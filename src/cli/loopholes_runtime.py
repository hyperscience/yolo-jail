"""Loophole / host-service runtime — the piece of the CLI that actually
launches the host-side daemons listed by ``yolo loopholes``.

What lives here:
  * LoopholeDaemon dataclass — the handle returned by start_loopholes
    and consumed by stop_loopholes / run()'s container command assembly.
  * _host_service_env_var, _host_service_default_jail_socket,
    _host_service_sockets_dir, _resolve_journal_mode,
    _substitute_socket_in_cmd — small helpers used by all daemons.
  * _should_mount_host_nix, _gpu_host_available — host-state probes
    used by run()'s mount-decision logic.  Kept here so the runtime
    plumbing all lives together.
  * cgroup delegate — _cgroup_delegate_handler,
    _cgd_ensure_agent_cgroup, _cgd_create_and_join, _cgd_destroy,
    _start_host_service_builtin_cgroup, _validate_cgroup_name,
    _parse_memory_value.  Implements the JSON socket protocol that
    backs ``yolo-cglimit``.
  * Journal bridge — JOURNAL_FRAME_*, _journal_send_frame,
    _journal_handle_client, _start_host_service_builtin_journal.
    Implements the framed binary-safe protocol behind
    ``yolo-journalctl``.
  * External daemons — _start_host_service_external.
  * start_loopholes / stop_loopholes — the lifecycle entry points
    called by run().
"""

import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src import loopholes as _loopholes

from .console import console
from .paths import (
    BUILTIN_CGROUP_LOOPHOLE_NAME,
    BUILTIN_JOURNAL_LOOPHOLE_NAME,
    CGD_SOCKET_NAME,
    GLOBAL_STORAGE,
    IS_LINUX,
    IS_MACOS,
    JAIL_HOST_SERVICES_DIR,
    JOURNAL_SOCKET_NAME,
)
from .runtime import _resolve_container_cgroup


@dataclass
class LoopholeDaemon:
    """A host-side service exposing a Unix socket inside the jail.

    Created by `start_loopholes` and torn down by `stop_loopholes`.
    Holds everything `run()` needs to wire the service into the container run command:
    bind mount, env var, optional client shim path.

    The `_stop` callable encapsulates the service's shutdown logic (SIGTERM for
    external, shutdown event for builtin) so both kinds look the same to the
    lifecycle manager.
    """

    name: str
    # Absolute path to the Unix socket on the host (inside the per-jail sockets dir).
    host_socket_path: Path
    # Absolute path where the socket appears inside the jail.
    jail_socket_path: str
    # Env var injected into the container so the agent can discover the socket.
    # Always set to YOLO_SERVICE_<sanitized-name>_SOCKET.
    env_var_name: str
    # Stop callable.  Called with no args at container exit.  Must not raise.
    _stop: "Callable[[], None]" = dataclasses.field(
        default_factory=lambda: lambda: None
    )
    # Optional path of a generated client-shim script, relative to the jail
    # filesystem.  If set, the shim is written into the per-workspace overlay
    # and appears on PATH inside the jail.
    client_shim_jail_path: Optional[str] = None


def _host_service_env_var(service_name: str) -> str:
    """Return the canonical env var name for a service's socket path."""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", service_name).strip("_").upper()
    return f"YOLO_SERVICE_{sanitized}_SOCKET"


def _host_service_default_jail_socket(name: str) -> str:
    """Default path where a service's socket appears inside the jail."""
    return f"{JAIL_HOST_SERVICES_DIR}/{name}.sock"


def _resolve_journal_mode(config: Dict[str, Any]) -> str:
    """Return the journal bridge mode from config.

    Accepts the canonical strings ("off", "user", "full").  `true` is
    treated as "user" (safer default for unprivileged agents), `false`
    and missing as "off".  Anything else is "off" — validation catches
    the invalid value separately and reports it to the user.
    """
    val = config.get("journal")
    if val is True:
        return "user"
    if val is False or val is None:
        return "off"
    if isinstance(val, str) and val in ("off", "user", "full"):
        return val
    return "off"


def _host_service_sockets_dir(cname: str) -> Path:
    """Per-jail directory holding all host-service sockets on the host side.

    Bind-mounted into the jail at JAIL_HOST_SERVICES_DIR.

    Lives under /tmp (not ws_state!) because Linux's AF_UNIX path limit is
    108 bytes and macOS's is 104 — workspace paths on CI runners or in
    nested directories can easily blow that when we append
    "<service-name>.sock" on top.  /tmp is always 4 bytes, leaving plenty
    of room.

    The directory name uses an 8-char hash of the container name to avoid
    collisions while keeping the path short:

        /tmp/yolo-host-services-<8hex>/cgroup-delegate.sock   (~53 bytes)

    macOS resolves /tmp → /private/tmp; we use the resolved form so paths
    we hand to Python's socket module match what the kernel sees.
    """
    short_hash = hashlib.sha1(cname.encode()).hexdigest()[:8]
    base = Path("/tmp").resolve() if IS_MACOS else Path("/tmp")
    return base / f"yolo-host-services-{short_hash}"


def _substitute_socket_in_cmd(args: List[str], socket_path: str) -> List[str]:
    """Replace '{socket}' in each arg with the actual socket path."""
    return [a.replace("{socket}", socket_path) for a in args]



def _should_mount_host_nix(
    runtime: str,
    *,
    nix_socket_exists: bool,
    nix_store_exists: bool,
    is_macos: bool,
    opt_in_env: Optional[str],
) -> bool:
    """Decide whether ``run()`` should bind-mount the host's Nix daemon + store.

    Linux: mount whenever both paths exist and runtime supports it.
    macOS: skip by default — the typical container runtime VM (Podman
    Machine, Apple container) does not share /nix, and bind-mounting
    statfs-errors at startup.  Setups that *do* share /nix (e.g.
    Colima with a custom mount) can opt back in by setting
    ``YOLO_NIX_HOST_DAEMON`` to a truthy value (``1``/``true``/``yes``).
    Apple Container can't share Unix sockets via -v bind mounts regardless,
    so the runtime gate handles that case.
    """
    if not (nix_socket_exists and nix_store_exists):
        return False
    if runtime == "container":
        return False
    if not is_macos:
        return True
    opt_in = (opt_in_env or "").lower() in ("1", "true", "yes")
    return opt_in


def _gpu_host_available(runtime: str) -> tuple[bool, Optional[str]]:
    """Probe whether NVIDIA GPU passthrough will actually work on this host.

    Returns ``(True, None)`` if the host has the drivers + toolkit
    podman needs, or ``(False, reason)`` explaining what's missing.
    ``reason`` is a single short phrase suitable for a one-line
    warning (e.g. ``"nvidia-smi not found on host"``).

    Used by :func:`run` so a workspace config with ``gpu.enabled: true``
    stays portable across a GPU box and a GPU-less laptop — the
    GPU-less machine sees a warning and starts without the CDI device
    flags instead of a hard podman error.

    GPU passthrough requires podman + CDI.  Other runtimes (macOS,
    Apple Container) return a skip reason; callers already warn/skip
    for those earlier.
    """
    if IS_MACOS or runtime == "container":
        return False, "runtime does not support NVIDIA passthrough"
    if runtime != "podman":
        return False, f"unsupported runtime: {runtime}"

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found on host"
    try:
        probe = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"nvidia-smi failed to run ({e})"
    if probe.returncode != 0:
        return False, "nvidia-smi reported no GPUs"

    cdi_paths = (Path("/etc/cdi/nvidia.yaml"), Path("/var/run/cdi/nvidia.yaml"))
    if not any(p.exists() for p in cdi_paths):
        return False, "no CDI spec at /etc/cdi/nvidia.yaml"
    return True, None




def _validate_cgroup_name(name: str) -> bool:
    """Validate that a cgroup name is safe (no path traversal)."""
    return (
        bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$", name)) and ".." not in name
    )


def _parse_memory_value(val: str) -> Optional[int]:
    """Parse a human-readable memory value to bytes.  Returns None on invalid input."""
    val = val.strip().lower()
    try:
        if val.endswith("g"):
            return int(float(val[:-1]) * 1073741824)
        if val.endswith("m"):
            return int(float(val[:-1]) * 1048576)
        if val.endswith("k"):
            return int(float(val[:-1]) * 1024)
        return int(val)
    except (ValueError, OverflowError):
        return None


def _cgroup_delegate_handler(
    conn: socket.socket,
    container_cgroup: Path,
    log_file,
):
    """Handle a single cgroup delegate request from the container.

    Protocol: single-line JSON request, single-line JSON response.
    """
    try:
        data = b""
        while b"\n" not in data and len(data) < 4096:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        if not data:
            return

        request = json.loads(data.decode("utf-8", errors="replace"))
        op = request.get("op", "")

        # Get the host-PID of the caller — only the PID is used; UID/GID
        # returned alongside it are ignored.
        # Linux: SO_PEERCRED (returns pid/uid/gid as three ints)
        # macOS: LOCAL_PEERPID (returns just the pid)
        try:
            if IS_LINUX:
                cred = conn.getsockopt(
                    socket.SOL_SOCKET,
                    getattr(socket, "SO_PEERCRED"),
                    struct.calcsize("3i"),
                )
                peer_pid = struct.unpack("3i", cred)[0]
            elif IS_MACOS:
                # macOS: LOCAL_PEERPID (0x002) returns the peer PID
                LOCAL_PEERPID = 0x002
                cred = conn.getsockopt(0, LOCAL_PEERPID, struct.calcsize("i"))
                peer_pid = struct.unpack("i", cred)[0]
            else:
                peer_pid = 0
        except (OSError, struct.error, AttributeError):
            peer_pid = 0

        # Log every request for auditability
        log_line = f"op={op} peer_pid={peer_pid} request={json.dumps(request)}"
        print(log_line, file=log_file, flush=True)

        if op == "status":
            # Check if delegation is available
            agent_cg = container_cgroup / "agent"
            controllers = ""
            if agent_cg.exists():
                try:
                    controllers = (agent_cg / "cgroup.controllers").read_text().strip()
                except OSError:
                    pass
            response = {
                "ok": True,
                "delegated": agent_cg.exists(),
                "controllers": controllers,
                "cgroup": str(container_cgroup),
            }

        elif op == "create_and_join":
            name = request.get("name", "")
            if not _validate_cgroup_name(name):
                response = {"ok": False, "error": f"Invalid cgroup name: {name!r}"}
            elif peer_pid <= 0:
                response = {"ok": False, "error": "Could not determine caller PID"}
            else:
                response = _cgd_create_and_join(
                    container_cgroup, name, request, peer_pid, log_file
                )

        elif op == "destroy":
            name = request.get("name", "")
            if not _validate_cgroup_name(name):
                response = {"ok": False, "error": f"Invalid cgroup name: {name!r}"}
            else:
                response = _cgd_destroy(container_cgroup, name, log_file)

        else:
            response = {"ok": False, "error": f"Unknown operation: {op!r}"}

        conn.sendall((json.dumps(response) + "\n").encode())
        print(f"  response={json.dumps(response)}", file=log_file, flush=True)

    except Exception as exc:
        try:
            conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
        except Exception:
            pass
    finally:
        conn.close()


def _cgd_ensure_agent_cgroup(container_cgroup: Path, log_file) -> Optional[Path]:
    """Ensure the agent cgroup subtree exists with controllers enabled.

    Returns the path to the agent cgroup, or None on failure.
    """
    agent_cg = container_cgroup / "agent"
    init_cg = container_cgroup / "init"

    if agent_cg.exists():
        return agent_cg

    try:
        init_cg.mkdir(exist_ok=True)
        agent_cg.mkdir(exist_ok=True)
    except OSError as e:
        print(f"  ERROR creating cgroup dirs: {e}", file=log_file, flush=True)
        return None

    # Move all existing processes to 'init' (cgroup v2 no-internal-process constraint)
    try:
        procs = (container_cgroup / "cgroup.procs").read_text().strip().split()
        for pid in procs:
            try:
                (init_cg / "cgroup.procs").write_text(pid)
            except OSError:
                pass  # Process may have exited or be a kthread
    except OSError:
        pass

    # Enable controllers on container root → agent subtree
    for cg in [container_cgroup, agent_cg]:
        try:
            available = (cg / "cgroup.controllers").read_text().strip().split()
            wanted = [c for c in ["cpu", "memory", "pids"] if c in available]
            if wanted:
                ctrl = " ".join(f"+{c}" for c in wanted)
                (cg / "cgroup.subtree_control").write_text(ctrl)
        except OSError:
            pass

    return agent_cg


def _cgd_create_and_join(
    container_cgroup: Path,
    name: str,
    request: dict,
    peer_pid: int,
    log_file,
) -> dict:
    """Create a child cgroup under agent/, set limits, and move the caller into it."""
    agent_cg = _cgd_ensure_agent_cgroup(container_cgroup, log_file)
    if agent_cg is None:
        return {"ok": False, "error": "Failed to set up agent cgroup hierarchy"}

    job_cg = agent_cg / name
    try:
        job_cg.mkdir(exist_ok=True)
    except OSError as e:
        return {"ok": False, "error": f"Cannot create cgroup {name}: {e}"}

    errors = []

    # CPU limit: percentage of all CPUs → cpu.max (quota period)
    cpu_pct = request.get("cpu_pct")
    if cpu_pct is not None:
        try:
            pct = int(cpu_pct)
            nproc = os.cpu_count() or 1
            if pct < 1 or pct > 100 * nproc:
                errors.append(f"cpu_pct out of range: {pct}")
            else:
                quota = pct * 1000 * nproc
                (job_cg / "cpu.max").write_text(f"{quota} 100000")
        except (ValueError, OSError) as e:
            errors.append(f"cpu.max: {e}")

    # Memory limit
    memory = request.get("memory")
    if memory is not None:
        mem_bytes = _parse_memory_value(str(memory))
        if mem_bytes is None or mem_bytes < 1048576:  # min 1MB
            errors.append(f"Invalid memory value: {memory}")
        else:
            try:
                (job_cg / "memory.max").write_text(str(mem_bytes))
            except OSError as e:
                errors.append(f"memory.max: {e}")

    # PID limit
    pids = request.get("pids")
    if pids is not None:
        try:
            pids_val = int(pids)
            if pids_val < 1 or pids_val > 1000000:
                errors.append(f"pids out of range: {pids_val}")
            else:
                (job_cg / "pids.max").write_text(str(pids_val))
        except (ValueError, OSError) as e:
            errors.append(f"pids.max: {e}")

    # Move the caller into the new cgroup (peer_pid is already host-namespace)
    try:
        (job_cg / "cgroup.procs").write_text(str(peer_pid))
    except OSError as e:
        return {
            "ok": False,
            "error": f"Cannot move PID {peer_pid} into cgroup: {e}",
            "limit_errors": errors,
        }

    cg_root = Path("/sys/fs/cgroup")
    try:
        cg_path = str(job_cg.relative_to(cg_root))
    except ValueError:
        cg_path = str(job_cg)
    result = {"ok": True, "cgroup": cg_path}
    if errors:
        result["warnings"] = errors
    return result


def _cgd_destroy(container_cgroup: Path, name: str, _log_file) -> dict:
    """Remove a child cgroup (must be empty of processes).

    `_log_file` is accepted to match the signature of sibling handlers
    (`_cgd_create`, `_cgd_status`) that all share a dispatch table; this
    handler doesn't need to log anything extra beyond the top-level request
    log line.
    """
    agent_cg = container_cgroup / "agent"
    job_cg = agent_cg / name
    if not job_cg.exists():
        return {"ok": True}  # Already gone — idempotent
    try:
        # Check for remaining processes
        procs = (job_cg / "cgroup.procs").read_text().strip()
        if procs:
            return {
                "ok": False,
                "error": f"Cgroup {name} still has processes: {procs}",
            }
        job_cg.rmdir()
        return {"ok": True}
    except OSError as e:
        return {"ok": False, "error": f"Cannot remove cgroup {name}: {e}"}


def _start_host_service_builtin_cgroup(
    cname: str, runtime: str, sockets_dir: Path
) -> Optional[LoopholeDaemon]:
    """Start the built-in cgroup delegate daemon as a host service.

    Listens on <sockets_dir>/cgroup.sock.  Returns a LoopholeDaemon handle, or
    None if cgroup v2 is not available (macOS or Linux without cgroup v2).

    This is functionally identical to the pre-refactor `start_cgroup_delegate`
    — same thread, same handler, same JSON protocol.  The only difference is
    that the socket now lives in the unified per-jail host-services directory
    instead of /tmp/yolo-cgd-<cname>.
    """
    if IS_MACOS:
        # macOS has no cgroup v2 — skip the delegation daemon entirely.
        return None

    # Quick sanity: is cgroup v2 available on the host?
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        return None

    sockets_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sockets_dir / CGD_SOCKET_NAME
    sock_path.unlink(missing_ok=True)

    log_dir = GLOBAL_STORAGE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"{cname}-cgd.log", "a")

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    sock_path.chmod(0o777)  # Container runs as mapped UID — must be accessible
    srv.listen(8)
    srv.settimeout(1.0)  # Allow periodic shutdown checks

    container_cgroup: Optional[Path] = None
    container_cgroup_lock = threading.Lock()
    shutdown = threading.Event()

    def serve():
        nonlocal container_cgroup
        while not shutdown.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # Lazy-resolve container cgroup on first request
            with container_cgroup_lock:
                if container_cgroup is None:
                    container_cgroup = _resolve_container_cgroup(cname, runtime)
                    if container_cgroup:
                        print(
                            f"Resolved container cgroup: {container_cgroup}",
                            file=log_file,
                            flush=True,
                        )
                    else:
                        print(
                            "WARNING: Could not resolve container cgroup",
                            file=log_file,
                            flush=True,
                        )
            if container_cgroup is None:
                try:
                    conn.sendall(
                        (
                            json.dumps(
                                {
                                    "ok": False,
                                    "error": "Container cgroup not yet available",
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    conn.close()
                except Exception:
                    pass
                continue

            _cgroup_delegate_handler(conn, container_cgroup, log_file)
        srv.close()
        log_file.close()

    t = threading.Thread(
        target=serve, daemon=True, name=f"host-service-{BUILTIN_CGROUP_LOOPHOLE_NAME}"
    )
    t.start()

    # Give the socket a moment to be ready
    time.sleep(0.05)

    def _stop():
        shutdown.set()
        # The srv.settimeout(1.0) means accept() will return within a second,
        # at which point the loop notices shutdown.is_set() and exits cleanly.
        t.join(timeout=3)

    return LoopholeDaemon(
        name=BUILTIN_CGROUP_LOOPHOLE_NAME,
        host_socket_path=sock_path,
        jail_socket_path=_host_service_default_jail_socket(
            BUILTIN_CGROUP_LOOPHOLE_NAME
        ),
        env_var_name=_host_service_env_var(BUILTIN_CGROUP_LOOPHOLE_NAME),
        _stop=_stop,
    )


# --- Journal bridge -------------------------------------------------------
#
# Wire protocol (framed, binary-safe — `journalctl -o export` is not
# line-delimited and `-f` follows indefinitely, so a plain newline-delimited
# stream wouldn't work):
#
#   Client → server:  single JSON line  {"args": ["-u", "foo", "-n", "50"]}\n
#   Server → client:  zero or more frames, each:
#                       [stream:1 byte][length:4 bytes BE][payload:length bytes]
#                     where stream ∈ {1=stdout, 2=stderr, 3=exit}.
#                     An "exit" frame has length=4 and payload=int32 BE.
#                     After the exit frame, the server closes the socket.
#
# The client script (~/.local/bin/yolo-journalctl, generated by
# entrypoint.py) decodes these frames back onto its own stdout/stderr and
# exits with the received code.
JOURNAL_FRAME_STDOUT = 1
JOURNAL_FRAME_STDERR = 2
JOURNAL_FRAME_EXIT = 3
JOURNAL_MAX_ARGS = 64
JOURNAL_MAX_ARG_LEN = 1024


def _journal_send_frame(conn: socket.socket, stream: int, payload: bytes) -> None:
    header = struct.pack(">BI", stream, len(payload))
    conn.sendall(header + payload)


def _journal_handle_client(conn: socket.socket, mode: str, log_file) -> None:
    """Serve one yolo-journalctl request end-to-end.

    `mode` is "user" (force --user) or "full" (pass args through).
    """
    try:
        # Read a single JSON request line.  Cap the header to avoid a
        # runaway client hanging the daemon thread.
        data = b""
        while b"\n" not in data and len(data) < 16384:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        if b"\n" not in data:
            _journal_send_frame(
                conn, JOURNAL_FRAME_STDERR, b"yolo-journal: malformed request\n"
            )
            _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 2))
            return
        header, _ = data.split(b"\n", 1)
        try:
            request = json.loads(header.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            msg = f"yolo-journal: invalid JSON: {e}\n".encode()
            _journal_send_frame(conn, JOURNAL_FRAME_STDERR, msg)
            _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 2))
            return

        args = request.get("args") or []
        if not isinstance(args, list) or len(args) > JOURNAL_MAX_ARGS:
            _journal_send_frame(
                conn,
                JOURNAL_FRAME_STDERR,
                f"yolo-journal: args must be a list of ≤{JOURNAL_MAX_ARGS} strings\n".encode(),
            )
            _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 2))
            return
        clean_args: List[str] = []
        for a in args:
            if not isinstance(a, str) or len(a) > JOURNAL_MAX_ARG_LEN:
                _journal_send_frame(
                    conn,
                    JOURNAL_FRAME_STDERR,
                    b"yolo-journal: each arg must be a string under 1024 bytes\n",
                )
                _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 2))
                return
            clean_args.append(a)

        # "user" mode: always force --user.  The user could technically add
        # their own --user already; a duplicate flag is harmless to
        # journalctl.  We do NOT strip conflicting flags (--system, -M) —
        # journalctl itself rejects those combinations and prints a clear
        # error, which we forward to the client.
        if mode == "user":
            clean_args = ["--user", *clean_args]

        print(
            f"[journal] mode={mode} args={json.dumps(clean_args)}",
            file=log_file,
            flush=True,
        )

        # Spawn journalctl.  start_new_session so SIGTERM reaches the
        # process and any children if the client disconnects.
        try:
            proc = subprocess.Popen(
                ["journalctl", *clean_args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            _journal_send_frame(
                conn,
                JOURNAL_FRAME_STDERR,
                b"yolo-journal: journalctl not found on host\n",
            )
            _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 127))
            return
        except OSError as e:
            _journal_send_frame(
                conn,
                JOURNAL_FRAME_STDERR,
                f"yolo-journal: spawn failed: {e}\n".encode(),
            )
            _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", 1))
            return

        send_lock = threading.Lock()

        def pump(stream_fd, frame_type: int):
            try:
                while True:
                    buf = stream_fd.read(4096)
                    if not buf:
                        return
                    with send_lock:
                        try:
                            _journal_send_frame(conn, frame_type, buf)
                        except OSError:
                            # Client went away — kill journalctl and bail.
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            return
            except Exception:
                return

        t_out = threading.Thread(
            target=pump, args=(proc.stdout, JOURNAL_FRAME_STDOUT), daemon=True
        )
        t_err = threading.Thread(
            target=pump, args=(proc.stderr, JOURNAL_FRAME_STDERR), daemon=True
        )
        t_out.start()
        t_err.start()

        rc = proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        with send_lock:
            try:
                _journal_send_frame(conn, JOURNAL_FRAME_EXIT, struct.pack(">i", rc))
            except OSError:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _start_host_service_builtin_journal(
    cname: str, sockets_dir: Path, mode: str
) -> Optional[LoopholeDaemon]:
    """Start the built-in journal bridge as a host service.

    `mode` is "user" or "full".  Returns None if journalctl isn't on the
    host's PATH (macOS or a minimal Linux without systemd).
    """
    if shutil.which("journalctl") is None:
        console.print(
            "[yellow]journal bridge requested but journalctl not found on host — "
            "skipping[/yellow]"
        )
        return None

    sockets_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sockets_dir / JOURNAL_SOCKET_NAME
    sock_path.unlink(missing_ok=True)

    log_dir = GLOBAL_STORAGE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"{cname}-journal.log", "a")

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    sock_path.chmod(0o777)
    srv.listen(8)
    srv.settimeout(1.0)

    shutdown = threading.Event()

    def serve():
        while not shutdown.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            handler = threading.Thread(
                target=_journal_handle_client,
                args=(conn, mode, log_file),
                daemon=True,
                name="journal-client",
            )
            handler.start()
        srv.close()
        log_file.close()

    t = threading.Thread(
        target=serve, daemon=True, name=f"host-service-{BUILTIN_JOURNAL_LOOPHOLE_NAME}"
    )
    t.start()
    time.sleep(0.05)

    def _stop():
        shutdown.set()
        t.join(timeout=3)

    return LoopholeDaemon(
        name=BUILTIN_JOURNAL_LOOPHOLE_NAME,
        host_socket_path=sock_path,
        jail_socket_path=_host_service_default_jail_socket(
            BUILTIN_JOURNAL_LOOPHOLE_NAME
        ),
        env_var_name=_host_service_env_var(BUILTIN_JOURNAL_LOOPHOLE_NAME),
        _stop=_stop,
    )


def _start_host_service_external(
    name: str,
    spec: Dict[str, Any],
    sockets_dir: Path,
    startup_timeout_secs: float = 5.0,
) -> Optional[LoopholeDaemon]:
    """Launch a user-configured external host service.

    The service's command is expected to bind a Unix socket at the path
    substituted for `{socket}` in its args.  Returns a LoopholeDaemon handle if
    the service bound the socket within `startup_timeout_secs`, or None on
    failure (command not found, socket not bound, process exited early).
    """
    sockets_dir.mkdir(parents=True, exist_ok=True)
    host_socket = sockets_dir / f"{name}.sock"
    host_socket.unlink(missing_ok=True)

    cmd_template = spec.get("command") or []
    if not isinstance(cmd_template, list) or not cmd_template:
        console.print(f"[red]Host service '{name}' has no command; skipping[/red]")
        return None

    cmd = _substitute_socket_in_cmd(
        [
            str(Path(str(a)).expanduser()) if a.startswith("~") else str(a)
            for a in cmd_template
        ],
        str(host_socket),
    )

    env = {**os.environ}
    for k, v in (spec.get("env") or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        env[k] = str(Path(v).expanduser()) if v.startswith("~") else v

    log_dir = GLOBAL_STORAGE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"host-service-{name}.log"
    log_file = open(log_path, "ab")

    def _print_log_tail(reason: str, max_lines: int = 5) -> None:
        """Surface the last few log lines so operators don't have to fish.

        The service's own stderr/stdout already captured the actionable
        error (e.g. a Python traceback ending in FileNotFoundError); we
        echo the tail to the console alongside the failure message.
        """
        try:
            log_file.flush()
        except Exception:
            pass
        try:
            with open(log_path, "rb") as f:
                tail = f.read()[-4096:].decode(errors="replace").rstrip()
        except OSError:
            return
        if not tail:
            return
        lines = tail.splitlines()[-max_lines:]
        console.print(
            f"[yellow]Last {len(lines)} line(s) of {log_path} ({reason}):[/yellow]"
        )
        for line in lines:
            console.print(f"  [dim]{line}[/dim]")

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,  # own process group so SIGTERM reaches kids
        )
    except (OSError, FileNotFoundError) as e:
        console.print(f"[red]Failed to launch host service '{name}': {e}[/red]")
        log_file.close()
        return None

    # Wait for the service to bind the socket (or exit early with an error).
    deadline = time.monotonic() + startup_timeout_secs
    while time.monotonic() < deadline:
        if host_socket.exists():
            break
        if proc.poll() is not None:
            console.print(
                f"[red]Host service '{name}' exited early with code "
                f"{proc.returncode} before binding {host_socket}[/red]"
            )
            _print_log_tail(f"exit code {proc.returncode}")
            log_file.close()
            return None
        time.sleep(0.05)
    else:
        console.print(
            f"[red]Host service '{name}' did not bind {host_socket} within "
            f"{startup_timeout_secs:.1f}s — killing[/red]"
        )
        _print_log_tail("startup timeout")
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass
        log_file.close()
        return None

    def _stop():
        # SIGTERM, give it 5s, SIGKILL if it's still around.
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
        try:
            log_file.close()
        except Exception:
            pass

    jail_socket = spec.get("jail_socket") or _host_service_default_jail_socket(name)
    return LoopholeDaemon(
        name=name,
        host_socket_path=host_socket,
        jail_socket_path=jail_socket,
        env_var_name=_host_service_env_var(name),
        _stop=_stop,
    )


def start_loopholes(
    cname: str,
    runtime: str,
    config: Dict[str, Any],
) -> List[LoopholeDaemon]:
    """Start all host services for this jail and return handles.

    Always attempts the built-in cgroup delegate (skipped gracefully on
    macOS and non-cgroup-v2 Linux).  Then launches any services declared in
    ``config["loopholes"]`` as external processes.

    The caller is responsible for passing the returned handles to
    ``stop_loopholes`` at container exit, along with the same socket
    directory (recoverable via ``_host_service_sockets_dir(cname)``).
    """
    sockets_dir = _host_service_sockets_dir(cname)
    sockets_dir.mkdir(parents=True, exist_ok=True)

    handles: List[LoopholeDaemon] = []

    # Apple Container caveats:
    #   - Cannot bind-mount the sockets *directory* into the jail
    #     (virtiofs directory shares don't carry AF_UNIX inodes
    #     correctly).  ``run_cmd.py`` uses per-socket ``-v`` mounts
    #     instead — see the ``runtime == "container"`` branch there.
    #   - The cgroup delegate is Linux-only regardless of runtime.
    if runtime == "container":
        # Apple Container: skip the directory-mount path; host services
        # still spawn so per-socket bind mounts above work.
        pass

    # 1. Built-in cgroup delegate (Linux only, cgroup v2 only).
    builtin = _start_host_service_builtin_cgroup(cname, runtime, sockets_dir)
    if builtin is not None:
        handles.append(builtin)

    # 2. Built-in journal bridge (opt in via top-level `journal` key).
    journal_mode = _resolve_journal_mode(config)
    if journal_mode != "off":
        journal = _start_host_service_builtin_journal(cname, sockets_dir, journal_mode)
        if journal is not None:
            handles.append(journal)

    # 3. External services.  Discovery unifies three sources:
    #      a) Bundled loopholes (ship in the wheel).
    #      b) User-installed loopholes (~/.local/share/yolo-jail/loopholes/).
    #      c) Inline ``loopholes:`` entries in yolo-jail.jsonc for daemons
    #         that don't need a file-backed manifest.
    #    Workspace config can also override (a) and (b) via name-matching
    #    entries — see ``_apply_workspace_overrides`` in src/loopholes.py.
    #    Inactive loopholes (disabled, or ``requires`` not met) are skipped.
    loopholes_config = config.get("loopholes")
    discovered = _loopholes.discover_loopholes(loopholes_config=loopholes_config)
    manifest_specs = _loopholes.manifest_host_daemon_specs(discovered)
    external_specs: Dict[str, Any] = dict(manifest_specs)
    # Config-inline loopholes (no matching file-backed entry) still end up
    # as unix-socket daemons — the synthesizer captured them, but
    # _start_host_service_external wants the original config dict shape,
    # so pull those straight from config for their command fields.
    if isinstance(loopholes_config, dict):
        for name, spec in loopholes_config.items():
            if name in external_specs:
                continue  # already covered by a file-backed manifest's host_daemon
            if isinstance(spec, dict) and "command" in spec:
                external_specs[name] = spec
    for name, spec in external_specs.items():
        if name in (BUILTIN_CGROUP_LOOPHOLE_NAME, BUILTIN_JOURNAL_LOOPHOLE_NAME):
            continue  # reserved builtins
        if not isinstance(spec, dict):
            continue
        h = _start_host_service_external(name, spec, sockets_dir)
        if h is not None:
            handles.append(h)

    return handles


def stop_loopholes(handles: List[LoopholeDaemon], sockets_dir: Optional[Path]) -> None:
    """Stop all host services and clean up the sockets directory."""
    for h in handles:
        try:
            h._stop()
        except Exception as e:
            console.print(
                f"[yellow]Error stopping host service '{h.name}': {e}[/yellow]"
            )
    if sockets_dir is not None and sockets_dir.exists():
        shutil.rmtree(sockets_dir, ignore_errors=True)
