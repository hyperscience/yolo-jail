#!/usr/bin/env python3
"""yolo-jail aws-credential-broker — host daemon.

Mints short-lived STS session credentials from the host's AWS
configuration and serves them to the jail over a unix socket.  The
jail's AWS SDK invokes ``yolo-aws-creds`` as ``credential_process`` and
the helper relays the daemon's reply verbatim.

We shell out to the host's ``aws`` CLI rather than depending on
boto3.  This inherits whatever SSO / role-chain / config the user has
already set up, sidesteps a 50MB dependency, and uses the same code
path that already works for the user on the host.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import host_service, loopholes as _loopholes
except ImportError:  # pragma: no cover — running as a script
    from src import host_service, loopholes as _loopholes  # type: ignore[no-redef]


# Limits per AWS docs.  STS won't issue session tokens shorter than
# 900s (15min) or longer than 43200s (12h) for IAM users, or 3600s
# (1h) for root user / temporary credentials.  We clamp client config
# into a safe range; the daemon surfaces "too long" via the same
# ttl_too_long error path the Anthropic broker uses.
STS_MIN_DURATION_S = 900
STS_MAX_DURATION_S = 43_200
DEFAULT_DURATION_S = 3_600


# How close to expiry we'll proactively re-mint, on top of the
# ``min_lifetime_ms`` the client requested.  60s — same shape as the
# Anthropic broker; gives us headroom for the network round-trip
# from helper invocation to first AWS call.
PROACTIVE_REFRESH_LEAD_MS = 60_000


# ARN sanity check — defensive for the "user pasted a role they don't
# own" foot-gun.  We don't try to validate semantically; just bound
# the shape so we fail before shelling out.
ROLE_ARN_RE = re.compile(r"^arn:aws[\w-]*:iam::\d{12}:role/[\w+=,.@/-]+$")


# Config + state paths.
CONFIG_PATH = Path.home() / ".config" / "yolo-jail" / "aws-broker.jsonc"
STATE_DIR = _loopholes.state_dir_for("aws-credential-broker")
CACHE_PATH = STATE_DIR / "cache.json"


log = logging.getLogger("aws-credential-broker")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    """Read the user's broker config.  Missing file → empty dict; the
    daemon falls back to AWS_PROFILE from the environment and a 1h
    default session duration.  We use pyjson5 if available (yolo's
    preferred jsonc parser) else strip comments quick-and-dirty."""
    if not CONFIG_PATH.exists():
        return {}
    raw = CONFIG_PATH.read_text()
    try:
        import pyjson5  # type: ignore[import-untyped]

        return pyjson5.loads(raw)
    except ImportError:
        # Strip // line comments and /* block comments before json.loads.
        no_line = re.sub(r"//[^\n]*", "", raw)
        no_block = re.sub(r"/\*.*?\*/", "", no_line, flags=re.DOTALL)
        try:
            return json.loads(no_block)
        except ValueError as e:
            log.error("malformed %s: %s", CONFIG_PATH, e)
            return {}


def _resolve_duration(cfg: Dict[str, Any]) -> int:
    raw = cfg.get("session_duration_seconds", DEFAULT_DURATION_S)
    try:
        d = int(raw)
    except (TypeError, ValueError):
        d = DEFAULT_DURATION_S
    return max(STS_MIN_DURATION_S, min(STS_MAX_DURATION_S, d))


HOST_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"


def _claude_settings_env(key: str) -> Optional[str]:
    """Read ``key`` from the host's ~/.claude/settings.json `env` block.

    Bedrock setups commonly write AWS_PROFILE / AWS_REGION there via
    Claude Code's ``/setup-bedrock`` wizard.  Mirroring those into the
    jail's broker means the host's existing Bedrock config Just Works
    without a separate aws-broker.jsonc file.  Returns None if the
    file is missing, unreadable, or doesn't carry the key.
    """
    try:
        if not HOST_CLAUDE_SETTINGS.is_file():
            return None
        data = json.loads(HOST_CLAUDE_SETTINGS.read_text())
    except (OSError, ValueError):
        return None
    env = data.get("env")
    if not isinstance(env, dict):
        return None
    val = env.get(key)
    return val if isinstance(val, str) and val else None


def _resolve_profile(cfg: Dict[str, Any]) -> Optional[str]:
    """Pick the AWS profile, in priority order:

    1. Explicit ``profile`` in ``~/.config/yolo-jail/aws-broker.jsonc``
       (highest precedence — operator override).
    2. ``AWS_PROFILE`` from the host's ``~/.claude/settings.json`` env
       block — what the user already wired for host Claude Code's
       Bedrock flow.
    3. ``AWS_PROFILE`` from the broker daemon's process environment.

    Skipping (3) when (2) is present matters because users sometimes
    have a default ``AWS_PROFILE`` exported in their shell that
    interferes with Bedrock auth (e.g. an SSO-only profile that
    refuses long-lived sessions); their ``settings.json`` workaround
    is to pin ``AWS_PROFILE=bedrock`` there.  We honor that pin.
    """
    return (
        cfg.get("profile")
        or _claude_settings_env("AWS_PROFILE")
        or os.environ.get("AWS_PROFILE")
    )


def _resolve_region(cfg: Dict[str, Any]) -> Optional[str]:
    """Pick the AWS region, same priority order as the profile."""
    return (
        cfg.get("region")
        or _claude_settings_env("AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )


def _resolve_role_arn(cfg: Dict[str, Any]) -> Optional[str]:
    arn = cfg.get("role_arn")
    if arn is None:
        return None
    if not isinstance(arn, str) or not ROLE_ARN_RE.match(arn):
        log.error("role_arn does not match expected shape: %r", arn)
        return None
    return arn


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _read_cache() -> Optional[Dict[str, Any]]:
    if not CACHE_PATH.is_file() or CACHE_PATH.stat().st_size == 0:
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except ValueError:
        return None


def _write_cache(payload: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(CACHE_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(payload, indent=2).encode())
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# STS calls via the host's aws CLI
# ---------------------------------------------------------------------------


def _aws_argv(cfg: Dict[str, Any], subcommand: str, *extras: str) -> list[str]:
    argv = ["aws", "sts", subcommand]
    profile = _resolve_profile(cfg)
    if profile:
        argv += ["--profile", profile]
    region = _resolve_region(cfg)
    if region:
        argv += ["--region", region]
    argv += list(extras)
    argv += ["--output", "json"]
    return argv


def _call_sts(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mint a fresh STS session.  Returns the ``Credentials`` dict.

    Uses ``sts:AssumeRole`` if ``role_arn`` is configured, else
    ``sts:GetSessionToken``.  Raises ``RuntimeError`` with the AWS CLI
    stderr on failure so callers can surface it to the helper.
    """
    duration = _resolve_duration(cfg)
    role_arn = _resolve_role_arn(cfg)
    if role_arn:
        argv = _aws_argv(
            cfg,
            "assume-role",
            "--role-arn",
            role_arn,
            "--role-session-name",
            "yolo-jail",
            "--duration-seconds",
            str(duration),
        )
    else:
        argv = _aws_argv(cfg, "get-session-token", "--duration-seconds", str(duration))
    log.info("aws sts %s (duration=%ds)", argv[2], duration)
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "aws sts failed")
    try:
        out = json.loads(proc.stdout)
    except ValueError as e:
        raise RuntimeError(f"aws sts returned non-JSON: {e}")
    creds = out.get("Credentials")
    if not isinstance(creds, dict):
        raise RuntimeError("aws sts response missing Credentials block")
    return creds


# ---------------------------------------------------------------------------
# Provide-token contract
# ---------------------------------------------------------------------------


def _expiration_ms(creds: Dict[str, Any]) -> int:
    """STS returns ``Expiration`` as ISO-8601.  Convert to ms since
    epoch.  Returns 0 if missing / unparseable so callers treat it as
    expired and re-mint."""
    raw = creds.get("Expiration")
    if not isinstance(raw, str):
        return 0
    try:
        # Python 3.11+ accepts trailing 'Z' via fromisoformat.
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError:
        return 0


def _remaining_ms(creds: Dict[str, Any]) -> int:
    return max(0, _expiration_ms(creds) - int(time.time() * 1000))


def _provide_credentials(min_lifetime_ms: int) -> Dict[str, Any]:
    """Return AWS-SDK-shape ``credential_process`` JSON if we can supply
    credentials whose remaining lifetime exceeds ``min_lifetime_ms``,
    else a structured error with a ``recommendation``.

    Output shape on success matches AWS's `credential_process` v1
    contract: {Version, AccessKeyId, SecretAccessKey, SessionToken,
    Expiration} (Expiration as ISO-8601, mirroring STS).
    """
    cfg = _load_config()

    cached = _read_cache()
    target_ms = min_lifetime_ms + PROACTIVE_REFRESH_LEAD_MS
    needs_refresh = (
        cached is None or _remaining_ms(cached.get("Credentials", {})) <= target_ms
    )

    if needs_refresh:
        try:
            creds = _call_sts(cfg)
        except subprocess.TimeoutExpired:
            return {
                "error": "sts_timeout",
                "message": "aws sts call timed out after 30s",
                "recommendation": (
                    "check host network and AWS endpoint reachability; "
                    "retry yolo doctor"
                ),
            }
        except RuntimeError as e:
            return {
                "error": "sts_failed",
                "message": str(e),
                "recommendation": (
                    "verify the configured profile / role / region are correct "
                    f"in {CONFIG_PATH}, and that `aws sts get-session-token` "
                    "works on the host"
                ),
            }
        cached = {"Credentials": creds, "MintedAtMs": int(time.time() * 1000)}
        _write_cache(cached)

    creds = cached["Credentials"]
    remaining = _remaining_ms(creds)
    if remaining <= min_lifetime_ms:
        recommended = max(60_000, remaining // 2)
        return {
            "error": "ttl_too_long",
            "message": (
                f"requested min_lifetime_ms={min_lifetime_ms} exceeds the STS "
                f"session lifetime ({remaining} ms remaining).  Increase "
                "session_duration_seconds in aws-broker.jsonc or lower the TTL."
            ),
            "recommendation": (
                f"lower CLAUDE_CODE_API_KEY_HELPER_TTL_MS to {recommended} "
                f"in the jail's settings, or raise session_duration_seconds in "
                f"{CONFIG_PATH} (max {STS_MAX_DURATION_S}s)"
            ),
        }

    return {
        "Version": 1,
        "AccessKeyId": creds.get("AccessKeyId"),
        "SecretAccessKey": creds.get("SecretAccessKey"),
        "SessionToken": creds.get("SessionToken"),
        "Expiration": creds.get("Expiration"),
        "expires_in_ms": remaining,
    }


# ---------------------------------------------------------------------------
# Daemon glue
# ---------------------------------------------------------------------------


def _handle(session: host_service.Session) -> None:
    request = session.request
    try:
        min_lifetime_ms = int(request.get("min_lifetime_ms", 60_000))
    except (TypeError, ValueError):
        session.json({"error": "bad_request", "message": "min_lifetime_ms must be int"})
        session.exit(2)
        return
    response = _provide_credentials(min_lifetime_ms)
    session.json(response)
    session.exit(0 if "error" not in response else 1)


def _self_check() -> int:
    """Validates the host environment without calling STS — STS calls
    cost real money / quota, and self-check runs on every yolo doctor."""
    if subprocess.run(["which", "aws"], capture_output=True).returncode != 0:
        print("aws CLI not on PATH", file=sys.stderr)
        return 1
    cfg = _load_config()
    profile = _resolve_profile(cfg)
    region = _resolve_region(cfg)
    duration = _resolve_duration(cfg)
    role = _resolve_role_arn(cfg)

    def _profile_source() -> str:
        if cfg.get("profile"):
            return f"from {CONFIG_PATH}"
        if _claude_settings_env("AWS_PROFILE"):
            return f"from {HOST_CLAUDE_SETTINGS} (env block)"
        if os.environ.get("AWS_PROFILE"):
            return "from broker process AWS_PROFILE"
        return "(unresolved — AWS SDK default chain)"

    def _region_source() -> str:
        if cfg.get("region"):
            return f"from {CONFIG_PATH}"
        if _claude_settings_env("AWS_REGION"):
            return f"from {HOST_CLAUDE_SETTINGS} (env block)"
        if os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"):
            return "from broker process env"
        return "(unresolved — aws CLI default)"

    print(f"config: {CONFIG_PATH} ({'present' if CONFIG_PATH.exists() else 'absent'})")
    print(f"profile: {profile or '(none)'}  [{_profile_source()}]")
    print(f"region:  {region or '(none)'}  [{_region_source()}]")
    print(f"duration: {duration}s (clamped to STS bounds)")
    print(f"role_arn: {role or '(none — sts:GetSessionToken)'}")
    cached = _read_cache()
    if cached is not None:
        print(f"cache: {CACHE_PATH} (remaining {_remaining_ms(cached.get('Credentials', {}))} ms)")
    else:
        print("cache: (empty — first request will mint a session)")
    return 0


def _refresh_now() -> int:
    cfg = _load_config()
    try:
        creds = _call_sts(cfg)
    except Exception as e:
        print(f"refresh failed: {e}", file=sys.stderr)
        return 2
    _write_cache({"Credentials": creds, "MintedAtMs": int(time.time() * 1000)})
    print(f"refreshed: expires {creds.get('Expiration')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="yolo-aws-credential-broker")
    p.add_argument("--socket", type=Path, help="unix socket path to bind on")
    p.add_argument("--self-check", action="store_true", help="report status and exit")
    p.add_argument(
        "--refresh-now", action="store_true", help="force an STS call and exit"
    )
    args = p.parse_args()

    logging.basicConfig(
        level=os.environ.get("YOLO_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.self_check:
        return _self_check()
    if args.refresh_now:
        return _refresh_now()
    if args.socket is None:
        p.error("--socket is required (or pass --self-check / --refresh-now)")
    host_service.serve(_handle, socket_path=args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
