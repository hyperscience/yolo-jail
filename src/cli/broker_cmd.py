"""Typer subcommand group: ``yolo broker {status,refresh,logs}``.

Manages the credential brokers — the host-side daemons that serve
Anthropic OAuth tokens (claude-credential-broker) and AWS STS sessions
(aws-credential-broker) into running jails.  Both brokers are
per-jail spawned daemons (no host-wide singleton like the legacy MITM
broker), so this module is mostly a thin wrapper around ``--self-check``
and ``--refresh-now`` on each broker plus log tailing.

Importing this module side-effect-attaches the subcommand group to the
top-level ``app``.  cli/__init__.py just imports it (no symbols
needed); registration happens via the @broker_app.command decorators.
"""

import subprocess
from pathlib import Path
from typing import List

import typer

from .console import console
from .paths import GLOBAL_STORAGE


broker_app = typer.Typer(
    help=(
        "Inspect the credential brokers — host-side daemons that serve "
        "Anthropic OAuth tokens and AWS STS sessions into jails."
    )
)


_BROKERS = {
    "claude": "yolo-claude-credential-broker",
    "aws": "yolo-aws-credential-broker",
}


def _run(argv: List[str]) -> int:
    """Run a broker subcommand and surface its output verbatim."""
    try:
        proc = subprocess.run(argv, check=False)
        return proc.returncode
    except FileNotFoundError:
        console.print(f"[red]not on PATH:[/red] {argv[0]}")
        return 127


@broker_app.command("status")
def broker_status_cmd():
    """Report status for each credential broker (delegates to the
    daemon's --self-check)."""
    overall = 0
    for label, binary in _BROKERS.items():
        console.print(f"[bold]{label}-credential-broker[/bold] ({binary})")
        rc = _run([binary, "--self-check"])
        if rc != 0:
            overall = rc
        console.print()
    raise typer.Exit(overall)


@broker_app.command("refresh")
def broker_refresh_cmd(
    target: str = typer.Argument(
        "all",
        help="Which broker to refresh: 'claude', 'aws', or 'all' (default).",
    ),
):
    """Force a fresh refresh against the upstream identity provider.

    Useful after rotating credentials on the host or to verify the
    broker's refresh-token chain still works.
    """
    targets = (
        list(_BROKERS.values())
        if target == "all"
        else [_BROKERS.get(target)]
    )
    if None in targets:
        console.print(f"[red]unknown broker:[/red] {target} (try 'claude', 'aws', or 'all')")
        raise typer.Exit(2)
    overall = 0
    for binary in targets:
        console.print(f"[bold]{binary}[/bold]")
        rc = _run([binary, "--refresh-now"])
        if rc != 0:
            overall = rc
    raise typer.Exit(overall)


@broker_app.command("logs")
def broker_logs_cmd(
    target: str = typer.Argument(
        "claude",
        help="Which broker's logs to tail: 'claude' or 'aws'.",
    ),
    lines: int = typer.Option(50, "-n", "--lines", help="Tail N lines"),
    follow: bool = typer.Option(False, "-f", "--follow", help="tail -f style"),
):
    """Tail the host-side log for a credential broker.

    Logs land in ~/.local/share/yolo-jail/logs/host-service-<broker>-<jail>.log
    (one log per jail).  This shows the most recent across all jails.
    """
    if target not in _BROKERS:
        console.print(f"[red]unknown broker:[/red] {target} (try 'claude' or 'aws')")
        raise typer.Exit(2)
    binary = _BROKERS[target]
    name = binary.replace("yolo-", "")
    log_dir = GLOBAL_STORAGE / "logs"
    matches = sorted(
        log_dir.glob(f"host-service-{name}-*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        console.print(f"[dim]No logs yet under {log_dir}/host-service-{name}-*.log[/dim]")
        raise typer.Exit(0)
    log_path: Path = matches[0]
    console.print(f"[dim]tailing {log_path}[/dim]")
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(log_path))
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
