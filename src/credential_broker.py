#!/usr/bin/env python3
"""yolo-jail claude-credential-broker — host daemon.

Listens on a unix socket; jail-side helper sends a request like
``{"min_lifetime_ms": 60000}`` and the daemon replies with the current
access token from its private OAuth state, refreshing first if the
remaining lifetime would be insufficient.

The on-disk state lives under
``~/.local/share/yolo-jail/state/claude-credential-broker/credentials.json``
in the same shape Claude Code itself writes (``{"claudeAiOauth": {...}}``)
so existing migration / upgrade tooling can read it.

This module is the MITM-free replacement for ``oauth_broker.py``.  See
``src/bundled_loopholes/claude-credential-broker/README.md`` for the
architectural overview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import host_service, loopholes as _loopholes
except ImportError:  # pragma: no cover — running as a script
    from src import host_service, loopholes as _loopholes  # type: ignore[no-redef]


# Upstream OAuth endpoint.  Same constants the legacy MITM broker uses;
# verified by extracting them from the Claude Code 2.1.x binary.  If
# Anthropic moves it, refreshes start failing with 404 and you re-verify
# with: ``rg -oab 'platform\.claude\.com|/v1/oauth/token' <claude-binary>``.
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_BETA_HEADER = "oauth-2025-04-20"


def _user_agent() -> str:
    """A non-default UA — Cloudflare drops Python-urllib's default with
    error 1010.  Identifying ourselves honestly also makes Anthropic-side
    log forensics sane."""
    try:
        from importlib.metadata import version as _pkg_version

        return f"yolo-jail-credential-broker/{_pkg_version('yolo-jail')}"
    except Exception:
        return "yolo-jail-credential-broker"


USER_AGENT = _user_agent()


# How close to expiry we'll proactively refresh, on top of the
# ``min_lifetime_ms`` the client requested.  60s means: if the token
# would expire 30s after the client's TTL, we refresh now instead of
# returning a token that's about to die.
PROACTIVE_REFRESH_LEAD_MS = 60_000


# Bootstrap sources, tried in order.
LEGACY_SHARED_CREDS = (
    Path.home()
    / ".local/share/yolo-jail/home/.claude-shared-credentials/.credentials.json"
)
HOST_CLAUDE_CREDS = Path.home() / ".claude/.credentials.json"


log = logging.getLogger("claude-credential-broker")


# ---------------------------------------------------------------------------
# State path resolution
# ---------------------------------------------------------------------------


STATE_DIR = _loopholes.state_dir_for("claude-credential-broker")
CREDS_PATH = STATE_DIR / "credentials.json"


def _state_path() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return CREDS_PATH


def _token_fp(tok: Optional[str]) -> str:
    """Stable 8-hex-char fingerprint of a token (sha256 prefix).  Safe to
    log; lets us correlate refresh chains across processes without
    leaking tokens."""
    if not tok:
        return "(none)"
    return hashlib.sha256(tok.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _read_creds(path: Path) -> Optional[Dict[str, Any]]:
    """Return the OAuth blob (``claudeAiOauth`` value) from a creds file,
    or None if missing / unreadable / malformed."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    return oauth


def _write_creds(path: Path, oauth: Dict[str, Any]) -> None:
    """Atomic write of the broker's credentials file (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"claudeAiOauth": oauth}, indent=2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, blob.encode())
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Refresh against Anthropic
# ---------------------------------------------------------------------------


def _refresh_upstream(refresh_token: str) -> Dict[str, Any]:
    """POST ``/v1/oauth/token`` with grant_type=refresh_token.  Returns
    the parsed JSON response on success; raises on HTTP / parse error."""
    body = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "anthropic-beta": OAUTH_BETA_HEADER,
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _normalize(upstream: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an upstream response to the on-disk shape, preserving any
    fields we don't manage (subscriptionType, scopes, etc.)."""
    now_ms = int(time.time() * 1000)
    expires_in = int(upstream.get("expires_in", 3600))
    out = dict(previous)
    out["accessToken"] = upstream["access_token"]
    if "refresh_token" in upstream:
        out["refreshToken"] = upstream["refresh_token"]
    out["expiresAt"] = now_ms + expires_in * 1000
    return out


def _refresh_and_persist(oauth: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh once and write the result to disk.  Returns the new
    OAuth blob.  Raises on refresh failure."""
    rt = oauth.get("refreshToken")
    if not rt:
        raise RuntimeError("no_refresh_token in stored credentials")
    log.info("refreshing rt=%s", _token_fp(rt))
    upstream = _refresh_upstream(rt)
    new_oauth = _normalize(upstream, oauth)
    _write_creds(_state_path(), new_oauth)
    log.info(
        "refreshed: at=%s rt=%s exp=%s",
        _token_fp(new_oauth.get("accessToken")),
        _token_fp(new_oauth.get("refreshToken")),
        new_oauth.get("expiresAt"),
    )
    return new_oauth


# ---------------------------------------------------------------------------
# Bootstrap — one-shot import from a source identity
# ---------------------------------------------------------------------------


def _bootstrap_from(source: Path) -> Optional[Dict[str, Any]]:
    """Capture a refresh token from ``source`` and immediately fork the
    chain by refreshing.  Writes the result to the broker's state file
    and returns the new OAuth blob, or None if the source isn't usable.

    The immediate refresh is what makes the broker's identity safe to
    coexist with the source identity: after this returns, the broker
    holds a *new* refresh token that the source has never seen.  Both
    chains are independent and can refresh on their own schedules.
    """
    src = _read_creds(source)
    if src is None:
        return None
    rt = src.get("refreshToken")
    if not rt:
        log.warning("bootstrap source %s missing refreshToken", source)
        return None
    log.info("bootstrapping from %s rt=%s", source, _token_fp(rt))
    try:
        upstream = _refresh_upstream(rt)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        log.warning("bootstrap refresh failed for %s: %s", source, e)
        return None
    new_oauth = _normalize(upstream, src)
    _write_creds(_state_path(), new_oauth)
    log.info(
        "bootstrap forked chain: rt %s → %s",
        _token_fp(rt),
        _token_fp(new_oauth.get("refreshToken")),
    )
    return new_oauth


def _ensure_credentials() -> Optional[Dict[str, Any]]:
    """Return the broker's OAuth blob, bootstrapping from a source
    identity if no broker state exists yet.  Returns None if the broker
    has no credentials and no source is available."""
    own = _read_creds(_state_path())
    if own is not None:
        return own
    for source in (LEGACY_SHARED_CREDS, HOST_CLAUDE_CREDS):
        bootstrapped = _bootstrap_from(source)
        if bootstrapped is not None:
            return bootstrapped
    return None


# ---------------------------------------------------------------------------
# Token-with-lifetime contract
# ---------------------------------------------------------------------------


def _remaining_ms(oauth: Dict[str, Any]) -> int:
    return max(0, int(oauth.get("expiresAt", 0)) - int(time.time() * 1000))


def _provide_token(min_lifetime_ms: int) -> Dict[str, Any]:
    """Return ``{"token": "..."}`` if we can supply a token whose remaining
    lifetime exceeds ``min_lifetime_ms``, else
    ``{"error": "...", "recommendation": "..."}``.

    The contract: the access token's remaining lifetime must be STRICTLY
    GREATER than ``min_lifetime_ms`` so Claude Code, which re-invokes the
    helper after that interval, doesn't cache a token about to die."""
    oauth = _ensure_credentials()
    if oauth is None:
        return {
            "error": "no_credentials",
            "message": (
                "yolo-jail credential broker has no Anthropic identity. "
                "Run `claude` on the host once and complete /login, then retry."
            ),
            "recommendation": "claude /login on the host",
        }

    target_ms = min_lifetime_ms + PROACTIVE_REFRESH_LEAD_MS
    if _remaining_ms(oauth) <= target_ms:
        try:
            oauth = _refresh_and_persist(oauth)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            return {
                "error": "refresh_failed",
                "message": f"refresh against Anthropic failed: {e}",
                "recommendation": (
                    "check network connectivity; if persistent, run "
                    "`claude /login` on the host to mint a fresh refresh token, "
                    "then `rm ~/.local/share/yolo-jail/state/claude-credential-broker/credentials.json` "
                    "and retry"
                ),
            }
        except RuntimeError as e:
            return {
                "error": "refresh_failed",
                "message": str(e),
                "recommendation": (
                    "broker state is missing a refresh token; "
                    "rm ~/.local/share/yolo-jail/state/claude-credential-broker/credentials.json "
                    "and retry to re-bootstrap"
                ),
            }

    remaining = _remaining_ms(oauth)
    if remaining <= min_lifetime_ms:
        # Refreshed but still can't satisfy the requested TTL — the user
        # has set CLAUDE_CODE_API_KEY_HELPER_TTL_MS too high.  Anthropic's
        # access tokens are typically ~1h; nothing we can do about that.
        recommended = max(60_000, remaining // 2)
        return {
            "error": "ttl_too_long",
            "message": (
                f"requested min_lifetime_ms={min_lifetime_ms} exceeds the "
                f"access-token lifetime ({remaining} ms remaining after "
                "refresh).  Anthropic access tokens are typically ~1h."
            ),
            "recommendation": (
                f"lower CLAUDE_CODE_API_KEY_HELPER_TTL_MS to {recommended} "
                "in the jail's ~/.claude/settings.json env block"
            ),
        }

    token = oauth.get("accessToken")
    if not token:
        return {
            "error": "no_access_token",
            "message": "broker state has no accessToken after refresh",
            "recommendation": "run `yolo-claude-credential-broker --self-check`",
        }
    return {"token": token, "expires_in_ms": remaining}


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
    response = _provide_token(min_lifetime_ms)
    session.json(response)
    session.exit(0 if "token" in response else 1)


def _self_check() -> int:
    """Cheap readiness check.  Does NOT call Anthropic — that would burn
    refresh tokens on every `yolo doctor` run.  Reports state-dir layout
    and credential file presence/freshness."""
    state = _state_path()
    print(f"state_dir: {state.parent}")
    if not state.exists():
        print(f"credentials: not yet bootstrapped (will bootstrap on first request)")
        legacy = LEGACY_SHARED_CREDS.exists() and LEGACY_SHARED_CREDS.stat().st_size > 0
        host = HOST_CLAUDE_CREDS.exists() and HOST_CLAUDE_CREDS.stat().st_size > 0
        if legacy:
            print(f"  bootstrap source available: {LEGACY_SHARED_CREDS}")
        if host:
            print(f"  bootstrap source available: {HOST_CLAUDE_CREDS}")
        if not (legacy or host):
            print(
                "  WARNING: no bootstrap source — run `claude /login` on the host first"
            )
            return 1
        return 0
    oauth = _read_creds(state)
    if oauth is None:
        print(f"credentials: {state} present but unreadable / malformed")
        return 2
    remaining = _remaining_ms(oauth)
    print(
        f"credentials: at={_token_fp(oauth.get('accessToken'))} "
        f"rt={_token_fp(oauth.get('refreshToken'))} "
        f"remaining_ms={remaining}"
    )
    return 0


def _refresh_now() -> int:
    """Force a refresh.  Useful for debugging / migration flows."""
    oauth = _ensure_credentials()
    if oauth is None:
        print("no credentials to refresh", file=sys.stderr)
        return 1
    try:
        _refresh_and_persist(oauth)
    except Exception as e:
        print(f"refresh failed: {e}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="yolo-claude-credential-broker")
    p.add_argument("--socket", type=Path, help="unix socket path to bind on")
    p.add_argument("--self-check", action="store_true", help="report status and exit")
    p.add_argument(
        "--refresh-now", action="store_true", help="force a refresh and exit"
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
