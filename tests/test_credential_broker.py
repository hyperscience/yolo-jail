"""Unit tests for src/credential_broker.py.

Exercises the TTL-assertion contract end-to-end (no real network):
the daemon must refuse to return a token whose remaining lifetime
fails to exceed the requested ``min_lifetime_ms``, with a clear
recommendation.  Also covers bootstrap from a source identity and
the no-credentials error surface.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import credential_broker as cb  # noqa: E402


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Redirect the broker's STATE_DIR / CREDS_PATH at the test's tmp dir."""
    state_dir = tmp_path / "state"
    creds = state_dir / "credentials.json"
    monkeypatch.setattr(cb, "STATE_DIR", state_dir)
    monkeypatch.setattr(cb, "CREDS_PATH", creds)
    monkeypatch.setattr(cb, "LEGACY_SHARED_CREDS", tmp_path / "legacy.json")
    monkeypatch.setattr(cb, "HOST_CLAUDE_CREDS", tmp_path / "host.json")
    return tmp_path, creds


def _write(path: Path, oauth: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claudeAiOauth": oauth}))


def _now_ms() -> int:
    return int(time.time() * 1000)


def test_no_credentials_returns_clear_error(tmp_state):
    """No broker state, no source identity → helpful error + recommendation."""
    response = cb._provide_token(min_lifetime_ms=60_000)
    assert response.get("error") == "no_credentials"
    assert "recommendation" in response
    assert "/login" in response["recommendation"]
    assert "token" not in response


def test_returns_token_when_lifetime_exceeds_ttl(tmp_state):
    """Token with comfortable headroom → returns the access token."""
    _, creds = tmp_state
    _write(
        creds,
        {
            "accessToken": "sk-OK",
            "refreshToken": "rt-OK",
            "expiresAt": _now_ms() + 3_600_000,  # 1h
        },
    )
    response = cb._provide_token(min_lifetime_ms=60_000)
    assert response.get("token") == "sk-OK"
    assert response.get("expires_in_ms", 0) > 60_000


def test_refuses_when_ttl_too_long(tmp_state):
    """Lifetime-after-refresh < requested TTL → ttl_too_long error.

    We simulate "after refresh" by starting from a token with 30 min
    remaining and asking for 60 min worth of headroom.  The proactive
    refresh path tries to call Anthropic; we patch that to leave the
    token unchanged so we can reach the post-refresh assertion.
    """
    _, creds = tmp_state
    _write(
        creds,
        {
            "accessToken": "sk-OK",
            "refreshToken": "rt-OK",
            "expiresAt": _now_ms() + 30 * 60_000,  # 30 min
        },
    )

    def _fake_refresh(refresh_token):
        # Pretend Anthropic re-issued the same shape with the same
        # remaining lifetime — the daemon then has to admit defeat.
        return {
            "access_token": "sk-OK",
            "refresh_token": "rt-OK",
            "expires_in": 30 * 60,  # still 30 min
        }

    with patch.object(cb, "_refresh_upstream", _fake_refresh):
        response = cb._provide_token(min_lifetime_ms=60 * 60_000)  # 60 min

    assert response.get("error") == "ttl_too_long"
    assert "recommendation" in response
    assert "CLAUDE_CODE_API_KEY_HELPER_TTL_MS" in response["recommendation"]
    assert "token" not in response


def test_refresh_failure_surfaces_recommendation(tmp_state):
    """Network / Anthropic failure → refresh_failed with actionable message."""
    import urllib.error

    _, creds = tmp_state
    _write(
        creds,
        {
            "accessToken": "sk-OLD",
            "refreshToken": "rt-OLD",
            "expiresAt": _now_ms() + 1_000,  # ~expired, will trigger refresh
        },
    )

    def _boom(refresh_token):
        raise urllib.error.URLError("connection refused")

    with patch.object(cb, "_refresh_upstream", _boom):
        response = cb._provide_token(min_lifetime_ms=60_000)

    assert response.get("error") == "refresh_failed"
    assert "recommendation" in response


def test_proactive_refresh_when_close_to_expiry(tmp_state):
    """Token whose remaining lifetime ≤ TTL+lead → daemon refreshes
    in-band and returns the new token."""
    _, creds = tmp_state
    _write(
        creds,
        {
            "accessToken": "sk-OLD",
            "refreshToken": "rt-OLD",
            "expiresAt": _now_ms() + 90_000,  # 90s left, lead is 60s
        },
    )

    refreshed = {
        "access_token": "sk-NEW",
        "refresh_token": "rt-NEW",
        "expires_in": 3600,
    }

    with patch.object(cb, "_refresh_upstream", lambda rt: refreshed):
        response = cb._provide_token(min_lifetime_ms=60_000)

    assert response.get("token") == "sk-NEW"
    # And the new state should be on disk.
    on_disk = json.loads(creds.read_text())
    assert on_disk["claudeAiOauth"]["accessToken"] == "sk-NEW"
    assert on_disk["claudeAiOauth"]["refreshToken"] == "rt-NEW"


def test_bootstrap_from_legacy_shared_creds(tmp_state):
    """No broker state but legacy shared file present → bootstrap forks
    the chain by performing an immediate refresh, then writes the
    NEW (post-refresh) tokens to the broker's state file."""
    tmp, creds = tmp_state
    legacy = tmp / "legacy.json"
    _write(
        legacy,
        {
            "accessToken": "sk-LEGACY",
            "refreshToken": "rt-LEGACY",
            "expiresAt": _now_ms() + 3_600_000,
        },
    )
    refreshed = {
        "access_token": "sk-FORKED",
        "refresh_token": "rt-FORKED",
        "expires_in": 3600,
    }

    with patch.object(cb, "_refresh_upstream", lambda rt: refreshed):
        response = cb._provide_token(min_lifetime_ms=60_000)

    assert response.get("token") == "sk-FORKED"
    # Broker state now holds the forked tokens.
    on_disk = json.loads(creds.read_text())
    assert on_disk["claudeAiOauth"]["refreshToken"] == "rt-FORKED"
    # Legacy file is untouched.
    legacy_data = json.loads(legacy.read_text())
    assert legacy_data["claudeAiOauth"]["refreshToken"] == "rt-LEGACY"


def test_bootstrap_prefers_legacy_over_host(tmp_state):
    """When both bootstrap sources exist, the legacy shared-creds
    file wins — it's already on a forked chain from the previous
    MITM-broker era and won't race with host Claude Code."""
    tmp, creds = tmp_state
    _write(
        tmp / "legacy.json",
        {
            "accessToken": "sk-LEGACY",
            "refreshToken": "rt-LEGACY",
            "expiresAt": _now_ms() + 3_600_000,
        },
    )
    _write(
        tmp / "host.json",
        {
            "accessToken": "sk-HOST",
            "refreshToken": "rt-HOST",
            "expiresAt": _now_ms() + 3_600_000,
        },
    )

    seen_refresh_tokens = []

    def _capture(rt):
        seen_refresh_tokens.append(rt)
        return {
            "access_token": "sk-FORKED",
            "refresh_token": "rt-FORKED",
            "expires_in": 3600,
        }

    with patch.object(cb, "_refresh_upstream", _capture):
        cb._provide_token(min_lifetime_ms=60_000)

    assert seen_refresh_tokens == ["rt-LEGACY"]


def test_bootstrap_falls_back_to_host_when_legacy_missing(tmp_state):
    """No legacy file, host has creds → bootstrap from host."""
    tmp, creds = tmp_state
    _write(
        tmp / "host.json",
        {
            "accessToken": "sk-HOST",
            "refreshToken": "rt-HOST",
            "expiresAt": _now_ms() + 3_600_000,
        },
    )

    seen_refresh_tokens = []

    def _capture(rt):
        seen_refresh_tokens.append(rt)
        return {
            "access_token": "sk-FORKED",
            "refresh_token": "rt-FORKED",
            "expires_in": 3600,
        }

    with patch.object(cb, "_refresh_upstream", _capture):
        cb._provide_token(min_lifetime_ms=60_000)

    assert seen_refresh_tokens == ["rt-HOST"]


def test_bad_request_min_lifetime(tmp_state):
    """Helper sends garbage min_lifetime_ms → handler returns a
    structured bad_request that the helper surfaces as exit-1+stderr."""
    # Drive _handle directly with a fake session.
    from src import host_service

    class _FakeSession:
        request = {"min_lifetime_ms": "not-a-number"}
        jail_id = "test"
        outputs = []
        exit_code = None

        def stdout(self, data):
            self.outputs.append(("out", data))

        def stderr(self, data):
            self.outputs.append(("err", data))

        def json(self, obj):
            self.outputs.append(("json", obj))

        def exit(self, code):
            self.exit_code = code

    s = _FakeSession()
    cb._handle(s)
    assert s.exit_code == 2
    assert any(kind == "json" and obj.get("error") == "bad_request" for kind, obj in s.outputs)
