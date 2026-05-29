"""Unit tests for src/aws_credential_broker.py.

Exercises the same TTL contract the Anthropic broker enforces, plus
the AWS-specific shape (role_arn handling, STS call shaping, session
lifetime parsing).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import aws_credential_broker as ab  # noqa: E402


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    cache = state_dir / "cache.json"
    cfg = tmp_path / "aws-broker.jsonc"
    monkeypatch.setattr(ab, "STATE_DIR", state_dir)
    monkeypatch.setattr(ab, "CACHE_PATH", cache)
    monkeypatch.setattr(ab, "CONFIG_PATH", cfg)
    yield tmp_path, cache, cfg


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _fake_sts(expires_in_seconds: int) -> dict:
    """Mimic STS GetSessionTokenResponse.Credentials shape."""
    exp = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return {
        "AccessKeyId": "AKIAFAKE",
        "SecretAccessKey": "secretFAKE",
        "SessionToken": "tokenFAKE",
        "Expiration": _iso(exp),
    }


def _patched_call(creds_to_return):
    """Patch ab._call_sts to return a constant value, recording calls."""
    calls = []

    def _fake(cfg):
        calls.append(cfg)
        return creds_to_return

    return _fake, calls


def test_returns_credentials_when_lifetime_exceeds_ttl(tmp_state):
    _, cache, _ = tmp_state
    fresh = _fake_sts(3600)
    fake, calls = _patched_call(fresh)
    with patch.object(ab, "_call_sts", fake):
        response = ab._provide_credentials(min_lifetime_ms=60_000)
    assert response.get("Version") == 1
    assert response.get("AccessKeyId") == "AKIAFAKE"
    assert response.get("SecretAccessKey") == "secretFAKE"
    assert response.get("SessionToken") == "tokenFAKE"
    assert response.get("Expiration") == fresh["Expiration"]
    assert response.get("expires_in_ms", 0) > 60_000
    # And we cached the response.
    cached = json.loads(cache.read_text())
    assert cached["Credentials"]["AccessKeyId"] == "AKIAFAKE"
    assert len(calls) == 1


def test_uses_cached_credentials_when_fresh_enough(tmp_state):
    _, cache, _ = tmp_state
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "Credentials": _fake_sts(3600),
                "MintedAtMs": int(time.time() * 1000),
            }
        )
    )
    fake, calls = _patched_call(_fake_sts(3600))
    with patch.object(ab, "_call_sts", fake):
        response = ab._provide_credentials(min_lifetime_ms=60_000)
    assert "AccessKeyId" in response
    # No STS call when cache is fresh.
    assert calls == []


def test_proactive_refresh_when_close_to_expiry(tmp_state):
    """Cache exists but remaining ≤ TTL+lead → re-mint."""
    _, cache, _ = tmp_state
    cache.parent.mkdir(parents=True, exist_ok=True)
    # 90 s remaining; lead is 60 s → refresh.
    cache.write_text(
        json.dumps(
            {
                "Credentials": _fake_sts(90),
                "MintedAtMs": int(time.time() * 1000) - 3500_000,
            }
        )
    )
    fresh = _fake_sts(3600)
    fake, calls = _patched_call(fresh)
    with patch.object(ab, "_call_sts", fake):
        response = ab._provide_credentials(min_lifetime_ms=60_000)
    assert response.get("Expiration") == fresh["Expiration"]
    assert len(calls) == 1


def test_refuses_when_ttl_too_long(tmp_state):
    """Lifetime-after-mint < requested TTL → ttl_too_long."""
    fake, _ = _patched_call(_fake_sts(900))  # 15 min, the STS minimum
    with patch.object(ab, "_call_sts", fake):
        response = ab._provide_credentials(min_lifetime_ms=60 * 60_000)  # 1 h floor
    assert response.get("error") == "ttl_too_long"
    assert "recommendation" in response
    assert "CLAUDE_CODE_API_KEY_HELPER_TTL_MS" in response["recommendation"]
    assert "AccessKeyId" not in response


def test_sts_failure_surfaces_recommendation(tmp_state):
    def _boom(cfg):
        raise RuntimeError("AccessDenied: not authorized")

    with patch.object(ab, "_call_sts", _boom):
        response = ab._provide_credentials(min_lifetime_ms=60_000)
    assert response.get("error") == "sts_failed"
    assert "AccessDenied" in response["message"]
    assert "recommendation" in response


def test_sts_timeout_surfaces_recommendation(tmp_state):
    import subprocess as sp

    def _boom(cfg):
        raise sp.TimeoutExpired(cmd=["aws"], timeout=30)

    with patch.object(ab, "_call_sts", _boom):
        response = ab._provide_credentials(min_lifetime_ms=60_000)
    assert response.get("error") == "sts_timeout"
    assert "recommendation" in response


def test_resolve_duration_clamps_to_sts_bounds(tmp_state):
    assert ab._resolve_duration({}) == ab.DEFAULT_DURATION_S
    assert ab._resolve_duration({"session_duration_seconds": 60}) == ab.STS_MIN_DURATION_S
    assert (
        ab._resolve_duration({"session_duration_seconds": 100_000}) == ab.STS_MAX_DURATION_S
    )
    assert ab._resolve_duration({"session_duration_seconds": 7200}) == 7200


def test_resolve_role_arn_validates_shape(tmp_state):
    valid = "arn:aws:iam::123456789012:role/my-role"
    assert ab._resolve_role_arn({"role_arn": valid}) == valid
    assert ab._resolve_role_arn({"role_arn": "garbage"}) is None
    assert ab._resolve_role_arn({"role_arn": "arn:aws:s3:::bucket"}) is None
    assert ab._resolve_role_arn({}) is None


def test_aws_argv_includes_profile_and_region(tmp_state):
    cfg = {"profile": "bedrock", "region": "us-east-1"}
    argv = ab._aws_argv(cfg, "get-session-token", "--duration-seconds", "3600")
    assert argv[:3] == ["aws", "sts", "get-session-token"]
    assert "--profile" in argv and "bedrock" in argv
    assert "--region" in argv and "us-east-1" in argv
    assert argv[-2:] == ["--output", "json"]


def test_assume_role_argv_uses_role_arn(tmp_state, monkeypatch):
    """Sanity-check the argv shape for the AssumeRole branch.  Asserts
    that the daemon emits ``aws sts assume-role --role-arn ...
    --role-session-name yolo-jail --duration-seconds N`` when role_arn
    is set."""
    captured = {}

    class _CompletedProcess:
        returncode = 0
        stdout = json.dumps({"Credentials": _fake_sts(3600)})
        stderr = ""

    def _fake_run(argv, **kw):
        captured["argv"] = argv
        return _CompletedProcess()

    monkeypatch.setattr(ab.subprocess, "run", _fake_run)
    cfg = {
        "profile": "p",
        "region": "us-west-2",
        "role_arn": "arn:aws:iam::123456789012:role/r",
    }
    creds = ab._call_sts(cfg)
    assert creds["AccessKeyId"] == "AKIAFAKE"
    assert "assume-role" in captured["argv"]
    assert "--role-arn" in captured["argv"]
    assert "arn:aws:iam::123456789012:role/r" in captured["argv"]
    assert "--role-session-name" in captured["argv"]
    assert "yolo-jail" in captured["argv"]


def test_load_config_handles_jsonc_comments(tmp_state):
    _, _, cfg = tmp_state
    cfg.write_text(
        """// Top comment
{
  "profile": "bedrock",  // inline
  /* block
     comment */
  "session_duration_seconds": 1800
}
"""
    )
    loaded = ab._load_config()
    assert loaded == {"profile": "bedrock", "session_duration_seconds": 1800}


def test_bad_request_min_lifetime(tmp_state):
    class _FakeSession:
        request = {"min_lifetime_ms": "nope"}
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
    ab._handle(s)
    assert s.exit_code == 2
    assert any(
        kind == "json" and obj.get("error") == "bad_request"
        for kind, obj in s.outputs
    )
