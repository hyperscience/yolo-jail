# claude-credential-broker loophole (bundled)

Replacement for `claude-oauth-broker`. Same goal — give jail-side Claude Code access to your Anthropic identity — but with no TLS interception, no CA, and no jail-side daemon.

## Architecture

- **Host daemon** (`yolo-claude-credential-broker`) — spawned per-jail by the loopholes pipeline. Owns its own OAuth refresh-token chain in `~/.local/share/yolo-jail/state/claude-credential-broker/credentials.json`. Refreshes proactively. Listens on a unix socket bind-mounted into the jail at `/run/yolo-services/claude-credential-broker.sock`.
- **Jail-side helper** (`/usr/local/bin/yolo-claude-creds`, baked into the jail image) — invoked by Claude Code as `apiKeyHelper`. Opens the socket, sends `{min_lifetime_ms: <CLAUDE_CODE_API_KEY_HELPER_TTL_MS>}`, prints the access token to stdout. On any error, exits non-zero with a clear stderr explanation that Claude Code surfaces to the agent.

## Bootstrap

On first run the daemon needs an Anthropic identity. It tries, in order:

1. The legacy `~/.local/share/yolo-jail/home/.claude-shared-credentials/.credentials.json` from the old MITM broker — auto-migrated if present (one-shot capture, then the broker forks its chain by performing an immediate refresh).
2. The host's own `~/.claude/.credentials.json` — same one-shot capture.

The capture step calls Anthropic's refresh endpoint immediately so the broker holds a fresh refresh token *before* the source identity does its next refresh. This is what keeps the two chains independent — the [2026-04-23 incident comments in the legacy broker](../../oauth_broker.py) describe the failure mode that motivates the fork.

If neither source has credentials, the helper fails non-zero with a message telling the user to run `claude` on the host once and complete `/login` first.

## TTL contract

Claude Code re-invokes `apiKeyHelper` every `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` ms (default 60_000 in the jail's settings.json). The daemon must return a token whose remaining lifetime is **strictly greater** than that TTL — otherwise Claude Code would cache a token that expires before the next refetch and start 401'ing.

When the daemon can't provide such a token (e.g. refresh failed, refresh token revoked, Anthropic outage), the helper exits non-zero with a stderr message including a recommended `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` value the user can set to recover.

## Files

| File | Location | Purpose |
|---|---|---|
| `manifest.jsonc` | bundled (in wheel, read-only) | Loophole definition |
| `credentials.json` | `~/.local/share/yolo-jail/state/claude-credential-broker/` | The broker's OAuth state. Owner-only `0600`. |

No CA, no certs, no flock — the daemon serializes refreshes naturally via the socket accept queue.

## Operations

```bash
# Self-check (also runs automatically via `yolo doctor` → manifest.doctor_cmd)
yolo-claude-credential-broker --self-check

# Force a refresh now (debugging)
yolo-claude-credential-broker --refresh-now

# Per-jail logs
ls ~/.local/share/yolo-jail/logs/host-service-claude-credential-broker-*.log
```
