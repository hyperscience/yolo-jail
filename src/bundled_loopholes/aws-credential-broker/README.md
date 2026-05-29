# aws-credential-broker loophole (bundled)

Routes AWS credentials into jails without copying long-lived keys. The host daemon mints short-lived STS sessions from your host AWS configuration and answers requests on a unix socket; inside the jail, the AWS SDK reaches the daemon via the standard `credential_process` mechanism in `~/.aws/config`.

## Architecture

- **Host daemon** (`yolo-aws-credential-broker`) — spawned per-jail by the loopholes pipeline. Reads `~/.config/yolo-jail/aws-broker.jsonc`, shells out to `aws sts get-session-token` (or `aws sts assume-role` if `role_arn` is set), caches the result, refreshes proactively when remaining lifetime would not exceed the jail's TTL request.
- **Jail-side helper** (`/usr/local/bin/yolo-aws-creds`, baked into the jail image) — invoked by the AWS SDK as `credential_process`. Opens the socket, prints AWS-SDK-shape JSON (`{Version, AccessKeyId, SecretAccessKey, SessionToken, Expiration}`) to stdout. On any error, exits non-zero with a clear stderr explanation.

## Configuration

For most Bedrock setups **no broker config is needed** — the daemon
auto-detects `AWS_PROFILE` and `AWS_REGION` from your host's
`~/.claude/settings.json` env block (the same place Claude Code's
`/setup-bedrock` wizard writes them).  Run `yolo-aws-credential-broker
--self-check` to see what it resolved.

To override or add settings absent from `~/.claude/settings.json`,
create `~/.config/yolo-jail/aws-broker.jsonc`:

```jsonc
{
  "profile": "bedrock",
  "region": "us-east-1",
  "session_duration_seconds": 3600,
  // Optional: assume a role across the session.  When set, the daemon
  // calls sts:AssumeRole instead of sts:GetSessionToken.
  // "role_arn": "arn:aws:iam::123456789012:role/jail-bedrock"
}
```

Resolution order (highest priority first):

1. Explicit value in `~/.config/yolo-jail/aws-broker.jsonc`.
2. Host's `~/.claude/settings.json` env block (`AWS_PROFILE`, `AWS_REGION`).
3. Broker process environment (e.g. `AWS_PROFILE` set in the shell that
   spawned `yolo`).
4. AWS SDK default chain.

This means a default-shell `AWS_PROFILE` that's incompatible with
Bedrock (a common SSO-only profile, for instance) won't override the
Bedrock-specific profile pinned in `settings.json` — matching what
Claude Code itself does on the host.

## TTL contract

Same shape as the Anthropic broker: when the jail asks for a token, it includes a `min_lifetime_ms` floor. The daemon must return credentials whose `Expiration` exceeds that floor; if STS won't issue a session that long (or the user requested a TTL longer than the maximum session duration), the helper exits non-zero with a recommendation telling the user what to lower.

## Bedrock auto-config

When this loophole is active, the jail entrypoint also writes `CLAUDE_CODE_USE_BEDROCK=1` and `AWS_REGION=<configured>` into the jail's `~/.claude/settings.json` `env` block. Claude Code then routes model calls to Bedrock by default. To opt out (run direct Anthropic alongside Bedrock for some workspaces), disable this loophole in that workspace's `yolo-jail.jsonc`:

```jsonc
{
  "loopholes": {
    "aws-credential-broker": { "enabled": false }
  }
}
```

The Anthropic-direct broker (`claude-credential-broker`) is independent and stays active.

## Files

| File | Location | Purpose |
|---|---|---|
| `manifest.jsonc` | bundled (in wheel, read-only) | Loophole definition |
| `aws-broker.jsonc` | `~/.config/yolo-jail/` | User config (profile, region, role_arn, duration) |
| `cache.json` | `~/.local/share/yolo-jail/state/aws-credential-broker/` | Last-minted STS session, owner-only `0600` |

## Operations

```bash
# Self-check (validates host aws CLI + config)
yolo-aws-credential-broker --self-check

# Force a fresh STS call now (debugging)
yolo-aws-credential-broker --refresh-now

# Per-jail logs
ls ~/.local/share/yolo-jail/logs/host-service-aws-credential-broker-*.log
```
