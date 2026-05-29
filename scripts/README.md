# yolo-jail scripts

Host-side scripts that support the jail. These run on the host, not inside any jail.

## Anthropic / AWS credential delivery — moved to bundled loopholes

Anthropic OAuth and AWS Bedrock credentials are now served to jails via
unix-socket loopholes that ship inside the wheel:

- [`src/bundled_loopholes/claude-credential-broker/`](../src/bundled_loopholes/claude-credential-broker/)
- [`src/bundled_loopholes/aws-credential-broker/`](../src/bundled_loopholes/aws-credential-broker/)

Earlier ad-hoc scripts and systemd units (`claude-token-refresher.timer`,
the MITM-based `claude-oauth-broker.service`) are gone. `just deploy`
removes any leftover artifacts on upgrade.
