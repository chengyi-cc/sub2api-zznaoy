# OpenAI OAuth RPM controls

OpenAI OAuth accounts can enable a strict per-minute request ceiling in the
create, edit, or bulk-edit account dialog. Enabling the control without a custom
value saves **15 requests per minute**. Existing accounts remain unchanged until
enabled; a missing or zero `extra.base_rpm` disables the limit. This is an
operator-configured ceiling, not an assertion of the upstream provider's limit.

Requests share the account counter across sessions and gateway instances.
Credential shadows use their parent account's counter. Redis server time defines
fixed calendar-minute buckets; this is not a rolling 60-second limit. The
counter expires after 120 seconds, but capacity resets at the next minute.

Scheduling prefers accounts with more headroom. At 80% utilization (12 of 15),
movable sticky requests yield to other accounts. Capacity above 80% remains
available when alternatives are exhausted. The hard ceiling is 15: sticky
continuations do not bypass it. An account-bound continuation cannot be moved
arbitrarily just to avoid the ceiling.

Before forwarding, atomic Redis admission reserves one slot. The first outbound
send claims that reservation without charging twice; retries, additional
WebSocket turns, Excel encrypted-history recovery and Excel tool corrections
acquire their own slots. Failed sends remain counted. Local failures after
reservation can consume a slot conservatively; reservations are not refunded.
Attachment transfers and model-catalog reads are outside this request ceiling.

When no eligible capacity remains, ordinary HTTP requests receive 429 and
`Retry-After`. Streams already started use the existing in-stream error path;
an already committed HTTP status cannot be rewritten. WebSocket clients receive
a retry-later close. A configured account cannot bypass admission when Redis is
unavailable: that condition returns 503 instead. Account listings show current
RPM and a pause/reset indication. Existing Anthropic strategies are preserved.

The RPM setting does not alter the Excel compaction threshold (920000 tokens),
repair network disconnections, or impose a limit on local tool execution.

## Source and adaptation

Ported from ranxi2001/sub2api commit
`833aa01319b9726d14488eb4430fb529770b97a6`, authored by sylarchen1389:
https://github.com/ranxi2001/sub2api/commit/833aa01319b9726d14488eb4430fb529770b97a6

The port uses this repository's service wiring and HTTP/plugin transport, keeps
its account eligibility and credential handling, and adds admission to its
Excel/BPS retry paths. Unrelated Mihomo, Copilot, and ticket-admission changes
were excluded.

Validation covers 100 concurrent acquisitions admitting exactly 15 requests,
minute reset and stale buckets, retry accounting without double charging,
Excel recovery at the hard ceiling, shared parent counters, scheduler modes,
cache failures, 429/retry metadata, frontend defaults and account dialog flows.
