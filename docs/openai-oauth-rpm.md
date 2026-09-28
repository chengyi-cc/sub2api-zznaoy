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
available when alternatives are exhausted. With overflow disabled, 15 is a hard ceiling: sticky
continuations do not bypass it. An account-bound continuation cannot be moved
arbitrarily just to avoid the ceiling.

Before forwarding, atomic Redis admission reserves one slot. The first outbound
send claims that reservation without charging twice; retries, additional
WebSocket turns, Excel encrypted-history recovery and Excel tool corrections
acquire their own slots. Failed sends remain counted. Local failures after
reservation can consume a slot conservatively; reservations are not refunded.
Attachment transfers and model-catalog reads are outside this request ceiling.

When no eligible capacity remains and no overflow-enabled account can be used,
ordinary HTTP requests receive 429 and
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


## Excel defaults, overflow, and monitoring

Enabling Excel/BPS now selects RPM (15 if unspecified), cache creation billed as
normal input, and automatic protocol disable on 403 by default. It also selects
`extra.openai_rpm_overflow`; the operator may turn any option off explicitly.
Existing enabled accounts retain saved choices until changed. These options
appear together in the Excel card; native OAuth accounts retain their RPM panel.

With overflow enabled, the scheduler first exhausts ordinary eligible capacity.
Only then can it select the lowest current-RPM account among opted-in candidates.
Group/model eligibility, disabled status, upstream cooldowns, concurrency limits
and account-bound continuation rules still apply. Overflow sends and retries are
counted beyond the configured ceiling. This prevents local RPM exhaustion from
being the only rejection cause; it cannot guarantee upstream acceptance.

Account capacity displays current-minute requests/limit with an RPM suffix;
missing counts show a dash. Both account and candy-monitor tables show an Excel
badge when enabled. A 403 disable persists an `upstream_403` transition and time,
shown as a subtle native/403 badge with details on hover. Stale 403 results cannot
undo a newer protocol activation.

Migration 244 adds the monitor template's `auto_excel_on_incorrect` option
(default false) and a nullable account override (null inherits, true/false
explicitly overrides). Only a completed `incorrect` verdict can enable Excel
for a supported direct OpenAI OAuth account. Invalid format, request failures,
API-key accounts and stale results do not switch protocols. The tested model is
included in Excel's model scope. The protocol change, verdict and scheduler
outbox event commit together. A 403 fallback is not automatically reversed by
subsequent incorrect results; manually re-enable Excel to retry that route.
A manual protocol change made after the probe started also takes precedence.

Verification includes a disposable local PostgreSQL instance: migrations,
template inheritance and both override directions, actual switch persistence,
403 markers and stale-request fencing, plus all scheduler modes and frontend
account/monitor controls. Production accounts and settings are not modified by
these tests.

Validation note: the broader scheduler race run exposed a shared test-state race
in grok_free_quota_gate_test.go (a global atomic value is reset while an earlier
background refresh still reads it). The normal broad suite and scoped RPM, Excel,
Candy, repository and handler race suites passed. That separate Grok test fixture
was not modified by this change. Frontend validation passed 218 tests and typechecking.
