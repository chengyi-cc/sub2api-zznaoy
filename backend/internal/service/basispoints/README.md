# Structured output compatibility

The adapter accepts Responses requests using text.format.type=json_object or
json_schema. BPS rejects native text/response_format request fields, so the
adapter sends the output requirements as developer instructions and validates
the final response locally. This does not provide upstream constrained decoding.

- Preserve the requested model and account. Model-access rejections remain
  upstream errors; structured output support does not grant model permissions.
- Return a successful final answer only when it is valid JSON and, for
  json_schema, satisfies the supplied schema. Do not retry, repair, strip fences,
  or replace invalid model output with fabricated data.
- Withhold structured message text until the terminal response is validated.
  Reconstruct message events from that validated response, rather than replaying
  unvalidated deltas. Ordinary text requests remain incremental.
- Preserve tool continuations and explicit refusals as protocol items, outside
  the final-answer JSON contract. Preserve upstream failure/incomplete status
  without exposing partial structured message text.
- Resolve schema references inside the submitted document only. Never fetch
  remote schemas or read local files. Limit schemas to 1 MiB and answer text to
  16 MiB, subject to the SSE event limit. The gateway supplies its configured
  `gateway.max_line_size` to both BPS stream readers; the standalone adapter
  defaults to 16 MiB. This does not change the independent answer-text or
  tool-repair budgets.

# Compaction policy and request timing

`gateway.excel_bps.compaction_threshold_tokens` (environment variable
`GATEWAY_EXCEL_BPS_COMPACTION_THRESHOLD_TOKENS`) defaults to 920000. Positive
values request that inline-compaction
threshold. Zero sends an empty `context_management` array when the caller has
not supplied one. This is a request policy, not a guarantee that the upstream
never compacts. Caller-supplied arrays, including empty ones, take precedence.
Explicit `compaction_trigger` items and the `/responses/compact` path remain
available. Negative configured values fail startup. Tool catalog inheritance
never inherits a previous request's compaction policy.

Do not infer the BPS context limit from a native client model manifest. Before
raising or disabling automatic compaction, verify the deployed BPS endpoint's
context capacity and the caller's explicit compaction path on a test instance.
Rollback is the previously deployed threshold unless separately customized.

Reference policy review (2026-09-28 UTC): ranxi2001/sub2api production at
`fe27f9895a75e12562f33eff70f21c660329a684` defaults to 920000.
Kaixxrua/excel-codex-bridge main at
`168ed925548d6149a9dde15651b40211a926a3c9` instead pairs client/backend limits:
180000/200000 for its 272000-token aliases, and 826000/872000 for its
918000-token long-context aliases. Its reported backend measurements are not
a capacity guarantee for this deployment. Adopt the paired policy only after
verifying this route and client; copying a larger backend threshold alone does
not synchronize the client's independent compaction limit.

Enable `gateway.excel_bps.log_request_timing` with
`GATEWAY_EXCEL_BPS_LOG_REQUEST_TIMING=true` to emit one
`excel_bps_request_timing` event per completed gateway call, including failures.
The existing logger carries request correlation IDs; the event adds an internal
account ID and numeric/boolean policy and timing fields. It does not log request
content, tool arguments, upstream bodies, hostnames or credentials. Disabled by
default. Server log ingestion and retention must also be enabled to query this
event through the admin system-log endpoint.

- Stage timestamps are milliseconds since server ingress when available,
  otherwise the gateway's request start. `forward_started_ms` captures earlier
  gateway work; it does not measure time before the server accepted the request.
- `http_transport` contains one entry per model HTTP attempt. Each entry uses
  its own attempt start for connection, DNS, TLS, request-write and first-byte
  offsets. Missing callbacks are omitted. Proxied/custom transports may not
  expose every callback; these measurements do not isolate upstream compute.
- `http_wait_total_ms` includes waiting until HTTP headers or an error, summed
  across the initial attempt, optional reasoning recovery and tool repairs.
  It is not first-token latency. Attachment transfers have a separate duration.
- `first_event_ms` and `first_output_ms` observe parsed upstream events; an
  output item can be reasoning or compaction, not necessarily visible text.
- `compaction_duration_ms` sums observed added-to-done compaction intervals.
  Missing start events yield no invented duration. It does not capture work
  performed before the upstream sends the start event. `compactions_completed`
  counts completion events; tool-correction streams are covered by repair time.
- `first_tool_buffer_ms` measures the first native tool-done event through the
  first translated tool written into the bridge pipe. It can include compaction,
  validation, repair and backpressure; it does not prove client execution.
- `validation_duration_ms` includes `repair_duration_ms`; do not add them.
  `tool_repairs` counts explicit corrections. `last_message_to_completion_ms`
  measures the tail after the last observed upstream message, not delivery time.
- Cancellation can end logging before every upstream stage finishes. Missing
  stages stay absent rather than being reported as zero.

For diagnosis, correlate a single request across client telemetry, access/Ops
logs and this timing event before changing thresholds. Test one change at a
time using the same model, effort, prompt/tool payload and comparable context
size. Preserve atomic tool-batch validation: dispatching calls before the
authoritative terminal response can execute a call later rejected by the batch.
Client time-to-headers overlaps server request duration; do not add it to the
server duration or label their difference as purely network overhead. Usage
records may store `client:<x-client-request-id>` rather than the response's
`x-request-id`; verify the correlation key before interpreting missing logs.
