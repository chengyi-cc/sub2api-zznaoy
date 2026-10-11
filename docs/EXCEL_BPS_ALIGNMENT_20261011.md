# Excel/BPS alignment ? 2026-10-11

Local baseline: `fb30198e0`. Reference: `ranxi2001/sub2api` tag `v2.10.5`, commit `d977e7a5e`.

## Implemented

- Official Excel browser OAuth and paired-client refresh/import; independent encrypted grants for existing password/TOTP accounts.
- Profile-aware local-worker queue, conditional grant writes, account identity checks, preparation marker, successful automatic activation, and cancellation of activation intent.
- Excel-only credentials excluded from native Codex, WebSocket, image and incompatible model paths.
- Excel 401 preserves its RT; subsequent refresh uses the Excel client. Invalid-grant errors survive sanitization as a fixed allowlisted code. Refresh/401 interleavings do not delete or overwrite newer authorization.
- Model access discovery, inference effort metadata, per-account/model permission cooldown, safe credential state, and account-test upstream evidence.
- Global BPS switch in settings/public settings; managed runtime pinned to 2.10.5 release digests.
- Dynamic tool catalog precedence, compact schema descriptions, declared argument validation, catalog-derived transport examples, unreadable new-agent-task validation, in-band failure classification and cancellation terminals.
- Dedicated 120-second BPS progress timeout; optional Redis-backed scoped tool replay/catalog state, off by default. Persistence failures prevent tool delivery.
- Repository scheduler projection preserves preparation state. Quality-triggered activation prepares authorization before routing. Production atomic account edits validate readiness before saving.
- Credential-key initialization/reset detects existing independent Excel ciphertext and refuses silent key replacement.

## Intentional local behavior

Retain explicit/default model selection, raw script/name compatibility, FUNCTION_CMD metadata passthrough, native fallback for encrypted history, official image attachments with existing detail behavior, and the 920000-token default compaction threshold. No local image relay, automatic image dropping, BPS image generation or unrelated reference features were imported.

## Validation

- Full unit runs passed for service, basispoints, handlers, admin handlers, DTOs, OpenAI package, configuration and migrations. API contract and server route suites passed after adding the public setting field.
- Real disposable PostgreSQL 17 tests passed for dual-grant storage, task exclusion, transactional rollback, engine selection, activation/cancellation, per-account bulk preparation, quality policy and credential recovery. Tests created and removed their own databases on a temporary loopback server.
- Redis state tests passed with an isolated Redis test server, covering cross-client access, compare-and-swap, isolation, expiry and bounded storage.
- Basispoints full race suite and focused Excel authorization race tests passed. Cross-instance bridge fixtures verify that a storage failure emits no tool call.
- Frontend production build/typecheck and targeted lint passed. Seven frontend suites passed (107 tests); the additional admin evidence rendering test and affected frontend subset passed afterward (99 tests).
- Linux amd64 builds passed, including the frontend via the `embed` build tag.
- The broad Windows run also exposed five platform-dependent failures in existing file-key/runtime tests: directory fsync/access permissions, Unix executable mode and Linux-only worker startup. These were not hidden or changed; actual Linux worker execution was unavailable because the local WSL VM could not start. The reference runtime packaging and Excel profile contract were inspected; no real account login or upstream generation was performed.

## Upgrade

Apply migration `247_openai_excel_dual_oauth.sql` with the new backend/frontend. Back up the database and its credential encryption key. Managed workers automatically use the pinned 2.10.5 package; externally managed workers must be upgraded separately. Keep Redis sharing disabled unless the deployment needs it and all instances use the updated code. Existing tokens alone cannot be converted to Excel OAuth; use official browser authorization or saved password/TOTP credentials.

All validation data is synthetic. These changes were prepared locally; they have not been pushed or deployed and do not establish upstream model quality.
