# Source attribution

Ported into chengyi-cc/sub2api-zznaoy from ranxi2001/sub2api production
(055a1cd1470b3b866d04aad8d1514bd34cac73ab), checked on 2026-09-25.
Includes release v2.8.11 and subsequent structured-output and billing fixes. The gateway
integration uses this repository's account and transport services; no unrelated
ticket, Copilot, account operations or deployment features are imported.

This package is adapted from hloolx/codex2api, whose README declares MIT License.
Original author of the Basispoints changes: hloolx. Source: https://github.com/hloolx/codex2api

Requested commits and required intervening fixes, in order:

- 9d02d3f5e5d69632ebb9590082a833c0a0916356 — initial Basispoints routing.
- c125e560eefb5fd15c995943eb1e111795b0635f — tool-loop identity and replay.
- 20ff3e860d9a149e2df731e37ba1d9b56ae053fc — envelope formatting and model access errors.
- d39f7e3697aab342e303bf4be0142e39b6a58515 — tool catalog and complete terminal items.
- 4dea83ec53b7668419edd2a9a9dd40fb55fdaacd — HTTPS image references.

Only the protocol package is imported; Sub2API supplies its own account setting,
OAuth credential lifecycle, proxy transport, usage recording and frontend.
The repositories have different layouts; this is a source port, not a Git merge
of the other application's deployment, database or account pool implementation.

Additional review reference: JaxsonWang/cpa-plugin-oai-basispoints at
05b2d97efa1bd117da6bd4d362d6e88f8e483680. Its tool/args envelope examples
were compared with this package. We retain scoped caches, incremental text
streaming and multiple-terminal-tool handling rather than its global call-ID
cache and single-transport extraction. No CPA plugin ABI is imported.


Protocol compatibility update, reviewed 2026-09-26:
- Reviewed ranxi2001/sub2api production 594cdf0d6027fe7097ef42fe029c22713b9cc989
  (latest release v2.8.14). Ported only single-invocation recovery and safe content
  diagnostics (3fb62aeb5 / 1fb21644f), plaintext collaboration argument metadata
  (d815d4a23), and raw string-code transport (5c1839b28), with their tests.
- Compared Kaixxrua/excel-codex-bridge main
  8a277df (latest release v0.5.1): parallel tool handling, official attachments,
  image generation and Codex sign-in changes. Its skip-unusable-calls behavior
  is not imported; this bridge validates every terminal tool before dispatch.
- Local fix resolves one optional `functions.` host display prefix for wrapped
  catalog calls just as for direct calls. Exact declared names take priority;
  unknown tools, ambiguous calls and arbitrary namespace suffixes remain denied.
- The native fallback routing and local image relay from ranxi are not imported.
  Our official attachment uploads, per-account model routing, templates and
  candy monitoring remain independent integrations.


Additional gateway/compatibility sync, 2026-09-26, same pinned references:
- Adapted ranxi commits 3355086e0 (bulk settings), 6df263ef2 (opt-in protocol-only
  HTTP 403 disable), 19becb835 (quota snapshots and immediate 429 cooldown),
  49bb7b049 (client-visible cache-write accounting), 1dfbc6f4d (bounded admin
  tool roundtrip), beb86d6ca (requested effort), a2d3ea37d (x/image v0.45.0).
- Reviewed bridge 2181910 (parallel calls); implemented the behavior in Go with
  full terminal validation, per-call identity preservation, parallel setting
  enforcement and result-round iteration tracking. No silent tool truncation.
- Implemented pre-dispatch native-capability routing, broader than ranxi's
  forced/high-context hosted-tool check; this never retries an already-sent call.
- Added missing Excel keys to our scheduler projection. Original model defaults,
  OAuth lifecycle, official attachments and existing candy monitor remain intact.
- Earlier note excluding fallback refers to the prior tool-only update. The
  current capability routing and deferred/non-applicable features are documented
  in docs/EXCEL_BPS.md; no Excel image-generation endpoints are imported.

Final head check on 2026-09-26: ranxi production advanced to
f671a8d30c34706d8526accadf6a6ad5f40f855e (client onboarding documentation only,
no feature-code differences from the reviewed 594cdf0d6); latest release remains
v2.8.14. The bridge main remains 8a277df / v0.5.1.
