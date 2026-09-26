# Tool catalog fixtures

exec_catalog_names.json records the 64 nested tool names visible in the Codex
runtime during the 2026-09-26 compatibility audit and whether their single input
is an object or string. It contains no credentials, user prompts, tool results or
business data. It is test data, not a production allowlist.

The matrix uses synthetic payloads to check identity and byte-preserving
transport, not to assert those payloads satisfy every tool's application schema.
Tests never execute these tools, install plugins, send messages or modify user
settings. Runtime names are always derived from the current client contract.
