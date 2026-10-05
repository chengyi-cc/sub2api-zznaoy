export default {
  astraBorrow: {
    title: 'Astra gateway borrowing', description: 'Borrow a source route after independent target verification.',
    refresh: 'Refresh status', guide: 'Start with one source whose Astra quality you have verified, one target, and a fixed exit. Saving an enabled configuration acquires a route and verifies the target with two requests. Sources and targets must not overlap.',
    boundary: 'HTTP requests are supported at the existing /v1/responses endpoint. Disable Excel on selected accounts. Native WebSocket connections are unavailable for selected targets. Probes consume upstream quota; a passing probe does not prove lasting model quality.',
    enabled: 'Enable gateway borrowing', on: 'Enabled', off: 'Disabled', preparing: 'Preparing', idle: 'Routes renew on demand',
    search: 'Search account name or ID', sources: 'Source accounts', targets: 'Target accounts', source: 'Source', target: 'Target',
    sourceHint: 'Obtain routes from accounts with verified Astra quality. Credentials and tickets are never copied.',
    targetHint: 'Requests use each target’s own credentials. Select up to 20; start with one.',
    emptyAccounts: 'No matching OAuth accounts', selected: 'Selected',
    followProxy: 'Follow source proxy (recommended)', proxyHint: 'Use the source route’s proxy for verification and business requests, including direct access when the source is direct. Keep the exit stable throughout verification and use. After persistent failure, try another stable exit and acquire a fresh source route. Automatic node rotation is not included. The same proxy URL does not guarantee the same public IP.',
    probeStatuses: 'Response status: first {mint} / continuation {continuation} (200 means accepted, not that verification passed)',
    probeTickets: 'Ticket length: first {mint} / continuation {continuation}; 0 means no ticket received. Ticket contents are never displayed.',
    ticketLengthWarning: 'The reference project only documents validation of this rule with 780-character tickets. This length differs; replacement alone cannot establish degraded quality or an IP problem.',
    ttl: 'Route lifetime (seconds)', ttlHint: '30–240 seconds, default 230. Usage never extends expiry; shorter upstream expiry takes precedence.',
    save: 'Save configuration', saving: 'Saving…', discard: 'Discard changes', unsaved: 'Unsaved changes; verification uses saved configuration',
    runtime: 'Runtime status · current instance', historyError: 'History could not be read or saved. Check the database.',
    remaining: '{seconds} seconds remaining', verify: 'Verify again', verifying: 'Verifying…', emptyRuntime: 'No accounts selected',
    history: 'Verification history · retained for 30 days', time: 'Time', result: 'Result', emptyHistory: 'No verification records', older: 'Load older records',
    invalidTTL: 'Lifetime must be an integer from 30 to 240.', tooMany: 'Select at most 20 sources and 20 targets.', overlap: 'Sources and targets must not overlap.', chooseBoth: 'Select sources and targets before enabling.',
    requestError: 'Request failed. Please retry.', accountsError: 'Could not load accounts. Reopen this page.', saved: 'Configuration saved. Preparation starts when enabled; check runtime status.', verified: 'Both verification requests passed.', upstreamError: 'Upstream returned {status}; verification failed',
    states: { ready: 'Ready', failed: 'Failed', expired: 'Expired', disabled: 'Disabled', not_tested: 'Not verified', verifying: 'Verifying' },
    reasons: {
      astra_proxy_unavailable: 'The account proxy is unavailable', astra_business_transport_failed: 'Business transport failed; request was not replayed',
      not_tested: 'Not verified yet', verifying: 'Verifying target', astra_probe_passed: 'Both requests completed without a replacement ticket', astra_source_ready: 'Source route acquired',
      astra_borrow_busy: 'Another verification is in progress', astra_account_unavailable: 'Account unavailable, paused or cooling down', astra_disable_excel_first: 'Disable Excel on the selected account first', astra_model_unavailable: 'Astra is unavailable or mapped to another model', astra_auth_failed: 'Could not obtain account credentials',
      astra_settings_unavailable: 'Configuration unavailable', astra_settings_invalid: 'Saved configuration invalid', astra_save_failed: 'Configuration save failed', astra_configuration_changed: 'Configuration, source identity or route changed; verify again', astra_account_changed: 'Account changed; verify again',
      astra_probe_cooldown: 'Target verification is cooling down', astra_source_cooldown: 'Source acquisition is cooling down', astra_no_source_route: 'No usable source route', astra_source_cookie_missing: 'Source did not return a valid route cookie',
      astra_probe_timeout: 'Probe timed out or was cancelled', astra_account_busy: 'Account concurrency is full', astra_rate_limited: 'Account request rate limited', astra_probe_request_failed: 'Could not construct probe', astra_network_error: 'Upstream connection failed',
      astra_stream_incomplete: 'Response incomplete or text missing', astra_stream_invalid: 'Invalid response format', astra_stream_failed: 'Upstream stream failed', astra_model_mismatch: 'Response model differs from Astra', astra_ticket_invalid: 'Invalid upstream ticket',
      astra_route_changed: 'Upstream changed the route during verification', astra_ticket_missing: 'First request returned no ticket', astra_route_expired: 'Route expired', astra_ticket_changed: 'Continuation returned a new ticket; verification failed', astra_business_route_invalidated: 'Business request failed or route changed; readiness revoked',
      astra_request_not_inspectable: 'Cannot determine request model', astra_http_responses_only: 'Borrowing supports HTTP Responses requests only'
    }
  }
}
