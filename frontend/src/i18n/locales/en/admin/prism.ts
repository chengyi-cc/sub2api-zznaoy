export default {
  prism: {
    title: 'Prism browser channel',
    description: 'Use a separate server browser service for Prism. Enable it per account and model. Off by default.',
    models: 'Models using Prism',
    modelsHint: 'Matches the account-mapped model name. Unselected models keep their original route; an empty selection disables Prism routing. Astra borrowing operates independently.',
    requirements: 'Check that the shared service is running. Reliable usage is unavailable, so requests are currently unbilled. Failures never switch channels automatically.',
    runtime: {
      title: 'Shared browser service', start: 'Start', stop: 'Stop', restart: 'Restart', check: 'Check connection', logs: 'Service logs', cancel: 'Cancel',
      scope: 'The account toggle applies only to this account when saved. Service operations apply immediately to every Prism account on this instance.',
      confirm: 'Stopping or restarting interrupts active requests for all Prism accounts on this instance. Session state is retained. Do not automatically replay requests with unknown outcomes. Continue?',
      states: { loading: 'Loading', not_installed: 'Bundled service not installed', unmanaged: 'Externally managed', misconfigured: 'Misconfigured', unreachable: 'Manager unreachable', stopped: 'Stopped', starting: 'Starting', running: 'Running', error: 'Error' },
      hints: {
        install: 'This runtime has no bundled browser. After publishing the updated image, run the upgrade command once from the original deployment directory. Future updates follow the usual process; browser installation and internal keys are automatic.',
        unmanaged: 'A separate browser service is configured. Upgrade to the bundled version to manage it here. Existing external services are never taken over or stopped.',
        gatewayDisabled: 'Browser management is available, but Prism is disabled in the gateway. Bundled deployments configure this automatically.'
      },
      copyUpgrade: 'Copy first-upgrade command',
      loadFailed: 'Unable to read service status. Check your connection and retry.', actionFailed: 'The operation was not confirmed. Check the connection and logs before retrying.',
      checkOK: 'The adapter process is reachable. Verify account access and model availability with an account test.', checkFailed: 'The adapter process is not ready. Check service logs.',
      logsHint: 'The latest 200 service events on this instance, refreshed automatically and cleared on manager restart. Credentials, conversation content and raw browser logs are excluded.', noLogs: 'No events yet',
      events: { browser_missing: 'Matching browser missing; rebuild bundled image', sandbox_missing: 'Browser sandbox missing; rebuild bundled image', non_root_required: 'Browser must run as a non-root user', adapter_configuration_invalid: 'Adapter configuration incomplete', dependencies_missing: 'Tool validation dependencies missing; rebuild bundled image', manager_ready: 'Manager ready', port_in_use: 'Adapter port occupied by another service', service_starting: 'Starting browser adapter', service_start_failed: 'Start failed', service_stopped: 'Service stopped', service_exited: 'Service exited unexpectedly; retry subject to cooldown', service_ready: 'Service ready', health_failed: 'Health check failed', check_ok: 'Connection check passed', check_failed: 'Connection check failed', control_failed: 'Service control failed', prism_adapter_error: 'Browser adapter request error', prism_worker_close_error: 'Browser shutdown error', unknown_event: 'Other service event' }
    },
    toolsHint: 'Use HTTP /v1/responses. Only 6.1 Sol supports client tools. Images, standalone context compaction and native WebSocket are not supported.'
  }
}
