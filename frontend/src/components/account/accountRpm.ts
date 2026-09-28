interface AccountRPMSettings {
  enabled: boolean
  baseRpm: number | null
  overflow?: boolean
  strict?: boolean
  strategy?: 'tiered' | 'sticky_exempt'
  stickyBuffer?: number | null
}

// Single edits replace extra; bulk edits merge it. Merge payloads must send
// explicit empty values to clear settings that already exist in the database.
export function applyAccountRPMSettings(
  extra: Record<string, unknown>,
  settings: AccountRPMSettings,
  mode: 'replace' | 'merge' = 'replace'
): void {
  const clear = (key: string, empty: unknown) => {
    if (mode === 'merge') extra[key] = empty
    else delete extra[key]
  }
  if (!settings.enabled) {
    if (settings.strict) extra.base_rpm = 0
    else clear('base_rpm', 0)
    if (settings.strict) extra.openai_rpm_overflow = false
    clear('rpm_strategy', '')
    clear('rpm_sticky_buffer', 0)
    return
  }
  extra.base_rpm = settings.baseRpm != null && settings.baseRpm > 0 ? settings.baseRpm : 15
  if (settings.strict) {
    if (settings.overflow !== undefined) extra.openai_rpm_overflow = settings.overflow
    clear('rpm_strategy', '')
    clear('rpm_sticky_buffer', 0)
    return
  }
  extra.rpm_strategy = settings.strategy ?? 'tiered'
  if (settings.stickyBuffer != null && settings.stickyBuffer > 0) {
    extra.rpm_sticky_buffer = settings.stickyBuffer
  } else if (mode === 'replace') {
    delete extra.rpm_sticky_buffer
  }
}
