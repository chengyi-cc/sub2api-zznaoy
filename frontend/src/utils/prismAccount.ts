export const PRISM_MODELS = ['gpt-6.1-sol', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-luna'] as const

export function readPrismAccount(extra?: Record<string, unknown>) {
  const configured = Object.prototype.hasOwnProperty.call(extra || {}, 'openai_prism_browser_models')
  const value = extra?.openai_prism_browser_models
  return {
    enabled: extra?.openai_prism_browser === true,
    models: configured
      ? Array.isArray(value) ? PRISM_MODELS.filter(model => value.includes(model)) : []
      : [...PRISM_MODELS]
  }
}

export function applyPrismAccount(extra: Record<string, unknown>, enabled: boolean, models: string[], supported: boolean) {
  if (supported && enabled) {
    extra.openai_prism_browser = true
    extra.openai_prism_browser_models = PRISM_MODELS.filter(model => models.includes(model))
  } else {
    delete extra.openai_prism_browser
    delete extra.openai_prism_browser_models
  }
}
