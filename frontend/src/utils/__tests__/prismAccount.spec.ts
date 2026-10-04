import { describe, expect, it } from 'vitest'
import { applyPrismAccount, readPrismAccount, PRISM_MODELS } from '../prismAccount'

describe('Prism account selection', () => {
  it('preserves an explicit empty scope and limits legacy scope to known models', () => {
    expect(readPrismAccount({ openai_prism_browser: true }).models).toEqual([...PRISM_MODELS])
    for (const value of [[], null, '*', ['gpt-6-astra']]) {
      expect(readPrismAccount({ openai_prism_browser: true, openai_prism_browser_models: value }).models).toEqual([])
    }
  })
  it('changes only Prism fields and never enables borrowing or Excel', () => {
    const extra: Record<string, unknown> = { openai_excel_bps: false, existing: 42 }
    applyPrismAccount(extra, true, ['gpt-6.1-sol', 'gpt-6-astra', 'gpt-6.1-sol'], true)
    expect(extra).toEqual({ openai_excel_bps: false, existing: 42, openai_prism_browser: true, openai_prism_browser_models: ['gpt-6.1-sol'] })
    applyPrismAccount(extra, true, [], true)
    expect(extra.openai_prism_browser_models).toEqual([])
    applyPrismAccount(extra, true, [...PRISM_MODELS], false)
    expect(extra).toEqual({ openai_excel_bps: false, existing: 42 })
  })
})
