import { describe, expect, it } from 'vitest'
import {
  isValidUnifiedSocialCreditCode,
  normalizeUnifiedSocialCreditCode
} from '../unifiedSocialCreditCode'

describe('unifiedSocialCreditCode', () => {
  it('accepts valid codes, case and surrounding whitespace insensitive', () => {
    expect(isValidUnifiedSocialCreditCode('91350100M000100Y43')).toBe(true)
    expect(isValidUnifiedSocialCreditCode('91110000600001760P')).toBe(true)
    expect(isValidUnifiedSocialCreditCode(' 91110108551385082q ')).toBe(true)
    expect(normalizeUnifiedSocialCreditCode(' 91110108551385082q ')).toBe('91110108551385082Q')
  })

  it('rejects wrong check digit, length and charset', () => {
    expect(isValidUnifiedSocialCreditCode('')).toBe(false)
    expect(isValidUnifiedSocialCreditCode('91350100M000100Y44')).toBe(false)
    expect(isValidUnifiedSocialCreditCode('91350100M000100Y4')).toBe(false)
    expect(isValidUnifiedSocialCreditCode('91350100I000100Y43')).toBe(false)
    expect(isValidUnifiedSocialCreditCode('91110000000000000X')).toBe(false)
  })
})
