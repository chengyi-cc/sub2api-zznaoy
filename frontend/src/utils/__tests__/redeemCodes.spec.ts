import { describe, expect, it } from 'vitest'
import type { RedeemCode } from '@/types'
import { createRedeemGenerationKey, formatGeneratedRedeemCodes, formatRedeemAmount } from '../redeemCodes'

const code = (secret: string, value: number, gift: boolean, pair?: string): RedeemCode => ({
  id: 1, code: secret, value, is_gift: gift, gift_pair_id: pair,
  type: 'balance', status: 'unused', used_by: null, used_at: null, created_at: ''
})

describe('generated redeem code text', () => {
  it('groups each gift with its paid code even if the response is reordered', () => {
    const codes = [code('GIFT-A', 0.1, true, 'a'), code('PAID-B', 20, false, 'b'), code('PAID-A', 10, false, 'a'), code('GIFT-B', 0.2, true, 'b')]
    const text = formatGeneratedRedeemCodes(codes, (c) => `${c.is_gift ? 'Gift (no invoice)' : 'Paid'} ${formatRedeemAmount(c.value)}: ${c.code}`)
    expect(text).toBe('Paid 10.00: PAID-A\nGift (no invoice) 0.10: GIFT-A\n\nPaid 20.00: PAID-B\nGift (no invoice) 0.20: GIFT-B')
  })
  it('preserves the existing plain-code copy format when no gifts were generated', () => {
    expect(formatGeneratedRedeemCodes([code('A', 10, false), code('B', 10, false)], () => 'unused')).toBe('A\nB')
  })
  it.each([[0.0001, '0.0001'], [0.2999, '0.2999'], [10, '10.00'], [0.1, '0.10']])('displays %s without losing gift value', (amount, text) => {
    expect(formatRedeemAmount(amount)).toBe(text)
  })
  it('creates independent operation keys without randomUUID', () => {
    const a = createRedeemGenerationKey(), b = createRedeemGenerationKey()
    expect(a).toMatch(/^redeem-generate-[a-f0-9]{32}$/)
    expect(a).not.toBe(b)
  })
})
