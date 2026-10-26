import type { RedeemCode } from '@/types'

export const formatRedeemAmount = (value: number): string =>
  value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8, useGrouping: false })

export function formatGeneratedRedeemCodes(
  codes: RedeemCode[],
  line: (code: RedeemCode) => string
): string {
  if (!codes.some((code) => code.is_gift)) return codes.map((code) => code.code).join('\n')
  const groups = new Map<string, RedeemCode[]>()
  for (const code of codes) {
    const key = code.gift_pair_id || code.code
    groups.set(key, [...(groups.get(key) || []), code])
  }
  return [...groups.values()].map((group) =>
    [...group].sort((a, b) => Number(!!a.is_gift) - Number(!!b.is_gift)).map(line).join('\n')
  ).join('\n\n')
}

// getRandomValues is available even for an admin panel served on an internal
// HTTP address, where randomUUID may be unavailable.
export function createRedeemGenerationKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  return 'redeem-generate-' + Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
}
