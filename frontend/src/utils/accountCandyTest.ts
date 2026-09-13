import type { Account } from '@/types'

export interface CandyTestResult {
  case_id: string
  verdict: 'pass' | 'incorrect' | 'invalid_format' | 'inconclusive'
  reason: string
  expected?: number
  actual?: number
  duration_ms?: number
}

export function supportsAccountCandyTest(account: Pick<Account, 'platform'> | null, model: string): boolean {
  return Boolean(account && ['openai', 'anthropic', 'gemini'].includes(account.platform) && model && !/image|imagine|video|audio|realtime|tts|whisper|embedding|moderation/i.test(model))
}

export function parseCandyTestResult(value: unknown): CandyTestResult | null {
  if (!value || typeof value !== 'object') return null
  const result = value as CandyTestResult
  if (result.case_id !== 'candy-shape-v1' || !['pass', 'incorrect', 'invalid_format', 'inconclusive'].includes(result.verdict)) return null
  if (typeof result.reason !== 'string') return null
  if (result.expected !== undefined && !Number.isSafeInteger(result.expected)) return null
  if (result.actual !== undefined && !Number.isSafeInteger(result.actual)) return null
  if (result.duration_ms !== undefined && (!Number.isFinite(result.duration_ms) || result.duration_ms < 0)) return null
  if (['pass', 'incorrect'].includes(result.verdict) && (result.expected === undefined || result.actual === undefined)) return null
  return result
}

export function missingCandyTestResult(): CandyTestResult {
  return { case_id: 'candy-shape-v1', verdict: 'inconclusive', reason: 'missing_result' }
}
