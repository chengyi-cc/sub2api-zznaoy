import { describe, expect, it } from 'vitest'
import { missingCandyTestResult, parseCandyTestResult, supportsAccountCandyTest } from '../accountCandyTest'

describe('accountCandyTest', () => {
  it('offers only supported text account models', () => {
    for (const platform of ['openai', 'anthropic', 'gemini'] as const) {
      expect(supportsAccountCandyTest({ platform }, 'text-model')).toBe(true)
      for (const model of ['', 'gpt-image-1', 'grok-imagine', 'video', 'tts', 'whisper', 'embedding', 'audio', 'realtime', 'moderation']) {
        expect(supportsAccountCandyTest({ platform }, model)).toBe(false)
      }
    }
    expect(supportsAccountCandyTest({ platform: 'grok' }, 'text-model')).toBe(false)
    expect(supportsAccountCandyTest(null, 'text-model')).toBe(false)
  })

  it('does not turn missing or malformed backend grades into success', () => {
    expect(missingCandyTestResult()).toEqual({ case_id: 'candy-shape-v1', verdict: 'inconclusive', reason: 'missing_result' })
    const correct = {
      case_id: 'candy-shape-v1', verdict: 'pass', reason: 'correct',
      actual: 21, expected: 21, duration_ms: 1200
    }
    expect(parseCandyTestResult(correct)).toEqual(correct)
    for (const invalid of [null, 'pass', {}, { ...correct, case_id: 'other' }, { ...correct, verdict: 'unknown' }, { ...correct, expected: null }, { ...correct, actual: undefined }, { ...correct, actual: '21' }, { ...correct, duration_ms: -1 }]) {
      expect(parseCandyTestResult(invalid)).toBeNull()
    }
  })
})
