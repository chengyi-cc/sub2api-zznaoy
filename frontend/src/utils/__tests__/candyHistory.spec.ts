import { describe, expect, it } from 'vitest'
import type { CandyResult } from '@/api/admin/candyMonitor'
import { candyNormalRate, recentCandyResults } from '../candyHistory'

const result = (id: number, verdict: CandyResult['verdict'], actual?: number): CandyResult => ({
  id, account_id: 1, model_id: 'gpt-6-astra', source: 'scheduled', verdict, actual,
  reason: '', expected: 21, duration_ms: 100, started_at: '2026-09-24T00:00:00Z'
})
describe('candy recent history', () => {
  it('excludes failed, invalid and running attempts from the rate, but includes other numeric answers', () => {
    const history = [result(6, 'running'), result(5, 'inconclusive'), result(4, 'invalid_format'), result(3, 'incorrect', 42), result(2, 'incorrect', 29), result(1, 'pass', 21)]
    expect(candyNormalRate(history)).toBe('33.3%')
    expect(candyNormalRate([result(1, 'inconclusive')])).toBe('—')
    expect(candyNormalRate()).toBe('—')
  })
  it('uses the latest ten completed tests without mutating the source or counting older passes', () => {
    const history = [result(1, 'pass', 21), ...Array.from({ length: 10 }, (_, i) => result(i + 2, 'incorrect', 29)), result(12, 'running')]
    expect(recentCandyResults(history).map(r => r.id)).toEqual([11, 10, 9, 8, 7, 6, 5, 4, 3, 2])
    expect(candyNormalRate(history)).toBe('0.0%')
    expect(history[0].id).toBe(1)
  })
})
