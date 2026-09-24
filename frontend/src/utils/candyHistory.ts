import type { CandyResult } from '@/api/admin/candyMonitor'

export function recentCandyResults(history: CandyResult[] = []): CandyResult[] {
  return history.filter(result => result.verdict !== 'running').sort((a, b) => b.id - a.id).slice(0, 10)
}

export function candyNormalRate(history: CandyResult[] = []): string {
  const valid = recentCandyResults(history).filter(result =>
    (result.verdict === 'pass' || result.verdict === 'incorrect') && result.actual != null)
  return valid.length ? `${(valid.filter(result => result.actual === 21).length / valid.length * 100).toFixed(1)}%` : '—'
}

export function candyBarClass(result: CandyResult | null): string {
  if (!result) return 'bg-gray-200 dark:bg-dark-600'
  if (result.verdict === 'pass') return 'bg-emerald-500'
  if (result.verdict === 'incorrect') return 'bg-red-500'
  return 'bg-amber-400'
}
