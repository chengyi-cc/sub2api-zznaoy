// 站点对账页面的格式化工具。金额单位沿用原对账站：¥。

export function money(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-'
  const abs = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  return v < 0 ? `-¥${abs}` : `¥${abs}`
}

export function num(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-'
  return v.toLocaleString('en-US')
}

export function pct(a: number | null | undefined, b: number | null | undefined): string {
  if (a === null || a === undefined || !b) return '-'
  return `${((a / b) * 100).toFixed(1)}%`
}

export function ratio(a: number | null | undefined, b: number | null | undefined): string {
  if (a === null || a === undefined || !b) return '-'
  return (a / b).toFixed(3)
}

export function signedClass(v: number | null | undefined): string {
  if (v === null || v === undefined || Math.abs(v) < 0.005) return ''
  return v > 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
}

export function ratesText(rates: number[] | null | undefined): string {
  if (!rates || rates.length === 0) return '-'
  const lo = Math.min(...rates)
  const hi = Math.max(...rates)
  const g = (x: number) => String(Number(x.toPrecision(6)))
  return lo === hi ? g(lo) : `${g(lo)}–${g(hi)}`
}

/** 请求差 = 上游 − 本站 */
export function diff(our: number | null | undefined, up: number | null | undefined): number | null {
  if (our === null || our === undefined || up === null || up === undefined) return null
  return up - our
}

export function signedNum(v: number | null): string {
  if (v === null) return '-'
  return `${v > 0 ? '+' : ''}${v.toLocaleString('en-US')}`
}

export function shortTime(v: string | null | undefined): string {
  if (!v) return '-'
  return v.slice(0, 19).replace('T', ' ')
}

export function delta(cur: number, old: number, has: boolean): number | null {
  if (!has || !old) return null
  return ((cur - old) / Math.abs(old)) * 100
}

// 表格单元格样式（写成字面量，Tailwind 才能扫描到）
export const TH = 'whitespace-nowrap px-3 py-2 text-right font-medium'
export const THL = 'whitespace-nowrap px-3 py-2 text-left font-medium'
export const TD = 'whitespace-nowrap px-3 py-2 text-right tabular-nums text-gray-700 dark:text-gray-200'
export const TDL = 'px-3 py-2 text-left text-gray-800 dark:text-gray-100'
export const NOTE = 'mt-2 text-xs leading-relaxed text-gray-500 dark:text-gray-400'
export const H2 = 'mb-2 mt-6 flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-white'
export const TAG = 'rounded bg-gray-100 px-1.5 py-0.5 text-[11px] font-normal text-gray-500 dark:bg-dark-700 dark:text-gray-400'
