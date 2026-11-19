/**
 * 站点对账（管理员）。上游访问令牌、Bark 地址只写不读：接口只回显尾号 / 主机名。
 */
import { apiClient } from '../client'

const BASE = '/admin/site-recon'

export interface SiteReconTotals {
  rev: number
  cost: number
  missing: number
  all: number
  has: boolean
  scanned: number
}

export interface SiteReconAlert {
  level: 'bad' | 'warn' | 'info'
  text: string
}

export interface SiteReconTrendPoint {
  day: string
  rev: number
  cost: number
  missing: number
}

export interface SiteReconPnLAccount {
  id: number
  name: string
  upstream: string
  rev: number
  cost: number | null
  profit: number | null
  req: number
  rates: number[]
}

export interface SiteReconGroupPnL {
  name: string
  rev: number
  cost: number | null
  missing: number
  known: number
  profit: number | null
  margin: number | null
  req: number
  rates: number[]
  accounts: SiteReconPnLAccount[]
}

export interface SiteReconAnomalyCounts {
  day?: string
  high: number
  mid: number
  low: number
  high_list: number
}

export interface SiteReconOverview {
  day: string
  days: string[]
  totals: SiteReconTotals
  prev: SiteReconTotals
  alerts: SiteReconAlert[]
  trend: SiteReconTrendPoint[]
  groups: SiteReconGroupPnL[]
  anomaly_counts: SiteReconAnomalyCounts
}

export interface SiteReconAccountRow {
  day: string
  account_id: number
  account_name: string
  upstream: string
  our_requests: number
  up_requests: number | null
  our_list_cost: number
  up_list_cost: number | null
  revenue: number
  cost: number | null
  up_balance: number | null
  scanned_rows: number | null
  anomalies: number | null
  mismatches: number | null
  error: string | null
  profit: number | null
  leak: number | null
  idle: boolean
}

export interface SiteReconWallet {
  host: string
  balance: number
  burn: number
  days_left: number | null
  level: 'empty' | 'low' | 'ok' | 'idle'
  names: string[]
}

export interface SiteReconModelRow {
  account_id: number
  account_name: string
  model: string
  our_requests: number
  up_requests: number | null
  revenue: number
  cost: number | null
  our_list_cost: number
}

export interface SiteReconPending {
  account_id: number
  account_name: string
  upstream: string
  error: string
  revenue: number
  rev7: number
}

export interface SiteReconHistoryRow {
  day: string
  busy: number
  done: number
  rev: number | null
  cost: number | null
  our_requests: number | null
  up_requests: number | null
  rev_missing: number | null
  scanned: number
}

export interface SiteReconDetail {
  day: string
  days: string[]
  accounts: SiteReconAccountRow[]
  wallets: SiteReconWallet[]
  balance_days: number
  models: SiteReconModelRow[]
  pending: SiteReconPending[]
  history: SiteReconHistoryRow[]
}

export type SiteReconKind = 'zero_paid' | 'unpriced' | 'free' | 'mismatch'

export interface SiteReconAnomalyStat {
  account_id: number
  account_name: string
  kind: SiteReconKind
  model: string
  detail: string
  requests: number
  list_cost: number
  actual_cost: number
}

export interface SiteReconAnomalySample {
  account_id: number
  log_id: number
  kind: SiteReconKind
  created_at: string
  user_id: number
  model: string
  total_cost: number
  actual_cost: number
  tokens: number
}

export interface SiteReconAnomalies {
  day: string
  days: string[]
  scanned: boolean
  stats: SiteReconAnomalyStat[]
  samples: SiteReconAnomalySample[]
  history: SiteReconAnomalyCounts[]
}

export interface SiteReconConfig {
  timezone: string
  schedule_enabled: boolean
  schedule_time: string
  days: number
  scan: boolean
  exclude: number[]
  push_enabled: boolean
  balance_days: number
}

export interface SiteReconConfigView {
  config: SiteReconConfig
  bark_configured: boolean
  bark_hint: string
  next_run: string
}

export interface SiteReconConfigUpdate extends SiteReconConfig {
  /** 新的 Bark 推送地址；不传表示不修改 */
  bark_secret_url?: string
  clear_bark?: boolean
}

export interface SiteReconRun {
  id: number
  trigger: 'schedule' | 'manual' | 'retry' | string
  target: string
  days: number
  started_at: string
  finished_at: string
  status: 'running' | 'ok' | 'failed' | 'aborted' | string
  summary: {
    problems?: Record<string, string[]>
    error?: string
    elapsed?: number
  }
  log?: string
}

export interface SiteReconRunsView {
  running: boolean
  active_id: number
  active_log: string
  runs: SiteReconRun[]
  next_run: string
  default_day: string
  config: SiteReconConfig
}

export interface SiteReconStartInput {
  date: string
  days: number
  only: number[]
  rescan: boolean
  push: boolean
}

export interface SiteReconToken {
  host: string
  hint: string
  updated_at: string
  readable: boolean
}

export interface SiteReconTokensView {
  tokens: SiteReconToken[]
  need_tokens: string[]
}

const dayParams = (day?: string) => (day ? { params: { day } } : undefined)

export const siteReconAPI = {
  async overview(day?: string): Promise<SiteReconOverview> {
    const { data } = await apiClient.get<SiteReconOverview>(`${BASE}/overview`, dayParams(day))
    return data
  },
  async detail(day?: string): Promise<SiteReconDetail> {
    const { data } = await apiClient.get<SiteReconDetail>(`${BASE}/detail`, dayParams(day))
    return data
  },
  async anomalies(day?: string): Promise<SiteReconAnomalies> {
    const { data } = await apiClient.get<SiteReconAnomalies>(`${BASE}/anomalies`, dayParams(day))
    return data
  },
  async runs(): Promise<SiteReconRunsView> {
    const { data } = await apiClient.get<SiteReconRunsView>(`${BASE}/runs`)
    return data
  },
  async run(id: number): Promise<SiteReconRun> {
    const { data } = await apiClient.get<SiteReconRun>(`${BASE}/runs/${id}`)
    return data
  },
  async startRun(input: SiteReconStartInput): Promise<{ id: number }> {
    const { data } = await apiClient.post<{ id: number }>(`${BASE}/runs`, input)
    return data
  },
  async config(): Promise<SiteReconConfigView> {
    const { data } = await apiClient.get<SiteReconConfigView>(`${BASE}/config`)
    return data
  },
  async updateConfig(input: SiteReconConfigUpdate): Promise<SiteReconConfigView> {
    const { data } = await apiClient.put<SiteReconConfigView>(`${BASE}/config`, input)
    return data
  },
  async testPush(): Promise<void> {
    await apiClient.post(`${BASE}/config/test-push`)
  },
  async tokens(): Promise<SiteReconTokensView> {
    const { data } = await apiClient.get<SiteReconTokensView>(`${BASE}/tokens`)
    return data
  },
  async setToken(host: string, accessToken: string): Promise<void> {
    await apiClient.put(`${BASE}/tokens`, { host, access_token: accessToken })
  },
  async deleteToken(host: string): Promise<void> {
    await apiClient.post(`${BASE}/tokens/delete`, { host })
  }
}

export default siteReconAPI
