import { apiClient } from '../client'
import type { PaginatedResponse } from '@/types'

export const CANDY_DEFAULT_MODEL = 'gpt-6-astra'
export interface CandySettings {
  enabled: boolean
  model_id: string
  interval_minutes: number
  max_results: number
}
export interface CandyConfig {
  enabled: boolean
  use_defaults: boolean
  model_id: string
  interval_minutes: number
}
export interface CandyResult {
  id: number
  account_id: number
  model_id: string
  source: 'manual' | 'scheduled'
  verdict: 'running' | 'pass' | 'incorrect' | 'invalid_format' | 'inconclusive'
  reason: string
  actual?: number
  expected: number
  duration_ms: number
  response_text?: string
  error_message?: string
  started_at: string
  finished_at?: string
}
export interface CandyAccount extends CandyConfig {
  total_tests: number
  answer_21_count: number
  answer_29_count: number
  other_answer_count: number
  inconclusive_count: number
  account_id: number
  name: string
  platform: string
  status: string
  type: string
  last_run_at: string | null
  next_run_at: string | null
  running_until: string | null
  latest: CandyResult | null
  history: CandyResult[]
}
export interface CandyFilter {
  page: number
  page_size: number
  group_id?: number
  search?: string
  enabled_only?: boolean
  enabled?: boolean
  ungrouped?: boolean
  type?: string
  privacy_mode?: string
  verdict?: string
  platform?: string
  status?: string
}
export interface CandyAccountState extends CandyConfig {
  account_id: number
  history: CandyResult[]
  last_valid_answer: number | null
  last_valid_at: string | null
}
export interface CandyAccountStates {
  scheduler_enabled: boolean
  items: CandyAccountState[]
}
const base = '/admin/candy-monitor'
export const candyMonitorAPI = {
  async states(account_ids: number[]) { return (await apiClient.get<CandyAccountStates>(`${base}/accounts/states`, { params: { account_ids: account_ids.join(',') } })).data },
  async setMonitoring(id: number, enabled: boolean) { return (await apiClient.put<CandyAccountStates>(`${base}/accounts/${id}/monitoring`, { enabled })).data },
  async settings() { return (await apiClient.get<CandySettings>(`${base}/settings`)).data },
  async saveSettings(settings: CandySettings) { return (await apiClient.put<CandySettings>(`${base}/settings`, settings)).data },
  async list(params: CandyFilter) { return (await apiClient.get<PaginatedResponse<CandyAccount>>(`${base}/accounts`, { params })).data },
  async configure(account_ids: number[], config: CandyConfig) { await apiClient.put(`${base}/accounts`, { account_ids, ...config }) },
  async setEnabled(account_ids: number[], enabled: boolean) { await apiClient.put(`${base}/accounts/enabled`, { account_ids, enabled }) },
  async run(id: number, model_id = CANDY_DEFAULT_MODEL) { return (await apiClient.post<CandyResult>(`${base}/accounts/${id}/run`, { model_id })).data },
  async result(id: number) { return (await apiClient.get<CandyResult>(`${base}/results/${id}`)).data },
  async history(id: number) { return (await apiClient.get<CandyResult[]>(`${base}/accounts/${id}/results`)).data }
}
