import { apiClient } from '../client'

export interface AstraBorrowSettings {
  enabled: boolean
  source_account_ids: number[]
  target_account_ids: number[]
  follow_source_proxy: boolean
  ttl_seconds: number
  revision: string
}
export interface AstraBorrowStatus {
  account_id: number
  source_account_id: number
  state: string
  reason: string
  checked_at: string
  expires_at?: string
}
export interface AstraBorrowSnapshot {
  settings: AstraBorrowSettings
  statuses: AstraBorrowStatus[]
  preparing: boolean
  history_error: boolean
}
export interface AstraBorrowHistory {
  id: number
  source_account_id: number
  target_account_id: number
  passed: boolean
  reason: string
  checked_at: string
}

export async function getAstraBorrow() {
  return (await apiClient.get<AstraBorrowSnapshot>('/admin/astra-borrow')).data
}
export async function saveAstraBorrow(value: AstraBorrowSettings) {
  return (await apiClient.put<AstraBorrowSettings>('/admin/astra-borrow', value)).data
}
export async function verifyAstraBorrow(id: number) {
  return (await apiClient.post(`/admin/astra-borrow/verify/${id}`, {}, { timeout: 160000 })).data
}
export async function getAstraBorrowHistory(before = 0) {
  return (await apiClient.get<AstraBorrowHistory[]>('/admin/astra-borrow/history', { params: { before } })).data
}
