import { apiClient } from '../client'

export type PrismRuntimeAction = 'start' | 'stop' | 'restart' | 'check'
export interface PrismRuntimeLog { id: number; time: string; code: string }
export interface PrismRuntimeStatus {
  managed: boolean
  gateway_enabled: boolean
  state: 'not_installed' | 'unmanaged' | 'misconfigured' | 'unreachable' | 'stopped' | 'starting' | 'running' | 'error'
  healthy: boolean
  desired_enabled: boolean
  logs?: PrismRuntimeLog[]
}
export async function getPrismRuntime(logs = false) {
  return (await apiClient.get<PrismRuntimeStatus>(`/admin/prism-runtime${logs ? '/logs' : ''}`, { timeout: 5000 })).data
}
export async function controlPrismRuntime(action: PrismRuntimeAction) {
  return (await apiClient.post<PrismRuntimeStatus>(`/admin/prism-runtime/${action}`, undefined, { timeout: 20000 })).data
}
