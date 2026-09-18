import { computed, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { useDocumentVisibility, useIntervalFn } from '@vueuse/core'
import { apiClient } from '@/api/client'
import type { AccountListItem } from '@/types'

export type TurnStateModelSummary = {
  model: string
  state: string
  expires_at?: string
  length?: number
  last_error?: string
}

export type TurnStateSummary = {
  enabled: boolean
  configured: boolean
  models: TurnStateModelSummary[]
  server_time?: string
  clockOffset: number
  failed?: boolean
}

export function supportsTurnState(account: Pick<AccountListItem, 'platform' | 'type'>): boolean {
  return account.platform === 'openai' && (account.type === 'oauth' || account.type === 'setup-token')
}

export function useTurnStateSummaries(accounts: Ref<AccountListItem[]>, enabled: Ref<boolean>) {
  const summaries = ref<Record<number, TurnStateSummary>>({})
  const now = ref(Date.now())
  const visibility = useDocumentVisibility()
  const ids = computed(() => accounts.value.filter(account => supportsTurnState(account) && account.extra?.codex_turn_state_auto_enabled === true).map(account => account.id).join(','))
  let controller: AbortController | undefined
  let generation = 0
  let running = false

  async function refresh(): Promise<void> {
    if (running || !enabled.value || visibility.value === 'hidden' || !ids.value) return
    running = true
    const current = generation
    const requestController = new AbortController()
    controller = requestController
    const pending = ids.value.split(',').map(Number)
    async function worker(): Promise<void> {
      while (pending.length && !requestController.signal.aborted) {
        const accountId = pending.shift()!
        try {
          const { data } = await apiClient.get<Omit<TurnStateSummary, 'clockOffset'>>(`/admin/accounts/${accountId}/turn-state`, { signal: requestController.signal, timeout: 10000 })
          if (current !== generation) return
          const serverTime = Date.parse(data.server_time || '')
          summaries.value[accountId] = { ...data, models: data.models || [], clockOffset: Number.isFinite(serverTime) ? serverTime - Date.now() : 0 }
        } catch {
          if (current !== generation) return
          summaries.value[accountId] = { enabled: true, configured: false, models: [], clockOffset: 0, failed: true }
        }
      }
    }
    try {
      await Promise.all(Array.from({ length: Math.min(4, pending.length) }, worker))
    } finally {
      if (current === generation) running = false
    }
  }

  watch([ids, enabled, visibility], () => {
    generation += 1
    controller?.abort()
    running = false
    const retained: Record<number, TurnStateSummary> = {}
    for (const accountId of ids.value.split(',').map(Number)) {
      if (summaries.value[accountId]) retained[accountId] = summaries.value[accountId]
    }
    summaries.value = retained
    void refresh()
  }, { immediate: true })
  useIntervalFn(() => { now.value = Date.now() }, 1000)
  useIntervalFn(() => void refresh(), 30000)
  onBeforeUnmount(() => { generation += 1; controller?.abort() })
  return { summaries, now, refresh }
}
