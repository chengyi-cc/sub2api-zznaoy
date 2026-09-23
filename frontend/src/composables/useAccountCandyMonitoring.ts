import { onMounted, onUnmounted, reactive, ref, watch, type Ref } from 'vue'
import { candyMonitorAPI, type CandyAccountState } from '@/api/admin/candyMonitor'

// One batched read per page, independent of whether the account table auto-refreshes.
export function useAccountCandyMonitoring(accountIDs: Ref<number[]>) {
  const states = ref<Record<number, CandyAccountState>>({})
  const pending = reactive(new Set<number>())
  const schedulerEnabled = ref(true)
  const loadError = ref(false)
  let generation = 0
  let disposed = false
  let refreshing = false
  let timer: ReturnType<typeof setInterval> | undefined

  async function refresh() {
    if (disposed || pending.size) return
    const current = ++generation
    const ids = [...accountIDs.value]
    if (!ids.length) { states.value = {}; return }
    refreshing = true
    try {
      const batches = []
      for (let index = 0; index < ids.length; index += 200) batches.push(ids.slice(index, index + 200))
      const responses = await Promise.all(batches.map(batch => candyMonitorAPI.states(batch)))
      if (disposed || current !== generation) return
      states.value = Object.fromEntries(responses.flatMap(response => response.items).map(item => [item.account_id, item]))
      schedulerEnabled.value = responses[0].scheduler_enabled
      loadError.value = false
    } catch {
      // Keep the last known answer on a failed read; never misreport a red account as healthy.
      if (!disposed && current === generation) loadError.value = true
    } finally { if (current === generation) refreshing = false }
  }
  async function setEnabled(id: number, enabled: boolean) {
    if (disposed || pending.has(id) || !states.value[id]) return
    generation++
    refreshing = false
    pending.add(id)
    try {
      const response = await candyMonitorAPI.setMonitoring(id, enabled)
      if (disposed) return
      for (const item of response.items) states.value[item.account_id] = item
      schedulerEnabled.value = response.scheduler_enabled
      return response
    } finally {
      pending.delete(id)
      if (!disposed && !pending.size) void refresh()
    }
  }
  watch(() => accountIDs.value.join(','), () => {
    const visible = new Set(accountIDs.value)
    states.value = Object.fromEntries(Object.entries(states.value).filter(([id]) => visible.has(Number(id))))
    void refresh()
  }, { immediate: true })
  onMounted(() => { timer = setInterval(() => { if (!document.hidden && !refreshing) void refresh() }, 10000) })
  onUnmounted(() => { disposed = true; generation++; clearInterval(timer) })
  return { states, pending, schedulerEnabled, loadError, refresh, setEnabled }
}
