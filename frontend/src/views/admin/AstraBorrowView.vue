<template>
  <AppLayout>
    <div class="space-y-5">
      <header class="flex flex-wrap items-center justify-between gap-3">
        <div><h1 class="text-xl font-semibold">{{ tr('title') }}</h1><p class="mt-2 text-sm text-gray-500">{{ tr('description') }}</p></div>
        <button class="btn btn-secondary" :disabled="loading" @click="refreshAll">{{ tr('refresh') }}</button>
      </header>
      <p v-if="error" role="alert" class="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30">{{ error }}</p>
      <p v-if="notice" role="status" class="text-sm text-emerald-600">{{ notice }}</p>
      <section class="card p-5 text-sm leading-6 text-gray-600 dark:text-gray-300">
        <p>{{ tr('guide') }}</p><p class="mt-2">{{ tr('boundary') }}</p>
      </section>
      <form v-if="draft" class="card space-y-5 p-5" @submit.prevent="save">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <label class="flex items-center gap-3 font-medium"><input v-model="draft.enabled" type="checkbox" data-testid="enabled" :disabled="saving" />{{ tr('enabled') }}</label>
          <span class="text-sm text-gray-500">{{ saved?.enabled ? tr('on') : tr('off') }} · {{ snapshot?.preparing ? tr('preparing') : tr('idle') }}</span>
        </div>
        <fieldset :disabled="saving || !accountsReady" class="space-y-5">
          <input v-model="search" class="input w-full" :placeholder="tr('search')" :aria-label="tr('search')" />
          <div class="grid gap-5 md:grid-cols-2">
            <section v-for="role in roles" :key="role.key">
              <h2 class="font-medium">{{ tr(role.key) }} <span class="text-sm text-gray-400">{{ draft[role.field].length }}/20</span></h2>
              <p class="mb-2 mt-1 text-xs text-gray-500">{{ tr(role.hint) }}</p>
              <div class="max-h-64 space-y-1 overflow-y-auto rounded-lg border border-gray-200 p-2 dark:border-dark-600">
                <label v-for="account in filteredAccounts" :key="account.id" class="flex items-center gap-2 rounded px-2 py-2 text-sm hover:bg-gray-50 dark:hover:bg-dark-700">
                  <input v-model="draft[role.field]" type="checkbox" :value="account.id" :disabled="draft[role.other].includes(account.id)" :data-testid="`${role.key}-${account.id}`" />
                  <span class="min-w-0 break-all">{{ account.name }} <span class="text-gray-400">#{{ account.id }}</span></span>
                </label>
                <p v-if="!filteredAccounts.length" class="p-2 text-sm text-gray-500">{{ tr('emptyAccounts') }}</p>
              </div>
              <p class="mt-2 break-words text-xs text-gray-500">{{ tr('selected') }}: {{ draft[role.field].map(accountName).join('、') || '—' }}</p>
            </section>
          </div>
          <div class="grid gap-5 md:grid-cols-2">
            <div><label class="flex items-center gap-2 text-sm"><input v-model="draft.follow_source_proxy" type="checkbox" />{{ tr('followProxy') }}</label><p class="mt-2 text-xs leading-5 text-gray-500">{{ tr('proxyHint') }}</p></div>
            <label class="text-sm">{{ tr('ttl') }}<input v-model.number="draft.ttl_seconds" class="input mt-2 block w-32" type="number" min="30" max="240" required /><span class="mt-1 block text-xs text-gray-500">{{ tr('ttlHint') }}</span></label>
          </div>
        </fieldset>
        <p v-if="validation" class="text-sm text-amber-600" role="alert">{{ validation }}</p>
        <div class="flex flex-wrap items-center gap-3">
          <button class="btn btn-primary" type="submit" :disabled="!dirty || saving || !!validation || (draft.enabled && !accountsReady)" data-testid="save">{{ saving ? tr('saving') : tr('save') }}</button>
          <button v-if="dirty" class="btn btn-secondary" type="button" :disabled="saving" @click="resetDraft">{{ tr('discard') }}</button>
          <span v-if="dirty" class="text-sm text-amber-600">{{ tr('unsaved') }}</span>
        </div>
      </form>
      <section class="card overflow-hidden">
        <h2 class="border-b border-gray-100 p-4 font-medium dark:border-dark-700">{{ tr('runtime') }}</h2>
        <p v-if="snapshot?.history_error" role="alert" class="p-4 text-sm text-amber-600">{{ tr('historyError') }}</p>
        <div v-for="row in snapshot?.statuses || []" :key="row.account_id" class="flex flex-wrap items-center justify-between gap-3 border-b border-gray-100 p-4 last:border-0 dark:border-dark-700">
          <div class="min-w-0"><p class="break-all text-sm font-medium">{{ accountName(row.account_id) }} <span class="text-gray-400">· {{ saved?.source_account_ids.includes(row.account_id) ? tr('source') : tr('target') }}</span></p>
            <p class="mt-1 text-sm" :class="row.state === 'ready' && remaining(row.expires_at) > 0 ? 'text-emerald-600' : 'text-gray-500'">{{ statusText(row) }} · {{ reasonText(row.reason) }}</p>
            <p v-if="row.source_account_id && row.source_account_id !== row.account_id" class="mt-1 text-xs text-gray-500">{{ tr('source') }}: {{ accountName(row.source_account_id) }}</p>
            <p v-if="row.expires_at" class="mt-1 text-xs text-gray-500">{{ tr('remaining', { seconds: remaining(row.expires_at) }) }}</p>
          </div>
          <button v-if="saved?.target_account_ids.includes(row.account_id)" type="button" class="btn btn-secondary btn-sm" :disabled="!saved.enabled || dirty || verifying !== null || snapshot?.preparing" @click="verify(row.account_id)">{{ verifying === row.account_id ? tr('verifying') : tr('verify') }}</button>
        </div>
        <p v-if="!snapshot?.statuses?.length" class="p-5 text-sm text-gray-500">{{ tr('emptyRuntime') }}</p>
      </section>
      <section class="card overflow-hidden">
        <h2 class="border-b border-gray-100 p-4 font-medium dark:border-dark-700">{{ tr('history') }}</h2>
        <div class="overflow-x-auto"><table class="w-full text-left text-sm"><thead><tr class="text-gray-500"><th class="p-3">{{ tr('time') }}</th><th class="p-3">{{ tr('source') }}</th><th class="p-3">{{ tr('target') }}</th><th class="p-3">{{ tr('result') }}</th></tr></thead><tbody>
          <tr v-for="row in history" :key="row.id" class="border-t border-gray-100 dark:border-dark-700"><td class="whitespace-nowrap p-3">{{ new Date(row.checked_at).toLocaleString() }}</td><td class="p-3">{{ accountName(row.source_account_id) }}</td><td class="p-3">{{ accountName(row.target_account_id) }}</td><td class="p-3" :class="row.passed ? 'text-emerald-600' : 'text-amber-600'">{{ reasonText(row.reason) }}</td></tr>
        </tbody></table></div>
        <p v-if="!history.length" class="p-4 text-sm text-gray-500">{{ tr('emptyHistory') }}</p>
        <button v-if="moreHistory" class="btn btn-secondary m-4" :disabled="historyLoading" @click="loadHistory(true)">{{ tr('older') }}</button>
      </section>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import AppLayout from '@/components/layout/AppLayout.vue'
import { list } from '@/api/admin/accounts'
import { getAstraBorrow, saveAstraBorrow, verifyAstraBorrow, getAstraBorrowHistory, type AstraBorrowSettings, type AstraBorrowSnapshot, type AstraBorrowStatus, type AstraBorrowHistory } from '@/api/admin/astraBorrow'

const { t, te } = useI18n()
const tr = (key: string, values: Record<string, unknown> = {}) => t(`admin.astraBorrow.${key}`, values)
const snapshot = ref<AstraBorrowSnapshot>()
const saved = ref<AstraBorrowSettings>()
const draft = ref<AstraBorrowSettings>()
const accounts = ref<{ id: number; name: string }[]>([])
const accountsReady = ref(false)
const history = ref<AstraBorrowHistory[]>([])
const loading = ref(false), saving = ref(false), historyLoading = ref(false), moreHistory = ref(false)
const verifying = ref<number | null>(null)
const error = ref(''), notice = ref(''), search = ref('')
const now = ref(Date.now())
let timer: ReturnType<typeof setInterval> | undefined
let disposed = false
let refreshVersion = 0
let historyPaginated = false
let pollCount = 0
const roles = [
  { key: 'sources', field: 'source_account_ids', other: 'target_account_ids', hint: 'sourceHint' },
  { key: 'targets', field: 'target_account_ids', other: 'source_account_ids', hint: 'targetHint' }
] as const
const filteredAccounts = computed(() => accounts.value.filter(a => `${a.name} ${a.id}`.toLowerCase().includes(search.value.toLowerCase())))
const dirty = computed(() => JSON.stringify(draft.value) !== JSON.stringify(saved.value))
const validation = computed(() => {
  const v = draft.value
  if (!v) return ''
  if (!Number.isInteger(v.ttl_seconds) || v.ttl_seconds < 30 || v.ttl_seconds > 240) return tr('invalidTTL')
  if (v.source_account_ids.length > 20 || v.target_account_ids.length > 20) return tr('tooMany')
  if (v.source_account_ids.some(id => v.target_account_ids.includes(id))) return tr('overlap')
  if (v.enabled && (!v.source_account_ids.length || !v.target_account_ids.length)) return tr('chooseBoth')
  return ''
})
const clone = (v: AstraBorrowSettings): AstraBorrowSettings => JSON.parse(JSON.stringify(v))
function resetDraft() { if (saved.value) draft.value = clone(saved.value) }
function accountName(id: number) { return id ? accounts.value.find(a => a.id === id)?.name || `#${id}` : '—' }
function remaining(expiry?: string) { return expiry ? Math.max(0, Math.ceil((Date.parse(expiry) - now.value) / 1000)) : 0 }
function reasonText(reason: string) {
  if (te(`admin.astraBorrow.reasons.${reason}`)) return tr(`reasons.${reason}`)
  const status = /^astra_upstream_(\d{3})$/.exec(reason)
  return status ? tr('upstreamError', { status: status[1] }) : reason
}
function statusText(row: AstraBorrowStatus) {
  const state = row.state === 'ready' && row.expires_at && remaining(row.expires_at) === 0 ? 'expired' : row.state
  return te(`admin.astraBorrow.states.${state}`) ? tr(`states.${state}`) : state
}
function errorText(err: unknown) {
  const msg = (err as { response?: { data?: { message?: string } }; message?: string }).response?.data?.message || (err as { message?: string }).message
  return msg ? reasonText(msg) : tr('requestError')
}
async function refresh() {
  if (loading.value || saving.value) return
  const version = ++refreshVersion
  loading.value = true
  try {
    const result = await getAstraBorrow()
    if (disposed || version !== refreshVersion) return
    const preserveDraft = draft.value && dirty.value
    snapshot.value = result
    saved.value = clone(result.settings)
    if (!preserveDraft) resetDraft()
  } catch (err) { error.value = errorText(err) }
  finally { loading.value = false }
}
async function loadHistory(append = false) {
  historyLoading.value = true
  try {
    const rows = await getAstraBorrowHistory(append ? history.value.at(-1)?.id || 0 : 0)
    if (!disposed) { history.value = append ? [...history.value, ...rows] : rows; moreHistory.value = rows.length === 50; historyPaginated = append }
  } catch { error.value = tr('historyError') }
  finally { historyLoading.value = false }
}
async function refreshAll() { error.value = ''; await Promise.allSettled([refresh(), loadHistory()]) }
async function save() {
  if (!draft.value || validation.value || saving.value) return
  saving.value = true; error.value = ''; notice.value = ''
  refreshVersion++
  try {
    saved.value = clone(await saveAstraBorrow(draft.value)); resetDraft(); notice.value = tr('saved')
  } catch (err) { error.value = errorText(err) }
  finally { saving.value = false; await refresh() }
}
async function verify(id: number) {
  verifying.value = id; error.value = ''; notice.value = ''
  try { await verifyAstraBorrow(id); notice.value = tr('verified') }
  catch (err) { error.value = errorText(err) }
  finally { verifying.value = null; await refresh(); await loadHistory() }
}
onMounted(async () => {
  await Promise.allSettled([refresh(), loadHistory(), (async () => {
    try {
      const rows: { id: number; name: string }[] = []
      for (let page = 1; !disposed; page++) {
        const result = await list(page, 100, { platform: 'openai', type: 'oauth', lite: 'true' })
        rows.push(...result.items.map(a => ({ id: a.id, name: a.name })))
        if (page >= result.pages || !result.items.length) break
      }
      if (!disposed) { accounts.value = rows; accountsReady.value = true }
    } catch { error.value = tr('accountsError') }
  })()])
  if (!disposed) timer = setInterval(() => {
    now.value = Date.now(); void refresh()
    if (++pollCount % 3 === 0 && !historyPaginated && !historyLoading.value) void loadHistory()
  }, 5000)
})
onUnmounted(() => { disposed = true; if (timer) clearInterval(timer) })
</script>
