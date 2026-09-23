<template>
  <AppLayout>
    <div class="space-y-5">
      <header>
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">🍬 {{ tr('title') }}</h1>
        <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">{{ tr('description') }}</p>
      </header>
      <p v-if="error" role="alert" class="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{{ error }}</p>
      <div v-if="initError" role="alert" class="flex items-center gap-3 text-sm text-red-600">{{ initError }}<button class="btn btn-secondary btn-sm" @click="loadSettingsAndGroups">{{ tr('refresh') }}</button></div>
      <p v-if="notice" role="status" class="text-sm text-emerald-600">{{ notice }}</p>
      <form class="card space-y-4 p-5" @submit.prevent="saveTemplate">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <h2 class="font-semibold">{{ tr('template') }}</h2>
          <label class="flex items-center gap-2 text-sm"><input v-model="settings.enabled" type="checkbox" :disabled="!ready || saving" />{{ tr('schedulerEnabled') }}</label>
        </div>
        <fieldset :disabled="!ready || saving" class="grid gap-4 md:grid-cols-3">
          <label class="text-sm">{{ tr('model') }}<input v-model="settings.model_id" data-testid="default-model" class="input mt-1 w-full" required maxlength="200" /></label>
          <label class="text-sm">{{ tr('interval') }}<input v-model.number="settings.interval_minutes" data-testid="default-interval" class="input mt-1 w-full" type="number" min="5" max="10080" required /></label>
          <label class="text-sm">{{ tr('retention') }}<input v-model.number="settings.max_results" class="input mt-1 w-full" type="number" min="10" max="500" required /></label>
        </fieldset>
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="max-w-3xl text-xs leading-5 text-gray-500">{{ tr('templateHint') }}</p>
          <button class="btn btn-primary" type="submit" :disabled="!ready || saving">{{ tr('saveTemplate') }}</button>
        </div>
      </form>
      <div class="rounded-xl border border-primary-200 bg-primary-50 p-4 text-sm text-primary-800 dark:border-primary-900 dark:bg-primary-950/30 dark:text-primary-200">
        <p>{{ tr('rule') }}</p><p class="mt-1 text-xs">{{ tr('scheduleHint') }}</p>
      </div>
      <section class="card overflow-hidden">
        <form class="flex flex-wrap items-end gap-3 border-b border-gray-200 p-4 dark:border-dark-700" @submit.prevent="applyFilters">
          <label class="min-w-40 text-sm">{{ tr('group') }}
            <select v-model.number="groupID" class="input mt-1 w-full" data-testid="group-filter" @change="applyFilters"><option :value="0">{{ tr('allGroups') }}</option><option v-for="group in groups" :key="group.id" :value="group.id">{{ group.name }}</option></select>
          </label>
          <label class="flex-1 text-sm">{{ tr('account') }}<input v-model="search" class="input mt-1 w-full" :placeholder="tr('search')" maxlength="200" /></label>
          <label class="flex items-center gap-2 py-2 text-sm"><input v-model="enabledOnly" type="checkbox" @change="applyFilters" />{{ tr('enabledOnly') }}</label>
          <button class="btn btn-secondary" type="submit">{{ tr('refresh') }}</button>
        </form>
        <div class="flex flex-wrap items-center gap-3 bg-gray-50 px-4 py-3 text-sm dark:bg-dark-900/40">
          <span>{{ tr('selected', { count: selected.length }) }}</span>
          <button class="btn btn-primary btn-sm" data-testid="apply-defaults" :disabled="!selected.length || saving || !ready" @click="applyDefaults">{{ tr('applyDefaults') }}</button>
          <button class="btn btn-secondary btn-sm" data-testid="pause-selected" :disabled="!selected.length || saving" @click="pauseSelected">{{ tr('pauseSelected') }}</button>
          <span v-if="loading" role="status" class="text-gray-500">{{ tr('loading') }}</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm">
            <thead class="border-b border-gray-200 text-xs text-gray-500 dark:border-dark-700"><tr>
              <th class="p-4"><input type="checkbox" :checked="allSelected" :aria-label="tr('selectPage')" @change="selectPage" /></th>
              <th class="p-3">{{ tr('account') }}</th><th class="p-3">{{ tr('configuration') }}</th><th class="p-3">{{ tr('latest') }}</th><th class="p-3">{{ tr('nextRun') }}</th><th class="p-3">{{ tr('actions') }}</th>
            </tr></thead>
            <tbody class="divide-y divide-gray-100 dark:divide-dark-700">
              <tr v-for="account in accounts" :key="account.account_id" :data-account-id="account.account_id">
                <td class="p-4"><input v-model="selected" type="checkbox" :value="account.account_id" :aria-label="account.name" /></td>
                <td class="p-3"><div class="max-w-56 truncate font-medium" :title="account.name">{{ account.name }}</div><div class="mt-1 text-xs text-gray-500">#{{ account.account_id }} · {{ account.platform }} · {{ account.status }}</div></td>
                <td class="p-3"><div :class="account.enabled ? 'text-emerald-600' : 'text-gray-500'">{{ account.enabled ? tr('enabled') : tr('paused') }} · {{ account.use_defaults ? tr('inherited') : tr('custom') }}</div><div class="mt-1 max-w-60 truncate text-xs text-gray-500" :title="account.model_id">{{ account.model_id }} · {{ tr('minutes', { count: account.interval_minutes }) }}</div></td>
                <td class="p-3"><CandyVerdictBadge :verdict="account.latest?.verdict" :actual="account.latest?.actual" /><div v-if="account.latest" class="mt-1 whitespace-nowrap text-xs text-gray-500">{{ date(account.latest.started_at) }}</div></td>
                <td class="whitespace-nowrap p-3 text-xs text-gray-500">{{ !schedulerEnabled ? tr('globalPaused') : account.enabled ? date(account.next_run_at) : '—' }}</td>
                <td class="p-3"><div class="flex items-center gap-2 whitespace-nowrap">
                  <button class="btn btn-secondary btn-sm" :disabled="account.latest?.verdict === 'running'" @click="runAccount = account">🍬 {{ tr('run') }}</button>
                  <button class="btn btn-ghost btn-sm" @click="edit(account)">{{ tr('configure') }}</button>
                  <button class="btn btn-ghost btn-sm" @click="showHistory(account)">{{ tr('history') }}</button>
                </div></td>
              </tr>
              <tr v-if="!accounts.length && !loading"><td colspan="6" class="p-10 text-center text-gray-500">{{ tr('empty') }}</td></tr>
            </tbody>
          </table>
        </div>
        <Pagination v-if="total > 0" :page="page" :page-size="pageSize" :total="total" :show-page-size-selector="false" @update:page="changePage" />
      </section>
    </div>
    <BaseDialog :show="!!editing" :title="`${tr('configure')} · ${editing?.name || ''}`" @close="editing = null">
      <form id="candy-account-settings" class="space-y-4" @submit.prevent="saveAccount">
        <label class="flex items-center gap-2 text-sm"><input v-model="config.enabled" type="checkbox" />{{ tr('accountEnabled') }}</label>
        <label class="flex items-center gap-2 text-sm"><input v-model="config.use_defaults" data-testid="use-defaults" type="checkbox" />{{ tr('useDefaults') }}</label>
        <fieldset :disabled="config.use_defaults" class="space-y-4">
          <label class="block text-sm">{{ tr('model') }}<input v-model="config.model_id" class="input mt-1 w-full" required maxlength="200" /></label>
          <label class="block text-sm">{{ tr('interval') }}<input v-model.number="config.interval_minutes" class="input mt-1 w-full" type="number" min="5" max="10080" required /></label>
        </fieldset>
        <p v-if="config.use_defaults" class="text-xs text-gray-500">{{ tr('inheritHint') }}</p>
        <p v-if="editError" role="alert" class="text-sm text-red-600">{{ editError }}</p>
      </form>
      <template #footer><button class="btn btn-secondary" @click="editing = null">{{ t('common.cancel') }}</button><button class="btn btn-primary" type="submit" form="candy-account-settings" :disabled="saving">{{ t('common.save') }}</button></template>
    </BaseDialog>
    <BaseDialog :show="!!historyAccount" :title="`${tr('history')} · ${historyAccount?.name || ''}`" width="wide" @close="historyAccount = null">
      <div class="space-y-4">
        <p class="text-xs text-gray-500">{{ tr('historyHint') }}</p>
        <button class="btn btn-secondary btn-sm" :disabled="historyLoading" @click="historyAccount && showHistory(historyAccount)">{{ tr('refresh') }}</button>
        <p v-if="historyError" role="alert" class="text-sm text-red-600">{{ historyError }}</p>
        <p v-if="historyLoading" role="status">{{ tr('loading') }}</p>
        <p v-else-if="!history.length" class="text-sm text-gray-500">{{ tr('noHistory') }}</p>
        <article v-for="result in history" :key="result.id" class="space-y-2 rounded-xl border border-gray-200 p-4 dark:border-dark-700">
          <div class="flex flex-wrap items-center justify-between gap-2"><CandyVerdictBadge :verdict="result.verdict" :actual="result.actual" /><span class="text-xs text-gray-500">{{ date(result.started_at) }}</span></div>
          <p class="text-xs text-gray-500">{{ result.model_id }} · {{ tr(result.source) }} · {{ (result.duration_ms / 1000).toFixed(1) }} s</p>
          <p v-if="result.error_message" class="break-words text-sm text-red-600">{{ result.error_message }}</p>
          <details v-if="result.response_text"><summary class="cursor-pointer text-sm">{{ tr('response') }}</summary><pre class="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs">{{ result.response_text }}</pre></details>
        </article>
      </div>
    </BaseDialog>
    <CandyMonitorRunDialog :show="!!runAccount" :account="runAccount ? { id: runAccount.account_id, name: runAccount.name } : null" :initial-model="runAccount?.model_id" @close="runAccount = null" @completed="loadAccounts()" />
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import AppLayout from '@/components/layout/AppLayout.vue'
import BaseDialog from '@/components/common/BaseDialog.vue'
import Pagination from '@/components/common/Pagination.vue'
import CandyVerdictBadge from '@/components/account/CandyVerdictBadge.vue'
import CandyMonitorRunDialog from '@/components/account/CandyMonitorRunDialog.vue'
import { candyMonitorAPI, CANDY_DEFAULT_MODEL, type CandyAccount, type CandyConfig, type CandyResult, type CandySettings } from '@/api/admin/candyMonitor'
import { getAllIncludingInactive } from '@/api/admin/groups'
import type { AdminGroup } from '@/types'

const { t } = useI18n()
const tr = (key: string, params: Record<string, string | number> = {}) => t(`admin.accounts.candyMonitor.${key}`, params)
const settings = reactive<CandySettings>({ enabled: true, model_id: CANDY_DEFAULT_MODEL, interval_minutes: 60, max_results: 50 })
const ready = ref(false)
const schedulerEnabled = ref(true)
const saving = ref(false)
const loading = ref(false)
const error = ref('')
const initError = ref('')
const notice = ref('')
const accounts = ref<CandyAccount[]>([])
const groups = ref<AdminGroup[]>([])
const groupID = ref(0)
const search = ref('')
const enabledOnly = ref(false)
const page = ref(1)
const pageSize = 30
const total = ref(0)
const selected = ref<number[]>([])
const editing = ref<CandyAccount | null>(null)
const config = reactive<CandyConfig>({ enabled: true, use_defaults: true, model_id: CANDY_DEFAULT_MODEL, interval_minutes: 60 })
const editError = ref('')
const runAccount = ref<CandyAccount | null>(null)
const historyAccount = ref<CandyAccount | null>(null)
const history = ref<CandyResult[]>([])
const historyLoading = ref(false)
const historyError = ref('')
let generation = 0
let historyGeneration = 0
let disposed = false
let timer: ReturnType<typeof setInterval> | undefined
const allSelected = computed(() => accounts.value.length > 0 && accounts.value.every(a => selected.value.includes(a.account_id)))
const date = (value: string | null) => value ? new Date(value).toLocaleString() : '—'
function failure(e: unknown) {
  const v = e as { response?: { data?: { message?: string } }; message?: string }
  return v.response?.data?.message || v.message || tr('failed')
}
async function loadAccounts(silent = false) {
  if (silent && loading.value) return
  const current = ++generation
  loading.value = true
  try {
    const response = await candyMonitorAPI.list({ page: page.value, page_size: pageSize, group_id: groupID.value || undefined, search: search.value.trim(), enabled_only: enabledOnly.value })
    if (disposed || current !== generation) return
    accounts.value = response.items
    total.value = response.total
    selected.value = selected.value.filter(id => accounts.value.some(a => a.account_id === id))
    error.value = ''
  } catch (e) { if (!disposed && current === generation) error.value = failure(e) }
  finally { if (current === generation) loading.value = false }
}
function applyFilters() { page.value = 1; selected.value = []; void loadAccounts() }
function changePage(value: number) { page.value = value; selected.value = []; void loadAccounts() }
function selectPage() { selected.value = allSelected.value ? [] : accounts.value.map(a => a.account_id) }
async function mutate(action: () => Promise<unknown>) {
  if (saving.value) return false
  saving.value = true; error.value = ''; notice.value = ''
  try { await action(); notice.value = tr('saved'); await loadAccounts(); return true }
  catch (e) { error.value = failure(e); return false }
  finally { saving.value = false }
}
async function saveTemplate() {
  await mutate(async () => { const saved = await candyMonitorAPI.saveSettings({ ...settings }); Object.assign(settings, saved); schedulerEnabled.value = saved.enabled })
}
async function applyDefaults() {
  await mutate(() => candyMonitorAPI.configure([...selected.value], { enabled: true, use_defaults: true, model_id: settings.model_id, interval_minutes: settings.interval_minutes }))
}
async function pauseSelected() { await mutate(() => candyMonitorAPI.setEnabled([...selected.value], false)) }
function edit(account: CandyAccount) { editing.value = account; editError.value = ''; Object.assign(config, { enabled: account.enabled, use_defaults: account.use_defaults, model_id: account.model_id, interval_minutes: account.interval_minutes }) }
async function saveAccount() {
  if (!editing.value) return
  const id = editing.value.account_id
  if (await mutate(() => candyMonitorAPI.configure([id], { ...config }))) editing.value = null
  else editError.value = error.value
}
async function showHistory(account: CandyAccount) {
  const current = ++historyGeneration
  historyAccount.value = account; history.value = []; historyLoading.value = true; historyError.value = ''
  try { const results = await candyMonitorAPI.history(account.account_id); if (!disposed && current === historyGeneration) history.value = results }
  catch (e) { if (!disposed && current === historyGeneration) historyError.value = failure(e) }
  finally { if (current === historyGeneration) historyLoading.value = false }
}
async function loadSettingsAndGroups() {
  initError.value = ''
  await Promise.allSettled([
    (async () => { try { const saved = await candyMonitorAPI.settings(); Object.assign(settings, saved); schedulerEnabled.value = saved.enabled; ready.value = true } catch (e) { initError.value = failure(e) } })(),
    (async () => { try { groups.value = await getAllIncludingInactive() } catch (e) { initError.value = failure(e) } })()
  ])
}
onMounted(async () => {
  await Promise.allSettled([loadSettingsAndGroups(), loadAccounts()])
  if (!disposed) timer = setInterval(() => void loadAccounts(true), 10000)
})
onUnmounted(() => { disposed = true; generation++; historyGeneration++; clearInterval(timer) })
</script>
