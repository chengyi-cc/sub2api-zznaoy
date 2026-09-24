<template>
  <AppLayout>
    <div class="candy-monitor space-y-4">
      <header class="flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <h1 class="text-xl font-semibold tracking-tight text-gray-900 dark:text-white">{{ tr('title') }}</h1>
          <span class="rounded-md bg-gray-100 px-2 py-1 text-xs tabular-nums text-gray-500 dark:bg-dark-800 dark:text-gray-400">{{ tr('accountCount', { count: total }) }}</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="mr-2 hidden text-xs text-gray-400 sm:inline">{{ tr('autoRefreshing') }}</span>
          <button class="btn btn-secondary btn-sm" :disabled="loading" :aria-label="tr('refresh')" @click="loadAccounts()"><Icon name="refresh" size="sm" :class="{ 'animate-spin': loading }" />{{ tr('refresh') }}</button>
          <button class="btn btn-secondary btn-sm" :aria-expanded="templateOpen" data-testid="toggle-template" @click="templateOpen = !templateOpen"><Icon name="cog" size="sm" />{{ tr('template') }}<Icon name="chevronDown" size="sm" :class="{ 'rotate-180': templateOpen }" /></button>
        </div>
      </header>
      <p v-if="error" role="alert" class="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{{ error }}</p>
      <div v-if="initError" role="alert" class="flex items-center gap-3 text-sm text-red-600">{{ initError }}<button class="btn btn-secondary btn-sm" @click="loadSettingsAndGroups">{{ tr('refresh') }}</button></div>
      <p v-if="notice" role="status" class="text-sm text-emerald-600 dark:text-emerald-400">{{ notice }}</p>
      <form v-show="templateOpen" class="rounded-xl border border-gray-200 bg-white p-4 dark:border-dark-700 dark:bg-dark-800" data-testid="template-form" @submit.prevent="saveTemplate">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 class="text-sm font-semibold">{{ tr('template') }}</h2>
          <label class="flex items-center gap-2 text-sm"><input v-model="settings.enabled" type="checkbox" :disabled="!ready || saving" />{{ tr('schedulerEnabled') }}</label>
        </div>
        <fieldset :disabled="!ready || saving" class="grid items-end gap-3 sm:grid-cols-2 xl:grid-cols-[2fr_1fr_1fr_auto]">
          <label class="text-xs text-gray-500 dark:text-gray-400">{{ tr('model') }}<input v-model="settings.model_id" data-testid="default-model" class="input mt-1.5 w-full" required maxlength="200" /></label>
          <label class="text-xs text-gray-500 dark:text-gray-400">{{ tr('interval') }}<input v-model.number="settings.interval_minutes" data-testid="default-interval" class="input mt-1.5 w-full" type="number" min="5" max="10080" required /></label>
          <label class="text-xs text-gray-500 dark:text-gray-400">{{ tr('retention') }}<input v-model.number="settings.max_results" class="input mt-1.5 w-full" type="number" min="10" max="500" required /></label>
          <button class="btn btn-primary" type="submit">{{ tr('saveTemplate') }}</button>
        </fieldset>
        <p class="mt-3 text-xs leading-5 text-gray-500">{{ tr('templateHint') }}</p>
      </form>
      <div v-if="ready && !schedulerEnabled" class="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400"><span class="h-3 w-1 rounded-[1px] bg-amber-500" />{{ tr('schedulerPausedHint') }}</div>
      <section class="overflow-hidden rounded-xl border border-gray-200 bg-white dark:border-dark-700 dark:bg-dark-800">
        <div class="space-y-3 border-b border-gray-200 p-4 dark:border-dark-700">
          <div class="flex flex-wrap items-center gap-2.5">
            <div class="relative min-w-52 flex-1 sm:max-w-80">
              <Icon name="search" size="sm" class="pointer-events-none absolute left-3 top-3 text-gray-400" />
              <input v-model="search" class="input w-full pl-9" data-testid="account-search" :aria-label="tr('account')" :placeholder="tr('search')" maxlength="200" @input="scheduleSearch" @keydown.enter.prevent="applyFilters" />
            </div>
            <Select v-model="platform" class="w-40" :options="platformOptions" :aria-label="tr('platform')" data-testid="platform-filter" @change="applyFilters" />
            <Select v-model="accountType" class="w-40" :options="typeOptions" :aria-label="t('admin.accounts.allTypes')" data-testid="type-filter" @change="applyFilters" />
            <Select v-model="accountStatus" class="w-40" :options="statusOptions" :aria-label="tr('accountStatus')" data-testid="status-filter" @change="applyFilters" />
            <Select v-model="groupID" class="w-48" :options="groupOptions" searchable :aria-label="tr('group')" data-testid="group-filter" @change="applyFilters" />
          </div>
          <div class="flex flex-wrap items-center gap-2.5">
            <Select v-model="monitorStatus" class="w-40" :options="monitorOptions" :aria-label="tr('monitorStatus')" data-testid="monitor-status-filter" @change="applyFilters" />
            <Select v-model="verdict" class="w-40" :options="verdictOptions" :aria-label="tr('latest')" data-testid="verdict-filter" @change="applyFilters" />
            <Select v-model="privacyMode" class="w-40" :options="privacyOptions" :aria-label="t('admin.accounts.allPrivacyModes')" data-testid="privacy-filter" @change="applyFilters" />
            <button v-if="hasFilters" class="btn btn-ghost btn-sm" data-testid="reset-filters" @click="resetFilters"><Icon name="x" size="sm" />{{ tr('reset') }}</button>
            <div class="ml-auto flex flex-wrap items-center gap-3 text-xs text-gray-500 dark:text-gray-400" :title="tr('rule')">
              <span class="inline-flex items-center gap-1.5"><span class="h-3 w-1 rounded-[1px] bg-emerald-500" />21 {{ tr('verdict.pass') }}</span>
              <span class="inline-flex items-center gap-1.5"><span class="h-3 w-1 rounded-[1px] bg-red-500" />29 {{ tr('verdict.incorrect') }}</span>
              <span class="inline-flex items-center gap-1.5"><span class="h-3 w-1 rounded-[1px] bg-amber-400" />{{ tr('verdict.inconclusive') }}</span>
            </div>
          </div>
        </div>
        <div class="flex min-h-12 flex-wrap items-center gap-3 border-b border-gray-200 px-4 py-2 text-xs dark:border-dark-700">
          <span class="tabular-nums text-gray-500">{{ tr('selected', { count: selected.length }) }}</span>
          <button class="btn btn-secondary btn-sm" data-testid="apply-defaults" :disabled="!selected.length || saving || !ready" @click="applyDefaults">{{ tr('applyDefaults') }}</button>
          <button class="btn btn-ghost btn-sm" data-testid="pause-selected" :disabled="!selected.length || saving" @click="pauseSelected">{{ tr('pauseSelected') }}</button>
          <span v-if="loading" role="status" class="ml-auto text-gray-400">{{ tr('loading') }}</span>
        </div>
        <div class="max-h-[calc(100vh-22rem)] min-h-56 overflow-auto" :aria-busy="loading">
          <table class="monitor-table w-full text-left text-sm">
            <thead class="sticky top-0 z-10 bg-gray-50 text-xs text-gray-500 dark:bg-dark-900 dark:text-gray-400"><tr>
              <th class="w-12"><input type="checkbox" :checked="allSelected" :indeterminate="selected.length > 0 && !allSelected" :aria-label="tr('selectPage')" @change="selectPage" /></th>
              <th>{{ tr('account') }}</th><th>{{ tr('latest') }}</th>
              <th class="!text-right">{{ tr('count21') }}</th><th class="!text-right">{{ tr('count29') }}</th><th class="!text-right">{{ tr('counts') }}</th>
              <th>{{ tr('historyStatus') }}<span class="ml-2 font-normal text-gray-400">{{ tr('lastTen') }}</span></th>
              <th>{{ tr('configuration') }}</th><th>{{ tr('nextRun') }}</th><th class="!text-right">{{ tr('actions') }}</th>
            </tr></thead>
            <tbody class="divide-y divide-gray-100 dark:divide-dark-700/70">
              <tr v-for="account in accounts" :key="account.account_id" :data-account-id="account.account_id" class="transition-colors hover:bg-gray-50 dark:hover:bg-dark-700/30" :class="{ 'bg-primary-50/60 dark:bg-primary-950/20': selected.includes(account.account_id) }">
                <td><input v-model="selected" type="checkbox" :value="account.account_id" :aria-label="account.name" /></td>
                <td class="min-w-52"><div class="max-w-72 truncate font-medium text-gray-900 dark:text-gray-100" :title="account.name">{{ account.name }}</div><div class="mt-1 flex items-center gap-1.5 text-xs text-gray-400"><span class="tabular-nums">#{{ account.account_id }}</span><span>&middot;</span><span>{{ platformLabel(account.platform) }}</span><span v-if="account.type" class="rounded bg-gray-100 px-1 text-[10px] dark:bg-dark-700">{{ typeLabel(account.type) }}</span></div></td>
                <td class="min-w-32"><div class="inline-flex items-center gap-2 whitespace-nowrap text-xs font-medium" data-testid="latest-status"><span class="h-3 w-1 shrink-0 rounded-[1px]" :class="statusBarClass(account.latest)" :data-verdict="account.latest?.verdict || 'untested'" />{{ resultLabel(account.latest) }}</div><div v-if="account.latest" class="mt-1 text-[11px] tabular-nums text-gray-400" :title="date(account.latest.started_at)">{{ shortDate(account.latest.started_at) }}</div></td>
                <td class="text-right text-base font-semibold tabular-nums text-emerald-600 dark:text-emerald-400" data-testid="count-21">{{ account.answer_21_count || 0 }}</td>
                <td class="text-right text-base font-semibold tabular-nums" :class="account.answer_29_count ? 'text-red-600 dark:text-red-400' : 'text-gray-400'" data-testid="count-29">{{ account.answer_29_count || 0 }}</td>
                <td class="text-right tabular-nums text-gray-500" :title="`${tr('otherAnswers', { count: account.other_answer_count || 0 })} / ${tr('inconclusiveCount', { count: account.inconclusive_count || 0 })}`" data-testid="total-tests">{{ account.total_tests || 0 }}</td>
                <td><CandyHistoryBars :history="account.history" @select="showHistory(account)" /></td>
                <td class="min-w-40"><div class="flex items-center gap-2 whitespace-nowrap text-xs"><span :class="account.enabled ? 'text-gray-700 dark:text-gray-200' : 'text-gray-400'">{{ !account.enabled ? tr('paused') : account.blocked_reason ? tr(`blocked.${account.blocked_reason}`) : tr('enabled') }}</span><span class="text-gray-300 dark:text-dark-500">/</span><span class="text-gray-500">{{ tr('minutes', { count: account.interval_minutes }) }}</span></div><div class="mt-1 max-w-48 truncate text-[11px] text-gray-400" :title="`${account.model_id} / ${account.use_defaults ? tr('inherited') : tr('custom')}`">{{ account.model_id }}<span v-if="!account.use_defaults"> &middot; {{ tr('custom') }}</span></div></td>
                <td class="whitespace-nowrap text-xs tabular-nums text-gray-500" :title="account.enabled && account.blocked_reason ? tr('blockedHint') : date(account.next_run_at)" data-testid="next-run">{{ !account.enabled ? '—' : !schedulerEnabled ? tr('globalPaused') : account.blocked_reason ? '—' : shortDate(account.next_run_at) }}</td>
                <td><div class="flex items-center justify-end gap-1 whitespace-nowrap">
                  <button class="btn btn-secondary btn-sm" :disabled="account.latest?.verdict === 'running'" @click="runAccount = account"><Icon name="play" size="xs" />{{ tr('run') }}</button>
                  <button class="btn btn-ghost btn-sm" @click="edit(account)">{{ tr('configure') }}</button>
                  <button class="btn btn-ghost btn-sm" @click="showHistory(account)">{{ tr('history') }}</button>
                </div></td>
              </tr>
              <tr v-if="!accounts.length"><td colspan="10" class="!py-16 text-center text-gray-400">{{ loading ? tr('loading') : tr('empty') }}</td></tr>
            </tbody>
          </table>
        </div>
        <Pagination v-if="total > 0" :page="page" :page-size="pageSize" :total="total" :page-size-options="[20, 50, 100]" :show-jump="true" @update:page="changePage" @update:page-size="changePageSize" />
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
import Select from '@/components/common/Select.vue'
import Icon from '@/components/icons/Icon.vue'
import CandyHistoryBars from '@/components/account/CandyHistoryBars.vue'
import CandyVerdictBadge from '@/components/account/CandyVerdictBadge.vue'
import CandyMonitorRunDialog from '@/components/account/CandyMonitorRunDialog.vue'
import { candyMonitorAPI, CANDY_DEFAULT_MODEL, type CandyAccount, type CandyConfig, type CandyResult, type CandySettings } from '@/api/admin/candyMonitor'
import { getAllIncludingInactive } from '@/api/admin/groups'
import type { AdminGroup } from '@/types'

const { t } = useI18n()
const tr = (key: string, params: Record<string, string | number> = {}) => t(`admin.accounts.candyMonitor.${key}`, params)
const settings = reactive<CandySettings>({ enabled: true, model_id: CANDY_DEFAULT_MODEL, interval_minutes: 60, max_results: 50 })
const templateOpen = ref(false)
const ready = ref(false)
const schedulerEnabled = ref(true)
const saving = ref(false)
const loading = ref(false)
const error = ref('')
const initError = ref('')
const notice = ref('')
const accounts = ref<CandyAccount[]>([])
const groups = ref<AdminGroup[]>([])
const groupID = ref('')
const platform = ref('')
const accountType = ref('')
const accountStatus = ref('')
const monitorStatus = ref('')
const privacyMode = ref('')
const verdict = ref('')
const search = ref('')
const page = ref(1)
const pageSize = ref(50)
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
let searchTimer: ReturnType<typeof setTimeout> | undefined
const platformOptions = computed(() => [{ value: '', label: tr('allPlatforms') }, ...['openai', 'anthropic', 'gemini'].map(value => ({ value, label: platformLabel(value) }))])
const typeOptions = computed(() => [{ value: '', label: t('admin.accounts.allTypes') }, ...['oauth', 'apikey', 'setup-token', 'bedrock'].map(value => ({ value, label: typeLabel(value) }))])
const statusOptions = computed(() => [{ value: '', label: tr('accountStatus') }, ...[
  ['active', 'active'], ['inactive', 'inactive'], ['error', 'error'], ['rate_limited', 'rateLimited'], ['temp_unschedulable', 'tempUnschedulable'], ['unschedulable', 'unschedulable']
].map(([value, key]) => ({ value, label: t(`admin.accounts.status.${key}`) }))])
const groupOptions = computed(() => [{ value: '', label: tr('allGroups') }, { value: 'ungrouped', label: t('admin.accounts.ungroupedGroup') }, ...groups.value.map(group => ({ value: String(group.id), label: group.name }))])
const monitorOptions = computed(() => [{ value: '', label: tr('monitorStatus') }, { value: 'enabled', label: tr('enabled') }, { value: 'paused', label: tr('paused') }])
const verdictOptions = computed(() => [{ value: '', label: tr('allResults') }, ...['pass', 'incorrect', 'inconclusive', 'invalid_format', 'running', 'untested'].map(value => ({ value, label: tr(`verdict.${value}`) }))])
const privacyOptions = computed(() => [
  { value: '', label: t('admin.accounts.allPrivacyModes') }, { value: '__unset__', label: t('admin.accounts.privacyUnset') },
  { value: 'training_off', label: 'Privacy' }, { value: 'training_set_cf_blocked', label: 'CF' }, { value: 'training_set_failed', label: 'Fail' }
])
const hasFilters = computed(() => !!(groupID.value || platform.value || accountType.value || accountStatus.value || monitorStatus.value || privacyMode.value || verdict.value || search.value))
const allSelected = computed(() => accounts.value.length > 0 && accounts.value.every(a => selected.value.includes(a.account_id)))
const date = (value: string | null) => value ? new Date(value).toLocaleString() : '—'
const shortDate = (value: string | null) => value ? new Date(value).toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'
const platformLabel = (value: string) => ({ openai: 'OpenAI', anthropic: 'Anthropic', gemini: 'Gemini' })[value] || value
const typeLabel = (value: string) => ({ oauth: 'OAuth', apikey: 'API Key', 'setup-token': 'Setup Token', bedrock: 'AWS Bedrock' })[value] || value
const resultLabel = (result: CandyResult | null | undefined) => `${tr(`verdict.${result?.verdict || 'untested'}`)}${result?.actual != null ? ` · ${result.actual}` : ''}`
function statusBarClass(result: CandyResult | null | undefined) {
  if (!result) return 'bg-gray-200 dark:bg-dark-600'
  if (result.verdict === 'pass') return 'bg-emerald-500'
  if (result.verdict === 'incorrect') return 'bg-red-500'
  if (result.verdict === 'running') return 'bg-amber-400 animate-pulse'
  return 'bg-amber-400'
}
function failure(e: unknown) {
  const v = e as { response?: { data?: { message?: string } }; message?: string }
  return v.response?.data?.message || v.message || tr('failed')
}
async function loadAccounts(silent = false) {
  if (silent && loading.value) return
  const current = ++generation
  loading.value = true
  try {
    const response = await candyMonitorAPI.list({
      page: page.value, page_size: pageSize.value,
      group_id: groupID.value && groupID.value !== 'ungrouped' ? Number(groupID.value) : undefined,
      ungrouped: groupID.value === 'ungrouped' || undefined, search: search.value.trim(),
      enabled: monitorStatus.value ? monitorStatus.value === 'enabled' : undefined,
      platform: platform.value || undefined, status: accountStatus.value || undefined,
      type: accountType.value || undefined, privacy_mode: privacyMode.value || undefined, verdict: verdict.value || undefined
    })
    if (disposed || current !== generation) return
    if (page.value > 1 && !response.items.length && response.total <= (page.value - 1) * pageSize.value) {
      page.value = Math.max(1, Math.ceil(response.total / pageSize.value))
      return await loadAccounts()
    }
    accounts.value = response.items
    total.value = response.total
    selected.value = selected.value.filter(id => accounts.value.some(a => a.account_id === id))
    error.value = ''
  } catch (e) { if (!disposed && current === generation) error.value = failure(e) }
  finally { if (current === generation) loading.value = false }
}
function applyFilters() { clearTimeout(searchTimer); page.value = 1; selected.value = []; void loadAccounts() }
function scheduleSearch() { clearTimeout(searchTimer); searchTimer = setTimeout(applyFilters, 300) }
function resetFilters() { groupID.value = ''; platform.value = ''; accountType.value = ''; accountStatus.value = ''; monitorStatus.value = ''; privacyMode.value = ''; verdict.value = ''; search.value = ''; applyFilters() }
function changePage(value: number) { page.value = value; selected.value = []; void loadAccounts() }
function changePageSize(value: number) { pageSize.value = value; applyFilters() }
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
onUnmounted(() => { disposed = true; generation++; historyGeneration++; clearInterval(timer); clearTimeout(searchTimer) })
</script>

<style scoped>
.monitor-table th {
  @apply whitespace-nowrap border-b border-gray-200 px-4 py-3 font-medium dark:border-dark-700;
}
.monitor-table td {
  @apply px-4 py-3.5;
}
.monitor-table th:first-child, .monitor-table td:first-child {
  @apply pr-0;
}
.candy-monitor :deep(.btn) {
  @apply gap-1.5;
}
</style>
