<template>
  <div>
    <div v-if="loading && !data" class="flex justify-center py-12"><LoadingSpinner size="md" /></div>
    <div v-else-if="error" class="card p-6 text-sm text-red-600 dark:text-red-400">{{ error }}</div>
    <div v-else-if="data" data-test="site-recon-runs">
      <div class="card p-4 text-sm text-gray-700 dark:text-gray-200">
        <template v-if="data.config.schedule_enabled">
          {{ t('siteRecon.runs.planOn', { time: data.config.schedule_time, tz: data.config.timezone, next: data.next_run || '-' }) }}
        </template>
        <template v-else>{{ t('siteRecon.runs.planOff') }}</template>
      </div>

      <h3 :class="H2">{{ t('siteRecon.runs.manual') }}</h3>
      <form class="card space-y-3 p-4" @submit.prevent="start">
        <div class="flex flex-wrap items-end gap-3">
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.runs.date') }}</span>
            <input v-model="form.date" type="date" class="input mt-1 w-44" required />
          </label>
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.runs.days') }}</span>
            <input v-model.number="form.days" type="number" min="1" max="90" class="input mt-1 w-24" />
          </label>
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.runs.only') }}</span>
            <input v-model="form.only" type="text" class="input mt-1 w-56" :placeholder="t('siteRecon.runs.onlyPlaceholder')" />
          </label>
        </div>
        <label class="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-200">
          <input v-model="form.rescan" type="checkbox" class="mt-0.5" />
          <span>{{ t('siteRecon.runs.rescan') }} <span class="text-xs text-gray-400">{{ t('siteRecon.runs.rescanHint') }}</span></span>
        </label>
        <label class="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200">
          <input v-model="form.push" type="checkbox" />
          <span>{{ t('siteRecon.runs.push') }}</span>
        </label>
        <button type="submit" class="btn btn-primary" :disabled="data.running || starting" data-test="site-recon-start">
          {{ starting ? t('common.processing') : t('siteRecon.runs.start') }}
        </button>
      </form>

      <template v-if="data.running || liveLog">
        <h3 :class="H2">{{ data.running ? t('siteRecon.runs.running') : t('siteRecon.runs.lastLog') }}</h3>
        <pre ref="liveBox" class="card max-h-96 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs text-gray-700 dark:text-gray-200" data-test="site-recon-live-log">{{ liveLog || '…' }}</pre>
      </template>

      <template v-if="shown">
        <h3 :class="H2">{{ t('siteRecon.runs.logOf', { id: shown.id }) }}</h3>
        <pre class="card max-h-96 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-xs text-gray-700 dark:text-gray-200">{{ shown.log || t('siteRecon.runs.noLog') }}</pre>
      </template>

      <h3 :class="H2">{{ t('siteRecon.runs.records') }}</h3>
      <SrTable
        :head="[
          t('siteRecon.runs.startedAt'), t('siteRecon.runs.trigger'), t('siteRecon.runs.target'), t('siteRecon.runs.dayCount'),
          t('siteRecon.runs.status'), t('siteRecon.runs.elapsed'), t('siteRecon.runs.note'), ''
        ]"
        :left="3"
        :empty="!data.runs.length"
        :empty-text="t('siteRecon.runs.empty')"
      >
        <tr v-for="r in data.runs" :key="r.id">
          <td :class="TDL" class="whitespace-nowrap text-xs">{{ shortTime(r.started_at) }}</td>
          <td :class="TDL">{{ triggerText(r.trigger) }}</td>
          <td :class="TDL">{{ r.target || '-' }}</td>
          <td :class="TD">{{ r.days }}</td>
          <td :class="TD">
            <span class="rounded px-1.5 py-0.5 text-xs" :class="statusClass(r.status)">{{ statusText(r.status) }}</span>
          </td>
          <td :class="TD">{{ r.summary?.elapsed != null ? t('siteRecon.runs.seconds', { n: r.summary.elapsed }) : '-' }}</td>
          <td :class="TDL" class="max-w-xs whitespace-normal break-words text-xs">{{ runNote(r) }}</td>
          <td :class="TD">
            <button type="button" class="text-primary-600 hover:underline dark:text-primary-400" @click="showLog(r.id)">{{ t('siteRecon.runs.viewLog') }}</button>
          </td>
        </tr>
      </SrTable>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { siteReconAPI, type SiteReconRun, type SiteReconRunsView } from '@/api/admin/siteRecon'
import { useAppStore } from '@/stores/app'
import { extractApiErrorMessage } from '@/utils/apiError'
import SrTable from './SrTable.vue'
import { H2, TD, TDL, shortTime } from './format'

const emit = defineEmits<{ (e: 'finished'): void }>()
const { t } = useI18n()
const appStore = useAppStore()

const data = ref<SiteReconRunsView | null>(null)
const loading = ref(false)
const error = ref('')
const starting = ref(false)
const liveLog = ref('')
const shown = ref<SiteReconRun | null>(null)
const liveBox = ref<HTMLElement | null>(null)
const form = reactive({ date: '', days: 7, only: '', rescan: false, push: false })
let timer: ReturnType<typeof setTimeout> | null = null
let initialized = false

const KNOWN_TRIGGERS = ['schedule', 'manual', 'retry']
const KNOWN_STATUS = ['ok', 'failed', 'running', 'aborted']

function triggerText(v: string) {
  return KNOWN_TRIGGERS.includes(v) ? t(`siteRecon.runs.triggerName.${v}`) : v
}

function statusText(v: string) {
  return KNOWN_STATUS.includes(v) ? t(`siteRecon.runs.statusName.${v}`) : v
}

function statusClass(v: string) {
  switch (v) {
    case 'ok':
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
    case 'failed':
      return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
    case 'running':
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
    default:
      return 'bg-gray-100 text-gray-600 dark:bg-dark-700 dark:text-gray-300'
  }
}

function runNote(r: SiteReconRun) {
  if (r.summary?.error) return r.summary.error.slice(0, 80)
  const n = Object.values(r.summary?.problems ?? {}).reduce((s, v) => s + v.length, 0)
  return n ? t('siteRecon.runs.problems', { n }) : ''
}

function schedule(ms: number) {
  if (timer) clearTimeout(timer)
  timer = setTimeout(load, ms)
}

async function load() {
  if (!data.value) loading.value = true
  try {
    const wasRunning = data.value?.running ?? false
    const view = await siteReconAPI.runs()
    data.value = view
    error.value = ''
    if (!initialized) {
      initialized = true
      form.date = view.default_day
      form.days = view.config.days
    }
    if (view.running) {
      liveLog.value = view.active_log
      await nextTick()
      if (liveBox.value) liveBox.value.scrollTop = liveBox.value.scrollHeight
      schedule(2000)
    } else if (wasRunning) {
      // 刚跑完：拿最终日志，通知父组件刷新报表
      const latest = view.runs[0]
      if (latest) {
        const full = await siteReconAPI.run(latest.id)
        liveLog.value = full.log || ''
      }
      emit('finished')
    }
  } catch (e) {
    error.value = extractApiErrorMessage(e, t('siteRecon.loadFailed'))
  } finally {
    loading.value = false
  }
}

async function start() {
  starting.value = true
  try {
    const only = form.only
      .replace(/，/g, ',')
      .split(',')
      .map((x) => Number(x.trim()))
      .filter((x) => Number.isInteger(x) && x > 0)
    await siteReconAPI.startRun({ date: form.date, days: form.days, only, rescan: form.rescan, push: form.push })
    appStore.showSuccess(t('siteRecon.runs.started'))
    liveLog.value = ''
    shown.value = null
    await load()
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  } finally {
    starting.value = false
  }
}

async function showLog(id: number) {
  try {
    shown.value = await siteReconAPI.run(id)
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  }
}

onMounted(load)
onBeforeUnmount(() => {
  if (timer) clearTimeout(timer)
})
</script>
