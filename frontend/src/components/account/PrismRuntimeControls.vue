<template>
  <div class="mt-4 rounded-lg border border-violet-200 p-3 dark:border-violet-900" data-testid="prism-runtime">
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2 text-sm">
        <span class="font-medium">{{ t('admin.prism.runtime.title') }}</span>
        <span role="status" class="rounded-full px-2 py-0.5 text-xs" :class="status?.healthy ? 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300' : 'bg-gray-100 text-gray-600 dark:bg-dark-700 dark:text-gray-300'">
          {{ t(`admin.prism.runtime.states.${status?.state ?? 'loading'}`) }}
        </span>
      </div>
      <div class="flex flex-wrap gap-2">
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy || !status?.managed || status.state === 'running' || status.state === 'starting'" data-testid="prism-start" @click="run('start')">{{ t('admin.prism.runtime.start') }}</button>
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy || !status?.managed || status.state === 'stopped'" data-testid="prism-stop" @click="pending = 'stop'">{{ t('admin.prism.runtime.stop') }}</button>
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy || !status?.managed" data-testid="prism-restart" @click="pending = 'restart'">{{ t('admin.prism.runtime.restart') }}</button>
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy" data-testid="prism-check" @click="check">{{ t('admin.prism.runtime.check') }}</button>
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy || !status?.managed" :aria-expanded="showLogs" data-testid="prism-logs" @click="toggleLogs">{{ t('admin.prism.runtime.logs') }}</button>
      </div>
    </div>
    <p class="mt-2 text-xs text-gray-500">{{ t('admin.prism.runtime.scope') }}</p>
    <p v-if="status && !status.managed" class="mt-2 text-xs text-amber-700 dark:text-amber-400">{{ t(`admin.prism.runtime.hints.${status.state === 'unmanaged' ? 'unmanaged' : 'install'}`) }}</p>
    <button v-if="status?.state === 'not_installed'" type="button" class="btn btn-sm btn-secondary mt-2" data-testid="prism-copy-upgrade" @click="copyToClipboard(upgradeCommand)">{{ t('admin.prism.runtime.copyUpgrade') }}</button>
    <p v-if="status?.managed && !status.gateway_enabled" class="mt-2 text-xs text-amber-700 dark:text-amber-400">{{ t('admin.prism.runtime.hints.gatewayDisabled') }}</p>
    <p v-if="actionError || error" role="alert" class="mt-2 text-xs text-red-600">{{ actionError || error }}</p>
    <p v-if="checkResult" role="status" class="mt-2 text-xs text-gray-500">{{ checkResult }}</p>
    <div v-if="pending" role="alert" class="mt-3 rounded border border-amber-400 p-3 text-sm">
      <p>{{ t('admin.prism.runtime.confirm') }}</p>
      <div class="mt-2 flex gap-2">
        <button type="button" class="btn btn-sm btn-danger" :disabled="busy" data-testid="prism-confirm" @click="run(pending!)">{{ t(`admin.prism.runtime.${pending}`) }}</button>
        <button type="button" class="btn btn-sm btn-secondary" :disabled="busy" @click="pending = null">{{ t('admin.prism.runtime.cancel') }}</button>
      </div>
    </div>
    <div v-if="showLogs" class="mt-3" data-testid="prism-log-panel">
      <p class="mb-2 text-xs text-gray-500">{{ t('admin.prism.runtime.logsHint') }}</p>
      <ul class="max-h-48 space-y-1 overflow-y-auto rounded bg-gray-950 p-3 font-mono text-xs text-gray-200" aria-live="polite">
        <li v-if="!logs.length">{{ t('admin.prism.runtime.noLogs') }}</li>
        <li v-for="entry in logs" :key="entry.id"><time>{{ formatTime(entry.time) }}</time> {{ eventLabel(entry.code) }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useClipboard } from '@/composables/useClipboard'
import { controlPrismRuntime, getPrismRuntime, type PrismRuntimeAction, type PrismRuntimeLog, type PrismRuntimeStatus } from '@/api/admin/prismRuntime'

const { t, te } = useI18n()
const { copyToClipboard } = useClipboard()
const upgradeCommand = "(f=$(mktemp) && curl -fsSL https://raw.githubusercontent.com/chengyi-cc/sub2api-zznaoy/main/deploy/upgrade-prism.sh -o \"$f\" && bash \"$f\"; rc=$?; [ -z \"${f:-}\" ] || rm -f -- \"$f\"; exit \"$rc\")"
const status = ref<PrismRuntimeStatus | null>(null)
const logs = ref<PrismRuntimeLog[]>([])
const operating = ref(false)
const reading = ref(false)
const busy = computed(() => operating.value || reading.value)
const showLogs = ref(false)
const error = ref('')
const actionError = ref('')
const checkResult = ref('')
const pending = ref<'stop' | 'restart' | null>(null)
let disposed = false
let timer: ReturnType<typeof setTimeout> | undefined

function accept(value: PrismRuntimeStatus) {
  status.value = value
  if (value.logs) logs.value = value.logs
}
async function refresh() {
  if (busy.value || disposed) return
  reading.value = true
  try {
    const value = await getPrismRuntime(showLogs.value && !!status.value?.managed)
    if (!disposed) { accept(value); error.value = '' }
  } catch {
    if (!disposed) error.value = t('admin.prism.runtime.loadFailed')
  } finally { reading.value = false }
}
async function poll() {
  await refresh()
  if (!disposed) timer = setTimeout(poll, 5000)
}
async function run(action: PrismRuntimeAction) {
  if (busy.value) return
  operating.value = true
  error.value = ''
  actionError.value = ''
  checkResult.value = ''
  pending.value = null
  try {
    const value = await controlPrismRuntime(action)
    if (!disposed) {
      accept(value)
      if (action === 'check') checkResult.value = t(`admin.prism.runtime.${value.healthy ? 'checkOK' : 'checkFailed'}`)
    }
  } catch {
    if (!disposed) actionError.value = t('admin.prism.runtime.actionFailed')
  } finally { operating.value = false }
  await refresh()
}
async function check() {
  if (status.value?.managed) await run('check')
  else await refresh()
}
async function toggleLogs() {
  showLogs.value = !showLogs.value
  await refresh()
}
function eventLabel(code: string) {
  const key = `admin.prism.runtime.events.${code}`
  return te(key) ? t(key) : t('admin.prism.runtime.events.unknown_event')
}
function formatTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString()
}
onMounted(poll)
onBeforeUnmount(() => { disposed = true; clearTimeout(timer) })
</script>
