<template>
  <BaseDialog :show="show" :title="`${t('admin.accounts.candyMonitor.quickTest')} · ${account?.name || ''}`" width="wide" @close="emit('close')">
    <div class="space-y-4">
      <p class="text-sm text-gray-600 dark:text-gray-300">{{ t('admin.accounts.candyMonitor.rule') }}</p>
      <label class="block text-sm">{{ t('admin.accounts.candyMonitor.model') }}
        <input v-model="model" class="input mt-1 w-full" :disabled="busy" maxlength="200" />
      </label>
      <p v-if="busy" role="status" class="text-sm text-primary-600">{{ t('admin.accounts.candyMonitor.runningHint') }}</p>
      <p v-if="error" role="alert" class="text-sm text-red-600">{{ error }}</p>
      <div v-if="result" class="space-y-3" aria-live="polite">
        <CandyVerdictBadge :verdict="result.verdict" :actual="result.actual" />
        <p class="text-xs text-gray-500">{{ result.model_id }} · {{ (result.duration_ms / 1000).toFixed(1) }} s</p>
        <p v-if="result.error_message" class="break-words text-sm text-red-600">{{ result.error_message }}</p>
        <details v-if="result.response_text">
          <summary class="cursor-pointer text-sm">{{ t('admin.accounts.candyMonitor.response') }}</summary>
          <pre class="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-gray-50 p-3 text-xs dark:bg-dark-900">{{ result.response_text }}</pre>
        </details>
      </div>
      <CandyTestPanel :result="null" />
      <RouterLink class="text-sm text-primary-600 hover:underline" to="/admin/candy-monitor" @click="emit('close')">{{ t('admin.accounts.candyMonitor.openMonitor') }} →</RouterLink>
    </div>
    <template #footer>
      <button class="btn btn-secondary" @click="emit('close')">{{ t('common.close') }}</button>
      <button v-if="result?.verdict === 'running' && error" class="btn btn-primary" @click="refresh">{{ t('admin.accounts.candyMonitor.refresh') }}</button>
      <button class="btn btn-primary" :disabled="busy || !model.trim()" @click="run">{{ t('admin.accounts.candyMonitor.runAgain') }}</button>
    </template>
  </BaseDialog>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import BaseDialog from '@/components/common/BaseDialog.vue'
import CandyTestPanel from './CandyTestPanel.vue'
import CandyVerdictBadge from './CandyVerdictBadge.vue'
import { candyMonitorAPI, CANDY_DEFAULT_MODEL, type CandyResult } from '@/api/admin/candyMonitor'

const props = withDefaults(defineProps<{ show: boolean; account: { id: number; name: string } | null; initialModel?: string }>(), { initialModel: CANDY_DEFAULT_MODEL })
const emit = defineEmits<{ close: []; completed: [] }>()
const { t } = useI18n()
const model = ref(CANDY_DEFAULT_MODEL)
const result = ref<CandyResult | null>(null)
const submitting = ref(false)
const error = ref('')
const busy = computed(() => submitting.value || result.value?.verdict === 'running')
let generation = 0
let timer: ReturnType<typeof setTimeout> | undefined
function stop() { generation++; clearTimeout(timer) }
function failure(e: unknown) {
  const message = (e as { response?: { data?: { message?: string } }; message?: string })
  return message.response?.data?.message || message.message || t('admin.accounts.candyMonitor.failed')
}
async function poll(id: number, current: number) {
  try {
    const next = await candyMonitorAPI.result(id)
    if (current !== generation || !props.show) return
    result.value = next
    error.value = ''
    if (next.verdict === 'running') timer = setTimeout(() => void poll(id, current), 2500)
    else emit('completed')
  } catch (e) {
    if (current === generation) error.value = failure(e)
  }
}
function refresh() { if (result.value) { clearTimeout(timer); void poll(result.value.id, generation) } }
async function run() {
  if (!props.account || busy.value) return
  stop()
  const current = generation
  submitting.value = true
  result.value = null
  error.value = ''
  try {
    const next = await candyMonitorAPI.run(props.account.id, model.value.trim())
    if (current !== generation || !props.show) return
    result.value = next
    if (next.verdict === 'running') timer = setTimeout(() => void poll(next.id, current), 1500)
    else emit('completed')
  } catch (e) {
    if (current === generation) error.value = failure(e)
  } finally { if (current === generation) submitting.value = false }
}
watch(() => [props.show, props.account?.id] as const, () => {
  stop()
  submitting.value = false
  result.value = null
  error.value = ''
  if (props.show && props.account) { model.value = props.initialModel; void run() }
}, { immediate: true })
onUnmounted(stop)
</script>
