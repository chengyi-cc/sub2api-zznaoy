<template>
  <div class="inline-flex shrink-0 items-center gap-2" @click.stop>
    <button type="button" class="rounded px-1 py-1 text-xs hover:underline" :class="state?.last_valid_answer === 29 ? 'text-red-600 dark:text-red-400' : 'text-primary-600 dark:text-primary-400'" :title="testTitle" data-testid="candy-test" @click="emit('test')">{{ tr('quickTest') }}</button>
    <label class="inline-flex items-center gap-1 whitespace-nowrap text-[11px] text-gray-500 dark:text-gray-400" :class="{ 'cursor-wait opacity-60': pending }" :title="monitorTitle">
      <input type="checkbox" class="h-3 w-3 rounded border-gray-300 text-primary-600 focus:ring-primary-500 dark:border-dark-500" :checked="state?.enabled || false" :disabled="!state || pending || loadError" :aria-label="tr('autoDetect')" data-testid="candy-auto" @change="changeMonitoring" />{{ tr('autoDetect') }}
    </label>
  </div>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { CandyAccountState } from '@/api/admin/candyMonitor'
const props = defineProps<{ state?: CandyAccountState; pending: boolean; schedulerEnabled: boolean; loadError: boolean }>()
const emit = defineEmits<{ test: []; toggle: [enabled: boolean] }>()
const { t } = useI18n()
const tr = (key: string, params: Record<string, string | number> = {}) => t(`admin.accounts.candyMonitor.${key}`, params)
function changeMonitoring(event: Event) {
  const input = event.target as HTMLInputElement
  const enabled = input.checked
  input.checked = props.state?.enabled || false
  emit('toggle', enabled)
}
const testTitle = computed(() => props.state?.last_valid_answer != null
  ? `${tr('lastValidAnswer', { answer: props.state.last_valid_answer })}${props.state.last_valid_at ? ` · ${new Date(props.state.last_valid_at).toLocaleString()}` : ''}`
  : tr('quickHint'))
const monitorTitle = computed(() => {
  if (props.loadError) return tr('stateLoadFailed')
  if (!props.state) return tr('loading')
  if (!props.schedulerEnabled) return tr('schedulerPausedHint')
  return tr(props.state.enabled ? 'autoDetectOnHint' : 'autoDetectOffHint', { model: props.state.model_id, minutes: props.state.interval_minutes })
})
</script>
