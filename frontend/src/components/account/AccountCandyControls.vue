<template>
  <div class="shrink-0" :class="state?.enabled ? 'w-40 space-y-0.5' : ''" @click.stop>
    <div class="flex items-center justify-between gap-2">
      <button v-if="state?.enabled" type="button" class="flex flex-1 items-center justify-between gap-2 text-xs font-medium hover:underline"
        :class="state.last_valid_answer === 29 ? 'text-red-600 dark:text-red-400' : state.last_valid_answer === 21 ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-500 dark:text-gray-400'"
        :title="testTitle" :aria-label="tr('quickTest')" data-testid="candy-test" @click="emit('test')">
        <span>{{ tr('intelligenceStatus') }}</span><span class="tabular-nums" :title="tr('normalRateHint')">{{ candyNormalRate(state.history) }}</span>
      </button>
      <label class="relative inline-flex h-5 w-8 shrink-0 cursor-pointer items-center" :class="{ 'cursor-wait opacity-60': pending, 'opacity-50': !state || loadError }" :title="monitorTitle">
        <input type="checkbox" role="switch" class="peer sr-only" :checked="state?.enabled || false" :disabled="!state || pending || loadError" :aria-label="tr('autoDetect')" data-testid="candy-auto" @change="changeMonitoring" />
        <span class="h-4 w-8 rounded-full bg-gray-300 transition-colors peer-checked:bg-primary-500 peer-focus-visible:ring-2 peer-focus-visible:ring-primary-500 peer-focus-visible:ring-offset-2 dark:bg-dark-500" />
        <span class="pointer-events-none absolute left-0.5 h-3 w-3 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-4" />
      </label>
    </div>
    <template v-if="state?.enabled">
      <CandyHistoryBars :history="state.history" :show-rate="false" :interactive="false" />
      <div class="truncate text-[10px] leading-4 text-gray-500 dark:text-gray-400" :title="monitorTitle">{{ state.model_id }}</div>
      <div class="text-[10px] leading-4 tabular-nums text-gray-400" :title="latestTime ? new Date(latestTime).toLocaleString() : tr('noRecord')">{{ latestTime ? new Date(latestTime).toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : tr('noRecord') }}</div>
    </template>
  </div>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { CandyAccountState } from '@/api/admin/candyMonitor'
import CandyHistoryBars from './CandyHistoryBars.vue'
import { candyNormalRate, recentCandyResults } from '@/utils/candyHistory'
const props = defineProps<{ state?: CandyAccountState; pending: boolean; schedulerEnabled: boolean; loadError: boolean }>()
const emit = defineEmits<{ test: []; toggle: [enabled: boolean] }>()
const { t } = useI18n()
const tr = (key: string, params: Record<string, string | number> = {}) => t(`admin.accounts.candyMonitor.${key}`, params)
const latestTime = computed(() => recentCandyResults(props.state?.history)[0]?.started_at)
function changeMonitoring(event: Event) {
  const input = event.target as HTMLInputElement
  const enabled = input.checked
  input.checked = props.state?.enabled || false
  emit('toggle', enabled)
}
const testTitle = computed(() => `${tr('quickTest')} · ${props.state?.last_valid_answer != null
  ? `${tr('lastValidAnswer', { answer: props.state.last_valid_answer })}${props.state.last_valid_at ? ` · ${new Date(props.state.last_valid_at).toLocaleString()}` : ''}`
  : tr('quickHint')}`)
const monitorTitle = computed(() => {
  if (props.loadError) return tr('stateLoadFailed')
  if (!props.state) return tr('loading')
  if (!props.schedulerEnabled) return tr('schedulerPausedHint')
  return tr(props.state.enabled ? 'autoDetectOnHint' : 'autoDetectOffHint', { model: props.state.model_id, minutes: props.state.interval_minutes })
})
</script>
