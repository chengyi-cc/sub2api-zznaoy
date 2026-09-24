<template>
  <div class="w-40" data-testid="history-bars">
    <div v-if="showRate" class="mb-1 flex items-center justify-between gap-3 text-[11px]" :title="tr('normalRateHint')">
      <span class="text-gray-500 dark:text-gray-400">{{ tr('normalRate') }}</span>
      <span class="font-medium tabular-nums text-gray-700 dark:text-gray-200" data-testid="normal-rate">{{ candyNormalRate(history) }}</span>
    </div>
    <div class="flex items-center justify-between" :aria-label="tr('historyStatus')">
      <component :is="interactive ? 'button' : 'span'" v-for="(result, index) in slots" :key="index" :type="interactive ? 'button' : undefined"
        class="flex h-6 w-3 items-center justify-center rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-500"
        :disabled="interactive ? !result : undefined" :title="label(result)" :aria-label="label(result)" @click="interactive && emit('select')">
        <span class="h-4 w-1.5 rounded-[1px]" :class="candyBarClass(result)" :data-verdict="result?.verdict || 'empty'" />
      </component>
    </div>
  </div>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { CandyResult } from '@/api/admin/candyMonitor'
import { candyBarClass, candyNormalRate, recentCandyResults } from '@/utils/candyHistory'
const props = withDefaults(defineProps<{ history?: CandyResult[]; showRate?: boolean; interactive?: boolean }>(), { history: () => [], showRate: true, interactive: true })
const emit = defineEmits<{ select: [] }>()
const { t } = useI18n()
const tr = (key: string) => t(`admin.accounts.candyMonitor.${key}`)
const slots = computed(() => {
  const recent = recentCandyResults(props.history).reverse()
  return [...Array<CandyResult | null>(10 - recent.length).fill(null), ...recent]
})
function label(result: CandyResult | null) {
  if (!result) return tr('noRecord')
  return `${new Date(result.started_at).toLocaleString()} / ${tr(`verdict.${result.verdict}`)}${result.actual == null ? '' : ` · ${result.actual}`} / ${result.model_id}`
}
</script>
