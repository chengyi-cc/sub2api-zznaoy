<template>
  <span class="inline-flex whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium" :class="color" :data-verdict="verdict || 'untested'">
    {{ t(`admin.accounts.candyMonitor.verdict.${verdict || 'untested'}`) }}<template v-if="actual != null"> · {{ actual }}</template>
  </span>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { CandyResult } from '@/api/admin/candyMonitor'
const props = defineProps<{ verdict?: CandyResult['verdict']; actual?: number }>()
const { t } = useI18n()
const color = computed(() => props.verdict === 'pass'
  ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300'
  : props.verdict === 'incorrect'
    ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
    : 'bg-gray-100 text-gray-700 dark:bg-dark-700 dark:text-gray-300')
</script>
