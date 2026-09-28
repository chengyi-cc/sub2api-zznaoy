<template>
  <span v-if="enabled || wasDisabled" data-testid="excel-protocol-badge"
    class="inline-flex items-center rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500 dark:bg-dark-700 dark:text-gray-400"
    :title="title">{{ enabled ? 'Excel' : t('admin.accounts.openai.excelReturnedNative') }}</span>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
const props = defineProps<{ extra?: Record<string, unknown> | null }>()
const { t } = useI18n()
const enabled = computed(() => props.extra?.openai_excel_bps === true)
const event = computed(() => props.extra?.openai_excel_bps_last_transition as { reason?: string; at?: string } | undefined)
const wasDisabled = computed(() => event.value?.reason === 'upstream_403')
const title = computed(() => {
  const reason = wasDisabled.value ? t('admin.accounts.openai.excel403History')
    : event.value?.reason === 'candy_incorrect' ? t('admin.accounts.openai.excelEnabledByMonitor')
    : t('admin.accounts.openai.excelBPS')
  const at = event.value?.at
  return at ? reason + ' · ' + new Date(at).toLocaleString() : reason
})
</script>
