<template>
  <div v-if="items.length" class="space-y-2" data-test="site-recon-alerts">
    <div
      v-for="(a, i) in items"
      :key="i"
      class="flex items-start gap-3 rounded-lg border px-4 py-2.5 text-sm"
      :class="boxClass(a.level)"
    >
      <span class="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[11px] font-semibold" :class="badgeClass(a.level)">
        {{ t(`siteRecon.level.${a.level}`) }}
      </span>
      <span class="break-words">{{ a.text }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { SiteReconAlert } from '@/api/admin/siteRecon'

defineProps<{ items: SiteReconAlert[] }>()
const { t } = useI18n()

function boxClass(level: SiteReconAlert['level']) {
  if (level === 'bad') return 'border-red-200 bg-red-50 text-red-800 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200'
  if (level === 'warn') return 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200'
  return 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-900/60 dark:bg-blue-900/20 dark:text-blue-200'
}

function badgeClass(level: SiteReconAlert['level']) {
  if (level === 'bad') return 'bg-red-600 text-white'
  if (level === 'warn') return 'bg-amber-500 text-white'
  return 'bg-blue-500 text-white'
}
</script>
