<template>
  <div class="space-y-3 rounded-xl border border-primary-200 bg-primary-50/60 p-4 text-sm dark:border-primary-800 dark:bg-primary-950/30" data-testid="candy-test-panel">
    <p class="text-gray-700 dark:text-gray-200">{{ t('admin.accounts.candy.hint') }}</p>
    <details class="text-xs text-gray-600 dark:text-gray-300">
      <summary class="cursor-pointer py-1 font-medium">{{ t('admin.accounts.candy.showQuestion') }}</summary>
      <div class="mt-2 space-y-2 leading-6">
        <p>{{ t('admin.accounts.candy.question') }}</p>
        <table class="w-full text-left"><thead><tr><th>{{ t('admin.accounts.candy.shape') }}</th><th>{{ t('admin.accounts.candy.apple') }}</th><th>{{ t('admin.accounts.candy.peach') }}</th><th>{{ t('admin.accounts.candy.melon') }}</th></tr></thead><tbody><tr><td>{{ t('admin.accounts.candy.round') }}</td><td>7</td><td>9</td><td>8</td></tr><tr><td>{{ t('admin.accounts.candy.star') }}</td><td>7</td><td>6</td><td>4</td></tr></tbody></table>
        <p>{{ t('admin.accounts.candy.formatHint') }}</p>
      </div>
    </details>
    <div v-if="result" class="space-y-2 border-t border-primary-200 pt-3 dark:border-primary-800" role="status" data-testid="candy-test-result" :data-verdict="result.verdict">
      <p class="font-semibold" :class="result.verdict === 'pass' ? 'text-green-700 dark:text-green-400' : 'text-amber-700 dark:text-amber-400'">{{ t(`admin.accounts.candy.verdict.${result.verdict}`) }}</p>
      <p v-if="result.actual !== undefined" class="text-gray-700 dark:text-gray-200">{{ t('admin.accounts.candy.actual', { answer: result.actual }) }}</p>
      <p v-if="result.expected !== undefined" class="text-gray-600 dark:text-gray-300">{{ t('admin.accounts.candy.expected', { answer: result.expected }) }}</p>
      <p v-if="result.duration_ms !== undefined" class="text-xs text-gray-500 dark:text-gray-400">{{ t('admin.accounts.candy.duration', { seconds: (result.duration_ms / 1000).toFixed(1) }) }}</p>
    </div>
    <p class="text-xs leading-5 text-gray-500 dark:text-gray-400">{{ t('admin.accounts.candy.disclaimer') }}</p>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { CandyTestResult } from '@/utils/accountCandyTest'

defineProps<{ result: CandyTestResult | null }>()
const { t } = useI18n()
</script>
