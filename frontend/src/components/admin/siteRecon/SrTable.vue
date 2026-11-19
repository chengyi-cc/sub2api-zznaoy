<template>
  <div class="card overflow-x-auto">
    <table class="min-w-full divide-y divide-gray-200 text-sm dark:divide-dark-700">
      <thead class="bg-gray-50 dark:bg-dark-800/40">
        <tr class="text-xs text-gray-500 dark:text-gray-400">
          <th v-for="(h, i) in head" :key="i" :class="i < left ? THL : TH">{{ h }}</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-100 dark:divide-dark-700/60">
        <slot />
        <tr v-if="empty">
          <td :colspan="head.length" class="px-3 py-6 text-center text-sm text-gray-400 dark:text-gray-500">
            {{ emptyText }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { TH, THL } from './format'

withDefaults(
  defineProps<{
    head: string[]
    left?: number
    empty?: boolean
    emptyText?: string
  }>(),
  { left: 1, empty: false, emptyText: '-' }
)
</script>
