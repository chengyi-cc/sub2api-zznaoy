<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { supportsTurnState, type TurnStateSummary, type TurnStateModelSummary } from '@/composables/useTurnStateSummaries'
import type { AccountListItem } from '@/types'
import TurnStateAcquireButton from './TurnStateAcquireButton.vue'

const props = defineProps<{ account: AccountListItem; summary?: TurnStateSummary; now: number }>()
defineEmits<{ open: []; queued: [] }>()
const { locale } = useI18n()
const chinese = computed(() => locale.value.startsWith('zh'))
const label = (zh: string, en: string) => chinese.value ? zh : en
const enabled = computed(() => props.account.extra?.codex_turn_state_auto_enabled === true && props.summary?.enabled !== false)
const models = computed(() => [...(props.summary?.models || [])].sort((first, second) => {
  if (first.model === 'gpt-6-astra') return -1
  if (second.model === 'gpt-6-astra') return 1
  return first.model.localeCompare(second.model)
}))
function details(model: TurnStateModelSummary) {
  const remaining = Date.parse(model.expires_at || '') - props.now - (props.summary?.clockOffset || 0)
  if (remaining > 0 && model.length) {
    return { text: label('可用', 'Ready'), ttl: `${Math.ceil(remaining / 60000)}${label('分', 'm')}`, valid: true }
  }
  const states: Record<string, string> = {
    queued: label('排队', 'Queued'), preparing: label('采集中', 'Acquiring'), refreshing: label('采集中', 'Acquiring'),
    unavailable: label('待重试', 'Retrying'), expired: label('已过期', 'Expired')
  }
  return { text: model.expires_at && remaining <= 0 ? label('已过期', 'Expired') : states[model.state] || label('待采集', 'Pending'), ttl: '', valid: false }
}
const tooltip = computed(() => models.value.map(model => {
  const status = details(model)
  return `${model.model} · ${status.text} ${status.ttl}${model.last_error ? '\n' + model.last_error : ''}`
}).join('\n'))
</script>

<template>
  <span v-if="!supportsTurnState(account)" class="text-xs text-gray-400">—</span>
  <div v-else class="max-w-[13rem] text-left text-[11px] leading-4 text-gray-500 dark:text-dark-400">
    <button v-if="!enabled" type="button" @click="$emit('open')">{{ label('未开启', 'Disabled') }}</button>
    <button v-else-if="summary?.failed" type="button" class="text-amber-600" @click="$emit('open')">{{ label('读取失败', 'Unavailable') }}</button>
    <button v-else-if="!summary" type="button" @click="$emit('open')">{{ label('读取中…', 'Loading…') }}</button>
    <template v-else-if="models.length">
      <span v-for="(model, index) in models.slice(0, 2)" :key="model.model" class="flex items-center gap-1 whitespace-nowrap" data-testid="turn-state-summary-model">
        <button type="button" class="flex min-w-0 items-center gap-1 text-left" :title="tooltip" @click="$emit('open')">
        <span class="max-w-[6rem] truncate font-mono">{{ model.model }}</span>
        <span class="text-gray-300 dark:text-gray-600">·</span>
        <span :class="details(model).valid ? '' : 'text-amber-600 dark:text-amber-400'">{{ details(model).text }}</span>
        <span class="font-mono tabular-nums">{{ details(model).ttl }}</span>
        <span v-if="index === 1 && models.length > 2" class="text-[10px]">+{{ models.length - 2 }}</span>
        </button>
        <TurnStateAcquireButton :account-id="account.id" :model="model.model" :disabled="!summary.configured" compact @queued="$emit('queued')" />
      </span>
    </template>
    <button v-else type="button" @click="$emit('open')">{{ summary.configured ? label('待采集', 'Pending') : label('未配置', 'Not configured') }}</button>
  </div>
</template>
