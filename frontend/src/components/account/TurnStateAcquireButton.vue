<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiClient } from '@/api/client'
import { extractApiErrorMessage } from '@/utils/apiError'
import Icon from '@/components/icons/Icon.vue'

const props = withDefaults(defineProps<{ accountId: number; model: string; disabled?: boolean; compact?: boolean }>(), { disabled: false, compact: false })
const emit = defineEmits<{ queued: [] }>()
const { locale } = useI18n()
const label = (zh: string, en: string) => locale.value.startsWith('zh') ? zh : en
const pending = ref(false)
const submitted = ref(false)
const error = ref('')
let controller: AbortController | undefined
let generation = 0
const title = computed(() => {
  if (error.value) return error.value
  if (submitted.value) return label('已提交采集任务，已有任务则继续；不表示已经获取成功。', 'Acquisition submitted or already running; this does not mean a new state is ready.')
  if (props.disabled) return label('请先保存账号并开启采集、配置所选出口。', 'Save and enable acquisition and configure the selected source first.')
  return label('重新获取 ', 'Reacquire ') + props.model + label('：使用已保存的账号规则和出口，未过期旧头继续使用。', ': uses saved account rules and source; a valid old state remains usable.')
})

async function acquire(): Promise<void> {
  if (pending.value || props.disabled || !props.model.trim()) return
  const current = generation
  pending.value = true
  submitted.value = false
  error.value = ''
  controller = new AbortController()
  try {
    await apiClient.post('/admin/accounts/' + props.accountId + '/turn-state/acquire', { model: props.model }, { signal: controller.signal })
    if (current !== generation) return
    submitted.value = true
    emit('queued')
  } catch (failure) {
    if (current !== generation) return
    error.value = extractApiErrorMessage(failure, label('启动失败，请重试。', 'Unable to start acquisition. Please retry.'))
  } finally {
    if (current === generation) pending.value = false
  }
}

watch(() => [props.accountId, props.model], () => {
  generation += 1
  controller?.abort()
  pending.value = false
  submitted.value = false
  error.value = ''
})
onBeforeUnmount(() => { generation += 1; controller?.abort() })
</script>

<template>
  <span class="inline-flex items-center gap-1" :class="compact ? 'shrink-0' : 'flex-wrap'">
    <button type="button" :disabled="disabled || pending" :title="title" :aria-label="label('重新获取 ', 'Reacquire ') + model" :aria-busy="pending" :data-testid="'turn-state-reacquire-' + model" :class="compact ? 'rounded p-0.5 text-gray-400 hover:text-primary-600 disabled:opacity-40' : 'btn btn-secondary px-2 py-1 text-xs disabled:opacity-40'" @click.stop="acquire">
      <Icon v-if="compact" :name="error ? 'exclamationCircle' : submitted ? 'check' : 'refresh'" size="sm" :class="[pending ? 'animate-spin' : '', error ? 'text-amber-600' : '']" />
      <template v-else>{{ pending ? label('提交中…', 'Submitting…') : label('重新获取', 'Reacquire') }}</template>
    </button>
    <span v-if="!compact && error" role="alert" class="text-xs text-amber-600">{{ error }}</span>
    <span v-else-if="!compact && submitted" role="status" class="text-xs text-gray-500">{{ label('已提交，等待采集结果', 'Submitted; awaiting acquisition result') }}</span>
    <span v-if="compact && (error || submitted)" class="sr-only" :role="error ? 'alert' : 'status'">{{ title }}</span>
  </span>
</template>
