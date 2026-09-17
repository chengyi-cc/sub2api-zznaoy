<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiClient } from '@/api/client'

const props = defineProps<{ accountId: number; modelValue: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: boolean] }>()
const { locale } = useI18n()
const chinese = computed(() => locale.value.startsWith('zh'))
type ModelStatus = { model: string; state: string; expires_at?: string; last_error?: string }
const status = ref<{ configured: boolean; enabled: boolean; models: ModelStatus[] } | null>(null)
const error = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined
let controller: AbortController | undefined
let generation = 0

function stateText(value: string): string {
  const labels: Record<string, string> = {
    preparing: '采集中', ready: '可用', refreshing: '刷新中', unavailable: '暂不可用', expired: '已过期'
  }
  return chinese.value ? labels[value] || value : value
}

async function refresh(current: number): Promise<void> {
  controller?.abort()
  controller = new AbortController()
  try {
    const { data } = await apiClient.get('/admin/accounts/' + props.accountId + '/turn-state', { signal: controller.signal })
    if (current !== generation) return
    status.value = data
    error.value = false
  } catch {
    if (current !== generation) return
    error.value = true
  } finally {
    if (current === generation) timer = setTimeout(() => void refresh(current), 5000)
  }
}

watch(() => props.accountId, () => {
  generation += 1
  clearTimeout(timer)
  status.value = null
  void refresh(generation)
}, { immediate: true })

onBeforeUnmount(() => {
  generation += 1
  clearTimeout(timer)
  controller?.abort()
})
</script>

<template>
  <section class="space-y-3 border-t border-gray-200 pt-4 dark:border-dark-600">
    <label class="flex cursor-pointer items-center justify-between gap-4">
      <span>
        <span class="input-label mb-1 block">{{ chinese ? '自动采集并注入轮次状态' : 'Automatic turn-state acquisition' }}</span>
        <span class="block text-xs text-gray-500 dark:text-gray-400">
          {{ chinese ? '按账号与模型隔离；通过一次性 IPv6 出口采集 292 字符状态，最终发送前覆盖。保存后生效，新导入账号默认开启。' : 'Isolated per account and model. Acquire 292-character states through disposable IPv6 exits and override at final dispatch. Save to apply; enabled by default for new imports.' }}
        </span>
      </span>
      <input type="checkbox" class="h-5 w-5 rounded border-gray-300 text-primary-600" :checked="modelValue" @change="emit('update:modelValue', ($event.target as HTMLInputElement).checked)">
    </label>
    <p v-if="error" class="text-xs text-amber-600">{{ chinese ? '状态暂时无法读取；不影响保存开关。' : 'Status unavailable; the switch can still be saved.' }}</p>
    <p v-else-if="status && !status.configured" class="text-xs text-amber-600">
      {{ chinese ? '出口池尚未配置或配置无效，当前不会自动采集。请配置服务端 TURN_STATE_POOL_* 环境变量（出口服务连接参数）。' : 'Pool configuration is missing or invalid. Configure the server TURN_STATE_POOL_* environment variables.' }}
    </p>
    <template v-else-if="status">
      <p class="text-xs text-gray-500">{{ chinese ? '生效状态：' : 'Saved state: ' }}{{ status.enabled ? (chinese ? '开启' : 'enabled') : (chinese ? '关闭' : 'disabled') }}</p>
      <p v-if="status.enabled && !status.models.length" class="text-xs text-gray-500">{{ chinese ? '等待该账号的模型请求。首次采集在后台进行，未准备好时保留原有请求流程。' : 'Waiting for model traffic. Acquisition runs in the background; original forwarding remains active until ready.' }}</p>
      <ul v-if="status.enabled" class="space-y-2 text-xs">
        <li v-for="model in status.models" :key="model.model" class="rounded bg-gray-50 p-2 dark:bg-dark-700">
          <span class="font-mono">{{ model.model }}</span> · {{ stateText(model.state) }}
          <span v-if="model.expires_at"> · {{ chinese ? '到期：' : 'Expires: ' }}{{ new Date(model.expires_at).toLocaleString() }}</span>
          <p v-if="model.last_error" class="mt-1 text-amber-600">{{ model.last_error }}</p>
        </li>
      </ul>
    </template>
  </section>
</template>
