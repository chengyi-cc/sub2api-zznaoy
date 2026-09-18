<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiClient } from '@/api/client'
import TurnStateSettingsPanel from './TurnStateSettingsPanel.vue'

const props = withDefaults(defineProps<{ accountId: number; modelValue: boolean; profile?: string; source?: string }>(), { profile: 'team', source: 'purchased' })
const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  'update:profile': [value: string]
  'update:source': [value: string]
}>()
const { locale } = useI18n()
const chinese = computed(() => locale.value.startsWith('zh'))
type ModelStatus = {
  model: string; state: string; expires_at?: string; refresh_at?: string; retry_at?: string
  last_error?: string; length?: number; country?: string; source_ip?: string
}
type Attempt = {
  at: string; model: string; profile: string; source: string; country?: string; actual_country?: string; kind?: string
  source_ip?: string; status: number; length: number; accepted: boolean; error?: string; duration_ms: number
}
type Snapshot = {
  configured: boolean; enabled: boolean; models: ModelStatus[]; server_time?: string
  sources?: Record<string, boolean>; countries?: string[]; history?: Attempt[]; history_error?: string
}
const status = ref<Snapshot | null>(null)
const error = ref(false)
const showHistory = ref(false)
const showSettings = ref(false)
const now = ref(Date.now())
const serverOffset = ref(0)
const selectedConfigured = computed(() => status.value?.sources?.[props.source] ?? status.value?.configured)
let timer: ReturnType<typeof setTimeout> | undefined
let controller: AbortController | undefined
let generation = 0
const clockTimer = setInterval(() => { now.value = Date.now() }, 1000)

function stateText(value: string): string {
  const labels: Record<string, string> = {
    queued: '等待采集', preparing: '采集中', ready: '可用', refreshing: '刷新中', unavailable: '暂不可用', expired: '已过期'
  }
  return chinese.value ? labels[value] || value : value
}

function countryName(value?: string): string {
  if (!value) return '—'
  const names: Record<string, string> = { US: '美国', SG: '新加坡', JP: '日本', ZA: '南非', BR: '巴西', AE: '阿联酋', DE: '德国', GB: '英国', FR: '法国', NL: '荷兰', AU: '澳大利亚', CA: '加拿大', IT: '意大利', ES: '西班牙', SE: '瑞典', NO: '挪威', CH: '瑞士', PL: '波兰', KR: '韩国', IN: '印度' }
  return chinese.value ? names[value] || value : value
}

function remaining(value?: string): string {
  if (!value) return '—'
  const seconds = Math.max(0, Math.floor((Date.parse(value) - now.value - serverOffset.value) / 1000))
  if (!Number.isFinite(seconds)) return '—'
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

async function refresh(current: number): Promise<void> {
  controller?.abort()
  controller = new AbortController()
  try {
    const suffix = showHistory.value ? '?history=1' : ''
    const { data } = await apiClient.get('/admin/accounts/' + props.accountId + '/turn-state' + suffix, { signal: controller.signal })
    if (current !== generation) return
    status.value = data
    const serverTime = Date.parse(data.server_time || '')
    if (Number.isFinite(serverTime)) serverOffset.value = serverTime - Date.now()
    now.value = Date.now()
    error.value = false
  } catch {
    if (current !== generation) return
    error.value = true
  } finally {
    if (current === generation) timer = setTimeout(() => void refresh(current), 5000)
  }
}

function toggleHistory(): void {
  showHistory.value = !showHistory.value
  generation += 1
  clearTimeout(timer)
  void refresh(generation)
}

function settingsSaved(): void {
  generation += 1
  clearTimeout(timer)
  void refresh(generation)
}

watch(() => props.accountId, () => {
  generation += 1
  clearTimeout(timer)
  status.value = null
  showHistory.value = false
  showSettings.value = false
  serverOffset.value = 0
  void refresh(generation)
}, { immediate: true })

onBeforeUnmount(() => {
  generation += 1
  clearTimeout(timer)
  clearInterval(clockTimer)
  controller?.abort()
})
</script>

<template>
  <section class="space-y-3 border-t border-gray-200 pt-4 dark:border-dark-600">
    <div class="flex items-center justify-between gap-4">
      <span>
        <span class="input-label mb-1 block">{{ chinese ? '自动采集并注入轮次状态' : 'Automatic turn-state acquisition' }}</span>
        <span class="block text-xs text-gray-500 dark:text-gray-400">
          {{ chinese ? '按账号和模型单独保存。新导入账号默认开启，默认使用 Team 规则和购买代理。修改后请保存账号。' : 'Stored separately per account and model. New imports default to enabled, Team rules and purchased proxies. Save the account to apply changes.' }}
        </span>
      </span>
      <div class="flex shrink-0 items-center gap-3">
        <button type="button" class="btn btn-secondary text-xs" :aria-expanded="showSettings" data-testid="turn-state-configure" @click="showSettings = !showSettings">{{ showSettings ? (chinese ? '收起配置' : 'Hide settings') : (chinese ? '配置' : 'Configure') }}</button>
        <input type="checkbox" class="h-5 w-5 rounded border-gray-300 text-primary-600" :aria-label="chinese ? '自动采集并注入轮次状态' : 'Automatic turn-state acquisition'" :checked="modelValue" @change="emit('update:modelValue', ($event.target as HTMLInputElement).checked)">
      </div>
    </div>
    <TurnStateSettingsPanel v-if="showSettings" :account-id="accountId" @saved="settingsSaved" />
    <div class="grid gap-3 sm:grid-cols-2">
      <label class="text-sm">
        <span class="input-label">{{ chinese ? '账号采集规则' : 'Account acquisition rules' }}</span>
        <select class="input mt-1 w-full" :value="profile" data-testid="turn-state-profile" @change="emit('update:profile', ($event.target as HTMLSelectElement).value)">
          <option value="team">Team · 332</option>
          <option value="pro">Pro · 292</option>
        </select>
      </label>
      <label class="text-sm">
        <span class="input-label">{{ chinese ? '采集出口' : 'Acquisition source' }}</span>
        <select class="input mt-1 w-full" :value="source" data-testid="turn-state-source" @change="emit('update:source', ($event.target as HTMLSelectElement).value)">
          <option value="purchased">{{ chinese ? '购买代理 IPv4 · 自动换国家' : 'Purchased IPv4 proxy · country rotation' }}</option>
          <option value="ipv6_pool">{{ chinese ? '自建 IPv6 池 · 保留原方式' : 'Self-hosted IPv6 pool' }}</option>
        </select>
      </label>
    </div>
    <p class="text-xs text-gray-500 dark:text-gray-400">
      {{ chinese ? '连续 3 次不合格或采集失败后换国家。按配置时间开始刷新（默认签发后48分钟）；失败时继续使用未过期旧值，满60分钟停止使用。' : 'Rotate countries after 3 rejected or failed probes. Refresh at the configured age (default 48 minutes after issuance); keep a valid old state on refresh failure, and stop using it after 60 minutes.' }}
    </p>
    <p v-if="source === 'purchased' && status?.countries?.length" class="text-xs text-gray-500">
      {{ chinese ? '候选国家：' : 'Candidate countries: ' }}{{ status.countries.map(countryName).join('、') }}
    </p>
    <p v-if="error" class="text-xs text-amber-600">{{ chinese ? '状态暂时无法读取；不影响保存设置。' : 'Status unavailable; settings can still be saved.' }}</p>
    <p v-else-if="status && !selectedConfigured" class="text-xs text-amber-600">
      {{ chinese ? '所选采集服务尚未配置或配置无效，请点击开关旁的“配置”填写并保存。' : 'The selected acquisition service is missing or invalid. Click Configure next to the switch.' }}
    </p>
    <template v-if="status">
      <p class="text-xs text-gray-500">{{ chinese ? '生效状态：' : 'Saved state: ' }}{{ status.enabled ? (chinese ? '开启' : 'enabled') : (chinese ? '关闭' : 'disabled') }}</p>
      <p v-if="status.enabled && !status.models.length" class="text-xs text-gray-500">{{ chinese ? '等待该账号的模型请求。首次采集在后台进行，未准备好时保留原有请求流程。' : 'Waiting for model traffic. Original forwarding remains active until a state is ready.' }}</p>
      <ul v-if="status.enabled" class="space-y-2 text-xs">
        <li v-for="model in status.models" :key="model.model" class="rounded bg-gray-50 p-2 dark:bg-dark-700">
          <span class="font-mono">{{ model.model }}</span> · {{ stateText(model.state) }}
          <span v-if="model.length"> · {{ model.length }}</span>
          <span v-if="model.country"> · {{ countryName(model.country) }}</span>
          <p v-if="model.expires_at" class="mt-1">
            {{ chinese ? '剩余有效期：' : 'Time remaining: ' }}<strong class="font-mono" data-testid="turn-state-ttl">{{ remaining(model.expires_at) }}</strong>
            · {{ chinese ? '到期：' : 'Expires: ' }}{{ new Date(model.expires_at).toLocaleString() }}
          </p>
          <p v-if="model.refresh_at" class="mt-1">{{ chinese ? '提前刷新倒计时：' : 'Refresh countdown: ' }}{{ remaining(model.refresh_at) }}</p>
          <p v-if="model.last_error" class="mt-1 text-amber-600">{{ model.last_error }}</p>
          <p v-if="model.retry_at && model.last_error" class="mt-1">{{ chinese ? '下次重试倒计时：' : 'Next retry: ' }}{{ remaining(model.retry_at) }}</p>
        </li>
      </ul>
    </template>
    <button type="button" class="btn btn-secondary text-xs" :aria-expanded="showHistory" data-testid="turn-state-history" @click="toggleHistory">
      {{ showHistory ? (chinese ? '收起检测记录' : 'Hide detection history') : (chinese ? '查看检测记录' : 'View detection history') }}
    </button>
    <div v-if="showHistory" class="max-h-80 overflow-auto rounded border border-gray-200 dark:border-dark-600">
      <p v-if="status?.history_error" class="p-3 text-xs text-amber-600">{{ chinese ? '检测记录暂时无法读取。' : 'Detection history is temporarily unavailable.' }}</p>
      <p v-else-if="!status?.history" class="p-3 text-xs text-gray-500">{{ chinese ? '正在读取检测记录…' : 'Loading history…' }}</p>
      <p v-else-if="!status.history.length" class="p-3 text-xs text-gray-500">{{ chinese ? '暂无检测记录；显示最近 7 天内的最多 200 条。' : 'No detections yet. Shows up to 200 entries from the past 7 days.' }}</p>
      <table v-else class="w-full text-left text-xs">
        <thead class="bg-gray-50 dark:bg-dark-700"><tr>
          <th class="p-2">{{ chinese ? '时间 / 模型' : 'Time / model' }}</th>
          <th class="p-2">{{ chinese ? '规则 / 出口' : 'Rules / source' }}</th>
          <th class="p-2">{{ chinese ? '结果' : 'Result' }}</th>
        </tr></thead>
        <tbody>
          <tr v-for="(entry, index) in status.history" :key="entry.at + ':' + index" class="border-t border-gray-100 dark:border-dark-600">
            <td class="p-2 align-top">{{ new Date(entry.at).toLocaleString() }}<br><span class="font-mono">{{ entry.model }}</span></td>
            <td class="p-2 align-top">
              {{ entry.profile === 'pro' ? 'Pro' : 'Team' }} · {{ entry.source === 'ipv6_pool' ? (chinese ? '自建 IPv6' : 'IPv6 pool') : 'IPv4' }} · {{ countryName(entry.country) }}
              <br><span class="break-all font-mono">{{ entry.source_ip || '—' }}</span>
              <p v-if="entry.actual_country && entry.actual_country !== entry.country">{{ chinese ? '实际国家：' : 'Actual country: ' }}{{ countryName(entry.actual_country) }}</p>
            </td>
            <td class="p-2 align-top">
              <span :class="entry.accepted ? 'text-green-600' : 'text-amber-600'">{{ entry.kind === 'response_rejection' ? (chinese ? '响应触发重采集' : 'Response triggered reacquisition') : entry.accepted ? (chinese ? '通过采集规则' : 'Accepted') : (chinese ? '未通过' : 'Rejected') }}</span>
              · {{ entry.length || '—' }}<template v-if="entry.kind !== 'response_rejection'"> · {{ entry.duration_ms }} ms</template>
              <p v-if="entry.error" class="mt-1 break-words text-gray-500">{{ entry.error }}</p>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
