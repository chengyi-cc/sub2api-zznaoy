<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { apiClient } from '@/api/client'

const emit = defineEmits<{ saved: [] }>()
const props = defineProps<{ accountId: number }>()
const { locale } = useI18n()
const chinese = computed(() => locale.value.startsWith('zh'))
const label = (zh: string, en: string) => chinese.value ? zh : en
type Settings = {
  revision: string; purchased_enabled: boolean; proxy_host: string; proxy_username: string
  proxy_password?: string; proxy_password_configured: boolean; proxy_upstream?: string
  proxy_upstream_configured: boolean; clear_proxy_upstream?: boolean; countries: string
  ipv6_enabled: boolean; pool_url: string; pool_token?: string; pool_token_configured: boolean
  pool_ca: string; attempts: number; concurrency: number
}
const settings = ref<Settings | null>(null)
const loading = ref(true)
const saving = ref(false)
const error = ref('')
const saved = ref(false)
const acquisitionModel = ref('gpt-6-astra')
const acquiring = ref(false)
const acquisitionStarted = ref(false)
const controller = new AbortController()
const endpoint = '/admin/accounts/turn-state/settings'

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    const { data } = await apiClient.get<Settings>(endpoint, { signal: controller.signal })
    settings.value = { ...data, proxy_password: '', pool_token: '', proxy_upstream: '', clear_proxy_upstream: false }
  } catch {
    if (!controller.signal.aborted) error.value = label('读取配置失败，请重试。', 'Unable to load settings. Please retry.')
  } finally {
    loading.value = false
  }
}

async function save(): Promise<void> {
  if (!settings.value || saving.value) return
  saving.value = true
  error.value = ''
  saved.value = false
  try {
    const { data } = await apiClient.put<Settings>(endpoint, settings.value)
    settings.value = { ...data, proxy_password: '', pool_token: '', proxy_upstream: '', clear_proxy_upstream: false }
    saved.value = true
    emit('saved')
  } catch (failure: unknown) {
    const response = failure as { response?: { data?: { message?: string } }; message?: string }
    error.value = response.response?.data?.message || response.message || label('保存失败，原配置保持不变。', 'Save failed. Previous settings are unchanged.')
  } finally {
    saving.value = false
  }
}

async function acquireNow(): Promise<void> {
  const model = acquisitionModel.value.trim()
  if (!model || acquiring.value) return
  acquiring.value = true
  error.value = ''
  acquisitionStarted.value = false
  try {
    await apiClient.post('/admin/accounts/' + props.accountId + '/turn-state/acquire', { model })
    acquisitionStarted.value = true
    emit('saved')
  } catch (failure: unknown) {
    const response = failure as { response?: { data?: { message?: string } }; message?: string }
    error.value = response.response?.data?.message || response.message || label('立即采集启动失败。', 'Unable to start acquisition.')
  } finally {
    acquiring.value = false
  }
}

async function readCertificate(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file || !settings.value) return
  error.value = ''
  if (file.size > 64 * 1024) {
    error.value = label('证书文件不能超过64KB。', 'Certificate must not exceed 64KB.')
    input.value = ''
    return
  }
  try {
    settings.value.pool_ca = await file.text()
    saved.value = false
  } catch {
    error.value = label('无法读取证书文件。', 'Unable to read certificate file.')
  }
  input.value = ''
}

function preventAccountSubmit(event: KeyboardEvent): void {
  if (event.key === 'Enter' && event.target instanceof HTMLInputElement) event.preventDefault()
}

onMounted(load)
onBeforeUnmount(() => { controller.abort(); settings.value = null })
</script>

<template>
  <section class="space-y-3 rounded-lg border border-primary-200 bg-gray-50 p-4 dark:border-dark-500 dark:bg-dark-700" data-testid="turn-state-settings" @keydown="preventAccountSubmit" @input="saved = false">
    <h4 class="font-medium">{{ label('采集服务配置（所有账号共用）', 'Acquisition settings (shared by all accounts)') }}</h4>
    <p class="text-xs text-gray-500 dark:text-gray-400">{{ label('这里只配置获取请求头的出口，不改变业务请求代理。保存后直接生效，无需终端或重启；其他服务实例最多15秒同步。账号开关与采集方式仍需保存账号。', 'Only configures header acquisition, not business request proxies. Applies without a terminal or restart; other instances sync within 15 seconds. Save account switches and source selections separately.') }}</p>
    <p v-if="loading" class="text-sm">{{ label('正在读取配置…', 'Loading settings…') }}</p>
    <fieldset v-if="settings" :disabled="saving" class="space-y-4">
      <div class="space-y-3">
        <label class="flex items-center gap-2 text-sm font-medium"><input v-model="settings.purchased_enabled" type="checkbox" data-testid="settings-ipv4-enabled">{{ label('启用购买代理 IPv4', 'Enable purchased IPv4 proxy') }}</label>
        <div class="grid gap-3 sm:grid-cols-2">
          <label class="text-xs">{{ label('代理地址（主机名:端口）', 'Proxy address (host:port)') }}<input v-model="settings.proxy_host" class="input mt-1 w-full" placeholder="gate1.ipweb.cc:7778" data-testid="settings-proxy-host"></label>
          <label class="text-xs">{{ label('用户名模板', 'Username template') }}<input v-model="settings.proxy_username" class="input mt-1 w-full" placeholder="YOUR_PREFIX_{country}___5_{session}" autocomplete="off" data-testid="settings-proxy-username"></label>
          <label class="text-xs">{{ label('代理密码', 'Proxy password') }}<input v-model="settings.proxy_password" type="password" autocomplete="new-password" class="input mt-1 w-full" :placeholder="settings.proxy_password_configured ? label('已配置；留空保留原密码', 'Configured; leave blank to keep') : label('请输入代理密码', 'Enter proxy password')" data-testid="settings-proxy-password"></label>
          <label class="text-xs">{{ label('候选国家（逗号分隔，最多20个）', 'Candidate countries (comma separated, up to 20)') }}<input v-model="settings.countries" class="input mt-1 w-full" placeholder="DE,GB,FR,SG,ZA,BR,AE,US,JP,NL,CA,AU,IT,ES,SE,NO,CH,PL,KR,IN" data-testid="settings-countries"></label>
        </div>
        <p class="text-xs text-gray-500">{{ label('用户名须包含 {country}（国家代码）和 {session}（自动生成的随机编号）。例如美国US、德国DE、日本JP。', 'Username must include {country} and {session}. Examples: US, DE, JP.') }}</p>
        <label class="block text-xs">{{ label('可选前置代理（用于连接购买代理，SOCKS5协议）', 'Optional upstream SOCKS5 proxy (connects to purchased proxy)') }}<input v-model="settings.proxy_upstream" type="password" autocomplete="new-password" class="input mt-1 w-full" :disabled="settings.clear_proxy_upstream" :placeholder="settings.proxy_upstream_configured ? label('已配置；留空保留，填写则替换', 'Configured; leave blank to keep, enter to replace') : 'socks5://host:7897'" data-testid="settings-upstream"></label>
        <label v-if="settings.proxy_upstream_configured" class="flex items-center gap-2 text-xs"><input v-model="settings.clear_proxy_upstream" type="checkbox">{{ label('清除已保存的前置代理', 'Clear saved upstream proxy') }}</label>
        <p class="text-xs text-gray-500">{{ label('海外服务器通常不需要前置代理。这里必须是服务器可访问的地址；容器内127.0.0.1不是你的电脑。', 'Overseas servers usually need no upstream proxy. Use an address reachable by the server; 127.0.0.1 inside a container is not your computer.') }}</p>
      </div>
      <div class="space-y-3 border-t border-gray-200 pt-3 dark:border-dark-500">
        <label class="flex items-center gap-2 text-sm font-medium"><input v-model="settings.ipv6_enabled" type="checkbox" data-testid="settings-ipv6-enabled">{{ label('启用自建 IPv6 池', 'Enable self-hosted IPv6 pool') }}</label>
        <div class="grid gap-3 sm:grid-cols-2">
          <label class="text-xs">{{ label('IPv6池地址（HTTPS加密地址）', 'IPv6 pool HTTPS URL') }}<input v-model="settings.pool_url" class="input mt-1 w-full" placeholder="https://your-server:18443" data-testid="settings-pool-url"></label>
          <label class="text-xs">{{ label('池管理密钥（至少32字符）', 'Pool token (at least 32 characters)') }}<input v-model="settings.pool_token" type="password" autocomplete="new-password" class="input mt-1 w-full" :placeholder="settings.pool_token_configured ? label('已配置；留空保留原密钥', 'Configured; leave blank to keep') : label('请输入池管理密钥', 'Enter pool token')" data-testid="settings-pool-token"></label>
        </div>
        <label class="block text-xs">{{ label('CA证书（验证自建池身份的证书，公有证书可留空）', 'CA certificate (trusts the pool; optional for public certificates)') }}<textarea v-model="settings.pool_ca" rows="4" class="input mt-1 w-full font-mono" placeholder="-----BEGIN CERTIFICATE-----" data-testid="settings-pool-ca" /></label>
        <label class="block text-xs">{{ label('或上传证书文件（读取内容后随配置保存）', 'Or upload a certificate file (saved with settings)') }}<input type="file" accept=".crt,.pem,.cer" class="mt-1 block w-full" data-testid="settings-ca-upload" @change="readCertificate"></label>
      </div>
      <div class="grid gap-3 border-t border-gray-200 pt-3 dark:border-dark-500 sm:grid-cols-2">
        <label class="text-xs">{{ label('每轮最多尝试次数（1–30）', 'Attempts per round (1–30)') }}<input v-model.number="settings.attempts" type="number" min="1" max="30" class="input mt-1 w-full"></label>
        <label class="text-xs">{{ label('同时采集任务数（1–16）', 'Concurrent acquisition tasks (1–16)') }}<input v-model.number="settings.concurrency" type="number" min="1" max="16" class="input mt-1 w-full"></label>
      </div>
      <div class="space-y-2 border-t border-gray-200 pt-3 dark:border-dark-500">
        <h5 class="text-sm font-medium">{{ label('立即采集（立即开始换出口并检测）', 'Acquire now (start rotating and probing immediately)') }}</h5>
        <div class="flex flex-col gap-2 sm:flex-row">
          <input v-model="acquisitionModel" type="text" maxlength="256" :aria-label="label('采集模型', 'Acquisition model')" class="input flex-1" placeholder="gpt-6-astra" data-testid="settings-acquisition-model">
          <button type="button" class="btn btn-secondary text-sm" :disabled="acquiring || !acquisitionModel.trim()" data-testid="settings-acquire-now" @click="acquireNow">{{ acquiring ? label('提交中…', 'Submitting…') : label('立即采集／重新获取', 'Acquire / refresh now') }}</button>
        </div>
        <p class="text-xs text-gray-500">{{ label('使用已保存的账号规则和出口，请先保存配置及账号修改。点击后后台立即尝试获取；并发满时排队，正在采集时不重复创建任务。仍有效的旧头继续使用，结果见检测记录。', 'Uses saved account rules and source; save configuration and account edits first. Starts in the background or queues when busy; an existing acquisition is reused. A valid old state remains usable. See detection history for results.') }}</p>
        <p v-if="acquisitionStarted" role="status" class="text-xs text-green-600">{{ label('已提交采集任务（已有任务则继续）；请查看检测记录。', 'Acquisition submitted (existing tasks continue). Check detection history.') }}</p>
      </div>
      <p class="text-xs text-gray-500">{{ label('密码和密钥加密保存在服务器数据库，不会返回网页。仅保存配置，不会立即发送测试请求；成功采集以检测记录为准。', 'Secrets are encrypted in the server database and never returned to the page. Saving does not send a test request; check detection history for acquisition results.') }}</p>
      <button type="button" class="btn btn-primary text-sm" :disabled="saving" data-testid="settings-save" @click="save">{{ saving ? label('保存中…', 'Saving…') : label('保存采集配置并生效', 'Save and apply acquisition settings') }}</button>
    </fieldset>
    <p v-if="error" role="alert" class="text-xs text-red-600">{{ error }}</p>
    <button v-if="!loading && !settings" type="button" class="btn btn-secondary text-xs" @click="load">{{ label('重新读取', 'Retry') }}</button>
    <p v-if="saved" role="status" class="text-xs text-green-600">{{ label('采集配置已保存并生效，不需要重启。', 'Acquisition settings saved and applied. No restart required.') }}</p>
  </section>
</template>
