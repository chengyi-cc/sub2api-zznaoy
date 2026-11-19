<template>
  <div>
    <div v-if="loading && !cfg" class="flex justify-center py-12"><LoadingSpinner size="md" /></div>
    <div v-else-if="error" class="card p-6 text-sm text-red-600 dark:text-red-400">{{ error }}</div>
    <div v-else-if="cfg" class="space-y-2" data-test="site-recon-settings">
      <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.settings.sub') }}</p>

      <h3 :class="H2">{{ t('siteRecon.settings.schedule') }}</h3>
      <form class="card space-y-4 p-4" @submit.prevent="saveConfig">
        <label class="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-200">
          <input v-model="cfg.schedule_enabled" type="checkbox" class="mt-0.5" data-test="site-recon-schedule-enabled" />
          <span>
            {{ t('siteRecon.settings.scheduleEnabled') }}
            <span class="block text-xs text-gray-400">{{ t('siteRecon.settings.scheduleHint') }}</span>
          </span>
        </label>
        <div class="flex flex-wrap gap-3">
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.settings.time') }}</span>
            <input v-model="cfg.schedule_time" type="time" class="input mt-1 w-32" required />
          </label>
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.settings.timezone') }}</span>
            <input v-model="cfg.timezone" type="text" class="input mt-1 w-48" placeholder="Asia/Shanghai" />
          </label>
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.settings.days') }}</span>
            <input v-model.number="cfg.days" type="number" min="1" max="90" class="input mt-1 w-24" />
          </label>
        </div>
        <p class="text-xs text-gray-400">{{ t('siteRecon.settings.daysHint') }}</p>
        <label class="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-200">
          <input v-model="cfg.scan" type="checkbox" class="mt-0.5" />
          <span>
            {{ t('siteRecon.settings.scan') }}
            <span class="block text-xs text-gray-400">{{ t('siteRecon.settings.scanHint') }}</span>
          </span>
        </label>
        <label class="block text-sm">
          <span class="input-label">{{ t('siteRecon.settings.exclude') }}</span>
          <input v-model="excludeText" type="text" class="input mt-1 w-full max-w-md" />
          <span class="mt-1 block text-xs text-gray-400">{{ t('siteRecon.settings.excludeHint') }}</span>
        </label>

        <div class="border-t border-gray-100 pt-4 dark:border-dark-700">
          <div class="mb-3 text-sm font-semibold text-gray-800 dark:text-gray-100">{{ t('siteRecon.settings.push') }}</div>
          <label class="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200">
            <input v-model="cfg.push_enabled" type="checkbox" />
            <span>{{ t('siteRecon.settings.pushEnabled') }}</span>
          </label>
          <label class="mt-3 block text-sm">
            <span class="input-label">{{ t('siteRecon.settings.barkUrl') }}</span>
            <input
              v-model="barkInput"
              type="password"
              autocomplete="new-password"
              class="input mt-1 w-full max-w-md"
              :placeholder="t('siteRecon.settings.barkPlaceholder')"
              data-test="site-recon-bark-input"
            />
            <span class="mt-1 block text-xs text-gray-400" data-test="site-recon-bark-hint">
              {{ view?.bark_configured ? t('siteRecon.settings.barkSaved', { hint: view.bark_hint }) : t('siteRecon.settings.barkNone') }}
            </span>
          </label>
          <label v-if="view?.bark_configured" class="mt-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200">
            <input v-model="clearBark" type="checkbox" />
            <span>{{ t('siteRecon.settings.clearBark') }}</span>
          </label>
          <label class="mt-3 block text-sm">
            <span class="input-label">{{ t('siteRecon.settings.balanceDays') }}</span>
            <input v-model.number="cfg.balance_days" type="number" min="0" step="0.5" class="input mt-1 w-24" />
            <span class="mt-1 block text-xs text-gray-400">{{ t('siteRecon.settings.balanceDaysHint') }}</span>
          </label>
        </div>

        <div class="flex flex-wrap items-center gap-3">
          <button type="submit" class="btn btn-primary" :disabled="saving" data-test="site-recon-save-config">
            {{ saving ? t('common.processing') : t('siteRecon.settings.save') }}
          </button>
          <button type="button" class="btn btn-secondary" :disabled="pushing || !view?.bark_configured" @click="testPush">
            {{ t('siteRecon.settings.testPush') }}
          </button>
          <span class="text-xs text-gray-500 dark:text-gray-400">
            {{ view?.next_run ? t('siteRecon.settings.nextRun', { value: view.next_run }) : t('siteRecon.settings.scheduleOff') }}
          </span>
        </div>
      </form>

      <h3 :class="H2">{{ t('siteRecon.settings.tokens') }}</h3>
      <div class="card space-y-4 p-4">
        <p class="text-xs leading-relaxed text-gray-500 dark:text-gray-400">{{ t('siteRecon.settings.tokensHint') }}</p>
        <div v-if="tokens.need_tokens.length" class="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200">
          {{ t('siteRecon.settings.needTokens') }}
          <button
            v-for="h in tokens.need_tokens"
            :key="h"
            type="button"
            class="ml-2 underline"
            @click="tokenForm.host = h"
          >
            {{ h }}（{{ t('siteRecon.settings.fillIt') }}）
          </button>
        </div>

        <div v-if="!tokens.tokens.length" class="text-sm text-gray-400">{{ t('siteRecon.settings.noTokens') }}</div>
        <ul v-else class="divide-y divide-gray-100 dark:divide-dark-700" data-test="site-recon-token-list">
          <li v-for="tk in tokens.tokens" :key="tk.host" class="flex flex-wrap items-center gap-3 py-2 text-sm">
            <span class="font-mono text-gray-800 dark:text-gray-100">{{ tk.host }}</span>
            <span v-if="tk.readable" class="text-xs text-gray-500">{{ t('siteRecon.settings.configured', { hint: tk.hint }) }}</span>
            <span v-else class="text-xs text-red-600 dark:text-red-400">{{ t('siteRecon.settings.unreadable') }}</span>
            <span class="text-xs text-gray-400">{{ t('siteRecon.settings.updatedAt', { value: shortTime(tk.updated_at) }) }}</span>
            <span class="flex-1"></span>
            <button type="button" class="text-xs text-primary-600 hover:underline dark:text-primary-400" @click="tokenForm.host = tk.host">
              {{ t('siteRecon.settings.replace') }}
            </button>
            <template v-if="confirmDelete === tk.host">
              <span class="text-xs text-red-600">{{ t('siteRecon.settings.removeConfirm', { host: tk.host }) }}</span>
              <button type="button" class="btn btn-danger btn-sm" @click="removeToken(tk.host)">{{ t('siteRecon.settings.remove') }}</button>
              <button type="button" class="btn btn-secondary btn-sm" @click="confirmDelete = ''">{{ t('common.cancel') }}</button>
            </template>
            <button v-else type="button" class="text-xs text-red-600 hover:underline" @click="confirmDelete = tk.host">
              {{ t('siteRecon.settings.remove') }}
            </button>
          </li>
        </ul>

        <form class="flex flex-wrap items-end gap-3 border-t border-gray-100 pt-4 dark:border-dark-700" @submit.prevent="saveToken">
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.settings.host') }}</span>
            <input v-model="tokenForm.host" type="text" class="input mt-1 w-64" :placeholder="t('siteRecon.settings.hostPlaceholder')" data-test="site-recon-token-host" />
          </label>
          <label class="text-sm">
            <span class="input-label">{{ t('siteRecon.settings.token') }}</span>
            <input
              v-model="tokenForm.token"
              type="password"
              autocomplete="new-password"
              class="input mt-1 w-72"
              :placeholder="t('siteRecon.settings.tokenPlaceholder')"
              data-test="site-recon-token-input"
            />
          </label>
          <button type="submit" class="btn btn-primary" :disabled="savingToken || !tokenForm.host.trim() || !tokenForm.token.trim()" data-test="site-recon-save-token">
            {{ savingToken ? t('common.processing') : t('siteRecon.settings.saveToken') }}
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { siteReconAPI, type SiteReconConfig, type SiteReconConfigView, type SiteReconTokensView } from '@/api/admin/siteRecon'
import { useAppStore } from '@/stores/app'
import { extractApiErrorMessage } from '@/utils/apiError'
import { H2, shortTime } from './format'

const { t } = useI18n()
const appStore = useAppStore()

const view = ref<SiteReconConfigView | null>(null)
const cfg = ref<SiteReconConfig | null>(null)
const tokens = ref<SiteReconTokensView>({ tokens: [], need_tokens: [] })
const loading = ref(false)
const error = ref('')
const saving = ref(false)
const pushing = ref(false)
const savingToken = ref(false)
const excludeText = ref('')
const barkInput = ref('')
const clearBark = ref(false)
const confirmDelete = ref('')
const tokenForm = reactive({ host: '', token: '' })

function applyView(v: SiteReconConfigView) {
  view.value = v
  cfg.value = { ...v.config, exclude: [...(v.config.exclude ?? [])] }
  excludeText.value = (v.config.exclude ?? []).join(', ')
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [v, tk] = await Promise.all([siteReconAPI.config(), siteReconAPI.tokens()])
    applyView(v)
    tokens.value = tk
  } catch (e) {
    error.value = extractApiErrorMessage(e, t('siteRecon.loadFailed'))
  } finally {
    loading.value = false
  }
}

async function saveConfig() {
  if (!cfg.value) return
  saving.value = true
  try {
    const exclude = excludeText.value
      .replace(/，/g, ',')
      .split(',')
      .map((x) => Number(x.trim()))
      .filter((x) => Number.isInteger(x) && x > 0)
    const payload = { ...cfg.value, exclude, clear_bark: clearBark.value } as Parameters<typeof siteReconAPI.updateConfig>[0]
    if (barkInput.value.trim()) payload.bark_secret_url = barkInput.value.trim()
    applyView(await siteReconAPI.updateConfig(payload))
    barkInput.value = '' // 只写不读：保存后清空
    clearBark.value = false
    appStore.showSuccess(t('siteRecon.settings.saved'))
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  } finally {
    saving.value = false
  }
}

async function testPush() {
  pushing.value = true
  try {
    await siteReconAPI.testPush()
    appStore.showSuccess(t('siteRecon.settings.testPushed'))
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  } finally {
    pushing.value = false
  }
}

async function refreshTokens() {
  tokens.value = await siteReconAPI.tokens()
}

async function saveToken() {
  savingToken.value = true
  try {
    await siteReconAPI.setToken(tokenForm.host.trim(), tokenForm.token.trim())
    tokenForm.token = '' // 只写不读：保存后立刻清空
    tokenForm.host = ''
    appStore.showSuccess(t('siteRecon.settings.tokenSaved'))
    await refreshTokens()
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  } finally {
    savingToken.value = false
  }
}

async function removeToken(host: string) {
  try {
    await siteReconAPI.deleteToken(host)
    confirmDelete.value = ''
    appStore.showSuccess(t('siteRecon.settings.removed'))
    await refreshTokens()
  } catch (e) {
    appStore.showError(extractApiErrorMessage(e, t('common.error')))
  }
}

onMounted(load)
</script>
