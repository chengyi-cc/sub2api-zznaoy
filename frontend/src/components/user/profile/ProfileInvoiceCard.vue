<template>
  <div class="card">
    <div class="border-b border-gray-100 px-6 py-4 dark:border-dark-700">
      <h2 class="text-lg font-medium text-gray-900 dark:text-white">
        {{ t('invoice.profile.cardTitle') }}
      </h2>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        {{ t('invoice.profile.cardDescription') }}
      </p>
    </div>
    <div class="px-6 py-6">
      <div v-if="loading" class="flex items-center justify-center py-6">
        <div class="h-8 w-8 animate-spin rounded-full border-b-2 border-primary-500"></div>
      </div>

      <div
        v-else-if="loadError"
        class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300"
      >
        {{ loadError }}
      </div>

      <!-- 已绑定：只读 -->
      <div v-else-if="profile" data-testid="invoice-profile-locked" class="space-y-3">
        <div class="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.profile.companyName') }}</div>
            <div class="mt-0.5 break-all font-medium text-gray-900 dark:text-white">{{ profile.title }}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.profile.taxNo') }}</div>
            <div class="mt-0.5 font-mono text-gray-900 dark:text-white">{{ profile.tax_no }}</div>
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span class="rounded bg-gray-100 px-2 py-0.5 font-medium text-gray-700 dark:bg-dark-700 dark:text-gray-300">
            {{ t('invoice.profile.locked') }}
          </span>
          <span v-if="profile.admin_modified">{{ t('invoice.profile.adminModified') }}</span>
          <span>{{ t('invoice.profile.lockedHint') }}</span>
        </div>
      </div>

      <!-- 未绑定：一次性绑定表单 -->
      <form v-else class="space-y-4" @submit.prevent="openConfirm">
        <div>
          <label class="input-label">{{ t('invoice.profile.companyName') }}</label>
          <input
            v-model="form.title"
            type="text"
            class="input mt-1 w-full"
            :placeholder="t('invoice.profile.companyNamePlaceholder')"
            maxlength="66"
          />
        </div>
        <div>
          <label class="input-label">{{ t('invoice.profile.taxNo') }}</label>
          <input
            v-model="form.tax_no"
            type="text"
            class="input mt-1 w-full font-mono uppercase"
            :placeholder="t('invoice.profile.taxNoPlaceholder')"
            maxlength="18"
          />
          <p v-if="taxNoInvalid" class="mt-1 text-xs text-red-600 dark:text-red-400">
            {{ t('invoice.profile.taxNoInvalid') }}
          </p>
        </div>
        <p class="text-xs text-amber-600 dark:text-amber-400">
          {{ t('invoice.profile.bindWarning') }}
        </p>
        <div class="flex justify-end">
          <button type="submit" class="btn btn-primary" :disabled="!canBind || binding">
            {{ binding ? t('common.processing') : t('invoice.profile.bind') }}
          </button>
        </div>
      </form>
    </div>

    <ConfirmDialog
      :show="showConfirm"
      :title="t('invoice.profile.confirmTitle')"
      :message="t('invoice.profile.confirmMessage', { title: form.title.trim(), taxNo: normalizedTaxNo })"
      :confirm-text="t('invoice.profile.bind')"
      @confirm="handleBind"
      @cancel="showConfirm = false"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'
import { invoiceAPI, type InvoiceProfile } from '@/api/invoice'
import { useAppStore } from '@/stores/app'
import { extractI18nErrorMessage } from '@/utils/apiError'
import {
  isValidUnifiedSocialCreditCode,
  normalizeUnifiedSocialCreditCode
} from '@/utils/unifiedSocialCreditCode'

const { t } = useI18n()
const appStore = useAppStore()

const loading = ref(true)
const loadError = ref('')
const binding = ref(false)
const showConfirm = ref(false)
const profile = ref<InvoiceProfile | null>(null)
const form = reactive({ title: '', tax_no: '' })

const normalizedTaxNo = computed(() => normalizeUnifiedSocialCreditCode(form.tax_no))
const taxNoInvalid = computed(
  () => normalizedTaxNo.value.length > 0 && !isValidUnifiedSocialCreditCode(normalizedTaxNo.value)
)
const canBind = computed(
  () => form.title.trim().length > 0 && isValidUnifiedSocialCreditCode(normalizedTaxNo.value)
)

function openConfirm() {
  if (!canBind.value) return
  showConfirm.value = true
}

async function handleBind() {
  showConfirm.value = false
  binding.value = true
  try {
    profile.value = await invoiceAPI.bindInvoiceProfile({
      title: form.title.trim(),
      tax_no: normalizedTaxNo.value
    })
    appStore.showSuccess(t('invoice.profile.bindSuccess'))
  } catch (e) {
    appStore.showError(extractI18nErrorMessage(e, t, 'invoice.errors', t('common.error')))
    // 已被锁定（例如另一个页面先绑定了）时刷新为只读视图
    await load()
  } finally {
    binding.value = false
  }
}

async function load() {
  loading.value = true
  loadError.value = ''
  try {
    profile.value = await invoiceAPI.getInvoiceProfile()
  } catch (e) {
    loadError.value = extractI18nErrorMessage(e, t, 'invoice.errors', t('common.error'))
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>
