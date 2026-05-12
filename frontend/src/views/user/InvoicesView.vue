<template>
  <AppLayout>
    <div class="space-y-4">
      <!-- Header card -->
      <div class="card p-4">
        <div class="flex flex-wrap items-center gap-3">
          <div class="flex-1 min-w-[200px]">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
              {{ t('invoice.title') }}
            </h2>
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {{ t('invoice.description') }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <select
              v-model="statusFilter"
              class="input h-9 w-36 text-sm"
              @change="reload"
            >
              <option value="">{{ t('invoice.allStatus') }}</option>
              <option value="pending">{{ t('invoice.status.pending') }}</option>
              <option value="approved">{{ t('invoice.status.approved') }}</option>
              <option value="rejected">{{ t('invoice.status.rejected') }}</option>
              <option value="issued">{{ t('invoice.status.issued') }}</option>
            </select>
            <button class="btn btn-secondary" @click="reload" :disabled="loading">
              {{ t('common.refresh') }}
            </button>
            <button class="btn btn-primary" @click="openCreateModal">
              {{ t('invoice.newRequest') }}
            </button>
          </div>
        </div>
      </div>

      <!-- List -->
      <div class="card overflow-hidden">
        <div v-if="loading" class="p-8 text-center text-gray-500 dark:text-gray-400">
          {{ t('common.loading') }}
        </div>
        <div
          v-else-if="items.length === 0"
          class="p-12 text-center text-gray-500 dark:text-gray-400"
        >
          {{ t('invoice.empty') }}
        </div>
        <div v-else class="divide-y divide-gray-100 dark:divide-dark-700">
          <div
            v-for="item in items"
            :key="item.id"
            class="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-dark-800/50"
          >
            <div class="flex-1 min-w-[220px]">
              <div class="flex items-center gap-2">
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ item.title }}
                </span>
                <span
                  class="rounded px-2 py-0.5 text-xs font-medium"
                  :class="invoiceTypeBadgeClass(item.invoice_type)"
                >
                  {{ t(`invoice.type.${item.invoice_type}`) }}
                </span>
                <span
                  class="rounded px-2 py-0.5 text-xs font-medium"
                  :class="statusBadgeClass(item.status)"
                >
                  {{ t(`invoice.status.${item.status}`) }}
                </span>
              </div>
              <div class="mt-1 text-xs text-gray-500 dark:text-gray-400">
                <span>#{{ item.id }}</span>
                <span class="mx-2">·</span>
                <span>{{ t('invoice.amount') }}: ¥{{ item.amount.toFixed(2) }}</span>
                <span class="mx-2">·</span>
                <span>{{ item.payment_order_ids.length }} {{ t('invoice.orders') }}</span>
                <span class="mx-2">·</span>
                <span>{{ formatDateTime(item.created_at) }}</span>
              </div>
              <div
                v-if="item.status === 'rejected' && item.reject_reason"
                class="mt-1 text-xs text-red-600 dark:text-red-400"
              >
                {{ t('invoice.rejectReason') }}: {{ item.reject_reason }}
              </div>
              <div
                v-if="item.status === 'issued' && item.invoice_no"
                class="mt-1 text-xs text-emerald-600 dark:text-emerald-400"
              >
                {{ t('invoice.invoiceNo') }}: {{ item.invoice_no }}
              </div>
            </div>
            <div class="flex items-center gap-2">
              <button
                v-if="item.status === 'issued' && item.has_file"
                class="btn btn-primary btn-sm"
                @click="handleDownload(item)"
                :disabled="downloadingId === item.id"
              >
                {{ downloadingId === item.id ? t('common.processing') : t('invoice.download') }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="pagination.total > pagination.page_size" class="flex items-center justify-end gap-2">
        <button
          class="btn btn-secondary btn-sm"
          :disabled="pagination.page <= 1"
          @click="changePage(pagination.page - 1)"
        >
          {{ t('common.previous') }}
        </button>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ pagination.page }} / {{ pagination.pages }}
        </span>
        <button
          class="btn btn-secondary btn-sm"
          :disabled="pagination.page >= pagination.pages"
          @click="changePage(pagination.page + 1)"
        >
          {{ t('common.next') }}
        </button>
      </div>
    </div>

    <!-- Create Modal -->
    <BaseDialog :show="showCreate" :title="t('invoice.newRequest')" @close="closeCreateModal">
      <div class="space-y-4">
        <!-- Eligible orders -->
        <div>
          <label class="input-label">{{ t('invoice.selectOrders') }}</label>
          <p class="input-hint">{{ t('invoice.selectOrdersHint') }}</p>
          <div
            v-if="loadingEligible"
            class="mt-2 rounded-md border border-gray-200 p-4 text-center text-sm text-gray-500 dark:border-dark-700 dark:text-gray-400"
          >
            {{ t('common.loading') }}
          </div>
          <div
            v-else-if="eligibleOrders.length === 0"
            class="mt-2 rounded-md border border-gray-200 p-4 text-center text-sm text-gray-500 dark:border-dark-700 dark:text-gray-400"
          >
            {{ t('invoice.noEligible') }}
          </div>
          <div
            v-else
            class="mt-2 max-h-56 overflow-y-auto rounded-md border border-gray-200 dark:border-dark-700"
          >
            <label
              v-for="o in eligibleOrders"
              :key="o.order_id"
              class="flex cursor-pointer items-center gap-3 border-b border-gray-100 px-3 py-2 last:border-b-0 hover:bg-gray-50 dark:border-dark-700 dark:hover:bg-dark-800"
            >
              <input
                type="checkbox"
                :value="o.order_id"
                v-model="form.payment_order_ids"
                class="h-4 w-4"
              />
              <div class="flex-1 text-sm">
                <div class="font-medium text-gray-900 dark:text-white">
                  ¥{{ o.amount.toFixed(2) }}
                  <span class="ml-2 text-xs text-gray-500 dark:text-gray-400">
                    #{{ o.order_id }} · {{ o.out_trade_no }}
                  </span>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400">
                  {{ formatDateTime(o.completed_at) }}
                </div>
              </div>
            </label>
          </div>
          <div v-if="form.payment_order_ids.length > 0" class="mt-2 text-sm text-gray-600 dark:text-gray-300">
            {{ t('invoice.totalAmount') }}:
            <span class="font-semibold">¥{{ totalAmount.toFixed(2) }}</span>
          </div>
        </div>

        <!-- Type -->
        <div>
          <label class="input-label">{{ t('invoice.invoiceType') }}</label>
          <div class="mt-2 flex gap-4">
            <label class="flex items-center gap-2 text-sm">
              <input type="radio" v-model="form.invoice_type" value="personal" />
              {{ t('invoice.type.personal') }}
            </label>
            <label class="flex items-center gap-2 text-sm">
              <input type="radio" v-model="form.invoice_type" value="company" />
              {{ t('invoice.type.company') }}
            </label>
          </div>
        </div>

        <!-- Title -->
        <div>
          <label class="input-label">{{ t('invoice.titleField') }}</label>
          <input
            v-model="form.title"
            type="text"
            class="input mt-1 w-full"
            :placeholder="t('invoice.titlePlaceholder')"
            maxlength="200"
          />
        </div>

        <!-- Tax No (company only) -->
        <div v-if="form.invoice_type === 'company'">
          <label class="input-label">{{ t('invoice.taxNo') }}</label>
          <input
            v-model="form.tax_no"
            type="text"
            class="input mt-1 w-full"
            :placeholder="t('invoice.taxNoPlaceholder')"
            maxlength="50"
          />
        </div>

        <!-- Recipient Email -->
        <div>
          <label class="input-label">{{ t('invoice.recipientEmail') }}</label>
          <input
            v-model="form.recipient_email"
            type="email"
            class="input mt-1 w-full"
            :placeholder="t('invoice.recipientEmailPlaceholder')"
          />
          <p class="input-hint">{{ t('invoice.recipientEmailHint') }}</p>
        </div>

        <!-- Remark -->
        <div>
          <label class="input-label">{{ t('invoice.remark') }}</label>
          <textarea
            v-model="form.remark"
            rows="2"
            class="input mt-1 w-full"
            :placeholder="t('invoice.remarkPlaceholder')"
            maxlength="1000"
          />
        </div>

        <div v-if="submitError" class="text-sm text-red-600 dark:text-red-400">
          {{ submitError }}
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-3">
          <button class="btn btn-secondary" @click="closeCreateModal">
            {{ t('common.cancel') }}
          </button>
          <button
            class="btn btn-primary"
            :disabled="submitting || !canSubmit"
            @click="handleSubmit"
          >
            {{ submitting ? t('common.processing') : t('invoice.submitRequest') }}
          </button>
        </div>
      </template>
    </BaseDialog>
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import AppLayout from '@/components/layout/AppLayout.vue'
import BaseDialog from '@/components/common/BaseDialog.vue'
import { formatDateTime } from '@/utils/format'
import {
  invoiceAPI,
  type InvoiceRequest,
  type InvoiceStatus,
  type EligibleOrder
} from '@/api/invoice'

const { t } = useI18n()

const items = ref<InvoiceRequest[]>([])
const loading = ref(false)
const statusFilter = ref<InvoiceStatus | ''>('')
const pagination = reactive({ total: 0, page: 1, page_size: 20, pages: 1 })
const downloadingId = ref<number | null>(null)

const showCreate = ref(false)
const submitting = ref(false)
const submitError = ref('')
const eligibleOrders = ref<EligibleOrder[]>([])
const loadingEligible = ref(false)
const form = reactive({
  payment_order_ids: [] as number[],
  invoice_type: 'personal' as 'personal' | 'company',
  title: '',
  tax_no: '',
  recipient_email: '',
  remark: ''
})

const totalAmount = computed(() => {
  const ids = new Set(form.payment_order_ids)
  return eligibleOrders.value
    .filter(o => ids.has(o.order_id))
    .reduce((sum, o) => sum + o.amount, 0)
})

const canSubmit = computed(() => {
  if (form.payment_order_ids.length === 0) return false
  if (!form.title.trim()) return false
  if (form.invoice_type === 'company' && !form.tax_no.trim()) return false
  return true
})

function statusBadgeClass(status: string): string {
  switch (status) {
    case 'pending':
      return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300'
    case 'approved':
      return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
    case 'rejected':
      return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
    case 'issued':
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
    default:
      return 'bg-gray-100 text-gray-700 dark:bg-dark-700 dark:text-gray-300'
  }
}

function invoiceTypeBadgeClass(type: string): string {
  return type === 'company'
    ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
    : 'bg-gray-100 text-gray-700 dark:bg-dark-700 dark:text-gray-300'
}

async function reload() {
  loading.value = true
  try {
    const resp = await invoiceAPI.listInvoiceRequests({
      status: statusFilter.value || undefined,
      page: pagination.page,
      page_size: pagination.page_size
    })
    items.value = resp.items
    pagination.total = resp.total
    pagination.pages = resp.pages
  } catch (e: any) {
    items.value = []
    console.error('load invoice requests failed', e)
  } finally {
    loading.value = false
  }
}

function changePage(n: number) {
  pagination.page = n
  reload()
}

async function openCreateModal() {
  showCreate.value = true
  submitError.value = ''
  form.payment_order_ids = []
  form.invoice_type = 'personal'
  form.title = ''
  form.tax_no = ''
  form.recipient_email = ''
  form.remark = ''
  loadingEligible.value = true
  try {
    eligibleOrders.value = await invoiceAPI.listEligibleOrders()
  } catch (e: any) {
    eligibleOrders.value = []
    submitError.value = e?.message || String(e)
  } finally {
    loadingEligible.value = false
  }
}

function closeCreateModal() {
  showCreate.value = false
}

async function handleSubmit() {
  submitting.value = true
  submitError.value = ''
  try {
    await invoiceAPI.createInvoiceRequest({
      payment_order_ids: form.payment_order_ids,
      invoice_type: form.invoice_type,
      title: form.title.trim(),
      tax_no: form.invoice_type === 'company' ? form.tax_no.trim() : undefined,
      recipient_email: form.recipient_email.trim() || undefined,
      remark: form.remark.trim() || undefined
    })
    closeCreateModal()
    pagination.page = 1
    await reload()
  } catch (e: any) {
    submitError.value = e?.message || String(e)
  } finally {
    submitting.value = false
  }
}

async function handleDownload(item: InvoiceRequest) {
  downloadingId.value = item.id
  try {
    const filename = item.invoice_no ? `invoice-${item.invoice_no}.pdf` : `invoice-${item.id}.pdf`
    await invoiceAPI.downloadInvoice(item.id, filename)
  } catch (e: any) {
    console.error('download failed', e)
  } finally {
    downloadingId.value = null
  }
}

onMounted(reload)
</script>