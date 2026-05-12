<template>
  <AppLayout>
    <div class="space-y-4">
      <!-- Filters -->
      <div class="card p-4">
        <div class="flex flex-wrap items-center gap-3">
          <div class="flex-1 min-w-[200px]">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
              {{ t('invoice.adminTitle') }}
            </h2>
          </div>
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
          <input
            v-model="keyword"
            class="input h-9 w-56 text-sm"
            :placeholder="t('invoice.adminSearchPlaceholder')"
            @keyup.enter="reload"
          />
          <button class="btn btn-secondary" @click="reload" :disabled="loading">
            {{ t('common.refresh') }}
          </button>
        </div>
      </div>

      <!-- Table -->
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
        <table v-else class="min-w-full divide-y divide-gray-200 text-sm dark:divide-dark-700">
          <thead class="bg-gray-50 dark:bg-dark-800/40">
            <tr class="text-left text-xs uppercase text-gray-500 dark:text-gray-400">
              <th class="px-4 py-2">ID</th>
              <th class="px-4 py-2">{{ t('invoice.adminUserId') }}</th>
              <th class="px-4 py-2">{{ t('invoice.titleField') }}</th>
              <th class="px-4 py-2">{{ t('invoice.invoiceType') }}</th>
              <th class="px-4 py-2">{{ t('invoice.amount') }}</th>
              <th class="px-4 py-2">{{ t('invoice.status.label') }}</th>
              <th class="px-4 py-2">{{ t('invoice.createdAt') }}</th>
              <th class="px-4 py-2 text-right">{{ t('invoice.actions') }}</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 dark:divide-dark-700">
            <tr
              v-for="item in items"
              :key="item.id"
              class="hover:bg-gray-50 dark:hover:bg-dark-800/50"
            >
              <td class="px-4 py-2 text-gray-500 dark:text-gray-400">#{{ item.id }}</td>
              <td class="px-4 py-2">{{ item.user_id }}</td>
              <td class="px-4 py-2">
                <div class="font-medium text-gray-900 dark:text-white">{{ item.title }}</div>
                <div v-if="item.tax_no" class="text-xs text-gray-500 dark:text-gray-400">
                  {{ t('invoice.taxNo') }}: {{ item.tax_no }}
                </div>
                <div v-if="item.invoice_no" class="text-xs text-emerald-600 dark:text-emerald-400">
                  {{ t('invoice.invoiceNo') }}: {{ item.invoice_no }}
                </div>
                <div v-if="item.reject_reason" class="text-xs text-red-600 dark:text-red-400">
                  {{ t('invoice.rejectReason') }}: {{ item.reject_reason }}
                </div>
              </td>
              <td class="px-4 py-2">{{ t(`invoice.type.${item.invoice_type}`) }}</td>
              <td class="px-4 py-2 font-medium text-gray-900 dark:text-white">
                ¥{{ item.amount.toFixed(2) }}
              </td>
              <td class="px-4 py-2">
                <span
                  class="rounded px-2 py-0.5 text-xs font-medium"
                  :class="statusBadgeClass(item.status)"
                >
                  {{ t(`invoice.status.${item.status}`) }}
                </span>
              </td>
              <td class="px-4 py-2 text-xs text-gray-500 dark:text-gray-400">
                {{ formatDateTime(item.created_at) }}
              </td>
              <td class="px-4 py-2">
                <div class="flex items-center justify-end gap-1">
                  <button
                    v-if="item.status === 'pending'"
                    class="btn btn-primary btn-sm"
                    @click="approve(item)"
                  >
                    {{ t('invoice.approve') }}
                  </button>
                  <button
                    v-if="item.status === 'pending'"
                    class="btn btn-secondary btn-sm"
                    @click="openReject(item)"
                  >
                    {{ t('invoice.reject') }}
                  </button>
                  <button
                    v-if="item.status === 'approved'"
                    class="btn btn-primary btn-sm"
                    @click="openIssue(item)"
                  >
                    {{ t('invoice.issue') }}
                  </button>
                  <button
                    v-if="item.status === 'issued' && item.has_file"
                    class="btn btn-secondary btn-sm"
                    @click="download(item)"
                    :disabled="downloadingId === item.id"
                  >
                    {{ downloadingId === item.id ? t('common.processing') : t('invoice.download') }}
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
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

    <!-- Reject Dialog -->
    <BaseDialog
      :show="!!rejectTarget"
      :title="t('invoice.rejectTitle')"
      width="narrow"
      @close="rejectTarget = null"
    >
      <div class="space-y-3">
        <p class="text-sm text-gray-600 dark:text-gray-300">
          {{ t('invoice.rejectConfirm', { id: rejectTarget?.id || '' }) }}
        </p>
        <div>
          <label class="input-label">{{ t('invoice.rejectReason') }}</label>
          <textarea
            v-model="rejectReason"
            rows="3"
            class="input mt-1 w-full"
            :placeholder="t('invoice.rejectReasonPlaceholder')"
          />
        </div>
        <div v-if="rejectError" class="text-sm text-red-600 dark:text-red-400">
          {{ rejectError }}
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-3">
          <button class="btn btn-secondary" @click="rejectTarget = null">
            {{ t('common.cancel') }}
          </button>
          <button
            class="btn btn-danger"
            :disabled="actionLoading || !rejectReason.trim()"
            @click="confirmReject"
          >
            {{ actionLoading ? t('common.processing') : t('invoice.reject') }}
          </button>
        </div>
      </template>
    </BaseDialog>

    <!-- Issue Dialog -->
    <BaseDialog
      :show="!!issueTarget"
      :title="t('invoice.issueTitle')"
      @close="issueTarget = null"
    >
      <div class="space-y-3">
        <p v-if="issueTarget" class="text-sm text-gray-600 dark:text-gray-300">
          {{ t('invoice.issueTargetInfo', { id: issueTarget.id, amount: issueTarget.amount.toFixed(2) }) }}
        </p>
        <div>
          <label class="input-label">{{ t('invoice.invoiceNo') }}</label>
          <input
            v-model="issueForm.invoiceNo"
            type="text"
            class="input mt-1 w-full"
            :placeholder="t('invoice.invoiceNoPlaceholder')"
            maxlength="100"
          />
        </div>
        <div>
          <label class="input-label">{{ t('invoice.uploadPdf') }}</label>
          <input
            type="file"
            accept="application/pdf,.pdf"
            class="block w-full text-sm"
            @change="onIssueFileChange"
          />
          <p class="input-hint">{{ t('invoice.uploadPdfHint') }}</p>
          <p v-if="issueForm.file" class="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {{ issueForm.file.name }} ({{ (issueForm.file.size / 1024).toFixed(1) }} KB)
          </p>
        </div>
        <div v-if="issueError" class="text-sm text-red-600 dark:text-red-400">
          {{ issueError }}
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-3">
          <button class="btn btn-secondary" @click="issueTarget = null">
            {{ t('common.cancel') }}
          </button>
          <button
            class="btn btn-primary"
            :disabled="actionLoading || !canIssue"
            @click="confirmIssue"
          >
            {{ actionLoading ? t('common.processing') : t('invoice.issue') }}
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
import { adminInvoiceAPI } from '@/api/admin/invoices'
import type { InvoiceRequest, InvoiceStatus } from '@/api/invoice'

const { t } = useI18n()

const items = ref<InvoiceRequest[]>([])
const loading = ref(false)
const statusFilter = ref<InvoiceStatus | ''>('')
const keyword = ref('')
const pagination = reactive({ total: 0, page: 1, page_size: 20, pages: 1 })
const downloadingId = ref<number | null>(null)
const actionLoading = ref(false)

// Reject state
const rejectTarget = ref<InvoiceRequest | null>(null)
const rejectReason = ref('')
const rejectError = ref('')

// Issue state
const issueTarget = ref<InvoiceRequest | null>(null)
const issueForm = reactive<{ invoiceNo: string; file: File | null }>({
  invoiceNo: '',
  file: null
})
const issueError = ref('')

const canIssue = computed(
  () => !!issueForm.file && issueForm.invoiceNo.trim().length > 0
)

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

async function reload() {
  loading.value = true
  try {
    const resp = await adminInvoiceAPI.list({
      status: statusFilter.value || undefined,
      q: keyword.value.trim() || undefined,
      page: pagination.page,
      page_size: pagination.page_size
    })
    items.value = resp.items
    pagination.total = resp.total
    pagination.pages = resp.pages
  } catch (e: any) {
    items.value = []
    console.error('admin invoice list failed', e)
  } finally {
    loading.value = false
  }
}

function changePage(n: number) {
  pagination.page = n
  reload()
}

async function approve(item: InvoiceRequest) {
  actionLoading.value = true
  try {
    await adminInvoiceAPI.approve(item.id)
    await reload()
  } catch (e: any) {
    console.error('approve failed', e)
  } finally {
    actionLoading.value = false
  }
}

function openReject(item: InvoiceRequest) {
  rejectTarget.value = item
  rejectReason.value = ''
  rejectError.value = ''
}

async function confirmReject() {
  if (!rejectTarget.value) return
  actionLoading.value = true
  rejectError.value = ''
  try {
    await adminInvoiceAPI.reject(rejectTarget.value.id, rejectReason.value.trim())
    rejectTarget.value = null
    await reload()
  } catch (e: any) {
    rejectError.value = e?.message || String(e)
  } finally {
    actionLoading.value = false
  }
}

function openIssue(item: InvoiceRequest) {
  issueTarget.value = item
  issueForm.invoiceNo = ''
  issueForm.file = null
  issueError.value = ''
}

function onIssueFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  issueForm.file = input.files && input.files.length > 0 ? input.files[0] : null
}

async function confirmIssue() {
  if (!issueTarget.value || !issueForm.file) return
  actionLoading.value = true
  issueError.value = ''
  try {
    await adminInvoiceAPI.issue(
      issueTarget.value.id,
      issueForm.invoiceNo.trim(),
      issueForm.file
    )
    issueTarget.value = null
    await reload()
  } catch (e: any) {
    issueError.value = e?.message || String(e)
  } finally {
    actionLoading.value = false
  }
}

async function download(item: InvoiceRequest) {
  downloadingId.value = item.id
  try {
    const filename = item.invoice_no ? `invoice-${item.invoice_no}.pdf` : `invoice-${item.id}.pdf`
    await adminInvoiceAPI.download(item.id, filename)
  } catch (e: any) {
    console.error('admin download failed', e)
  } finally {
    downloadingId.value = null
  }
}

onMounted(reload)
</script>