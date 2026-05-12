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
              <td class="px-4 py-2 text-right">
                <button class="btn btn-secondary btn-sm" @click="openDetail(item)">
                  {{ t('invoice.viewDetail') }}
                </button>
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

    <!-- Detail Dialog -->
    <BaseDialog
      :show="!!detailTarget"
      :title="t('invoice.detailTitle')"
      width="wide"
      @close="closeDetail"
    >
      <div v-if="detailLoading" class="py-8 text-center text-gray-500 dark:text-gray-400">
        {{ t('common.loading') }}
      </div>
      <div v-else-if="detailError" class="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
        {{ detailError }}
      </div>
      <div v-else-if="detail" class="space-y-5">
        <!-- Meta -->
        <div class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.colId') }}</div>
            <div class="font-medium text-gray-900 dark:text-white">#{{ detail.request.id }}</div>
          </div>
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.detailApplicant') }}</div>
            <div class="text-gray-900 dark:text-white">
              <span v-if="detail.user_email">{{ detail.user_email }}</span>
              <span v-else class="text-gray-400">user #{{ detail.request.user_id }}</span>
              <span v-if="detail.user_name" class="ml-1 text-xs text-gray-500">({{ detail.user_name }})</span>
            </div>
          </div>
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.status.label') }}</div>
            <span
              class="rounded px-2 py-0.5 text-xs font-medium"
              :class="statusBadgeClass(detail.request.status)"
            >
              {{ t(`invoice.status.${detail.request.status}`) }}
            </span>
          </div>
          <div>
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.invoiceType') }}</div>
            <div class="text-gray-900 dark:text-white">{{ t(`invoice.type.${detail.request.invoice_type}`) }}</div>
          </div>
          <div class="col-span-2 sm:col-span-1">
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.createdAt') }}</div>
            <div class="text-gray-700 dark:text-gray-300">{{ formatDateTime(detail.request.created_at) }}</div>
          </div>
          <div v-if="detail.request.recipient_email" class="col-span-2 sm:col-span-3">
            <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.recipientEmail') }}</div>
            <div class="text-gray-700 dark:text-gray-300">{{ detail.request.recipient_email }}</div>
          </div>
        </div>

        <!-- Invoice header info -->
        <div class="rounded-md border border-gray-200 p-4 dark:border-dark-700">
          <div class="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            {{ t('invoice.detailHeaderSection') }}
          </div>
          <div class="mt-2 grid grid-cols-1 gap-y-2 text-sm sm:grid-cols-2">
            <div>
              <span class="text-gray-500 dark:text-gray-400">{{ t('invoice.titleField') }}: </span>
              <span class="font-medium text-gray-900 dark:text-white">{{ detail.request.title }}</span>
            </div>
            <div v-if="detail.request.tax_no">
              <span class="text-gray-500 dark:text-gray-400">{{ t('invoice.taxNo') }}: </span>
              <span class="text-gray-900 dark:text-white">{{ detail.request.tax_no }}</span>
            </div>
            <div v-if="detail.request.invoice_no" class="sm:col-span-2">
              <span class="text-gray-500 dark:text-gray-400">{{ t('invoice.invoiceNo') }}: </span>
              <span class="font-medium text-emerald-600 dark:text-emerald-400">{{ detail.request.invoice_no }}</span>
            </div>
            <div v-if="detail.request.remark" class="sm:col-span-2">
              <span class="text-gray-500 dark:text-gray-400">{{ t('invoice.remark') }}: </span>
              <span class="text-gray-700 dark:text-gray-300">{{ detail.request.remark }}</span>
            </div>
            <div v-if="detail.request.reject_reason" class="sm:col-span-2">
              <span class="text-gray-500 dark:text-gray-400">{{ t('invoice.rejectReason') }}: </span>
              <span class="text-red-600 dark:text-red-400">{{ detail.request.reject_reason }}</span>
            </div>
          </div>
        </div>

        <!-- User reconciliation overview -->
        <div
          class="rounded-md border p-4"
          :class="detail.user_overview.exceeds_total_recharge
            ? 'border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-900/10'
            : 'border-gray-200 dark:border-dark-700'"
        >
          <div class="mb-2 flex items-center justify-between">
            <div class="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              {{ t('invoice.detailUserOverview') }}
            </div>
            <div
              v-if="detail.user_overview.exceeds_total_recharge"
              class="text-xs font-semibold text-red-600 dark:text-red-400"
            >
              ⚠ {{ t('invoice.detailExceedsRecharge') }}
            </div>
          </div>
          <div class="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
            <div>
              <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.userTotalRecharged') }}</div>
              <div class="mt-0.5 text-base font-semibold text-gray-900 dark:text-white">
                ¥{{ detail.user_overview.total_recharged.toFixed(2) }}
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.userBalance') }}</div>
              <div class="mt-0.5 text-base font-semibold text-gray-900 dark:text-white">
                ¥{{ detail.user_overview.balance.toFixed(2) }}
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.userIssuedTotal') }}</div>
              <div class="mt-0.5 text-base font-semibold text-emerald-600 dark:text-emerald-400">
                ¥{{ detail.user_overview.issued_total.toFixed(2) }}
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.userInFlightTotal') }}</div>
              <div class="mt-0.5 text-base font-semibold text-yellow-600 dark:text-yellow-400">
                ¥{{ detail.user_overview.in_flight_total.toFixed(2) }}
              </div>
            </div>
          </div>
          <div
            v-if="detail.user_overview.total_recharged > 0"
            class="mt-3 border-t border-gray-200 pt-3 text-xs dark:border-dark-700"
            :class="detail.user_overview.exceeds_total_recharge ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'"
          >
            {{ t('invoice.userIssuedPlusInFlight') }}: ¥{{ detail.user_overview.issued_plus_in_flight.toFixed(2) }}
            /
            ¥{{ detail.user_overview.total_recharged.toFixed(2) }}
            ({{ ((detail.user_overview.issued_plus_in_flight / detail.user_overview.total_recharged) * 100).toFixed(1) }}%)
          </div>
        </div>

        <!-- Amount reconciliation -->
        <div
          class="rounded-md border p-4"
          :class="detail.amount_match
            ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-900/10'
            : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/10'"
        >
          <div class="flex items-center justify-between text-sm">
            <div>
              <div class="font-medium text-gray-900 dark:text-white">
                {{ t('invoice.detailAmountTitle') }}
              </div>
              <div class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
                {{ t('invoice.detailRequestAmount') }}: ¥{{ detail.request.amount.toFixed(2) }}
                ·
                {{ t('invoice.detailComputedSum') }}: ¥{{ detail.computed_sum.toFixed(2) }}
              </div>
            </div>
            <div
              class="text-sm font-medium"
              :class="detail.amount_match ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'"
            >
              {{ detail.amount_match ? t('invoice.detailAmountMatch') : t('invoice.detailAmountMismatch') }}
            </div>
          </div>
        </div>

        <!-- Orders -->
        <div>
          <div class="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            {{ t('invoice.sourceOrders') }}
            <span v-if="detail.orders.length > 0" class="ml-1 text-gray-400">({{ detail.orders.length }})</span>
          </div>
          <div v-if="detail.orders.length === 0" class="rounded-md border border-gray-200 p-3 text-center text-xs text-gray-500 dark:border-dark-700 dark:text-gray-400">
            —
          </div>
          <div v-else class="overflow-hidden rounded-md border border-gray-200 dark:border-dark-700">
            <div
              v-for="o in detail.orders"
              :key="o.order_id"
              class="flex items-center gap-3 border-b border-gray-100 px-4 py-3 last:border-b-0 dark:border-dark-700"
            >
              <div class="min-w-0 flex-1">
                <div class="truncate text-sm font-medium text-gray-900 dark:text-white">
                  #{{ o.order_id }} · {{ o.out_trade_no || '—' }}
                </div>
                <div class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
                  {{ o.completed_at ? formatDateTime(o.completed_at) : '—' }}
                </div>
              </div>
              <div class="flex flex-shrink-0 items-center gap-3">
                <span
                  class="rounded px-2 py-0.5 text-xs font-medium"
                  :class="o.eligible
                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'"
                >
                  {{ o.status }}
                </span>
                <span class="text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                  ¥{{ o.amount.toFixed(2) }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Redeem codes -->
        <div v-if="detail.redeem_codes.length > 0">
          <div class="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
            {{ t('invoice.sourceRedeemCodes') }}
            <span class="ml-1 text-gray-400">({{ detail.redeem_codes.length }})</span>
          </div>
          <div class="overflow-hidden rounded-md border border-gray-200 dark:border-dark-700">
            <div
              v-for="c in detail.redeem_codes"
              :key="c.redeem_code_id"
              class="flex items-center gap-3 border-b border-gray-100 px-4 py-3 last:border-b-0 dark:border-dark-700"
            >
              <div class="min-w-0 flex-1">
                <div class="truncate font-mono text-sm font-medium text-gray-900 dark:text-white">
                  {{ c.code }}
                </div>
                <div class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
                  {{ c.used_at ? formatDateTime(c.used_at) : '—' }}
                  <span v-if="!c.eligible" class="ml-2 text-red-600 dark:text-red-400">
                    · {{ c.type }} / {{ c.status }}
                    <span v-if="c.used_by !== detail.request.user_id">· {{ t('invoice.detailOwnershipMismatch') }}</span>
                  </span>
                </div>
              </div>
              <div class="flex flex-shrink-0 items-center gap-3">
                <span
                  class="rounded px-2 py-0.5 text-xs font-medium"
                  :class="c.eligible
                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'"
                >
                  {{ c.status }}
                </span>
                <span class="text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                  ¥{{ c.value.toFixed(2) }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Overall warning -->
        <div
          v-if="!detail.all_eligible"
          class="rounded-md border border-yellow-200 bg-yellow-50 p-3 text-xs text-yellow-700 dark:border-yellow-800 dark:bg-yellow-900/10 dark:text-yellow-300"
        >
          {{ t('invoice.detailEligibilityWarning') }}
        </div>

        <!-- Issue form (only when approved + issuing) -->
        <div
          v-if="detail.request.status === 'approved' && showIssueForm"
          class="space-y-3 rounded-md border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/10"
        >
          <div class="text-sm font-medium text-gray-900 dark:text-white">
            {{ t('invoice.issueTitle') }}
          </div>
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

        <!-- Reject form (only when pending + rejecting) -->
        <div
          v-if="detail.request.status === 'pending' && showRejectForm"
          class="space-y-3 rounded-md border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/10"
        >
          <div class="text-sm font-medium text-gray-900 dark:text-white">
            {{ t('invoice.rejectTitle') }}
          </div>
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

        <div v-if="actionError" class="text-sm text-red-600 dark:text-red-400">
          {{ actionError }}
        </div>
      </div>

      <template #footer>
        <div v-if="detail" class="flex justify-end gap-2">
          <!-- Download (any state with file) -->
          <button
            v-if="detail.request.status === 'issued' && detail.request.has_file"
            class="btn btn-secondary"
            @click="download(detail.request)"
            :disabled="downloadingId === detail.request.id"
          >
            {{ downloadingId === detail.request.id ? t('common.processing') : t('invoice.download') }}
          </button>

          <!-- Pending: reject / approve -->
          <template v-if="detail.request.status === 'pending'">
            <button
              v-if="!showRejectForm"
              class="btn btn-secondary"
              @click="showRejectForm = true"
            >
              {{ t('invoice.reject') }}
            </button>
            <template v-else>
              <button class="btn btn-secondary" @click="showRejectForm = false">
                {{ t('common.cancel') }}
              </button>
              <button
                class="btn btn-danger"
                :disabled="actionLoading || !rejectReason.trim()"
                @click="confirmReject"
              >
                {{ actionLoading ? t('common.processing') : t('invoice.confirmReject') }}
              </button>
            </template>
            <button
              v-if="!showRejectForm"
              class="btn btn-primary"
              :disabled="actionLoading"
              @click="approve"
            >
              {{ actionLoading ? t('common.processing') : t('invoice.approve') }}
            </button>
          </template>

          <!-- Approved: issue -->
          <template v-if="detail.request.status === 'approved'">
            <button
              v-if="!showIssueForm"
              class="btn btn-primary"
              @click="showIssueForm = true"
            >
              {{ t('invoice.issue') }}
            </button>
            <template v-else>
              <button class="btn btn-secondary" @click="showIssueForm = false">
                {{ t('common.cancel') }}
              </button>
              <button
                class="btn btn-primary"
                :disabled="actionLoading || !canIssue"
                @click="confirmIssue"
              >
                {{ actionLoading ? t('common.processing') : t('invoice.confirmIssue') }}
              </button>
            </template>
          </template>

          <button class="btn btn-secondary" @click="closeDetail">
            {{ t('common.close') }}
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
import { adminInvoiceAPI, type InvoiceRequestDetail } from '@/api/admin/invoices'
import type { InvoiceRequest, InvoiceStatus } from '@/api/invoice'

const { t } = useI18n()

const items = ref<InvoiceRequest[]>([])
const loading = ref(false)
const statusFilter = ref<InvoiceStatus | ''>('')
const keyword = ref('')
const pagination = reactive({ total: 0, page: 1, page_size: 20, pages: 1 })
const downloadingId = ref<number | null>(null)

// Detail dialog state
const detailTarget = ref<InvoiceRequest | null>(null)
const detail = ref<InvoiceRequestDetail | null>(null)
const detailLoading = ref(false)
const detailError = ref('')

const actionLoading = ref(false)
const actionError = ref('')

const showRejectForm = ref(false)
const rejectReason = ref('')
const rejectError = ref('')

const showIssueForm = ref(false)
const issueForm = reactive<{ invoiceNo: string; file: File | null }>({
  invoiceNo: '',
  file: null
})
const issueError = ref('')

const canIssue = computed(() => !!issueForm.file && issueForm.invoiceNo.trim().length > 0)

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

async function openDetail(item: InvoiceRequest) {
  detailTarget.value = item
  detail.value = null
  detailLoading.value = true
  detailError.value = ''
  actionError.value = ''
  showRejectForm.value = false
  showIssueForm.value = false
  rejectReason.value = ''
  rejectError.value = ''
  issueForm.invoiceNo = ''
  issueForm.file = null
  issueError.value = ''
  try {
    detail.value = await adminInvoiceAPI.detail(item.id)
  } catch (e: any) {
    detailError.value = e?.message || String(e)
  } finally {
    detailLoading.value = false
  }
}

function closeDetail() {
  detailTarget.value = null
  detail.value = null
}

async function refreshDetail() {
  if (!detailTarget.value) return
  try {
    detail.value = await adminInvoiceAPI.detail(detailTarget.value.id)
  } catch (e: any) {
    detailError.value = e?.message || String(e)
  }
}

async function approve() {
  if (!detail.value) return
  actionLoading.value = true
  actionError.value = ''
  try {
    await adminInvoiceAPI.approve(detail.value.request.id)
    await refreshDetail()
    await reload()
  } catch (e: any) {
    actionError.value = e?.message || String(e)
  } finally {
    actionLoading.value = false
  }
}

async function confirmReject() {
  if (!detail.value) return
  actionLoading.value = true
  rejectError.value = ''
  try {
    await adminInvoiceAPI.reject(detail.value.request.id, rejectReason.value.trim())
    showRejectForm.value = false
    await refreshDetail()
    await reload()
  } catch (e: any) {
    rejectError.value = e?.message || String(e)
  } finally {
    actionLoading.value = false
  }
}

function onIssueFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  issueForm.file = input.files && input.files.length > 0 ? input.files[0] : null
}

async function confirmIssue() {
  if (!detail.value || !issueForm.file) return
  actionLoading.value = true
  issueError.value = ''
  try {
    await adminInvoiceAPI.issue(
      detail.value.request.id,
      issueForm.invoiceNo.trim(),
      issueForm.file
    )
    showIssueForm.value = false
    await refreshDetail()
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