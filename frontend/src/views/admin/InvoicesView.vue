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
            @change="reload()"
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
            @keyup.enter="reload()"
          />
          <button class="btn btn-secondary" @click="reload()" :disabled="loading">
            {{ t('common.refresh') }}
          </button>
          <button class="btn btn-primary" @click="exportApproved" :disabled="exporting">
            {{ exporting ? t('invoice.exporting') : t('invoice.exportApproved') }}
          </button>
        </div>
      </div>

      <!-- Toolbar: 待审核批量通过 -->
      <div
        v-if="isPendingMode && selectedCount > 0"
        class="card flex items-center gap-3 p-3"
      >
        <span class="text-sm text-gray-600 dark:text-gray-300">
          {{ t('invoice.selectedCount', { n: selectedCount }) }}
        </span>
        <button class="btn btn-primary btn-sm" :disabled="batchApproving" @click="batchApprove">
          {{ batchApproving ? t('common.processing') : t('invoice.batchApprove', { n: selectedCount }) }}
        </button>
      </div>

      <!-- Toolbar: 待开票批量上传 PDF -->
      <div v-if="isApprovedMode" class="card flex flex-wrap items-center gap-3 p-3">
        <button class="btn btn-secondary btn-sm" @click="triggerBatchUpload">
          {{ t('invoice.batchUpload') }}
        </button>
        <input
          ref="batchUploadInput"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          class="hidden"
          @change="onBatchUploadChange"
        />
        <span class="text-xs text-gray-500 dark:text-gray-400">{{ t('invoice.batchUploadHint') }}</span>
        <span v-if="batchUploadSummary" class="text-xs font-medium text-blue-600 dark:text-blue-400">
          {{ batchUploadSummary }}
        </span>
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
              <th v-if="isPendingMode" class="px-3 py-2">
                <input
                  type="checkbox"
                  :checked="allPendingSelected"
                  @change="toggleSelectAll"
                />
              </th>
              <th class="px-4 py-2">ID</th>
              <th class="px-4 py-2">{{ t('invoice.adminUserId') }}</th>
              <th class="px-4 py-2">{{ t('invoice.titleField') }}</th>
              <th class="px-4 py-2">{{ t('invoice.userEmailCol') }}</th>
              <th class="px-4 py-2">{{ t('invoice.invoiceType') }}</th>
              <th class="px-4 py-2">{{ t('invoice.amount') }}</th>
              <th class="px-4 py-2">{{ t('invoice.sourceCountCol') }}</th>
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
              <td v-if="isPendingMode" class="px-3 py-2">
                <input
                  v-if="item.status === 'pending'"
                  type="checkbox"
                  :checked="selectedIds.has(item.id)"
                  @change="toggleSelect(item.id)"
                />
              </td>
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
              <td class="px-4 py-2 text-xs text-gray-600 dark:text-gray-300">
                {{ item.user_email || '—' }}
              </td>
              <td class="px-4 py-2">{{ t(`invoice.type.${item.invoice_type}`) }}</td>
              <td class="px-4 py-2 font-medium text-gray-900 dark:text-white">
                ¥{{ item.amount.toFixed(2) }}
              </td>
              <td class="px-4 py-2 text-xs text-gray-500 dark:text-gray-400">
                {{ sourceCount(item) }}
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
                <div class="flex flex-wrap items-center justify-end gap-2">
                  <!-- pending: 行内通过 / 驳回 / 详情 -->
                  <template v-if="item.status === 'pending'">
                    <button
                      class="btn btn-primary btn-sm"
                      :disabled="rowActionId === item.id"
                      @click="approveRow(item)"
                    >
                      {{ rowActionId === item.id ? t('common.processing') : t('invoice.approve') }}
                    </button>
                    <button class="btn btn-danger btn-sm" @click="openRejectDialog(item)">
                      {{ t('invoice.reject') }}
                    </button>
                    <button class="btn btn-secondary btn-sm" @click="openDetail(item)">
                      {{ t('invoice.viewDetail') }}
                    </button>
                  </template>

                  <!-- approved: 行内传 PDF + 开具 -->
                  <template v-else-if="item.status === 'approved'">
                    <div class="flex flex-col items-end gap-1">
                      <label class="cursor-pointer text-xs text-blue-600 hover:underline dark:text-blue-400">
                        {{ rowIssue[item.id]?.file ? rowIssue[item.id]?.invoiceNo : t('invoice.chooseFile') }}
                        <input
                          type="file"
                          accept="application/pdf,.pdf"
                          class="hidden"
                          @change="(e) => onRowFileChange(e, item)"
                        />
                      </label>
                      <span
                        v-if="rowIssue[item.id]?.amountMismatch"
                        class="text-xs font-medium text-red-600 dark:text-red-400"
                      >
                        ⚠ {{ t('invoice.amountMismatchTag', { parsed: rowIssue[item.id]?.parsedAmount }) }}
                      </span>
                    </div>
                    <button
                      class="btn btn-primary btn-sm"
                      :disabled="!rowIssue[item.id]?.file || rowIssuingId === item.id"
                      @click="issueRow(item)"
                    >
                      {{ rowIssuingId === item.id ? t('common.processing') : t('invoice.issue') }}
                    </button>
                    <button class="btn btn-secondary btn-sm" @click="openDetail(item)">
                      {{ t('invoice.viewDetail') }}
                    </button>
                  </template>

                  <!-- issued / rejected: 详情 + 下载 -->
                  <template v-else>
                    <button
                      v-if="item.status === 'issued' && item.has_file"
                      class="btn btn-secondary btn-sm"
                      :disabled="downloadingId === item.id"
                      @click="download(item)"
                    >
                      {{ downloadingId === item.id ? t('common.processing') : t('invoice.download') }}
                    </button>
                    <button class="btn btn-secondary btn-sm" @click="openDetail(item)">
                      {{ t('invoice.viewDetail') }}
                    </button>
                  </template>
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
          <div class="flex items-center justify-between">
            <div class="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
              {{ t('invoice.detailHeaderSection') }}
            </div>
            <button
              class="btn btn-secondary btn-sm"
              :title="t('invoice.copyInfoHint')"
              @click="copyInvoiceInfo(detail.request)"
            >
              {{ copied ? t('invoice.copySuccess') : t('invoice.copyInfo') }}
            </button>
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
              @input="onInvoiceNoInput"
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

    <!-- 行内驳回弹窗（与详情驳回共用逻辑） -->
    <BaseDialog
      :show="!!rejectDialogTarget"
      :title="t('invoice.rejectTitle')"
      @close="closeRejectDialog"
    >
      <div v-if="rejectDialogTarget" class="space-y-3">
        <div class="text-sm text-gray-600 dark:text-gray-300">
          {{ t('invoice.rejectConfirm', { id: rejectDialogTarget.id }) }}
        </div>
        <div>
          <label class="input-label">{{ t('invoice.rejectReason') }}</label>
          <textarea
            v-model="rejectDialogReason"
            rows="3"
            class="input mt-1 w-full"
            :placeholder="t('invoice.rejectReasonPlaceholder')"
          />
        </div>
        <div v-if="rejectDialogError" class="text-sm text-red-600 dark:text-red-400">
          {{ rejectDialogError }}
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-2">
          <button class="btn btn-secondary" @click="closeRejectDialog">
            {{ t('common.cancel') }}
          </button>
          <button
            class="btn btn-danger"
            :disabled="rejectDialogLoading || !rejectDialogReason.trim()"
            @click="confirmRejectDialog"
          >
            {{ rejectDialogLoading ? t('common.processing') : t('invoice.confirmReject') }}
          </button>
        </div>
      </template>
    </BaseDialog>
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import * as XLSX from 'xlsx'
import { saveAs } from 'file-saver'
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
const exporting = ref(false)
const copied = ref(false)

// 待审核：勾选批量通过
const selectedIds = ref<Set<number>>(new Set())
const batchApproving = ref(false)

// 行内审核操作（通过/驳回）：记录正在处理的行 id
const rowActionId = ref<number | null>(null)

// 行内驳回弹窗（行内 + 详情共用）
const rejectDialogTarget = ref<InvoiceRequest | null>(null)
const rejectDialogReason = ref('')
const rejectDialogError = ref('')
const rejectDialogLoading = ref(false)

// 待开票：每行待开具的 PDF 文件 + 状态（文件名/金额校验/匹配状态）
interface RowIssueState {
  file: File | null
  invoiceNo: string // 文件名去 .pdf
  parsedAmount: number | null // 从文件名解析的金额
  amountMismatch: boolean // 解析金额与申请金额不一致
}
const rowIssue = reactive<Record<number, RowIssueState>>({})
const rowIssuingId = ref<number | null>(null)
const batchUploadInput = ref<HTMLInputElement | null>(null)
const batchUploadSummary = ref('')

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
// 用户是否手动编辑过发票号：true 时不再用 PDF 文件名自动覆盖
const invoiceNoTouched = ref(false)
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

// 当前是否处于「待审核」/「待开票」工作模式（由状态筛选决定行内操作）
const isPendingMode = computed(() => statusFilter.value === 'pending')
const isApprovedMode = computed(() => statusFilter.value === 'approved')

// 当前页可勾选（pending）的行
const pendingItems = computed(() => items.value.filter((i) => i.status === 'pending'))
const allPendingSelected = computed(
  () => pendingItems.value.length > 0 && pendingItems.value.every((i) => selectedIds.value.has(i.id))
)
const selectedCount = computed(() => selectedIds.value.size)

// 来源数（订单 + 兑换码），从列表已有数组直接取
function sourceCount(item: InvoiceRequest): number {
  return (item.payment_order_ids?.length || 0) + (item.redeem_code_ids?.length || 0)
}

function toggleSelect(id: number) {
  const next = new Set(selectedIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedIds.value = next
}

function toggleSelectAll() {
  const next = new Set(selectedIds.value)
  if (allPendingSelected.value) {
    pendingItems.value.forEach((i) => next.delete(i.id))
  } else {
    pendingItems.value.forEach((i) => next.add(i.id))
  }
  selectedIds.value = next
}

// 解析 PDF 文件名：^金额[-]公司_时间戳.pdf -> { amount, company, invoiceNo(去.pdf) }
// 金额与公司之间的 - 可有可无。
function parseInvoiceFileName(name: string): { amount: number | null; company: string; invoiceNo: string } {
  const invoiceNo = name.replace(/\.pdf$/i, '').trim()
  const m = invoiceNo.match(/^(\d+(?:\.\d+)?)-?(.+?)_\d+$/)
  if (!m) {
    return { amount: null, company: '', invoiceNo }
  }
  return { amount: parseFloat(m[1]), company: m[2].trim(), invoiceNo }
}

// reload 重新拉取列表。
// keepRowIssue=true 时保留各行暂存的待开票 PDF 文件槽（用于「逐行开具」后刷新列表，
// 避免清空其它行已匹配/已选的文件）；默认（换页/筛选/手动刷新）清空，避免指向旧数据。
async function reload(opts?: { keepRowIssue?: boolean }) {
  const keepRowIssue = opts?.keepRowIssue === true
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
    selectedIds.value = new Set()
    if (!keepRowIssue) {
      Object.keys(rowIssue).forEach((k) => delete rowIssue[Number(k)])
      batchUploadSummary.value = ''
    } else {
      // 保留文件槽，但清掉已不在当前列表（如已开具离开 approved）的残留项
      const liveIds = new Set(items.value.map((i) => i.id))
      Object.keys(rowIssue).forEach((k) => {
        if (!liveIds.has(Number(k))) delete rowIssue[Number(k)]
      })
    }
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

// 行内通过（pending -> approved）
async function approveRow(item: InvoiceRequest) {
  rowActionId.value = item.id
  try {
    await adminInvoiceAPI.approve(item.id)
    await reload()
  } catch (e: any) {
    window.alert(e?.message || String(e))
  } finally {
    rowActionId.value = null
  }
}

// 批量通过勾选行
async function batchApprove() {
  const ids = Array.from(selectedIds.value)
  if (ids.length === 0) return
  batchApproving.value = true
  try {
    const res = await adminInvoiceAPI.batchApprove(ids)
    const okN = res.succeeded_ids.length
    const failN = res.failed.length
    if (failN === 0) {
      window.alert(t('invoice.batchApproveDone', { ok: okN }))
    } else {
      window.alert(t('invoice.batchApprovePartial', { ok: okN, fail: failN }))
    }
    await reload()
  } catch (e: any) {
    window.alert(e?.message || String(e))
  } finally {
    batchApproving.value = false
  }
}

// 打开驳回弹窗（行内或详情触发）
function openRejectDialog(item: InvoiceRequest) {
  rejectDialogTarget.value = item
  rejectDialogReason.value = ''
  rejectDialogError.value = ''
}

function closeRejectDialog() {
  rejectDialogTarget.value = null
}

async function confirmRejectDialog() {
  const target = rejectDialogTarget.value
  if (!target) return
  const reason = rejectDialogReason.value.trim()
  if (!reason) {
    rejectDialogError.value = t('invoice.rejectReasonRequired')
    return
  }
  rejectDialogLoading.value = true
  rejectDialogError.value = ''
  try {
    await adminInvoiceAPI.reject(target.id, reason)
    rejectDialogTarget.value = null
    // 详情弹窗开着的话同步刷新
    if (detailTarget.value && detailTarget.value.id === target.id) {
      await refreshDetail()
    }
    await reload()
  } catch (e: any) {
    rejectDialogError.value = e?.message || String(e)
  } finally {
    rejectDialogLoading.value = false
  }
}

// 为某行设置待开具的 PDF 文件，并按文件名解析金额做一致性校验
function setRowFile(item: InvoiceRequest, file: File | null) {
  if (!file) {
    delete rowIssue[item.id]
    return
  }
  const parsed = parseInvoiceFileName(file.name)
  rowIssue[item.id] = {
    file,
    invoiceNo: parsed.invoiceNo,
    parsedAmount: parsed.amount,
    amountMismatch: parsed.amount !== null && Math.abs(parsed.amount - item.amount) >= 0.01
  }
}

// 行内文件选择
function onRowFileChange(e: Event, item: InvoiceRequest) {
  const input = e.target as HTMLInputElement
  setRowFile(item, input.files && input.files.length > 0 ? input.files[0] : null)
}

// 行内开具：approved -> issued，发票号 = 文件名去 .pdf
async function issueRow(item: InvoiceRequest) {
  const st = rowIssue[item.id]
  if (!st || !st.file) return
  if (st.amountMismatch) {
    if (!window.confirm(t('invoice.amountMismatchConfirm', { parsed: st.parsedAmount, amount: item.amount.toFixed(2) }))) {
      return
    }
  }
  rowIssuingId.value = item.id
  try {
    await adminInvoiceAPI.issue(item.id, st.invoiceNo, st.file)
    delete rowIssue[item.id]
    // 保留其它行已匹配的文件槽，避免逐行开具时被清空
    await reload({ keepRowIssue: true })
  } catch (e: any) {
    window.alert(e?.message || String(e))
  } finally {
    rowIssuingId.value = null
  }
}

// 批量选择 PDF：按「金额相等 + 公司名互相包含」匹配到 approved 行
function triggerBatchUpload() {
  batchUploadInput.value?.click()
}

function onBatchUploadChange(e: Event) {
  const input = e.target as HTMLInputElement
  const files = input.files ? Array.from(input.files) : []
  input.value = '' // 允许重复选同一批
  if (files.length === 0) return

  const approvedRows = items.value.filter((i) => i.status === 'approved')
  // 已被本批某个 PDF 占用的行，不再参与后续匹配，避免一行被多个文件覆盖
  const usedRowIds = new Set<number>()
  let matched = 0
  let unmatched = 0
  for (const file of files) {
    const parsed = parseInvoiceFileName(file.name)
    const available = approvedRows.filter((r) => !usedRowIds.has(r.id))
    // 候选：金额相等
    let candidates = available.filter(
      (r) => parsed.amount !== null && Math.abs(parsed.amount - r.amount) < 0.01
    )
    // 多个同金额时，再用公司名互相包含缩小
    if (candidates.length > 1 && parsed.company) {
      const narrowed = candidates.filter(
        (r) => r.title.includes(parsed.company) || parsed.company.includes(r.title)
      )
      if (narrowed.length >= 1) candidates = narrowed
    }
    // 没金额匹配时退化为纯公司名匹配
    if (candidates.length === 0 && parsed.company) {
      candidates = available.filter(
        (r) => r.title.includes(parsed.company) || parsed.company.includes(r.title)
      )
    }
    if (candidates.length === 1) {
      setRowFile(candidates[0], file)
      usedRowIds.add(candidates[0].id)
      matched++
    } else {
      unmatched++
    }
  }
  batchUploadSummary.value = t('invoice.batchUploadSummary', { matched, unmatched })
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
  invoiceNoTouched.value = false
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
  // 自动用 PDF 文件名（去掉 .pdf 扩展名）填充发票号；
  // 仅当用户尚未手动编辑过发票号框时才覆盖，避免冲掉手填值。
  if (issueForm.file && !invoiceNoTouched.value) {
    const name = issueForm.file.name.replace(/\.pdf$/i, '').trim()
    if (name) {
      issueForm.invoiceNo = name
    }
  }
}

// 发票号输入框被手动编辑时打标记，停止文件名自动填充
function onInvoiceNoInput() {
  invoiceNoTouched.value = true
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

// 复制开票信息：抬头 ⇥ 识别号 ⇥ 金额（Tab 分隔），粘贴到 Excel 正好落 3 格。
async function copyInvoiceInfo(r: InvoiceRequest) {
  const cells = [r.title || '', r.tax_no || '', r.amount.toFixed(2)]
  const text = cells.join('\t')
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 1500)
  } catch (e: any) {
    console.error('copy invoice info failed', e)
  }
}

// 导出「已通过待开票」为 Excel：序号 / 公司名称·个人抬头 / 纳税人识别号 / 金额 / 用户邮箱 / 接收邮箱
async function exportApproved() {
  exporting.value = true
  try {
    const rows = await adminInvoiceAPI.exportApproved()
    if (rows.length === 0) {
      window.alert(t('invoice.exportEmpty'))
      return
    }
    const headers = [
      t('invoice.exportColSeq'),
      t('invoice.exportColTitle'),
      t('invoice.exportColTaxNo'),
      t('invoice.exportColAmount'),
      t('invoice.exportColUserEmail'),
      t('invoice.exportColRecipientEmail')
    ]
    const data = rows.map((r, idx) => [
      idx + 1,
      r.title,
      r.tax_no,
      r.amount,
      r.user_email,
      // 接收邮箱未填时回退注册邮箱（与开票通知实际去向一致）
      r.recipient_email || r.user_email
    ])
    const ws = XLSX.utils.aoa_to_sheet([headers, ...data])
    ws['!cols'] = [
      { wch: 6 },
      { wch: 32 },
      { wch: 24 },
      { wch: 12 },
      { wch: 28 },
      { wch: 28 }
    ]
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, t('invoice.exportSheetName'))
    const today = new Date().toISOString().slice(0, 10)
    saveAs(
      new Blob([XLSX.write(wb, { bookType: 'xlsx', type: 'array' })], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      }),
      `${t('invoice.exportFilePrefix')}_${today}.xlsx`
    )
  } catch (e: any) {
    console.error('export approved invoices failed', e)
    window.alert(t('invoice.exportFailed'))
  } finally {
    exporting.value = false
  }
}

onMounted(() => reload())
</script>