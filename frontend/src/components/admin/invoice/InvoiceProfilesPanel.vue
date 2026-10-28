<template>
  <div class="space-y-4">
    <div class="card p-4">
      <div class="flex flex-wrap items-center gap-3">
        <div class="min-w-[200px] flex-1">
          <p class="text-sm text-gray-500 dark:text-gray-400">{{ t('invoice.profile.adminDescription') }}</p>
        </div>
        <input
          v-model="keyword"
          class="input h-9 w-64 text-sm"
          :placeholder="t('invoice.profile.adminSearchPlaceholder')"
          @keyup.enter="search"
        />
        <button class="btn btn-secondary" :disabled="loading" @click="search">
          {{ t('common.refresh') }}
        </button>
        <button class="btn btn-primary" @click="openCreate">
          {{ t('invoice.profile.adminSetForUser') }}
        </button>
      </div>
    </div>

    <div class="card overflow-hidden">
      <div v-if="loading" class="p-8 text-center text-gray-500 dark:text-gray-400">
        {{ t('common.loading') }}
      </div>
      <div v-else-if="items.length === 0" class="p-12 text-center text-gray-500 dark:text-gray-400">
        {{ t('invoice.profile.adminEmpty') }}
      </div>
      <div v-else class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200 text-sm dark:divide-dark-700">
          <thead class="bg-gray-50 dark:bg-dark-800/40">
            <tr class="text-left text-xs uppercase text-gray-500 dark:text-gray-400">
              <th class="px-4 py-2">{{ t('invoice.adminUserId') }}</th>
              <th class="px-4 py-2">{{ t('invoice.userEmailCol') }}</th>
              <th class="px-4 py-2">{{ t('invoice.profile.companyName') }}</th>
              <th class="px-4 py-2">{{ t('invoice.taxNo') }}</th>
              <th class="px-4 py-2">{{ t('invoice.profile.source') }}</th>
              <th class="px-4 py-2">{{ t('invoice.profile.updatedAt') }}</th>
              <th class="px-4 py-2 text-right">{{ t('invoice.actions') }}</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100 dark:divide-dark-700">
            <tr v-for="item in items" :key="item.user_id" class="hover:bg-gray-50 dark:hover:bg-dark-800/50">
              <td class="px-4 py-2">{{ item.user_id }}</td>
              <td class="px-4 py-2 text-xs text-gray-600 dark:text-gray-300">{{ item.user_email || '—' }}</td>
              <td class="px-4 py-2 font-medium text-gray-900 dark:text-white">{{ item.title }}</td>
              <td class="px-4 py-2">
                <div class="font-mono text-gray-900 dark:text-white">{{ item.tax_no }}</div>
                <button
                  v-if="item.shared_user_count > 0"
                  type="button"
                  class="mt-0.5 text-xs font-medium text-amber-600 hover:underline dark:text-amber-400"
                  @click="filterByTaxNo(item.tax_no)"
                >
                  ⚠ {{ t('invoice.profile.sharedTag', { n: item.shared_user_count }) }}
                </button>
              </td>
              <td class="px-4 py-2 text-xs text-gray-500 dark:text-gray-400">
                {{ item.updated_by ? t('invoice.profile.updatedBy', { id: item.updated_by }) : t('invoice.profile.selfBound') }}
              </td>
              <td class="px-4 py-2 text-xs text-gray-500 dark:text-gray-400">{{ formatDateTime(item.updated_at) }}</td>
              <td class="px-4 py-2">
                <div class="flex justify-end gap-2">
                  <button class="btn btn-secondary btn-sm" @click="openEdit(item)">
                    {{ t('invoice.profile.edit') }}
                  </button>
                  <button class="btn btn-danger btn-sm" @click="resetTarget = item">
                    {{ t('invoice.profile.reset') }}
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div v-if="pagination.total > pagination.page_size" class="flex items-center justify-end gap-2">
      <button class="btn btn-secondary btn-sm" :disabled="pagination.page <= 1" @click="changePage(pagination.page - 1)">
        {{ t('common.previous') }}
      </button>
      <span class="text-sm text-gray-500 dark:text-gray-400">{{ pagination.page }} / {{ pagination.pages }}</span>
      <button
        class="btn btn-secondary btn-sm"
        :disabled="pagination.page >= pagination.pages"
        @click="changePage(pagination.page + 1)"
      >
        {{ t('common.next') }}
      </button>
    </div>

    <!-- 修改 / 为用户设置 -->
    <BaseDialog
      :show="showEdit"
      :title="editUserLocked ? t('invoice.profile.editTitle') : t('invoice.profile.createTitle')"
      @close="showEdit = false"
    >
      <form class="space-y-4" @submit.prevent="save">
        <div>
          <label class="input-label">{{ t('invoice.adminUserId') }}</label>
          <input
            v-model.number="editForm.user_id"
            type="number"
            min="1"
            class="input mt-1 w-full"
            :disabled="editUserLocked"
          />
        </div>
        <div>
          <label class="input-label">{{ t('invoice.profile.companyName') }}</label>
          <input v-model="editForm.title" type="text" class="input mt-1 w-full" maxlength="66" />
        </div>
        <div>
          <label class="input-label">{{ t('invoice.profile.taxNo') }}</label>
          <input v-model="editForm.tax_no" type="text" class="input mt-1 w-full font-mono uppercase" maxlength="18" />
          <p v-if="editTaxNoInvalid" class="mt-1 text-xs text-red-600 dark:text-red-400">
            {{ t('invoice.profile.taxNoInvalid') }}
          </p>
        </div>
        <p v-if="editError" class="text-sm text-red-600 dark:text-red-400">{{ editError }}</p>
      </form>
      <template #footer>
        <div class="flex justify-end gap-3">
          <button class="btn btn-secondary" @click="showEdit = false">{{ t('common.cancel') }}</button>
          <button class="btn btn-primary" :disabled="!canSave || saving" @click="save">
            {{ saving ? t('common.processing') : t('common.save') }}
          </button>
        </div>
      </template>
    </BaseDialog>

    <ConfirmDialog
      :show="!!resetTarget"
      :title="t('invoice.profile.resetConfirmTitle')"
      :message="resetTarget ? t('invoice.profile.resetConfirmMessage', { userId: resetTarget.user_id, title: resetTarget.title }) : ''"
      :confirm-text="t('invoice.profile.reset')"
      danger
      @confirm="confirmReset"
      @cancel="resetTarget = null"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import BaseDialog from '@/components/common/BaseDialog.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'
import { adminInvoiceAPI, type AdminInvoiceProfile } from '@/api/admin/invoices'
import { useAppStore } from '@/stores/app'
import { extractI18nErrorMessage } from '@/utils/apiError'
import { formatDateTime } from '@/utils/format'
import {
  isValidUnifiedSocialCreditCode,
  normalizeUnifiedSocialCreditCode
} from '@/utils/unifiedSocialCreditCode'

const { t } = useI18n()
const appStore = useAppStore()

const items = ref<AdminInvoiceProfile[]>([])
const loading = ref(false)
const keyword = ref('')
const pagination = reactive({ total: 0, page: 1, page_size: 20, pages: 1 })

const showEdit = ref(false)
const editUserLocked = ref(false)
const saving = ref(false)
const editError = ref('')
const editForm = reactive({ user_id: 0 as number, title: '', tax_no: '' })
const resetTarget = ref<AdminInvoiceProfile | null>(null)

const editTaxNo = computed(() => normalizeUnifiedSocialCreditCode(editForm.tax_no))
const editTaxNoInvalid = computed(() => editTaxNo.value.length > 0 && !isValidUnifiedSocialCreditCode(editTaxNo.value))
const canSave = computed(
  () =>
    Number.isInteger(editForm.user_id) &&
    editForm.user_id > 0 &&
    editForm.title.trim().length > 0 &&
    isValidUnifiedSocialCreditCode(editTaxNo.value)
)

function errorMessage(e: unknown): string {
  return extractI18nErrorMessage(e, t, 'invoice.errors', t('common.error'))
}

async function reload(): Promise<void> {
  loading.value = true
  try {
    const resp = await adminInvoiceAPI.listProfiles({
      q: keyword.value.trim() || undefined,
      page: pagination.page,
      page_size: pagination.page_size
    })
    // 当前页已越过末页（如重置了末页最后一条）时回退到末页重新加载；pages 至少为 1
    if (pagination.page > 1 && pagination.page > resp.pages) {
      pagination.page = resp.pages
      return reload()
    }
    items.value = resp.items || []
    pagination.total = resp.total
    pagination.pages = resp.pages
  } catch (e) {
    items.value = []
    appStore.showError(errorMessage(e))
  } finally {
    loading.value = false
  }
}

function search() {
  pagination.page = 1
  reload()
}

function filterByTaxNo(taxNo: string) {
  keyword.value = taxNo
  search()
}

function changePage(n: number) {
  pagination.page = n
  reload()
}

function openCreate() {
  editUserLocked.value = false
  editForm.user_id = 0
  editForm.title = ''
  editForm.tax_no = ''
  editError.value = ''
  showEdit.value = true
}

function openEdit(item: AdminInvoiceProfile) {
  editUserLocked.value = true
  editForm.user_id = item.user_id
  editForm.title = item.title
  editForm.tax_no = item.tax_no
  editError.value = ''
  showEdit.value = true
}

async function save() {
  if (!canSave.value || saving.value) return
  saving.value = true
  editError.value = ''
  try {
    await adminInvoiceAPI.upsertProfile(editForm.user_id, {
      title: editForm.title.trim(),
      tax_no: editTaxNo.value
    })
    showEdit.value = false
    appStore.showSuccess(t('invoice.profile.saveSuccess'))
    await reload()
  } catch (e) {
    editError.value = errorMessage(e)
  } finally {
    saving.value = false
  }
}

async function confirmReset() {
  const target = resetTarget.value
  resetTarget.value = null
  if (!target) return
  try {
    await adminInvoiceAPI.resetProfile(target.user_id)
    appStore.showSuccess(t('invoice.profile.resetSuccess'))
    await reload()
  } catch (e) {
    appStore.showError(errorMessage(e))
  }
}

onMounted(reload)
</script>
