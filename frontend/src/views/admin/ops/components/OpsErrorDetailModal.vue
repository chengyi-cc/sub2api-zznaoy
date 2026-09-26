<template>
  <BaseDialog :show="show" :title="title" width="full" :close-on-click-outside="true" @close="close">
    <div v-if="loading" class="flex items-center justify-center py-16">
      <div class="flex flex-col items-center gap-3">
        <div class="h-8 w-8 animate-spin rounded-full border-b-2 border-primary-600"></div>
        <div class="text-sm font-medium text-gray-500 dark:text-gray-400">{{ t('admin.ops.errorDetail.loading') }}</div>
      </div>
    </div>

    <div v-else-if="!detail" class="py-10 text-center text-sm text-gray-500 dark:text-gray-400">
      {{ emptyText }}
    </div>

    <div v-else class="space-y-6 p-6">
      <!-- Summary -->
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.requestId') }}</div>
          <div class="mt-1 break-all font-mono text-sm font-medium text-gray-900 dark:text-white">
            {{ requestId || '—' }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.time') }}</div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            {{ formatDateTime(detail.created_at) }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">
            {{ isUpstreamError(detail) ? t('admin.ops.errorDetail.account') : t('admin.ops.errorDetail.user') }}
          </div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            <template v-if="isUpstreamError(detail)">
              {{ detail.account_name || (detail.account_id != null ? String(detail.account_id) : '—') }}
            </template>
            <template v-else>
              {{ detail.user_email || (detail.user_id != null ? String(detail.user_id) : '—') }}
            </template>
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.platform') }}</div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            {{ detail.platform || '—' }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.group') }}</div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            {{ detail.group_name || (detail.group_id != null ? String(detail.group_id) : '—') }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.model') }}</div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            <template v-if="hasModelMapping(detail)">
              <span class="font-mono">{{ detail.requested_model }}</span>
              <span class="mx-1 text-gray-400">→</span>
              <span class="font-mono text-primary-600 dark:text-primary-400">{{ detail.upstream_model }}</span>
            </template>
            <template v-else>
              {{ displayModel(detail) || '—' }}
            </template>
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.inboundEndpoint') }}</div>
          <div class="mt-1 break-all font-mono text-sm font-medium text-gray-900 dark:text-white">
            {{ detail.inbound_endpoint || '—' }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.upstreamEndpoint') }}</div>
          <div class="mt-1 break-all font-mono text-sm font-medium text-gray-900 dark:text-white">
            {{ detail.upstream_endpoint || '—' }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.clientStatus') }}</div>
          <div class="mt-1">
            <span :class="['inline-flex items-center rounded-lg px-2 py-1 text-xs font-black ring-1 ring-inset shadow-sm', statusClass]">
              {{ detail.client_status_code ?? '—' }}
            </span>
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.upstreamStatus') }}</div>
          <div class="mt-1">
            <span :class="['inline-flex items-center rounded-lg px-2 py-1 text-xs font-black ring-1 ring-inset shadow-sm', upstreamStatusClass]">
              {{ detail.upstream_status_code ?? '—' }}
            </span>
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.requestType') }}</div>
          <div class="mt-1 text-sm font-medium text-gray-900 dark:text-white">
            {{ formatRequestTypeLabel(detail.request_type) }}
          </div>
        </div>

        <div class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.message') }}</div>
          <div class="mt-1 break-words text-sm font-medium text-gray-900 dark:text-white" :title="rootCauseMessage">
            {{ rootCauseMessage || '—' }}
          </div>
        </div>

        <div v-if="detail.api_key_prefix" class="rounded-xl bg-gray-50 p-4 dark:bg-dark-900">
          <div class="text-xs font-bold uppercase tracking-wider text-gray-400">{{ t('admin.ops.errorDetail.apiKeyPrefix') }}</div>
          <div class="mt-1 font-mono text-sm font-medium text-gray-900 dark:text-white">
            {{ detail.api_key_prefix }}
          </div>
        </div>

      </div>

      <div v-if="rootCauseMessage" class="rounded-xl bg-amber-50 p-6 dark:bg-amber-900/10">
        <h3 class="text-sm font-black uppercase tracking-wider text-amber-900 dark:text-amber-200">{{ t('admin.ops.errorDetail.rootCause') }}</h3>
        <div class="mt-3 break-words text-sm font-medium text-amber-900 dark:text-amber-100">{{ rootCauseMessage }}</div>
      </div>

      <div class="rounded-xl bg-gray-50 p-6 dark:bg-dark-900">
        <h3 class="text-sm font-black uppercase tracking-wider text-gray-900 dark:text-white">{{ t('admin.ops.errorDetail.diagnosticPayloads') }}</h3>
        <div v-if="archiveRef" class="mt-3 space-y-2 text-sm text-gray-600 dark:text-gray-300">
          <p>{{ t('admin.ops.errorDetail.archiveNotice') }}</p>
          <p v-if="archiveRef.expires_at">{{ t('admin.ops.errorDetail.archiveExpires') }} {{ formatDateTime(archiveRef.expires_at) }}</p>
          <p v-if="archiveRef.request_truncated">{{ t('admin.ops.errorDetail.archiveTruncated') }}</p>
          <p v-if="archiveRef.upload_complete === true" data-testid="archive-upload-complete">{{ t('admin.ops.errorDetail.archiveUploadComplete') }}</p>
          <p v-else-if="archiveRef.upload_complete === false" data-testid="archive-upload-incomplete">{{ t('admin.ops.errorDetail.archiveUploadIncomplete') }}</p>
          <p v-if="archiveRef.upload_complete !== null" data-testid="archive-upload-bytes">
            {{ t('admin.ops.errorDetail.archiveReceivedBytes') }}: {{ archiveRef.received_bytes ?? '—' }};
            {{ t('admin.ops.errorDetail.archiveExpectedBytes') }}: {{ archiveRef.content_length !== null && archiveRef.content_length >= 0 ? archiveRef.content_length : '—' }}
            <span v-if="archiveRef.missing_bytes">; {{ t('admin.ops.errorDetail.archiveMissingBytes') }}: {{ archiveRef.missing_bytes }}</span>
          </p>
          <p v-if="archiveRef.capture_limited">{{ t('admin.ops.errorDetail.archiveMemoryLimited') }}</p>
          <p v-if="archiveRef.read_error_kind">{{ t('admin.ops.errorDetail.archiveReadError') }}: {{ archiveRef.read_error_kind }}</p>
          <button v-if="archiveRef.id" type="button" class="btn btn-secondary" data-testid="download-error-archive" :disabled="archiveDownloading" @click="downloadArchive">
            {{ archiveDownloading ? t('common.loading') : t('admin.ops.errorDetail.archiveDownload') }}
          </button>
          <p v-else data-testid="archive-not-captured">{{ archiveStateMessage }}</p>
        </div>
        <div v-if="!diagnosticPayloadSections.length" class="mt-4 text-sm text-gray-500 dark:text-gray-400">{{ t('common.noData') }}</div>
        <div v-else class="mt-4 space-y-4">
          <div v-for="section in diagnosticPayloadSections" :key="section.key">
            <div class="mb-2 text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">{{ diagnosticPayloadLabel(section.key) }}</div>
            <pre class="max-h-[520px] overflow-auto rounded-xl border border-gray-200 bg-white p-4 text-xs text-gray-800 dark:border-dark-700 dark:bg-dark-800 dark:text-gray-100"><code>{{ prettyJSON(section.value) }}</code></pre>
          </div>
        </div>
      </div>

      <!-- Upstream errors list (only for request errors) -->
      <div v-if="showUpstreamList" class="rounded-xl bg-gray-50 p-6 dark:bg-dark-900">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h3 class="text-sm font-black uppercase tracking-wider text-gray-900 dark:text-white">{{ t('admin.ops.errorDetails.upstreamErrors') }}</h3>
          <div class="text-xs text-gray-500 dark:text-gray-400" v-if="correlatedUpstreamLoading">{{ t('common.loading') }}</div>
        </div>

        <div v-if="!correlatedUpstreamLoading && !correlatedUpstreamErrors.length" class="mt-3 text-sm text-gray-500 dark:text-gray-400">
          {{ t('common.noData') }}
        </div>

        <div v-else class="mt-4 space-y-3">
          <div
            v-for="(ev, idx) in correlatedUpstreamErrors"
            :key="ev.id"
            class="rounded-xl border border-gray-200 bg-white p-4 dark:border-dark-700 dark:bg-dark-800"
          >
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="text-xs font-black text-gray-900 dark:text-white">
                #{{ idx + 1 }}
                <span v-if="ev.type" class="ml-2 rounded-md bg-gray-100 px-2 py-0.5 font-mono text-[10px] font-bold text-gray-700 dark:bg-dark-700 dark:text-gray-200">{{ ev.type }}</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="font-mono text-xs text-gray-500 dark:text-gray-400">
                  {{ ev.status_code ?? '—' }}
                </div>
                <button
                  type="button"
                  class="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[10px] font-bold text-primary-700 hover:bg-primary-50 disabled:cursor-not-allowed disabled:opacity-60 dark:text-primary-200 dark:hover:bg-dark-700"
                  :disabled="!getUpstreamResponsePreview(ev)"
                  :title="getUpstreamResponsePreview(ev) ? '' : t('common.noData')"
                  @click="toggleUpstreamDetail(ev.id)"
                >
                  <Icon
                    :name="expandedUpstreamDetailIds.has(ev.id) ? 'chevronDown' : 'chevronRight'"
                    size="xs"
                    :stroke-width="2"
                  />
                  <span>
                    {{
                      expandedUpstreamDetailIds.has(ev.id)
                        ? t('admin.ops.errorDetail.responsePreview.collapse')
                        : t('admin.ops.errorDetail.responsePreview.expand')
                    }}
                  </span>
                </button>
              </div>
            </div>

            <div class="mt-3 grid grid-cols-1 gap-2 text-xs text-gray-600 dark:text-gray-300 sm:grid-cols-2">
              <div>
                <span class="text-gray-400">{{ t('admin.ops.errorDetail.upstreamEvent.status') }}:</span>
                <span class="ml-1 font-mono">{{ ev.status_code ?? '—' }}</span>
              </div>
              <div>
                <span class="text-gray-400">{{ t('admin.ops.errorDetail.upstreamEvent.requestId') }}:</span>
                <span class="ml-1 font-mono">{{ ev.request_id || ev.client_request_id || '—' }}</span>
              </div>
            </div>

            <div v-if="ev.message" class="mt-3 break-words text-sm font-medium text-gray-900 dark:text-white">{{ ev.message }}</div>

            <pre
              v-if="expandedUpstreamDetailIds.has(ev.id)"
              class="mt-3 max-h-[240px] overflow-auto rounded-xl border border-gray-200 bg-gray-50 p-3 text-xs text-gray-800 dark:border-dark-700 dark:bg-dark-900 dark:text-gray-100"
            ><code>{{ prettyJSON(getUpstreamResponsePreview(ev)) }}</code></pre>
          </div>
        </div>
      </div>
    </div>
    <template v-if="backToList" #footer>
      <button
        type="button"
        class="btn btn-secondary"
        data-testid="error-detail-back-to-list"
        @click="goBack"
      >
        {{ t('admin.ops.errorDetail.backToList') }}
      </button>
    </template>
  </BaseDialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import BaseDialog from '@/components/common/BaseDialog.vue'
import Icon from '@/components/icons/Icon.vue'
import { useAppStore } from '@/stores'
import { opsAPI, type OpsErrorDetail } from '@/api/admin/ops'
import { formatDateTime } from '@/utils/format'
import { resolveUpstreamPayload } from '../utils/errorDetailResponse'

interface Props {
  show: boolean
  errorId: number | null
  errorType?: 'request' | 'upstream'
  backToList?: boolean
}

interface Emits {
  (e: 'update:show', value: boolean): void
  (e: 'back'): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const { t } = useI18n()
const appStore = useAppStore()

const loading = ref(false)
const detail = ref<OpsErrorDetail | null>(null)
const archiveDownloading = ref(false)
const archiveRef = computed(() => {
  try {
    const ref = JSON.parse(detail.value?.error_body || '{}').diagnostic_archive
    if (!ref || typeof ref !== 'object') return null
    return {
      id: typeof ref.id === 'string' && /^[a-f0-9]{32}$/.test(ref.id) ? ref.id : '',
      expires_at: typeof ref.expires_at === 'string' && !ref.expires_at.startsWith('0001-') && Number.isFinite(Date.parse(ref.expires_at)) ? ref.expires_at : '',
      state: typeof ref.state === 'string' ? ref.state : '',
      capture_limited: ref.capture_limited === true,
      request_truncated: ref.request_truncated === true,
      upload_complete: typeof ref.upload_complete === 'boolean' ? ref.upload_complete : null,
      received_bytes: Number.isSafeInteger(ref.received_bytes) && ref.received_bytes >= 0 ? ref.received_bytes : null,
      content_length: Number.isSafeInteger(ref.content_length) && ref.content_length >= -1 ? ref.content_length : null,
      missing_bytes: Number.isSafeInteger(ref.missing_bytes) && ref.missing_bytes > 0 ? ref.missing_bytes : null,
      read_error_kind: typeof ref.read_error_kind === 'string' ? ref.read_error_kind : ''
    }
  } catch { return null }
})

const archiveStateMessage = computed(() => {
  if (archiveRef.value?.state === 'capacity_exhausted') return t('admin.ops.errorDetail.archiveCapacityExhausted')
  if (archiveRef.value?.state === 'queue_full') return t('admin.ops.errorDetail.archiveQueueFull')
  return t('admin.ops.errorDetail.archiveNotCaptured')
})

async function downloadArchive() {
  const id = archiveRef.value?.id
  if (!id || archiveDownloading.value) return
  archiveDownloading.value = true
  try {
    const blob = await opsAPI.downloadErrorArchive(id)
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'error-capture-' + id + '.json'
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  } catch {
    appStore.showError(t('admin.ops.errorDetail.archiveUnavailable'))
  } finally {
    archiveDownloading.value = false
  }
}

const showUpstreamList = computed(() => props.errorType === 'request')

const requestId = computed(() => detail.value?.request_id || detail.value?.client_request_id || '')

type DiagnosticPayloadKey = 'client' | 'upstream_message' | 'upstream_detail' | 'upstream_events'

const rootCauseMessage = computed(() => {
  const current = detail.value
  if (!current) return ''
  for (const candidate of [current.upstream_error_message, current.upstream_error_detail, current.message, current.error_body]) {
    const value = meaningfulPayload(candidate)
    if (value) return value
  }
  return ''
})

const diagnosticPayloadSections = computed(() => {
  const current = detail.value
  if (!current) return []
  const candidates: Array<{ key: DiagnosticPayloadKey; value: string }> = [
    { key: 'client', value: meaningfulPayload(current.error_body) },
    { key: 'upstream_message', value: meaningfulPayload(current.upstream_error_message) },
    { key: 'upstream_detail', value: meaningfulPayload(current.upstream_error_detail) },
    { key: 'upstream_events', value: meaningfulPayload(current.upstream_errors) }
  ]
  return candidates.filter((section, index, all) => {
    return section.value && all.findIndex(candidate => candidate.value === section.value) === index
  })
})

function meaningfulPayload(candidate: unknown): string {
  const value = String(candidate || '').trim()
  if (!value || value === '[]' || value === '{}' || value.toLowerCase() === 'null') return ''
  return value
}

function diagnosticPayloadLabel(key: DiagnosticPayloadKey): string {
  return t(`admin.ops.errorDetail.payloads.${key}`)
}

const title = computed(() => {
  if (!props.errorId) return t('admin.ops.errorDetail.title')
  return t('admin.ops.errorDetail.titleWithId', { id: String(props.errorId) })
})

const emptyText = computed(() => t('admin.ops.errorDetail.noErrorSelected'))

function isUpstreamError(d: OpsErrorDetail | null): boolean {
  if (!d) return false
  const phase = String(d.phase || '').toLowerCase()
  const owner = String(d.error_owner || '').toLowerCase()
  return phase === 'upstream' && owner === 'provider'
}

function formatRequestTypeLabel(type: number | null | undefined): string {
  switch (type) {
    case 1: return t('admin.ops.errorDetail.requestTypeSync')
    case 2: return t('admin.ops.errorDetail.requestTypeStream')
    case 3: return t('admin.ops.errorDetail.requestTypeWs')
    default: return t('admin.ops.errorDetail.requestTypeUnknown')
  }
}

function hasModelMapping(d: OpsErrorDetail | null): boolean {
  if (!d) return false
  const requested = String(d.requested_model || '').trim()
  const upstream = String(d.upstream_model || '').trim()
  return !!requested && !!upstream && requested !== upstream
}

function displayModel(d: OpsErrorDetail | null): string {
  if (!d) return ''
  const upstream = String(d.upstream_model || '').trim()
  if (upstream) return upstream
  const requested = String(d.requested_model || '').trim()
  if (requested) return requested
  return String(d.model || '').trim()
}

const correlatedUpstream = ref<OpsErrorDetail[]>([])
const correlatedUpstreamLoading = ref(false)

const correlatedUpstreamErrors = computed<OpsErrorDetail[]>(() => correlatedUpstream.value)

const expandedUpstreamDetailIds = ref(new Set<number>())

function getUpstreamResponsePreview(ev: OpsErrorDetail): string {
  const upstreamPayload = resolveUpstreamPayload(ev)
  if (upstreamPayload) return upstreamPayload
  return String(ev.error_body || '').trim()
}

function toggleUpstreamDetail(id: number) {
  const next = new Set(expandedUpstreamDetailIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expandedUpstreamDetailIds.value = next
}

async function fetchCorrelatedUpstreamErrors(requestErrorId: number) {
  correlatedUpstreamLoading.value = true
  try {
    const res = await opsAPI.listRequestErrorUpstreamErrors(
      requestErrorId,
      { page: 1, page_size: 100, view: 'all' },
      { include_detail: true }
    )
    correlatedUpstream.value = res.items || []
  } catch (err) {
    console.error('[OpsErrorDetailModal] Failed to load correlated upstream errors', err)
    correlatedUpstream.value = []
  } finally {
    correlatedUpstreamLoading.value = false
  }
}

function close() {
  emit('update:show', false)
}

function goBack() {
  emit('update:show', false)
  emit('back')
}

function prettyJSON(raw?: string): string {
  if (!raw) return 'N/A'
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

async function fetchDetail(id: number) {
  loading.value = true
  try {
    const kind = props.errorType || (detail.value?.phase === 'upstream' ? 'upstream' : 'request')
    const d = kind === 'upstream' ? await opsAPI.getUpstreamErrorDetail(id) : await opsAPI.getRequestErrorDetail(id)
    detail.value = d
  } catch (err: any) {
    detail.value = null
    appStore.showError(err?.message || t('admin.ops.failedToLoadErrorDetail'))
  } finally {
    loading.value = false
  }
}

watch(
  () => [props.show, props.errorId] as const,
  ([show, id]) => {
    if (!show) {
      detail.value = null
      return
    }
    if (typeof id === 'number' && id > 0) {
      expandedUpstreamDetailIds.value = new Set()
      fetchDetail(id)
      if (props.errorType === 'request') {
        fetchCorrelatedUpstreamErrors(id)
      } else {
        correlatedUpstream.value = []
      }
    }
  },
  { immediate: true }
)

function statusBadgeClass(code: number): string {
  if (code >= 500) return 'bg-red-50 text-red-700 ring-red-600/20 dark:bg-red-900/30 dark:text-red-400 dark:ring-red-500/30'
  if (code === 429) return 'bg-purple-50 text-purple-700 ring-purple-600/20 dark:bg-purple-900/30 dark:text-purple-400 dark:ring-purple-500/30'
  if (code >= 400) return 'bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-900/30 dark:text-amber-400 dark:ring-amber-500/30'
  return 'bg-gray-50 text-gray-700 ring-gray-600/20 dark:bg-gray-900/30 dark:text-gray-400 dark:ring-gray-500/30'
}

const statusClass = computed(() => statusBadgeClass(detail.value?.client_status_code ?? 0))

const upstreamStatusClass = computed(() => statusBadgeClass(detail.value?.upstream_status_code ?? 0))

</script>
