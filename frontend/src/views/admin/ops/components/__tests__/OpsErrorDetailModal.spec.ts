import { flushPromises, shallowMount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import OpsErrorDetailModal from '../OpsErrorDetailModal.vue'

const mocks = vi.hoisted(() => ({
  getRequestErrorDetail: vi.fn(),
  listRequestErrorUpstreamErrors: vi.fn(),
  downloadErrorArchive: vi.fn(),
  showError: vi.fn()
}))

vi.mock('@/api/admin/ops', () => ({
  opsAPI: {
    getRequestErrorDetail: mocks.getRequestErrorDetail,
    getUpstreamErrorDetail: vi.fn(),
    listRequestErrorUpstreamErrors: mocks.listRequestErrorUpstreamErrors,
    downloadErrorArchive: mocks.downloadErrorArchive
  }
}))

vi.mock('@/stores', () => ({
  useAppStore: () => ({ showError: mocks.showError })
}))

vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-i18n')>()
  return {
    ...actual,
    useI18n: () => ({ t: (key: string) => key })
  }
})

describe('OpsErrorDetailModal', () => {
  beforeEach(() => {
    mocks.downloadErrorArchive.mockReset()
    mocks.showError.mockReset()
    mocks.getRequestErrorDetail.mockReset()
    mocks.listRequestErrorUpstreamErrors.mockReset()
    mocks.listRequestErrorUpstreamErrors.mockResolvedValue({ items: [] })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('downloads an archived request only after the administrator clicks', async () => {
    vi.useFakeTimers()
    const id = 'a'.repeat(32)
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 2, error_body: JSON.stringify({ diagnostic_archive: { id, expires_at: '2026-09-29T00:00:00Z', request_truncated: true, read_error_kind: 'truncated_body' } }) })
    mocks.downloadErrorArchive.mockResolvedValue(new Blob(['{}'], { type: 'application/json' }))
    const createObjectURL = vi.fn().mockReturnValue('blob:diagnostic')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 2, errorType: 'request' }, global: { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true } } })
    await flushPromises()
    expect(mocks.downloadErrorArchive).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('admin.ops.errorDetail.archiveTruncated')
    expect(wrapper.text()).toContain('truncated_body')
    await wrapper.get('[data-testid="download-error-archive"]').trigger('click')
    await flushPromises()
    expect(mocks.downloadErrorArchive).toHaveBeenCalledOnce()
    expect(mocks.downloadErrorArchive).toHaveBeenCalledWith(id)
    expect(click).toHaveBeenCalledOnce()
    vi.advanceTimersByTime(1000)
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:diagnostic')
    wrapper.unmount()
  })

  it('reports expired captures without attempting to replay the request', async () => {
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 2, error_body: JSON.stringify({ diagnostic_archive: { id: 'b'.repeat(32) } }) })
    mocks.downloadErrorArchive.mockRejectedValue({ status: 404 })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 2, errorType: 'request' }, global: { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true } } })
    await flushPromises()
    await wrapper.get('[data-testid="download-error-archive"]').trigger('click')
    await flushPromises()
    expect(mocks.showError).toHaveBeenCalledWith('admin.ops.errorDetail.archiveUnavailable')
    expect(mocks.downloadErrorArchive).toHaveBeenCalledOnce()
    wrapper.unmount()
  })

  it.each(['capacity_exhausted', 'queue_full'])('explains missing samples for %s without displaying year one', async (state) => {
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 3, error_body: JSON.stringify({ diagnostic_archive: { state, expires_at: '0001-01-01T00:00:00Z', read_error_kind: 'read_timeout' } }) })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 3, errorType: 'request' }, global: { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true } } })
    await flushPromises()
    expect(wrapper.find('[data-testid="download-error-archive"]').exists()).toBe(false)
    expect(wrapper.get('[data-testid="archive-not-captured"]').text()).toContain(state === 'queue_full' ? 'archiveQueueFull' : 'archiveCapacityExhausted')
    expect(wrapper.text()).not.toContain('admin.ops.errorDetail.archiveExpires')
    expect(wrapper.text()).toContain('read_timeout')
    expect(mocks.downloadErrorArchive).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('allows downloading metadata when the body memory budget is exhausted', async () => {
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 4, error_body: JSON.stringify({ diagnostic_archive: { id: 'c'.repeat(32), state: 'queued', capture_limited: true, request_truncated: true, read_error_kind: 'truncated_body' } }) })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 4, errorType: 'request' }, global: { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true } } })
    await flushPromises()
    expect(wrapper.find('[data-testid="download-error-archive"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('admin.ops.errorDetail.archiveMemoryLimited')
    expect(wrapper.text()).toContain('truncated_body')
    wrapper.unmount()
  })

  it('prioritizes upstream root cause and deduplicates diagnostic payloads', async () => {
    mocks.getRequestErrorDetail.mockResolvedValue({
      id: 1,
      created_at: '2026-08-19T00:00:00Z',
      phase: 'request',
      type: 'upstream_error',
      error_owner: 'provider',
      error_source: 'gateway',
      severity: 'P1',
      status_code: 502,
      upstream_status_code: 429,
      platform: 'openai',
      model: 'gpt-5.6',
      resolved: false,
      request_id: 'rid-1',
      message: 'All available accounts exhausted',
      error_body: '{"error":"same"}',
      upstream_error_message: 'provider rate limit exhausted',
      upstream_error_detail: '{"error":"same"}',
      upstream_errors: '[]',
      account_name: 'account',
      group_name: 'group',
      is_business_limited: false
    })

    const wrapper = shallowMount(OpsErrorDetailModal, {
      props: { show: true, errorId: 1, errorType: 'request' },
      global: {
        stubs: {
          BaseDialog: { template: '<div><slot /></div>' },
          Icon: true
        }
      }
    })
    await flushPromises()

    expect(wrapper.text()).toContain('provider rate limit exhausted')
    expect(wrapper.text()).toContain('admin.ops.errorDetail.upstreamStatus')
    expect(wrapper.text()).toContain('429')
    expect(wrapper.findAll('pre')).toHaveLength(2)
    expect(wrapper.text()).not.toContain('admin.ops.errorDetail.payloads.upstream_detail')
  })

  it.each([true, false, undefined])('separates upload status %s from archive truncation', async (complete) => {
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 5, error_body: JSON.stringify({ diagnostic_archive: {
      id: 'd'.repeat(32), state: 'queued', request_truncated: true,
      upload_complete: complete, received_bytes: 15925230, content_length: 22870745,
      missing_bytes: complete === false ? 6945515 : 0
    } }) })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 5, errorType: 'request' }, global: { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true } } })
    await flushPromises()
    expect(wrapper.text()).toContain('archiveTruncated')
    expect(wrapper.find('[data-testid="archive-upload-complete"]').exists()).toBe(complete === true)
    expect(wrapper.find('[data-testid="archive-upload-incomplete"]').exists()).toBe(complete === false)
    expect(wrapper.find('[data-testid="archive-upload-bytes"]').exists()).toBe(complete !== undefined)
    if (complete !== undefined) expect(wrapper.get('[data-testid="archive-upload-bytes"]').text()).toContain('15925230')
    if (complete === false) expect(wrapper.text()).toContain('6945515')
    wrapper.unmount()
  })
})
