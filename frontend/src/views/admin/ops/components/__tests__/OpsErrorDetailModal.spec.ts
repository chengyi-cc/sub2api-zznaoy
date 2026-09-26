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
})
