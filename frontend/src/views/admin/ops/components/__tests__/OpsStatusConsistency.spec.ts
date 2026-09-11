import { flushPromises, shallowMount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import OpsErrorDetailModal from '../OpsErrorDetailModal.vue'
import OpsRequestDetailsModal from '../OpsRequestDetailsModal.vue'

const mocks = vi.hoisted(() => ({
  desktop: true,
  getRequestErrorDetail: vi.fn(),
  listRequestDetails: vi.fn()
}))

vi.mock('@/api/admin/ops', () => ({
  opsAPI: {
    getRequestErrorDetail: mocks.getRequestErrorDetail,
    listRequestDetails: mocks.listRequestDetails,
    listRequestErrorUpstreamErrors: vi.fn().mockResolvedValue({ items: [] })
  }
}))
vi.mock('@/stores', () => ({ useAppStore: () => ({ showError: vi.fn() }) }))
vi.mock('@/composables/useClipboard', () => ({ useClipboard: () => ({ copyToClipboard: vi.fn() }) }))
vi.mock('@vueuse/core', () => ({ useMediaQuery: () => mocks.desktop }))
vi.mock('vue-i18n', async (importOriginal) => ({
  ...await importOriginal<typeof import('vue-i18n')>(),
  useI18n: () => ({ t: (key: string) => key })
}))

const global = { stubs: { BaseDialog: { template: '<div><slot /></div>' }, Icon: true, Pagination: true } }

describe('Ops client/upstream status consistency', () => {
  it.each([403, null])('separates actual client status from monitoring status (upstream %s)', async (upstream) => {
    mocks.getRequestErrorDetail.mockResolvedValue({
      id: 1, created_at: '2026-09-11T00:00:00Z', phase: 'request', type: 'upstream_error',
      status_code: upstream ?? 502, client_status_code: 502, upstream_status_code: upstream,
      resolved: false, error_body: '{}', upstream_errors: '[]'
    })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 1, errorType: 'request' }, global })
    await flushPromises()
    const card = (key: string) => wrapper.findAll('div').find((node) => node.element.children.length === 0 && node.text() === key)?.element.parentElement
    const clientCard = card('admin.ops.errorDetail.clientStatus')
    expect(clientCard?.textContent).toContain('502')
    expect(clientCard?.querySelector('span')?.className).toContain('text-red-700')
    const upstreamCard = card('admin.ops.errorDetail.upstreamStatus')
    expect(upstreamCard?.textContent).toContain(upstream === null ? '—' : '403')
    wrapper.unmount()
  })

  it('does not guess a missing client status from the monitoring status', async () => {
    mocks.getRequestErrorDetail.mockResolvedValue({ id: 1, status_code: 403, client_status_code: null, upstream_status_code: 403 })
    const wrapper = shallowMount(OpsErrorDetailModal, { props: { show: true, errorId: 1, errorType: 'request' }, global })
    await flushPromises()
    const label = wrapper.findAll('div').find((node) => node.element.children.length === 0 && node.text() === 'admin.ops.errorDetail.clientStatus')
    expect(label?.element.parentElement?.textContent).toContain('—')
    wrapper.unmount()
  })

  it.each([true, false])('shows both status sources in request rows (desktop=%s)', async (desktop) => {
    mocks.desktop = desktop
    mocks.listRequestDetails.mockResolvedValue({ total: 2, items: [
      { kind: 'error', created_at: '2026-09-11T00:00:00Z', status_code: 502, upstream_status_code: 403, error_id: 1 },
      { kind: 'error', created_at: '2026-09-11T00:00:00Z', status_code: 400, upstream_status_code: null, error_id: 2 }
    ] })
    const wrapper = shallowMount(OpsRequestDetailsModal, { props: { modelValue: false, timeRange: '1h', preset: { title: 'test' } }, global })
    await wrapper.setProps({ modelValue: true })
    await flushPromises()
    expect(wrapper.text()).toContain('admin.ops.requestDetails.table.clientStatus')
    expect(wrapper.text()).toContain('admin.ops.requestDetails.table.upstreamStatus')
    if (desktop) {
      const rows = wrapper.findAll('tbody tr')
      expect(rows[0].findAll('td').slice(5, 7).map((cell) => cell.text())).toEqual(['502', '403'])
      expect(rows[1].findAll('td').slice(5, 7).map((cell) => cell.text())).toEqual(['400', '-'])
    } else {
      expect(wrapper.text()).toContain('admin.ops.requestDetails.table.clientStatus: 502')
      expect(wrapper.text()).toContain('admin.ops.requestDetails.table.upstreamStatus: 403')
      expect(wrapper.text()).toContain('admin.ops.requestDetails.table.upstreamStatus: -')
    }
    wrapper.unmount()
  })
})
