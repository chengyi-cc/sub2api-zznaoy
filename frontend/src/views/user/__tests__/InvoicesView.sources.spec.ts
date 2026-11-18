import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import InvoicesView from '../InvoicesView.vue'

const { listInvoiceRequests, getInvoiceRequest } = vi.hoisted(() => ({
  listInvoiceRequests: vi.fn(),
  getInvoiceRequest: vi.fn(),
}))

vi.mock('@/api/invoice', () => ({
  invoiceAPI: {
    listInvoiceRequests,
    getInvoiceRequest,
    listEligibleSources: vi.fn(),
    getInvoiceProfile: vi.fn(),
    createInvoiceRequest: vi.fn(),
    downloadInvoice: vi.fn(),
  },
}))

vi.mock('@/stores/app', () => ({
  useAppStore: () => ({
    cachedPublicSettings: {},
    showError: vi.fn(),
    showSuccess: vi.fn(),
  }),
}))

vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-i18n')>()
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, fallback?: unknown) => (typeof fallback === 'string' ? fallback : key),
    }),
  }
})

const issued = {
  id: 7,
  user_id: 1,
  payment_order_ids: [11, 12],
  redeem_code_ids: [21],
  amount: 150,
  invoice_type: 'company',
  title: 'Acme Inc.',
  status: 'issued',
  invoice_no: 'INV-2026-0007',
  has_file: true,
  issued_at: '2026-11-10T08:00:00Z',
  created_at: '2026-11-09T08:00:00Z',
  updated_at: '2026-11-10T08:00:00Z',
}

const mountView = () =>
  mount(InvoicesView, {
    global: {
      stubs: {
        AppLayout: { template: '<main><slot /></main>' },
        BaseDialog: {
          props: ['show', 'title'],
          template: '<section v-if="show" data-test="dialog"><slot /><slot name="footer" /></section>',
        },
      },
    },
  })

describe('user InvoicesView sources detail', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    listInvoiceRequests.mockResolvedValue({ items: [issued], total: 1, page: 1, page_size: 20, pages: 1 })
  })

  it('shows the orders and redeem codes linked to an issued invoice', async () => {
    getInvoiceRequest.mockResolvedValue({
      ...issued,
      orders: [
        { order_id: 11, out_trade_no: 'OUT-AAA', amount: 60, order_type: 'balance', payment_type: 'alipay', completed_at: '2026-11-01T08:00:00Z' },
        { order_id: 12, out_trade_no: 'OUT-BBB', amount: 50, order_type: 'balance', payment_type: 'wxpay', completed_at: '2026-11-02T08:00:00Z' },
      ],
      redeem_codes: [{ redeem_code_id: 21, code: 'BAL-CODE-1234', value: 40, used_at: '2026-11-03T08:00:00Z' }],
    })
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-test="invoice-sources-7"]').trigger('click')
    await flushPromises()

    expect(getInvoiceRequest).toHaveBeenCalledWith(7)
    const detail = wrapper.get('[data-test="invoice-sources-detail"]')
    expect(detail.text()).toContain('INV-2026-0007')
    expect(detail.text()).toContain('OUT-AAA')
    expect(detail.text()).toContain('OUT-BBB')
    expect(detail.text()).toContain('BAL-CODE-1234')
    expect(detail.text()).toContain('¥150.00')
    wrapper.unmount()
  })

  it('shows a dash instead of year 0001 for missing source records', async () => {
    getInvoiceRequest.mockResolvedValue({
      ...issued,
      orders: [{ order_id: 11, out_trade_no: '', amount: 0, order_type: '', payment_type: '', completed_at: '0001-01-01T00:00:00Z' }],
      redeem_codes: [],
    })
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-test="invoice-sources-7"]').trigger('click')
    await flushPromises()

    const detail = wrapper.get('[data-test="invoice-sources-detail"]')
    expect(detail.text()).toContain('#11')
    expect(detail.text()).not.toContain('0001')
    wrapper.unmount()
  })

  it('reports a load failure inside the dialog', async () => {
    getInvoiceRequest.mockRejectedValue(new Error('boom'))
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-test="invoice-sources-7"]').trigger('click')
    await flushPromises()

    expect(wrapper.find('[data-test="invoice-sources-detail"]').exists()).toBe(false)
    expect(wrapper.get('[data-test="dialog"]').text()).not.toBe('')
    wrapper.unmount()
  })
})
