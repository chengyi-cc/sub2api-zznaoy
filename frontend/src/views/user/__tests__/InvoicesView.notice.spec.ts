import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import InvoicesView from '../InvoicesView.vue'

const { publicSettings } = vi.hoisted(() => ({
  publicSettings: { value: {} as Record<string, unknown> },
}))

vi.mock('@/api/invoice', () => ({
  invoiceAPI: {
    listInvoiceRequests: vi.fn().mockResolvedValue({ items: [], total: 0, page: 1, page_size: 20, pages: 1 }),
    getInvoiceRequest: vi.fn(),
    listEligibleSources: vi.fn(),
    getInvoiceProfile: vi.fn(),
    createInvoiceRequest: vi.fn(),
    downloadInvoice: vi.fn(),
  },
}))

vi.mock('@/stores/app', () => ({
  useAppStore: () => ({
    get cachedPublicSettings() {
      return publicSettings.value
    },
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

const mountView = () =>
  mount(InvoicesView, {
    global: {
      stubs: {
        AppLayout: { template: '<main><slot /></main>' },
        BaseDialog: { template: '<section><slot /></section>' },
      },
    },
  })

describe('user InvoicesView notice', () => {
  beforeEach(() => {
    publicSettings.value = {}
  })

  it('shows the admin-configured invoice notice with line breaks kept', async () => {
    publicSettings.value = { invoice_notice: '  发票每周四开具。\n暂不支持加急。  ' }
    const wrapper = mountView()
    await flushPromises()

    const notice = wrapper.get('[data-test="invoice-notice"]')
    expect(notice.text()).toContain('发票每周四开具。')
    expect(notice.find('p').element.textContent).toBe('发票每周四开具。\n暂不支持加急。')
    wrapper.unmount()
  })

  it('hides the notice when it is empty or whitespace', async () => {
    publicSettings.value = { invoice_notice: '   ' }
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.find('[data-test="invoice-notice"]').exists()).toBe(false)
    wrapper.unmount()
  })
})
