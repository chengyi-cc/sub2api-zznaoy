import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import InvoicesView from '../InvoicesView.vue'

const { list, reject } = vi.hoisted(() => ({
  list: vi.fn(),
  reject: vi.fn(),
}))

vi.mock('@/api/admin/invoices', () => ({
  adminInvoiceAPI: {
    list,
    reject,
    detail: vi.fn(),
    approve: vi.fn(),
    batchApprove: vi.fn(),
    issue: vi.fn(),
    download: vi.fn(),
    exportApproved: vi.fn(),
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
      t: (key: string) => key,
    }),
  }
})

const approved = {
  id: 9,
  user_id: 3,
  payment_order_ids: [31],
  redeem_code_ids: [],
  amount: 300,
  invoice_type: 'company',
  title: 'Acme Inc.',
  tax_no: 'BAD-TAX-NO',
  status: 'approved',
  has_file: false,
  created_at: '2026-11-09T08:00:00Z',
  updated_at: '2026-11-10T08:00:00Z',
}

const mountView = () =>
  mount(InvoicesView, {
    global: {
      stubs: {
        AppLayout: { template: '<main><slot /></main>' },
        InvoiceProfilesPanel: true,
        BaseDialog: {
          props: ['show', 'title'],
          template: '<section v-if="show" data-test="dialog"><slot /><slot name="footer" /></section>',
        },
      },
    },
  })

describe('admin InvoicesView reject approved request', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    list.mockResolvedValue({ items: [approved], total: 1, page: 1, page_size: 20, pages: 1 })
    reject.mockResolvedValue({ ...approved, status: 'rejected' })
  })

  it('lets admins reject an approved-but-not-issued request from the row', async () => {
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[data-test="invoice-reject-9"]').trigger('click')
    const dialog = wrapper.get('[data-test="dialog"]')
    await dialog.get('textarea').setValue('税号查询不到，请核对后重新申请')
    const confirm = dialog.findAll('button').find((b) => b.text() === 'invoice.confirmReject')
    expect(confirm).toBeTruthy()
    await confirm!.trigger('click')
    await flushPromises()

    expect(reject).toHaveBeenCalledWith(9, '税号查询不到，请核对后重新申请')
    expect(list).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })
})
