import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import InvoiceProfilesPanel from '@/components/admin/invoice/InvoiceProfilesPanel.vue'
import type { AdminInvoiceProfile } from '@/api/admin/invoices'

const { listProfilesMock, resetProfileMock, showSuccessMock, showErrorMock } = vi.hoisted(() => ({
  listProfilesMock: vi.fn(),
  resetProfileMock: vi.fn(),
  showSuccessMock: vi.fn(),
  showErrorMock: vi.fn()
}))

vi.mock('@/api/admin/invoices', () => ({
  adminInvoiceAPI: {
    listProfiles: listProfilesMock,
    upsertProfile: vi.fn(),
    resetProfile: resetProfileMock
  }
}))

vi.mock('@/stores/app', () => ({
  useAppStore: () => ({
    showSuccess: showSuccessMock,
    showError: showErrorMock
  })
}))

vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-i18n')>()
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string) => key
    })
  }
})

const ConfirmDialogStub = {
  name: 'ConfirmDialog',
  props: ['show', 'title', 'message', 'confirmText', 'danger'],
  emits: ['confirm', 'cancel'],
  template: '<div v-if="show"><button data-testid="confirm-ok" @click="$emit(\'confirm\')" /></div>'
}

function profile(userId: number): AdminInvoiceProfile {
  return {
    user_id: userId,
    user_email: `u${userId}@example.com`,
    title: `Company ${userId}`,
    tax_no: '91350100M000100Y43',
    shared_user_count: 0,
    created_at: '2026-10-28T00:00:00Z',
    updated_at: '2026-10-28T00:00:00Z'
  }
}

function page(items: AdminInvoiceProfile[], total: number, pageNo: number) {
  return { items, total, page: pageNo, page_size: 20, pages: Math.max(1, Math.ceil(total / 20)) }
}

describe('InvoiceProfilesPanel', () => {
  beforeEach(() => {
    listProfilesMock.mockReset()
    resetProfileMock.mockReset()
    showSuccessMock.mockReset()
    showErrorMock.mockReset()
  })

  it('falls back to the last page after resetting the only row on the final page', async () => {
    const firstPage = Array.from({ length: 20 }, (_, i) => profile(i + 1))
    listProfilesMock
      .mockResolvedValueOnce(page(firstPage, 21, 1)) // 初次加载
      .mockResolvedValueOnce(page([profile(21)], 21, 2)) // 翻到第 2 页
      .mockResolvedValueOnce(page([], 20, 2)) // 重置后仍请求第 2 页：已越界
      .mockResolvedValueOnce(page(firstPage, 20, 1)) // 回退到第 1 页
    resetProfileMock.mockResolvedValue(undefined)

    const wrapper = mount(InvoiceProfilesPanel, {
      global: { stubs: { ConfirmDialog: ConfirmDialogStub, BaseDialog: true } }
    })
    await flushPromises()

    const next = wrapper.findAll('button').find((b) => b.text() === 'common.next')!
    await next.trigger('click')
    await flushPromises()
    expect(wrapper.findAll('tbody tr')).toHaveLength(1)

    const reset = wrapper.findAll('button').find((b) => b.text() === 'invoice.profile.reset')!
    await reset.trigger('click')
    await wrapper.find('[data-testid="confirm-ok"]').trigger('click')
    await flushPromises()

    expect(resetProfileMock).toHaveBeenCalledWith(21)
    expect(listProfilesMock).toHaveBeenCalledTimes(4)
    expect(listProfilesMock.mock.calls[3][0]).toMatchObject({ page: 1 })
    expect(wrapper.findAll('tbody tr')).toHaveLength(20)
    expect(wrapper.text()).not.toContain('invoice.profile.adminEmpty')
  })
})
