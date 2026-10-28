import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ProfileInvoiceCard from '@/components/user/profile/ProfileInvoiceCard.vue'

const { getInvoiceProfileMock, bindInvoiceProfileMock, showSuccessMock, showErrorMock } = vi.hoisted(() => ({
  getInvoiceProfileMock: vi.fn(),
  bindInvoiceProfileMock: vi.fn(),
  showSuccessMock: vi.fn(),
  showErrorMock: vi.fn()
}))

vi.mock('@/api/invoice', () => ({
  invoiceAPI: {
    getInvoiceProfile: getInvoiceProfileMock,
    bindInvoiceProfile: bindInvoiceProfileMock
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
  props: ['show', 'title', 'message', 'confirmText'],
  emits: ['confirm', 'cancel'],
  template: '<div v-if="show" data-testid="confirm"><button data-testid="confirm-ok" @click="$emit(\'confirm\')" /></div>'
}

function mountCard() {
  return mount(ProfileInvoiceCard, {
    global: { stubs: { ConfirmDialog: ConfirmDialogStub } }
  })
}

describe('ProfileInvoiceCard', () => {
  beforeEach(() => {
    getInvoiceProfileMock.mockReset()
    bindInvoiceProfileMock.mockReset()
    showSuccessMock.mockReset()
    showErrorMock.mockReset()
  })

  it('shows the bound profile read-only without a form', async () => {
    getInvoiceProfileMock.mockResolvedValue({
      title: 'Acme Inc.',
      tax_no: '91350100M000100Y43',
      locked: true,
      admin_modified: false,
      created_at: '2026-10-28T00:00:00Z',
      updated_at: '2026-10-28T00:00:00Z'
    })
    const wrapper = mountCard()
    await flushPromises()

    expect(wrapper.find('[data-testid="invoice-profile-locked"]').text()).toContain('Acme Inc.')
    expect(wrapper.find('form').exists()).toBe(false)
    expect(wrapper.findAll('input')).toHaveLength(0)
  })

  it('blocks binding until the tax number passes the checksum', async () => {
    getInvoiceProfileMock.mockResolvedValue(null)
    const wrapper = mountCard()
    await flushPromises()

    const [titleInput, taxInput] = wrapper.findAll('input')
    await titleInput.setValue('Acme Inc.')
    await taxInput.setValue('91350100M000100Y44')
    expect(wrapper.text()).toContain('invoice.profile.taxNoInvalid')
    expect(wrapper.find('button[type="submit"]').attributes('disabled')).toBeDefined()

    await taxInput.setValue('91350100m000100y43')
    expect(wrapper.text()).not.toContain('invoice.profile.taxNoInvalid')
    expect(wrapper.find('button[type="submit"]').attributes('disabled')).toBeUndefined()
  })

  it('binds after confirmation with normalised values and then locks', async () => {
    getInvoiceProfileMock.mockResolvedValue(null)
    bindInvoiceProfileMock.mockResolvedValue({
      title: 'Acme Inc.',
      tax_no: '91350100M000100Y43',
      locked: true,
      admin_modified: false,
      created_at: '2026-10-28T00:00:00Z',
      updated_at: '2026-10-28T00:00:00Z'
    })
    const wrapper = mountCard()
    await flushPromises()

    const [titleInput, taxInput] = wrapper.findAll('input')
    await titleInput.setValue('  Acme Inc. ')
    await taxInput.setValue(' 91350100m000100y43 ')
    await wrapper.find('form').trigger('submit')
    expect(bindInvoiceProfileMock).not.toHaveBeenCalled()

    await wrapper.find('[data-testid="confirm-ok"]').trigger('click')
    await flushPromises()

    expect(bindInvoiceProfileMock).toHaveBeenCalledWith({ title: 'Acme Inc.', tax_no: '91350100M000100Y43' })
    expect(showSuccessMock).toHaveBeenCalled()
    expect(wrapper.find('[data-testid="invoice-profile-locked"]').exists()).toBe(true)
  })
})
