import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import type { Account } from '@/types'
import TurnStateAccountModal from '../TurnStateAccountModal.vue'

const { getById, update, refreshStatus } = vi.hoisted(() => ({ getById: vi.fn(), update: vi.fn(), refreshStatus: vi.fn() }))
vi.mock('@/api/admin', () => ({ adminAPI: { accounts: { getById, update } } }))
vi.mock('@/api/client', () => ({ apiClient: { get: vi.fn() } }))
vi.mock('vue-i18n', () => ({ useI18n: () => ({ locale: ref('zh-CN') }) }))
const account = { id: 42, name: 'account', extra: { codex_turn_state_auto_enabled: true, untouched: 'old' } } as unknown as Account
const mountModal = () => mount(TurnStateAccountModal, { props: { account }, global: { stubs: {
  BaseDialog: { template: '<div><slot /></div>' },
  TurnStateAutoField: { name: 'TurnStateAutoField', template: '<div />', methods: { refreshStatus } }
} } })
beforeEach(() => { vi.clearAllMocks() })

describe('TurnStateAccountModal', () => {
  it('saves only acquisition settings while preserving fresh account extras and keeps the panel open', async () => {
    getById.mockResolvedValue({ ...account, extra: { untouched: 'new', quota_daily_limit: 10 } })
    update.mockResolvedValue(account)
    const wrapper = mountModal()
    wrapper.getComponent({ name: 'TurnStateAutoField' }).vm.$emit('update:profile', 'pro')
    await wrapper.get('[data-testid="turn-state-save-account"]').trigger('click')
    await flushPromises()
    expect(update).toHaveBeenCalledWith(42, { extra: { untouched: 'new', quota_daily_limit: 10, codex_turn_state_auto_enabled: true, codex_turn_state_profile: 'pro', codex_turn_state_source: 'purchased' } })
    expect(wrapper.emitted('updated')).toEqual([[account]])
    expect(wrapper.emitted('close')).toBeUndefined()
    expect(refreshStatus).toHaveBeenCalledOnce()
    wrapper.unmount()
  })

  it('does not save with stale extras if refreshing the account fails', async () => {
    getById.mockRejectedValue(new Error('unavailable'))
    const wrapper = mountModal()
    await wrapper.get('[data-testid="turn-state-save-account"]').trigger('click')
    await flushPromises()
    expect(update).not.toHaveBeenCalled()
    expect(wrapper.find('[role="alert"]').exists()).toBe(true)
    wrapper.unmount()
  })
})
