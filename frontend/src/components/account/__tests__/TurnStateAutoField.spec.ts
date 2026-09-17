import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import TurnStateAutoField from '../TurnStateAutoField.vue'

const { getStatus } = vi.hoisted(() => ({ getStatus: vi.fn() }))
vi.mock('@/api/client', () => ({ apiClient: { get: getStatus } }))
vi.mock('vue-i18n', () => ({ useI18n: () => ({ locale: ref('zh-CN') }) }))

beforeEach(() => { vi.useFakeTimers(); getStatus.mockReset() })
afterEach(() => { vi.useRealTimers() })

describe('TurnStateAutoField', () => {
  it('emits the account switch without mutating the saved state', async () => {
    getStatus.mockResolvedValue({ data: { configured: true, enabled: false, models: [] } })
    const wrapper = mount(TurnStateAutoField, { props: { accountId: 42, modelValue: false } })
    await flushPromises()
    await wrapper.get('input').setValue(true)
    expect(wrapper.emitted('update:modelValue')).toEqual([[true]])
    expect(wrapper.text()).toContain('生效状态：关闭')
    expect(getStatus).toHaveBeenCalledWith('/admin/accounts/42/turn-state', expect.anything())
    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(10000)
    expect(getStatus).toHaveBeenCalledTimes(1)
  })

  it('shows missing global configuration instead of pretending the feature is ready', async () => {
    getStatus.mockResolvedValue({ data: { configured: false, enabled: true, models: [] } })
    const wrapper = mount(TurnStateAutoField, { props: { accountId: 42, modelValue: true } })
    await flushPromises()
    expect(wrapper.text()).toContain('出口池尚未配置或配置无效')
    wrapper.unmount()
  })

  it('shows model status and expiry without displaying raw states', async () => {
    getStatus.mockResolvedValue({ data: { configured: true, enabled: true, models: [{ model: 'actual-model', state: 'ready', expires_at: '2026-09-17T16:00:00Z' }] } })
    const wrapper = mount(TurnStateAutoField, { props: { accountId: 42, modelValue: true } })
    await flushPromises()
    expect(wrapper.text()).toContain('actual-model')
    expect(wrapper.text()).toContain('可用')
    expect(wrapper.text()).toContain('到期：')
    wrapper.unmount()
  })
})
