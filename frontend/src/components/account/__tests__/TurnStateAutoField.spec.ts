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
    expect(wrapper.text()).toContain('所选采集服务尚未配置或配置无效')
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

  it('defaults to Team and purchased proxies and allows manual legacy selections', async () => {
    getStatus.mockResolvedValue({ data: { configured: true, enabled: true, models: [] } })
    const wrapper = mount(TurnStateAutoField, { props: { accountId: 42, modelValue: true } })
    await flushPromises()
    expect((wrapper.get('[data-testid="turn-state-profile"]').element as HTMLSelectElement).value).toBe('team')
    expect((wrapper.get('[data-testid="turn-state-source"]').element as HTMLSelectElement).value).toBe('purchased')
    await wrapper.get('[data-testid="turn-state-profile"]').setValue('pro')
    await wrapper.get('[data-testid="turn-state-source"]').setValue('ipv6_pool')
    expect(wrapper.emitted('update:profile')).toEqual([['pro']])
    expect(wrapper.emitted('update:source')).toEqual([['ipv6_pool']])
    wrapper.unmount()
  })

  it('counts down using server time and loads history only when requested', async () => {
    vi.setSystemTime(new Date('2026-09-18T01:00:00Z'))
    const snapshot = { configured: true, enabled: true, server_time: '2026-09-18T02:00:00Z', models: [{ model: 'model-a', state: 'refreshing', expires_at: '2026-09-18T02:30:00Z', refresh_at: '2026-09-18T02:00:00Z' }] }
    getStatus.mockResolvedValue({ data: snapshot })
    const wrapper = mount(TurnStateAutoField, { props: { accountId: 42, modelValue: true } })
    await flushPromises()
    expect(wrapper.get('[data-testid="turn-state-ttl"]').text()).toBe('30:00')
    await vi.advanceTimersByTimeAsync(1000)
    expect(wrapper.get('[data-testid="turn-state-ttl"]').text()).toBe('29:59')
    expect(getStatus).toHaveBeenCalledTimes(1)
    getStatus.mockResolvedValue({ data: { ...snapshot, history: [{ at: '2026-09-18T01:59:00Z', model: 'model-a', profile: 'team', source: 'purchased', country: 'JP', source_ip: '198.51.100.1', status: 200, length: 356, accepted: false, duration_ms: 200, error: 'candidate rejected', value: 'private-state-not-for-ui' }] } })
    await wrapper.get('[data-testid="turn-state-history"]').trigger('click')
    await flushPromises()
    expect(getStatus).toHaveBeenLastCalledWith('/admin/accounts/42/turn-state?history=1', expect.anything())
    expect(wrapper.text()).toContain('日本')
    expect(wrapper.text()).toContain('356')
    expect(wrapper.text()).toContain('candidate rejected')
    expect(wrapper.text()).not.toContain('private-state-not-for-ui')
    wrapper.unmount()
  })
})
