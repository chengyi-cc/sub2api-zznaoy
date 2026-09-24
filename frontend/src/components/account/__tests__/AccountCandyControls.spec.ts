import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import AccountCandyControls from '../AccountCandyControls.vue'
vi.mock('vue-i18n', () => ({ useI18n: () => ({ t: (key: string) => key }) }))
describe('account candy controls', () => {
  it('uses plain red text for 29 and restores ordinary text for the next valid answer', async () => {
    const state = { account_id: 1, enabled: true, history: [], use_defaults: false, model_id: 'custom', interval_minutes: 17, last_valid_answer: 29, last_valid_at: null }
    const wrapper = mount(AccountCandyControls, { props: { state, pending: false, schedulerEnabled: true, loadError: false } })
    expect(wrapper.get('[data-testid="candy-test"]').classes()).toContain('text-red-600')
    expect(wrapper.find('svg').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('🍬')
    expect(wrapper.get('[data-testid="history-bars"]').findAll('[data-verdict]')).toHaveLength(10)
    await wrapper.get('[data-testid="candy-test"]').trigger('click')
    expect(wrapper.emitted('test')).toHaveLength(1)
    await wrapper.setProps({ state: { ...state, last_valid_answer: 21 } })
    expect(wrapper.get('[data-testid="candy-test"]').classes()).not.toContain('text-red-600')
    expect(wrapper.find('[data-testid="candy-auto"]').exists()).toBe(false)
    await wrapper.get('[data-testid="candy-pause"]').trigger('click')
    expect(wrapper.emitted('toggle')).toEqual([[false]])
    expect(wrapper.find('[data-testid="candy-pause"]').exists()).toBe(true)
    await wrapper.setProps({ pending: true })
    expect(wrapper.get('[data-testid="candy-pause"]').attributes('disabled')).toBeDefined()
    await wrapper.setProps({ pending: false, state: { ...state, enabled: false } })
    expect(wrapper.find('button').exists()).toBe(false)
    expect(wrapper.find('[data-testid="history-bars"]').exists()).toBe(false)
    expect(wrapper.text()).toBe('')
    await wrapper.get('[data-testid="candy-auto"]').setValue(true)
    expect(wrapper.emitted('toggle')).toEqual([[false], [true]])
    expect((wrapper.get('[data-testid="candy-auto"]').element as HTMLInputElement).checked).toBe(false)
    wrapper.unmount()
  })
  it('shows only a disabled switch when the state cannot be loaded', () => {
    const wrapper = mount(AccountCandyControls, { props: { pending: false, schedulerEnabled: true, loadError: true } })
    expect(wrapper.get('[data-testid="candy-auto"]').attributes('disabled')).toBeDefined()
    expect(wrapper.find('button').exists()).toBe(false)
    expect(wrapper.text()).toBe('')
    wrapper.unmount()
  })
  it('keeps the compact pause action and red answer while the account is unavailable', async () => {
    const state = { account_id: 1, enabled: true, history: [], use_defaults: true, model_id: 'gpt-6-astra', interval_minutes: 60, last_valid_answer: 29, last_valid_at: null, blocked_reason: 'expired' }
    const wrapper = mount(AccountCandyControls, { props: { state, pending: false, schedulerEnabled: true, loadError: false } })
    expect(wrapper.text()).toContain('admin.accounts.candyMonitor.blocked.expired')
    expect(wrapper.get('[data-testid="candy-test"]').classes()).toContain('text-red-600')
    expect(wrapper.find('[role="switch"]').exists()).toBe(false)
    await wrapper.get('[data-testid="candy-pause"]').trigger('click')
    expect(wrapper.emitted('toggle')).toEqual([[false]])
    // A failed write leaves server state enabled and still allows retrying.
    expect(wrapper.find('[data-testid="candy-pause"]').exists()).toBe(true)
    await wrapper.setProps({ loadError: true })
    expect(wrapper.get('[data-testid="candy-pause"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
})
