import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import AccountCandyControls from '../AccountCandyControls.vue'
vi.mock('vue-i18n', () => ({ useI18n: () => ({ t: (key: string) => key }) }))
describe('account candy controls', () => {
  it('uses plain red text for 29 and restores ordinary text for the next valid answer', async () => {
    const state = { account_id: 1, enabled: true, use_defaults: false, model_id: 'custom', interval_minutes: 17, last_valid_answer: 29, last_valid_at: null }
    const wrapper = mount(AccountCandyControls, { props: { state, pending: false, schedulerEnabled: true, loadError: false } })
    expect(wrapper.get('[data-testid="candy-test"]').classes()).toContain('text-red-600')
    expect(wrapper.find('svg').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('🍬')
    await wrapper.setProps({ state: { ...state, last_valid_answer: 21 } })
    expect(wrapper.get('[data-testid="candy-test"]').classes()).not.toContain('text-red-600')
    await wrapper.get('[data-testid="candy-auto"]').setValue(false)
    expect(wrapper.emitted('toggle')).toEqual([[false]])
    expect((wrapper.get('[data-testid="candy-auto"]').element as HTMLInputElement).checked).toBe(true)
    await wrapper.setProps({ pending: true })
    expect(wrapper.get('[data-testid="candy-auto"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
  })
  it('disables unknown or failed state reads without disabling manual tests', async () => {
    const wrapper = mount(AccountCandyControls, { props: { pending: false, schedulerEnabled: true, loadError: true } })
    expect(wrapper.get('[data-testid="candy-auto"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-testid="candy-test"]').trigger('click')
    expect(wrapper.emitted('test')).toHaveLength(1)
    wrapper.unmount()
  })
})
