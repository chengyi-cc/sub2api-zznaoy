import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import AccountTodayStatsCell from '../AccountTodayStatsCell.vue'
import type { WindowStats } from '@/types'

vi.mock('vue-i18n', async (importOriginal) => ({
  ...await importOriginal<typeof import('vue-i18n')>(),
  useI18n: () => ({ t: (key: string) => key })
}))

const today: WindowStats = { requests: 7, tokens: 1200, cost: 1.25, user_cost: 0.75 }

describe('AccountTodayStatsCell lifetime totals', () => {
  it('adds lifetime totals while retaining today requests and both billing views', () => {
    const wrapper = mount(AccountTodayStatsCell, { props: { stats: { ...today, lifetime_tokens: 8000000, lifetime_cost: 1904.56 } } })
    expect(wrapper.get('[data-test="lifetime-tokens"]').text()).toBe('8.00M')
    expect(wrapper.get('[data-test="lifetime-cost"]').text()).toContain('1,904.56')
    expect(wrapper.text()).toContain('admin.accounts.stats.requests')
    expect(wrapper.text()).toContain('1.2K')
    expect(wrapper.text()).toContain('usage.accountBilled')
    expect(wrapper.text()).toContain('usage.userBilled')
    expect(wrapper.text()).toContain('0.75')
  })

  it('shows unavailable totals as a dash for older servers or failed lifetime queries', () => {
    const wrapper = mount(AccountTodayStatsCell, { props: { stats: today } })
    expect(wrapper.get('[data-test="lifetime-tokens"]').text()).toBe('—')
    expect(wrapper.get('[data-test="lifetime-cost"]').text()).toBe('—')
    expect(wrapper.text()).toContain('1.2K')
  })

  it('distinguishes successful zero totals from unavailable values', () => {
    const wrapper = mount(AccountTodayStatsCell, { props: { stats: { ...today, lifetime_tokens: 0, lifetime_cost: 0 } } })
    expect(wrapper.get('[data-test="lifetime-tokens"]').text()).toBe('0')
    expect(wrapper.get('[data-test="lifetime-cost"]').text()).not.toBe('—')
  })

  it('supports explicit null totals and preserves loading and error states', () => {
    const wrapper = mount(AccountTodayStatsCell, { props: { stats: { ...today, lifetime_tokens: null, lifetime_cost: null } } })
    expect(wrapper.get('[data-test="lifetime-tokens"]').text()).toBe('—')
    const loading = mount(AccountTodayStatsCell, { props: { loading: true } })
    expect(loading.find('[data-test="lifetime-tokens"]').exists()).toBe(false)
    const error = mount(AccountTodayStatsCell, { props: { error: 'Statistics unavailable' } })
    expect(error.text()).toBe('Statistics unavailable')
  })
})
