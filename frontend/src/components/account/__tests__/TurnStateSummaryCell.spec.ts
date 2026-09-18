import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { ref } from 'vue'
import type { AccountListItem } from '@/types'
import TurnStateSummaryCell from '../TurnStateSummaryCell.vue'

vi.mock('vue-i18n', () => ({ useI18n: () => ({ locale: ref('zh-CN') }) }))
vi.mock('@/api/client', () => ({ apiClient: { get: vi.fn() } }))
const account = { id: 42, platform: 'openai', type: 'oauth', extra: { codex_turn_state_auto_enabled: true } } as AccountListItem
const now = Date.parse('2026-09-18T00:00:00Z')

describe('TurnStateSummaryCell', () => {
  it('keeps two compact rows, prioritizes astra and expires using server time', async () => {
    const wrapper = mount(TurnStateSummaryCell, { props: { account, now, summary: {
      enabled: true, configured: true, clockOffset: 60000, models: [
        { model: 'gpt-5.6-sol', state: 'preparing' },
        { model: 'gpt-6-astra', state: 'refreshing', length: 332, expires_at: '2026-09-18T00:03:00Z', last_error: 'refresh failed' },
        { model: 'another-model', state: 'queued' }
      ]
    } } })
    expect(wrapper.findAll('[data-testid="turn-state-summary-model"]')).toHaveLength(2)
    expect(wrapper.get('[data-testid="turn-state-summary-model"]').text()).toContain('gpt-6-astra')
    expect(wrapper.text()).toContain('可用2分')
    expect(wrapper.text()).toContain('+1')
    expect(wrapper.get('button').attributes('title')).toContain('gpt-5.6-sol')
    await wrapper.setProps({ now: now + 120000 })
    expect(wrapper.text()).toContain('已过期')
    expect(wrapper.text()).not.toContain('可用')
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted('open')).toHaveLength(1)
  })

  it('distinguishes invalidated state, read failure and disabled collection', async () => {
    const wrapper = mount(TurnStateSummaryCell, { props: { account, now, summary: {
      enabled: true, configured: true, clockOffset: 0, models: [{ model: 'gpt-6-astra', state: 'unavailable' }]
    } } })
    expect(wrapper.text()).toContain('待重试')
    await wrapper.setProps({ summary: { enabled: true, configured: true, clockOffset: 0, models: [], failed: true } })
    expect(wrapper.text()).toBe('读取失败')
    await wrapper.setProps({ account: { ...account, extra: {} } })
    expect(wrapper.text()).toBe('未开启')
  })
})
