import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CandyMonitorView from '../CandyMonitorView.vue'
import type { CandyResult } from '@/api/admin/candyMonitor'

const api = vi.hoisted(() => ({ settings: vi.fn(), list: vi.fn(), saveSettings: vi.fn(), configure: vi.fn(), setEnabled: vi.fn(), history: vi.fn() }))
vi.mock('@/api/admin/candyMonitor', () => ({ candyMonitorAPI: api, CANDY_DEFAULT_MODEL: 'gpt-6-astra' }))
vi.mock('@/api/admin/groups', () => ({ default: {}, getAllIncludingInactive: async () => [{ id: 8, name: 'Group eight' }] }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) }
})
const defaults = { enabled: true, model_id: 'gpt-6-astra', interval_minutes: 60, max_results: 50 }
const account = { account_id: 42, name: 'Custom account', platform: 'openai', type: 'oauth', status: 'active', enabled: true, use_defaults: false, model_id: 'custom-model', interval_minutes: 17, latest: null, history: [], total_tests: 100, answer_21_count: 75, answer_29_count: 20, other_answer_count: 3, inconclusive_count: 2 }
function setup() {
  return mount(CandyMonitorView, { global: { stubs: {
    AppLayout: { template: '<div><slot /></div>' },
    BaseDialog: { props: ['show'], template: '<div v-if="show"><slot /><slot name="footer" /></div>' },
    Pagination: true, CandyMonitorRunDialog: true,
    Select: { props: ['modelValue', 'options'], emits: ['update:modelValue', 'change'], template: '<select :value="modelValue" @change="$emit(\'update:modelValue\', $event.target.value); $emit(\'change\')"><option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option></select>' }
  } } })
}
describe('CandyMonitorView', () => {
  beforeEach(() => {
    vi.useFakeTimers(); vi.clearAllMocks()
    api.settings.mockResolvedValue(defaults)
    api.list.mockResolvedValue({ items: [account], total: 1 })
    api.saveSettings.mockImplementation(async value => value)
    api.configure.mockResolvedValue(undefined); api.setEnabled.mockResolvedValue(undefined)
    api.history.mockResolvedValue([])
  })
  afterEach(() => vi.useRealTimers())
  it('explains skipped unavailable accounts without showing a scheduled time or changing history', async () => {
    api.list.mockResolvedValue({ items: [{ ...account, blocked_reason: 'account_error', next_run_at: '2026-09-24T00:00:00Z' }], total: 1 })
    const wrapper = setup(); await flushPromises()
    expect(wrapper.text()).toContain('admin.accounts.candyMonitor.blocked.account_error')
    expect(wrapper.get('[data-testid="next-run"]').text()).toBe('—')
    expect(wrapper.get('[data-testid="next-run"]').attributes('title')).toBe('admin.accounts.candyMonitor.blockedHint')
    expect(wrapper.get('[data-testid="total-tests"]').text()).toBe('100')
    expect(wrapper.get('[data-testid="history-bars"]').findAll('[data-verdict="empty"]')).toHaveLength(10)
    api.list.mockResolvedValue({ items: [{ ...account, blocked_reason: '' }], total: 1 })
    await vi.advanceTimersByTimeAsync(10000); await flushPromises()
    expect(wrapper.text()).not.toContain('admin.accounts.candyMonitor.blocked.account_error')
    wrapper.unmount()
  })
  it('filters by group and applies the saved template to selected accounts', async () => {
    const wrapper = setup(); await flushPromises()
    expect(wrapper.get('[data-testid="count-21"]').text()).toBe('75')
    expect(wrapper.get('[data-testid="count-29"]').text()).toBe('20')
    expect(wrapper.get('[data-testid="total-tests"]').text()).toBe('100')
    await wrapper.get('[data-testid="group-filter"]').setValue('8'); await flushPromises()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ group_id: 8, page: 1 }))
    await wrapper.get('input[aria-label="Custom account"]').setValue(true)
    await wrapper.get('[data-testid="apply-defaults"]').trigger('click'); await flushPromises()
    expect(api.configure).toHaveBeenCalledWith([42], { enabled: true, use_defaults: true, model_id: 'gpt-6-astra', interval_minutes: 60 })
    wrapper.unmount()
  })
  it('batch pause preserves overrides instead of rewriting them', async () => {
    const wrapper = setup(); await flushPromises()
    await wrapper.get('input[aria-label="Custom account"]').setValue(true)
    await wrapper.get('[data-testid="pause-selected"]').trigger('click'); await flushPromises()
    expect(api.setEnabled).toHaveBeenCalledTimes(1)
    expect(api.setEnabled).toHaveBeenCalledWith([42], false)
    expect(api.configure).not.toHaveBeenCalled()
    wrapper.unmount()
  })
  it('saves per-account model and interval separately from the template', async () => {
    const wrapper = setup(); await flushPromises()
    await wrapper.findAll('button').find(b => b.text().endsWith('.configure'))!.trigger('click')
    const form = wrapper.get('#candy-account-settings')
    await form.get('input[maxlength="200"]').setValue('another-text-model')
    await form.get('input[type="number"]').setValue(25)
    await form.trigger('submit'); await flushPromises()
    expect(api.configure).toHaveBeenCalledWith([42], { enabled: true, use_defaults: false, model_id: 'another-text-model', interval_minutes: 25 })
    expect(api.saveSettings).not.toHaveBeenCalled()
    wrapper.unmount()
  })
  it('saves the default template and stops polling when unmounted', async () => {
    const wrapper = setup(); await flushPromises()
    expect((wrapper.get('[data-testid="template-form"]').element as HTMLElement).style.display).toBe('none')
    await wrapper.get('[data-testid="toggle-template"]').trigger('click')
    expect((wrapper.get('[data-testid="template-form"]').element as HTMLElement).style.display).not.toBe('none')
    await wrapper.get('[data-testid="default-interval"]').setValue(120)
    await wrapper.find('form').trigger('submit'); await flushPromises()
    expect(api.saveSettings).toHaveBeenCalledWith({ ...defaults, interval_minutes: 120 })
    const before = api.list.mock.calls.length
    wrapper.unmount(); await vi.advanceTimersByTimeAsync(30000)
    expect(api.list).toHaveBeenCalledTimes(before)
  })
  it('combines account and monitor filters, including explicitly paused accounts, and resets them', async () => {
    const wrapper = setup(); await flushPromises()
    for (const [id, value] of [['platform', 'openai'], ['type', 'oauth'], ['status', 'rate_limited'], ['group', '8'], ['monitor-status', 'paused'], ['verdict', 'incorrect'], ['privacy', 'training_off']]) {
      await wrapper.get(`[data-testid="${id}-filter"]`).setValue(value)
    }
    await flushPromises()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ platform: 'openai', type: 'oauth', status: 'rate_limited', group_id: 8, enabled: false, verdict: 'incorrect', privacy_mode: 'training_off', page: 1 }))
    await wrapper.get('[data-testid="monitor-status-filter"]').setValue('enabled'); await flushPromises()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: true }))
    await wrapper.get('[data-testid="group-filter"]').setValue('ungrouped'); await flushPromises()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ group_id: undefined, ungrouped: true }))
    await wrapper.get('[data-testid="reset-filters"]').trigger('click'); await flushPromises()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ platform: undefined, type: undefined, status: undefined, group_id: undefined, enabled: undefined, verdict: undefined, privacy_mode: undefined, ungrouped: undefined }))
    wrapper.unmount()
  })
  it('debounces search, clears selections when filters change, and ignores stale results', async () => {
    const wrapper = setup(); await flushPromises()
    await wrapper.get('input[aria-label="Custom account"]').setValue(true)
    const before = api.list.mock.calls.length
    let resolve!: (value: unknown) => void
    api.list.mockImplementationOnce(() => new Promise(r => { resolve = r }))
    await wrapper.get('[data-testid="account-search"]').setValue('older')
    await vi.advanceTimersByTimeAsync(300)
    expect(api.list).toHaveBeenCalledTimes(before + 1)
    await wrapper.get('[data-testid="account-search"]').setValue('newer')
    await vi.advanceTimersByTimeAsync(300); await flushPromises()
    resolve({ items: [{ ...account, name: 'Stale account' }], total: 1 }); await flushPromises()
    expect(wrapper.text()).not.toContain('Stale account')
    expect(wrapper.get('[data-testid="apply-defaults"]').attributes('disabled')).toBeDefined()
    expect(api.list).toHaveBeenLastCalledWith(expect.objectContaining({ search: 'newer', page: 1 }))
    wrapper.unmount()
  })
  it('renders ten chronological bars, keeps empty positions neutral, and opens history', async () => {
    const result = (id: number, verdict: CandyResult['verdict'], actual?: number): CandyResult => ({ id, verdict, actual, account_id: 42, model_id: 'gpt-6-astra', source: 'manual', reason: '', expected: 21, duration_ms: 1000, started_at: `2026-09-24T00:0${id}:00Z` })
    api.list.mockResolvedValue({ items: [{ ...account, latest: result(3, 'incorrect', 29), history: [result(3, 'incorrect', 29), result(2, 'inconclusive'), result(1, 'pass', 21)] }], total: 1 })
    const wrapper = setup(); await flushPromises()
    const dots = wrapper.get('[data-testid="history-bars"]').findAll('[data-verdict]')
    expect(dots).toHaveLength(10)
    expect(wrapper.get('[data-testid="normal-rate"]').text()).toBe('50.0%')
    expect(dots.every(dot => !dot.classes().includes('rounded-full'))).toBe(true)
    expect(dots.map(dot => dot.attributes('data-verdict'))).toEqual([...Array(7).fill('empty'), 'pass', 'inconclusive', 'incorrect'])
    expect(dots[7].classes()).toContain('bg-emerald-500')
    expect(dots[8].classes()).toContain('bg-amber-400')
    expect(dots[9].classes()).toContain('bg-red-500')
    expect(wrapper.get('[data-testid="latest-status"] [data-verdict]').classes()).toContain('bg-red-500')
    const buttons = wrapper.get('[data-testid="history-bars"]').findAll('button')
    expect(buttons[0].attributes('disabled')).toBeDefined()
    expect(buttons[9].attributes('title')).toContain('29')
    await buttons[9].trigger('click'); await flushPromises()
    expect(api.history).toHaveBeenCalledWith(42)
    wrapper.unmount()
  })
  it('updates bars after automatic refresh and limits history to ten completed results', async () => {
    const results = Array.from({ length: 12 }, (_, index) => ({ id: 12 - index, verdict: 'pass', actual: 21, started_at: '2026-09-24T00:00:00Z' }))
    const wrapper = setup(); await flushPromises()
    api.list.mockResolvedValue({ items: [{ ...account, latest: results[0], history: [{ id: 13, verdict: 'running' }, ...results] }], total: 1 })
    await vi.advanceTimersByTimeAsync(10000); await flushPromises()
    expect(wrapper.get('[data-testid="history-bars"]').findAll('[data-verdict="pass"]')).toHaveLength(10)
    expect(wrapper.get('[data-testid="latest-status"] [data-verdict]').classes()).toContain('bg-emerald-500')
    wrapper.unmount()
  })
})
