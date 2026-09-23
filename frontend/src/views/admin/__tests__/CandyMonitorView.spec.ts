import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CandyMonitorView from '../CandyMonitorView.vue'

const api = vi.hoisted(() => ({ settings: vi.fn(), list: vi.fn(), saveSettings: vi.fn(), configure: vi.fn(), setEnabled: vi.fn(), history: vi.fn() }))
vi.mock('@/api/admin/candyMonitor', () => ({ candyMonitorAPI: api, CANDY_DEFAULT_MODEL: 'gpt-6-astra' }))
vi.mock('@/api/admin/groups', () => ({ default: {}, getAllIncludingInactive: async () => [{ id: 8, name: 'Group eight' }] }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) }
})
const defaults = { enabled: true, model_id: 'gpt-6-astra', interval_minutes: 60, max_results: 50 }
const account = { account_id: 42, name: 'Custom account', platform: 'openai', status: 'active', enabled: true, use_defaults: false, model_id: 'custom-model', interval_minutes: 17, latest: null }
function setup() {
  return mount(CandyMonitorView, { global: { stubs: {
    AppLayout: { template: '<div><slot /></div>' },
    BaseDialog: { props: ['show'], template: '<div v-if="show"><slot /><slot name="footer" /></div>' },
    Pagination: true, CandyMonitorRunDialog: true
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
  it('filters by group and applies the saved template to selected accounts', async () => {
    const wrapper = setup(); await flushPromises()
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
    await wrapper.get('[data-testid="default-interval"]').setValue(120)
    await wrapper.find('form').trigger('submit'); await flushPromises()
    expect(api.saveSettings).toHaveBeenCalledWith({ ...defaults, interval_minutes: 120 })
    const before = api.list.mock.calls.length
    wrapper.unmount(); await vi.advanceTimersByTimeAsync(30000)
    expect(api.list).toHaveBeenCalledTimes(before)
  })
})
