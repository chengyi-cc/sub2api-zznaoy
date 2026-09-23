import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CandyMonitorRunDialog from '../CandyMonitorRunDialog.vue'

const api = vi.hoisted(() => ({ run: vi.fn(), result: vi.fn() }))
vi.mock('@/api/admin/candyMonitor', () => ({ candyMonitorAPI: api, CANDY_DEFAULT_MODEL: 'gpt-6-astra' }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) }
})
const running = { id: 7, account_id: 42, model_id: 'gpt-6-astra', verdict: 'running', duration_ms: 0 }
function setup() {
  return mount(CandyMonitorRunDialog, {
    props: { show: true, account: { id: 42, name: 'Account' } },
    global: { stubs: {
      BaseDialog: { props: ['show'], template: '<div v-if="show"><slot /><slot name="footer" /></div>' },
      CandyTestPanel: true, RouterLink: { template: '<a><slot /></a>' }
    } }
  })
}
describe('CandyMonitorRunDialog', () => {
  beforeEach(() => { vi.useFakeTimers(); vi.clearAllMocks(); api.run.mockResolvedValue(running) })
  afterEach(() => vi.useRealTimers())
  it.each([['pass', 21], ['incorrect', 29], ['inconclusive', undefined]])('starts once with Astra and displays %s', async (verdict, actual) => {
    api.result.mockResolvedValue({ ...running, verdict, actual, response_text: '<script>untrusted</script>' })
    const wrapper = setup()
    await flushPromises()
    expect(api.run).toHaveBeenCalledTimes(1)
    expect(api.run).toHaveBeenCalledWith(42, 'gpt-6-astra')
    await vi.advanceTimersByTimeAsync(1500)
    await flushPromises()
    expect(wrapper.find(`[data-verdict="${verdict}"]`).exists()).toBe(true)
    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.find('pre').text()).toBe('<script>untrusted</script>')
    expect(wrapper.emitted('completed')).toHaveLength(1)
    wrapper.unmount()
  })
  it('closing stops polling without cancelling the server run', async () => {
    const wrapper = setup(); await flushPromises()
    await wrapper.setProps({ show: false })
    await vi.advanceTimersByTimeAsync(10000)
    expect(api.result).not.toHaveBeenCalled()
    expect(api.run).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })
  it('ignores late results when switching accounts', async () => {
    let finishOld!: (value: unknown) => void
    api.run.mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve }))
    const wrapper = setup(); await flushPromises()
    await wrapper.setProps({ account: { id: 43, name: 'Second' } })
    await flushPromises()
    finishOld({ ...running, verdict: 'incorrect', actual: 29 })
    await flushPromises()
    expect(wrapper.find('[data-verdict="incorrect"]').exists()).toBe(false)
    expect(wrapper.find('[data-verdict="running"]').exists()).toBe(true)
    expect(api.run).toHaveBeenLastCalledWith(43, 'gpt-6-astra')
    wrapper.unmount()
  })
  it('polling failure allows refresh without charging for a new test', async () => {
    api.result.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce({ ...running, verdict: 'pass', actual: 21 })
    const wrapper = setup(); await flushPromises(); await vi.advanceTimersByTimeAsync(1500)
    expect(wrapper.find('[role="alert"]').text()).toBe('offline')
    const refresh = wrapper.findAll('button').find(b => b.text().endsWith('.refresh'))!
    await refresh.trigger('click'); await flushPromises()
    expect(api.run).toHaveBeenCalledTimes(1)
    expect(wrapper.find('[data-verdict="pass"]').exists()).toBe(true)
    wrapper.unmount()
  })
})
