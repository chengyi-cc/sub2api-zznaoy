import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import PrismRuntimeControls from '../PrismRuntimeControls.vue'
import zh from '@/i18n/locales/zh/admin/prism'
import { controlPrismRuntime, getPrismRuntime } from '@/api/admin/prismRuntime'

vi.unmock('vue-i18n')
vi.mock('@/api/admin/prismRuntime', () => ({ getPrismRuntime: vi.fn(), controlPrismRuntime: vi.fn() }))
const stopped = { managed: true, gateway_enabled: true, state: 'stopped' as const, healthy: false, desired_enabled: false }
const running = { ...stopped, state: 'running' as const, healthy: true, desired_enabled: true }
let wrapper: VueWrapper | undefined
async function render() {
  wrapper = mount(PrismRuntimeControls, { global: { plugins: [createI18n({ legacy: false, locale: 'zh', messageCompiler: message => () => String(message), messages: { zh: { admin: zh } } })] } })
  await flushPromises()
  return wrapper
}
beforeEach(() => { vi.useFakeTimers(); vi.resetAllMocks(); vi.mocked(getPrismRuntime).mockResolvedValue(stopped) })
afterEach(() => { wrapper?.unmount(); vi.useRealTimers() })

describe('Prism service controls', () => {
  it('starts the shared service without changing account settings', async () => {
    vi.mocked(controlPrismRuntime).mockResolvedValue({ ...stopped, state: 'starting', desired_enabled: true })
    const w = await render()
    await w.get('[data-testid="prism-start"]').trigger('click')
    await flushPromises()
    expect(controlPrismRuntime).toHaveBeenCalledWith('start')
    expect(w.emitted('update:enabled')).toBeUndefined()
  })
  it('requires explicit confirmation before stopping every account', async () => {
    vi.mocked(getPrismRuntime).mockResolvedValue(running)
    vi.mocked(controlPrismRuntime).mockResolvedValue(stopped)
    const w = await render()
    await w.get('[data-testid="prism-stop"]').trigger('click')
    expect(controlPrismRuntime).not.toHaveBeenCalled()
    expect(w.text()).toContain('所有 Prism 账号')
    await w.get('[data-testid="prism-confirm"]').trigger('click')
    await flushPromises()
    expect(controlPrismRuntime).toHaveBeenCalledWith('stop')
  })
  it('disables process control for an external adapter', async () => {
    vi.mocked(getPrismRuntime).mockResolvedValue({ ...stopped, managed: false, state: 'unmanaged' })
    const w = await render()
    expect(w.get('[data-testid="prism-start"]').attributes('disabled')).toBeDefined()
    expect(w.text()).toContain('外部管理')
  })
  it('keeps a failed operation visible after status refresh', async () => {
    vi.mocked(controlPrismRuntime).mockRejectedValue(new Error('not confirmed'))
    const w = await render()
    await w.get('[data-testid="prism-start"]').trigger('click')
    await flushPromises()
    expect(w.get('[role="alert"]').text()).toContain('操作未确认成功')
    await vi.advanceTimersByTimeAsync(5000)
    expect(w.get('[role="alert"]').text()).toContain('操作未确认成功')
  })
  it('shows translated safe log events and stops polling on close', async () => {
    const w = await render()
    vi.mocked(getPrismRuntime).mockResolvedValue({ ...stopped, logs: [{ id: 1, time: '2026-10-04T12:00:00Z', code: 'service_ready' }] })
    await w.get('[data-testid="prism-logs"]').trigger('click')
    await flushPromises()
    expect(getPrismRuntime).toHaveBeenLastCalledWith(true)
    expect(w.get('[data-testid="prism-log-panel"]').text()).toContain('服务已就绪')
    w.unmount()
    const count = vi.mocked(getPrismRuntime).mock.calls.length
    await vi.advanceTimersByTimeAsync(15000)
    expect(getPrismRuntime).toHaveBeenCalledTimes(count)
    wrapper = undefined
  })
})
