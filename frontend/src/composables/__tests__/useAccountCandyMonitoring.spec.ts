import { defineComponent, ref } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useAccountCandyMonitoring } from '../useAccountCandyMonitoring'
const api = vi.hoisted(() => ({ states: vi.fn(), setMonitoring: vi.fn() }))
vi.mock('@/api/admin/candyMonitor', () => ({ candyMonitorAPI: api }))
const state = (id: number, enabled = false, answer: number | null = 29) => ({ account_id: id, enabled, use_defaults: false, model_id: 'custom-model', interval_minutes: 17, last_valid_answer: answer, last_valid_at: '2026-09-24T00:00:00Z' })
function setup(initial = [1, 2]) {
  const ids = ref(initial)
  let controls!: ReturnType<typeof useAccountCandyMonitoring>
  const wrapper = mount(defineComponent({ setup() { controls = useAccountCandyMonitoring(ids); return () => null } }))
  return { wrapper, ids, controls }
}
describe('account candy monitoring', () => {
  beforeEach(() => {
    vi.useFakeTimers(); vi.resetAllMocks()
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(false)
    api.states.mockImplementation(async (ids: number[]) => ({ scheduler_enabled: true, items: ids.map(id => state(id)) }))
    api.setMonitoring.mockImplementation(async (id: number, enabled: boolean) => ({ scheduler_enabled: true, items: [state(id, enabled)] }))
  })
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })
  it('batches page reads and keeps the valid answer through failed refreshes', async () => {
    const { wrapper, controls } = setup(); await flushPromises()
    expect(api.states).toHaveBeenCalledTimes(1)
    expect(api.states).toHaveBeenCalledWith([1, 2])
    expect(controls.states.value[1].last_valid_answer).toBe(29)
    api.states.mockRejectedValueOnce(new Error('offline'))
    await vi.advanceTimersByTimeAsync(10000); await flushPromises()
    expect(controls.states.value[1].last_valid_answer).toBe(29)
    expect(controls.loadError.value).toBe(true)
    const calls = api.states.mock.calls.length
    wrapper.unmount(); await vi.advanceTimersByTimeAsync(30000)
    expect(api.states).toHaveBeenCalledTimes(calls)
  })
  it('ignores late responses from the previous page', async () => {
    let resolve!: (value: unknown) => void
    api.states.mockImplementationOnce(() => new Promise(r => { resolve = r }))
    const { wrapper, controls, ids } = setup()
    ids.value = [3]; await flushPromises()
    resolve({ scheduler_enabled: true, items: [state(1)] }); await flushPromises()
    expect(Object.keys(controls.states.value)).toEqual(['3'])
    wrapper.unmount()
  })
  it('only changes enabled state, prevents duplicate writes, and ignores stale reads', async () => {
    const { wrapper, controls } = setup(); await flushPromises()
    let resolveRead!: (value: unknown) => void
    let resolveWrite!: (value: unknown) => void
    api.states.mockImplementationOnce(() => new Promise(r => { resolveRead = r }))
    const refresh = controls.refresh()
    api.setMonitoring.mockImplementationOnce(() => new Promise(r => { resolveWrite = r }))
    const write = controls.setEnabled(1, true)
    expect(controls.pending.has(1)).toBe(true)
    await controls.setEnabled(1, true)
    expect(api.setMonitoring).toHaveBeenCalledTimes(1)
    expect(api.setMonitoring).toHaveBeenCalledWith(1, true)
    resolveRead({ scheduler_enabled: true, items: [state(1, false, 21)] }); await refresh
    expect(controls.states.value[1].last_valid_answer).toBe(29)
    api.states.mockResolvedValue({ scheduler_enabled: false, items: [state(1, true)] })
    resolveWrite({ scheduler_enabled: false, items: [state(1, true)] }); await write; await flushPromises()
    expect(controls.states.value[1]).toMatchObject({ enabled: true, model_id: 'custom-model', interval_minutes: 17 })
    expect(controls.schedulerEnabled.value).toBe(false)
    expect(controls.pending.size).toBe(0)
    wrapper.unmount()
  })
  it('keeps the checkbox state on write failure and can recover', async () => {
    const { wrapper, controls } = setup(); await flushPromises()
    api.setMonitoring.mockRejectedValueOnce(new Error('offline'))
    await expect(controls.setEnabled(1, true)).rejects.toThrow('offline'); await flushPromises()
    expect(controls.states.value[1].enabled).toBe(false)
    expect(controls.pending.size).toBe(0)
    wrapper.unmount()
  })
  it('bounds batches for large pages and skips empty pages', async () => {
    const { wrapper, ids } = setup(Array.from({ length: 501 }, (_, index) => index + 1)); await flushPromises()
    expect(api.states.mock.calls.map(call => call[0].length)).toEqual([200, 200, 101])
    ids.value = []; await flushPromises()
    expect(api.states).toHaveBeenCalledTimes(3)
    wrapper.unmount()
  })
})
