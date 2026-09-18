import { afterEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, ref } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import type { AccountListItem } from '@/types'
import { useTurnStateSummaries } from '../useTurnStateSummaries'

const { get } = vi.hoisted(() => ({ get: vi.fn() }))
vi.mock('@/api/client', () => ({ apiClient: { get } }))
afterEach(() => { vi.useRealTimers(); vi.resetAllMocks() })

describe('useTurnStateSummaries', () => {
  it('limits reads to four, cancels old pages and pauses when hidden', async () => {
    vi.useFakeTimers()
    const accounts = ref(Array.from({ length: 6 }, (_, index) => ({ id: index + 1, platform: 'openai', type: 'oauth', extra: { codex_turn_state_auto_enabled: true } })) as AccountListItem[])
    const enabled = ref(true)
    const waiting: Array<(value: unknown) => void> = []
    get.mockImplementation(() => new Promise(resolve => waiting.push(resolve)))
    let state!: ReturnType<typeof useTurnStateSummaries>
    const wrapper = mount(defineComponent({ setup() { state = useTurnStateSummaries(accounts, enabled); return () => null } }))
    expect(get).toHaveBeenCalledTimes(4)
    const oldSignal = get.mock.calls[0][1].signal as AbortSignal
    accounts.value = [{ ...accounts.value[0], id: 99 }]
    await flushPromises()
    expect(oldSignal.aborted).toBe(true)
    expect(get).toHaveBeenCalledTimes(5)
    waiting[0]({ data: { enabled: true, configured: true, models: [] } })
    waiting[4]({ data: { enabled: true, configured: true, models: [] } })
    await flushPromises()
    expect(Object.keys(state.summaries.value)).toEqual(['99'])
    expect(get.mock.calls.every(([url]) => !url.includes('history'))).toBe(true)
    enabled.value = false
    await flushPromises()
    await vi.advanceTimersByTimeAsync(30000)
    expect(get).toHaveBeenCalledTimes(5)
    wrapper.unmount()
  })
})
