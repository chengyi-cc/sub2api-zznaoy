import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AccountTestModal from '../AccountTestModal.vue'
import AdminAccountTestModal from '../../admin/account/AccountTestModal.vue'

const { getAvailableModels } = vi.hoisted(() => ({ getAvailableModels: vi.fn() }))
vi.mock('@/api/admin', () => ({ adminAPI: { accounts: { getAvailableModels } } }))
vi.mock('@/composables/useClipboard', () => ({ useClipboard: () => ({ copyToClipboard: vi.fn() }) }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) }
})

const expected = 21

function streamResponse(events: unknown[]) {
  const chunks = [new TextEncoder().encode(events.map(event => `data:${JSON.stringify(event)}\r\n\r\n`).join(''))]
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: vi.fn(async () => chunks.length ? { done: false, value: chunks.shift() } : { done: true }),
        releaseLock: vi.fn()
      })
    }
  }
}

describe.each([
  ['admin', AdminAccountTestModal],
  ['shared', AccountTestModal]
] as const)('%s candy account test', (_name, component) => {
  function mountModal(platform = 'openai') {
    return mount(component, {
      props: { show: false, account: { id: 42, name: 'test account', platform, type: 'apikey', credentials: {}, extra: {}, status: 'active' } } as any,
      global: {
        stubs: {
          BaseDialog: { template: '<div><slot /><slot name="footer" /></div>' },
          Select: { props: ['modelValue', 'options'], template: '<div />' },
          TextArea: true,
          Icon: true
        }
      }
    })
  }

  beforeEach(() => {
    getAvailableModels.mockResolvedValue([{ id: 'text-model', display_name: 'Text model' }, { id: 'gpt-image-1', display_name: 'Image model' }])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([])))
  })

  afterEach(() => vi.unstubAllGlobals())

  it.each(['pass', 'incorrect', 'invalid_format', 'inconclusive'])('renders %s separately from transport completion', async verdict => {
    const result = { case_id: 'candy-shape-v1', verdict, reason: 'test', expected, actual: verdict === 'incorrect' ? 29 : expected, duration_ms: 1000 }
    vi.mocked(fetch).mockResolvedValue(streamResponse([
      { type: 'test_start', model: 'text-model' },
      { type: 'candy_result', data: result },
      { type: 'test_complete', success: true }
    ]) as Response)
    const wrapper = mountModal()
    await wrapper.setProps({ show: true })
    await flushPromises()
    const state = wrapper.vm as any
    state.selectedModelId = 'text-model'
    state.testMode = 'candy'
    await flushPromises()
    expect(wrapper.find('[data-testid="candy-test-panel"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="candy-test-result"]').exists()).toBe(false)
    await state.startTest()
    await flushPromises()
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)).toEqual({ model_id: 'text-model', prompt: '', mode: 'candy' })
    expect(wrapper.find('[data-testid="candy-test-result"]').attributes('data-verdict')).toBe(verdict)
    expect(wrapper.text()).toContain('admin.accounts.candy.connected')
    expect(wrapper.text()).toContain(`admin.accounts.candy.verdict.${verdict}`)
    expect(wrapper.text()).not.toContain('admin.accounts.testCompleted')
    state.selectedModelId = 'gpt-image-1'
    await flushPromises()
    expect(state.testMode).toBe('default')
    expect(state.candyResult).toBeNull()
    wrapper.unmount()
  })

  it.each(['anthropic', 'gemini'])('sends candy mode for %s text accounts', async platform => {
    const wrapper = mountModal(platform)
    await wrapper.setProps({ show: true })
    await flushPromises()
    const state = wrapper.vm as any
    state.selectedModelId = 'text-model'
    state.testMode = 'candy'
    await flushPromises()
    expect(state.openAITestModeOptions.map((option: { value: string }) => option.value)).toEqual(['default', 'candy'])
    await state.startTest()
    expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string).mode).toBe('candy')
    wrapper.unmount()
  })

  it.each([
    ['missing backend grade', [{ type: 'test_complete', success: true }]],
    ['early end', [{ type: 'content', text: '21' }]],
    ['upstream error', [{ type: 'error', error: 'timed out' }]]
  ])('shows inconclusive for %s', async (_description, events) => {
    vi.mocked(fetch).mockResolvedValue(streamResponse(events as unknown[]) as Response)
    const wrapper = mountModal()
    await wrapper.setProps({ show: true })
    await flushPromises()
    const state = wrapper.vm as any
    state.selectedModelId = 'text-model'
    state.testMode = 'candy'
    await flushPromises()
    await state.startTest()
    await flushPromises()
    expect(wrapper.find('[data-testid="candy-test-result"]').attributes('data-verdict')).toBe('inconclusive')
    wrapper.unmount()
  })

  it('does not accept results from a closed test after reopening', async () => {
    let finishOldRequest!: (response: Response) => void
    vi.mocked(fetch).mockImplementationOnce(() => new Promise(resolve => { finishOldRequest = resolve }))
    const wrapper = mountModal()
    await wrapper.setProps({ show: true })
    await flushPromises()
    const state = wrapper.vm as any
    state.selectedModelId = 'text-model'
    state.testMode = 'candy'
    await flushPromises()
    const oldRequest = state.startTest()
    await flushPromises()
    const oldSignal = vi.mocked(fetch).mock.calls[0][1]!.signal
    await wrapper.setProps({ show: false })
    expect(oldSignal?.aborted).toBe(true)
    await wrapper.setProps({ show: true })
    await flushPromises()
    finishOldRequest(streamResponse([
      { type: 'candy_result', data: { case_id: 'candy-shape-v1', verdict: 'pass', reason: 'correct', expected, actual: expected } },
      { type: 'test_complete', success: true }
    ]) as Response)
    await oldRequest
    await flushPromises()
    expect(state.status).toBe('idle')
    expect(state.candyResult).toBeNull()
    wrapper.unmount()
  })
})
