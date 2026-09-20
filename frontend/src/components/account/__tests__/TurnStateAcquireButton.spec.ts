import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import TurnStateAcquireButton from '../TurnStateAcquireButton.vue'

const { post } = vi.hoisted(() => ({ post: vi.fn() }))
vi.mock('@/api/client', () => ({ apiClient: { post } }))
vi.mock('vue-i18n', () => ({ useI18n: () => ({ locale: ref('zh-CN') }) }))
beforeEach(() => { post.mockReset() })

describe('TurnStateAcquireButton', () => {
  it('submits the exact account and model, disables duplicate clicks and reports queued rather than ready', async () => {
    let finish!: (value: unknown) => void
    post.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const wrapper = mount(TurnStateAcquireButton, { props: { accountId: 42, model: 'gpt-6-astra' } })
    await wrapper.get('button').trigger('click')
    await wrapper.get('button').trigger('click')
    expect(post).toHaveBeenCalledTimes(1)
    expect(post).toHaveBeenCalledWith('/admin/accounts/42/turn-state/acquire', { model: 'gpt-6-astra' }, { signal: expect.any(AbortSignal) })
    expect((wrapper.get('button').element as HTMLButtonElement).disabled).toBe(true)
    expect(wrapper.text()).toContain('提交中')
    finish({ data: { queued: true } })
    await flushPromises()
    expect(wrapper.emitted('queued')).toHaveLength(1)
    expect(wrapper.text()).toContain('等待采集结果')
    expect((wrapper.get('button').element as HTMLButtonElement).disabled).toBe(false)
    wrapper.unmount()
  })

  it('allows a different model to submit independently and never bubbles the click', async () => {
    post.mockResolvedValue({ data: { queued: true } })
    const wrapper = mount({ components: { TurnStateAcquireButton }, data: () => ({ clicks: 0 }), template: '<div @click="clicks++"><span data-clicks>{{ clicks }}</span><TurnStateAcquireButton :account-id="42" model="gpt-5.6-sol" compact /><TurnStateAcquireButton :account-id="42" model="gpt-5.5" compact /></div>' })
    for (const button of wrapper.findAll('button')) await button.trigger('click')
    await flushPromises()
    expect(post.mock.calls.map(call => call[1].model)).toEqual(['gpt-5.6-sol', 'gpt-5.5'])
    expect(wrapper.get('[data-clicks]').text()).toBe('0')
    wrapper.unmount()
  })

  it('shows server failure and permits a retry without claiming success', async () => {
    post.mockRejectedValueOnce({ response: { data: { message: '该模型不在采集名单内' } } }).mockResolvedValueOnce({ data: { queued: true } })
    const wrapper = mount(TurnStateAcquireButton, { props: { accountId: 42, model: 'gpt-5.5' } })
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('不在采集名单内')
    expect(wrapper.emitted('queued')).toBeUndefined()
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.emitted('queued')).toHaveLength(1)
    wrapper.unmount()
  })

  it('ignores late completions after switching accounts and cancels on unmount', async () => {
    let finish!: (value: unknown) => void
    post.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const wrapper = mount(TurnStateAcquireButton, { props: { accountId: 42, model: 'gpt-6-astra' } })
    await wrapper.get('button').trigger('click')
    const signal = post.mock.calls[0][2].signal as AbortSignal
    await wrapper.setProps({ accountId: 43 })
    expect(signal.aborted).toBe(true)
    finish({ data: { queued: true } })
    await flushPromises()
    expect(wrapper.emitted('queued')).toBeUndefined()
    await wrapper.get('button').trigger('click')
    const nextSignal = post.mock.calls[1][2].signal as AbortSignal
    wrapper.unmount()
    expect(nextSignal.aborted).toBe(true)
  })

  it('does not submit when disabled', async () => {
    const wrapper = mount(TurnStateAcquireButton, { props: { accountId: 42, model: 'gpt-6-astra', disabled: true } })
    await wrapper.get('button').trigger('click')
    expect(post).not.toHaveBeenCalled()
    wrapper.unmount()
  })
})
