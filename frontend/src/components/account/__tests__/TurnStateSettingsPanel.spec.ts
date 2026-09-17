import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import TurnStateSettingsPanel from '../TurnStateSettingsPanel.vue'

const { get, put, post } = vi.hoisted(() => ({ get: vi.fn(), put: vi.fn(), post: vi.fn() }))
vi.mock('@/api/client', () => ({ apiClient: { get, put, post } }))
vi.mock('vue-i18n', () => ({ useI18n: () => ({ locale: ref('zh-CN') }) }))
const configuration = {
  revision: 'first', purchased_enabled: true, proxy_host: 'proxy.example:7778', proxy_username: 'test_{country}_{session}',
  proxy_password_configured: true, proxy_upstream_configured: true, countries: 'US,DE',
  ipv6_enabled: true, pool_url: 'https://pool.example:18443', pool_token_configured: true,
  pool_ca: 'certificate', attempts: 9, concurrency: 4
}
beforeEach(() => { vi.clearAllMocks(); get.mockResolvedValue({ data: { ...configuration } }); post.mockResolvedValue({ data: { started: true } }) })

describe('TurnStateSettingsPanel', () => {
  it('loads shared configuration and keeps secrets empty', async () => {
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    expect(wrapper.text()).toContain('所有账号共用')
    expect((wrapper.get('[data-testid="settings-proxy-password"]').element as HTMLInputElement).value).toBe('')
    expect((wrapper.get('[data-testid="settings-pool-token"]').element as HTMLInputElement).value).toBe('')
    expect(wrapper.get('[data-testid="settings-proxy-password"]').attributes('placeholder')).toContain('已配置')
    wrapper.unmount()
  })

  it('saves both sources without submitting the account form and clears new secrets', async () => {
    put.mockResolvedValue({ data: { ...configuration, revision: 'second' } })
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    await wrapper.get('[data-testid="settings-proxy-password"]').setValue('new-secret')
    await wrapper.get('[data-testid="settings-pool-token"]').setValue('new-pool-token')
    await wrapper.get('[data-testid="settings-countries"]').setValue('US,JP')
    expect(wrapper.get('[data-testid="settings-save"]').attributes('type')).toBe('button')
    await wrapper.get('[data-testid="settings-save"]').trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith('/admin/accounts/turn-state/settings', expect.objectContaining({ proxy_password: 'new-secret', pool_token: 'new-pool-token', countries: 'US,JP', revision: 'first' }))
    expect(wrapper.emitted('saved')).toHaveLength(1)
    expect(wrapper.text()).toContain('配置已保存并生效')
    expect((wrapper.get('[data-testid="settings-proxy-password"]').element as HTMLInputElement).value).toBe('')
    wrapper.unmount()
  })

  it('starts an immediate acquisition for the selected model', async () => {
    put.mockResolvedValue({ data: { ...configuration, revision: 'second' } })
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    await wrapper.get('[data-testid="settings-acquisition-model"]').setValue('gpt-6-astra')
    await wrapper.get('[data-testid="settings-acquire-now"]').trigger('click')
    await flushPromises()
    expect(post).toHaveBeenCalledWith('/admin/accounts/42/turn-state/acquire', { model: 'gpt-6-astra' })
    expect(wrapper.text()).toContain('已提交采集任务')
    wrapper.unmount()
  })

  it('does not report acquisition success on a rejected request', async () => {
    post.mockRejectedValueOnce({ message: '请先保存账号并开启自动采集' })
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    await wrapper.get('[data-testid="settings-acquire-now"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('请先保存账号')
    expect(wrapper.text()).not.toContain('已提交采集任务')
    expect(wrapper.emitted('saved')).toBeUndefined()
    wrapper.unmount()
  })

  it('shows save errors without falsely reporting success', async () => {
    put.mockRejectedValue({ message: 'Invalid pool certificate' })
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    await wrapper.get('[data-testid="settings-save"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('Invalid pool certificate')
    expect(wrapper.emitted('saved')).toBeUndefined()
    expect(wrapper.find('[role="status"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('allows certificate upload without a server file path', async () => {
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    const input = wrapper.get('[data-testid="settings-ca-upload"]')
    Object.defineProperty(input.element, 'files', { value: [{ size: 200, text: async () => 'uploaded-certificate' }] })
    await input.trigger('change')
    await flushPromises()
    expect((wrapper.get('[data-testid="settings-pool-ca"]').element as HTMLTextAreaElement).value).toBe('uploaded-certificate')
    wrapper.unmount()
  })

  it('does not allow saving when loading failed', async () => {
    get.mockRejectedValue(new Error('network'))
    const wrapper = mount(TurnStateSettingsPanel, { props: { accountId: 42 } })
    await flushPromises()
    expect(wrapper.text()).toContain('读取配置失败')
    expect(wrapper.find('[data-testid="settings-save"]').exists()).toBe(false)
    wrapper.unmount()
  })
})
