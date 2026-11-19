import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SiteReconSettings from '../SiteReconSettings.vue'

const { api, showSuccess, showError } = vi.hoisted(() => ({
  api: {
    config: vi.fn(),
    tokens: vi.fn(),
    updateConfig: vi.fn(),
    testPush: vi.fn(),
    setToken: vi.fn(),
    deleteToken: vi.fn()
  },
  showSuccess: vi.fn(),
  showError: vi.fn()
}))

vi.mock('@/api/admin/siteRecon', () => ({ siteReconAPI: api, default: api }))
vi.mock('@/stores/app', () => ({ useAppStore: () => ({ showSuccess, showError }) }))
vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-i18n')>()
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, params?: Record<string, unknown>) => (params ? `${key}:${Object.values(params).join(',')}` : key)
    })
  }
})

const configView = {
  config: {
    timezone: 'Asia/Shanghai',
    schedule_enabled: false,
    schedule_time: '08:00',
    days: 7,
    scan: true,
    exclude: [3],
    push_enabled: true,
    balance_days: 2
  },
  bark_configured: true,
  bark_hint: 'https://api.day.app/****',
  next_run: ''
}

describe('SiteReconSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.config.mockResolvedValue(configView)
    api.tokens.mockResolvedValue({
      tokens: [{ host: 'up.example.com', hint: '****ABCD', updated_at: '2026-11-19T08:00:00+08:00', readable: true }],
      need_tokens: ['new.example.com']
    })
    api.updateConfig.mockResolvedValue(configView)
    api.setToken.mockResolvedValue(undefined)
  })

  it('shows only token hints and clears the token input after saving', async () => {
    const wrapper = mount(SiteReconSettings)
    await flushPromises()

    const list = wrapper.get('[data-test="site-recon-token-list"]')
    expect(list.text()).toContain('up.example.com')
    expect(list.text()).toContain('****ABCD')

    await wrapper.get('[data-test="site-recon-token-host"]').setValue('https://new.example.com/console')
    const tokenInput = wrapper.get('[data-test="site-recon-token-input"]')
    expect(tokenInput.attributes('type')).toBe('password')
    await tokenInput.setValue('secret-access-token')
    await wrapper.findAll('form')[1].trigger('submit')
    await flushPromises()

    expect(api.setToken).toHaveBeenCalledWith('https://new.example.com/console', 'secret-access-token')
    expect((wrapper.get('[data-test="site-recon-token-input"]').element as HTMLInputElement).value).toBe('')
    expect(wrapper.html()).not.toContain('secret-access-token')
    expect(api.tokens).toHaveBeenCalledTimes(2)
  })

  it('sends the bark url only when a new one is typed and clears it afterwards', async () => {
    const wrapper = mount(SiteReconSettings)
    await flushPromises()
    expect(wrapper.get('[data-test="site-recon-bark-hint"]').text()).toContain('https://api.day.app/****')

    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(api.updateConfig.mock.calls[0][0]).not.toHaveProperty('bark_secret_url')
    expect(api.updateConfig.mock.calls[0][0].exclude).toEqual([3])

    await wrapper.get('[data-test="site-recon-bark-input"]').setValue('https://api.day.app/NEWKEY')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(api.updateConfig.mock.calls[1][0].bark_secret_url).toBe('https://api.day.app/NEWKEY')
    expect((wrapper.get('[data-test="site-recon-bark-input"]').element as HTMLInputElement).value).toBe('')
  })
})
