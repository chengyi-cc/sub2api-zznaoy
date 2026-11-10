import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount, RouterLinkStub } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import HomeView from '../HomeView.vue'
import zh from '@/i18n/locales/zh/landing'
import en from '@/i18n/locales/en/landing'

const { appStore, authStore } = vi.hoisted(() => ({
  appStore: {
    cachedPublicSettings: {} as Record<string, unknown>,
    siteName: 'Fallback', siteLogo: '', docUrl: '', publicSettingsLoaded: true,
    fetchPublicSettings: vi.fn(),
  },
  authStore: { isAuthenticated: false, isAdmin: false, user: null, checkAuth: vi.fn() },
}))
vi.mock('@/stores', () => ({ useAppStore: () => appStore, useAuthStore: () => authStore }))
vi.mock('@/stores/app', () => ({ useAppStore: () => appStore }))
enableAutoUnmount(afterEach)

let systemPreference: MediaQueryList

function mountHome(locale = 'zh') {
  return mount(HomeView, {
    attachTo: document.body,
    global: {
      plugins: [createI18n({ legacy: false, locale, fallbackLocale: 'en', messages: { zh, en }, messageCompiler: message => () => typeof message === 'string' ? message : '' })],
      stubs: { RouterLink: RouterLinkStub, LocaleSwitcher: true },
    },
  })
}

const theme = (wrapper: ReturnType<typeof mountHome>) => wrapper.get('[data-home-theme]').attributes('data-home-theme')
const linksTo = (wrapper: ReturnType<typeof mountHome>, to: string) =>
  wrapper.findAllComponents(RouterLinkStub).filter(link => link.props('to') === to)

describe('flow-diagram homepage', () => {
  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
    document.documentElement.classList.remove('dark')
    appStore.cachedPublicSettings = { site_name: 'Our platform', site_subtitle: '', model_plaza_enabled: true }
    authStore.isAuthenticated = false
    authStore.isAdmin = false
    systemPreference = Object.assign(new EventTarget(), { matches: false }) as MediaQueryList
    vi.spyOn(window, 'matchMedia').mockImplementation(query => query.includes('reduced-motion')
      ? Object.assign(new EventTarget(), { matches: true }) as MediaQueryList
      : systemPreference)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    document.documentElement.classList.remove('dark')
    document.body.replaceChildren()
  })

  it('renders the light theme by default and switches using the existing saved theme', async () => {
    const wrapper = mountHome()
    expect(theme(wrapper)).toBe('light')
    expect(wrapper.get('h1').text()).toBe('一把密钥，通行所有模型')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(theme(wrapper)).toBe('dark')
    expect(wrapper.get('.designed-home').classes()).toContain('dh-dark')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(localStorage.getItem('theme')).toBe('dark')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(theme(wrapper)).toBe('light')
    expect(localStorage.getItem('theme')).toBe('light')
  })

  it.each(['dark', 'light'])('restores an explicit %s preference instead of the system default', async saved => {
    localStorage.setItem('theme', saved)
    Object.assign(systemPreference, { matches: saved !== 'dark' })
    const wrapper = mountHome()
    await flushPromises()
    expect(theme(wrapper)).toBe(saved)
  })

  it('follows system changes only when no explicit preference has been saved', async () => {
    const wrapper = mountHome()
    Object.assign(systemPreference, { matches: true })
    systemPreference.dispatchEvent(new Event('change'))
    await flushPromises()
    expect(theme(wrapper)).toBe('dark')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    systemPreference.dispatchEvent(new Event('change'))
    await flushPromises()
    expect(theme(wrapper)).toBe('light')
  })

  it('tracks theme changes from other pages and browser tabs', async () => {
    const wrapper = mountHome()
    document.documentElement.classList.add('dark')
    await flushPromises()
    expect(theme(wrapper)).toBe('dark')
    localStorage.setItem('theme', 'light')
    window.dispatchEvent(new StorageEvent('storage', { key: 'theme', newValue: 'light' }))
    await flushPromises()
    expect(theme(wrapper)).toBe('light')
  })

  it('keeps switching usable when browser storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    const wrapper = mountHome()
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(theme(wrapper)).toBe('dark')
  })

  it.each([[false, false, '/login'], [true, false, '/dashboard'], [true, true, '/admin/dashboard']] as const)('routes the primary action for authenticated=%s admin=%s', (authenticated, admin, destination) => {
    authStore.isAuthenticated = authenticated
    authStore.isAdmin = admin
    const wrapper = mountHome()
    expect(wrapper.findAllComponents(RouterLinkStub).find(link => link.attributes('data-testid') === 'home-primary-action')?.props('to')).toBe(destination)
  })

  it('leads with registration and keeps sign-in secondary when registration is enabled', () => {
    appStore.cachedPublicSettings.registration_enabled = true
    const wrapper = mountHome()
    const primary = wrapper.findAllComponents(RouterLinkStub).find(link => link.attributes('data-testid') === 'home-primary-action')
    expect(primary?.props('to')).toBe('/register')
    expect(primary?.text()).toContain('注册账号')
    expect(wrapper.get('[data-testid="home-signin-action"]').text()).toBe('登录控制台')
    expect(wrapper.get('[data-testid="home-header-login"]').text()).toBe('登录')
    expect(wrapper.get('[data-testid="home-header-login"]').classes()).not.toContain('dh-account-primary')
    expect(wrapper.get('[data-testid="home-header-register"]').classes()).toContain('dh-account-primary')
    expect(linksTo(wrapper, '/register')).toHaveLength(2)
  })

  it.each([false, undefined])('offers only sign-in when registration is %s', enabled => {
    appStore.cachedPublicSettings.registration_enabled = enabled
    const wrapper = mountHome()
    expect(wrapper.get('[data-testid="home-primary-action"]').text()).toContain('登录控制台')
    expect(wrapper.find('[data-testid="home-signin-action"]').exists()).toBe(false)
    expect(wrapper.get('[data-testid="home-header-login"]').classes()).toContain('dh-account-primary')
    expect(linksTo(wrapper, '/register')).toHaveLength(0)
  })

  it.each(['dark', 'light'])('shows a single dashboard action to signed-in users in %s mode', saved => {
    localStorage.setItem('theme', saved)
    authStore.isAuthenticated = true
    appStore.cachedPublicSettings.registration_enabled = true
    const wrapper = mountHome()
    expect(wrapper.findAll('.dh-actions a')).toHaveLength(1)
    expect(wrapper.get('[data-testid="home-primary-action"]').text()).toContain('进入控制台')
    expect(wrapper.get('[data-testid="home-header-login"]').classes()).toContain('dh-account-primary')
    expect(linksTo(wrapper, '/register')).toHaveLength(0)
  })

  it.each(['zh', 'en'])('renders translated content, valid anchors and one diagram in %s', async locale => {
    appStore.cachedPublicSettings.site_name = 'A deliberately long site name for a narrow viewport'
    appStore.cachedPublicSettings.site_subtitle = 'Our own subtitle'
    appStore.cachedPublicSettings.doc_url = 'https://example.com/docs'
    const wrapper = mountHome(locale)
    expect(wrapper.findAll('h1')).toHaveLength(1)
    expect(wrapper.findAll('[data-testid="home-flow"]')).toHaveLength(1)
    expect(wrapper.get('.dh-lead').text()).toBe('Our own subtitle')
    expect(wrapper.get('.dh-brand').text()).toContain('A deliberately long')
    expect(wrapper.text()).not.toContain('home.designed.')
    expect(wrapper.get('.dh-nav').text()).toContain(locale === 'zh' ? '模型广场' : 'Models')
    for (const link of wrapper.findAll('a[href^="#"]')) expect(wrapper.find(link.attributes('href')).exists()).toBe(true)
    for (const link of wrapper.findAll('a[target="_blank"]')) expect(link.attributes('rel')).toBe('noopener noreferrer')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.text()).not.toContain('home.designed.')
  })

  it('uses meaningful product copy when the configured subtitle only repeats the brand', () => {
    appStore.cachedPublicSettings.site_subtitle = ' Our platform '
    const wrapper = mountHome()
    expect(wrapper.get('.dh-lead').text()).toContain('用同一个接口地址调用')
  })

  it('shows the configured API host at the centre of the diagram', () => {
    appStore.cachedPublicSettings.api_base_url = 'https://api.example.com/v1'
    const wrapper = mountHome()
    expect(wrapper.get('.dh-hub-host').text()).toBe('api.example.com')
    expect(wrapper.get('.dh-cf-hub strong').text()).toBe('api.example.com')
  })

  it.each(['', 'not a url'])('falls back to the current host when the API address is %j', apiBaseUrl => {
    appStore.cachedPublicSettings.api_base_url = apiBaseUrl
    const wrapper = mountHome()
    expect(wrapper.get('.dh-hub-host').text()).toBe(window.location.host)
  })

  it('draws every tool and model with a link into the gateway', () => {
    const wrapper = mountHome()
    const flow = wrapper.get('[data-testid="home-flow"]')
    const models = flow.findAll('[data-model]')
    expect(models.map(model => model.attributes('data-model'))).toEqual(['claude', 'openai', 'gemini', 'xai', 'deepseek', 'qwen', 'moonshot', 'more'])
    expect(models.at(-1)?.text()).toContain('更多模型')
    expect(flow.text()).toContain('你的应用 / SDK')
    expect(flow.findAll('.dh-line')).toHaveLength(4 + 8)
    expect(flow.findAll('.dh-streak')).toHaveLength(4 + 8)
    expect(flow.findAll('[data-model] svg path').length).toBeGreaterThanOrEqual(7)
  })

  it('does not expose the model plaza link when it requires authentication', () => {
    appStore.cachedPublicSettings.model_plaza_require_auth = true
    const wrapper = mountHome()
    expect(linksTo(wrapper, '/model-plaza')).toHaveLength(0)
  })

  it.each(['zh', 'en'])('links to the configured documentation from the header in %s', locale => {
    appStore.cachedPublicSettings.doc_url = 'https://example.com/docs'
    const wrapper = mountHome(locale)
    const headerDocs = wrapper.get('[data-testid="home-header-docs"]')
    expect(headerDocs.text()).toBe(locale === 'zh' ? '文档' : 'Docs')
    expect(headerDocs.attributes('href')).toBe('https://example.com/docs')
    expect(headerDocs.attributes('target')).toBe('_blank')
    expect(headerDocs.attributes('rel')).toBe('noopener noreferrer')
  })

  it.each(['', 'javascript:alert(1)'])('hides the documentation link without a safe configured URL: %s', docUrl => {
    appStore.cachedPublicSettings.doc_url = docUrl
    const wrapper = mountHome()
    expect(wrapper.find('[data-testid="home-header-docs"]').exists()).toBe(false)
  })
})
