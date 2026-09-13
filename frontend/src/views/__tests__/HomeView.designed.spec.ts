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

describe('selected dual-theme homepage', () => {
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

  it('uses Command in normal mode and switches to Silver using the existing saved theme', async () => {
    const wrapper = mountHome()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('12-command')
    expect(wrapper.get('h1').text()).toBe('你的应用，下一层智能。')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('15-silver')
    expect(wrapper.get('h1').text()).toBe('连接所想。专注所长。')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(localStorage.getItem('theme')).toBe('dark')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('12-command')
    expect(localStorage.getItem('theme')).toBe('light')
  })

  it.each(['dark', 'light'])('restores an explicit %s preference instead of the system default', async theme => {
    localStorage.setItem('theme', theme)
    Object.assign(systemPreference, { matches: theme !== 'dark' })
    const wrapper = mountHome()
    await flushPromises()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe(theme === 'dark' ? '15-silver' : '12-command')
  })

  it('preserves the selected sculpture, rotation angle and pause control across theme changes', async () => {
    const wrapper = mountHome()
    await vi.dynamicImportSettled()
    await flushPromises()
    const slot = wrapper.get('[data-hc-artwork]')
    const identifier = slot.attributes('data-hc-current-art')
    expect(identifier).toBeTruthy()
    await slot.trigger('click')
    const angle = slot.attributes('data-hc-yaw')
    const svg = slot.get('svg').element
    const selected = sessionStorage.getItem('hc:last-hero-artwork')
    expect(Number(angle)).toBe(32)
    await wrapper.get('#dh-motion-pause').setValue(true)
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-hc-artwork]').element).toBe(slot.element)
    expect(slot.get('svg').element).toBe(svg)
    expect(slot.attributes('data-hc-current-art')).toBe(identifier)
    expect(slot.attributes('data-hc-yaw')).toBe(angle)
    expect(sessionStorage.getItem('hc:last-hero-artwork')).toBe(selected)
    expect((wrapper.get('#dh-motion-pause').element as HTMLInputElement).checked).toBe(true)
    wrapper.unmount()
    expect((slot.element as HTMLElement).dataset.hcInteractive).toBeUndefined()
  })

  it('follows system changes only when no explicit preference has been saved', async () => {
    const wrapper = mountHome()
    Object.assign(systemPreference, { matches: true })
    systemPreference.dispatchEvent(new Event('change'))
    await flushPromises()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('15-silver')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    systemPreference.dispatchEvent(new Event('change'))
    await flushPromises()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('12-command')
  })

  it('tracks theme changes from other pages and browser tabs', async () => {
    const wrapper = mountHome()
    document.documentElement.classList.add('dark')
    await flushPromises()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('15-silver')
    localStorage.setItem('theme', 'light')
    window.dispatchEvent(new StorageEvent('storage', { key: 'theme', newValue: 'light' }))
    await flushPromises()
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('12-command')
  })

  it.each([[false, false, '/login'], [true, false, '/dashboard'], [true, true, '/admin/dashboard']] as const)('routes the primary action for authenticated=%s admin=%s', (authenticated, admin, destination) => {
    authStore.isAuthenticated = authenticated
    authStore.isAdmin = admin
    const wrapper = mountHome()
    expect(wrapper.findAllComponents(RouterLinkStub).find(link => link.attributes('data-testid') === 'home-primary-action')?.props('to')).toBe(destination)
  })

  it.each(['zh', 'en'])('renders translated content, valid anchors and a single sculpture in %s', async locale => {
    appStore.cachedPublicSettings.site_name = 'A deliberately long site name for a narrow viewport'
    appStore.cachedPublicSettings.site_subtitle = 'Our own subtitle'
    appStore.cachedPublicSettings.doc_url = 'https://example.com/docs'
    const wrapper = mountHome(locale)
    expect(wrapper.findAll('h1')).toHaveLength(1)
    expect(wrapper.findAll('[data-hc-artwork]')).toHaveLength(1)
    expect(wrapper.get('.dh-lead').text()).toBe('Our own subtitle')
    expect(wrapper.get('.dh-brand').text()).toContain('A deliberately long')
    expect(wrapper.text()).not.toContain('home.designed.')
    expect(wrapper.get('.dh-footer').text()).toContain(locale === 'zh' ? '模型广场' : 'Models')
    for (const link of wrapper.findAll('a[href^="#"]')) expect(wrapper.find(link.attributes('href')).exists()).toBe(true)
    for (const link of wrapper.findAll('a[target="_blank"]')) expect(link.attributes('rel')).toBe('noopener noreferrer')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.text()).not.toContain('home.designed.')
  })

  it.each(['dark', 'light'])('keeps only account actions above the fold in %s mode', theme => {
    localStorage.setItem('theme', theme)
    authStore.isAuthenticated = true
    appStore.cachedPublicSettings.registration_enabled = true
    const wrapper = mountHome()
    expect(wrapper.find('.dh-nav').exists()).toBe(false)
    expect(wrapper.find('a[href="#dh-guide"]').exists()).toBe(false)
    expect(wrapper.find('.dh-mini-grid').exists()).toBe(false)
    expect(wrapper.findAll('.dh-actions a')).toHaveLength(1)
    expect(wrapper.get('[data-testid="home-primary-action"]').text()).toBe('进入控制台')
    expect(wrapper.findAllComponents(RouterLinkStub).filter(link => link.props('to') === '/register')).toHaveLength(0)
    expect(wrapper.get('[data-testid="home-header-login"]').classes()).toContain('dh-account-primary')
  })

  it.each([true, false, undefined])('respects the registration setting: %s', enabled => {
    appStore.cachedPublicSettings.registration_enabled = enabled
    const wrapper = mountHome()
    expect(wrapper.get('[data-testid="home-primary-action"]').text()).toBe('登录控制台')
    expect(wrapper.get('[data-testid="home-header-login"]').text()).toBe('登录')
    const registrationLinks = wrapper.findAllComponents(RouterLinkStub).filter(link => link.props('to') === '/register')
    expect(registrationLinks).toHaveLength(enabled === true ? 2 : 0)
    if (enabled) {
      expect(wrapper.get('[data-testid="home-register-action"]').text()).toBe('注册账号')
      expect(wrapper.get('[data-testid="home-header-register"]').classes()).toContain('dh-account-primary')
    }
  })

  it('uses meaningful product copy when the configured subtitle only repeats the brand', async () => {
    appStore.cachedPublicSettings.site_subtitle = ' Our platform '
    const wrapper = mountHome()
    expect(wrapper.get('.dh-lead').text()).toContain('在一个控制台连接模型')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('.dh-lead').text()).toContain('模型接入、密钥管理与用量记录')
  })

  it('does not expose the model plaza footer link when it requires authentication', () => {
    appStore.cachedPublicSettings.model_plaza_require_auth = true
    const wrapper = mountHome()
    expect(wrapper.findAllComponents(RouterLinkStub).filter(link => link.props('to') === '/model-plaza')).toHaveLength(0)
  })

  it.each(['zh', 'en'])('adds the same documentation link immediately before the locale selector in %s', async locale => {
    appStore.cachedPublicSettings.doc_url = 'https://example.com/docs'
    const wrapper = mountHome(locale)
    const headerDocs = wrapper.get('[data-testid="home-header-docs"]')
    const footerDocs = wrapper.get('.dh-footer a[target="_blank"]')
    expect(headerDocs.text()).toBe(locale === 'zh' ? '文档' : 'Docs')
    expect(headerDocs.attributes('href')).toBe(footerDocs.attributes('href'))
    expect(headerDocs.attributes('target')).toBe('_blank')
    expect(headerDocs.attributes('rel')).toBe('noopener noreferrer')
    expect(headerDocs.element.nextElementSibling).toBe(wrapper.get('.dh-locale').element)
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-testid="home-header-docs"]').attributes('href')).toBe('https://example.com/docs')
  })

  it.each(['', 'javascript:alert(1)'])('hides the header documentation link without a safe configured URL: %s', docUrl => {
    appStore.cachedPublicSettings.doc_url = docUrl
    const wrapper = mountHome()
    expect(wrapper.find('[data-testid="home-header-docs"]').exists()).toBe(false)
    expect(wrapper.find('.dh-footer a[target="_blank"]').exists()).toBe(false)
  })

  it('keeps switching usable when browser storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    const wrapper = mountHome()
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('15-silver')
  })
})
