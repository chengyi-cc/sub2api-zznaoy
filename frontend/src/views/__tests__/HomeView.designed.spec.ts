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
    expect(wrapper.get('.dh-nav').text()).toContain(locale === 'zh' ? '模型广场' : 'Models')
    for (const link of wrapper.findAll('a[href^="#"]')) expect(wrapper.find(link.attributes('href')).exists()).toBe(true)
    for (const link of wrapper.findAll('a[target="_blank"]')) expect(link.attributes('rel')).toBe('noopener noreferrer')
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.text()).not.toContain('home.designed.')
  })

  it('keeps switching usable when browser storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    const wrapper = mountHome()
    await wrapper.get('[data-testid="home-theme-toggle"]').trigger('click')
    expect(wrapper.get('[data-home-design]').attributes('data-home-design')).toBe('15-silver')
  })
})
