import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import HomeView from '../HomeView.vue'

const { appStore, authStore } = vi.hoisted(() => ({
  appStore: {
    cachedPublicSettings: {} as Record<string, unknown>,
    siteName: 'Test site',
    siteLogo: '',
    docUrl: '',
    publicSettingsLoaded: true,
    fetchPublicSettings: vi.fn(),
  },
  authStore: {
    isAuthenticated: false,
    isAdmin: false,
    user: null,
    checkAuth: vi.fn(),
  },
}))

vi.mock('@/stores', () => ({ useAppStore: () => appStore, useAuthStore: () => authStore }))
vi.mock('@/stores/app', () => ({ useAppStore: () => appStore }))
vi.mock('vue-i18n', async (importOriginal) => ({
  ...await importOriginal<typeof import('vue-i18n')>(),
  useI18n: () => ({ t: (key: string) => key }),
}))

describe('paste-ready homepage concepts', () => {
  beforeEach(() => {
    localStorage.clear()
    window.sessionStorage.clear()
    vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: false } as MediaQueryList)
  })

  it.each(['01-orbit', '02-porcelain', '03-prism', '04-terminal', '05-atelier', '06-armillary', '07-crystal', '08-corona', '09-trefoil', '10-lattice'])(
    'renders %s through the existing custom homepage setting without scripts',
    (identifier) => {
      const fragment = readFileSync(resolve(__dirname, '../../../public/homepage-concepts/fragments', `${identifier}.html`), 'utf8')
      appStore.cachedPublicSettings = { home_content: fragment, compact_home_enabled: true }
      const wrapper = mount(HomeView, {
        global: { stubs: { RouterLink: true, LocaleSwitcher: true, Icon: true } },
      })

      expect(wrapper.findAll('.hc-site')).toHaveLength(1)
      expect(wrapper.findAll('h1')).toHaveLength(1)
      expect(wrapper.get('h1').text().length).toBeGreaterThan(5)
      expect(wrapper.findAll('details')).toHaveLength(3)
      expect(wrapper.findAll('script')).toHaveLength(0)
      expect(wrapper.findAll('iframe')).toHaveLength(0)
      expect(wrapper.get('style').text()).toContain('@media(max-width:760px)')
      expect(wrapper.get('a[href="/dashboard"]').attributes('target')).toBe('_top')
      expect(wrapper.find('[data-testid="compact-home"]').exists()).toBe(false)

      for (const anchor of wrapper.findAll('a[href^="#"]')) {
        expect(wrapper.find(anchor.attributes('href')).exists()).toBe(true)
      }

      wrapper.unmount()
    },
  )

  it.each(['11-cloud', '12-command', '13-editorial', '14-cobalt', '15-silver'])(
    'renders %s and initializes the six-artwork pack through native homepage code',
    async (identifier) => {
      const fragment = readFileSync(resolve(__dirname, '../../../public/homepage-concepts/fragments', `${identifier}.html`), 'utf8')
      appStore.cachedPublicSettings = { home_content: fragment, compact_home_enabled: true }
      const wrapper = mount(HomeView, {
        global: { stubs: { RouterLink: true, LocaleSwitcher: true, Icon: true } },
      })
      await vi.dynamicImportSettled()
      await flushPromises()
      expect(wrapper.findAll('.hf-site')).toHaveLength(1)
      expect(wrapper.findAll('h1')).toHaveLength(1)
      expect(wrapper.findAll('details')).toHaveLength(3)
      expect(wrapper.findAll('script,iframe')).toHaveLength(0)
      expect(wrapper.get('[data-hc-artwork="random"]').attributes('data-hc-current-art')).toMatch(/^(05-atelier|06-armillary|07-crystal|08-corona|09-trefoil|10-lattice)$/)
      expect(wrapper.get('[data-hc-artwork-name]').text().length).toBeGreaterThan(0)
      expect(wrapper.findAll('[data-hc-artwork="random"] svg')).toHaveLength(1)
      expect(wrapper.findAll('[data-art-surface="transparent"]')).toHaveLength(1)
      await wrapper.get('#hf-motion-pause').setValue(true)
      expect((wrapper.get('#hf-motion-pause').element as HTMLInputElement).checked).toBe(true)
      expect(wrapper.get('.hf-motion-control').attributes('for')).toBe('hf-motion-pause')
      expect(wrapper.get('[data-hc-artwork]').attributes('data-hc-interactive')).toBe('true')
      await wrapper.get('[data-hc-artwork]').trigger('click')
      expect(Number(wrapper.get('[data-hc-artwork]').attributes('data-hc-yaw'))).toBe(32)
      await wrapper.get('[data-hc-art-reset]').trigger('click')
      expect(Number(wrapper.get('[data-hc-artwork]').attributes('data-hc-yaw'))).toBe(0)
      expect(wrapper.find('[data-testid="compact-home"]').exists()).toBe(false)
      for (const anchor of wrapper.findAll('a[href^="#"]')) {
        expect(wrapper.find(anchor.attributes('href')).exists()).toBe(true)
      }
      wrapper.unmount()
    },
  )

  it('does not immediately repeat the last sculpture when re-entering the homepage', async () => {
    appStore.cachedPublicSettings = { home_content: '<div data-hc-artwork="random">Fallback</div>' }
    const mountOptions = { global: { stubs: { RouterLink: true, LocaleSwitcher: true, Icon: true } } }
    const first = mount(HomeView, mountOptions)
    await vi.dynamicImportSettled()
    await flushPromises()
    const previous = first.get('[data-hc-artwork]').attributes('data-hc-current-art')
    first.unmount()
    const second = mount(HomeView, mountOptions)
    await vi.dynamicImportSettled()
    await flushPromises()
    expect(second.get('[data-hc-artwork]').attributes('data-hc-current-art')).toBeTruthy()
    expect(second.get('[data-hc-artwork]').attributes('data-hc-current-art')).not.toBe(previous)
    second.unmount()
  })
})
