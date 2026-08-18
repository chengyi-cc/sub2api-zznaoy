import { createI18n } from 'vue-i18n'

import {
  DEFAULT_LOCALE,
  getIntlLocaleFor,
  getLocaleListSeparatorFor,
  isLocaleCode,
  type LocaleCode
} from './locale-format'

type LocaleMessages = Record<string, any>

const LOCALE_KEY = 'sub2api_locale'

// 纯 locale 映射与判定住在 ./locale-format（不 import vue-i18n），这样只需要
// Intl 标签的组件不必被本文件顶层的 createI18n 副作用连坐。
export { DEFAULT_LOCALE, INTL_LOCALE_MAP, isLocaleCode, type LocaleCode } from './locale-format'

const localeLoaders: Record<LocaleCode, () => Promise<{ default: LocaleMessages }>> = {
  en: () => import('./locales/en'),
  ru: () => import('./locales/ru'),
  zh: () => import('./locales/zh')
}

function getDefaultLocale(): LocaleCode {
  const saved = localStorage.getItem(LOCALE_KEY)
  if (saved && isLocaleCode(saved)) {
    return saved
  }

  const browserLang = navigator.language.toLowerCase()
  if (browserLang.startsWith('zh')) {
    return 'zh'
  }
  if (browserLang.startsWith('ru')) {
    return 'ru'
  }

  return DEFAULT_LOCALE
}

export const i18n = createI18n({
  legacy: false,
  locale: getDefaultLocale(),
  fallbackLocale: DEFAULT_LOCALE,
  messages: {},
  // Some onboarding content intentionally renders trusted HTML.
  warnHtmlMessage: false
})

const loadedLocales = new Set<LocaleCode>()

export async function loadLocaleMessages(locale: LocaleCode): Promise<void> {
  if (loadedLocales.has(locale)) {
    return
  }

  const loader = localeLoaders[locale]
  const module = await loader()
  i18n.global.setLocaleMessage(locale, module.default)
  loadedLocales.add(locale)
}

export async function initI18n(): Promise<void> {
  const current = getLocale()
  await loadLocaleMessages(current)
  document.documentElement.setAttribute('lang', current)
}

export async function setLocale(locale: string): Promise<void> {
  if (!isLocaleCode(locale)) {
    return
  }

  await loadLocaleMessages(locale)
  i18n.global.locale.value = locale
  localStorage.setItem(LOCALE_KEY, locale)
  document.documentElement.setAttribute('lang', locale)

  // 同步更新浏览器页签标题，使其跟随语言切换
  const { resolveRouteDocumentTitle } = await import('@/router/title')
  const { default: router } = await import('@/router')
  const { useAppStore } = await import('@/stores/app')
  const { useAuthStore } = await import('@/stores/auth')
  const { useAdminSettingsStore } = await import('@/stores/adminSettings')
  const route = router.currentRoute.value
  const appStore = useAppStore()
  const authStore = useAuthStore()
  const adminSettingsStore = useAdminSettingsStore()
  const customMenuItems = [
    ...(appStore.cachedPublicSettings?.custom_menu_items ?? []),
    ...(authStore.isAdmin ? adminSettingsStore.customMenuItems : []),
  ]
  document.title = resolveRouteDocumentTitle(route, appStore.siteName, customMenuItems)
}

export function getLocale(): LocaleCode {
  const current = i18n.global.locale.value
  return isLocaleCode(current) ? current : DEFAULT_LOCALE
}

// \u8fd9\u4e24\u4e2a\u4fdd\u7559\u300c\u4e0d\u4f20\u53c2 = \u53d6\u5f53\u524d locale\u300d\u7684\u4fbf\u5229\u91cd\u8f7d\uff0c\u56e0\u4e3a\u9ed8\u8ba4\u503c\u4f9d\u8d56 i18n \u5b9e\u4f8b\u3002
// \u5df2\u7ecf\u62ff\u5230 locale \u7684\u8c03\u7528\u65b9\u8bf7\u76f4\u63a5\u7528 ./locale-format \u7684 *For \u7248\u672c\u3002
export function getIntlLocale(locale: string = getLocale()): string {
  return getIntlLocaleFor(locale)
}

export function getLocaleListSeparator(locale: string = getLocale()): string {
  return getLocaleListSeparatorFor(locale)
}

export const availableLocales = [
  { code: 'en', name: 'English', flag: '\uD83C\uDDFA\uD83C\uDDF8' },
  { code: 'ru', name: '\u0420\u0443\u0441\u0441\u043A\u0438\u0439', flag: '\uD83C\uDDF7\uD83C\uDDFA' },
  { code: 'zh', name: '\u4E2D\u6587', flag: '\uD83C\uDDE8\uD83C\uDDF3' }
] as const

export default i18n
