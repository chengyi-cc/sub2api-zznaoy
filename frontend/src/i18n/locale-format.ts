// 纯粹的 locale 常量与格式化 helper，**不 import vue-i18n、不创建 i18n 实例**。
//
// 为什么单独一个文件：`@/i18n`（index.ts）在模块顶层就调用 `createI18n()`。
// 组件只要 import 它，任何只 stub 了 `useI18n` 的 vue-i18n 测试 mock 都会炸
// （"No createI18n export is defined on the vue-i18n mock"）。Intl 格式化需要的
// 只是「locale 字符串 → BCP 47 标签」这层映射，与 i18n 实例无关，所以拆出来。
//
// 需要 Intl 标签的组件请从这里 import，而不是从 `@/i18n`。

export type LocaleCode = 'en' | 'ru' | 'zh'

export const DEFAULT_LOCALE: LocaleCode = 'en'

export const INTL_LOCALE_MAP: Record<LocaleCode, string> = {
  en: 'en-US',
  ru: 'ru-RU',
  zh: 'zh-CN'
}

export function isLocaleCode(value: string): value is LocaleCode {
  return value === 'en' || value === 'ru' || value === 'zh'
}

/** 把 locale code 映射成 Intl / toLocaleString 用的 BCP 47 标签。 */
export function getIntlLocaleFor(locale: string): string {
  return isLocaleCode(locale) ? INTL_LOCALE_MAP[locale] : INTL_LOCALE_MAP[DEFAULT_LOCALE]
}

/** 列表分隔符：中文用顿号，其它用逗号加空格。 */
export function getLocaleListSeparatorFor(locale: string): string {
  return locale === 'zh' ? '、' : ', '
}
