import { createApp, defineComponent, h } from 'vue'
import { createI18n } from 'vue-i18n'
import SelectedHomePreview from './SelectedHomePreview.vue'
import zh from '../../frontend/src/i18n/locales/zh/landing'
import en from '../../frontend/src/i18n/locales/en/landing'

const app = createApp(SelectedHomePreview)
app.use(createI18n({ legacy: false, locale: 'zh', fallbackLocale: 'en', messages: { zh, en } }))
app.component('RouterLink', defineComponent({
  inheritAttrs: false,
  props: { to: { type: String, required: true } },
  setup(props, { slots, attrs }) {
    return () => h('a', { ...attrs, href: props.to }, slots.default?.())
  },
}))
app.mount('#app')
