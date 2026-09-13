<template>
  <DesignedHome
    :is-dark="isDark"
    site-name="zz.naoy"
    :site-logo="logo"
    site-subtitle=""
    doc-url=""
    :is-authenticated="false"
    dashboard-path="/dashboard"
    :show-model-plaza-entry="false"
    @toggle-theme="toggleTheme"
  >
    <template #locale><button class="preview-language" type="button" @click="locale = locale === 'zh' ? 'en' : 'zh'">{{ locale === 'zh' ? 'EN' : '中文' }}</button></template>
  </DesignedHome>
  <aside class="preview-note">{{ locale === 'zh' ? '定稿本地预览 · 正常模式 12 深蓝 / 暗黑模式 15 雾银 · 不改变线上设置' : 'Local preview · Light: 12 Command / Dark: 15 Silver · No live settings changed' }}<a href="page-styles.html">{{ locale === 'zh' ? '返回方案库' : 'All concepts' }} ↗</a></aside>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { watch } from 'vue'
import DesignedHome from '../../frontend/src/components/home/DesignedHome.vue'
import { useSiteTheme } from '../../frontend/src/composables/useSiteTheme'
import logo from '../../frontend/public/logo.svg?url'

const { isDark, toggleTheme } = useSiteTheme()
const { locale } = useI18n()
watch(locale, value => { document.documentElement.lang = value })
</script>

<style>
html, body { margin: 0; padding: 0; }
.preview-language { min-width: 44px; min-height: 44px; border: 0; border-radius: 8px; background: transparent; color: #bcc9e0; font: 13px "Segoe UI", "Microsoft YaHei", sans-serif; cursor: pointer; }
.preview-note { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px 24px; padding: 18px 24px; background: #0d1119; color: #a7b4c9; font: 12px/1.8 "Segoe UI", "Microsoft YaHei", sans-serif; text-align: center; }
.preview-note a { color: #bbd0f5; }
</style>
