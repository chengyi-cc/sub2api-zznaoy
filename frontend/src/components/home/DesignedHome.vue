<template>
  <div
    id="dh-top"
    class="designed-home"
    :class="{ 'dh-dark': isDark }"
    :data-home-theme="isDark ? 'dark' : 'light'"
  >
    <a class="dh-skip" href="#dh-main">{{ t('home.designed.skip') }}</a>

    <header class="dh-bar">
      <a class="dh-brand" href="#dh-top" :title="siteName">
        <img :src="logo" alt="" width="32" height="32" />
        <span>{{ siteName }}</span>
      </a>
      <nav class="dh-nav" :aria-label="t('home.designed.navigation')">
        <a v-if="docUrl" :href="docUrl" target="_blank" rel="noopener noreferrer" data-testid="home-header-docs">{{ t('home.docs') }}</a>
        <RouterLink to="/key-usage">{{ t('home.designed.usage') }}</RouterLink>
        <RouterLink v-if="showModelPlazaEntry" to="/model-plaza">{{ t('home.designed.modelPlaza') }}</RouterLink>
      </nav>
      <div class="dh-bar-end">
        <div class="dh-locale"><slot name="locale" /></div>
        <button
          type="button"
          class="dh-icon-btn"
          data-testid="home-theme-toggle"
          :title="themeLabel"
          :aria-label="themeLabel"
          @click="emit('toggleTheme')"
        >
          <Icon :name="isDark ? 'sun' : 'moon'" size="md" />
        </button>
        <RouterLink
          :to="headerLoginTarget"
          class="dh-bar-link"
          :class="{ 'dh-account-primary': !showRegister }"
          data-testid="home-header-login"
        >{{ isAuthenticated ? t('home.goToDashboard') : t('home.login') }}</RouterLink>
        <RouterLink v-if="showRegister" to="/register" class="dh-bar-link dh-account-primary" data-testid="home-header-register">
          {{ t('home.designed.register') }}
        </RouterLink>
      </div>
    </header>

    <main id="dh-main" class="dh-hero" tabindex="-1" aria-labelledby="dh-title">
      <p class="dh-eyebrow"><i aria-hidden="true"></i>{{ t('home.designed.eyebrow') }}</p>
      <h1 id="dh-title">{{ t('home.designed.titleLead') }}<span class="dh-grad">{{ t('home.designed.titleAccent') }}</span></h1>
      <p class="dh-lead">{{ heroDescription }}</p>
      <div class="dh-actions">
        <RouterLink class="dh-pill" :to="primaryTarget" data-testid="home-primary-action">
          {{ primaryLabel }}
          <span class="dh-pill-arrow" aria-hidden="true"><Icon name="arrowRight" size="sm" /></span>
        </RouterLink>
        <RouterLink v-if="showRegister" class="dh-pill dh-pill-ghost" to="/login" data-testid="home-signin-action">
          {{ t('home.designed.signIn') }}
        </RouterLink>
      </div>

      <div class="dh-flow-wrap">
        <!-- 宽屏：完整线路图 -->
        <svg class="dh-flow" :viewBox="FLOW_VIEWBOX" role="img" :aria-label="diagramLabel" data-testid="home-flow">
          <defs>
            <linearGradient id="dh-flow-gradient" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1200" y2="0">
              <stop offset="0" class="dh-stop-1" />
              <stop offset="0.5" class="dh-stop-2" />
              <stop offset="1" class="dh-stop-3" />
            </linearGradient>
            <clipPath id="dh-flow-logo-clip">
              <rect :x="FLOW_HUB.cx - 30" :y="FLOW_HUB.cy - 52" width="60" height="60" rx="16" />
            </clipPath>
          </defs>
          <text class="dh-cap" x="22" y="-10">{{ t('home.designed.toolsCaption') }}</text>
          <text class="dh-cap" x="1180" y="-10" text-anchor="end">{{ t('home.designed.modelsCaption') }}</text>
          <circle class="dh-ring" :cx="FLOW_HUB.cx" :cy="FLOW_HUB.cy" r="96" />
          <circle class="dh-ring dh-ring-late" :cx="FLOW_HUB.cx" :cy="FLOW_HUB.cy" r="96" />
          <path v-for="(link, i) in layout.links" :key="`b${i}`" class="dh-line" :d="link.d" />
          <path
            v-for="(link, i) in layout.links"
            :key="`s${i}`"
            class="dh-streak"
            pathLength="100"
            :d="link.d"
            :style="{ animationDelay: link.delay }"
          />
          <rect
            class="dh-hub"
            :x="FLOW_HUB.cx - FLOW_HUB.halfWidth"
            :y="FLOW_HUB.cy - FLOW_HUB.halfHeight"
            :width="FLOW_HUB.halfWidth * 2"
            :height="FLOW_HUB.halfHeight * 2"
            rx="32"
          />
          <image
            :href="logo"
            :x="FLOW_HUB.cx - 30"
            :y="FLOW_HUB.cy - 52"
            width="60"
            height="60"
            preserveAspectRatio="xMidYMid slice"
            clip-path="url(#dh-flow-logo-clip)"
          />
          <text class="dh-hub-host" :x="FLOW_HUB.cx" :y="FLOW_HUB.cy + 40" :style="{ fontSize: `${hostFontSize}px` }">{{ host }}</text>
          <rect class="dh-key" :x="FLOW_HUB.cx - 62" :y="FLOW_HUB.cy + FLOW_HUB.halfHeight + 16" width="124" height="28" rx="14" />
          <text class="dh-key-text" :x="FLOW_HUB.cx" :y="FLOW_HUB.cy + FLOW_HUB.halfHeight + 34.5">{{ keySample }}</text>
          <g class="dh-chips">
            <g v-for="(tool, i) in tools" :key="tool">
              <rect class="dh-chip" :x="layout.tools[i].x" :y="layout.tools[i].y" :width="layout.tools[i].width" :height="layout.tools[i].height" rx="13" />
              <text class="dh-glyph" :x="layout.tools[i].x + 16" :y="layout.tools[i].y + 25.5">&gt;_</text>
              <text class="dh-chip-label" :x="layout.tools[i].x + 46" :y="layout.tools[i].y + 26">{{ tool }}</text>
            </g>
            <g v-for="(model, i) in models" :key="model.label" :data-model="model.icon ?? 'more'">
              <rect
                class="dh-chip"
                :class="{ 'dh-chip-more': !model.icon }"
                :x="layout.models[i].x"
                :y="layout.models[i].y"
                :width="layout.models[i].width"
                :height="layout.models[i].height"
                rx="12"
              />
              <template v-if="model.glyph">
                <svg :x="layout.models[i].x + 14" :y="layout.models[i].y + 9" width="20" height="20" viewBox="0 0 24 24" :fill="model.glyph.fill" fill-rule="evenodd" aria-hidden="true">
                  <path v-for="(p, j) in model.glyph.paths" :key="j" :d="p" />
                </svg>
              </template>
              <text v-else class="dh-glyph" :x="layout.models[i].x + 17" :y="layout.models[i].y + 24">+</text>
              <text class="dh-chip-label" :class="{ 'dh-chip-label-dim': !model.icon }" :x="layout.models[i].x + 44" :y="layout.models[i].y + 24">{{ model.label }}</text>
            </g>
          </g>
        </svg>

        <!-- 窄屏：纵向简化版 -->
        <div class="dh-flow-compact" role="img" :aria-label="diagramLabel">
          <ul class="dh-cf-tools">
            <li v-for="tool in tools" :key="tool"><span class="dh-cf-glyph">&gt;_</span>{{ tool }}</li>
          </ul>
          <span class="dh-cf-line"></span>
          <div class="dh-cf-hub">
            <img :src="logo" alt="" width="44" height="44" />
            <strong>{{ host }}</strong>
            <span>{{ keySample }}</span>
          </div>
          <span class="dh-cf-line dh-cf-line-late"></span>
          <ul class="dh-cf-models">
            <li v-for="model in models" :key="model.label" :class="{ 'dh-cf-more': !model.icon }">
              <svg v-if="model.glyph" viewBox="0 0 24 24" :fill="model.glyph.fill" fill-rule="evenodd" aria-hidden="true">
                <path v-for="(p, j) in model.glyph.paths" :key="j" :d="p" />
              </svg>
              <span v-else class="dh-cf-glyph">+</span>
              {{ model.label }}
            </li>
          </ul>
        </div>
      </div>
    </main>

    <footer class="dh-footer">© {{ currentYear }} {{ siteName }}</footer>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import Icon from '@/components/icons/Icon.vue'
import {
  FLOW_HUB,
  FLOW_VIEWBOX,
  HOME_FLOW_MODELS,
  HOME_FLOW_TOOLS,
  buildHomeFlowLayout,
  flowIcon
} from './homeFlowLayout'
import './designed-home.css'

const props = defineProps<{
  isDark: boolean
  siteName: string
  siteLogo: string
  siteSubtitle: string
  docUrl: string
  apiBaseUrl?: string
  isAuthenticated: boolean
  dashboardPath: string
  showModelPlazaEntry: boolean
  registrationEnabled: boolean
}>()
const emit = defineEmits<{ toggleTheme: [] }>()
const { t } = useI18n()

const keySample = 'sk-••••7f3a'
const currentYear = new Date().getFullYear()
const logo = computed(() => props.siteLogo || '/logo.svg')
const themeLabel = computed(() => props.isDark ? t('home.switchToLight') : t('home.switchToDark'))

const showRegister = computed(() => !props.isAuthenticated && props.registrationEnabled)
const headerLoginTarget = computed(() => props.isAuthenticated ? props.dashboardPath : '/login')
const primaryTarget = computed(() => {
  if (props.isAuthenticated) return props.dashboardPath
  return props.registrationEnabled ? '/register' : '/login'
})
const primaryLabel = computed(() => {
  if (props.isAuthenticated) return t('home.goToDashboard')
  return props.registrationEnabled ? t('home.designed.register') : t('home.designed.signIn')
})

const heroDescription = computed(() => {
  const subtitle = props.siteSubtitle.trim()
  return subtitle && subtitle.toLowerCase() !== props.siteName.trim().toLowerCase()
    ? subtitle
    : t('home.designed.description')
})

/** 网关中心显示的域名：优先后台配置的 API 地址，其次当前站点。 */
const host = computed(() => {
  const configured = (props.apiBaseUrl || '').trim()
  if (configured) {
    try { return new URL(configured).host } catch { /* 非法地址时回退 */ }
  }
  return typeof window !== 'undefined' ? window.location.host : ''
})
// 网关方块内宽约 120，域名过长时缩小字号避免溢出
const hostFontSize = computed(() => Math.max(10, Math.min(15.5, 120 / Math.max(1, host.value.length * 0.6))))

const tools = computed(() => [...HOME_FLOW_TOOLS, t('home.designed.yourApp')])
const models = computed(() => [
  ...HOME_FLOW_MODELS.map(model => ({
    icon: model.icon as string | null,
    label: model.labelKey ? t(`home.designed.models.${model.labelKey}`) : model.label ?? '',
    glyph: flowIcon(model.icon)
  })),
  { icon: null, label: t('home.designed.moreModels'), glyph: null }
])
const layout = computed(() => buildHomeFlowLayout(tools.value.length, models.value.length))
const diagramLabel = computed(() => t('home.designed.diagramLabel', {
  host: host.value,
  models: models.value.filter(model => model.icon).map(model => model.label).join(t('home.designed.listSeparator'))
}))
</script>
