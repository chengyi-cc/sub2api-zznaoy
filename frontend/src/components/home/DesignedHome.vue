<template>
  <div
    id="dh-top"
    class="designed-home hf-site"
    :class="isDark ? 'hf-silver' : 'hf-command'"
    :data-home-design="isDark ? '15-silver' : '12-command'"
  >
    <a class="dh-skip" href="#dh-main">{{ t('home.designed.skip') }}</a>
    <header class="dh-header dh-wrap">
      <a class="dh-brand" href="#dh-top" :title="siteName">
        <img :src="siteLogo || '/logo.svg'" alt="" width="34" height="34" />
        <span>{{ siteName }}</span>
      </a>
      <div class="dh-header-actions">
        <div class="dh-preferences">
          <div class="dh-locale"><slot name="locale" /></div>
          <button
            type="button"
            class="dh-theme"
            data-testid="home-theme-toggle"
            :title="themeLabel"
            :aria-label="themeLabel"
            @click="emit('toggleTheme')"
          >
            <Icon :name="isDark ? 'sun' : 'moon'" size="md" />
          </button>
        </div>
        <div class="dh-account-actions">
          <RouterLink :to="destination" class="dh-login" :class="{ 'dh-account-primary': isAuthenticated || !registrationEnabled }" data-testid="home-header-login">
            {{ isAuthenticated ? t('home.goToDashboard') : t('home.login') }}
            <Icon name="arrowRight" size="sm" />
          </RouterLink>
          <RouterLink v-if="!isAuthenticated && registrationEnabled" to="/register" class="dh-login dh-account-primary" data-testid="home-header-register">
            {{ t('home.designed.register') }}
          </RouterLink>
        </div>
      </div>
    </header>

    <main id="dh-main" tabindex="-1">
      <section class="dh-hero dh-wrap" aria-labelledby="dh-title">
        <div class="dh-hero-copy">
          <div class="dh-copy">
            <p class="dh-eyebrow dh-hero-eyebrow"><span class="dh-signal" aria-hidden="true"></span>{{ t(`home.designed.${mode}.eyebrow`) }}</p>
            <h1 id="dh-title"><span>{{ t(`home.designed.${mode}.title`) }}</span><strong>{{ t(`home.designed.${mode}.accent`) }}</strong></h1>
            <p class="dh-lead">{{ heroDescription }}</p>
            <div class="dh-actions">
              <RouterLink class="dh-button dh-primary" :to="destination" data-testid="home-primary-action">
                {{ isAuthenticated ? t('home.goToDashboard') : t('home.designed.signIn') }}
                <Icon name="arrowRight" size="sm" />
              </RouterLink>
              <RouterLink v-if="!isAuthenticated && registrationEnabled" class="dh-button dh-secondary" to="/register" data-testid="home-register-action">{{ t('home.designed.register') }}</RouterLink>
            </div>
            <ul class="dh-benefits">
              <li v-for="feature in features" :key="feature.id"><Icon :name="feature.icon" size="sm" />{{ t(`home.designed.${feature.id}.short`) }}</li>
            </ul>
          </div>
        </div>

        <div ref="artworkRoot" class="dh-figure hf-stage">
          <input id="dh-motion-pause" class="hf-motion-pause" type="checkbox" :aria-label="t('home.designed.pause')" />
          <div v-once class="hf-art" data-hc-artwork="random"><img class="dh-art-fallback" :src="artworkFallback" alt="" /></div>
          <p class="hf-interaction-hint">{{ t('home.designed.interaction') }}<span>{{ t('home.designed.keyboard') }}</span></p>
          <div class="hf-stage-bottom">
            <span class="dh-art-name"><i aria-hidden="true"></i><span data-hc-artwork-name>{{ t('home.designed.sculpture') }}</span></span>
            <span class="hf-art-controls">
              <button type="button" class="hf-art-reset" data-hc-art-reset :aria-label="t('home.designed.resetLabel')">{{ t('home.designed.reset') }} ↺</button>
              <label class="hf-motion-control" for="dh-motion-pause"><span class="hf-motion-running">{{ t('home.designed.pause') }} Ⅱ</span><span class="hf-motion-stopped">{{ t('home.designed.resume') }} ▷</span></label>
            </span>
          </div>
        </div>
      </section>

      <section id="dh-models" class="dh-models dh-wrap" aria-labelledby="dh-models-title">
        <div><h2 id="dh-models-title">{{ t('home.designed.modelsTitle') }}</h2><p class="dh-model-note">{{ t('home.designed.modelsNote') }}</p></div>
        <div class="dh-model-names"><span>Claude</span><span>GPT</span><span>Gemini</span></div>
      </section>

      <section id="dh-capabilities" class="dh-section dh-wrap" aria-labelledby="dh-features-title">
        <div class="dh-section-heading"><h2 id="dh-features-title">{{ t('home.designed.featuresTitle') }}</h2><p>{{ t('home.designed.featuresDescription') }}</p></div>
        <div class="dh-features">
          <article v-for="(feature, index) in features" :key="feature.id">
            <div class="dh-feature-top"><Icon :name="feature.icon" size="lg" /><span>0{{ index + 1 }}</span></div>
            <h3>{{ t(`home.designed.${feature.id}.title`) }}</h3><p>{{ t(`home.designed.${feature.id}.description`) }}</p>
          </article>
        </div>
      </section>

      <section class="dh-faq dh-wrap" aria-labelledby="dh-faq-title">
        <h2 id="dh-faq-title">{{ t('home.designed.faqTitle') }}</h2>
        <div><details v-for="question in [1, 2, 3]" :key="question"><summary>{{ t(`home.designed.faq${question}.question`) }}<Icon name="plus" size="sm" /></summary><p>{{ t(`home.designed.faq${question}.answer`) }}</p></details></div>
      </section>

    </main>

    <footer class="dh-footer dh-wrap"><a class="dh-footer-brand" href="#dh-top">{{ siteName }}</a><p>© {{ currentYear }} {{ siteName }}</p><div><RouterLink v-if="showModelPlazaEntry" to="/model-plaza">{{ t('home.designed.modelPlaza') }}</RouterLink><RouterLink to="/key-usage">{{ t('home.designed.usage') }}</RouterLink><a v-if="docUrl" :href="docUrl" target="_blank" rel="noopener noreferrer">{{ t('home.docs') }}</a></div></footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import Icon from '@/components/icons/Icon.vue'
import artworkFallback from '@/assets/home-artwork-fallback.svg'
import '@/assets/home-artwork.css'
import './designed-home.css'

const props = defineProps<{
  isDark: boolean
  siteName: string
  siteLogo: string
  siteSubtitle: string
  docUrl: string
  isAuthenticated: boolean
  dashboardPath: string
  showModelPlazaEntry: boolean
  registrationEnabled: boolean
}>()
const emit = defineEmits<{ toggleTheme: [] }>()
const { t } = useI18n()
const mode = computed(() => props.isDark ? 'silver' : 'command')
const destination = computed(() => props.isAuthenticated ? props.dashboardPath : '/login')
const heroDescription = computed(() => {
  const subtitle = props.siteSubtitle.trim()
  return subtitle && subtitle.toLowerCase() !== props.siteName.trim().toLowerCase()
    ? subtitle
    : t(`home.designed.${mode.value}.description`)
})
const themeLabel = computed(() => props.isDark ? t('home.switchToLight') : t('home.switchToDark'))
const currentYear = new Date().getFullYear()
const artworkRoot = ref<HTMLElement | null>(null)
const features = [{ id: 'connect', icon: 'bolt' }, { id: 'organize', icon: 'key' }, { id: 'understand', icon: 'chart' }] as const
let disposed = false
let cleanupArtwork: (() => void) | undefined

onMounted(async () => {
  try {
    const { initializeHomeArtwork } = await import('@/utils/homeArtwork')
    if (!disposed && artworkRoot.value) cleanupArtwork = initializeHomeArtwork(artworkRoot.value)
  } catch {
    return
  }
})
onBeforeUnmount(() => { disposed = true; cleanupArtwork?.() })
</script>
