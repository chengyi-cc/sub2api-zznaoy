<template>
  <AppLayout>
    <div class="space-y-4">
      <div class="flex flex-wrap items-center gap-3">
        <div class="min-w-[200px] flex-1">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">{{ t('siteRecon.title') }}</h2>
          <p class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.description') }}</p>
        </div>
        <div v-if="isDayTab && days.length" class="flex items-center gap-1" data-test="site-recon-daybar">
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            :disabled="!olderDay"
            :title="t('siteRecon.prevDay')"
            :aria-label="t('siteRecon.prevDay')"
            @click="setDay(olderDay)"
          >
            ‹
          </button>
          <select :value="day" class="input h-8 w-36 text-sm" :aria-label="t('siteRecon.selectDay')" @change="setDay(($event.target as HTMLSelectElement).value)">
            <option v-for="d in days" :key="d" :value="d">{{ d }}</option>
          </select>
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            :disabled="!newerDay"
            :title="t('siteRecon.nextDay')"
            :aria-label="t('siteRecon.nextDay')"
            @click="setDay(newerDay)"
          >
            ›
          </button>
        </div>
      </div>

      <div class="flex gap-1 overflow-x-auto border-b border-gray-200 dark:border-dark-700">
        <button
          v-for="tab in TABS"
          :key="tab"
          type="button"
          class="-mb-px flex items-center gap-1 whitespace-nowrap border-b-2 px-4 py-2 text-sm font-medium"
          :class="activeTab === tab
            ? 'border-primary-600 text-primary-600 dark:border-primary-400 dark:text-primary-400'
            : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'"
          :data-test="`site-recon-tab-${tab}`"
          @click="setTab(tab)"
        >
          {{ t(`siteRecon.tabs.${tab}`) }}
        </button>
      </div>

      <SiteReconOverview v-if="activeTab === 'overview'" :day="day" @loaded="onLoaded" @select-day="setDay" />
      <SiteReconDetail v-else-if="activeTab === 'detail'" :day="day" @loaded="onLoaded" @select-day="setDay" />
      <SiteReconAnomalies v-else-if="activeTab === 'anomalies'" :day="day" @loaded="onLoaded" @select-day="setDay" />
      <SiteReconRuns v-else-if="activeTab === 'runs'" />
      <SiteReconSettings v-else />
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import AppLayout from '@/components/layout/AppLayout.vue'
import SiteReconOverview from '@/components/admin/siteRecon/SiteReconOverview.vue'
import SiteReconDetail from '@/components/admin/siteRecon/SiteReconDetail.vue'
import SiteReconAnomalies from '@/components/admin/siteRecon/SiteReconAnomalies.vue'
import SiteReconRuns from '@/components/admin/siteRecon/SiteReconRuns.vue'
import SiteReconSettings from '@/components/admin/siteRecon/SiteReconSettings.vue'

const TABS = ['overview', 'detail', 'anomalies', 'runs', 'settings'] as const
type Tab = (typeof TABS)[number]

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

const days = ref<string[]>([])
const activeTab = computed<Tab>(() => {
  const q = String(route.query.tab || '')
  return (TABS as readonly string[]).includes(q) ? (q as Tab) : 'overview'
})
const day = computed(() => String(route.query.day || ''))
const isDayTab = computed(() => ['overview', 'detail', 'anomalies'].includes(activeTab.value))
const dayIndex = computed(() => days.value.indexOf(day.value))
const olderDay = computed(() => (dayIndex.value >= 0 && dayIndex.value < days.value.length - 1 ? days.value[dayIndex.value + 1] : ''))
const newerDay = computed(() => (dayIndex.value > 0 ? days.value[dayIndex.value - 1] : ''))

function setTab(tab: Tab) {
  void router.replace({ query: { ...route.query, tab } })
}

function setDay(d: string) {
  if (!d) return
  void router.replace({ query: { ...route.query, day: d, tab: isDayTab.value ? activeTab.value : 'overview' } })
}

function onLoaded(payload: { day: string; days: string[] }) {
  days.value = payload.days
  if (payload.day && payload.day !== day.value) {
    void router.replace({ query: { ...route.query, day: payload.day } })
  }
}
</script>
