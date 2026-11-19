<template>
  <div>
    <div v-if="loading && !data" class="flex justify-center py-12"><LoadingSpinner size="md" /></div>
    <div v-else-if="error" class="card p-6 text-sm text-red-600 dark:text-red-400">{{ error }}</div>
    <div v-else-if="data && !data.day" class="card p-6 text-sm text-gray-500 dark:text-gray-400">{{ t('siteRecon.noData') }}</div>
    <div v-else-if="data" data-test="site-recon-anomalies">
      <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.anomalies.sub') }}</p>
      <SrAlerts v-if="!data.scanned" class="mt-3" :items="[{ level: 'info', text: t('siteRecon.anomalies.notScanned') }]" />

      <h3 :class="H2">{{ t('siteRecon.anomalies.needLook') }}</h3>
      <div v-if="!serious.length" class="card p-6 text-center text-sm text-gray-400">
        {{ data.scanned ? t('siteRecon.anomalies.none') : '-' }}
      </div>
      <div v-for="kind in seriousKinds" :key="kind" class="mb-4">
        <div class="mb-1 flex items-center gap-2 text-sm font-semibold text-gray-800 dark:text-gray-100">
          {{ t(`siteRecon.anomalies.kind.${kind}`) }}
          <span
            class="rounded px-1.5 py-0.5 text-[11px]"
            :class="kind === 'free' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'"
          >
            {{ kind === 'free' ? t('siteRecon.anomalies.confirm') : t('siteRecon.anomalies.mustHandle') }}
          </span>
        </div>
        <p class="mb-2 text-xs text-gray-500 dark:text-gray-400">{{ t(`siteRecon.anomalies.help.${kind}`) }}</p>
        <SrTable
          :head="[t('siteRecon.detail.walletAccounts'), t('siteRecon.detail.model'), t('siteRecon.anomalies.count'), t('siteRecon.anomalies.listSum'), t('siteRecon.anomalies.paidSum')]"
          :left="2"
        >
          <tr v-for="r in byKind(kind)" :key="`${r.account_id}-${r.model}`">
            <td :class="TDL">{{ r.account_name }} <span :class="TAG">#{{ r.account_id }}</span></td>
            <td :class="TDL" class="font-mono text-xs">{{ r.model }}</td>
            <td :class="TD">{{ num(r.requests) }}</td>
            <td :class="TD">{{ money(r.list_cost) }}</td>
            <td :class="TD">{{ money(r.actual_cost) }}</td>
          </tr>
        </SrTable>
        <details v-if="samplesOf(kind).length" class="mt-2">
          <summary class="cursor-pointer text-xs text-primary-600 dark:text-primary-400">
            {{ t('siteRecon.anomalies.samples', { n: samplesOf(kind).length }) }}
          </summary>
          <div class="mt-2">
            <SrTable
              :head="[
                t('siteRecon.anomalies.time'), t('siteRecon.detail.walletAccounts'), t('siteRecon.anomalies.userId'), t('siteRecon.detail.model'),
                t('siteRecon.anomalies.tokens'), t('siteRecon.anomalies.listCost'), t('siteRecon.anomalies.paid'), t('siteRecon.anomalies.logId')
              ]"
              :left="2"
            >
              <tr v-for="s in samplesOf(kind)" :key="`${s.account_id}-${s.log_id}`">
                <td :class="TDL" class="whitespace-nowrap text-xs">{{ shortTime(s.created_at) }}</td>
                <td :class="TDL">{{ accountName(s.account_id) }}</td>
                <td :class="TD">{{ s.user_id }}</td>
                <td :class="TD" class="font-mono text-xs">{{ s.model }}</td>
                <td :class="TD">{{ num(s.tokens) }}</td>
                <td :class="TD">{{ money(s.total_cost) }}</td>
                <td :class="TD">{{ money(s.actual_cost) }}</td>
                <td :class="TD">{{ s.log_id }}</td>
              </tr>
            </SrTable>
          </div>
        </details>
      </div>

      <template v-if="variants.length">
        <h3 :class="H2">
          {{ t('siteRecon.anomalies.variants') }}
          <span :class="TAG">{{ t('siteRecon.anomalies.variantsTag') }}</span>
        </h3>
        <details>
          <summary class="cursor-pointer text-sm text-gray-600 dark:text-gray-300">
            {{ t('siteRecon.anomalies.variantsSummary', { n: num(variantTotal) }) }}
          </summary>
          <p :class="NOTE" class="mb-2">{{ t('siteRecon.anomalies.variantsNote') }}</p>
          <SrTable
            :head="[
              t('siteRecon.detail.walletAccounts'), t('siteRecon.anomalies.requested'), t('siteRecon.anomalies.returned'),
              t('siteRecon.anomalies.count'), t('siteRecon.anomalies.listSum'), t('siteRecon.anomalies.paidSum')
            ]"
            :left="3"
          >
            <tr v-for="r in variants" :key="`${r.account_id}-${r.model}-${r.detail}`">
              <td :class="TDL">{{ r.account_name }} <span :class="TAG">#{{ r.account_id }}</span></td>
              <td :class="TDL" class="font-mono text-xs">{{ r.model }}</td>
              <td :class="TDL" class="font-mono text-xs">{{ r.detail || '-' }}</td>
              <td :class="TD">{{ num(r.requests) }}</td>
              <td :class="TD">{{ money(r.list_cost) }}</td>
              <td :class="TD">{{ money(r.actual_cost) }}</td>
            </tr>
          </SrTable>
        </details>
      </template>

      <h3 :class="H2">{{ t('siteRecon.anomalies.recent') }}</h3>
      <SrTable
        :head="[t('siteRecon.overview.dayCol'), t('siteRecon.anomalies.highCol'), t('siteRecon.anomalies.midCol'), t('siteRecon.anomalies.lowCol')]"
        :empty="!data.history.length"
        :empty-text="t('siteRecon.anomalies.recentEmpty')"
      >
        <tr v-for="h in data.history" :key="h.day">
          <td :class="TDL">
            <button type="button" class="text-primary-600 hover:underline dark:text-primary-400" @click="emit('select-day', h.day || '')">{{ h.day }}</button>
          </td>
          <td :class="TD">
            <span v-if="h.high" class="rounded bg-red-100 px-1.5 py-0.5 text-red-700 dark:bg-red-900/30 dark:text-red-300">{{ num(h.high) }}</span>
            <span v-else>0</span>
          </td>
          <td :class="TD">{{ num(h.mid) }}</td>
          <td :class="TD" class="text-gray-400">{{ num(h.low) }}</td>
        </tr>
      </SrTable>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { siteReconAPI, type SiteReconAnomalies, type SiteReconKind } from '@/api/admin/siteRecon'
import { extractApiErrorMessage } from '@/utils/apiError'
import SrAlerts from './SrAlerts.vue'
import SrTable from './SrTable.vue'
import { H2, NOTE, TAG, TD, TDL, money, num, shortTime } from './format'

const props = defineProps<{ day: string }>()
const emit = defineEmits<{
  (e: 'loaded', payload: { day: string; days: string[] }): void
  (e: 'select-day', day: string): void
}>()
const { t } = useI18n()

const data = ref<SiteReconAnomalies | null>(null)
const loading = ref(false)
const error = ref('')

const ORDER: SiteReconKind[] = ['zero_paid', 'unpriced', 'free']
const serious = computed(() => (data.value?.stats ?? []).filter((r) => r.kind !== 'mismatch'))
const seriousKinds = computed(() => ORDER.filter((k) => serious.value.some((r) => r.kind === k)))
const variants = computed(() => (data.value?.stats ?? []).filter((r) => r.kind === 'mismatch'))
const variantTotal = computed(() => variants.value.reduce((s, r) => s + r.requests, 0))

function byKind(kind: SiteReconKind) {
  return serious.value.filter((r) => r.kind === kind)
}

function samplesOf(kind: SiteReconKind) {
  return (data.value?.samples ?? []).filter((s) => s.kind === kind)
}

function accountName(id: number) {
  const hit = data.value?.stats.find((r) => r.account_id === id)
  return hit?.account_name || `#${id}`
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await siteReconAPI.anomalies(props.day || undefined)
    emit('loaded', { day: data.value.day, days: data.value.days })
  } catch (e) {
    error.value = extractApiErrorMessage(e, t('siteRecon.loadFailed'))
  } finally {
    loading.value = false
  }
}

watch(() => props.day, load, { immediate: true })
defineExpose({ reload: load })
</script>
