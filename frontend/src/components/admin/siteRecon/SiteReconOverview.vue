<template>
  <div>
    <div v-if="loading && !data" class="flex justify-center py-12"><LoadingSpinner size="md" /></div>
    <div v-else-if="error" class="card p-6 text-sm text-red-600 dark:text-red-400">{{ error }}</div>
    <div v-else-if="data && !data.day" class="card p-6 text-sm text-gray-500 dark:text-gray-400" data-test="site-recon-empty">
      {{ t('siteRecon.noData') }}
    </div>
    <div v-else-if="data" class="space-y-4" data-test="site-recon-overview">
      <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.unit') }}</p>
      <SrAlerts :items="data.alerts" />

      <div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div class="card p-4">
          <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.overview.revenue') }}</div>
          <div class="mt-1 text-xl font-semibold tabular-nums text-gray-900 dark:text-white">{{ money(data.totals.rev) }}</div>
          <div class="mt-1 text-xs" :class="signedClass(deltaRev) || 'text-gray-500'" v-if="deltaRev !== null">
            {{ t('siteRecon.overview.vsPrev', { value: fmtDelta(deltaRev) }) }}
          </div>
        </div>
        <div class="card p-4">
          <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.overview.cost') }}</div>
          <div class="mt-1 text-xl font-semibold tabular-nums text-gray-900 dark:text-white">{{ money(data.totals.cost) }}</div>
          <div class="mt-1 text-xs text-gray-500" v-if="deltaCost !== null">
            {{ t('siteRecon.overview.vsPrev', { value: fmtDelta(deltaCost) }) }}
          </div>
        </div>
        <div class="card p-4 ring-1 ring-primary-200 dark:ring-primary-900/50">
          <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.overview.profit') }}</div>
          <div class="mt-1 text-xl font-semibold tabular-nums" :class="signedClass(profit) || 'text-gray-900 dark:text-white'" data-test="site-recon-profit">
            {{ money(profit) }}
          </div>
          <div class="mt-1 text-xs" :class="signedClass(deltaProfit) || 'text-gray-500'" v-if="deltaProfit !== null">
            {{ t('siteRecon.overview.vsPrev', { value: fmtDelta(deltaProfit) }) }}
          </div>
        </div>
        <div class="card p-4">
          <div class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.overview.margin') }}</div>
          <div class="mt-1 text-xl font-semibold tabular-nums text-gray-900 dark:text-white">{{ pct(profit, data.totals.rev) }}</div>
          <div class="mt-1 text-xs text-gray-500" v-if="data.totals.all">
            {{ t('siteRecon.overview.coverage', { value: pct(data.totals.rev, data.totals.all) }) }}
          </div>
        </div>
      </div>

      <template v-if="data.trend.length >= 2">
        <h3 :class="H2">{{ t('siteRecon.overview.trendTitle', { n: data.trend.length }) }}</h3>
        <div class="card p-4">
          <p class="mb-2 text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.overview.trendHint') }}</p>
          <div class="h-64"><Line :data="chartData" :options="chartOptions" /></div>
        </div>
      </template>

      <h3 :class="H2">
        {{ t('siteRecon.overview.groupsTitle') }}
        <span :class="TAG">{{ t('siteRecon.overview.groupsTag') }}</span>
      </h3>
      <div class="card overflow-x-auto">
        <div v-if="!data.groups.length" class="px-4 py-6 text-center text-sm text-gray-400">{{ t('siteRecon.overview.noGroups') }}</div>
        <table v-else class="min-w-full text-sm">
          <thead class="bg-gray-50 dark:bg-dark-800/40">
            <tr class="text-xs text-gray-500 dark:text-gray-400">
              <th :class="THL">{{ t('siteRecon.overview.groupCol') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.rateCol') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.revenue') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.cost') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.profit') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.margin') }}</th>
              <th :class="TH">{{ t('siteRecon.overview.reqCol') }}</th>
            </tr>
          </thead>
          <tbody v-for="g in data.groups" :key="g.name" class="border-t border-gray-100 dark:border-dark-700/60">
            <tr class="cursor-pointer hover:bg-gray-50 dark:hover:bg-dark-800/40" @click="toggle(g.name)">
              <td :class="TDL">
                <div class="flex flex-wrap items-center gap-1.5">
                  <span class="text-gray-400">{{ open[g.name] ? '▾' : '▸' }}</span>
                  <span class="font-medium">{{ g.name }}</span>
                  <span :class="TAG">{{ t('siteRecon.overview.accountsCount', { n: g.accounts.length }) }}</span>
                  <span v-if="g.missing" :class="TAG">{{ t('siteRecon.overview.missingTag', { value: money(g.missing) }) }}</span>
                </div>
                <div v-if="g.margin !== null" class="mt-1 h-1 w-32 overflow-hidden rounded bg-gray-100 dark:bg-dark-700">
                  <div class="h-full bg-emerald-500" :style="{ width: barWidth(g.margin) }"></div>
                </div>
              </td>
              <td :class="TD">{{ ratesText(g.rates) }}</td>
              <td :class="TD">{{ money(g.rev) }}</td>
              <td :class="TD">{{ money(g.profit === null ? null : g.cost) }}</td>
              <td :class="[TD, signedClass(g.profit)]">{{ money(g.profit) }}</td>
              <td :class="TD">{{ pct(g.profit, g.known) }}</td>
              <td :class="TD">{{ num(g.req) }}</td>
            </tr>
            <template v-if="open[g.name]">
              <tr v-for="a in g.accounts" :key="a.id" class="bg-gray-50/60 text-xs dark:bg-dark-800/30">
                <td :class="TDL" class="pl-9">
                  {{ a.name }} <span :class="TAG">#{{ a.id }}</span> <span :class="TAG">{{ a.upstream }}</span>
                </td>
                <td :class="TD">{{ ratesText(a.rates) }}</td>
                <td :class="TD">{{ money(a.rev) }}</td>
                <td :class="TD">{{ money(a.cost) }}</td>
                <td :class="[TD, signedClass(a.profit)]">{{ money(a.profit) }}</td>
                <td :class="TD">{{ pct(a.profit, a.rev) }}</td>
                <td :class="TD">{{ num(a.req) }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
      <p :class="NOTE">{{ t('siteRecon.overview.groupsNote') }}</p>

      <h3 :class="H2">{{ t('siteRecon.overview.last7') }}</h3>
      <SrTable
        :head="[t('siteRecon.overview.dayCol'), t('siteRecon.overview.revenue'), t('siteRecon.overview.cost'), t('siteRecon.overview.profit'), t('siteRecon.overview.margin')]"
        :empty="!last7.length"
      >
        <tr v-for="r in last7" :key="r.day">
          <td :class="TDL">
            <button type="button" class="text-primary-600 hover:underline dark:text-primary-400" @click="emit('select-day', r.day)">{{ r.day }}</button>
          </td>
          <td :class="TD">{{ money(r.rev) }}</td>
          <td :class="TD">{{ money(r.cost) }}</td>
          <td :class="[TD, signedClass(r.rev - r.cost)]">{{ money(r.rev - r.cost) }}</td>
          <td :class="TD">{{ pct(r.rev - r.cost, r.rev) }}</td>
        </tr>
      </SrTable>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler } from 'chart.js'
import { Line } from 'vue-chartjs'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { siteReconAPI, type SiteReconOverview } from '@/api/admin/siteRecon'
import { extractApiErrorMessage } from '@/utils/apiError'
import SrAlerts from './SrAlerts.vue'
import SrTable from './SrTable.vue'
import { H2, NOTE, TAG, TD, TDL, TH, THL, delta, money, num, pct, ratesText, signedClass } from './format'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const props = defineProps<{ day: string }>()
const emit = defineEmits<{
  (e: 'loaded', payload: { day: string; days: string[] }): void
  (e: 'select-day', day: string): void
}>()
const { t } = useI18n()

const data = ref<SiteReconOverview | null>(null)
const loading = ref(false)
const error = ref('')
const open = reactive<Record<string, boolean>>({})

const profit = computed(() => (data.value ? data.value.totals.rev - data.value.totals.cost : null))
const prevProfit = computed(() => (data.value ? data.value.prev.rev - data.value.prev.cost : 0))
const deltaRev = computed(() => (data.value ? delta(data.value.totals.rev, data.value.prev.rev, data.value.prev.has) : null))
const deltaCost = computed(() => (data.value ? delta(data.value.totals.cost, data.value.prev.cost, data.value.prev.has) : null))
const deltaProfit = computed(() =>
  data.value && profit.value !== null ? delta(profit.value, prevProfit.value, data.value.prev.has) : null
)
const last7 = computed(() => (data.value ? [...data.value.trend].slice(-7).reverse() : []))

function fmtDelta(v: number) {
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
}

function barWidth(margin: number) {
  return `${Math.max(0, Math.min(100, margin * 100)).toFixed(0)}%`
}

function toggle(name: string) {
  open[name] = !open[name]
}

const chartData = computed(() => {
  const trend = data.value?.trend ?? []
  return {
    labels: trend.map((r) => r.day.slice(5)),
    datasets: [
      {
        label: t('siteRecon.overview.revenue'),
        data: trend.map((r) => Number(r.rev.toFixed(2))),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 2
      },
      {
        label: t('siteRecon.overview.cost'),
        data: trend.map((r) => Number(r.cost.toFixed(2))),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.05)',
        fill: false,
        tension: 0.3,
        pointRadius: 2
      }
    ]
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { intersect: false, mode: 'index' as const },
  plugins: { legend: { position: 'top' as const } },
  scales: { y: { beginAtZero: true } }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await siteReconAPI.overview(props.day || undefined)
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
