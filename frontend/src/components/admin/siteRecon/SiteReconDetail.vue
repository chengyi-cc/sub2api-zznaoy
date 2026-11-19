<template>
  <div>
    <div v-if="loading && !data" class="flex justify-center py-12"><LoadingSpinner size="md" /></div>
    <div v-else-if="error" class="card p-6 text-sm text-red-600 dark:text-red-400">{{ error }}</div>
    <div v-else-if="data && !data.day" class="card p-6 text-sm text-gray-500 dark:text-gray-400">{{ t('siteRecon.noData') }}</div>
    <div v-else-if="data" data-test="site-recon-detail">
      <p class="text-xs text-gray-500 dark:text-gray-400">{{ t('siteRecon.detail.sub') }}</p>

      <h3 :class="H2">{{ t('siteRecon.detail.accounts') }}</h3>
      <SrTable
        :head="[
          t('siteRecon.detail.account'), t('siteRecon.overview.revenue'), t('siteRecon.overview.cost'), t('siteRecon.overview.profit'),
          t('siteRecon.overview.margin'), t('siteRecon.detail.ourRate'), t('siteRecon.detail.upRate'), t('siteRecon.detail.ourReq'),
          t('siteRecon.detail.upReq'), t('siteRecon.detail.reqDiff'), t('siteRecon.detail.leak')
        ]"
        :empty="!busyAccounts.length"
      >
        <tr v-for="a in busyAccounts" :key="a.account_id">
          <td :class="TDL">
            {{ a.account_name }} <span :class="TAG">#{{ a.account_id }}</span>
            <div class="text-xs text-gray-400">{{ a.upstream }}</div>
          </td>
          <td :class="TD">{{ money(a.revenue) }}</td>
          <td :class="TD">{{ money(a.cost) }}</td>
          <td :class="[TD, signedClass(a.profit)]">{{ money(a.profit) }}</td>
          <td :class="TD">{{ pct(a.profit, a.revenue) }}</td>
          <td :class="TD">{{ ratio(a.revenue, a.our_list_cost) }}</td>
          <td :class="TD">{{ ratio(a.cost, a.up_list_cost) }}</td>
          <td :class="TD">{{ num(a.our_requests) }}</td>
          <td :class="TD">{{ num(a.up_requests) }}</td>
          <td :class="[TD, diffClass(diff(a.our_requests, a.up_requests))]">{{ signedNum(diff(a.our_requests, a.up_requests)) }}</td>
          <td :class="TD">{{ money(a.leak) }}</td>
        </tr>
      </SrTable>
      <p :class="NOTE">
        {{ t('siteRecon.detail.accountsNote') }}
        <template v-if="idleCount">{{ t('siteRecon.detail.idleNote', { n: idleCount }) }}</template>
      </p>

      <template v-if="data.wallets.length">
        <h3 :class="H2">{{ t('siteRecon.detail.wallets') }}</h3>
        <SrTable
          :head="[t('siteRecon.detail.upstream'), t('siteRecon.detail.walletAccounts'), t('siteRecon.detail.balance'), t('siteRecon.detail.burn'), t('siteRecon.detail.daysLeft'), t('siteRecon.detail.status')]"
          :left="2"
        >
          <tr v-for="(w, i) in data.wallets" :key="i">
            <td :class="TDL">{{ w.host }}</td>
            <td :class="TDL" class="max-w-xs text-xs">{{ w.names.join('、') }}</td>
            <td :class="TD">{{ money(w.balance) }}</td>
            <td :class="TD">{{ money(w.burn || null) }}</td>
            <td :class="TD">{{ w.days_left === null ? '-' : t('siteRecon.detail.daysUnit', { n: w.days_left.toFixed(1) }) }}</td>
            <td :class="TD">
              <span class="rounded px-1.5 py-0.5 text-xs" :class="walletClass(w.level)">{{ t(`siteRecon.detail.walletLevel.${w.level}`) }}</span>
            </td>
          </tr>
        </SrTable>
        <p :class="NOTE">{{ t('siteRecon.detail.walletsNote', { n: data.balance_days }) }}</p>
      </template>

      <h3 :class="H2">{{ t('siteRecon.detail.models') }}</h3>
      <SrTable
        :head="[
          t('siteRecon.detail.walletAccounts'), t('siteRecon.detail.model'), t('siteRecon.overview.revenue'), t('siteRecon.overview.cost'),
          t('siteRecon.overview.profit'), t('siteRecon.overview.margin'), t('siteRecon.detail.ourRate'), t('siteRecon.detail.ourReq'),
          t('siteRecon.detail.upReq'), t('siteRecon.detail.reqDiff')
        ]"
        :left="2"
        :empty="!data.models.length"
        :empty-text="t('siteRecon.detail.modelsEmpty')"
      >
        <tr v-for="m in data.models" :key="`${m.account_id}-${m.model}`">
          <td :class="TDL">{{ m.account_name || `#${m.account_id}` }}</td>
          <td :class="TDL" class="font-mono text-xs">{{ m.model }}</td>
          <td :class="TD">{{ money(m.revenue) }}</td>
          <td :class="TD">{{ money(m.cost) }}</td>
          <td :class="[TD, signedClass(modelProfit(m))]">{{ money(modelProfit(m)) }}</td>
          <td :class="TD">{{ pct(modelProfit(m), m.revenue) }}</td>
          <td :class="TD">{{ ratio(m.revenue, m.our_list_cost) }}</td>
          <td :class="TD">{{ num(m.our_requests) }}</td>
          <td :class="TD">{{ num(m.up_requests) }}</td>
          <td :class="[TD, diffClass(diff(m.our_requests, m.up_requests))]">{{ signedNum(diff(m.our_requests, m.up_requests)) }}</td>
        </tr>
      </SrTable>

      <h3 :class="H2">{{ t('siteRecon.detail.pending') }}</h3>
      <SrTable
        :head="[t('siteRecon.detail.walletAccounts'), t('siteRecon.detail.upstream'), t('siteRecon.detail.dayRevenue'), t('siteRecon.detail.rev7'), t('siteRecon.detail.reason')]"
        :left="2"
        :empty="!data.pending.length"
        :empty-text="t('siteRecon.detail.pendingEmpty')"
      >
        <tr v-for="p in data.pending" :key="p.account_id">
          <td :class="TDL">{{ p.account_name }} <span :class="TAG">#{{ p.account_id }}</span></td>
          <td :class="TDL">{{ p.upstream }}</td>
          <td :class="TD">{{ money(p.revenue) }}</td>
          <td :class="TD">{{ money(p.rev7) }}</td>
          <td :class="TDL" class="max-w-md whitespace-normal break-words text-xs">{{ p.error }}</td>
        </tr>
      </SrTable>
      <p :class="NOTE">{{ t('siteRecon.detail.pendingNote') }}</p>

      <h3 :class="H2">{{ t('siteRecon.detail.history') }}</h3>
      <SrTable
        :head="[
          t('siteRecon.overview.dayCol'), t('siteRecon.overview.revenue'), t('siteRecon.overview.cost'), t('siteRecon.overview.profit'),
          t('siteRecon.overview.margin'), t('siteRecon.detail.ourReq'), t('siteRecon.detail.upReq'), t('siteRecon.detail.reqDiff'),
          t('siteRecon.detail.revMissing'), t('siteRecon.detail.doneBusy')
        ]"
        :empty="!data.history.length"
      >
        <tr v-for="h in data.history" :key="h.day">
          <td :class="TDL">
            <button type="button" class="text-primary-600 hover:underline dark:text-primary-400" @click="emit('select-day', h.day)">{{ h.day }}</button>
            <span v-if="!h.scanned" :class="TAG" class="ml-1">{{ t('siteRecon.detail.notScanned') }}</span>
          </td>
          <td :class="TD">{{ money(h.rev) }}</td>
          <td :class="TD">{{ money(h.cost) }}</td>
          <td :class="[TD, signedClass(historyProfit(h))]">{{ money(historyProfit(h)) }}</td>
          <td :class="TD">{{ pct(historyProfit(h), h.rev) }}</td>
          <td :class="TD">{{ num(h.our_requests) }}</td>
          <td :class="TD">{{ num(h.up_requests) }}</td>
          <td :class="[TD, diffClass(diff(h.our_requests, h.up_requests))]">{{ signedNum(diff(h.our_requests, h.up_requests)) }}</td>
          <td :class="TD">{{ money(h.rev_missing || null) }}</td>
          <td :class="TD">{{ h.done }}/{{ h.busy }}</td>
        </tr>
      </SrTable>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { siteReconAPI, type SiteReconDetail, type SiteReconHistoryRow, type SiteReconModelRow, type SiteReconWallet } from '@/api/admin/siteRecon'
import { extractApiErrorMessage } from '@/utils/apiError'
import SrTable from './SrTable.vue'
import { H2, NOTE, TAG, TD, TDL, diff, money, num, pct, ratio, signedClass, signedNum } from './format'

const props = defineProps<{ day: string }>()
const emit = defineEmits<{
  (e: 'loaded', payload: { day: string; days: string[] }): void
  (e: 'select-day', day: string): void
}>()
const { t } = useI18n()

const data = ref<SiteReconDetail | null>(null)
const loading = ref(false)
const error = ref('')

const busyAccounts = computed(() => (data.value?.accounts ?? []).filter((a) => !a.idle))
const idleCount = computed(() => (data.value?.accounts ?? []).filter((a) => a.idle).length)

function diffClass(v: number | null) {
  return v ? 'text-amber-600 dark:text-amber-400' : ''
}

function modelProfit(m: SiteReconModelRow) {
  return m.cost === null ? null : (m.revenue || 0) - m.cost
}

function historyProfit(h: SiteReconHistoryRow) {
  return h.rev === null || h.cost === null ? null : h.rev - h.cost
}

function walletClass(level: SiteReconWallet['level']) {
  switch (level) {
    case 'empty':
      return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
    case 'low':
      return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
    case 'ok':
      return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
    default:
      return 'bg-gray-100 text-gray-600 dark:bg-dark-700 dark:text-gray-300'
  }
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await siteReconAPI.detail(props.day || undefined)
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
