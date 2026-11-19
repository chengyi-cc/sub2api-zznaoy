import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SiteReconOverview from '../SiteReconOverview.vue'

const { overview } = vi.hoisted(() => ({ overview: vi.fn() }))

vi.mock('@/api/admin/siteRecon', () => ({ siteReconAPI: { overview } }))
vi.mock('vue-chartjs', () => ({ Line: { template: '<canvas />' } }))
vi.mock('vue-i18n', async (importOriginal) => {
  const actual = await importOriginal<typeof import('vue-i18n')>()
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, params?: Record<string, unknown>) => (params ? `${key}:${Object.values(params).join(',')}` : key)
    })
  }
})

const totals = (rev: number, cost: number) => ({ rev, cost, missing: 0, all: rev, has: true, scanned: 1 })

describe('SiteReconOverview', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders totals, alerts and expandable group profit', async () => {
    overview.mockResolvedValue({
      day: '2026-11-09',
      days: ['2026-11-09', '2026-11-08'],
      totals: totals(100, 40),
      prev: totals(80, 40),
      alerts: [{ level: 'bad', text: '上游 a.com 余额已用完' }],
      trend: [
        { day: '2026-11-08', rev: 80, cost: 40, missing: 0 },
        { day: '2026-11-09', rev: 100, cost: 40, missing: 0 }
      ],
      groups: [
        {
          name: 'VIP',
          rev: 100,
          cost: 40,
          missing: 0,
          known: 100,
          profit: 60,
          margin: 0.6,
          req: 10,
          rates: [0.8],
          accounts: [{ id: 7, name: 'relay-A', upstream: 'a.com', rev: 100, cost: 40, profit: 60, req: 10, rates: [0.8] }]
        }
      ],
      anomaly_counts: { high: 0, mid: 0, low: 0, high_list: 0 }
    })
    const wrapper = mount(SiteReconOverview, { props: { day: '' } })
    await flushPromises()

    expect(overview).toHaveBeenCalledWith(undefined)
    expect(wrapper.emitted('loaded')?.[0]).toEqual([{ day: '2026-11-09', days: ['2026-11-09', '2026-11-08'] }])
    expect(wrapper.get('[data-test="site-recon-profit"]').text()).toBe('¥60.00')
    expect(wrapper.get('[data-test="site-recon-alerts"]').text()).toContain('余额已用完')
    expect(wrapper.text()).toContain('VIP')
    expect(wrapper.text()).not.toContain('relay-A')

    await wrapper.get('tbody tr').trigger('click')
    expect(wrapper.text()).toContain('relay-A')
  })

  it('shows the empty hint when nothing has been reconciled', async () => {
    overview.mockResolvedValue({
      day: '',
      days: [],
      totals: totals(0, 0),
      prev: totals(0, 0),
      alerts: [],
      trend: [],
      groups: [],
      anomaly_counts: { high: 0, mid: 0, low: 0, high_list: 0 }
    })
    const wrapper = mount(SiteReconOverview, { props: { day: '' } })
    await flushPromises()
    expect(wrapper.find('[data-test="site-recon-empty"]').exists()).toBe(true)
  })
})
