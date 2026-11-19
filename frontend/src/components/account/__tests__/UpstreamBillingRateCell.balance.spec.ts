import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import UpstreamBillingRateCell from '../UpstreamBillingRateCell.vue'
import type { Account, UpstreamBillingData } from '@/types'

vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, params?: Record<string, unknown>) =>
        params ? `${key}:${Object.values(params).join(',')}` : key
    })
  }
})

const makeAccount = (data: Partial<UpstreamBillingData>, receivedAt = '2026-11-19T00:00:00Z'): Account => ({
  id: 1,
  name: 'relay',
  platform: 'openai',
  type: 'apikey',
  proxy_id: null,
  concurrency: 1,
  priority: 1,
  status: 'active',
  error_message: null,
  last_used_at: null,
  expires_at: null,
  auto_pause_on_expired: false,
  created_at: '2026-11-19T00:00:00Z',
  updated_at: '2026-11-19T00:00:00Z',
  schedulable: true,
  rate_limited_at: null,
  rate_limit_reset_at: null,
  overload_until: null,
  temp_unschedulable_until: null,
  temp_unschedulable_reason: null,
  session_window_start: null,
  session_window_end: null,
  session_window_status: null,
  extra: {
    upstream_billing_probe_enabled: true,
    upstream_billing_probe: {
      status: 'ok',
      data: data as UpstreamBillingData,
      received_at: receivedAt,
      fresh_until: '2026-11-19T01:00:00Z',
      last_attempt_at: receivedAt,
      next_probe_at: '2026-11-19T00:30:00Z'
    }
  }
})

const newapiRate = {
  object: 'newapi.token_billing' as const,
  schema_version: 1 as const,
  billing_scope: 'token' as const,
  group_rate_multiplier: 0.085,
  resolved_rate_multiplier: 0.085,
  peak_rate_enabled: false,
  effective_rate_multiplier: 0.085,
  observed_at: '2026-11-19T00:00:00Z',
  upstream_kind: 'newapi' as const,
  upstream_group: 'vip',
  rate_source: 'recent_log' as const
}

const mountCell = (account: Account) =>
  mount(UpstreamBillingRateCell, {
    props: { account, now: Date.parse('2026-11-19T00:10:00Z') },
    attachTo: document.body
  })

const openTooltip = async (wrapper: ReturnType<typeof mountCell>) => {
  await wrapper.get('[data-testid="upstream-billing-details"]').trigger('mouseenter')
  await flushPromises()
  const tooltips = document.body.querySelectorAll('[role="tooltip"]')
  return tooltips[tooltips.length - 1] as HTMLElement
}

describe('UpstreamBillingRateCell balance', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-19T00:10:00Z'))
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('shows rate and upstream balance side by side', async () => {
    const wrapper = mountCell(makeAccount({ ...newapiRate, upstream_balance: 1045.14, upstream_balance_scope: 'token' }))
    expect(wrapper.get('[data-testid="upstream-billing-rate"]').text().replace(/\s+/g, ' ')).toBe('0.085x / 1,045.14')
    const tooltip = await openTooltip(wrapper)
    const kind = tooltip.querySelector('[data-testid="upstream-billing-kind"]')?.textContent ?? ''
    expect(kind).toContain('kindNewAPIGroup')
    expect(kind).toContain('vip')
    expect(tooltip.querySelector('[data-testid="upstream-billing-balance-detail"]')?.textContent).toContain('$1,045.14')
    wrapper.unmount()
  })

  it('shows the balance with a dash when the NewAPI rate is unknown', async () => {
    const wrapper = mountCell(makeAccount({
      object: 'newapi.token_billing',
      schema_version: 1,
      observed_at: '2026-11-19T00:00:00Z',
      upstream_kind: 'newapi',
      upstream_balance_unlimited: true
    }))
    expect(wrapper.get('[data-testid="upstream-billing-rate"]').text().replace(/\s+/g, ' ')).toBe('- / ∞')
    const tooltip = await openTooltip(wrapper)
    expect(tooltip.querySelector('[data-testid="upstream-billing-rate-unknown"]')).not.toBeNull()
    wrapper.unmount()
  })

  it('keeps the plain rate when no balance was reported', () => {
    const { upstream_kind: _kind, upstream_group: _group, rate_source: _source, ...plain } = newapiRate
    const wrapper = mountCell(makeAccount({ ...plain, object: 'sub2api.key_billing' }))
    expect(wrapper.get('[data-testid="upstream-billing-rate"]').text()).toBe('0.085x')
    expect(wrapper.find('[data-testid="upstream-billing-balance"]').exists()).toBe(false)
  })

  it('hides the balance once the snapshot is stale', () => {
    const wrapper = mount(UpstreamBillingRateCell, {
      props: {
        account: makeAccount({ ...newapiRate, upstream_balance: 5 }),
        now: Date.parse('2026-11-19T03:00:00Z')
      }
    })
    expect(wrapper.find('[data-testid="upstream-billing-balance"]').exists()).toBe(false)
  })
})
