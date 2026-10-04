import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AstraBorrowView from '../AstraBorrowView.vue'

const api = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn(), verify: vi.fn(), history: vi.fn(), accounts: vi.fn() }))
vi.mock('@/api/admin/astraBorrow', () => ({ getAstraBorrow: api.get, saveAstraBorrow: api.save, verifyAstraBorrow: api.verify, getAstraBorrowHistory: api.history }))
vi.mock('@/api/admin/accounts', () => ({ default: {}, list: api.accounts }))
vi.mock('vue-i18n', async () => ({ ...await vi.importActual<typeof import('vue-i18n')>('vue-i18n'), useI18n: () => ({ t: (key: string) => key, te: () => true }) }))
const defaults = { enabled: false, source_account_ids: [] as number[], target_account_ids: [] as number[], follow_source_proxy: true, ttl_seconds: 230, revision: '' }
const setup = () => mount(AstraBorrowView, { global: { stubs: { AppLayout: { template: '<div><slot /></div>' } } } })
beforeEach(() => {
  vi.useFakeTimers(); vi.clearAllMocks()
  api.get.mockResolvedValue({ settings: { ...defaults }, statuses: [], preparing: false, history_error: false })
  api.accounts.mockResolvedValue({ items: [{ id: 1, name: 'Source' }, { id: 2, name: 'Target' }], pages: 1 })
  api.history.mockResolvedValue([])
  api.save.mockImplementation(async value => ({ ...value, revision: 'new' }))
  api.verify.mockResolvedValue({ passed: true })
})
afterEach(() => vi.useRealTimers())

describe('Astra borrowing', () => {
  it('reading and polling never save or probe and polling stops on unmount', async () => {
    const w = setup(); await flushPromises(); await vi.advanceTimersByTimeAsync(15000)
    expect(api.get).toHaveBeenCalledTimes(4); expect(api.save).not.toHaveBeenCalled(); expect(api.verify).not.toHaveBeenCalled()
    w.unmount(); await vi.advanceTimersByTimeAsync(10000); expect(api.get).toHaveBeenCalledTimes(4)
  })
  it('requires both roles and prevents selecting the same account for both', async () => {
    const w = setup(); await flushPromises()
    await w.get('[data-testid="enabled"]').setValue(true)
    expect(w.get('[data-testid="save"]').attributes('disabled')).toBeDefined()
    await w.get('[data-testid="sources-1"]').setValue(true)
    expect(w.get('[data-testid="targets-1"]').attributes('disabled')).toBeDefined()
    await w.get('[data-testid="targets-2"]').setValue(true)
    expect(w.get('[data-testid="save"]').attributes('disabled')).toBeUndefined()
    await w.get('form').trigger('submit'); await flushPromises()
    expect(api.save).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, source_account_ids: [1], target_account_ids: [2], follow_source_proxy: true, ttl_seconds: 230 }))
    w.unmount()
  })
  it('keeps unsaved selections when the runtime refreshes', async () => {
    const w = setup(); await flushPromises()
    await w.get('[data-testid="sources-1"]').setValue(true)
    api.get.mockResolvedValue({ settings: { ...defaults, revision: 'other-admin' }, statuses: [], preparing: true })
    await vi.advanceTimersByTimeAsync(5000); await flushPromises()
    expect((w.get('[data-testid="sources-1"]').element as HTMLInputElement).checked).toBe(true)
    expect(api.save).not.toHaveBeenCalled(); w.unmount()
  })
  it('shows failed verification instead of a success notice', async () => {
    api.get.mockResolvedValue({ settings: { ...defaults, enabled: true, source_account_ids: [1], target_account_ids: [2] }, statuses: [{ account_id: 2, source_account_id: 1, state: 'failed', reason: 'astra_ticket_changed' }], preparing: false })
    api.verify.mockRejectedValue({ response: { data: { message: 'astra_ticket_changed' } } })
    const w = setup(); await flushPromises()
    const button = w.findAll('button').find(b => b.text() === 'admin.astraBorrow.verify')!
    await button.trigger('click'); await flushPromises()
    expect(api.verify).toHaveBeenCalledWith(2)
    expect(w.get('[role="alert"]').text()).toContain('astra_ticket_changed')
    expect(w.find('[role="status"]').exists()).toBe(false); w.unmount()
  })
  it('allows disabling even when the account list cannot load', async () => {
    api.accounts.mockRejectedValue(new Error('offline'))
    api.get.mockResolvedValue({ settings: { ...defaults, enabled: true, source_account_ids: [1], target_account_ids: [2] }, statuses: [], preparing: false })
    const w = setup(); await flushPromises()
    await w.get('[data-testid="enabled"]').setValue(false)
    expect(w.get('[data-testid="save"]').attributes('disabled')).toBeUndefined()
    await w.get('form').trigger('submit'); await flushPromises()
    expect(api.save).toHaveBeenCalledWith(expect.objectContaining({ enabled: false })); w.unmount()
  })
})
