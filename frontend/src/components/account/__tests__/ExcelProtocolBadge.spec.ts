import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import ExcelProtocolBadge from '../ExcelProtocolBadge.vue'
vi.mock('vue-i18n', () => ({ useI18n: () => ({ t: (key: string) => key }) }))
describe('Excel protocol status', () => {
  it('shows enabled state regardless of manual or automatic source', () => {
    const wrapper = mount(ExcelProtocolBadge, { props: { extra: { openai_excel_bps: true, openai_excel_bps_last_transition: { reason: 'candy_incorrect', at: '2026-09-28T00:00:00Z' } } } })
    expect(wrapper.text()).toBe('Excel')
    expect(wrapper.attributes('title')).toContain('excelEnabledByMonitor')
  })
  it('keeps a visible explanation after automatic 403 fallback', async () => {
    const wrapper = mount(ExcelProtocolBadge, { props: { extra: { openai_excel_bps: false, openai_excel_bps_last_transition: { reason: 'upstream_403', at: '2026-09-28T00:00:00Z' } } } })
    expect(wrapper.text()).toContain('excelReturnedNative')
    expect(wrapper.attributes('title')).toContain('excel403History')
    await wrapper.setProps({ extra: {} })
    expect(wrapper.find('[data-testid="excel-protocol-badge"]').exists()).toBe(false)
  })
})
