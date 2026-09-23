import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import OpenAIAccountTemplateBar from '../OpenAIAccountTemplateBar.vue'
import { bindTemplateValue } from '@/utils/openaiAccountTemplate'

const api = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn() }))
vi.mock('@/api/admin/openaiAccountTemplate', () => ({ openaiAccountTemplateAPI: api }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) }
})
function setup() {
  const concurrency = ref(10); const priority = ref(7); const fingerprint = ref('machine')
  const wrapper = mount(OpenAIAccountTemplateBar, {
    props: { kind: 'oauth', proxies: [], groups: [], profiles: [], bindings: { concurrency: bindTemplateValue(concurrency), priority: bindTemplateValue(priority), codexFingerprintMode: bindTemplateValue(fingerprint) } },
    global: { stubs: { BaseDialog: { props: ['show'], template: '<div v-if="show"><slot /><slot name="footer" /></div>' } } }
  })
  return { wrapper, concurrency, priority, fingerprint }
}
describe('OpenAIAccountTemplateBar', () => {
  beforeEach(() => { vi.clearAllMocks(); api.get.mockResolvedValue({ version: 1, fields: { concurrency: 3, codexFingerprintMode: 'off', openAIResponsesMode: 'force_responses' } }); api.save.mockImplementation(async v => v) })
  it('applies saved fields and leaves unselected form values unchanged', async () => {
    const { wrapper, concurrency, priority, fingerprint } = setup()
    expect(api.get).not.toHaveBeenCalled()
    await wrapper.get('[data-testid="apply-template"]').trigger('click'); await flushPromises()
    expect(concurrency.value).toBe(3); expect(priority.value).toBe(7); expect(fingerprint.value).toBe('off')
    expect(api.save).not.toHaveBeenCalled()
    wrapper.unmount()
  })
  it('edits selected values, removes unchecked fields, and preserves other account type fields', async () => {
    const { wrapper, concurrency } = setup()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-template-field="codexFingerprintMode"] input[type="checkbox"]').setValue(false)
    await wrapper.get('[data-template-field="concurrency"] input[type="number"]').setValue(6)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(api.save).toHaveBeenCalledWith({ version: 1, fields: { concurrency: 6, openAIResponsesMode: 'force_responses' } })
    expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
  it('cancelling editor makes no changes to the template or account', async () => {
    const { wrapper, concurrency } = setup()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-template-field="concurrency"] input[type="number"]').setValue(99)
    await wrapper.findAll('button').find(b => b.text() === 'common.cancel')!.trigger('click')
    expect(api.save).not.toHaveBeenCalled(); expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
  it('ignores a late response after account type changes', async () => {
    let resolve!: (value: unknown) => void
    api.get.mockImplementationOnce(() => new Promise(r => { resolve = r }))
    const { wrapper, concurrency } = setup()
    await wrapper.get('[data-testid="apply-template"]').trigger('click')
    await wrapper.setProps({ kind: 'apikey' })
    resolve({ version: 1, fields: { concurrency: 99 } }); await flushPromises()
    expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
  it('shows storage failures without clearing account values', async () => {
    api.get.mockRejectedValueOnce(new Error('offline'))
    const { wrapper, concurrency } = setup()
    await wrapper.get('[data-testid="apply-template"]').trigger('click'); await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toBe('offline'); expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
})
