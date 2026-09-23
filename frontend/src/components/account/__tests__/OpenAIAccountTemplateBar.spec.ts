import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import OpenAIAccountTemplateBar from '../OpenAIAccountTemplateBar.vue'
import { bindTemplateValue, cloneTemplateValue, type OpenAIAccountTemplates } from '@/utils/openaiAccountTemplate'

const api = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn() }))
vi.mock('@/api/admin/openaiAccountTemplate', () => ({ openaiAccountTemplateAPI: api }))
vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return { ...actual, useI18n: () => ({ t: (key: string, params?: Record<string, unknown>) => {
    if (key.endsWith('.applyNamed')) return `Use ${params?.name}`
    if (key.endsWith('.defaultName')) return `Template ${params?.number}`
    return key
  } }) }
})
function setup() {
  const concurrency = ref(10); const priority = ref(7); const fingerprint = ref('machine')
  const wrapper = mount(OpenAIAccountTemplateBar, {
    props: { kind: 'oauth', proxies: [], groups: [], profiles: [], bindings: { concurrency: bindTemplateValue(concurrency), priority: bindTemplateValue(priority), codexFingerprintMode: bindTemplateValue(fingerprint) } },
    global: { stubs: { BaseDialog: { props: ['show'], template: '<div v-if="show"><slot /><slot name="footer" /></div>' } } }
  })
  return { wrapper, concurrency, priority, fingerprint }
}
let stored: OpenAIAccountTemplates
describe('OpenAIAccountTemplateBar', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    stored = { version: 2, templates: [
      { id: 'first', name: 'Template 1', enabled: true, fields: { concurrency: 3, codexFingerprintMode: 'off', openAIResponsesMode: 'force_responses' } },
      { id: 'second', name: 'Template 2', enabled: true, fields: { concurrency: 8 } },
      { id: 'hidden', name: 'Hidden', enabled: false, fields: { priority: 2 } }
    ] }
    api.get.mockImplementation(async () => cloneTemplateValue(stored))
    api.save.mockImplementation(async value => { stored = cloneTemplateValue(value); return cloneTemplateValue(value) })
  })
  it('applies saved fields and leaves unselected form values unchanged', async () => {
    const { wrapper, concurrency, priority, fingerprint } = setup()
    await flushPromises()
    expect(wrapper.findAll('[data-testid="apply-template"]').map(button => button.text())).toEqual(['Use Template 1', 'Use Template 2'])
    await wrapper.get('[data-testid="apply-template"]').trigger('click'); await flushPromises()
    expect(concurrency.value).toBe(3); expect(priority.value).toBe(7); expect(fingerprint.value).toBe('off')
    await wrapper.get('[data-template-id="second"]').trigger('click'); await flushPromises()
    expect(concurrency.value).toBe(8); expect(priority.value).toBe(7); expect(fingerprint.value).toBe('off')
    expect(api.save).not.toHaveBeenCalled()
    wrapper.unmount()
  })
  it('edits selected values, removes unchecked fields, and preserves other account type fields', async () => {
    const { wrapper, concurrency } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-template-field="codexFingerprintMode"] input[type="checkbox"]').setValue(false)
    await wrapper.get('[data-template-field="concurrency"] input[type="number"]').setValue(6)
    await wrapper.get('[data-select-template="second"]').trigger('click')
    await wrapper.get('[data-template-field="concurrency"] input[type="number"]').setValue(12)
    await wrapper.get('[data-select-template="first"]').trigger('click')
    expect((wrapper.get('[data-template-field="concurrency"] input[type="number"]').element as HTMLInputElement).value).toBe('6')
    expect((wrapper.get('[data-template-field="codexFingerprintMode"] input[type="checkbox"]').element as HTMLInputElement).checked).toBe(false)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(stored.templates[0].fields).toEqual({ concurrency: 6, openAIResponsesMode: 'force_responses' })
    expect(stored.templates[1].fields).toEqual({ concurrency: 12 })
    expect(stored.templates[2].fields).toEqual({ priority: 2 })
    expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
  it('cancelling editor makes no changes to the template or account', async () => {
    const { wrapper, concurrency } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-template-field="concurrency"] input[type="number"]').setValue(99)
    await wrapper.get('[data-testid="remove-template"]').trigger('click')
    await wrapper.get('[data-testid="add-template"]').trigger('click')
    await wrapper.findAll('button').find(b => b.text() === 'common.cancel')!.trigger('click')
    expect(api.save).not.toHaveBeenCalled(); expect(concurrency.value).toBe(10)
    expect(wrapper.get('[data-template-id="first"]').text()).toBe('Use Template 1')
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    expect(wrapper.findAll('[data-select-template]')).toHaveLength(3)
    expect((wrapper.get('[data-template-field="concurrency"] input[type="number"]').element as HTMLInputElement).value).toBe('3')
    wrapper.unmount()
  })
  it('ignores a late response after account type changes', async () => {
    const { wrapper, concurrency } = setup()
    await flushPromises()
    let resolve!: (value: unknown) => void
    api.get.mockImplementationOnce(() => new Promise(r => { resolve = r }))
    await wrapper.get('[data-testid="apply-template"]').trigger('click')
    await wrapper.setProps({ kind: 'apikey' })
    resolve(stored); await flushPromises()
    expect(concurrency.value).toBe(10)
    wrapper.unmount()
  })
  it('shows storage failures without clearing account values', async () => {
    api.get.mockRejectedValueOnce(new Error('offline'))
    const { wrapper, concurrency } = setup()
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toBe('offline'); expect(concurrency.value).toBe(10)
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    expect(wrapper.find('form').exists()).toBe(true)
    wrapper.unmount()
  })
  it('adds, names, enables, saves and reloads an independent template', async () => {
    const { wrapper } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-testid="add-template"]').trigger('click')
    expect((wrapper.get('[data-testid="template-enabled"]').element as HTMLInputElement).checked).toBe(false)
    expect((wrapper.get('[data-template-field="concurrency"] input[type="checkbox"]').element as HTMLInputElement).checked).toBe(false)
    await wrapper.get('[data-testid="template-name"]').setValue('Batch import')
    await wrapper.get('[data-testid="template-enabled"]').setValue(true)
    await wrapper.get('[data-template-field="priority"] input[type="checkbox"]').setValue(true)
    await wrapper.get('[data-template-field="priority"] input[type="number"]').setValue(4)
    expect(wrapper.findAll('[data-testid="apply-template"]')).toHaveLength(2)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(wrapper.findAll('[data-testid="apply-template"]').map(button => button.text())).toEqual(['Use Template 1', 'Use Template 2', 'Use Batch import'])
    const created = stored.templates[3]
    expect(created).toMatchObject({ name: 'Batch import', enabled: true, fields: { priority: 4 } })
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get(`[data-select-template="${created.id}"]`).trigger('click')
    expect((wrapper.get('[data-testid="template-name"]').element as HTMLInputElement).value).toBe('Batch import')
    expect((wrapper.get('[data-template-field="priority"] input[type="number"]').element as HTMLInputElement).value).toBe('4')
    wrapper.unmount()
  })
  it('hides disabled templates only after save and preserves their settings', async () => {
    const { wrapper } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-testid="template-enabled"]').setValue(false)
    expect(wrapper.find('[data-template-id="first"]').exists()).toBe(true)
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(wrapper.find('[data-template-id="first"]').exists()).toBe(false)
    expect(stored.templates[0].fields.concurrency).toBe(3)
    wrapper.unmount()
  })
  it('saves an empty collection when all templates are removed', async () => {
    const { wrapper } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    for (let index = 0; index < 3; index++) await wrapper.get('[data-testid="remove-template"]').trigger('click')
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(stored.templates).toEqual([])
    expect(wrapper.findAll('[data-testid="apply-template"]')).toHaveLength(0)
    wrapper.unmount()
  })
  it('blocks duplicate or empty names', async () => {
    const { wrapper } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    for (const name of ['Template 2', '  ']) {
      await wrapper.get('[data-testid="template-name"]').setValue(name)
      await wrapper.get('form').trigger('submit'); await flushPromises()
      expect(wrapper.get('[role="alert"]').text()).toContain('invalidNames')
    }
    expect(api.save).not.toHaveBeenCalled()
    wrapper.unmount()
  })
  it.each(['disabled', 'deleted'])('does not apply a template %s by another window', async mode => {
    const { wrapper, concurrency } = setup()
    await flushPromises()
    if (mode === 'disabled') stored.templates[0].enabled = false
    else stored.templates.shift()
    await wrapper.get('[data-template-id="first"]').trigger('click'); await flushPromises()
    expect(concurrency.value).toBe(10)
    expect(wrapper.find('[data-template-id="first"]').exists()).toBe(false)
    expect(wrapper.get('[role="status"]').text()).toContain('noLongerEnabled')
    wrapper.unmount()
  })
  it('keeps drafts and existing buttons if saving fails', async () => {
    api.save.mockRejectedValueOnce(new Error('offline'))
    const { wrapper } = setup()
    await flushPromises()
    await wrapper.get('[data-testid="edit-template"]').trigger('click'); await flushPromises()
    await wrapper.get('[data-testid="template-name"]').setValue('Renamed')
    await wrapper.get('form').trigger('submit'); await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toBe('offline')
    expect(wrapper.get('[data-template-id="first"]').text()).toBe('Use Template 1')
    expect((wrapper.get('[data-testid="template-name"]').element as HTMLInputElement).value).toBe('Renamed')
    wrapper.unmount()
  })
})
