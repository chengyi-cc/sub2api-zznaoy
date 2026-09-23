import { describe, expect, it } from 'vitest'
import { reactive, ref, watch } from 'vue'
import { applyOpenAIAccountTemplate, bindTemplateValue, templateGroup, templateProperty } from '../openaiAccountTemplate'

describe('OpenAI account template application', () => {
  it('overwrites only selected fields, including false, null and empty arrays', async () => {
    const form = reactive({ concurrency: 20, priority: 7, proxy_id: 12 as number | null, group_ids: [1, 2], credentials: { api_key: 'keep-secret' } })
    const passthrough = ref(true)
    const count = await applyOpenAIAccountTemplate({ version: 1, fields: { concurrency: 4, proxy_id: null, group_ids: [], openaiPassthroughEnabled: false, credentials: { api_key: 'replace' } } }, {
      concurrency: templateProperty(form, 'concurrency'), priority: templateProperty(form, 'priority'), proxy_id: templateProperty(form, 'proxy_id'), group_ids: templateProperty(form, 'group_ids'),
      openaiPassthroughEnabled: bindTemplateValue(passthrough), credentials: templateProperty(form, 'credentials')
    }, 'oauth')
    expect(count).toBe(4)
    expect(form).toEqual({ concurrency: 4, priority: 7, proxy_id: null, group_ids: [], credentials: { api_key: 'keep-secret' } })
    expect(passthrough.value).toBe(false)
  })
  it('does not apply OAuth-only fields to an API key account', async () => {
    const fingerprint = ref('machine')
    const responses = ref('auto')
    const count = await applyOpenAIAccountTemplate({ version: 1, fields: { codexFingerprintMode: 'off', openAIResponsesMode: 'force_responses' } }, {
      codexFingerprintMode: bindTemplateValue(fingerprint), openAIResponsesMode: bindTemplateValue(responses)
    }, 'apikey')
    expect(count).toBe(1); expect(fingerprint.value).toBe('machine'); expect(responses.value).toBe('force_responses')
  })
  it('restores explicit whitelist after mode watchers and does not share mutable arrays', async () => {
    const mode = ref('mapping'); const models = ref(['old']); const mappings = ref([])
    const stop = watch(mode, () => { models.value = ['autofilled'] })
    const original = { mode: 'whitelist', allowed_models: ['chosen-text'], mappings: [] }
    await applyOpenAIAccountTemplate({ version: 1, fields: { modelConfig: original } }, {
      modelConfig: templateGroup({ mode: bindTemplateValue(mode), allowed_models: bindTemplateValue(models), mappings: bindTemplateValue(mappings) })
    }, 'oauth')
    expect(models.value).toEqual(['chosen-text'])
    models.value.push('another-model')
    expect(original.allowed_models).toEqual(['chosen-text'])
    stop()
  })
})
