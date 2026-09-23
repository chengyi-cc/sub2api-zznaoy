import { nextTick, type Ref } from 'vue'

export interface OpenAIAccountTemplate { version: 1; fields: Record<string, unknown> }
export interface TemplateBinding { read: () => unknown; write: (value: unknown) => void | Promise<void> }
export type TemplateBindings = Record<string, TemplateBinding>
export interface TemplateField {
  key: string
  kind: 'number' | 'boolean' | 'select' | 'proxy' | 'groups' | 'tls' | 'cli' | 'mappings' | 'models' | 'pool' | 'capabilities'
  scope?: 'oauth' | 'apikey'
  options?: string[]
  min?: number
  max?: number
  nullable?: boolean
  default: unknown
}
export const OPENAI_TEMPLATE_FIELDS: TemplateField[] = [
  { key: 'concurrency', kind: 'number', min: 1, max: 100000, default: 10 },
  { key: 'load_factor', kind: 'number', min: 1, max: 10000, nullable: true, default: null },
  { key: 'priority', kind: 'number', min: 0, max: 100000, default: 1 },
  { key: 'rate_multiplier', kind: 'number', min: 0, max: 10000, default: 1 },
  { key: 'proxy_id', kind: 'proxy', default: null },
  { key: 'group_ids', kind: 'groups', default: [] },
  { key: 'autoPauseOnExpired', kind: 'boolean', default: true },
  { key: 'codexFingerprintMode', kind: 'select', scope: 'oauth', options: ['off', 'device', 'machine', 'session', 'full'], default: 'off' },
  { key: 'tls', kind: 'tls', scope: 'oauth', default: { enabled: false, profile_id: null } },
  { key: 'openaiPassthroughEnabled', kind: 'boolean', default: false },
  { key: 'openaiResponsesWebSocketV2Mode', kind: 'select', options: ['off', 'ctx_pool', 'passthrough', 'http_bridge'], default: 'off' },
  { key: 'codexCLI', kind: 'cli', scope: 'oauth', default: { enabled: false, allow_app_server: false } },
  { key: 'openaiFlattenNamespacesEnabled', kind: 'boolean', scope: 'oauth', default: false },
  { key: 'openAILongContextBillingEnabled', kind: 'boolean', default: false },
  { key: 'openAICompactMode', kind: 'select', options: ['auto', 'force_on', 'force_off'], default: 'auto' },
  { key: 'openAICompactModelMappings', kind: 'mappings', default: [] },
  { key: 'modelConfig', kind: 'models', default: { mode: 'whitelist', allowed_models: [], mappings: [] } },
  { key: 'openAIResponsesMode', kind: 'select', scope: 'apikey', options: ['auto', 'force_responses', 'force_chat_completions'], default: 'auto' },
  { key: 'openAIImagesUrlToB64JsonEnabled', kind: 'boolean', scope: 'apikey', default: false },
  { key: 'openAIEndpointCapabilities', kind: 'capabilities', scope: 'apikey', options: ['chat_completions', 'embeddings', 'seedance'], default: ['chat_completions', 'embeddings'] },
  { key: 'poolConfig', kind: 'pool', scope: 'apikey', default: { enabled: false, retry_count: 3, status_codes: '' } },
  { key: 'editQuotaLimit', kind: 'number', scope: 'apikey', min: 0, max: 1e12, nullable: true, default: null },
  { key: 'editQuotaDailyLimit', kind: 'number', scope: 'apikey', min: 0, max: 1e12, nullable: true, default: null },
  { key: 'editQuotaWeeklyLimit', kind: 'number', scope: 'apikey', min: 0, max: 1e12, nullable: true, default: null }
]
export function cloneTemplateValue<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T }
export function bindTemplateValue<T>(value: Ref<T>): TemplateBinding {
  return { read: () => cloneTemplateValue(value.value), write: next => { value.value = cloneTemplateValue(next) as T } }
}
export function templateProperty<T extends object, K extends keyof T>(form: T, key: K): TemplateBinding {
  return { read: () => cloneTemplateValue(form[key]), write: next => { form[key] = cloneTemplateValue(next) as T[K] } }
}
export function templateGroup(bindings: TemplateBindings): TemplateBinding {
  return {
    read: () => Object.fromEntries(Object.entries(bindings).map(([key, binding]) => [key, binding.read()])),
    async write(value) {
      const object = value as Record<string, unknown>
      // Let mode watchers finish before writing dependent lists (e.g. whitelist).
      for (const [key, binding] of Object.entries(bindings)) {
        if (Object.prototype.hasOwnProperty.call(object, key)) { await binding.write(object[key]); await nextTick() }
      }
    }
  }
}
export async function applyOpenAIAccountTemplate(template: OpenAIAccountTemplate, bindings: TemplateBindings, scope: 'oauth' | 'apikey'): Promise<number> {
  let count = 0
  for (const field of OPENAI_TEMPLATE_FIELDS) {
    if (field.scope && field.scope !== scope) continue
    if (!Object.prototype.hasOwnProperty.call(template.fields, field.key) || !bindings[field.key]) continue
    await bindings[field.key].write(cloneTemplateValue(template.fields[field.key]))
    await nextTick()
    count++
  }
  return count
}
