<template>
  <input v-if="field.kind === 'number'" class="input w-full" type="number" :value="modelValue ?? ''" :min="field.min" :max="field.max" :step="['concurrency', 'priority', 'load_factor'].includes(field.key) ? 1 : 'any'" :required="!field.nullable" :placeholder="tr('emptyDefault')" @input="setNumber" />
  <label v-else-if="field.kind === 'boolean'" class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="modelValue === true" @change="emit('update:modelValue', checked($event))" />{{ tr('enabled') }}</label>
  <select v-else-if="field.kind === 'select'" class="input w-full" :value="modelValue" @change="emit('update:modelValue', value($event))"><option v-for="option in field.options" :key="option" :value="option">{{ tr(`options.${option}`) }}</option></select>
  <select v-else-if="field.kind === 'proxy'" class="input w-full" :value="modelValue ?? ''" @change="emit('update:modelValue', value($event) ? Number(value($event)) : null)"><option value="">{{ tr('noProxy') }}</option><option v-for="proxy in proxies" :key="proxy.id" :value="proxy.id">{{ proxy.name }}</option><option v-if="modelValue && !proxies.some(p => p.id === modelValue)" :value="modelValue">#{{ modelValue }} · {{ tr('unavailable') }}</option></select>
  <div v-else-if="field.kind === 'groups'" class="max-h-44 space-y-2 overflow-auto">
    <label v-for="group in groups" :key="group.id" class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="list.includes(group.id)" @change="toggle(group.id, checked($event))" />{{ group.name }}</label>
    <p class="text-xs text-gray-500">{{ tr('emptyGroups') }}</p>
  </div>
  <div v-else-if="field.kind === 'capabilities'" class="space-y-2"><label v-for="option in field.options" :key="option" class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="list.includes(option)" @change="toggle(option, checked($event))" />{{ tr(`options.${option}`) }}</label></div>
  <div v-else-if="field.kind === 'tls'" class="space-y-2">
    <label class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="object.enabled === true" @change="part('enabled', checked($event))" />{{ tr('enabled') }}</label>
    <select class="input w-full" :disabled="!object.enabled" :value="object.profile_id ?? ''" @change="part('profile_id', value($event) ? Number(value($event)) : null)"><option value="">{{ tr('defaultProfile') }}</option><option v-for="profile in profiles" :key="profile.id" :value="profile.id">{{ profile.name }}</option><option v-if="object.profile_id && !profiles.some(p => p.id === object.profile_id)" :value="object.profile_id">#{{ object.profile_id }} · {{ tr('unavailable') }}</option></select>
  </div>
  <div v-else-if="field.kind === 'cli'" class="space-y-2">
    <label class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="object.enabled === true" @change="part('enabled', checked($event))" />{{ tr('onlyCLI') }}</label>
    <label class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="object.allow_app_server === true" @change="part('allow_app_server', checked($event))" />{{ tr('allowAppServer') }}</label>
  </div>
  <div v-else-if="field.kind === 'pool'" class="space-y-2">
    <label class="flex items-center gap-2 text-sm"><input type="checkbox" :checked="object.enabled === true" @change="part('enabled', checked($event))" />{{ tr('enabled') }}</label>
    <label class="block text-xs">{{ tr('retryCount') }}<input class="input mt-1 w-full" type="number" min="1" max="10" required :value="object.retry_count" @input="part('retry_count', Number(value($event)))" /></label>
    <label class="block text-xs">{{ tr('statusCodes') }}<input class="input mt-1 w-full" :value="object.status_codes" maxlength="200" placeholder="401, 403, 429" @input="part('status_codes', value($event))" /></label>
  </div>
  <div v-else-if="field.kind === 'models' || field.kind === 'mappings'" class="space-y-3">
    <template v-if="field.kind === 'models'">
      <select class="input w-full" :value="object.mode" @change="part('mode', value($event))"><option value="whitelist">{{ tr('options.whitelist') }}</option><option value="mapping">{{ tr('options.mapping') }}</option></select>
      <label v-if="object.mode === 'whitelist'" class="block text-xs">{{ tr('modelList') }}<textarea class="input mt-1 w-full" rows="4" :value="(object.allowed_models as string[] || []).join('\n')" @input="part('allowed_models', value($event).split(/\r?\n/).map(s => s.trim()).filter(Boolean))" /></label>
    </template>
    <template v-if="field.kind === 'mappings' || object.mode === 'mapping'">
      <div v-for="(mapping, index) in mappings" :key="index" class="flex items-center gap-2">
        <input class="input min-w-0 flex-1" :aria-label="tr('sourceModel')" :placeholder="tr('sourceModel')" :value="mapping.from" required maxlength="200" @input="updateMapping(index, 'from', value($event))" /><span>→</span>
        <input class="input min-w-0 flex-1" :aria-label="tr('targetModel')" :placeholder="tr('targetModel')" :value="mapping.to" required maxlength="200" @input="updateMapping(index, 'to', value($event))" />
        <button class="btn btn-ghost btn-sm" type="button" :aria-label="tr('remove')" @click="setMappings(mappings.filter((_, i) => i !== index))">×</button>
      </div>
      <button class="btn btn-secondary btn-sm" type="button" @click="setMappings([...mappings, { from: '', to: '' }])">{{ tr('addMapping') }}</button>
    </template>
  </div>
</template>
<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { TemplateField } from '@/utils/openaiAccountTemplate'
const props = defineProps<{ field: TemplateField; modelValue: unknown; proxies: { id: number; name: string }[]; groups: { id: number; name: string }[]; profiles: { id: number; name: string }[] }>()
const emit = defineEmits<{ 'update:modelValue': [value: unknown] }>()
const { t } = useI18n()
const tr = (key: string) => t(`admin.accounts.accountTemplate.${key}`)
const object = computed(() => (props.modelValue || {}) as Record<string, unknown>)
const list = computed(() => (Array.isArray(props.modelValue) ? props.modelValue : []) as (string | number)[])
const mappings = computed(() => (props.field.kind === 'mappings' ? props.modelValue : object.value.mappings) as { from: string; to: string }[] || [])
const value = (event: Event) => (event.target as HTMLInputElement).value
const checked = (event: Event) => (event.target as HTMLInputElement).checked
const part = (key: string, next: unknown) => emit('update:modelValue', { ...object.value, [key]: next })
function setNumber(event: Event) { const raw = value(event); emit('update:modelValue', raw === '' ? null : Number(raw)) }
function toggle(item: string | number, on: boolean) { emit('update:modelValue', on ? [...list.value, item] : list.value.filter(v => v !== item)) }
function setMappings(next: { from: string; to: string }[]) { if (props.field.kind === 'mappings') emit('update:modelValue', next); else part('mappings', next) }
function updateMapping(index: number, key: 'from' | 'to', next: string) { setMappings(mappings.value.map((entry, i) => i === index ? { ...entry, [key]: next } : entry)) }
</script>
