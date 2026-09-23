<template>
  <section class="rounded-xl border-2 border-primary-300 bg-primary-50 p-4 dark:border-primary-700 dark:bg-primary-950/40" data-testid="openai-template-bar">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div><h3 class="font-semibold text-primary-900 dark:text-primary-100">{{ tr('title') }}</h3><p class="mt-1 text-xs leading-5 text-primary-700 dark:text-primary-300">{{ tr('hint') }}</p></div>
      <div class="flex flex-wrap gap-2">
        <button type="button" class="btn btn-secondary border-primary-300" :disabled="busy" data-testid="edit-template" @click="openEditor">{{ tr('edit') }}</button>
        <button type="button" class="btn btn-primary shadow-sm" :disabled="busy" data-testid="apply-template" @click="apply">{{ tr('apply') }}</button>
      </div>
    </div>
    <p v-if="notice" role="status" class="mt-2 text-sm text-primary-700 dark:text-primary-200">{{ notice }}</p>
    <p v-if="error" role="alert" class="mt-2 text-sm text-red-600">{{ error }}</p>
  </section>
  <BaseDialog :show="editing" :title="tr('editTitle')" width="wide" :z-index="70" @close="closeEditor">
    <form :id="formID" class="space-y-4" @submit.stop.prevent="save">
      <p class="rounded-lg bg-primary-50 p-3 text-sm leading-6 dark:bg-primary-950/30">{{ tr('editorHint') }}</p>
      <div class="flex flex-wrap items-center gap-3 text-sm">
        <span>{{ tr('selected', { count: visibleFields.filter(f => selected[f.key]).length }) }}</span>
        <button class="btn btn-secondary btn-sm" type="button" :disabled="busy" @click="captureSelected">{{ tr('capture') }}</button>
        <button class="btn btn-ghost btn-sm" type="button" :disabled="busy" @click="visibleFields.forEach(f => selected[f.key] = false)">{{ tr('clearSelection') }}</button>
      </div>
      <fieldset :disabled="busy" class="grid gap-3 md:grid-cols-2">
        <div v-for="field in visibleFields" :key="field.key" class="space-y-3 rounded-xl border p-4" :class="selected[field.key] ? 'border-primary-300 bg-primary-50/30 dark:border-primary-700' : 'border-gray-200 dark:border-dark-600'" :data-template-field="field.key">
          <label class="flex cursor-pointer items-start gap-2 text-sm font-medium"><input v-model="selected[field.key]" type="checkbox" class="mt-0.5" :aria-label="tr('cover', { field: tr(`fields.${field.key}`) })" /><span>{{ tr(`fields.${field.key}`) }}</span></label>
          <fieldset :disabled="!selected[field.key]" :class="!selected[field.key] ? 'opacity-45' : ''">
            <OpenAIAccountTemplateField v-model="draft[field.key]" :field="field" :proxies="proxies" :groups="groups" :profiles="profiles" />
          </fieldset>
          <p class="text-xs text-gray-500">{{ selected[field.key] ? tr('willCover') : tr('keepCurrent') }}</p>
        </div>
      </fieldset>
      <p v-if="editorError" role="alert" class="text-sm text-red-600">{{ editorError }}</p>
    </form>
    <template #footer><button class="btn btn-secondary" :disabled="busy" type="button" @click="closeEditor">{{ t('common.cancel') }}</button><button class="btn btn-primary" :disabled="busy" type="submit" :form="formID">{{ tr('save') }}</button></template>
  </BaseDialog>
</template>
<script setup lang="ts">
import { computed, onUnmounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import BaseDialog from '@/components/common/BaseDialog.vue'
import OpenAIAccountTemplateField from './OpenAIAccountTemplateField.vue'
import { openaiAccountTemplateAPI } from '@/api/admin/openaiAccountTemplate'
import { OPENAI_TEMPLATE_FIELDS, applyOpenAIAccountTemplate, cloneTemplateValue, type OpenAIAccountTemplate, type TemplateBindings } from '@/utils/openaiAccountTemplate'

const props = defineProps<{ kind: 'oauth' | 'apikey'; bindings: TemplateBindings; proxies: { id: number; name: string }[]; groups: { id: number; name: string }[]; profiles: { id: number; name: string }[] }>()
const { t } = useI18n()
const tr = (key: string, params: Record<string, string | number> = {}) => t(`admin.accounts.accountTemplate.${key}`, params)
const formID = `account-template-${Math.random().toString(36).slice(2)}`
const busy = ref(false)
const editing = ref(false)
const notice = ref('')
const error = ref('')
const editorError = ref('')
const selected = reactive<Record<string, boolean>>({})
const draft = reactive<Record<string, unknown>>({})
let saved: OpenAIAccountTemplate = { version: 1, fields: {} }
let generation = 0
const visibleFields = computed(() => OPENAI_TEMPLATE_FIELDS.filter(f => (!f.scope || f.scope === props.kind) && props.bindings[f.key]))
function failure(e: unknown) { const v = e as { response?: { data?: { message?: string } }; message?: string }; return v.response?.data?.message || v.message || tr('failed') }
async function openEditor() {
  if (busy.value) return
  const current = ++generation
  busy.value = true; error.value = ''; notice.value = ''
  try {
    const template = await openaiAccountTemplateAPI.get()
    if (current !== generation) return
    saved = template
    for (const field of visibleFields.value) {
      selected[field.key] = Object.prototype.hasOwnProperty.call(template.fields, field.key)
      draft[field.key] = cloneTemplateValue(selected[field.key] ? template.fields[field.key] : props.bindings[field.key]?.read() ?? field.default)
    }
    editorError.value = ''; editing.value = true
  } catch (e) { if (current === generation) error.value = failure(e) }
  finally { if (current === generation) busy.value = false }
}
function captureSelected() { for (const field of visibleFields.value) if (selected[field.key]) draft[field.key] = cloneTemplateValue(props.bindings[field.key].read()) }
function closeEditor() { if (!busy.value) editing.value = false }
async function save() {
  if (busy.value) return
  const current = ++generation
  busy.value = true; editorError.value = ''
  // Preserve fields for the other account type; unchecked visible fields are removed.
  const fields = cloneTemplateValue(saved.fields)
  for (const field of visibleFields.value) { if (selected[field.key]) fields[field.key] = cloneTemplateValue(draft[field.key]); else delete fields[field.key] }
  try {
    const result = await openaiAccountTemplateAPI.save({ version: 1, fields })
    if (current !== generation) return
    saved = result; editing.value = false; notice.value = tr('saved')
  } catch (e) { if (current === generation) editorError.value = failure(e) }
  finally { if (current === generation) busy.value = false }
}
async function apply() {
  if (busy.value) return
  const current = ++generation
  busy.value = true; error.value = ''; notice.value = ''
  try {
    const template = await openaiAccountTemplateAPI.get()
    if (current !== generation) return
    const count = await applyOpenAIAccountTemplate(template, props.bindings, props.kind)
    if (current === generation) notice.value = count ? tr('applied', { count }) : tr('empty')
  } catch (e) { if (current === generation) error.value = failure(e) }
  finally { if (current === generation) busy.value = false }
}
watch(() => props.kind, () => { generation++; busy.value = false; editing.value = false; notice.value = ''; error.value = '' })
onUnmounted(() => generation++)
</script>
