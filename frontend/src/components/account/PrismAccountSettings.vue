<template>
  <section data-testid="prism-account-settings" class="rounded-xl border border-violet-200 bg-violet-50/40 p-5 dark:border-violet-900 dark:bg-violet-950/20">
    <div class="flex items-start justify-between gap-4">
      <div><h3 class="text-base font-semibold text-gray-900 dark:text-gray-100">{{ t('admin.prism.title') }}</h3><p class="mt-2 text-sm text-gray-500 dark:text-gray-400">{{ t('admin.prism.description') }}</p></div>
      <button type="button" role="switch" data-testid="prism-toggle" :aria-checked="enabled" :aria-label="t('admin.prism.title')" :class="['relative inline-flex h-8 w-14 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent focus:outline-none focus:ring-2 focus:ring-violet-500', enabled ? 'bg-violet-600' : 'bg-gray-200 dark:bg-dark-600']" @click="$emit('update:enabled', !enabled)">
        <span :class="['pointer-events-none inline-block h-7 w-7 rounded-full bg-white shadow transition', enabled ? 'translate-x-6' : 'translate-x-0']" />
      </button>
    </div>
    <fieldset v-if="enabled" class="mt-4 border-t border-violet-200 pt-4 dark:border-violet-900">
      <legend class="text-sm font-medium">{{ t('admin.prism.models') }}</legend>
      <div class="mt-2 grid gap-3 sm:grid-cols-2">
        <label v-for="model in PRISM_MODELS" :key="model" class="flex items-center gap-2 text-sm">
          <input type="checkbox" :checked="models.includes(model)" :data-testid="`prism-model-${model}`" @change="toggleModel(model, ($event.target as HTMLInputElement).checked)" />{{ model }}
        </label>
      </div>
      <p class="mt-3 text-xs leading-5 text-gray-500">{{ t('admin.prism.modelsHint') }}</p>
      <p class="mt-2 text-xs leading-5 text-amber-700 dark:text-amber-400">{{ t('admin.prism.requirements') }}</p>
      <p class="mt-2 text-xs leading-5 text-gray-500">{{ t('admin.prism.toolsHint') }}</p>
    </fieldset>
  </section>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { PRISM_MODELS } from '@/utils/prismAccount'
const props = defineProps<{ enabled: boolean; models: string[] }>()
const emit = defineEmits<{ 'update:enabled': [value: boolean]; 'update:models': [value: string[]] }>()
const { t } = useI18n()
function toggleModel(model: string, checked: boolean) {
  emit('update:models', PRISM_MODELS.filter(value => value === model ? checked : props.models.includes(value)))
}
</script>
