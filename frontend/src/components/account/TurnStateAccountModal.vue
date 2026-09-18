<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { adminAPI } from '@/api/admin'
import { extractApiErrorMessage } from '@/utils/apiError'
import type { Account } from '@/types'
import BaseDialog from '@/components/common/BaseDialog.vue'
import TurnStateAutoField from './TurnStateAutoField.vue'

const props = defineProps<{ account: Account }>()
const emit = defineEmits<{ close: []; updated: [account: Account] }>()
const { locale } = useI18n()
const chinese = computed(() => locale.value.startsWith('zh'))
const label = (zh: string, en: string) => chinese.value ? zh : en
const enabled = ref(props.account.extra?.codex_turn_state_auto_enabled === true)
const profile = ref(String(props.account.extra?.codex_turn_state_profile || 'team'))
const source = ref(String(props.account.extra?.codex_turn_state_source || 'purchased'))
const saving = ref(false)
const error = ref('')
const saved = ref(false)
const field = ref<InstanceType<typeof TurnStateAutoField>>()

async function save(): Promise<void> {
  if (saving.value) return
  saving.value = true
  error.value = ''
  saved.value = false
  try {
    const current = await adminAPI.accounts.getById(props.account.id)
    const updated = await adminAPI.accounts.update(props.account.id, { extra: {
      ...current.extra,
      codex_turn_state_auto_enabled: enabled.value,
      codex_turn_state_profile: profile.value,
      codex_turn_state_source: source.value
    } })
    saved.value = true
    emit('updated', updated)
    field.value?.refreshStatus()
  } catch (failure) {
    error.value = extractApiErrorMessage(failure, label('保存失败，请重试。', 'Save failed. Please retry.'))
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <BaseDialog :show="true" :title="label('请求头采集', 'State acquisition') + ' · ' + account.name" width="wide" :close-on-escape="!saving" :show-close-button="!saving" @close="emit('close')">
    <div class="mb-3 flex items-center gap-3">
      <button type="button" class="btn btn-primary text-xs" :disabled="saving" data-testid="turn-state-save-account" @click="save">{{ saving ? label('保存中…', 'Saving…') : label('保存账号采集设置', 'Save account acquisition settings') }}</button>
      <span v-if="saved" role="status" class="text-xs text-green-600">{{ label('账号设置已保存', 'Account settings saved') }}</span>
    </div>
    <p v-if="error" role="alert" class="mb-3 text-xs text-red-600">{{ error }}</p>
    <fieldset :disabled="saving" @change="saved = false">
      <TurnStateAutoField ref="field" :account-id="account.id" v-model="enabled" v-model:profile="profile" v-model:source="source" :initial-settings="true" />
    </fieldset>
  </BaseDialog>
</template>
