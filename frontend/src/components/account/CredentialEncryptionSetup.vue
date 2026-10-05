<template>
  <section class="rounded-xl border border-gray-200 bg-white p-4 dark:border-dark-600 dark:bg-dark-800" data-testid="credential-encryption-setup" :aria-busy="busy">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h3 class="font-semibold text-gray-900 dark:text-gray-100">{{ t('tokenGuardV2.encryption.title') }}</h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400" role="status">{{ t(status?.configured ? 'tokenGuardV2.encryption.ready' : busy ? 'tokenGuardV2.encryption.checking' : error ? 'tokenGuardV2.encryption.unavailable' : 'tokenGuardV2.encryption.required') }}</p>
      </div>
      <button v-if="error" type="button" class="btn btn-secondary" :disabled="busy" @click="check">{{ t('tokenGuardV2.encryption.retry') }}</button>
      <button v-else-if="status && !status.configured" type="button" class="btn btn-primary" :disabled="busy" data-testid="initialize-credential-encryption" @click="initialize">{{ t(busy ? 'tokenGuardV2.encryption.initializing' : 'tokenGuardV2.encryption.initialize') }}</button>
    </div>
    <p v-if="error" role="alert" class="mt-2 text-sm text-red-600 dark:text-red-400">{{ error }}</p>
    <div v-if="recoverable" class="mt-3">
      <button v-if="!confirmReset" type="button" class="btn btn-danger" :disabled="busy" data-testid="reset-credential-encryption" @click="confirmReset = true">{{ t('tokenGuardV2.encryption.reset') }}</button>
      <div v-else class="space-y-2 rounded-lg border border-red-300 p-3" role="alert">
        <p class="text-sm">{{ t('tokenGuardV2.encryption.resetConfirm') }}</p>
        <button type="button" class="btn btn-danger" :disabled="busy" data-testid="confirm-reset-credential-encryption" @click="request('reset')">{{ t('tokenGuardV2.encryption.resetConfirmButton') }}</button>
        <button type="button" class="btn btn-secondary ml-2" :disabled="busy" @click="confirmReset = false">{{ t('tokenGuardV2.encryption.cancel') }}</button>
      </div>
    </div>
    <p v-if="status && !status.configured" class="mt-2 text-sm text-gray-500 dark:text-gray-400">{{ t('tokenGuardV2.encryption.description') }}</p>
    <p v-if="status?.source === 'local_file'" class="mt-2 text-xs text-gray-500 dark:text-gray-400">{{ t('tokenGuardV2.encryption.backupHint') }}</p>
    <p v-if="status?.source === 'database'" class="mt-2 text-xs text-gray-500 dark:text-gray-400">{{ t('tokenGuardV2.encryption.databaseHint') }}</p>
    <slot />
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCredentialEncryption, initializeCredentialEncryption, resetCredentialEncryption, type CredentialEncryptionStatus } from '@/api/admin/credentialEncryption'

const emit = defineEmits<{ ready: [value: boolean] }>()
const { t } = useI18n()
const status = ref<CredentialEncryptionStatus | null>(null)
const busy = ref(false)
const error = ref('')
const recoverable = ref(false)
const confirmReset = ref(false)

async function request(action: 'check' | 'initialize' | 'reset') {
  if (busy.value) return
  busy.value = true
  error.value = ''
  recoverable.value = false
  confirmReset.value = false
  emit('ready', false)
  try {
    status.value = await (action === 'reset' ? resetCredentialEncryption() : action === 'initialize' ? initializeCredentialEncryption() : getCredentialEncryption())
    emit('ready', status.value.configured)
  } catch (err) {
    status.value = null
    const reason = (err as { reason?: string })?.reason
    const messages: Record<string, string> = {
      CREDENTIAL_ENCRYPTION_KEY_MISSING: 'keyMissing',
      CREDENTIAL_RECOVERY_IN_USE: 'recoveryInUse',
      CREDENTIAL_RECOVERY_NOT_NEEDED: 'recoveryNotNeeded',
      CREDENTIAL_RECOVERY_CLEAR_FAILED: 'clearFailed',
      CREDENTIAL_RECOVERY_INITIALIZE_FAILED: 'recoveryInitializeFailed'
    }
    recoverable.value = reason === 'CREDENTIAL_ENCRYPTION_KEY_MISSING' || reason === 'CREDENTIAL_RECOVERY_IN_USE' || reason === 'CREDENTIAL_RECOVERY_CLEAR_FAILED'
    error.value = t(`tokenGuardV2.encryption.${(reason && messages[reason]) || (action === 'check' ? 'loadFailed' : 'initializeFailed')}`)
  } finally {
    busy.value = false
  }
}
const check = () => request('check')
const initialize = () => request('initialize')
onMounted(check)
</script>
