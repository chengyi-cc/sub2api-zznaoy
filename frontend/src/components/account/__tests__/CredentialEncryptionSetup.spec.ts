import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CredentialEncryptionSetup from '../CredentialEncryptionSetup.vue'
import { getCredentialEncryption, initializeCredentialEncryption, resetCredentialEncryption } from '@/api/admin/credentialEncryption'

vi.mock('vue-i18n', async importOriginal => ({ ...await importOriginal<typeof import('vue-i18n')>(), useI18n: () => ({ t: (key: string) => key }) }))
vi.mock('@/api/admin/credentialEncryption', () => ({ getCredentialEncryption: vi.fn(), initializeCredentialEncryption: vi.fn(), resetCredentialEncryption: vi.fn() }))
enableAutoUnmount(afterEach)
beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(getCredentialEncryption).mockResolvedValue({ configured: false, source: 'unconfigured' })
  vi.mocked(initializeCredentialEncryption).mockResolvedValue({ configured: true, source: 'local_file' })
})

describe('CredentialEncryptionSetup', () => {
  it('clears old credentials only after explicit confirmation of a missing-key recovery', async () => {
    vi.mocked(getCredentialEncryption).mockRejectedValue({ reason: 'CREDENTIAL_ENCRYPTION_KEY_MISSING' })
    vi.mocked(resetCredentialEncryption).mockResolvedValue({ configured: true, source: 'database' })
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    expect(wrapper.text()).toContain('keyMissing')
    expect(resetCredentialEncryption).not.toHaveBeenCalled()
    await wrapper.get('[data-testid="reset-credential-encryption"]').trigger('click')
    expect(resetCredentialEncryption).not.toHaveBeenCalled()
    await wrapper.get('[data-testid="confirm-reset-credential-encryption"]').trigger('click')
    await flushPromises()
    expect(resetCredentialEncryption).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([true])
    expect(wrapper.text()).toContain('databaseHint')
  })
  it('does not offer destructive recovery while database availability is unknown', async () => {
    vi.mocked(getCredentialEncryption).mockRejectedValue({ reason: 'CREDENTIAL_ENCRYPTION_KEY_MISSING' })
    vi.mocked(resetCredentialEncryption).mockRejectedValue({ reason: 'CREDENTIAL_ENCRYPTION_STORAGE_FAILED' })
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    await wrapper.get('[data-testid="reset-credential-encryption"]').trigger('click')
    await wrapper.get('[data-testid="confirm-reset-credential-encryption"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('initializeFailed')
    expect(wrapper.find('[data-testid="reset-credential-encryption"]').exists()).toBe(false)
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([false])
  })
  it('checks status without generating a key and initializes only on a click', async () => {
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    expect(initializeCredentialEncryption).not.toHaveBeenCalled()
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([false])
    expect(wrapper.find('input').exists()).toBe(false)
    await wrapper.get('[data-testid="initialize-credential-encryption"]').trigger('click')
    await flushPromises()
    expect(initializeCredentialEncryption).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([true])
    expect(wrapper.find('button').exists()).toBe(false)
    expect(wrapper.text()).toContain('tokenGuardV2.encryption.backupHint')
  })

  it('keeps an existing server key without offering replacement', async () => {
    vi.mocked(getCredentialEncryption).mockResolvedValue({ configured: true, source: 'server_config' })
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([true])
    expect(wrapper.find('button').exists()).toBe(false)
    expect(initializeCredentialEncryption).not.toHaveBeenCalled()
  })

  it('keeps saves blocked and allows a status retry when initialization fails', async () => {
    vi.mocked(initializeCredentialEncryption).mockRejectedValue(new Error('raw-private-error'))
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([false])
    expect(wrapper.get('[role="alert"]').text()).toContain('initializeFailed')
    expect(wrapper.text()).not.toContain('raw-private-error')
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(getCredentialEncryption).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[data-testid="initialize-credential-encryption"]').exists()).toBe(true)
  })

  it('does not offer initialization when key status is unavailable', async () => {
    vi.mocked(getCredentialEncryption).mockRejectedValue(new Error('private-file-path'))
    const wrapper = mount(CredentialEncryptionSetup)
    await flushPromises()
    expect(wrapper.emitted('ready')?.at(-1)).toEqual([false])
    expect(wrapper.find('[data-testid="initialize-credential-encryption"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('private-file-path')
  })
})
