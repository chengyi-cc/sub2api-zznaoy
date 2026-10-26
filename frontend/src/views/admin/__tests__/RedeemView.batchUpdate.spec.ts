import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import RedeemView from '../RedeemView.vue'

const { listRedeemCodes, batchUpdateRedeemCodes, getAllGroups, showSuccess, showError, showInfo, generateRedeemCodes, clipboardCopy } =
  vi.hoisted(() => ({
    generateRedeemCodes: vi.fn(),
    clipboardCopy: vi.fn(),
    listRedeemCodes: vi.fn(),
    batchUpdateRedeemCodes: vi.fn(),
    getAllGroups: vi.fn(),
    showSuccess: vi.fn(),
    showError: vi.fn(),
    showInfo: vi.fn()
  }))

vi.mock('@/api/admin', () => ({
  adminAPI: {
    redeem: {
      list: listRedeemCodes,
      generate: generateRedeemCodes,
      delete: vi.fn(),
      batchDelete: vi.fn(),
      batchUpdate: batchUpdateRedeemCodes,
      exportCodes: vi.fn()
    },
    groups: {
      getAll: getAllGroups
    }
  }
}))

vi.mock('@/stores/app', () => ({
  useAppStore: () => ({
    showSuccess,
    showError,
    showInfo
  })
}))

vi.mock('@/composables/useClipboard', () => ({
  useClipboard: () => ({
    copyToClipboard: clipboardCopy
  })
}))

vi.mock('vue-i18n', async () => {
  const actual = await vi.importActual<typeof import('vue-i18n')>('vue-i18n')
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, params?: Record<string, unknown>) => params && key.endsWith('CodeLine') ? `${key} ${JSON.stringify(params)}` : key
    })
  }
})

const DataTableStub = {
  props: ['columns', 'data'],
  template: `
    <table>
      <thead>
        <tr>
          <th v-for="column in columns" :key="column.key">
            <slot :name="'header-' + column.key" :column="column">{{ column.label }}</slot>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in data" :key="row.id">
          <td v-for="column in columns" :key="column.key">
            <slot :name="'cell-' + column.key" :row="row" :value="row[column.key]">
              {{ row[column.key] }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
  `
}

const SelectStub = {
  props: ['modelValue', 'options'],
  emits: ['update:modelValue', 'change'],
  setup(props: { options: Array<{ value: unknown; label: string }> }, { emit }: { emit: (event: string, ...args: unknown[]) => void }) {
    const onChange = (event: Event) => {
      const raw = (event.target as HTMLSelectElement).value
      const option = props.options.find((item) => String(item.value ?? '') === raw)
      const value = option ? option.value : raw
      emit('update:modelValue', value)
      emit('change', value, option ?? null)
    }
    return { onChange }
  },
  template: `
    <select v-bind="$attrs" :value="modelValue ?? ''" @change="onChange">
      <option v-for="option in options" :key="String(option.value ?? '')" :value="option.value ?? ''">
        {{ option.label }}
      </option>
    </select>
  `
}

describe('admin RedeemView batch update', () => {
  beforeEach(() => {
    localStorage.clear()
    document.body.innerHTML = ''

    generateRedeemCodes.mockReset()
    clipboardCopy.mockReset()
    clipboardCopy.mockResolvedValue(true)
    generateRedeemCodes.mockResolvedValue([])
    listRedeemCodes.mockReset()
    batchUpdateRedeemCodes.mockReset()
    getAllGroups.mockReset()
    showSuccess.mockReset()
    showError.mockReset()
    showInfo.mockReset()

    listRedeemCodes.mockResolvedValue({
      items: [
        {
          id: 1,
          code: 'CODE-1',
          type: 'balance',
          value: 10,
          status: 'unused',
          used_by: null,
          used_at: null,
          created_at: '2026-01-01T00:00:00Z',
          expires_at: null
        },
        {
          id: 2,
          code: 'CODE-2',
          type: 'balance',
          value: 20,
          status: 'unused',
          used_by: null,
          used_at: null,
          created_at: '2026-01-01T00:00:00Z',
          expires_at: null
        }
      ],
      total: 2,
      page: 1,
      page_size: 20,
      pages: 1
    })
    batchUpdateRedeemCodes.mockResolvedValue({ updated: 1, message: 'ok' })
    getAllGroups.mockResolvedValue([])
  })

  it('submits only checked fields for selected redeem codes', async () => {
    const wrapper = mount(RedeemView, {
      attachTo: document.body,
      global: {
        stubs: {
          AppLayout: { template: '<div><slot /></div>' },
          TablePageLayout: {
            template: '<div><slot name="filters" /><slot name="table" /><slot name="pagination" /></div>'
          },
          DataTable: DataTableStub,
          Pagination: true,
          ConfirmDialog: true,
          Select: SelectStub,
          GroupBadge: true,
          GroupOptionItem: true,
          Icon: true,
          Teleport: true
        }
      }
    })

    await flushPromises()
    await wrapper.findAll('[data-test="select-code"]')[0].setValue(true)
    await wrapper.get('[data-test="batch-update-open"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-test="batch-field-status"]').setValue(true)
    await wrapper.get('[data-test="batch-status-select"]').setValue('disabled')
    await wrapper.get('[data-test="batch-field-notes"]').setValue(true)
    await wrapper.get('[data-test="batch-notes-input"]').setValue('maintenance')
    await wrapper.get('[data-test="batch-update-form"]').trigger('submit')
    await flushPromises()

    expect(batchUpdateRedeemCodes).toHaveBeenCalledWith([1], {
      status: 'disabled',
      notes: 'maintenance'
    })
    expect(showSuccess).toHaveBeenCalledWith('admin.redeem.batchUpdateSuccess')
  })
  const mountGeneration = async () => {
    const wrapper = mount(RedeemView, {
      attachTo: document.body,
      global: { stubs: {
        AppLayout: { template: '<div><slot /></div>' },
        TablePageLayout: { template: '<div><slot name="filters" /><slot name="table" /></div>' },
        DataTable: true, Pagination: true, ConfirmDialog: true, Select: SelectStub,
        GroupBadge: true, GroupOptionItem: true, Icon: true, Teleport: true
      } }
    })
    await flushPromises()
    await wrapper.get('[data-test="generate-open"]').trigger('click')
    return wrapper
  }

  it('requests one gift per paid code and copies the entire result in pairs', async () => {
    generateRedeemCodes.mockResolvedValue([
      { id: 1, code: 'PAID-CODE', type: 'balance', value: 10, is_gift: false, gift_pair_id: 'pair-a' },
      { id: 2, code: 'GIFT-CODE', type: 'balance', value: 0.1, is_gift: true, gift_pair_id: 'pair-a' }
    ])
    const wrapper = await mountGeneration()
    expect((wrapper.get('[data-test="generate-include-gift"]').element as HTMLInputElement).checked).toBe(false)
    await wrapper.get('[data-test="generate-include-gift"]').setValue(true)
    await wrapper.get('[data-test="generate-form"]').trigger('submit')
    await flushPromises()
    expect(generateRedeemCodes).toHaveBeenCalledWith(1, 'balance', 10, undefined, undefined, undefined, true, expect.stringMatching(/^redeem-generate-/))
    await wrapper.get('[data-test="copy-generated"]').trigger('click')
    const text = clipboardCopy.mock.calls[0][0]
    expect(text).toContain('PAID-CODE')
    expect(text).toContain('GIFT-CODE')
    expect(text).toContain('admin.redeem.giftCodeLine')
    expect(text).toContain('0.10')
    wrapper.unmount()
  })

  it('keeps the operation key on a failed request and creates a new key when input changes', async () => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => {})
    generateRedeemCodes.mockRejectedValue(new Error('Network timeout'))
    const wrapper = await mountGeneration()
    await wrapper.get('[data-test="generate-include-gift"]').setValue(true)
    await wrapper.get('[data-test="generate-form"]').trigger('submit'); await flushPromises()
    const firstKey = generateRedeemCodes.mock.calls[0][7]
    await wrapper.get('[data-test="generate-form"]').trigger('submit'); await flushPromises()
    expect(generateRedeemCodes.mock.calls[1][7]).toBe(firstKey)
    await wrapper.get('[data-test="generate-value"]').setValue(20)
    await wrapper.get('[data-test="generate-form"]').trigger('submit'); await flushPromises()
    expect(generateRedeemCodes.mock.calls[2][7]).not.toBe(firstKey)
    errorLog.mockRestore()
    wrapper.unmount()
  })

  it('blocks a second submit while generation is running', async () => {
    let finish!: (codes: never[]) => void
    generateRedeemCodes.mockReturnValue(new Promise((resolve) => { finish = resolve }))
    const wrapper = await mountGeneration()
    await wrapper.get('[data-test="generate-form"]').trigger('submit')
    await wrapper.get('[data-test="generate-form"]').trigger('submit')
    expect(generateRedeemCodes).toHaveBeenCalledTimes(1)
    finish([]); await flushPromises()
    wrapper.unmount()
  })

  it('clears gift selection when switching to another code type', async () => {
    const wrapper = await mountGeneration()
    await wrapper.get('[data-test="generate-include-gift"]').setValue(true)
    await wrapper.get('[data-test="generate-type"]').setValue('concurrency')
    expect(wrapper.find('[data-test="generate-include-gift"]').exists()).toBe(false)
    await wrapper.get('[data-test="generate-type"]').setValue('balance')
    expect((wrapper.get('[data-test="generate-include-gift"]').element as HTMLInputElement).checked).toBe(false)
    wrapper.unmount()
  })

})
