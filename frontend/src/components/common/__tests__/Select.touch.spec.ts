import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import Select from '../Select.vue'

vi.mock('vue-i18n', () => ({ useI18n: () => ({ t: (key: string) => key }) }))

const options = Array.from({ length: 12 }, (_, i) => ({ value: `g${i}`, label: `Group ${i}` }))
let wrapper: ReturnType<typeof mount>

afterEach(() => {
  wrapper?.unmount()
  document.body.innerHTML = ''
  vi.unstubAllGlobals()
})

async function openSelect() {
  wrapper = mount(Select, { attachTo: document.body, props: { modelValue: 'g0', searchable: true, options } })
  await wrapper.get('button').trigger('click')
  await nextTick()
  await nextTick()
}

describe('Select on touch devices', () => {
  it('does not focus the search input (no soft keyboard) on a coarse pointer', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === '(pointer: coarse)', media: query, addEventListener() {}, removeEventListener() {} }))
    await openSelect()
    expect(document.querySelector('.select-search-input')).not.toBeNull()
    expect(document.activeElement).not.toBe(document.querySelector('.select-search-input'))
    expect(document.activeElement).toBe(document.querySelector('[role="listbox"]'))
  })

  it('still focuses the search input on a fine pointer', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: query === '(pointer: fine)', media: query, addEventListener() {}, removeEventListener() {} }))
    await openSelect()
    expect(document.activeElement).toBe(document.querySelector('.select-search-input'))
  })

  async function openAt(top: number, bottom: number, viewportHeight: number) {
    wrapper = mount(Select, { attachTo: document.body, props: { modelValue: 'g0', options } })
    vi.spyOn(wrapper.element as HTMLElement, 'getBoundingClientRect').mockReturnValue(
      { top, bottom, left: 0, right: 200, width: 200, height: bottom - top, x: 0, y: top, toJSON() {} } as DOMRect
    )
    vi.stubGlobal('innerHeight', viewportHeight)
    await wrapper.get('button').trigger('click')
    await nextTick()
    await nextTick()
    return {
      list: document.querySelector<HTMLElement>('.select-options')!,
      dropdown: document.querySelector<HTMLElement>('.select-dropdown-portal')!,
    }
  }

  it('shrinks the option list to the room left below the trigger', async () => {
    const { list, dropdown } = await openAt(100, 140, 300)
    expect(dropdown.style.top).toBe('144px')
    expect(list.style.maxHeight).toBe('152px')
  })

  it('opens upward when there is more room above', async () => {
    const { list, dropdown } = await openAt(500, 540, 700)
    expect(dropdown.style.bottom).not.toBe('')
    expect(list.style.maxHeight).toBe('320px')
  })
})
