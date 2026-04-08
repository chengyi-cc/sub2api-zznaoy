import { describe, expect, it } from 'vitest'

import en from '../locales/en'
import ru from '../locales/ru'
import zh from '../locales/zh'

describe('usage service tier locale keys', () => {
  it('contains zh labels for service tier tooltip', () => {
    expect(zh.usage.serviceTier).toBe('服务档位')
    expect(zh.usage.serviceTierPriority).toBe('Fast')
    expect(zh.usage.serviceTierFlex).toBe('Flex')
    expect(zh.usage.serviceTierStandard).toBe('Standard')
  })

  it('contains en labels for service tier tooltip', () => {
    expect(en.usage.serviceTier).toBe('Service tier')
    expect(en.usage.serviceTierPriority).toBe('Fast')
    expect(en.usage.serviceTierFlex).toBe('Flex')
    expect(en.usage.serviceTierStandard).toBe('Standard')
  })

  it('contains ru labels for service tier tooltip', () => {
    expect(ru.usage.serviceTier).toBe('\u0423\u0440\u043E\u0432\u0435\u043D\u044C \u0441\u0435\u0440\u0432\u0438\u0441\u0430')
    expect(ru.usage.serviceTierPriority).toBe('\u0411\u044B\u0441\u0442\u0440\u044B\u0439')
    expect(ru.usage.serviceTierFlex).toBe('\u0413\u0438\u0431\u043A\u0438\u0439')
    expect(ru.usage.serviceTierStandard).toBe('\u0421\u0442\u0430\u043D\u0434\u0430\u0440\u0442\u043D\u044B\u0439')
  })
})
