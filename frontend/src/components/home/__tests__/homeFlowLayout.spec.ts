import { describe, expect, it } from 'vitest'
import { FLOW_HUB, HOME_FLOW_MODELS, buildHomeFlowLayout, flowIcon, flowIconFill } from '../homeFlowLayout'

describe('home flow layout', () => {
  it('keeps every chip inside the 1200-wide drawing area', () => {
    const { tools, models } = buildHomeFlowLayout(4, 8)
    expect(tools).toHaveLength(4)
    expect(models).toHaveLength(8)
    for (const chip of [...tools, ...models]) {
      expect(chip.x).toBeGreaterThanOrEqual(0)
      expect(chip.x + chip.width).toBeLessThanOrEqual(1200)
    }
  })

  it('routes tool links into the left edge of the hub and model links out of the right edge', () => {
    const { links } = buildHomeFlowLayout(4, 8)
    expect(links).toHaveLength(12)
    for (const link of links.slice(0, 4)) expect(link.d).toMatch(new RegExp(` ${FLOW_HUB.cx - FLOW_HUB.halfWidth},[\\d.]+$`))
    for (const link of links.slice(4)) expect(link.d.startsWith(`M${FLOW_HUB.cx + FLOW_HUB.halfWidth},`)).toBe(true)
  })

  it('centres a single chip on the hub instead of dividing by zero', () => {
    const { tools, links } = buildHomeFlowLayout(1, 1)
    expect(tools[0].y + tools[0].height / 2).toBe(FLOW_HUB.cy)
    expect(links.every(link => !link.d.includes('NaN'))).toBe(true)
  })

  it('lets near-black brand marks follow the text colour so they stay visible on dark', () => {
    expect(flowIconFill('#000000')).toBe('currentColor')
    expect(flowIconFill('#16191E')).toBe('currentColor')
    expect(flowIconFill('#D97706')).toBe('#D97706')
    expect(flowIconFill('not-a-colour')).toBe('currentColor')
  })

  it('has icon artwork for every model shown on the homepage', () => {
    for (const model of HOME_FLOW_MODELS) expect(flowIcon(model.icon)?.paths.length, model.icon).toBeGreaterThan(0)
    expect(flowIcon('does-not-exist')).toBeNull()
  })
})
