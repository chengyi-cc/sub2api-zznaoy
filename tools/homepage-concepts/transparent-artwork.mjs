import { originalGoldArt } from './concepts.mjs'

let stopIndex = 0

export const transparentGoldArt = originalGoldArt
  .replace('class="hc-gold-art"', 'class="hc-gold-art" data-art-surface="transparent"')
  .replace('fill="#131410"', 'fill="none"')
  .replace('stroke="#bda477"', 'stroke="#bda477" style="stroke:var(--hf-metal-3,#bda477)"')
  .replace('stroke="#d2bc94"', 'stroke="#d2bc94" style="stroke:var(--hf-metal-1,#d2bc94)"')
  .replace(/<stop([^>]*?)stop-color="([^"]+)"/g, (_match, attributes, color) => `<stop${attributes}stop-color="${color}" style="stop-color:var(--hf-metal-${++stopIndex},${color})"`)
