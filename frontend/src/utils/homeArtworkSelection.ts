export interface HomeArtwork {
  id: string
  name: string
  markup: string
}

export interface HomeArtworkOptions {
  random?: () => number
  storage?: Pick<Storage, 'getItem' | 'setItem'> | null
  force?: boolean
}

export function pickHomeArtwork(artworks: readonly HomeArtwork[], previous: string | null, random = Math.random): HomeArtwork | undefined {
  const alternatives = artworks.filter(artwork => artwork.id !== previous)
  const pool = alternatives.length ? alternatives : artworks
  if (!pool.length) return undefined
  const value = random()
  const fraction = Number.isFinite(value) ? Math.max(0, Math.min(value, 0.999999999)) : 0
  return pool[Math.floor(fraction * pool.length)]
}

export function applyRandomHomeArtwork(root: ParentNode, artworks: readonly HomeArtwork[], options: HomeArtworkOptions = {}): string | undefined {
  const slots = Array.from(root.querySelectorAll<HTMLElement>('[data-hc-artwork="random"]'))
  if (!slots.length || !artworks.length) return undefined
  const existing = slots[0].dataset.hcCurrentArt
  if (existing && !options.force && slots.every(slot => slot.dataset.hcCurrentArt === existing)) return existing
  let storage = options.storage
  let previous: string | null = existing || null
  try {
    if (storage === undefined) storage = window.sessionStorage
    previous = previous || storage?.getItem('hc:last-hero-artwork') || null
  } catch {
    storage = null
  }
  const selected = pickHomeArtwork(artworks, previous, options.random)
  if (!selected) return undefined
  for (const slot of slots) {
    slot.innerHTML = selected.markup
    slot.dataset.hcCurrentArt = selected.id
  }
  for (const label of root.querySelectorAll<HTMLElement>('[data-hc-artwork-name]')) label.textContent = selected.name
  try {
    storage?.setItem('hc:last-hero-artwork', selected.id)
  } catch {
    storage = null
  }
  return selected.id
}
