import { beforeEach, describe, expect, it, vi } from 'vitest'
import artworks from '@/assets/home-artworks.json'
import { applyRandomHomeArtwork, pickHomeArtwork } from '../homeArtworkSelection'

function createRoot() {
  const root = document.createElement('div')
  root.innerHTML = '<h1>Keep this layout</h1><div data-hc-artwork="random">Fallback</div><span data-hc-artwork-name>Original</span>'
  return root
}

describe('random homepage artwork', () => {
  beforeEach(() => {
    window.sessionStorage.clear()
  })

  it('keeps the original plus five accepted sculptures, without the rejected ribbon', () => {
    expect(artworks.map(artwork => artwork.id)).toEqual(['05-atelier', '06-armillary', '07-crystal', '08-corona', '09-trefoil', '10-lattice'])
    expect(artworks.every(artwork => !artwork.markup.includes('hc-gold-thread'))).toBe(true)
  })

  it.each(artworks.map((artwork, index) => [artwork.id, index] as const))('can select %s', (identifier, index) => {
    expect(pickHomeArtwork(artworks, null, () => (index + 0.5) / artworks.length)?.id).toBe(identifier)
  })

  it('excludes only the immediately previous sculpture', () => {
    for (const previous of artworks) {
      const choices = Array.from({ length: 5 }, (_unused, index) => pickHomeArtwork(artworks, previous.id, () => (index + 0.5) / 5)?.id)
      expect(new Set(choices).size).toBe(5)
      expect(choices).not.toContain(previous.id)
    }
  })

  it('handles empty and single-entry collections', () => {
    expect(pickHomeArtwork([], null)).toBeUndefined()
    expect(pickHomeArtwork([artworks[0]], artworks[0].id)).toEqual(artworks[0])
    const root = createRoot()
    expect(applyRandomHomeArtwork(root, [])).toBeUndefined()
    expect(root.querySelector('[data-hc-artwork]')?.textContent).toBe('Fallback')
  })

  it.each([-1, 1, 100, NaN, Infinity])('clamps invalid random input %s', value => {
    expect(artworks).toContain(pickHomeArtwork(artworks, null, () => value))
  })

  it('updates only opt-in artwork and labels, leaving the layout intact', () => {
    const root = createRoot()
    const identifier = applyRandomHomeArtwork(root, artworks, { random: () => 0, storage: null })
    expect(identifier).toBe('05-atelier')
    expect(root.querySelector('h1')?.textContent).toBe('Keep this layout')
    expect(root.querySelectorAll('svg')).toHaveLength(1)
    expect(root.querySelector('[data-hc-artwork-name]')?.textContent).toBe('金色轨道')
    expect(root.querySelector('[data-hc-current-art]')?.getAttribute('data-hc-current-art')).toBe(identifier)
    const untouched = document.createElement('div')
    untouched.innerHTML = '<div>Existing homepage</div>'
    expect(applyRandomHomeArtwork(untouched, artworks)).toBeUndefined()
    expect(untouched.innerHTML).toBe('<div>Existing homepage</div>')
  })

  it('initializes once, but supports an explicit reroll without replacing the layout', () => {
    const root = createRoot()
    const random = vi.fn(() => 0)
    expect(applyRandomHomeArtwork(root, artworks, { random })).toBe('05-atelier')
    expect(applyRandomHomeArtwork(root, artworks, { random })).toBe('05-atelier')
    expect(random).toHaveBeenCalledTimes(1)
    expect(applyRandomHomeArtwork(root, artworks, { random, force: true })).toBe('06-armillary')
    expect(root.querySelector('h1')?.textContent).toBe('Keep this layout')
  })

  it('remembers the previous entry across homepage mounts within one session', () => {
    expect(applyRandomHomeArtwork(createRoot(), artworks, { random: () => 0 })).toBe('05-atelier')
    expect(applyRandomHomeArtwork(createRoot(), artworks, { random: () => 0 })).toBe('06-armillary')
    expect(window.sessionStorage.getItem('hc:last-hero-artwork')).toBe('06-armillary')
  })

  it.each(['read', 'write'])('still renders when storage blocks %s', operation => {
    const storage = {
      getItem: () => { if (operation === 'read') throw new Error('Blocked'); return null },
      setItem: () => { if (operation === 'write') throw new Error('Blocked') },
    }
    const root = createRoot()
    expect(applyRandomHomeArtwork(root, artworks, { random: () => 0, storage })).toBe('05-atelier')
    expect(root.querySelector('svg')).not.toBeNull()
  })
})
