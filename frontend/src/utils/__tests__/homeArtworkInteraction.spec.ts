import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import artworks from '@/assets/home-artworks.json'
import { applyRandomHomeArtwork } from '../homeArtworkSelection'
import { initializeHomeArtworkInteractions } from '../homeArtworkInteraction'
import { createHomeArtworkRenderer, renderHomeArtwork } from '../homeArtworkRenderer'

let clock = 0
let frameId = 0
let reducedMotion: MediaQueryList
let frames: Map<number, FrameRequestCallback>
let cleanups: Array<() => void>

function advance(milliseconds = 900, frameDuration = 16) {
  const end = clock + milliseconds
  while (clock < end) {
    clock += frameDuration
    vi.advanceTimersByTime(frameDuration)
    const callbacks = [...frames.values()]
    frames.clear()
    callbacks.forEach(callback => callback(clock))
  }
}

function setup(ownerDocument = document) {
  const root = ownerDocument.createElement('div')
  root.innerHTML = '<div class="hf-stage"><input type="checkbox" class="hf-motion-pause"><div class="hf-art" data-hc-artwork="random"></div><button data-hc-art-reset>Reset</button></div>'
  ownerDocument.body.append(root)
  const slot = root.querySelector<HTMLElement>('.hf-art')!
  vi.spyOn(slot, 'getBoundingClientRect').mockReturnValue({ width: 400, height: 394, x: 0, y: 0, top: 0, left: 0, right: 400, bottom: 394, toJSON: () => ({}) })
  const captured = new Set<number>()
  slot.setPointerCapture = vi.fn(identifier => { captured.add(identifier) })
  slot.hasPointerCapture = vi.fn(identifier => captured.has(identifier))
  slot.releasePointerCapture = vi.fn(identifier => { captured.delete(identifier) })
  applyRandomHomeArtwork(root, [artworks[5]], { storage: null })
  const cleanup = initializeHomeArtworkInteractions(root, artworks)
  cleanups.push(cleanup)
  return { root, slot, cleanup, pause: root.querySelector<HTMLInputElement>('.hf-motion-pause')!, reset: root.querySelector<HTMLButtonElement>('[data-hc-art-reset]')!, svg: slot.querySelector('svg')! }
}

function pointer(target: EventTarget, type: string, horizontal: number, vertical: number, options: { pointerId?: number; pointerType?: string; button?: number; isPrimary?: boolean } = {}) {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: horizontal, clientY: vertical, button: options.button ?? 0 })
  Object.defineProperties(event, {
    pointerId: { value: options.pointerId ?? 1 },
    pointerType: { value: options.pointerType ?? 'mouse' },
    isPrimary: { value: options.isPrimary ?? true },
  })
  target.dispatchEvent(event)
}

describe('interactive homepage sculptures', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
    clock = 0
    frameId = 0
    frames = new Map()
    cleanups = []
    reducedMotion = Object.assign(new EventTarget(), { matches: false, media: '(prefers-reduced-motion: reduce)' }) as MediaQueryList
    vi.spyOn(window, 'matchMedia').mockReturnValue(reducedMotion)
    vi.spyOn(window.performance, 'now').mockImplementation(() => clock)
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++frameId, callback); return frameId })
    vi.stubGlobal('cancelAnimationFrame', (identifier: number) => { frames.delete(identifier) })
  })

  afterEach(() => {
    cleanups.forEach(cleanup => cleanup())
    document.body.replaceChildren()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('starts the animation loop immediately without an introduction timer', () => {
    const { slot } = setup()
    expect(frames.size).toBe(1)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    advance(16)
    expect(vi.getTimerCount()).toBe(0)
    expect(slot.dataset.hcInteractive).toBe('true')
    expect(slot.tabIndex).toBe(0)
    expect(slot.getAttribute('role')).toBe('button')
    expect(slot.getAttribute('aria-label')).toContain('按住拖拽旋转')
  })

  it('clicks rotate the geometry and central emblem rather than a flat image transform', () => {
    const { slot, svg } = setup()
    const previous = svg.querySelector('[data-depth-layer="core"] path')!.getAttribute('d')
    slot.click()
    advance()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(32, 1)
    expect(svg.querySelector('[data-depth-layer="core"] path')!.getAttribute('d')).not.toBe(previous)
    expect(svg.innerHTML).toContain('hc-live-10-lattice')
    expect(svg.style.transform).toBe('')
    expect(slot.dataset.hcActiveInteraction).toBeUndefined()
    expect(frames.size).toBe(0)
  })

  it('drags track the pointer in both axes and retain the pointer outside the sculpture', () => {
    const { slot } = setup()
    pointer(slot, 'pointerdown', 100, 100)
    advance(32)
    pointer(window, 'pointermove', 200, 150)
    advance(32)
    expect(slot.setPointerCapture).toHaveBeenCalledWith(1)
    expect(slot.dataset.hcDragging).toBe('true')
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(60, 1)
    expect(Number(slot.dataset.hcPitch)).toBeCloseTo(-22.5, 1)
    const beforeRelease = Number(slot.dataset.hcYaw)
    pointer(window, 'pointerup', 200, 150)
    advance()
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(beforeRelease)
    expect(slot.releasePointerCapture).toHaveBeenCalledWith(1)
    expect(slot.dataset.hcDragging).toBeUndefined()
    expect(frames.size).toBe(0)
    const afterInertia = slot.dataset.hcYaw
    slot.click()
    advance()
    expect(slot.dataset.hcYaw).toBe(afterInertia)
  })

  it('a stationary click still turns after pointer down and up', () => {
    const { slot } = setup()
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointerup', 100, 100)
    slot.click()
    advance()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(32, 1)
  })

  it('leaves vertical touch scrolling alone, but supports horizontal touch dragging', () => {
    const { slot } = setup()
    pointer(slot, 'pointerdown', 100, 100, { pointerType: 'touch' })
    pointer(window, 'pointermove', 102, 160, { pointerType: 'touch' })
    advance()
    expect(slot.dataset.hcYaw).toBeUndefined()
    expect(slot.dataset.hcActiveInteraction).toBeUndefined()
    pointer(slot, 'pointerdown', 100, 100, { pointerType: 'touch' })
    pointer(window, 'pointermove', 160, 102, { pointerType: 'touch' })
    advance(40)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(20)
    pointer(window, 'pointercancel', 160, 102, { pointerType: 'touch' })
    advance()
    expect(frames.size).toBe(0)
  })

  it('ignores right click, non-primary pointers and unrelated pointer IDs', () => {
    const { slot } = setup()
    pointer(slot, 'pointerdown', 100, 100, { button: 2 })
    pointer(window, 'pointermove', 180, 100)
    pointer(slot, 'pointerdown', 100, 100, { isPrimary: false })
    pointer(window, 'pointermove', 180, 100)
    expect(slot.setPointerCapture).not.toHaveBeenCalled()
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointermove', 180, 100, { pointerId: 2 })
    advance()
    expect(slot.dataset.hcYaw).toBeUndefined()
    pointer(window, 'pointercancel', 100, 100)
  })

  it('allows vertical rotation beyond a full turn while reduced motion disables inertia, not manual control', () => {
    Object.assign(reducedMotion, { matches: true })
    const { slot } = setup()
    slot.click()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(32, 1)
    expect(frames.size).toBe(0)
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointermove', 200, -900)
    advance(32)
    pointer(window, 'pointerup', 200, -900)
    const yaw = slot.dataset.hcYaw
    advance()
    expect(Number(slot.dataset.hcPitch)).toBe(450)
    expect(slot.dataset.hcYaw).toBe(yaw)
    expect(frames.size).toBe(0)
  })

  it('pause stops click easing and inertia, while deliberate rotation remains available', () => {
    const { slot, pause } = setup()
    slot.click()
    advance(64)
    pause.checked = true
    pause.dispatchEvent(new Event('change'))
    const stopped = slot.dataset.hcYaw
    advance()
    expect(slot.dataset.hcYaw).toBe(stopped)
    expect(frames.size).toBe(0)
    slot.click()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(Number(stopped) + 32, 0)
    expect(frames.size).toBe(0)
  })

  it('keyboard rotation and reset restore the exact accepted starting sculpture', () => {
    const { slot, svg, reset } = setup()
    const initial = svg.innerHTML
    slot.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }))
    advance()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(15, 1)
    slot.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true, cancelable: true }))
    advance()
    expect(Number(slot.dataset.hcPitch)).toBeCloseTo(15, 1)
    reset.click()
    expect(svg.innerHTML).toBe(initial)
    expect(slot.dataset.hcYaw).toBe('0.0')
    slot.click()
    advance(48)
    slot.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true }))
    expect(svg.innerHTML).toBe(initial)
    expect(frames.size).toBe(1)
  })

  it.each(['pointercancel', 'lostpointercapture'])('%s ends dragging without leaving an animation running', cancellation => {
    const { slot } = setup()
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointermove', 200, 100)
    pointer(cancellation === 'lostpointercapture' ? slot : window, cancellation, 200, 100)
    advance()
    expect(slot.dataset.hcActiveInteraction).toBeUndefined()
    expect(slot.dataset.hcDragging).toBeUndefined()
    expect(frames.size).toBe(0)
  })

  it('losing focus does not stop visible animation, and cleanup removes listeners and pending frames', () => {
    const { slot, cleanup } = setup()
    slot.click()
    advance(32)
    window.dispatchEvent(new Event('blur'))
    advance(500)
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(32, 1)
    slot.click()
    expect(frames.size).toBeGreaterThan(0)
    cleanup()
    expect(frames.size).toBe(0)
    expect(slot.dataset.hcInteractive).toBeUndefined()
    expect(slot.hasAttribute('role')).toBe(false)
    slot.click()
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointermove', 200, 100)
    advance()
    expect(slot.dataset.hcYaw).toBeUndefined()
    expect(frames.size).toBe(0)
  })

  it('reinitialization does not install duplicate click or pointer handlers', () => {
    const { root, slot } = setup()
    cleanups.push(initializeHomeArtworkInteractions(root, artworks))
    slot.click()
    advance()
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(32, 1)
  })

  it.each(artworks)('reuses $id elements while preserving geometry and occlusion through full turns', artwork => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    const expected = svg.cloneNode(false) as SVGSVGElement
    document.body.append(svg)
    const render = createHomeArtworkRenderer(svg, artwork.model)
    let definitions: Element | null = null
    let core: Element | null = null
    for (const [pitch, yaw] of [[0, 0], [0.35, 0.6], [1.57, 2.7], [3.14, 4.2], [0, Math.PI * 2], [-4, -5], [0, 0]]) {
      render(pitch, yaw)
      expected.innerHTML = renderHomeArtwork(artwork.model, pitch, yaw)
      expect(svg.isEqualNode(expected)).toBe(true)
      definitions ??= svg.querySelector('defs')
      core ??= svg.querySelector('[data-depth-layer="core"]')
      expect(svg.querySelector('defs')).toBe(definitions)
      expect(svg.querySelector('[data-depth-layer="core"]')).toBe(core)
    }
  })

  it('rotates the geometry on the first frame and continues without user input', () => {
    const { slot, svg } = setup()
    const original = svg.innerHTML
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    advance(16)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(0)
    expect(svg.innerHTML).not.toBe(original)
    advance(1000)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(8)
    expect(Number(slot.dataset.hcPitch)).toBeGreaterThan(0)
    const angle = Number(slot.dataset.hcYaw)
    advance(1000)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(angle + 7)
  })

  it('dragging takes over auto-rotation immediately and restarts it only after an idle delay', () => {
    const { slot } = setup()
    advance(6200)
    pointer(slot, 'pointerdown', 100, 100)
    expect(slot.dataset.hcAutoPhase).toBe('waiting')
    const start = Number(slot.dataset.hcYaw)
    pointer(window, 'pointermove', 200, 100)
    advance(48)
    expect(Number(slot.dataset.hcYaw)).toBeCloseTo(start + 60, 0)
    advance(120)
    pointer(window, 'pointerup', 200, 100)
    const released = slot.dataset.hcYaw
    advance(2200)
    expect(slot.dataset.hcYaw).toBe(released)
    expect(slot.dataset.hcAutoPhase).toBe('waiting')
    advance(1200)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(Number(released))
  })

  it('mouse dragging can rotate vertically through multiple turns and reverse without hitting a stop', () => {
    const { slot, pause } = setup()
    pause.checked = true
    pause.dispatchEvent(new Event('change'))
    pointer(slot, 'pointerdown', 100, 100)
    pointer(window, 'pointermove', 100, -1900)
    advance(48)
    expect(Number(slot.dataset.hcPitch)).toBe(900)
    pointer(window, 'pointermove', 100, 2100)
    advance(48)
    expect(Number(slot.dataset.hcPitch)).toBe(-900)
    pointer(window, 'pointerup', 100, 2100)
    expect(frames.size).toBe(0)
  })

  it('pause and reduced-motion changes stop auto-rotation and cancel its delayed start', () => {
    const { slot, pause } = setup()
    advance(6200)
    pause.checked = true
    pause.dispatchEvent(new Event('change'))
    const angle = slot.dataset.hcYaw
    advance(6000)
    expect(slot.dataset.hcYaw).toBe(angle)
    expect(slot.dataset.hcAutoPhase).toBe('paused')
    expect(frames.size).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
    pause.checked = false
    pause.dispatchEvent(new Event('change'))
    Object.assign(reducedMotion, { matches: true })
    reducedMotion.dispatchEvent(new Event('change'))
    advance(6000)
    expect(slot.dataset.hcYaw).toBe(angle)
    expect(vi.getTimerCount()).toBe(0)
    Object.assign(reducedMotion, { matches: false })
    reducedMotion.dispatchEvent(new Event('change'))
    advance(3400)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
  })

  it('reset restores the exact original and resumes rotation on the next frame', () => {
    const { slot, svg, reset } = setup()
    const original = svg.innerHTML
    advance(6200)
    reset.click()
    expect(svg.innerHTML).toBe(original)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    expect(vi.getTimerCount()).toBe(0)
    advance(16)
    expect(svg.innerHTML).not.toBe(original)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(0)
  })

  it('keeps rotating when a visible preview loses focus but suspends while its document is hidden', () => {
    const { slot } = setup()
    advance(6200)
    window.dispatchEvent(new Event('blur'))
    const stopped = Number(slot.dataset.hcYaw)
    advance(10000)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(stopped)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
    document.dispatchEvent(new Event('visibilitychange'))
    advance(6000)
    expect(frames.size).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
    hidden.mockRestore()
    document.dispatchEvent(new Event('visibilitychange'))
    advance(3400)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
  })

  it.each([8, 16])('updates every %s ms frame without rebuilding the SVG tree', frameDuration => {
    const { svg } = setup()
    advance(frameDuration, frameDuration)
    const definitions = svg.querySelector('defs')!
    const core = svg.querySelector('[data-depth-layer="core"] path')!
    const path = svg.querySelector('[data-depth-layer="back"] path')!
    const updates = vi.spyOn(core, 'setAttribute')
    const replacements = vi.spyOn(svg, 'innerHTML', 'set')
    advance(frameDuration * 20, frameDuration)
    expect(updates.mock.calls.filter(([name]) => name === 'd')).toHaveLength(20)
    expect(replacements).not.toHaveBeenCalled()
    expect(svg.querySelector('defs')).toBe(definitions)
    expect(svg.querySelector('[data-depth-layer="back"] path')).toBe(path)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('uses one clock even when the frame callback timestamp belongs to a different time origin', () => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.set(++frameId, () => callback(clock - 1000000))
      return frameId
    })
    const { slot } = setup()
    advance(3100)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(3)
    slot.click()
    const previous = Number(slot.dataset.hcYaw)
    advance(500)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(previous + 25)
  })

  it('rotates a sandboxed preview using the parent scheduler when child timers and animation frames cannot run', () => {
    const frame = document.createElement('iframe')
    frame.setAttribute('sandbox', 'allow-same-origin')
    document.body.append(frame)
    const child = frame.contentWindow!
    vi.spyOn(child.document, 'hidden', 'get').mockReturnValue(false)
    const childTimeout = vi.spyOn(child, 'setTimeout').mockReturnValue(0)
    const childFrame = vi.fn(() => 0)
    Object.defineProperty(child, 'requestAnimationFrame', { configurable: true, value: childFrame })
    vi.spyOn(child.performance, 'now').mockReturnValue(1000000)
    const { slot } = setup(child.document)
    expect(slot.dataset.hcAutoPhase).toBe('spinning')
    advance(16)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(0)
    advance(1000)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(3)
    expect(childTimeout).not.toHaveBeenCalled()
    expect(childFrame).not.toHaveBeenCalled()
    child.dispatchEvent(new child.Event('blur'))
    const previous = Number(slot.dataset.hcYaw)
    advance(1000)
    expect(Number(slot.dataset.hcYaw)).toBeGreaterThan(previous)
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
    document.dispatchEvent(new Event('visibilitychange'))
    advance(5000)
    expect(slot.dataset.hcAutoPhase).toBe('suspended')
    expect(frames.size).toBe(0)
    hidden.mockRestore()
  })

  it('disconnects visibility observation and clears autoplay when disposed', () => {
    let notify: IntersectionObserverCallback = () => {}
    const disconnect = vi.fn()
    vi.stubGlobal('IntersectionObserver', class {
      constructor(callback: IntersectionObserverCallback) { notify = callback }
      observe() {}
      disconnect = disconnect
    })
    const { slot, cleanup } = setup()
    advance(6200)
    notify([{ target: slot, isIntersecting: false } as IntersectionObserverEntry], {} as IntersectionObserver)
    expect(slot.dataset.hcAutoPhase).toBe('suspended')
    advance(6000)
    expect(frames.size).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
    notify([{ target: slot, isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)
    expect(vi.getTimerCount()).toBe(0)
    expect(frames.size).toBe(1)
    cleanup()
    advance(6000)
    expect(disconnect).toHaveBeenCalledOnce()
    expect(slot.dataset.hcAutoPhase).toBeUndefined()
    expect(frames.size).toBe(0)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe('spatial artwork projection', () => {
  it.each(artworks.map(artwork => [artwork.id, artwork.model] as const))('%s stays finite and keeps its center inside the depth layers through a full turn', (_identifier, model) => {
    for (const yaw of [0, Math.PI / 2, Math.PI, Math.PI * 2]) {
      const markup = renderHomeArtwork(model, 0.35, yaw)
      expect(markup).not.toMatch(/NaN|Infinity|<script/)
      const document = new DOMParser().parseFromString(`<svg>${markup}</svg>`, 'text/html')
      expect([...document.querySelectorAll('[data-depth-layer]')].map(layer => layer.getAttribute('data-depth-layer'))).toEqual(['back', 'core', 'front'])
      expect(document.querySelector('[data-world-center]')?.getAttribute('data-world-center')).toBe('0 0 0')
      expect(document.querySelectorAll('mask')).toHaveLength(1)
      expect(document.querySelectorAll('clipPath')).toHaveLength(1)
      expect(document.querySelector('[data-depth-layer="back"]')?.children).toHaveLength(model.nodes.length)
      expect(document.querySelectorAll('svg>rect,svg>circle')).toHaveLength(0)
    }
    expect(renderHomeArtwork(model, 0, 0)).not.toBe(renderHomeArtwork(model, 0, 0.6))
    expect(renderHomeArtwork(model, 0, 0)).toBe(renderHomeArtwork(model, 0, Math.PI * 2))
    expect(renderHomeArtwork(model, 0, 0)).toBe(renderHomeArtwork(model, Math.PI * 2, 0))
    expect(renderHomeArtwork(model, Math.PI * 3.5, 0.3)).not.toMatch(/NaN|Infinity/)
  })
})
