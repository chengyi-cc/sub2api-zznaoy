import { createHomeArtworkRenderer, type HomeArtworkModel } from './homeArtworkRenderer'

interface InteractiveArtwork {
  id: string
  name: string
  model?: HomeArtworkModel
}

interface ArtworkGesture {
  pointerId: number
  pointerType: string
  startX: number
  startY: number
  startPitch: number
  startYaw: number
  lastTime: number
  velocityPitch: number
  velocityYaw: number
  moved: boolean
}

const artworkControllers = new WeakMap<HTMLElement, () => void>()

export function initializeHomeArtworkInteractions(root: ParentNode, artworks: readonly InteractiveArtwork[]): () => void {
  const cleanups: Array<() => void> = []
  for (const slot of root.querySelectorAll<HTMLElement>('[data-hc-artwork="random"]')) {
    artworkControllers.get(slot)?.()
    const artwork = artworks.find(item => item.id === slot.dataset.hcCurrentArt)
    const model = artwork?.model
    const svg = slot.querySelector('svg')
    const stage = slot.closest('.hf-stage')
    const host = slot.ownerDocument.defaultView
    if (!model || !svg || !host) continue
    const scheduler = window
    const resetButton = stage?.querySelector<HTMLButtonElement>('[data-hc-art-reset]')
    const pause = stage?.querySelector<HTMLInputElement>('.hf-motion-pause')
    const reducedMotion = host.matchMedia?.('(prefers-reduced-motion: reduce)')
    const originalMarkup = svg.innerHTML
    const render = createHomeArtworkRenderer(svg, model)
    const originalAttributes = ['role', 'tabindex', 'aria-label'].map(name => [name, slot.getAttribute(name)] as const)
    const listeners: Array<() => void> = []
    let pitch = 0
    let yaw = 0
    let gesture: ArtworkGesture | null = null
    let frame: number | null = null
    let animation: ((now: number) => boolean) | null = null
    let dirty = false
    let disposed = false
    let suppressClick = false
    let autoTimer: number | null = null
    let inViewport = true
    const radians = Math.PI / 180
    const clamp = (value: number, limit: number) => Math.max(-limit, Math.min(limit, value))
    const automaticAllowed = () => !reducedMotion?.matches && !pause?.checked
    const automaticVisible = () => automaticAllowed() && inViewport && !slot.ownerDocument.hidden && !scheduler.document.hidden
    const requestFrame = (callback: FrameRequestCallback) => scheduler.requestAnimationFrame ? scheduler.requestAnimationFrame(callback) : scheduler.setTimeout(() => callback(scheduler.performance.now()), 16)
    const cancelFrame = (identifier: number) => scheduler.cancelAnimationFrame ? scheduler.cancelAnimationFrame(identifier) : scheduler.clearTimeout(identifier)
    const listen = (target: EventTarget, event: string, callback: EventListener) => {
      target.addEventListener(event, callback)
      listeners.push(() => target.removeEventListener(event, callback))
    }
    const setActive = (active: boolean) => {
      if (active) slot.dataset.hcActiveInteraction = 'true'
      else delete slot.dataset.hcActiveInteraction
    }
    const draw = () => {
      if (disposed || !slot.contains(svg)) return
      render(pitch, yaw)
      slot.dataset.hcPitch = (pitch / radians).toFixed(1)
      slot.dataset.hcYaw = (yaw / radians).toFixed(1)
      dirty = false
    }
    const tick = () => {
      frame = null
      if (disposed) return
      const now = scheduler.performance.now()
      const wasAnimating = animation !== null
      if (animation) animation = animation(now) ? animation : null
      if (dirty) draw()
      if (animation || dirty) frame = requestFrame(tick)
      else if (!gesture) {
        setActive(false)
        if (wasAnimating) queueAutoRotation()
      }
    }
    const requestDraw = () => {
      dirty = true
      if (frame === null) frame = requestFrame(tick)
    }
    const stopAnimation = () => {
      if (autoTimer !== null) scheduler.clearTimeout(autoTimer)
      autoTimer = null
      animation = null
      if (frame !== null) cancelFrame(frame)
      frame = null
      if (dirty) draw()
      if (!gesture) setActive(false)
      slot.dataset.hcAutoPhase = automaticAllowed() ? 'waiting' : 'paused'
    }
    const startAutoRotation = () => {
      if (autoTimer !== null) scheduler.clearTimeout(autoTimer)
      autoTimer = null
      if (disposed || !automaticVisible() || gesture || animation) return
      slot.dataset.hcAutoPhase = 'spinning'
      let previousTime = scheduler.performance.now()
      animation = now => {
        const delta = Math.min(50, Math.max(0, now - previousTime))
        previousTime = now
        yaw += delta * 9 * radians / 1000
        pitch += delta * 1.8 * radians / 1000
        dirty = true
        return true
      }
      if (frame === null) frame = requestFrame(tick)
    }
    const queueAutoRotation = (immediate = false) => {
      if (autoTimer !== null) scheduler.clearTimeout(autoTimer)
      autoTimer = null
      if (disposed) return
      if (!automaticAllowed()) { slot.dataset.hcAutoPhase = 'paused'; return }
      if (!automaticVisible()) { slot.dataset.hcAutoPhase = 'suspended'; return }
      if (gesture || animation) return
      if (immediate) { startAutoRotation(); return }
      slot.dataset.hcAutoPhase = 'waiting'
      autoTimer = scheduler.setTimeout(startAutoRotation, 2400)
    }
    const releaseGesture = () => {
      const previous = gesture
      gesture = null
      delete slot.dataset.hcDragging
      if (previous) {
        try {
          if (slot.hasPointerCapture?.(previous.pointerId)) slot.releasePointerCapture(previous.pointerId)
        } catch {
          setActive(false)
        }
      }
      return previous
    }
    const reset = () => {
      releaseGesture()
      stopAnimation()
      pitch = 0
      yaw = 0
      dirty = false
      svg.innerHTML = originalMarkup
      slot.dataset.hcPitch = '0.0'
      slot.dataset.hcYaw = '0.0'
      setActive(false)
      queueAutoRotation(true)
    }
    const turn = (nextPitch: number, nextYaw: number) => {
      releaseGesture()
      stopAnimation()
      const startPitch = pitch
      const startYaw = yaw
      const endPitch = nextPitch
      if (!automaticAllowed()) {
        pitch = endPitch
        yaw = nextYaw
        draw()
        return
      }
      const start = scheduler.performance.now()
      setActive(true)
      animation = now => {
        const progress = Math.min(1, (now - start) / 360)
        const eased = 1 - (1 - progress) ** 3
        pitch = startPitch + (endPitch - startPitch) * eased
        yaw = startYaw + (nextYaw - startYaw) * eased
        dirty = true
        return progress < 1
      }
      requestDraw()
    }
    const finishGesture = (allowInertia: boolean) => {
      const previous = releaseGesture()
      if (!previous) return
      suppressClick = previous.moved
      stopAnimation()
      const recentlyMoved = scheduler.performance.now() - previous.lastTime < 90
      if (!allowInertia || !previous.moved || !recentlyMoved || !automaticAllowed()) {
        queueAutoRotation()
        return
      }
      let velocityPitch = previous.velocityPitch
      let velocityYaw = previous.velocityYaw
      let previousTime = scheduler.performance.now()
      const start = previousTime
      setActive(true)
      animation = now => {
        const delta = Math.min(40, now - previousTime)
        previousTime = now
        const decay = Math.exp(-delta / 150)
        velocityPitch *= decay
        velocityYaw *= decay
        pitch += velocityPitch * delta
        yaw += velocityYaw * delta
        dirty = true
        return now - start < 700 && Math.abs(velocityPitch) + Math.abs(velocityYaw) > 0.00006
      }
      requestDraw()
    }
    slot.dataset.hcInteractive = 'true'
    slot.tabIndex = 0
    slot.setAttribute('role', 'button')
    slot.setAttribute('aria-label', `${artwork.name}，点击转动，按住拖拽旋转；方向键调整，Home 键复位`)
    listen(slot, 'pointerdown', event => {
      const pointer = event as PointerEvent
      if (gesture || pointer.isPrimary === false || pointer.button !== 0) return
      stopAnimation()
      suppressClick = false
      gesture = { pointerId: pointer.pointerId, pointerType: pointer.pointerType, startX: pointer.clientX, startY: pointer.clientY, startPitch: pitch, startYaw: yaw, lastTime: scheduler.performance.now(), velocityPitch: 0, velocityYaw: 0, moved: false }
      setActive(true)
      if (pointer.pointerType !== 'touch') slot.focus({ preventScroll: true })
      try { slot.setPointerCapture?.(pointer.pointerId) } catch { return }
    })
    listen(host, 'pointermove', event => {
      const pointer = event as PointerEvent
      if (!gesture || pointer.pointerId !== gesture.pointerId) return
      const horizontal = pointer.clientX - gesture.startX
      const vertical = pointer.clientY - gesture.startY
      const threshold = gesture.pointerType === 'touch' ? 8 : 4
      if (!gesture.moved && Math.hypot(horizontal, vertical) < threshold) return
      if (!gesture.moved && gesture.pointerType === 'touch' && Math.abs(vertical) > Math.abs(horizontal)) {
        finishGesture(false)
        return
      }
      gesture.moved = true
      slot.dataset.hcDragging = 'true'
      const width = Math.max(160, slot.getBoundingClientRect().width || 540)
      const nextYaw = gesture.startYaw + horizontal / width * 240 * radians
      const nextPitch = gesture.startPitch - vertical / width * 180 * radians
      const now = scheduler.performance.now()
      const delta = Math.max(8, now - gesture.lastTime)
      gesture.velocityPitch = clamp((nextPitch - pitch) / delta, 0.004)
      gesture.velocityYaw = clamp((nextYaw - yaw) / delta, 0.005)
      gesture.lastTime = now
      pitch = nextPitch
      yaw = nextYaw
      requestDraw()
    })
    listen(host, 'pointerup', event => {
      if ((event as PointerEvent).pointerId === gesture?.pointerId) finishGesture(true)
    })
    const cancelGesture: EventListener = event => {
      if ((event as PointerEvent).pointerId === gesture?.pointerId) finishGesture(false)
    }
    listen(host, 'pointercancel', cancelGesture)
    listen(slot, 'lostpointercapture', cancelGesture)
    listen(slot, 'click', event => {
      if (suppressClick) { suppressClick = false; return }
      event.preventDefault()
      turn(pitch, yaw + 32 * radians)
    })
    listen(slot, 'keydown', event => {
      const keyboard = event as KeyboardEvent
      if (keyboard.altKey || keyboard.ctrlKey || keyboard.metaKey) return
      const step = (keyboard.shiftKey ? 30 : 15) * radians
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', ' ', 'Home', 'Escape'].includes(keyboard.key)) return
      keyboard.preventDefault()
      if (keyboard.key === 'Home' || keyboard.key === 'Escape') { reset(); return }
      if (keyboard.key === 'ArrowLeft') turn(pitch, yaw - step)
      else if (keyboard.key === 'ArrowUp') turn(pitch + step, yaw)
      else if (keyboard.key === 'ArrowDown') turn(pitch - step, yaw)
      else turn(pitch, yaw + step)
    })
    if (resetButton) listen(resetButton, 'click', reset)
    const updateMotionPreference = () => {
      stopAnimation()
      queueAutoRotation(true)
    }
    if (pause) listen(pause, 'change', updateMotionPreference)
    if (reducedMotion && typeof reducedMotion.addEventListener === 'function') listen(reducedMotion, 'change', updateMotionPreference)
    listen(host, 'blur', () => { if (gesture) finishGesture(false) })
    const updateVisibility = () => {
      if (slot.ownerDocument.hidden || scheduler.document.hidden) { finishGesture(false); stopAnimation() }
      queueAutoRotation(true)
    }
    listen(slot.ownerDocument, 'visibilitychange', updateVisibility)
    if (scheduler.document !== slot.ownerDocument) listen(scheduler.document, 'visibilitychange', updateVisibility)
    const observer = typeof host.IntersectionObserver === 'function' ? new host.IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.target !== slot || inViewport === entry.isIntersecting) continue
        inViewport = entry.isIntersecting
        if (!inViewport) { finishGesture(false); stopAnimation() }
        queueAutoRotation(true)
      }
    }, { threshold: 0 }) : null
    observer?.observe(slot)
    const cleanup = () => {
      if (disposed) return
      releaseGesture()
      stopAnimation()
      disposed = true
      observer?.disconnect()
      if (slot.contains(svg)) svg.innerHTML = originalMarkup
      listeners.forEach(remove => remove())
      delete slot.dataset.hcInteractive
      delete slot.dataset.hcActiveInteraction
      delete slot.dataset.hcPitch
      delete slot.dataset.hcYaw
      delete slot.dataset.hcAutoPhase
      for (const [name, value] of originalAttributes) {
        if (value === null) slot.removeAttribute(name)
        else slot.setAttribute(name, value)
      }
      artworkControllers.delete(slot)
    }
    artworkControllers.set(slot, cleanup)
    cleanups.push(cleanup)
    queueAutoRotation(true)
  }
  return () => cleanups.forEach(cleanup => cleanup())
}
