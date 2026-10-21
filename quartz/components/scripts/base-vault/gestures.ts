import type { VaultCamera } from './camera'
const DRAG_PX = 5
const PULL_ALIGN = Math.cos((24 * Math.PI) / 180)

interface GestureOptions {
  viewport: HTMLElement
  dialog: HTMLDialogElement
  signal: AbortSignal
  camera: VaultCamera
  cans: Map<number, SVGGElement>
  pulled: Map<SVGGElement, SVGGElement>
  pullOf: (can: SVGGElement) => [number, number]
  pullProgress: (can: SVGGElement) => number
  current: () => number
  release: () => void
  mountClone: (index: number) => SVGGElement | undefined
  placePull: (index: number, progress: number) => void
  finishPull: (index: number, progress: number, speed: number) => void
  openCard: (index: number) => void
  cancelPull: () => void
}

export function bindVaultGestures(options: GestureOptions) {
  const {
    viewport,
    dialog,
    signal,
    camera,
    cans,
    pulled,
    pullOf,
    pullProgress,
    current,
    release,
    mountClone,
    placePull,
    finishPull,
    openCard,
    cancelPull,
  } = options
  type PendingGesture = {
    mode: 'pending'
    id: number
    x: number
    y: number
    left: number
    top: number
    index: number
    px: number
    py: number
  }
  type PanGesture = {
    mode: 'pan'
    id: number
    x: number
    y: number
    left: number
    top: number
    moved: boolean
  }
  type PullGesture = {
    mode: 'pull'
    id: number
    x: number
    y: number
    index: number
    px: number
    py: number
    start: number
    t: number
    /** Progress per ms, smoothed over the last moves. */
    speed: number
    at: number
  }
  let gesture: PendingGesture | PanGesture | PullGesture | undefined
  let dragged = false
  let draggedResetTimer: ReturnType<typeof setTimeout> | undefined
  const clearDragged = () => {
    dragged = false
    if (draggedResetTimer !== undefined) {
      clearTimeout(draggedResetTimer)
      draggedResetTimer = undefined
    }
  }
  const markDragged = () => {
    dragged = true
    if (draggedResetTimer !== undefined) clearTimeout(draggedResetTimer)
    draggedResetTimer = setTimeout(() => {
      draggedResetTimer = undefined
      dragged = false
    }, 0)
  }
  const capturePointer = (pointerId: number) => {
    try {
      viewport.setPointerCapture?.(pointerId)
    } catch {}
  }
  const releasePointer = (pointerId: number) => {
    try {
      if (viewport.hasPointerCapture?.(pointerId)) viewport.releasePointerCapture(pointerId)
    } catch {}
  }

  // A press does not focus the stage: a focus change wakes the site's :has(:focus-visible) rules and
  // restyles every element on the page. Tab still focuses it for the keys.
  viewport.addEventListener(
    'mousedown',
    event => {
      if (event.button === 0) event.preventDefault()
    },
    { signal },
  )
  viewport.addEventListener(
    'pointerdown',
    event => {
      clearDragged()
      if (event.button !== 0 || event.isPrimary === false || dialog.open) return
      camera.commitZoom()
      const node = camera.canAt(event)
      const index = Number(node?.dataset.index ?? -1)
      const can = cans.get(index)
      const [px, py] = can ? pullOf(can) : [0, 0]
      gesture = {
        mode: 'pending',
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        left: viewport.scrollLeft,
        top: viewport.scrollTop,
        index: can && px * px + py * py > 1 ? index : -1,
        px,
        py,
      }
    },
    { signal },
  )
  viewport.addEventListener(
    'pointermove',
    event => {
      if (!gesture || event.pointerId !== gesture.id) return
      if (event.buttons === 0) {
        endGesture(event)
        return
      }
      const dx = event.clientX - gesture.x
      const dy = event.clientY - gesture.y
      const dist = Math.hypot(dx, dy)
      if (gesture.mode === 'pending') {
        if (dist < DRAG_PX) return
        const len = Math.hypot(gesture.px, gesture.py)
        const dot =
          gesture.index >= 0 && len > 0 ? (dx * gesture.px + dy * gesture.py) / (len * dist) : 0
        const can = gesture.index >= 0 ? cans.get(gesture.index) : undefined
        const existing = can ? pulled.get(can) : undefined
        const isOut = Boolean(existing && existing.dataset.out !== undefined)
        const live = existing ? pullProgress(existing) : 0
        const canPullOut = !isOut && dot >= PULL_ALIGN
        const canPushIn = isOut && dot <= -PULL_ALIGN
        if (can && (canPullOut || canPushIn)) {
          const startT = canPushIn ? Math.max(0.5, live) : live
          if (current() >= 0 && current() !== gesture.index) release()
          cancelPull()
          const clone = existing ?? mountClone(gesture.index)
          if (!clone) {
            gesture = {
              mode: 'pan',
              id: gesture.id,
              x: gesture.x,
              y: gesture.y,
              left: gesture.left,
              top: gesture.top,
              moved: false,
            }
          } else {
            placePull(gesture.index, startT)
            capturePointer(event.pointerId)
            camera.hold(true)
            gesture = {
              mode: 'pull',
              id: gesture.id,
              x: gesture.x,
              y: gesture.y,
              index: gesture.index,
              px: gesture.px,
              py: gesture.py,
              start: startT,
              t: startT,
              speed: 0,
              at: event.timeStamp,
            }
          }
        } else {
          gesture = {
            mode: 'pan',
            id: gesture.id,
            x: gesture.x,
            y: gesture.y,
            left: gesture.left,
            top: gesture.top,
            moved: false,
          }
        }
      }
      if (gesture.mode === 'pan') {
        const panDx = event.clientX - gesture.x
        const panDy = event.clientY - gesture.y
        if (!gesture.moved && Math.hypot(panDx, panDy) < DRAG_PX) return
        if (!gesture.moved) {
          gesture.moved = true
          capturePointer(event.pointerId)
          camera.hold(true)
        }
        camera.writeScroll(gesture.left - panDx, gesture.top - panDy)
        return
      }
      if (gesture.mode !== 'pull') return
      const k = camera.scale
      const along =
        ((event.clientX - gesture.x) / k) * gesture.px +
        ((event.clientY - gesture.y) / k) * gesture.py
      const t = Math.min(
        1,
        Math.max(0, gesture.start + along / (gesture.px * gesture.px + gesture.py * gesture.py)),
      )
      const elapsed = event.timeStamp - gesture.at
      if (elapsed > 0) gesture.speed = gesture.speed * 0.5 + ((t - gesture.t) / elapsed) * 0.5
      gesture.at = event.timeStamp
      gesture.t = t
      placePull(gesture.index, t)
    },
    { signal },
  )
  const endGesture = (event?: PointerEvent) => {
    const active = gesture
    if (!active) return
    if (event && event.pointerId !== undefined && event.pointerId !== active.id) return
    gesture = undefined
    camera.hold(false)
    releasePointer(active.id)
    if (active.mode === 'pan') {
      if (active.moved) markDragged()
      return
    }
    if (active.mode === 'pull') {
      markDragged()
      // A hand that stopped before letting go has no speed left to carry.
      const still = (event?.timeStamp ?? performance.now()) - active.at > 80
      finishPull(active.index, active.t, still ? 0 : active.speed)
    }
  }
  viewport.addEventListener('pointerup', endGesture, { signal })
  viewport.addEventListener('pointercancel', endGesture, { signal })
  viewport.addEventListener('lostpointercapture', endGesture, { signal })
  viewport.addEventListener(
    'click',
    event => {
      if (dragged) {
        clearDragged()
        return
      }
      const can = camera.canAt(event)
      if (can) openCard(Number(can.dataset.index))
      else release()
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      const pointer = gesture?.id
      gesture = undefined
      if (pointer !== undefined) releasePointer(pointer)
      clearDragged()
    },
    { once: true },
  )
  return {
    clearDragged,
    get active() {
      return gesture !== undefined
    },
  }
}
