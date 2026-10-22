import type { VaultCamera } from './vault-camera'

const DRAG_PX = 5
const PULL_ALIGN = Math.cos((24 * Math.PI) / 180)
const ZOOM = 1.25

export interface VaultGestureOptions {
  host: HTMLElement
  viewport: HTMLElement
  controls: HTMLElement
  signal: AbortSignal
  camera: VaultCamera
  cans: Map<number, SVGGElement>
  pulled: Map<SVGGElement, SVGGElement>
  pullOf: (can: SVGGElement) => [number, number]
  pullProgress: (clone: SVGGElement) => number
  current: () => number
  release: () => void
  mountClone: (index: number) => SVGGElement | undefined
  placePull: (index: number, progress: number) => void
  finishPull: (index: number, progress: number, speed: number) => void
  cancelPull: () => void
  onCanClick: (index: number, evt: MouseEvent) => void
  onPulledHover: (evt: PointerEvent, can: SVGGElement) => void
  onStep: (delta: number) => void
  onSelect: (index: number) => void
  onOpenNote: (index: number, evt: KeyboardEvent) => void
  onToggleFullscreen: () => void
}

export function bindVaultGestures(options: VaultGestureOptions) {
  const {
    host,
    viewport,
    controls,
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
    cancelPull,
    onCanClick,
    onPulledHover,
    onStep,
    onSelect,
    onOpenNote,
    onToggleFullscreen,
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
    speed: number
    at: number
  }
  let gesture: PendingGesture | PanGesture | PullGesture | undefined
  let dragged = false
  let draggedResetTimer: number | null = null
  const win = viewport.ownerDocument.defaultView ?? window
  const clearDragged = () => {
    dragged = false
    if (draggedResetTimer !== null) {
      win.clearTimeout(draggedResetTimer)
      draggedResetTimer = null
    }
  }
  const markDragged = () => {
    dragged = true
    if (draggedResetTimer !== null) win.clearTimeout(draggedResetTimer)
    draggedResetTimer = win.setTimeout(() => {
      draggedResetTimer = null
      dragged = false
    }, 0)
  }
  const capturePointer = (pointerId: number) => {
    try {
      viewport.setPointerCapture(pointerId)
    } catch {}
  }
  const releasePointer = (pointerId: number) => {
    try {
      if (viewport.hasPointerCapture(pointerId)) viewport.releasePointerCapture(pointerId)
    } catch {}
  }

  // A press does not focus the rack: a focus change restyles every :focus-within and
  // :has(:focus-visible) ancestor. Tab still focuses it for the keys.
  viewport.addEventListener(
    'mousedown',
    evt => {
      if (evt.button === 0) evt.preventDefault()
    },
    { signal },
  )
  viewport.addEventListener(
    'pointerdown',
    evt => {
      clearDragged()
      if (evt.button !== 0 || evt.isPrimary === false) return
      camera.commitZoom()
      camera.cancelGlide()
      const node = camera.canAt(evt)
      const index = Number(node?.dataset.index ?? -1)
      const can = cans.get(index)
      const [px, py] = can ? pullOf(can) : [0, 0]
      gesture = {
        mode: 'pending',
        id: evt.pointerId,
        x: evt.clientX,
        y: evt.clientY,
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
    evt => {
      if (gesture && evt.pointerId === gesture.id) {
        if (evt.buttons === 0) {
          endGesture(evt)
          return
        }
        const dx = evt.clientX - gesture.x
        const dy = evt.clientY - gesture.y
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
              capturePointer(evt.pointerId)
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
                at: evt.timeStamp,
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
          const panDx = evt.clientX - gesture.x
          const panDy = evt.clientY - gesture.y
          if (!gesture.moved && Math.hypot(panDx, panDy) < DRAG_PX) return
          if (!gesture.moved) {
            gesture.moved = true
            capturePointer(evt.pointerId)
            camera.hold(true)
          }
          camera.writeScroll(gesture.left - panDx, gesture.top - panDy)
          return
        }
        if (gesture.mode !== 'pull') return
        const k = camera.scale
        const along =
          ((evt.clientX - gesture.x) / k) * gesture.px +
          ((evt.clientY - gesture.y) / k) * gesture.py
        const t = Math.min(
          1,
          Math.max(0, gesture.start + along / (gesture.px * gesture.px + gesture.py * gesture.py)),
        )
        const elapsed = evt.timeStamp - gesture.at
        if (elapsed > 0) gesture.speed = gesture.speed * 0.5 + ((t - gesture.t) / elapsed) * 0.5
        gesture.at = evt.timeStamp
        gesture.t = t
        placePull(gesture.index, t)
        return
      }
      // A pointer that moves onto the pulled can opens its page preview, once per lift.
      if (evt.pointerType === 'touch' || evt.buttons !== 0 || current() < 0) return
      const can = (evt.target as Element).closest<SVGGElement>('.cn-can.cn-pulled')
      if (can) onPulledHover(evt, can)
    },
    { signal },
  )
  const endGesture = (evt?: PointerEvent) => {
    const active = gesture
    if (!active) return
    if (evt && evt.pointerId !== undefined && evt.pointerId !== active.id) return
    gesture = undefined
    camera.hold(false)
    releasePointer(active.id)
    if (active.mode === 'pan') {
      if (active.moved) markDragged()
      return
    }
    if (active.mode === 'pull') {
      markDragged()
      const still = (evt?.timeStamp ?? performance.now()) - active.at > 80
      finishPull(active.index, active.t, still ? 0 : active.speed)
    }
  }
  viewport.addEventListener('pointerup', endGesture, { signal })
  viewport.addEventListener('pointercancel', endGesture, { signal })
  viewport.addEventListener('lostpointercapture', endGesture, { signal })
  viewport.addEventListener(
    'click',
    evt => {
      if (dragged) {
        clearDragged()
        return
      }
      const can = camera.canAt(evt)
      if (!can) return release()
      onCanClick(Number(can.dataset.index), evt)
    },
    { signal },
  )
  viewport.addEventListener(
    'keydown',
    evt => {
      clearDragged()
      const cur = current()
      if (evt.altKey || evt.ctrlKey || evt.metaKey) {
        if (evt.key === 'Enter' && cur >= 0) onOpenNote(cur, evt)
        return
      }
      const key = evt.key
      if (key === 'ArrowRight' || key === 'ArrowDown') onStep(1)
      else if (key === 'ArrowLeft' || key === 'ArrowUp') onStep(-1)
      else if (key === 'Enter' && cur >= 0) onOpenNote(cur, evt)
      else if (key === 'Enter') onSelect(0)
      else if (key === ' ') onSelect(cur < 0 ? 0 : cur)
      else if (key === '+' || key === '=') camera.setScale(camera.scale * ZOOM)
      else if (key === '-') camera.setScale(camera.scale / ZOOM)
      else if (key === '0') camera.fit()
      else return
      evt.preventDefault()
    },
    { signal },
  )
  controls.addEventListener(
    'click',
    evt => {
      const action = (evt.target as Element).closest<HTMLElement>('[data-camera]')?.dataset.camera
      if (action === 'fit') camera.fit()
      else if (action === 'in') camera.setScale(camera.scale * ZOOM)
      else if (action === 'out') camera.setScale(camera.scale / ZOOM)
      else if (action === 'fullscreen') onToggleFullscreen()
    },
    { signal },
  )
  host.addEventListener(
    'click',
    evt => {
      const row = (evt.target as Element).closest<HTMLElement>('.vault-entry')
      if (row) onSelect(Number(row.dataset.index))
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
