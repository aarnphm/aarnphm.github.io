// Camera gestures for the latent bench: the 3D scatter of document vectors. The bench owns the
// points and drawing; this module owns how pointers, pinches and the wheel move its view.

/** The view fields a gesture changes, in radians, a scale factor and css px. */
export type BenchView = {
  yaw: number
  pitch: number
  zoom: number
  panX: number
  panY: number
  w: number
  h: number
}

export const BENCH_VIEW = { yaw: 0.6, pitch: 0.5, zoom: 1, panX: 0, panY: 0 }

export type BenchHost = {
  view(): BenchView | null
  /** true while a point is under the pointer */
  hovering(): boolean
  hover(x: number, y: number): void
  redraw(): void
  /** follow the hovered point */
  open(): void
}

/**
 * Mouse: drag turns the space; shift-drag, right-drag or middle-drag moves it. Touch: one finger
 * turns, two fingers move and pinch. The wheel zooms about the pointer. A drag never counts as a
 * click, and a double click on empty space restores the opening view.
 */
export function bindBenchGestures(
  canvas: HTMLCanvasElement,
  signal: AbortSignal,
  host: BenchHost,
): void {
  const pointers = new Map<number, { x: number; y: number }>()
  let mode: 'rotate' | 'pan' | null = null
  let travel = 0
  const local = (x: number, y: number) => {
    const rect = canvas.getBoundingClientRect()
    return [x - rect.left, y - rect.top] as const
  }
  const pair = () => {
    const [a, b] = [...pointers.values()]
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) }
  }
  let pinch = { x: 0, y: 0, d: 0 }
  canvas.addEventListener(
    'pointerdown',
    event => {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
      canvas.setPointerCapture(event.pointerId)
      travel = 0
      if (pointers.size === 2) pinch = pair()
      mode =
        pointers.size > 1 || event.shiftKey || event.button === 1 || event.button === 2
          ? 'pan'
          : 'rotate'
      canvas.dataset.drag = mode
    },
    { signal },
  )
  canvas.addEventListener(
    'pointermove',
    event => {
      const bench = host.view()
      if (!bench) return
      const last = pointers.get(event.pointerId)
      if (!last || !mode) {
        if (event.pointerType === 'touch') return
        host.hover(...local(event.clientX, event.clientY))
        return
      }
      const dx = event.clientX - last.x
      const dy = event.clientY - last.y
      last.x = event.clientX
      last.y = event.clientY
      travel += Math.abs(dx) + Math.abs(dy)
      if (pointers.size === 2) {
        const next = pair()
        const [cx, cy] = local(next.x, next.y)
        panBench(bench, next.x - pinch.x, next.y - pinch.y)
        if (pinch.d > 0) zoomBench(bench, next.d / pinch.d, cx, cy)
        pinch = next
      } else if (mode === 'pan') {
        panBench(bench, dx, dy)
      } else {
        bench.yaw += dx * 0.01
        bench.pitch = Math.max(0.05, Math.min(1.2, bench.pitch + dy * 0.005))
      }
      host.redraw()
    },
    { signal },
  )
  const release = (event: PointerEvent) => {
    pointers.delete(event.pointerId)
    if (pointers.size === 0) {
      mode = null
      delete canvas.dataset.drag
    } else if (pointers.size === 1) {
      // lifting one finger of a pinch keeps moving the space instead of snapping into a turn
      mode = 'pan'
    }
  }
  canvas.addEventListener('pointerup', release, { signal })
  canvas.addEventListener('pointercancel', release, { signal })
  canvas.addEventListener('contextmenu', event => event.preventDefault(), { signal })
  canvas.addEventListener(
    'wheel',
    event => {
      const bench = host.view()
      if (!bench) return
      event.preventDefault()
      // trackpad pinches arrive as ctrl-wheel with small deltas; scale by the delta for both
      const k = Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.002))
      zoomBench(bench, k, ...local(event.clientX, event.clientY))
      host.redraw()
    },
    { signal, passive: false },
  )
  canvas.addEventListener(
    'click',
    () => {
      if (travel <= 4) host.open()
    },
    { signal },
  )
  canvas.addEventListener(
    'dblclick',
    () => {
      const bench = host.view()
      if (!bench || host.hovering()) return
      Object.assign(bench, BENCH_VIEW)
      host.redraw()
    },
    { signal },
  )
}

// Keeps the origin within a canvas width or so of the centre at any zoom, so the cloud stays findable.
function panBench(bench: BenchView, dx: number, dy: number): void {
  const room = Math.max(bench.w, bench.h) * Math.max(1, bench.zoom)
  bench.panX = Math.max(-room, Math.min(room, bench.panX + dx))
  bench.panY = Math.max(-room, Math.min(room, bench.panY + dy))
}

// Scales about (px, py) in css px, so the point under the pointer stays put.
function zoomBench(bench: BenchView, k: number, px: number, py: number): void {
  const zoom = Math.max(0.4, Math.min(12, bench.zoom * k))
  const f = zoom / bench.zoom
  bench.zoom = zoom
  const ox = px - bench.w / 2
  const oy = py - bench.h / 2
  panBench(bench, ox - (ox - bench.panX) * f - bench.panX, oy - (oy - bench.panY) * f - bench.panY)
}
