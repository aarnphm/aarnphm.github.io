import type { Point } from './landscape'

export const trailOpeningDuration = 600
export const trailThemeDuration = 600
export const trailOpeningCell = 6

export function mountSceneReveal(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!view || !svg) throw new Error('The trail opening needs its surface')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let epoch: number | null = null
  let updated = 0
  let revealed = 0
  let camp = 1
  let width = 1
  let height = 1
  let unbuiltBottom: number | null = Infinity
  let frame = 0
  let fallbackEnabled = false
  let mask: {
    canvas: HTMLCanvasElement
    paint: CanvasRenderingContext2D
    pixels: ImageData
  } | null = null
  let painted = -1
  const clearMask = () => {
    svg.style.removeProperty('mask-image')
    svg.style.removeProperty('mask-size')
    svg.style.removeProperty('mask-repeat')
    painted = -1
  }
  const complete = () => {
    revealed = 1
    scene.dataset.trailOpening = 'complete'
    scene.dataset.trailOpeningProgress = '1'
    clearMask()
    return 1
  }
  const progress = (time: number, ceiling = 1, ready = true) => {
    if (scene.dataset.trailOpening === 'complete') return 1
    if (host.hidden || scene.dataset.campPositioned !== 'true') return 0
    // Generation and reveal share the camp's deadline. Coverage holds unfinished rows;
    // bounded catch-up keeps a late upload from turning the remaining sweep into a flash.
    if (epoch === null) {
      epoch = time
      updated = time
    }
    const elapsed = Math.max(0, (time - epoch) / trailOpeningDuration)
    // Texture coverage cannot see the next detached forest batch before it is published.
    if (unbuiltBottom !== null) {
      const origin = Math.min(camp, height)
      ceiling = Math.min(ceiling, 0.82 * Math.max(0, (origin - unbuiltBottom) / origin))
    }
    const advance = Math.min(32, Math.max(0, time - updated)) / (trailOpeningDuration / 2)
    const value = reduced.matches
      ? ready
        ? 1
        : 0
      : Math.max(revealed, Math.min(ceiling, elapsed, revealed + advance))
    updated = time
    revealed = value
    if (value === 1 && ready) return complete()
    if (value > 0) scene.dataset.trailOpening = 'revealing'
    scene.dataset.trailOpeningProgress = value.toFixed(3)
    return value
  }
  const fallback = () => {
    frame = 0
    if (signal.aborted || host.hidden || document.hidden) return
    if (scene.dataset.trailRenderer === 'webgl' && scene.dataset.trailOpeningLive !== 'true') {
      clearMask()
      return
    }
    const value = progress(view.performance.now(), 1, scene.dataset.trailRendered === 'true')
    if (value >= 1) return
    if (scene.dataset.campPositioned === 'true') {
      const step = Math.floor(value * 120)
      const columns = Math.max(1, Math.ceil(width / trailOpeningCell))
      const rows = Math.max(1, Math.ceil(height / trailOpeningCell))
      if (!mask || mask.canvas.width !== columns || mask.canvas.height !== rows) {
        const canvas = document.createElement('canvas')
        canvas.width = columns
        canvas.height = rows
        const paint = canvas.getContext('2d', { willReadFrequently: true })
        if (!paint) throw new Error('The trail opening needs its pixel mask')
        mask = { canvas, paint, pixels: paint.createImageData(columns, rows) }
        painted = -1
      }
      if (painted !== step) {
        painted = step
        const origin = Math.min(camp, height)
        for (let y = 0; y < rows; y++) {
          const distance = Math.max(0, (origin - y * trailOpeningCell) / origin)
          for (let x = 0; x < columns; x++) {
            const noise = (x * 0.754877666 + y * 0.569840296) % 1
            const offset = (y * columns + x) * 4
            mask.pixels.data[offset] = 255
            mask.pixels.data[offset + 1] = 255
            mask.pixels.data[offset + 2] = 255
            mask.pixels.data[offset + 3] = 0.82 * distance + 0.18 * noise <= value ? 255 : 0
          }
        }
        mask.paint.putImageData(mask.pixels, 0, 0)
        svg.style.maskImage = `url("${mask.canvas.toDataURL()}")`
        svg.style.maskSize = '100% 100%'
        svg.style.maskRepeat = 'no-repeat'
      }
    }
    frame = view.requestAnimationFrame(fallback)
  }
  const schedule = () => {
    if (fallbackEnabled && !frame && !host.hidden && !document.hidden)
      frame = view.requestAnimationFrame(fallback)
  }
  const reset = () => {
    view.cancelAnimationFrame(frame)
    frame = 0
    epoch = null
    revealed = 0
    unbuiltBottom = Infinity
    clearMask()
    scene.dataset.trailOpening = 'pending'
    scene.dataset.trailOpeningProgress = '0'
  }
  reduced.addEventListener(
    'change',
    () => {
      if (reduced.matches && epoch !== null && scene.dataset.trailRendered === 'true') complete()
    },
    { signal },
  )
  host.addEventListener('trail:ready', schedule, { signal })
  document.addEventListener('visibilitychange', schedule, { signal })
  signal.addEventListener('abort', reset, { once: true })
  reset()
  return {
    progress,
    value: () => revealed,
    reset,
    generatedThrough(bottom: number | null) {
      unbuiltBottom = bottom
    },
    origin: () => camp,
    update(point: Point, size: { width: number; height: number }) {
      camp = Math.max(1, point.y)
      width = size.width
      height = size.height
      scene.dataset.trailOpeningOrigin = String(camp)
      if (scene.dataset.trailOpening === 'pending') progress(view.performance.now(), 1, false)
      schedule()
    },
    useFallback() {
      fallbackEnabled = true
      if (!frame) fallback()
    },
  }
}

export type SceneReveal = ReturnType<typeof mountSceneReveal>
