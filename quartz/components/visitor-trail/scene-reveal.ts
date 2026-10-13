import type { Point } from './landscape'

export const trailOpeningDuration = 2400
export const trailOpeningCell = 6

export function mountSceneReveal(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!view || !svg) throw new Error('The trail opening needs its surface')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let epoch: number | null = null
  let camp = 1
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
    scene.dataset.trailOpening = 'complete'
    scene.dataset.trailOpeningProgress = '1'
    clearMask()
    return 1
  }
  const progress = (time: number) => {
    if (scene.dataset.trailOpening === 'complete') return 1
    if (host.hidden || scene.dataset.trailRendered !== 'true') return 0
    if (reduced.matches) return complete()
    if (epoch === null) {
      epoch = time
      scene.dataset.trailOpening = 'revealing'
    }
    const value = Math.min(1, Math.max(0, (time - epoch) / trailOpeningDuration))
    if (value === 1) return complete()
    scene.dataset.trailOpeningProgress = value.toFixed(3)
    return value
  }
  const fallback = () => {
    frame = 0
    if (signal.aborted || host.hidden || document.hidden) return
    if (scene.dataset.trailRenderer === 'webgl') {
      clearMask()
      return
    }
    const value = progress(view.performance.now())
    if (value >= 1) return
    if (scene.dataset.trailRendered === 'true') {
      const step = Math.floor(value * 120)
      const width = Math.max(1, Math.ceil(svg.clientWidth / trailOpeningCell))
      const height = Math.max(1, Math.ceil(svg.clientHeight / trailOpeningCell))
      if (!mask || mask.canvas.width !== width || mask.canvas.height !== height) {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const paint = canvas.getContext('2d')
        if (!paint) throw new Error('The trail opening needs its pixel mask')
        mask = { canvas, paint, pixels: paint.createImageData(width, height) }
        painted = -1
      }
      if (painted !== step) {
        painted = step
        const origin = Math.min(camp, svg.clientHeight)
        for (let y = 0; y < height; y++) {
          const distance = Math.max(0, (origin - y * trailOpeningCell) / origin)
          for (let x = 0; x < width; x++) {
            const noise = (x * 0.754877666 + y * 0.569840296) % 1
            const offset = (y * width + x) * 4
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
    clearMask()
    scene.dataset.trailOpening = 'pending'
    scene.dataset.trailOpeningProgress = '0'
  }
  reduced.addEventListener(
    'change',
    () => {
      if (reduced.matches && epoch !== null) complete()
    },
    { signal },
  )
  host.addEventListener('trail:ready', schedule, { signal })
  document.addEventListener('visibilitychange', schedule, { signal })
  signal.addEventListener('abort', reset, { once: true })
  reset()
  return {
    progress,
    reset,
    origin: () => camp,
    update(point: Point) {
      camp = Math.max(1, point.y)
      scene.dataset.trailOpeningOrigin = String(camp)
    },
    useFallback() {
      fallbackEnabled = true
      if (!frame) fallback()
    },
  }
}

export type SceneReveal = ReturnType<typeof mountSceneReveal>
