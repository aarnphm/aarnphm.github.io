import type { Point } from './landscape'

export const trailOpeningDuration = 600
export const trailThemeDuration = 600
export const trailOpeningCell = 6
export const trailRevealSoftness = 1.5
const maskPixel = 1

export function trailRevealThreshold(
  x: number,
  y: number,
  origin: number,
  cell = trailOpeningCell,
) {
  const height = Math.max(1, origin)
  const radius = cell / 2
  const q = (x / Math.sqrt(3) - y / 3) / radius
  const r = ((2 / 3) * y) / radius
  const s = -q - r
  let rq = Math.floor(q + 0.5)
  let rr = Math.floor(r + 0.5)
  const rs = Math.floor(s + 0.5)
  const dq = Math.abs(rq - q)
  const dr = Math.abs(rr - r)
  const ds = Math.abs(rs - s)
  if (dq > dr && dq > ds) rq = -rr - rs
  else if (dr > ds) rr = -rq - rs
  const column = Math.floor(rq + rr / 2) * 0.754877666
  const offset = (column - Math.floor(column)) * radius * 0.5
  const distance = Math.max(0, height - rr * radius * 1.5)
  return (distance + offset) / (height + cell)
}

export function trailRevealLimit(bottom: number, origin: number, cell = trailOpeningCell) {
  return Math.max(0, (origin - bottom - cell * 2) / (Math.max(1, origin) + cell))
}

export function createTrailRevealMask(
  document: Document,
  width: number,
  height: number,
  origin: number,
  offset: Point = { x: 0, y: 0 },
) {
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.ceil(width / maskPixel))
  const band = trailOpeningCell * 2
  canvas.height = Math.ceil((band * 2) / maskPixel)
  const paint = canvas.getContext('2d', { willReadFrequently: true })
  if (!paint) return null
  const pixels = paint.createImageData(canvas.width, canvas.height)
  pixels.data.fill(255)
  const softness = trailRevealSoftness / (Math.max(1, origin) + trailOpeningCell)
  return {
    width,
    height,
    origin,
    draw(progress: number, inverse = false) {
      if (progress <= 0 || progress >= 1) {
        const color = progress >= 1 !== inverse ? '#000' : 'transparent'
        return { image: `linear-gradient(${color}, ${color})`, size: '100% 100%', position: '0 0' }
      }
      const front = origin - progress * (Math.max(1, origin) + trailOpeningCell)
      const top = Math.floor((front - band - offset.y) / maskPixel) * maskPixel
      // Only the moving edge needs a bitmap; a solid gradient fills the completed scenery.
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const threshold = trailRevealThreshold(
            (x + 0.5) * maskPixel + offset.x,
            top + (y + 0.5) * maskPixel + offset.y,
            origin,
          )
          const blend = Math.min(1, Math.max(0, (progress - threshold) / (softness * 2) + 0.5))
          const alpha = blend * blend * (3 - 2 * blend)
          pixels.data[(y * canvas.width + x) * 4 + 3] = Math.round(
            (inverse ? 1 - alpha : alpha) * 255,
          )
        }
      }
      paint.putImageData(pixels, 0, 0)
      const bottom = top + canvas.height * maskPixel
      const fill = inverse
        ? `linear-gradient(to bottom, #000 ${top}px, transparent ${top}px)`
        : `linear-gradient(to bottom, transparent ${bottom}px, #000 ${bottom}px)`
      return {
        image: `url("${canvas.toDataURL()}"), ${fill}`,
        size: `${canvas.width * maskPixel}px ${canvas.height * maskPixel}px, 100% 100%`,
        position: `0 ${top}px, 0 0`,
      }
    },
  }
}

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
  let mask: ReturnType<typeof createTrailRevealMask> | null = null
  let painted = -1
  const clearMask = () => {
    svg.style.removeProperty('mask-image')
    svg.style.removeProperty('mask-size')
    svg.style.removeProperty('mask-repeat')
    svg.style.removeProperty('mask-position')
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
      ceiling = Math.min(ceiling, trailRevealLimit(unbuiltBottom, origin))
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
      const origin = Math.min(camp, height)
      if (!mask || mask.width !== width || mask.height !== height || mask.origin !== origin) {
        mask = createTrailRevealMask(document, width, height, origin)
        if (!mask) throw new Error('The trail opening needs its mask surface')
        painted = -1
      }
      if (painted !== step) {
        painted = step
        const drawn = mask.draw(value)
        svg.style.maskImage = drawn.image
        svg.style.maskSize = drawn.size
        svg.style.maskPosition = drawn.position
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
