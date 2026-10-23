import type { Point } from './landscape'

export const trailThemeDuration = 600
export const trailOpeningCell = 6

export function sceneMotionReady(scene: HTMLElement) {
  return scene.dataset.trailOpening === 'complete' && scene.dataset.trailReveal === undefined
}

export function trailRevealProgress(elapsed: number) {
  const progress = Math.max(0, Math.min(1, elapsed))
  return progress * progress * (3 - 2 * progress)
}

export function trailRevealOrigin(document: Document): Point {
  const clock = document.querySelector('.trail-local-time')?.getBoundingClientRect()
  return clock && clock.width > 0
    ? { x: clock.left + clock.width / 2, y: clock.top + clock.height / 2 }
    : { x: document.documentElement.clientWidth, y: document.documentElement.clientHeight }
}

export function trailRevealRadius(width: number, height: number, origin: Point) {
  return Math.hypot(Math.max(origin.x, width - origin.x), Math.max(origin.y, height - origin.y))
}

export function createTrailRevealClip(
  width: number,
  height: number,
  origin: Point,
  radius = trailRevealRadius(width, height, origin),
) {
  const cell = trailOpeningCell
  const clampX = (x: number) => Math.max(0, Math.min(width, x))
  return (progress: number, inverse = false) => {
    const reach = progress * (radius + cell)
    const left: string[] = []
    const right: string[] = []
    // Trace the pixel edge once. The SVG stays live beneath a vector clip, with no bitmap uploads.
    for (let y = 0; y < height; y += cell) {
      const bottom = Math.min(height, y + cell)
      const distance = Math.abs((y + bottom) / 2 - origin.y)
      const span = Math.sqrt(Math.max(0, reach * reach - distance * distance))
      const x1 = clampX(span ? Math.floor((origin.x - span) / cell) * cell : origin.x)
      const x2 = clampX(span ? Math.ceil((origin.x + span) / cell) * cell : origin.x)
      left.push(`${x1}px ${y}px`, `${x1}px ${bottom}px`)
      right.push(`${x2}px ${y}px`, `${x2}px ${bottom}px`)
    }
    const edge = [...left, ...right.reverse(), left[0]]
    if (inverse)
      edge.unshift('0 0', `${width}px 0`, `${width}px ${height}px`, `0 ${height}px`, '0 0')
    return `polygon(evenodd, ${edge.join(', ')})`
  }
}

export function mountSceneReveal(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!view || !svg) throw new Error('The trail opening needs its surface')
  let compositor = false
  let origin: Point = { x: 0, y: 0 }
  let radius = 1
  let midpointRadius = 0
  let midpointFrame = 0
  let opensFromZero = true
  let requested = false
  let animation: Animation | null = null
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  const finish = () => {
    const phase = scene.dataset.trailReveal
    view.cancelAnimationFrame(midpointFrame)
    midpointFrame = 0
    requested = false
    if (animation) {
      animation.onfinish = null
      animation.cancel()
      animation = null
    }
    host.style.removeProperty('clip-path')
    delete scene.dataset.trailReveal
    if (phase === 'closing') host.dispatchEvent(new Event('trail:closed'))
    else if (phase === 'revealing') host.dispatchEvent(new Event('trail:reveal-end'))
  }
  // Which way the running sweep goes. Read from here, not from `playbackRate`: a reversal is
  // pending until the animation is ready, and the getter reports the old rate until then.
  let forward = true
  const checkMidpoint = () => {
    midpointFrame = 0
    if (!animation || scene.dataset.trailReveal !== 'closing') return
    const progress = animation.effect?.getComputedTiming().progress
    if (progress !== null && progress !== undefined) {
      const reach = Math.ceil(radius) * (opensFromZero ? progress : 1 - progress)
      if (reach <= midpointRadius) {
        host.dispatchEvent(new Event('trail:close-midpoint'))
        return
      }
    }
    midpointFrame = view.requestAnimationFrame(checkMidpoint)
  }
  const animate = (open: boolean) => {
    view.cancelAnimationFrame(midpointFrame)
    midpointFrame = open ? 0 : view.requestAnimationFrame(checkMidpoint)
    scene.dataset.trailReveal = open ? 'revealing' : 'closing'
    if (animation) {
      // An interrupted sweep turns round from where it is.
      if (forward !== open) {
        animation.reverse()
        forward = open
      }
      return
    }
    // A closing sweep is its own animation, played forward from the full circle. Starting a
    // forward animation at its end and reversing it left a frame in which the animation was
    // "finished" with the reversal still pending, and its finish event closed the scene at once.
    const center = `${origin.x}px ${origin.y}px`
    const shut = { clipPath: `circle(0px at ${center})` }
    const full = { clipPath: `circle(${Math.ceil(radius)}px at ${center})` }
    forward = open
    opensFromZero = open
    animation = host.animate(open ? [shut, full] : [full, shut], {
      duration: trailThemeDuration,
      easing: open ? 'ease-out' : 'ease-in',
      fill: 'both',
    })
    animation.onfinish = () => finish()
  }
  const reveal = () => {
    if (!requested || host.hidden || scene.dataset.trailOpening !== 'complete') return
    requested = false
    if (reduced.matches) return finish()
    animate(true)
  }
  const open = () => {
    if (animation) return animate(true)
    requested = true
    host.style.clipPath = 'inset(50%)'
    reveal()
  }
  const close = (event: Event) => {
    requested = false
    if (host.hidden || scene.dataset.trailOpening !== 'complete' || reduced.matches) return finish()
    event.preventDefault()
    animate(false)
  }
  const present = () => {
    if (
      signal.aborted ||
      scene.dataset.trailOpening === 'complete' ||
      scene.dataset.trailRendered !== 'true' ||
      (compositor && scene.dataset.trailPaintReady !== 'true')
    )
      return
    scene.dataset.trailOpening = 'complete'
    scene.dataset.trailOpeningProgress = '1'
    reveal()
    host.dispatchEvent(new Event('trail:prepared'))
    if (!host.hidden) host.dispatchEvent(new Event('trail:revealed'))
  }
  host.addEventListener('trail:open', open, { signal })
  host.addEventListener('trail:close', close, { signal })
  host.addEventListener('trail:hide', finish, { signal })
  view.addEventListener(
    'resize',
    () => {
      // Layout can settle while the first frame is still preparing. Keep its requested
      // reveal; drawScene updates the origin and radius for the new viewport.
      if (scene.dataset.trailOpening === 'complete') finish()
    },
    { signal },
  )
  reduced.addEventListener('change', () => reduced.matches && finish(), { signal })
  signal.addEventListener('abort', finish, { once: true })
  host.addEventListener('trail:ready', () => !compositor && present(), { signal })
  host.addEventListener('trail:show', present, { signal })
  scene.dataset.trailOpening = 'pending'
  scene.dataset.trailOpeningProgress = '0'
  if (!host.hidden) open()
  return {
    present,
    origin: () => origin,
    radius: () => radius,
    useCompositor() {
      compositor = true
    },
    useFallback() {
      compositor = false
      present()
    },
    update(next: { width: number; height: number }) {
      const bounds = svg.getBoundingClientRect()
      const clock = trailRevealOrigin(document)
      origin = { x: clock.x - bounds.left, y: clock.y - bounds.top }
      midpointRadius = Math.hypot(view.innerWidth / 2 - clock.x, view.innerHeight / 2 - clock.y)
      radius = trailRevealRadius(next.width, next.height, origin)
      scene.dataset.trailOpeningOriginX = String(origin.x)
      scene.dataset.trailOpeningOriginY = String(origin.y)
      scene.dataset.trailOpeningRadius = String(radius)
    },
  }
}

export type SceneReveal = ReturnType<typeof mountSceneReveal>
