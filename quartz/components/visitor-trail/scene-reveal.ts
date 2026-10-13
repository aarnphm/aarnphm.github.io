import type { Point } from './landscape'

type RevealRoute = {
  id: string
  samples: (Point & { position: number })[]
  delay: number
  duration: number
}

export const trailOpeningTiming = (order: number) => ({ delay: 450 + order * 260, duration: 7000 })

export function mountSceneReveal(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('The trail opening needs its document')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  const animations = new Set<Animation>()
  let painted = new WeakSet<Element>()
  let epoch: number | null = null
  let completion = 0
  const now = () => {
    const time = document.timeline.currentTime
    return typeof time === 'number' ? time : view.performance.now()
  }
  const cancel = () => {
    view.clearTimeout(completion)
    for (const animation of animations) animation.cancel()
    animations.clear()
  }
  const reset = () => {
    cancel()
    painted = new WeakSet()
    epoch = null
    scene.dataset.trailOpening = 'pending'
  }
  const start = () => {
    if (epoch === null) epoch = now()
  }
  const animate = (element: Element, frames: Keyframe[], delay: number, duration = 650) => {
    if (painted.has(element)) return
    painted.add(element)
    if (epoch === null || reducedMotion.matches || now() - epoch >= delay + duration) return
    const animation = element.animate(frames, {
      id: 'trail-opening',
      delay,
      duration,
      easing: 'linear',
      fill: 'backwards',
    })
    // Generation and depth sorting insert objects at different times into the same opening.
    animation.startTime = epoch
    animations.add(animation)
    animation.addEventListener('finish', () => animations.delete(animation), { once: true })
  }
  const fade = (element: Element, delay: number, duration?: number) => {
    if (painted.has(element)) return
    animate(
      element,
      [{ opacity: 0 }, { opacity: view.getComputedStyle(element).opacity }],
      delay,
      duration,
    )
  }
  const update = (routes: RevealRoute[], pond: Point) => {
    if (!routes.length) return
    start()
    view.clearTimeout(completion)
    // The ranges come up right behind the terrain, tone by tone, so the scene has its horizon
    // while the routes are still setting out.
    const ridge = 250
    const reading = Math.max(...routes.map(route => route.delay + route.duration)) + 400
    const remaining =
      reading +
      3600 +
      Math.max(0, scene.querySelectorAll('[data-pet-walk-reveal]').length - 1) * 180 -
      (now() - (epoch ?? now()))
    if (reducedMotion.matches || remaining <= 0) {
      scene.dataset.trailOpening = 'complete'
      return
    }
    const arrival = ({ x, y }: Point) => {
      let closest = Infinity
      let delay = 450
      for (const route of routes) {
        for (let index = 0; index < route.samples.length; index += 4) {
          const sample = route.samples[index]
          const distance = (sample.x - x) ** 2 + (sample.y - y) ** 2
          if (distance >= closest) continue
          closest = distance
          delay = route.delay + (1 - sample.position) * route.duration
        }
      }
      return delay + Math.min(600, Math.sqrt(closest) * 2)
    }
    for (const mask of scene.querySelectorAll<SVGElement>('[data-trail-reveal]')) {
      const route = routes.find(route => route.id === mask.dataset.trailReveal)
      if (route)
        animate(
          mask,
          [{ strokeDashoffset: -1 }, { strokeDashoffset: 0 }],
          route.delay,
          route.duration,
        )
    }
    for (const [index, mask] of scene.querySelectorAll('[data-pet-walk-reveal]').entries())
      animate(mask, [{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }], reading + index * 180, 3600)

    for (const element of scene.querySelectorAll('.trail-upland, .trail-terrain'))
      fade(element, 200)
    for (const element of scene.querySelectorAll('.trail-pond')) fade(element, arrival(pond), 900)
    for (const element of scene.querySelectorAll<SVGElement>('.trail-mountain-paint')) {
      const step = Number(element.style.getPropertyValue('--trail-paint-step'))
      const foothill = element.closest('[data-formation="foothills"]') ? 300 : 0
      fade(element, ridge + foothill + step * 90, 800)
    }
    for (const element of scene.querySelectorAll(
      '.trail-mountain-roads, .trail-alpine-tracks, .trail-alpine-climbers',
    ))
      fade(element, ridge + 900)
    for (const element of scene.querySelectorAll<SVGElement>(
      '.trail-tree, .trail-woodland-tree, [data-trail-wander], [data-holiday-decoration], .trail-map-reader, .trail-backpacker, [data-alpine-traveler], [data-camp-person], [data-camp-tent], .trail-extra-camp, [data-park-ranger], [data-trail-visitor]',
    )) {
      if (painted.has(element)) continue
      const x = Number(element.dataset.groundX)
      const y = Number(element.dataset.groundY)
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      const far =
        element.dataset.treeDepth === 'foothill' || element.hasAttribute('data-alpine-traveler')
      fade(element, far ? Math.max(ridge + 900, arrival({ x, y })) : arrival({ x, y }))
    }
    for (const element of scene.querySelectorAll(
      '.trail-hiker-tracks, .trail-reader-tracks, .trail-camp-tracks, .trail-visitor-tracks, .trail-ranger-tracks',
    ))
      fade(element, 450)
    scene.dataset.trailOpening = 'revealing'
    completion = view.setTimeout(() => {
      scene.dataset.trailOpening = 'complete'
    }, remaining)
  }
  reducedMotion.addEventListener(
    'change',
    () => {
      if (reducedMotion.matches) {
        cancel()
        scene.dataset.trailOpening = 'complete'
      }
    },
    { signal },
  )
  signal.addEventListener('abort', cancel, { once: true })
  reset()
  return { start, reset, update }
}
