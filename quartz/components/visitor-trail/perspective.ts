export const parallaxMovement = { horizontal: 8, vertical: 4, scroll: 12 }
export const parallaxDepth = { alpine: 0.25, foothill: 0.6 }

export function mountTrailPerspective(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('Trail perspective needs its document')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let layers: { element: SVGGElement; depth: number }[] = []
  let frame: number | null = null
  let previous: number | null = null
  let pointer = { x: 0, y: 0 }
  let current = { x: 0, y: 0 }
  let target = { x: 0, y: 0 }

  const render = () => {
    for (const { element, depth } of layers)
      element.setAttribute(
        'transform',
        `translate(${(current.x * depth).toFixed(2)} ${(current.y * depth).toFixed(2)})`,
      )
  }
  const pause = () => {
    if (frame !== null) view.cancelAnimationFrame(frame)
    frame = null
    previous = null
    scene.dataset.perspectiveMotion = 'paused'
  }
  const tick = (time: number) => {
    frame = null
    if (host.hidden || document.hidden || signal.aborted || reducedMotion.matches) return pause()
    const elapsed = previous === null ? 16 : Math.min(64, time - previous)
    previous = time
    const step = 1 - Math.exp(-elapsed / 90)
    current.x += (target.x - current.x) * step
    current.y += (target.y - current.y) * step
    if (Math.hypot(target.x - current.x, target.y - current.y) < 0.02) {
      current = { ...target }
      previous = null
      scene.dataset.perspectiveMotion = 'idle'
    } else frame = view.requestAnimationFrame(tick)
    render()
  }
  const schedule = () => {
    if (host.hidden || document.hidden || signal.aborted || reducedMotion.matches) return
    target = {
      x: pointer.x * parallaxMovement.horizontal,
      y:
        pointer.y * parallaxMovement.vertical +
        Math.min(parallaxMovement.scroll, view.scrollY * 0.025),
    }
    if (frame === null && Math.hypot(target.x - current.x, target.y - current.y) >= 0.02) {
      scene.dataset.perspectiveMotion = 'active'
      frame = view.requestAnimationFrame(tick)
    }
  }
  document.addEventListener(
    'pointermove',
    event => {
      if (event.pointerType !== 'mouse' || host.hidden || reducedMotion.matches) return
      pointer = {
        x: Math.max(-1, Math.min(1, (event.clientX / view.innerWidth - 0.5) * 2)),
        y: Math.max(-1, Math.min(1, (event.clientY / view.innerHeight - 0.5) * 2)),
      }
      schedule()
    },
    { signal, passive: true },
  )
  document.addEventListener(
    'pointerleave',
    () => {
      pointer = { x: 0, y: 0 }
      schedule()
    },
    { signal },
  )
  view.addEventListener('scroll', schedule, { signal, passive: true })
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : schedule()), {
    signal,
  })
  reducedMotion.addEventListener(
    'change',
    () => {
      pause()
      if (reducedMotion.matches) {
        current = { x: 0, y: 0 }
        target = { x: 0, y: 0 }
        render()
        scene.dataset.perspectiveMotion = 'still'
      } else schedule()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })
  return {
    pause,
    resume: schedule,
    update() {
      layers = [...scene.querySelectorAll<SVGGElement>('[data-trail-depth]')].map(element => ({
        element,
        depth:
          element.dataset.trailDepth === 'alpine'
            ? parallaxDepth.alpine
            : element.dataset.trailDepth === 'foothill'
              ? parallaxDepth.foothill
              : Math.max(0, Math.min(1, Number(element.dataset.trailDepth) || 0)),
      }))
      render()
      schedule()
    },
  }
}
