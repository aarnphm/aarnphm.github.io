import {
  createTrailRevealClip,
  trailRevealOrigin,
  trailRevealProgress,
  trailThemeDuration,
} from '../visitor-trail/scene-reveal'

// The old page is a snapshot; its clip follows the live trail's reveal clock.
export function mountHomepageTheme(signal: AbortSignal) {
  const root = document.documentElement
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  let cancel = () => {}
  let generation = 0
  let pending: { theme: 'light' | 'dark'; update: () => void } | null = null

  const stop = () => {
    const request = pending
    cancel()
    if (request) {
      generation += 1
      request.update()
    }
  }
  signal.addEventListener('abort', stop, { once: true })
  window.addEventListener('resize', stop, { signal })
  window.addEventListener('scroll', stop, { signal, passive: true })
  document.addEventListener('visibilitychange', stop, { signal })
  reduced.addEventListener('change', stop, { signal })

  return (update: () => void, theme: 'light' | 'dark') => {
    if (pending?.theme === theme) {
      pending.update = update
      return
    }
    const changed = root.getAttribute('saved-theme') !== theme
    // A preference can change (realtime to system) without changing the resolved palette.
    if (!changed && !pending) {
      update()
      return
    }
    cancel()
    const current = ++generation
    if (
      signal.aborted ||
      !changed ||
      document.body.dataset.slug !== 'index' ||
      reduced.matches ||
      document.hidden ||
      typeof document.startViewTransition !== 'function'
    ) {
      update()
      return
    }

    const host = document.querySelector<HTMLElement>('.homepage-trail-scene')
    const scene = host?.shadowRoot?.querySelector<HTMLElement>('.trail-scene')
    const surface = scene?.querySelector<SVGSVGElement>('.trail-canvas')
    const followsScene = Boolean(host && !host.hidden && surface)
    const opening = followsScene && scene?.dataset.trailOpening !== 'complete'
    const gpu = followsScene && scene?.dataset.trailRenderer === 'webgl'
    const rect = followsScene ? surface?.getBoundingClientRect() : undefined
    const clock = trailRevealOrigin(document)
    const radius = Number(scene?.dataset.trailOpeningRadius)
    const origin =
      (opening || gpu) && rect && scene && radius > 0
        ? {
            x: Number(scene.dataset.trailOpeningOriginX) + rect.left,
            y: Number(scene.dataset.trailOpeningOriginY) + rect.top,
          }
        : clock
    const clip = createTrailRevealClip(
      window.innerWidth,
      window.innerHeight,
      origin,
      (opening || gpu) && radius > 0 ? radius : undefined,
    )

    const style = document.createElement('style')
    style.textContent = ':root[data-home-theme-reveal]::view-transition-old(root) {}'
    document.head.append(style)
    const rule = style.sheet?.cssRules[0]
    if (!(rule instanceof CSSStyleRule)) {
      style.remove()
      update()
      return
    }
    const draw = (progress: number) => {
      rule.style.clipPath = clip(progress, true)
    }
    draw(0)
    if (gpu && scene && !opening) scene.dataset.trailThemeStarted = String(performance.now())
    root.dataset.homeThemeReveal = 'pending'
    let frame = 0
    let finished = false
    let seenScene = false
    const observer = new MutationObserver(records => {
      if (
        scene?.dataset.trailThemePhase === 'revealing' ||
        records.some(record => record.oldValue === 'revealing')
      )
        seenScene = true
    })
    if (gpu && scene && !opening)
      observer.observe(scene, {
        attributes: true,
        attributeOldValue: true,
        attributeFilter: ['data-trail-theme-phase'],
      })
    const request = { theme, update }
    pending = request
    const transition = document.startViewTransition(() => {
      if (current === generation && !signal.aborted) {
        pending = null
        request.update()
      }
    })
    const cleanup = () => {
      if (finished) return
      finished = true
      observer.disconnect()
      window.cancelAnimationFrame(frame)
      style.remove()
      host?.removeEventListener('trail:hide', cleanup)
      transition.skipTransition()
      if (current === generation) {
        pending = null
        if (scene) delete scene.dataset.trailThemeStarted
        delete root.dataset.homeThemeReveal
        cancel = () => {}
      }
    }
    cancel = cleanup
    host?.addEventListener('trail:hide', cleanup, { once: true })
    void transition.ready.then(() => {
      if (finished) return
      root.dataset.homeThemeReveal = 'revealing'
      const started = performance.now()
      const tick = (now: number) => {
        if (finished) return
        let progress = trailRevealProgress((now - started) / trailThemeDuration)
        if (opening && scene) {
          progress = Number(scene.dataset.trailOpeningProgress) || 0
          if (scene.dataset.trailOpening === 'complete') progress = 1
        } else if (gpu && scene) {
          progress = seenScene ? Number(scene.dataset.trailThemeProgress) || 0 : 0
          if (seenScene && scene.dataset.trailThemePhase === 'complete') progress = 1
        }
        draw(progress)
        if (progress >= 1) cleanup()
        else frame = window.requestAnimationFrame(tick)
      }
      frame = window.requestAnimationFrame(tick)
    }, cleanup)
    void transition.finished.then(cleanup, cleanup)
  }
}
