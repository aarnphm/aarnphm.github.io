import { createTrailRevealMask, trailThemeDuration } from '../visitor-trail/scene-reveal'

// The old page is a snapshot; the new page, including the trail canvas, stays live.
// Its mask follows the compositor's existing reveal instead of starting another clock.
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
    const surface = scene?.querySelector<HTMLCanvasElement>('.trail-compositor')
    if (scene?.dataset.trailResizePhase) {
      update()
      return
    }
    const followsScene = Boolean(
      host &&
      !host.hidden &&
      surface &&
      (scene?.dataset.trailRenderer === 'webgl' ||
        (scene?.dataset.trailOpeningLive === 'true' && scene.dataset.trailOpening !== 'complete')),
    )
    const started = performance.now()
    const opening = followsScene && scene?.dataset.trailOpening !== 'complete'
    const rect = followsScene ? surface?.getBoundingClientRect() : undefined
    const origin = rect
      ? Math.min(Number(scene?.dataset.trailOpeningOrigin) || rect.height, rect.height)
      : window.innerHeight
    const mask = createTrailRevealMask(document, window.innerWidth, window.innerHeight, origin, {
      x: -(rect?.left ?? 0),
      y: -(rect?.top ?? 0),
    })
    if (!mask) {
      update()
      return
    }

    const style = document.createElement('style')
    style.textContent = ':root[data-home-theme-reveal]::view-transition-old(root) {}'
    document.head.append(style)
    const rule = style.sheet?.cssRules[0]
    if (!(rule instanceof CSSStyleRule)) {
      style.remove()
      update()
      return
    }
    rule.style.maskRepeat = 'no-repeat'
    let painted = -1
    const draw = (progress: number) => {
      const step = Math.floor(progress * 240)
      if (painted === step) return
      painted = step
      const drawn = mask.draw(progress, true)
      rule.style.maskImage = drawn.image
      rule.style.maskSize = drawn.size
      rule.style.maskPosition = drawn.position
    }
    draw(0)
    if (followsScene && scene && !opening) scene.dataset.trailThemeStarted = String(started)
    root.dataset.homeThemeReveal = 'pending'
    let frame = 0
    let finished = false
    let seenScene = false
    const observer = new MutationObserver(records => {
      if (
        records.some(record => record.oldValue !== 'revealing') &&
        scene?.dataset.trailThemePhase === 'revealing'
      )
        seenScene = true
    })
    const request = { theme, update }
    pending = request
    const transition = document.startViewTransition(() => {
      if (current === generation && !signal.aborted) {
        pending = null
        request.update()
        if (followsScene && scene && !opening)
          observer.observe(scene, {
            attributes: true,
            attributeOldValue: true,
            attributeFilter: ['data-trail-theme-phase'],
          })
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
      let revealed = 0
      let previous = started
      const tick = (now: number) => {
        if (finished) return
        let progress = Math.min(
          1,
          (now - started) / trailThemeDuration,
          revealed + Math.min(32, now - previous) / (trailThemeDuration / 2),
        )
        if (followsScene && scene) {
          const phase = opening ? scene.dataset.trailOpening : scene.dataset.trailThemePhase
          progress =
            seenScene || opening
              ? Number(
                  opening ? scene.dataset.trailOpeningProgress : scene.dataset.trailThemeProgress,
                ) || 0
              : 0
          if (phase === 'complete' && (seenScene || opening)) progress = 1
        }
        revealed = progress
        previous = now
        draw(progress)
        if (progress >= 1) cleanup()
        else frame = window.requestAnimationFrame(tick)
      }
      frame = window.requestAnimationFrame(tick)
    }, cleanup)
    void transition.finished.then(cleanup, cleanup)
  }
}
