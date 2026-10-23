import { trailScene } from '../../../util/trail-scene'
import { trailClockRegion } from '../../visitor-trail/clock-clearance'
import { center, clockEl, onClock, petState, railY, reducedMotion } from './actor-model'
import { render } from './appearance'
import { moveTo } from './perch-motion'
import { floorNear, petObstacles, pickEdge } from './perch-placement'
import { railFor } from './scroll-carry'
import { trailPointClear } from './trail-clearance'
import { enterTrail } from './trail-drive'
import { currentTrailPath, petTrail, trailGeometry } from './trail-geometry'
import { clearTrailActor, leaveTrail, pauseTrail, rememberTrail } from './trail-pose'
import { walkTrail } from './trail-walk'

// The home page's clock is the door to the trail: the scene opens out of it and shuts back into
// it. The pet rides both sweeps sitting on the clock, as the 404 pet keeps the keystone over its
// gate: a leap there on the click, a beat on the clock, then onto the route or back to the page.
// It sits past the clock's left end with its legs over the paper, so the digits stay readable;
// every other perch keeps out of the clock's corner.
const clockDwellMs = 900

const perchClock = () => {
  const clock = clockEl()
  if (!petState.root || !clock || petState.carry) return false
  const rect = clock.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return false
  petState.clockLanded = performance.now()
  petState.afterMove = () => {
    petState.clockLanded = performance.now()
  }
  moveTo({ el: clock, kind: 'sit', t: -(0.36 * petState.unit) / rect.width }, 'leap')
  return true
}

// The home page's trail scene: the pet goes onto the route once the scene shows and has drawn it,
// off it when the scene hides or a board covers the pet, and onto the clock as the scene opens and
// closes. `trailNav.trailWanted` is shared with the page's scroll and resize handlers.
export function watchTrail(
  trailNav: { trailWanted: boolean },
  scene: HTMLElement | null,
  placed: boolean | undefined,
  signal: AbortSignal,
) {
  let trailFrame = 0
  let clockTimer = 0
  let waiting: Animation | null = null
  const refreshTrail = () => {
    trailFrame = 0
    if (signal.aborted || !scene || scene.hidden || document.hidden || reducedMotion()) return
    const drawing = trailScene(document)
    if (drawing?.dataset.trailOpening !== 'complete' || drawing.dataset.trailReveal !== undefined)
      return
    if (scene.dataset.trailResizing === 'true') {
      trailNav.trailWanted ||= !!petState.trailWalk
      if (petState.trailWalk?.duration) pauseTrail()
      return
    }
    // The frame replaces its routes over several frames of a redraw and reports rendered only at
    // the end. A walk whose route has moved or gone stops at once, before the rendered check, so
    // the pet does not keep hopping along a path that is no longer there; it resumes on the new
    // route once the scene is ready.
    const obstacles = petObstacles()
    const text = { ...obstacles, camps: [] }
    const currentTrail = petState.trailWalk
    if (
      currentTrail &&
      (!currentTrailPath(currentTrail.path) ||
        currentTrail.geometry !== trailGeometry(currentTrail.path) ||
        currentTrail.feet.some(foot => !trailPointClear(foot, currentTrail.path, text)))
    ) {
      pauseTrail()
      trailNav.trailWanted = true
    }
    if (trailScene(document)?.dataset.trailRendered !== 'true') return
    if (!trailNav.trailWanted || petState.carry) return
    // On the trail a move finishes first, and the leap to the clock lands and sits a beat there;
    // from anywhere else on the page the pet comes in at once.
    const current = petState.trailWalk || onClock() ? petState.hop : null
    if (current) {
      if (waiting === current) return
      waiting = current
      const ready = () => {
        if (waiting !== current || signal.aborted) return
        waiting = null
        queueTrail()
      }
      void current.finished.then(ready, ready)
      return
    }
    if (onClock()) {
      const left = clockDwellMs - (performance.now() - petState.clockLanded)
      if (left > 0) {
        window.clearTimeout(clockTimer)
        clockTimer = window.setTimeout(queueTrail, left + 20)
        return
      }
    }
    if (petState.trailWalk ? walkTrail(obstacles) : enterTrail(obstacles)) {
      trailNav.trailWanted = false
      return
    }
    const resting = petState.trailWalk
    if (
      resting &&
      (!currentTrailPath(resting.path) ||
        !trailPointClear(resting.feet[0], resting.path, obstacles))
    ) {
      const anchor = center()
      const next = pickEdge(anchor, resting.path) ?? floorNear(anchor.x)
      if (next) moveTo(next, 'fade')
      else if (petState.root) petState.root.hidden = true
    }
  }
  // The frame can redraw in its own animation callback after the host's reflow event.
  const queueTrail = () => {
    if (trailFrame || signal.aborted) return
    trailFrame = requestAnimationFrame(() => {
      trailFrame = requestAnimationFrame(refreshTrail)
    })
  }
  const pausePageTrail = () => {
    cancelAnimationFrame(trailFrame)
    window.clearTimeout(clockTimer)
    trailFrame = 0
    waiting = null
    trailNav.trailWanted = false
    pauseTrail()
  }
  const closeTrail = () => {
    pausePageTrail()
    leaveTrail()
    petState.trailJourney = null
  }
  const clearClock = () => {
    const region = trailClockRegion(document)
    if (!petState.root || !region) return
    if (petState.carry) {
      petState.hop?.cancel()
      petState.hop = null
      petState.override = null
      petState.landAfterJet = false
      petState.carry = railFor()
      petState.root.style.setProperty('--pet-rail-from', `${petState.carry.from.toFixed(1)}px`)
      petState.root.style.setProperty('--pet-rail-to', `${petState.carry.to.toFixed(1)}px`)
      petState.root.style.transform = `translate3d(${petState.carry.x.toFixed(1)}px, 0, 0)`
      petState.root.style.translate = `0 ${railY(petState.carry).toFixed(1)}px`
      render()
      return
    }
    const box = petState.root.getBoundingClientRect()
    if (box.right <= region.left || box.bottom <= region.top || box.top >= region.bottom) return
    trailNav.trailWanted ||= !!petState.trailWalk
    rememberTrail()
    const next = floorNear(region.left - petState.width) ?? pickEdge(center(), null)
    if (next) moveTo(next, 'fade')
    else petState.root.hidden = true
  }
  // Fresh page placement already checked the clock in its obstacle snapshot. Slides have their
  // own perch policy; retained pets still need the live check after the page changes. A trail
  // still opening when the pet mounts takes it to the clock instead.
  const openingAtLoad =
    !!scene && !scene.hidden && trailScene(document)?.dataset.trailOpening !== 'complete'
  if ((!placed || document.body.dataset.slides === 'true') && !openingAtLoad) clearClock()
  if (scene) {
    const openTrail = () => {
      if (scene.hidden) return
      trailNav.trailWanted = true
      queueTrail()
    }
    const leaveForClock = () => {
      perchClock()
      pausePageTrail()
    }
    scene.addEventListener('trail:before-toggle', leaveForClock, { signal })
    scene.addEventListener('trail:show', openTrail, { signal })
    scene.addEventListener('trail:hide', closeTrail, { signal })
    scene.addEventListener('trail:dispose', closeTrail, { signal })
    scene.addEventListener('trail:reflow', queueTrail, { signal })
    scene.addEventListener('trail:ready', openTrail, { signal })
    scene.addEventListener('trail:revealed', openTrail, { signal })
    scene.addEventListener('load', openTrail, { capture: true, signal })
    // The host's events bracket a redraw; the route paths change in between, so they are watched.
    let paths: MutationObserver | null = null
    const watchPaths = () => {
      paths?.disconnect()
      const trail = trailScene(document)
      if (!trail) return
      const canvas = trail.querySelector('svg.trail-canvas')
      paths = new MutationObserver(records => {
        if (
          records.some(
            record =>
              record.target !== canvas ||
              [...record.addedNodes, ...record.removedNodes].some(
                node => node instanceof Element && petTrail(node),
              ),
          )
        )
          queueTrail()
      })
      paths.observe(trail, { attributes: true, attributeFilter: ['data-trail-rendered'] })
      if (canvas) paths.observe(canvas, { childList: true })
      const expedition = trail.querySelector('[data-pet-expedition]')
      if (expedition)
        paths.observe(expedition, {
          attributes: true,
          attributeFilter: ['d', 'data-pet-scale-stops', 'data-pet-alpine-span'],
        })
    }
    scene.addEventListener('trail:mounted', watchPaths, { signal })
    scene.addEventListener('trail:ready', watchPaths, { signal })
    watchPaths()
    signal.addEventListener('abort', () => paths?.disconnect(), { once: true })
    const controlsChanged = () => {
      if (petState.root && !petState.carry) {
        const pet = petState.root.getBoundingClientRect()
        const covered = [...document.querySelectorAll<HTMLElement>('.homepage-board')].some(
          panel => {
            const box = panel.getBoundingClientRect()
            return (
              box.left < pet.right &&
              box.right > pet.left &&
              box.top < pet.bottom &&
              box.bottom > pet.top
            )
          },
        )
        if (covered) {
          pauseTrail()
          const anchor = center()
          const next = pickEdge(anchor, petState.perch?.el ?? null) ?? floorNear(anchor.x)
          if (next) moveTo(next, 'fade')
          else {
            clearTrailActor()
            petState.root.hidden = true
          }
        }
      }
      trailNav.trailWanted = true
      queueTrail()
    }
    const controlsObserver = new MutationObserver(controlsChanged)
    controlsObserver.observe(scene, {
      subtree: true,
      attributes: true,
      attributeFilter: ['hidden'],
    })
    const observeControls = () => {
      const drawing = trailScene(document)
      if (drawing)
        controlsObserver.observe(drawing, {
          subtree: true,
          attributes: true,
          attributeFilter: ['hidden'],
        })
    }
    observeControls()
    scene.addEventListener('trail:mounted', observeControls, { signal })
    signal.addEventListener('abort', () => controlsObserver.disconnect(), { once: true })
    window.addEventListener('resize', queueTrail, { passive: true, signal })
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    motion.addEventListener('change', () => (motion.matches ? pausePageTrail() : openTrail()), {
      signal,
    })
    document.addEventListener(
      'visibilitychange',
      () => {
        if (document.hidden) {
          const resume = trailNav.trailWanted || !!petState.trailWalk
          pausePageTrail()
          trailNav.trailWanted = resume
        } else queueTrail()
      },
      { signal },
    )
    if (!scene.hidden) openTrail()
    if (openingAtLoad) perchClock()
  }
  return { queueTrail, clearClock, closeTrail }
}
