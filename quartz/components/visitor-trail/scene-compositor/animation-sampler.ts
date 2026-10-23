import type { Overlays } from './retained-presentation'
import type { SceneTree } from './scene-partition'
import { motionProperties, type Scope } from './render-model'

// Animations: paused around the opening, and sampled each frame into moved or dirtied textures.
export function createAnimationSampler(scope: Scope, sceneTree: SceneTree, overlays: Overlays) {
  const { is, shared, svg, signal, view } = scope
  const { fading, groups, promote, touch, dirtyGroup, touchDefinition, settle } = sceneTree
  const { holdResize, capturePalette } = overlays
  const animations = new Set<Animation>()
  let discoveryPending = true
  const discover = () => {
    discoveryPending = true
  }
  for (const name of ['animationstart', 'transitionrun', 'trail:animation'])
    svg.addEventListener(name, discover, { signal })
  const readAnimations = () => {
    if (discoveryPending) {
      discoveryPending = false
      for (const animation of svg.getAnimations({ subtree: true })) animations.add(animation)
    }
    return animations
  }
  const pausedAnimations = new Set<Animation>()
  let pauseFrame = 0
  const cancelSuspension = () => {
    view.cancelAnimationFrame(pauseFrame)
    pauseFrame = 0
  }
  signal.addEventListener('abort', cancelSuspension, { once: true })
  const pauseAnimations = (animations: Iterable<Animation>) => {
    const running = [...animations].filter(
      animation => !pausedAnimations.has(animation) && animation.playState === 'running',
    )
    for (const animation of running) {
      pausedAnimations.add(animation)
      animation.pause()
    }
  }
  const suspendAnimations = () => {
    cancelSuspension()
    const pending = [...readAnimations()].filter(
      animation => !pausedAnimations.has(animation) && animation.playState === 'running',
    )
    let next = 0
    const suspend = () => {
      pauseFrame = 0
      // The closing wipe uses retained pixels. Spread SVG style invalidation over its frames.
      for (const animation of pending.slice(next, (next += 8))) {
        pausedAnimations.add(animation)
        animation.pause()
      }
      if (next < pending.length) pauseFrame = view.requestAnimationFrame(suspend)
    }
    if (pending.length) pauseFrame = view.requestAnimationFrame(suspend)
  }
  const resumeAnimations = () => {
    cancelSuspension()
    const paused = [...pausedAnimations].filter(animation => animation.playState === 'paused')
    for (const animation of paused) animation.play()
    pausedAnimations.clear()
  }
  let settled = 0
  // Whether an animation changes pixels inside its target, rather than only moving or fading it.
  const paints = new WeakMap<
    Animation,
    {
      content: boolean
      state: AnimationPlayState
      style: CSSStyleDeclaration
      properties: string[]
      values: string[]
    }
  >()
  const timing = new Set(['offset', 'computedOffset', 'easing', 'composite'])
  const scan = (sample: Iterable<Animation>) => {
    if (shared.palettePending || shared.paletteTransition) {
      const fullPalette = shared.palettePending
      shared.palettePending = false
      // Palette changes during a resize join its replacement instead of stacking two reveals.
      if (shared.resizeFrame?.started !== null && shared.resizeFrame) holdResize()
      if (fullPalette) shared.palette += 1
      if (shared.paletteTransition && !shared.resizeFrame) capturePalette()
      shared.paletteTransition = false
      // Theme colors do not move the scene. Preserve matrices and surface uploads;
      // the palette version already invalidates the rasters that need new colors.
      if (fullPalette) {
        shared.flush = true
        for (const owner of groups.values()) owner.fresh = false
      }
      shared.revision += 1
    }
    for (const animation of sample) {
      const effect = animation.effect
      if (!is(effect, 'KeyframeEffect') || !effect.target || !svg.contains(effect.target)) {
        animations.delete(animation)
        continue
      }
      const state = animation.playState
      // Sample the final or cancelled state once before releasing its target.
      if (state === 'finished' || state === 'idle') animations.delete(animation)
      let sampled = paints.get(animation)
      // A stepped fade reaches its final opacity when it finishes. Sample that state once
      // before retaining it, otherwise footprints stay invisible until their textures regroup.
      if (state !== 'running' && sampled?.state === state) continue
      const target = effect.target
      if (target.closest('defs')) {
        touchDefinition(target)
        continue
      }
      if (!sampled) {
        const properties = is(animation, 'CSSTransition')
          ? [animation.transitionProperty]
          : effect.getKeyframes().flatMap(frame => Object.keys(frame))
        sampled = {
          content: properties.some(name => !timing.has(name) && !motionProperties.has(name)),
          state,
          style: groups.get(target)?.style ?? view.getComputedStyle(target),
          properties: [
            ...new Set(
              properties
                .filter(name => !timing.has(name))
                .map(name => name.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`)),
            ),
          ],
          values: [],
        }
        paints.set(animation, sampled)
      }
      sampled.state = state
      if (!sampled.content && state === 'running') fading.set(target, shared.clock)
      let changed = sampled.values.length !== sampled.properties.length
      for (const [index, name] of sampled.properties.entries()) {
        const value = sampled.style.getPropertyValue(name)
        changed ||= value !== sampled.values[index]
        sampled.values[index] = value
      }
      if (!changed) continue
      const known = groups.get(target)
      if (known) {
        if (sampled.content) dirtyGroup(known)
        else {
          fading.set(target, shared.clock)
          known.moved = true
          known.fresh = false
        }
      } else if (!sampled.content) promote(target, 'motion')
      else touch(target)
    }
    if (shared.clock - settled > 1000) {
      settled = shared.clock
      settle()
    }
  }
  return { pauseAnimations, suspendAnimations, resumeAnimations, readAnimations, scan }
}
export type AnimationSampler = ReturnType<typeof createAnimationSampler>
