import type { Overlays } from './retained-presentation'
import type { SceneTree } from './scene-partition'
import { motionProperties, type Scope } from './render-model'

// Animations: paused around the opening, and sampled each frame into moved or dirtied textures.
export function createAnimationSampler(scope: Scope, sceneTree: SceneTree, overlays: Overlays) {
  const { is, shared } = scope
  const { fading, groups, promote, touch, dirtyGroup, touchDefinition, settle } = sceneTree
  const { holdResize, capturePalette } = overlays
  const pausedAnimations = new Set<Animation>()
  const pauseAnimations = (animations: Animation[]) => {
    const running = animations.filter(animation => animation.playState === 'running')
    for (const animation of running) {
      pausedAnimations.add(animation)
      animation.pause()
    }
  }
  const resumeAnimations = () => {
    const paused = [...pausedAnimations].filter(animation => animation.playState === 'paused')
    for (const animation of paused) animation.play()
    pausedAnimations.clear()
  }
  let settled = 0
  // Whether an animation changes pixels inside its target, rather than only moving or fading it.
  const paints = new WeakMap<Animation, { content: boolean; state: AnimationPlayState }>()
  const timing = new Set(['offset', 'computedOffset', 'easing', 'composite'])
  const scan = (animations: Animation[]) => {
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
    for (const animation of animations) {
      const state = animation.playState
      let sampled = paints.get(animation)
      // A stepped fade reaches its final opacity when it finishes. Sample that state once
      // before retaining it, otherwise footprints stay invisible until their textures regroup.
      if (state !== 'running' && sampled?.state === state) continue
      const effect = animation.effect
      if (!is(effect, 'KeyframeEffect') || !effect.target) continue
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
        }
        paints.set(animation, sampled)
      }
      sampled.state = state
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
  return { pauseAnimations, resumeAnimations, scan }
}
export type AnimationSampler = ReturnType<typeof createAnimationSampler>
