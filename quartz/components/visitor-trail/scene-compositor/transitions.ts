import type { Gpu } from './gpu-textures'
import type { Scope } from './render-model'
import {
  cloneGrade,
  equalGrade,
  formatGrade,
  interpolateGrade,
  type Grade,
} from '../../../util/scene-grade'
import { trailOpeningCell } from '../scene-reveal'

// Transition clocks: the lighting grade, the theme fade and the resize reveal.
export function createTransitions(scope: Scope, gpu: Gpu) {
  const { scene, host, document, view, gl, opening, reducedMotion, coverage, shared } = scope
  const { uniforms } = gpu
  const pageThemeReveal = () => document.documentElement.dataset.homeThemeReveal === 'revealing'
  const resizeProgress = () =>
    shared.resizeFrame?.started === null || !shared.resizeFrame
      ? 0
      : Math.min(1, Math.max(0, (shared.clock - shared.resizeFrame.started) / 420))
  const themeProgress = () => shared.fade?.progress ?? 0
  const preparePresentation = () => {
    if (!host.hidden) return
    scene.dataset.trailOpening = 'pending'
    scene.dataset.trailOpeningProgress = '0'
    scene.dataset.trailPaintReady = 'false'
  }
  const gradeAt = (now: number): Grade => {
    const progress = reducedMotion.matches
      ? 1
      : Math.min(1, Math.max(0, (now - shared.gradeStarted) / 1200))
    const share = progress * progress * (3 - 2 * progress)
    return interpolateGrade(shared.gradeFrom, shared.gradeTo, share)
  }
  const applyGrade = (next: Grade) => {
    if (equalGrade(next, shared.gradeTo)) return
    const now = view.performance.now()
    shared.gradeFrom = host.hidden ? cloneGrade(next) : gradeAt(now)
    shared.gradeTo = cloneGrade(next)
    shared.gradeStarted = host.hidden ? now - 1200 : now
    preparePresentation()
    scene.dataset.sceneGrade = formatGrade(next)
  }
  // Frame-wide uniforms: the grade, the theme and resize reveals and the opening sweep.
  const frameUniforms = (grade: Grade) => {
    gl.uniformMatrix3fv(uniforms.gradeM, true, grade.m)
    gl.uniform3fv(uniforms.gradeB, grade.b)
    gl.uniform1f(uniforms.loadActive, Number(!!shared.fade && coverage.active))
    gl.uniform4f(
      uniforms.themeReveal,
      shared.fade && !pageThemeReveal() ? 1 : 0,
      themeProgress(),
      shared.fade?.origin ?? 1,
      trailOpeningCell * shared.resolution,
    )
    const origin = opening.origin()
    gl.uniform3f(
      uniforms.sweep,
      origin.x * shared.resolution,
      origin.y * shared.resolution,
      opening.radius() * shared.resolution,
    )
    gl.uniform3f(
      uniforms.reveal,
      shared.resizeFrame && shared.resizeFrame.started !== null ? 1 : 0,
      resizeProgress(),
      6 * shared.resolution,
    )
  }
  return {
    pageThemeReveal,
    resizeProgress,
    themeProgress,
    preparePresentation,
    gradeAt,
    applyGrade,
    frameUniforms,
  }
}
export type Transitions = ReturnType<typeof createTransitions>
