import type { Gpu } from './gpu-textures'
import type { ScenePainter } from './paint-scene'
import type { Scope } from './render-model'
import type { LoadCoverage } from './reveal-coverage'
import type { SceneTree } from './scene-partition'
import type { SegmentCache } from './segment-cache'
import type { Transitions } from './transitions'
import { IDENTITY_GRADE } from '../../../util/scene-grade'
import { trailOpeningCell, trailThemeDuration } from '../scene-reveal'
import { none, place } from './affine-geometry'

// Presentation: the canvas surface, the theme crossfade and the held resize frame drawn over
// the scene from copies of the retained canvas, and the drawing-buffer trim while hidden.
export function createOverlays(
  scope: Scope,
  gpu: Gpu,
  sceneTree: SceneTree,
  load: LoadCoverage,
  transitions: Transitions,
  cache: SegmentCache,
  painter: ScenePainter,
) {
  const {
    scene,
    host,
    view,
    svg,
    canvas,
    gl,
    opening,
    reducedMotion,
    size,
    coverage,
    active,
    shared,
  } = scope
  const { uniforms, textures, filter, release } = gpu
  const { root, repaint } = sceneTree
  const { coverageCeiling } = load
  const { pageThemeReveal, themeProgress, resizeProgress, preparePresentation } = transitions
  const { dropSegments } = cache
  const { paintScene } = painter
  let surfaceTimer = 0
  const surface = (width: number, height: number, clear = true) => {
    if (canvas.width !== width || canvas.height !== height) shared.clipVersion += 1
    if (canvas.width !== width) canvas.width = width
    if (canvas.height !== height) canvas.height = height
    const cssWidth = `${width / shared.resolution}px`
    const cssHeight = `${height / shared.resolution}px`
    if (canvas.style.width !== cssWidth) canvas.style.width = cssWidth
    if (canvas.style.height !== cssHeight) canvas.style.height = cssHeight
    gl.viewport(0, 0, canvas.width, canvas.height)
    gl.clearColor(0, 0, 0, 0)
    if (clear) gl.clear(gl.COLOR_BUFFER_BIT)
    gl.uniform2f(uniforms.viewport, canvas.width, canvas.height)
    gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
  }
  const composite = (width = size.width, height = size.height) => {
    // The scene clears what it repaints; the retained canvas keeps the rest.
    surface(width, height, false)
    paintScene()
  }
  const drawCopy = (
    texture: WebGLTexture,
    alpha: number,
    {
      width = canvas.width,
      height = canvas.height,
      left = 0,
      mirror = false,
      reveal = false,
      theme = false,
    }: {
      width?: number
      height?: number
      left?: number
      mirror?: boolean
      reveal?: boolean
      theme?: boolean
    } = {},
  ) => {
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, gpu.blank)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, gpu.blank)
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, coverage.texture ?? gpu.blank)
    gl.uniform1f(uniforms.loadActive, Number(theme && coverage.active))
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1f(uniforms.alpha, alpha)
    gl.uniform3f(uniforms.reveal, reveal ? 2 : 0, resizeProgress(), 6 * shared.resolution)
    gl.uniform4f(
      uniforms.themeReveal,
      theme ? 2 : -1,
      themeProgress(),
      shared.fade?.origin ?? 1,
      trailOpeningCell * shared.resolution,
    )
    // Theme and resize copies already contain the lighting of their captured frame.
    gl.uniformMatrix3fv(uniforms.gradeM, true, IDENTITY_GRADE.m)
    gl.uniform3fv(uniforms.gradeB, IDENTITY_GRADE.b)
    gl.uniformMatrix3fv(uniforms.maskPlace, false, none)
    // The copy holds the drawing buffer bottom row first, so the quad is flipped to match.
    gl.uniformMatrix3fv(
      uniforms.place,
      false,
      place([mirror ? -width : width, 0, 0, -height, mirror ? left + width : left, height]),
    )
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
  const trimSurface = () => {
    surfaceTimer = 0
    if (
      !shared.surfaceExposed ||
      !host.hidden ||
      !shared.ready ||
      !active() ||
      canvas.width > gpu.maxTexture ||
      canvas.height > gpu.maxTexture
    )
      return
    // The retained pixels come back unchanged below, so only the segment textures are released.
    dropSegments()
    const texture = gl.createTexture()
    if (!texture) return
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    if (gl.getError() !== gl.NO_ERROR) {
      gl.deleteTexture(texture)
      return
    }
    textures.add(texture)
    const { width, height } = canvas
    // Release pooled browser buffers, then restore while hidden so reopening needs no copy.
    canvas.width = 1
    canvas.height = 1
    // Restoring the same pixels does not invalidate full-resolution clip textures.
    canvas.width = width
    canvas.height = height
    gl.viewport(0, 0, canvas.width, canvas.height)
    gl.uniform2f(uniforms.viewport, canvas.width, canvas.height)
    gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    drawCopy(texture, 1)
    release(texture)
    shared.surfaceExposed = false
  }
  const cancelSurfaceTrim = () => {
    view.clearTimeout(surfaceTimer)
    surfaceTimer = 0
  }
  const scheduleSurfaceTrim = () => {
    if (!host.hidden || !shared.surfaceExposed) return
    cancelSurfaceTrim()
    // Finish pending hidden paints before releasing the browser's drawing-buffer pool.
    surfaceTimer = view.setTimeout(trimSurface, 250)
  }
  const endFade = () => {
    if (!shared.fade) return
    release(shared.fade.texture)
    shared.fade = null
    scene.dataset.trailThemePhase = 'complete'
    scene.dataset.trailThemeProgress = '1'
  }
  const capturePalette = () => {
    scene.dataset.trailAmbientReady = 'false'
    if (
      host.hidden ||
      !shared.ready ||
      reducedMotion.matches ||
      scene.dataset.trailOpening !== 'complete'
    ) {
      endFade()
      return
    }
    const texture = gl.createTexture()
    if (!texture) return
    // The retained buffer includes any interrupted pixel reveal. Copy it before replacing
    // textures or releasing that reveal, so a reversal starts at the actual visible frame.
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    textures.add(texture)
    endFade()
    shared.fade = {
      texture,
      started: Number(scene.dataset.trailThemeStarted) || shared.clock,
      updated: shared.clock,
      progress: 0,
      origin: opening.radius() * shared.resolution,
      scrollX: view.scrollX,
      scrollY: view.scrollY,
    }
    scene.dataset.trailThemePhase = 'revealing'
    scene.dataset.trailThemeProgress = '0'
  }
  const advanceFade = () => {
    if (!shared.fade) return
    const duration = trailThemeDuration
    const elapsed = Math.min(1, Math.max(0, (shared.clock - shared.fade.started) / duration))
    // Preparation and reveal share one deadline. Recover time spent behind an upload
    // without jumping the frontier across the scene after a delayed frame.
    const advance = Math.min(32, Math.max(0, shared.clock - shared.fade.updated)) / (duration / 2)
    const progress = Math.max(
      shared.fade.progress,
      Math.min(elapsed, coverageCeiling(), shared.fade.progress + advance),
    )
    shared.fade.updated = shared.clock
    shared.fade.progress = progress
    scene.dataset.trailThemeProgress = progress.toFixed(3)
    if (progress >= 1 && !coverage.active) endFade()
  }
  const overlay = () => {
    // The homepage snapshot masks the UI and scene together using this fade's progress.
    if (shared.fade && !pageThemeReveal()) drawCopy(shared.fade.texture, 1, { theme: true })
  }
  const endResize = () => {
    if (shared.resizeFrame) release(shared.resizeFrame.texture)
    shared.resizeFrame = null
    delete scene.dataset.trailResizePhase
  }
  const overlayResize = () => {
    if (!shared.resizeFrame) return
    // Reflect the held edge into newly exposed space. Every cell keeps its native dimensions;
    // the generated landscape replaces this temporary fill when its textures are complete.
    for (let left = 0, tile = 0; left < canvas.width; left += shared.resizeFrame.width, tile++)
      drawCopy(shared.resizeFrame.texture, 1, {
        width: shared.resizeFrame.width,
        height: shared.resizeFrame.height,
        left,
        mirror: tile % 2 !== 0,
        reveal: shared.resizeFrame.started !== null,
      })
  }
  const presentResize = () => {
    if (!shared.resizeFrame || shared.resizeFrame.started !== null) return
    // Grow in buckets and clip on compression, avoiding a drawing-buffer allocation on each
    // pixel of a drag. The committed frame returns to the exact settled dimensions.
    const width =
      host.clientWidth * shared.resolution > canvas.width
        ? Math.ceil(host.clientWidth / 256) * 256 * shared.resolution
        : canvas.width
    const height = Math.max(canvas.height, Math.round(host.clientHeight * shared.resolution))
    surface(width, height)
    overlayResize()
  }
  const holdResize = () => {
    scene.dataset.trailAmbientReady = 'false'
    if (host.hidden) {
      preparePresentation()
      endResize()
      return
    }
    if (!shared.ready || !active()) return
    if (shared.resizeFrame?.started === null) {
      presentResize()
      return
    }
    if (canvas.width > gpu.maxTexture || canvas.height > gpu.maxTexture) return
    shared.clock = view.performance.now()
    // Recompose the cached frame before geometry changes, including active theme overlays.
    composite(canvas.width, canvas.height)
    overlay()
    overlayResize()
    const texture = shared.resizeFrame?.texture ?? gl.createTexture()
    if (!texture) return
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, canvas.width, canvas.height, 0)
    filter()
    textures.add(texture)
    endFade()
    shared.resizeFrame = { texture, width: canvas.width, height: canvas.height, started: null }
    scene.dataset.trailResizePhase = 'holding'
    presentResize()
  }
  const resize = () => {
    const nextResolution = Math.max(1, view.devicePixelRatio)
    if (!shared.sizeDirty && shared.resolution === nextResolution) return
    shared.sizeDirty = false
    if (shared.resolution !== nextResolution) {
      shared.resolution = nextResolution
      repaint()
    }
    const width = Math.max(1, Math.round(svg.clientWidth * shared.resolution))
    const height = Math.max(1, Math.round(svg.clientHeight * shared.resolution))
    if (size.width === width && size.height === height) return
    size.width = width
    size.height = height
    shared.clipVersion += 1
    endFade()
    root.moved = true
  }
  return {
    composite,
    cancelSurfaceTrim,
    scheduleSurfaceTrim,
    endFade,
    capturePalette,
    advanceFade,
    overlay,
    endResize,
    overlayResize,
    presentResize,
    holdResize,
    resize,
  }
}
export type Overlays = ReturnType<typeof createOverlays>
