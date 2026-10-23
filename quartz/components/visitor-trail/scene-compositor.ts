import type { Box, Classes, Group, Scope, Surface } from './scene-compositor/render-model'
import type { SceneReveal } from './scene-reveal'
import { cloneGrade, IDENTITY_GRADE } from '../../util/scene-grade'
import { createAnimationSampler } from './scene-compositor/animation-sampler'
import { createFrameLoop } from './scene-compositor/frame-loop'
import { createGpu } from './scene-compositor/gpu-textures'
import { createMutations } from './scene-compositor/mutation-invalidation'
import { createScenePainter } from './scene-compositor/paint-scene'
import { createRaster } from './scene-compositor/raster-layers'
import { createOverlays } from './scene-compositor/retained-presentation'
import { createLoadCoverage } from './scene-compositor/reveal-coverage'
import { createSceneTree } from './scene-compositor/scene-partition'
import { createSegmentCache } from './scene-compositor/segment-cache'
import { createTransitions } from './scene-compositor/transitions'
import { subscribeSceneGrade } from './scene-lighting'

// The SVG stays the scene model: journeys, depth order, the pet and camp hints still read and
// write it. This module rasterizes that model into textures and composites them with WebGL2, so
// Blink lays the SVG out but never paints, layerizes or rasterizes its eight thousand nodes.
export function mountSceneCompositor(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  opening: SceneReveal,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('svg.trail-canvas')
  if (!view || !svg) throw new Error('The scene compositor needs the trail canvas')
  const canvas = document.createElement('canvas')
  canvas.className = 'trail-compositor'
  canvas.setAttribute('aria-hidden', 'true')
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    premultipliedAlpha: true,
    // Deferred uploads keep the last complete frame through browser presentation.
    preserveDrawingBuffer: true,
    antialias: false,
    depth: false,
    stencil: false,
  })
  if (!gl) return false
  svg.after(canvas)
  const scratch = document.createElement('canvas')
  const context = scratch.getContext('2d', { willReadFrequently: true })
  if (!context) {
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    canvas.remove()
    return false
  }

  // The trail frame builds nodes in the page document, and some of them carry the frame's classes,
  // so a type test checks both realms.
  const realms = [view, globalThis as unknown as typeof view]
  const is = <K extends keyof Classes>(value: unknown, name: K): value is Classes[K] =>
    realms.some(realm => value instanceof (realm[name] as unknown as typeof Object))
  // The scene lives in a shadow root, so its ids and stylesheets belong to that tree.
  const tree = () => scene.getRootNode() as Document | ShadowRoot
  const scope: Scope = {
    scene,
    host,
    signal,
    opening,
    document,
    view,
    svg,
    canvas,
    gl,
    is,
    tree,
    active: () => !signal.aborted && !document.hidden && scene.isConnected && !gl.isContextLost(),
    reducedMotion: view.matchMedia('(prefers-reduced-motion: reduce)'),
    size: { width: canvas.width, height: canvas.height },
    coverage: { texture: null, cells: new Uint8Array(), width: 0, height: 0, active: false },
    repainted: new Map(),
    shared: {
      resolution: Math.max(1, view.devicePixelRatio),
      clock: 0,
      palette: 0,
      revision: 0,
      flush: false,
      sizeDirty: true,
      clipVersion: 0,
      ready: false,
      firstPaint: null,
      surfaceExposed: false,
      generationPending: true,
      palettePending: false,
      paletteTransition: false,
      fade: null,
      resizeFrame: null,
      gradeFrom: cloneGrade(IDENTITY_GRADE),
      gradeTo: cloneGrade(IDENTITY_GRADE),
      gradeStarted: 0,
      initialAnimations: null,
    },
  }
  const { reducedMotion, coverage, shared } = scope
  const gpu = createGpu(gl)
  const sceneTree = createSceneTree(scope, gpu)
  const raster = createRaster(scope, gpu, sceneTree, context)
  const load = createLoadCoverage(scope, gpu, raster)
  const transitions = createTransitions(scope, gpu)
  const cache = createSegmentCache(scope, gpu)
  const painter = createScenePainter(scope, gpu, sceneTree, raster, transitions, cache)
  const overlays = createOverlays(scope, gpu, sceneTree, load, transitions, cache, painter)
  const sampler = createAnimationSampler(scope, sceneTree, overlays)
  const mutations = createMutations(scope, sceneTree, transitions, () => loop.schedule())
  const observer = new MutationObserver(mutations.record)
  const loop = createFrameLoop(
    scope,
    gpu,
    sceneTree,
    raster,
    load,
    transitions,
    overlays,
    sampler,
    mutations,
    observer,
  )
  const { textures, clipTextures, setup } = gpu
  const { groups, promoted, root } = sceneTree
  const { trimRaster } = raster
  const { applyGrade, preparePresentation } = transitions
  const { segments } = cache
  const { cancelSurfaceTrim, scheduleSurfaceTrim, endFade, endResize, presentResize, holdResize } =
    overlays
  const { pauseAnimations, resumeAnimations, readAnimations } = sampler
  const { schedule } = loop
  observer.observe(svg, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeOldValue: true,
    characterData: true,
  })
  observer.observe(scene, { attributes: true, attributeOldValue: true })
  observer.observe(document.documentElement, {
    attributes: true,
    attributeOldValue: true,
    attributeFilter: ['saved-theme'],
  })
  host.addEventListener('trail:prepare', schedule, { signal })
  host.addEventListener('trail:ready', schedule, { signal })

  scene.addEventListener(
    'trail:canopy-paint',
    event => {
      const frameElement = (event.target as Element | null)?.closest('foreignObject')
      if (!frameElement) return
      let owner = frameElement.parentElement
      while (owner && !groups.has(owner)) owner = owner.parentElement
      const item =
        owner &&
        groups
          .get(owner)
          ?.items.find(
            (item): item is Surface => item.kind === 'surface' && item.frame === frameElement,
          )
      if (!item) return
      // A painter can name the rects it changed; otherwise the whole surface counts as repainted.
      const rects = (event as CustomEvent<{ rects?: Box[] } | null>).detail?.rects
      if (!item.dirty) item.damage = []
      if (rects && item.damage) item.damage.push(...rects)
      else item.damage = null
      item.dirty = true
    },
    { signal },
  )
  document.addEventListener('visibilitychange', schedule, { signal })
  view.addEventListener('resize', schedule, { signal })
  reducedMotion.addEventListener(
    'change',
    () => {
      if (reducedMotion.matches) {
        endFade()
        shared.gradeFrom = cloneGrade(shared.gradeTo)
        shared.gradeStarted = view.performance.now() - 1200
      }
      schedule()
    },
    { signal },
  )
  subscribeSceneGrade(
    document,
    next => {
      applyGrade(next)
      schedule()
    },
    signal,
  )
  host.addEventListener('trail:resize-start', holdResize, { signal })
  host.addEventListener(
    'trail:open',
    () => {
      cancelSurfaceTrim()
      shared.surfaceExposed = true
    },
    { signal },
  )
  host.addEventListener('trail:reveal-end', schedule, { signal })
  host.addEventListener(
    'trail:before-toggle',
    () => {
      if (host.getAttribute('aria-hidden') === 'false') pauseAnimations(readAnimations())
    },
    { signal, capture: true },
  )
  host.addEventListener(
    'trail:show',
    () => {
      if (scene.dataset.trailOpening === 'complete') resumeAnimations()
      schedule()
    },
    { signal },
  )
  host.addEventListener(
    'trail:hide',
    () => {
      scheduleSurfaceTrim()
      trimRaster(true)
      endFade()
      endResize()
      shared.gradeFrom = cloneGrade(shared.gradeTo)
      shared.gradeStarted = view.performance.now() - 1200
      schedule()
    },
    { signal },
  )
  const resizeObserver = new ResizeObserver(() => {
    shared.sizeDirty = true
    preparePresentation()
    presentResize()
    root.moved = true
    schedule()
  })
  resizeObserver.observe(svg)
  resizeObserver.observe(host)
  canvas.addEventListener(
    'webglcontextlost',
    event => {
      event.preventDefault()
      cancelSurfaceTrim()
      delete scene.dataset.trailRenderer
      opening.useFallback()
      shared.ready = false
      shared.firstPaint = null
      shared.surfaceExposed = false
      endResize()
      shared.fade = null
    },
    { signal },
  )
  canvas.addEventListener(
    'webglcontextrestored',
    () => {
      textures.clear()
      // The lost context already freed these; deleting them here would raise errors.
      segments.clear()
      painter.lose()
      coverage.texture = null
      coverage.active = false
      endResize()
      shared.fade = null
      setup()
      const reset = (owner: Group) => {
        if (owner.mask) Object.assign(owner.mask, { texture: null, dirty: true, palette: -1 })
        for (const item of owner.items) {
          if (item.kind === 'group') reset(item)
          else Object.assign(item, { texture: null, dirty: true, palette: -1 })
        }
      }
      reset(root)
      opening.useCompositor()
      scene.dataset.trailRenderer = 'webgl'
      schedule()
    },
    { signal },
  )
  setup()
  opening.useCompositor()
  scene.dataset.trailRenderer = 'webgl'
  schedule()
  signal.addEventListener(
    'abort',
    () => {
      loop.stop()
      cancelSurfaceTrim()
      observer.disconnect()
      resizeObserver.disconnect()
      endResize()
      if (shared.firstPaint) gl.deleteSync(shared.firstPaint)
      shared.firstPaint = null
      for (const texture of textures) gl.deleteTexture(texture)
      textures.clear()
      groups.clear()
      promoted.clear()
      clipTextures.clear()
      trimRaster(true)
      gl.deleteTexture(gpu.blank)
      gl.deleteBuffer(gpu.vertexBuffer)
      painter.dispose()
      gl.deleteProgram(gpu.program)
      canvas.width = 1
      canvas.height = 1
      gl.getExtension('WEBGL_lose_context')?.loseContext()
      delete scene.dataset.trailRenderer
      canvas.remove()
    },
    { once: true },
  )
  return true
}
