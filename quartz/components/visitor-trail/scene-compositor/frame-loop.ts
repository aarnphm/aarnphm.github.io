import type { AnimationSampler } from './animation-sampler'
import type { Gpu } from './gpu-textures'
import type { Mutations } from './mutation-invalidation'
import type { Raster } from './raster-layers'
import type { Overlays } from './retained-presentation'
import type { LoadCoverage } from './reveal-coverage'
import type { SceneTree } from './scene-partition'
import type { Transitions } from './transitions'
import { ctm, multiply, scaleOf, scaling } from './affine-geometry'
import { chunkVisible } from './draw-list'
import { ambientLayers, type Group, type Layer, type Run, type Scope } from './render-model'

// Frame loop: each frame reads the scene's records and animations, repaints stale textures within
// a time budget, then composites and presents.
export function createFrameLoop(
  scope: Scope,
  gpu: Gpu,
  sceneTree: SceneTree,
  raster: Raster,
  load: LoadCoverage,
  transitions: Transitions,
  overlays: Overlays,
  sampler: AnimationSampler,
  mutations: Mutations,
  observer: MutationObserver,
) {
  const {
    scene,
    host,
    document,
    view,
    svg,
    gl,
    opening,
    reducedMotion,
    coverage,
    repainted,
    active,
    shared,
  } = scope
  const { upload } = gpu
  const { groups, root, partition, split } = sceneTree
  const { trimRaster, rasterize, rasterizeMask } = raster
  const { paintCoverage, paintPriority } = load
  const { resizeProgress } = transitions
  const { composite, overlay, overlayResize, advanceFade, endResize, resize, scheduleSurfaceTrim } =
    overlays
  const { pauseAnimations, resumeAnimations, scan } = sampler
  const { readSelectors, record } = mutations
  let frame = 0
  let timer = 0
  let continuation = 0
  const uploads = new view.MessageChannel()
  uploads.port1.onmessage = (event: MessageEvent<number>) => {
    if (event.data === continuation) tick()
  }
  const cancelContinuation = () => {
    continuation += 1
    view.clearTimeout(timer)
    timer = 0
  }
  // Groups stand for their mask here.
  const pending: (Run | Group)[] = []
  const stale = (layer: Layer, scale: number) => {
    const due =
      layer.dirty ||
      layer.palette !== shared.palette ||
      (layer.scale > 0 && Math.abs(scale / layer.scale - 1) > 0.2)
    if (due && !layer.waiting) layer.waiting = shared.clock
    return due
  }
  const prepare = (owner: Group, moved: boolean) => {
    if (!chunkVisible(scope, owner)) return
    if (owner.structure) partition(owner)
    if (!owner.fresh) {
      owner.fresh = true
      owner.shown = owner.element === svg || owner.style.display !== 'none'
      owner.opacity = owner.element === svg ? 1 : Number(owner.style.opacity)
      for (const item of owner.items)
        if (item.kind === 'surface')
          item.shown = item.style.display !== 'none' && item.style.visibility === 'visible'
    }
    if (!owner.shown) return
    moved ||= owner.moved
    if (moved) {
      owner.moved = false
      owner.matrix = ctm(owner.element)
      const screen = multiply(scaling(shared.resolution), owner.matrix)
      if (screen.some((value, index) => value !== owner.screen[index])) {
        owner.screen = screen
        if (owner.mask) owner.mask.placed = null
        for (const item of owner.items) if (item.kind !== 'group') item.placed = null
      }
    }
    const scale = scaleOf(owner.matrix) * shared.resolution
    if (owner.mask && stale(owner.mask, scale)) pending.push(owner)
    for (const item of owner.items) {
      if (item.kind === 'group') prepare(item, moved)
      else if (item.kind === 'surface') {
        if (!item.dirty && item.texture) continue
        item.dirty = false
        if (item.canvas.width && item.canvas.height) {
          const damage = item.texture ? item.damage : null
          item.texture = upload(item.texture, item.canvas)
          const known = repainted.get(item)
          repainted.set(item, damage && known !== null ? [...(known ?? []), ...damage] : null)
        }
        item.damage = null
      } else if (stale(item, scale)) pending.push(item)
    }
  }
  const tick = () => {
    frame = 0
    if (!active()) return
    if (shared.firstPaint) {
      if (gl.clientWaitSync(shared.firstPaint, 0, 0) === gl.TIMEOUT_EXPIRED) {
        frame = view.requestAnimationFrame(tick)
        return
      }
      gl.deleteSync(shared.firstPaint)
      shared.firstPaint = null
      if (
        !shared.sizeDirty &&
        !shared.flush &&
        !shared.palettePending &&
        !shared.paletteTransition &&
        !shared.generationPending
      )
        opening.present()
      if (scene.dataset.trailOpening === 'complete') {
        if (!host.hidden && scene.dataset.trailReveal !== 'revealing') resumeAnimations()
        trimRaster(Boolean(host.hidden))
        scheduleSurfaceTrim()
        schedule()
        return
      }
    }
    if (scene.dataset.trailRendered !== 'true') return
    // The native reveal can present the retained buffer without resampling the live SVG.
    if (
      scene.dataset.trailReveal !== undefined &&
      scene.dataset.trailOpening === 'complete' &&
      !shared.sizeDirty &&
      !shared.flush &&
      !shared.palettePending &&
      !shared.paletteTransition &&
      !shared.fade &&
      !shared.resizeFrame
    )
      return
    frame = -1
    // Initial uploads yield without the nested-timer delay. Once motion starts, queue after
    // this paint so the next frame reads the journeys' writes in one style and layout pass.
    if (!host.hidden || scene.dataset.trailPaintReady !== 'true') {
      if (!host.hidden && scene.dataset.trailOpening !== 'complete')
        uploads.port2.postMessage(++continuation)
      else
        timer = view.setTimeout(() => {
          timer = 0
          if (scene.dataset.trailOpening === 'complete') frame = view.requestAnimationFrame(tick)
          else tick()
        }, 0)
    } else frame = 0
    shared.clock = view.performance.now()
    record(observer.takeRecords())
    // Resizes retain their frame until camp placement settles.
    if (host.dataset.trailResizing === 'true' || scene.dataset.campPositioned !== 'true') return
    readSelectors()
    resize()
    const animations =
      scene.dataset.trailOpening === 'complete'
        ? svg.getAnimations({ subtree: true })
        : (shared.initialAnimations ??= svg.getAnimations({ subtree: true }))
    if (scene.dataset.trailOpening !== 'complete' || host.hidden) pauseAnimations(animations)
    scan(animations)
    split()
    pending.length = 0
    prepare(root, false)
    // Spend the saved layout work on bounded palette uploads during the shared sweep.
    const budget =
      scene.dataset.trailOpening !== 'complete' ? (host.hidden ? 10 : 64) : shared.fade ? 10 : 4
    const unfinished: (Run | Group)[] = []
    let deferred = 0
    let holes = 0
    let replacements = 0
    const unpainted = (target: Run | Group) =>
      (target.kind === 'group' ? (target.mask?.palette ?? -1) : target.palette) < 0
    const outdated = (target: Run | Group) => {
      const layer = target.kind === 'group' ? target.mask : target
      return layer?.palette !== shared.palette || layer.revision !== shared.revision
    }
    // A scene switch already holds a complete frame. Keep its camp-first order even when
    // winter clothing or crowns introduce new textures higher up the valley.
    // Initial presentation waits for every texture, so only subsequent updates need a
    // spatial repaint order. Avoid rescanning the forest's ancestors for each initial batch.
    if (scene.dataset.trailOpening === 'complete') {
      const priorities = new Map(pending.map(target => [target, paintPriority(target)]))
      pending.sort(
        (a, b) =>
          Number(outdated(b)) - Number(outdated(a)) ||
          (shared.fade ? 0 : Number(unpainted(b)) - Number(unpainted(a))) ||
          (priorities.get(a) ?? 0) - (priorities.get(b) ?? 0) ||
          (a.kind === 'group' ? (a.mask?.waiting ?? 0) : a.waiting) -
            (b.kind === 'group' ? (b.mask?.waiting ?? 0) : b.waiting),
      )
    }
    // Live updates share their budget with layout and preparation. Always advance one
    // texture, even when those reads consume the budget, so replacement cannot starve.
    const painting =
      scene.dataset.trailOpening === 'complete' ? shared.clock : view.performance.now()
    let paintedLayers = 0
    for (const target of pending) {
      const owner = target.kind === 'group' ? target : target.owner
      const layer = target.kind === 'group' ? target.mask : target
      // Weather and tracks resume after a palette transition finishes.
      if (shared.fade && owner.element.closest(ambientLayers)) continue
      const painted = !unpainted(target)
      if (paintedLayers > 0 && view.performance.now() - painting > budget) {
        if (layer && (layer.palette !== shared.palette || layer.revision !== shared.revision))
          unfinished.push(target)
        if ((target.kind === 'group' ? target.mask?.revision : target.revision) !== shared.revision)
          replacements += 1
        if (!painted) holes += 1
        else if (
          (target.kind === 'group' ? target.mask?.palette : target.palette) !== shared.palette
        )
          deferred += 1
        continue
      }
      if (target.kind === 'group') rasterizeMask(target)
      else
        rasterize(
          target,
          target.owner.element,
          target.nodes,
          target.nodes[0] === target.owner.element,
        )
      paintedLayers += 1
    }
    // Ongoing motion can dirty a replacement again; it must not hold the completed scene forever.
    const painted =
      holes || deferred || replacements || scene.dataset.trailRendered !== 'true' ? 'false' : 'true'
    if (scene.dataset.trailPaintReady !== painted) scene.dataset.trailPaintReady = painted
    // Starting motion can split a retained layer into new textures. Keep the last complete
    // canvas until those textures exist, so budgeted uploads cannot flash empty patches.
    if (
      (shared.ready && holes && !shared.fade) ||
      (shared.resizeFrame && (holes || (shared.ready && shared.flush && replacements)))
    ) {
      return
    }
    if (shared.fade) paintCoverage(unfinished)
    else {
      coverage.active = false
      scene.dataset.trailLoadProgress = painted === 'true' ? '1.000' : '0.000'
    }
    shared.flush = false
    advanceFade()
    if (shared.resizeFrame) {
      if (reducedMotion.matches || resizeProgress() >= 1) endResize()
      else if (shared.resizeFrame.started === null) {
        shared.resizeFrame.started = shared.clock
        scene.dataset.trailResizePhase = 'revealing'
      }
    }
    const ambient = String(
      scene.dataset.trailRendered === 'true' &&
        (scene.dataset.trailOpening === 'complete' || painted === 'true') &&
        !shared.fade &&
        !shared.resizeFrame,
    )
    if (scene.dataset.trailAmbientReady !== ambient) scene.dataset.trailAmbientReady = ambient
    // The first visible frame uses the finished scene, with every texture at its final scale.
    if (scene.dataset.trailOpening !== 'complete' && painted !== 'true') return
    composite()
    overlay()
    overlayResize()
    if (!host.hidden) shared.surfaceExposed = true
    if (painted === 'true' && !shared.fade && !shared.resizeFrame)
      scene.dataset.trailPaintTheme =
        document.documentElement.getAttribute('saved-theme') ?? 'light'
    if (!shared.ready) {
      shared.ready = true
      scene.dataset.trailRenderer = 'webgl'
      for (const owner of groups.values()) owner.fresh = false
    }
    const presenting = scene.dataset.trailOpening !== 'complete'
    if (presenting) {
      // Uploaded textures can still be queued on the GPU when the native reveal starts.
      // Poll between frames so its first draw finishes before the 600 ms sweep begins.
      shared.firstPaint = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
      if (shared.firstPaint) {
        gl.flush()
        cancelContinuation()
        frame = view.requestAnimationFrame(tick)
        return
      }
    }
    opening.present()
    if (presenting && !host.hidden && scene.dataset.trailReveal !== 'revealing') resumeAnimations()
    trimRaster(host.hidden && painted === 'true')
    scheduleSurfaceTrim()
    if (painted !== 'true') schedule()
  }
  const schedule = () => {
    if (!frame && active()) frame = view.requestAnimationFrame(tick)
  }
  const stop = () => {
    view.cancelAnimationFrame(frame)
    cancelContinuation()
    uploads.port1.close()
    uploads.port2.close()
  }
  return { schedule, stop }
}
