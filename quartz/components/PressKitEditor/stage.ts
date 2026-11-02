import type { CardLayer, LayerBounds } from '../scripts/press-kit-layers'
import { readCardBackground } from '../scripts/press-kit-background'
import { paintRulers } from '../scripts/press-kit-rulers'
import { colorImage } from './color'
import {
  bounds,
  composition,
  copy,
  layers,
  placement,
  placeSelection,
  remember,
  selection,
  type Composition,
  type EditorState,
  type LayerRaster,
  type Layout,
} from './layout'

export const positionOverlay = (state: EditorState) => {
  const scene = state.scene,
    parent = state.canvas.parentElement
  if (!scene || !parent) return
  const rect = state.canvas.getBoundingClientRect(),
    host = parent.getBoundingClientRect()
  state.overlay.style.cssText = `left:${rect.left - host.left}px;top:${rect.top - host.top}px;width:${rect.width}px;height:${rect.height}px`
  const position = (node: HTMLElement, box: LayerBounds) => {
    node.style.cssText = `left:${(box.x / scene.width) * 100}%;top:${(box.y / scene.height) * 100}%;width:${(box.width / scene.width) * 100}%;height:${(box.height / scene.height) * 100}%`
    const handleSize = window.matchMedia('(pointer: coarse)').matches ? 44 : 24
    node.toggleAttribute(
      'data-compact-x',
      (box.height * rect.height) / scene.height < handleSize * 2,
    )
    node.toggleAttribute('data-compact-y', (box.width * rect.width) / scene.width < handleSize * 2)
  }
  for (const node of state.overlay.querySelectorAll<HTMLElement>('[data-editor-layer]')) {
    const layer = scene.layers.find(item => item.id === node.dataset.editorLayer)
    if (!layer) continue
    const p = placement(state, layer)
    position(node, {
      x: p.x,
      y: p.y,
      width: layer.width * p.scaleX,
      height: layer.height * p.scaleY,
    })
    node.dataset.x = String(p.x)
    node.dataset.y = String(p.y)
    node.dataset.scaleX = String(p.scaleX)
    node.dataset.scaleY = String(p.scaleY)
  }
  const group = state.overlay.querySelector<HTMLElement>('[data-editor-selection]'),
    selBounds = bounds(selection(state))
  if (group && selBounds) position(group, selBounds)
  const marquee = state.overlay.querySelector<HTMLElement>('.press-kit-editor-marquee')
  if (marquee && state.gesture?.kind === 'marquee') position(marquee, state.gesture.bounds)
  if (!state.rulers.hidden) {
    state.rulers.style.cssText = `left:${rect.left - host.left - 24}px;top:${rect.top - host.top - 24}px`
    paintRulers(
      state.rulers,
      rect.width,
      rect.height,
      scene.width,
      scene.height,
      selBounds,
      state.pointer,
    )
    const coordinates = state.controls.querySelector('output')
    if (coordinates)
      coordinates.textContent = selBounds
        ? `x ${Math.round(selBounds.x)} · y ${Math.round(selBounds.y)} px`
        : state.pointer
          ? `x ${Math.round(state.pointer.x)} · y ${Math.round(state.pointer.y)} px`
          : `${scene.width} × ${scene.height} px`
  }
}

const layerImage = (
  state: EditorState,
  layer: CardLayer,
  width: number,
  height: number,
): Pick<LayerRaster, 'image' | 'width' | 'height'> => {
  const renderer = layer.renderer
  if (!renderer || (width === layer.width && height === layer.height))
    return { image: layer.image, width, height }
  let raster = state.rasters.get(layer)
  if (!raster) {
    raster = { width: layer.width, height: layer.height, image: layer.image }
    state.rasters.set(layer, raster)
  }
  if (
    !raster.pending &&
    (raster.width !== width || raster.height !== height) &&
    (raster.failed?.width !== width || raster.failed?.height !== height)
  ) {
    const target = raster
    const scene = state.scene
    target.failed = undefined
    target.pending = (async () => {
      const image = await renderer.render(width, height)
      if (state.signal.aborted || state.scene !== scene) return
      target.image = image
      target.width = width
      target.height = height
    })()
      .catch(() => {
        target.failed = { width, height }
      })
      .finally(() => {
        target.pending = undefined
        if (state.signal.aborted || state.scene !== scene) return
        state.changed()
        // Read the latest placement after decoding; one pending render per chart coalesces drags.
        compose(state)
        state.syncView()
      })
  }
  return raster
}

export const redrawFailed = (state: EditorState) =>
  layers(state).some(layer => {
    const p = placement(state, layer)
    const failed = state.rasters.get(layer)?.failed
    return (
      !p.hidden &&
      failed?.width === layer.width * p.scaleX &&
      failed?.height === layer.height * p.scaleY
    )
  })

export const compose = (state: EditorState) => {
  const scene = state.scene
  if (!scene || !state.available || state.signal.aborted) return
  const ctx = state.canvas.getContext('2d')
  if (!ctx) return
  const width = Math.round(scene.width * scene.scale),
    height = Math.round(scene.height * scene.scale)
  if (state.canvas.width !== width) state.canvas.width = width
  if (state.canvas.height !== height) state.canvas.height = height
  ctx.setTransform(scene.scale, 0, 0, scene.scale, 0, 0)
  ctx.clearRect(0, 0, scene.width, scene.height)
  if (scene.background) {
    ctx.fillStyle = scene.background
    ctx.fillRect(0, 0, scene.width, scene.height)
  }
  const bg = state.draft.value.background
  if (bg) {
    const ratio =
      (bg.fit === 'cover' ? Math.max : Math.min)(
        scene.width / bg.image.naturalWidth,
        scene.height / bg.image.naturalHeight,
      ) * bg.zoom
    const bw = bg.image.naturalWidth * ratio,
      bh = bg.image.naturalHeight * ratio
    ctx.globalAlpha = bg.opacity
    ctx.drawImage(bg.image, (scene.width - bw) * bg.x, (scene.height - bh) * bg.y, bw, bh)
    ctx.globalAlpha = 1
  }
  for (const layer of layers(state)) {
    const p = placement(state, layer)
    if (!p.hidden) {
      const lw = layer.width * p.scaleX,
        lh = layer.height * p.scaleY
      const raster = layerImage(state, layer, lw, lh)
      ctx.save()
      ctx.beginPath()
      ctx.rect(p.x, p.y, lw, lh)
      ctx.clip()
      ctx.drawImage(colorImage(state, layer, raster.image), p.x, p.y, raster.width, raster.height)
      ctx.restore()
    }
  }
  state.details.setAttribute(
    'aria-busy',
    String([...state.rasters.values()].some(raster => raster.pending)),
  )
  positionOverlay(state)
}

export const queue = (state: EditorState) => {
  state.changed()
  if (!state.frame)
    state.frame = requestAnimationFrame(() => {
      state.frame = 0
      compose(state)
    })
}

export const flushStage = async (state: EditorState) => {
  await state.pendingImage
  while (!state.signal.aborted) {
    if (state.frame) {
      cancelAnimationFrame(state.frame)
      state.frame = 0
    }
    compose(state)
    const pending = [...state.rasters.values()].flatMap(raster =>
      raster.pending ? [raster.pending] : [],
    )
    if (!pending.length) return !redrawFailed(state)
    await Promise.all(pending)
  }
  return false
}

export const edit = (state: EditorState, mutate: (comp: Composition, doc: Layout) => void) => {
  const before = copy(state.draft.value)
  mutate(composition(state), state.draft.value)
  remember(state, before)
  const doc = state.draft.value
  if (
    before.metricGap !== doc.metricGap ||
    before.railTextScale !== doc.railTextScale ||
    before.title !== doc.title
  )
    state.requestScene()
  else queue(state)
  state.syncView()
}

export const editBounds = (state: EditorState, after: LayerBounds) => {
  const items = selection(state),
    before = bounds(items)
  if (before) edit(state, layout => placeSelection(layout, items, before, after))
}

export const resizeRailText = (state: EditorState, scale: number) => {
  if (scale === state.draft.value.railTextScale) return
  state.railAnchors.clear()
  const allLayers = layers(state)
  const blocks = new Set(
    allLayers
      .filter(layer => layer.id.includes('/stat/'))
      .map(layer => layer.id.split('/stat/')[0]),
  )
  const comp = composition(state)
  for (const block of blocks) {
    const items = allLayers
      .filter(layer => layer.id.startsWith(`${block}/stat/`) && !placement(state, layer).hidden)
      .map(layer => ({ layer, placement: placement(state, layer) }))
    if (!items.some(({ layer }) => comp.placements[layer.id])) continue
    const box = bounds(items)
    if (box) state.railAnchors.set(block, { right: box.x + box.width, bottom: box.y + box.height })
  }
  edit(state, (_, doc) => {
    doc.railTextScale = scale
  })
}

export const importImage = (state: EditorState, file?: File) => {
  if (!file) return
  const revision = ++state.revision
  state.loading = true
  state.error = ''
  state.syncView()
  state.pendingImage = (async () => {
    const image = await readCardBackground(file)
    if (state.signal.aborted || revision !== state.revision) return
    edit(state, (_, doc) => {
      doc.background = { image, name: file.name, fit: 'cover', zoom: 1, x: 0.5, y: 0.5, opacity: 1 }
    })
  })()
    .catch(error => {
      if (revision === state.revision)
        state.error = error instanceof Error ? error.message : 'image illisible'
    })
    .finally(() => {
      if (revision !== state.revision || state.signal.aborted) return
      state.loading = false
      state.syncView()
    })
}
