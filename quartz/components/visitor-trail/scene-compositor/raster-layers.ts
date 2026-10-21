import type { Gpu } from './gpu-textures'
import type { SceneTree } from './scene-partition'
import {
  ctm,
  invert,
  multiply,
  scaleOf,
  scaling,
  translation,
  type Matrix,
} from './affine-geometry'
import { padding, type Box, type Clip, type Group, type Run, type Scope } from './render-model'
import { createSvgPaint } from './svg-painter'

// Raster: draws runs, masks and clip chains into a staging canvas and uploads them.
export function createRaster(
  scope: Scope,
  gpu: Gpu,
  sceneTree: SceneTree,
  context: CanvasRenderingContext2D,
) {
  const { document, view, canvas, shared } = scope
  const { clipTextures, upload, release } = gpu
  const { clipVersions, boxOf, localBox } = sceneTree
  // Reuse CPU staging across uploads, then shed a large scene repaint's allocation once
  // recent paints only need small sprites. Durable pixels live in the WebGL textures.
  let scratch = context.canvas
  let paint = context
  const largeRaster = { canvas: scratch, paint }
  const smallCanvas = document.createElement('canvas')
  const smallPaint = smallCanvas.getContext('2d', { willReadFrequently: true })
  let rasterWidth = 1
  let rasterHeight = 1
  let rasterSince = view.performance.now()
  const trimRaster = (release = false) => {
    const now = view.performance.now()
    if (!release && now - rasterSince < 1000) return
    const width = release ? 1 : rasterWidth
    const height = release ? 1 : rasterHeight
    if (release || largeRaster.canvas.width * largeRaster.canvas.height > 2 * width * height) {
      largeRaster.canvas.width = width
      largeRaster.canvas.height = height
    }
    if (release) {
      smallCanvas.width = 1
      smallCanvas.height = 1
    }
    rasterWidth = 1
    rasterHeight = 1
    rasterSince = now
  }
  const clearRaster = (width: number, height: number) => {
    // Small uploads keep their own staging surface instead of snapshotting the scenery canvas.
    const small = width <= 512 && height <= 512 && smallPaint !== null
    scratch = small ? smallCanvas : largeRaster.canvas
    paint = small && smallPaint ? smallPaint : largeRaster.paint
    const nextWidth = Math.ceil(width / 256) * 256
    const nextHeight = Math.ceil(height / 256) * 256
    rasterWidth = Math.max(rasterWidth, nextWidth)
    rasterHeight = Math.max(rasterHeight, nextHeight)
    if (scratch.width < width) scratch.width = nextWidth
    if (scratch.height < height) scratch.height = nextHeight
    paint.setTransform(1, 0, 0, 1, 0, 0)
    paint.clearRect(0, 0, width, height)
  }
  const { render, clipPathOf } = createSvgPaint(scope, sceneTree, () => paint)
  const bounds = (nodes: Element[], space: Matrix, self: boolean) => {
    const toSpace = invert(space)
    let total: Box | null = null
    for (const node of nodes) {
      const box = boxOf(node, toSpace, self)
      if (!box) continue
      total = total
        ? {
            left: Math.min(total.left, box.left),
            top: Math.min(total.top, box.top),
            right: Math.max(total.right, box.right),
            bottom: Math.max(total.bottom, box.bottom),
          }
        : box
    }
    return total
  }
  const rasterize = (target: Run, space: SVGGraphicsElement, nodes: Element[], self: boolean) => {
    const spaceMatrix = space === target.owner.element ? target.owner.matrix : ctm(space)
    const sceneScale = scaleOf(spaceMatrix) * shared.resolution
    // A palette change keeps the geometry and texel allocation from the previous paint.
    const recolor = !target.dirty && target.width > 0 && target.scale === sceneScale
    const box = recolor ? null : (target.box ?? bounds(nodes, spaceMatrix, self))
    if (box) target.box = box
    target.dirty = false
    target.waiting = 0
    target.palette = shared.palette
    target.revision = shared.revision
    const scale = Math.max(0.25, sceneScale)
    target.scale = sceneScale
    if (!recolor && !box) {
      release(target.texture)
      target.texture = null
      return
    }
    if (box) {
      const margin = padding + 2 / scale
      target.width = Math.min(
        gpu.maxTexture,
        Math.ceil((box.right - box.left + margin * 2) * scale),
      )
      target.height = Math.min(
        gpu.maxTexture,
        Math.ceil((box.bottom - box.top + margin * 2) * scale),
      )
      target.base = multiply(scaling(scale), translation(margin - box.left, margin - box.top))
      target.placed = null
    }
    const { width, height } = target
    if (width <= 0 || height <= 0) return
    clearRaster(width, height)
    const toTexels = multiply(target.base, invert(spaceMatrix))
    let painted = false
    for (const node of nodes) painted = render(node, target.base, 1, toTexels, self) || painted
    if (!painted) {
      release(target.texture)
      target.texture = null
      return
    }
    target.texture = upload(target.texture, scratch, false, width, height)
  }
  const rasterizeMask = (owner: Group) => {
    const mask = owner.mask
    if (!mask) return
    const box = localBox(owner.element)
    if (!box) return
    mask.dirty = false
    mask.waiting = 0
    mask.palette = shared.palette
    mask.revision = shared.revision
    const spaceMatrix = owner.matrix
    const scale = Math.max(0.25, scaleOf(spaceMatrix) * shared.resolution)
    mask.scale = scaleOf(spaceMatrix) * shared.resolution
    const margin = padding + 2 / scale
    const width = Math.min(gpu.maxTexture, Math.max(1, Math.ceil((box.width + margin * 2) * scale)))
    const height = Math.min(
      gpu.maxTexture,
      Math.max(1, Math.ceil((box.height + margin * 2) * scale)),
    )
    mask.base = multiply(scaling(scale), translation(margin - box.x, margin - box.y))
    mask.placed = null
    mask.width = width
    mask.height = height
    clearRaster(width, height)
    const toTexels = multiply(mask.base, invert(spaceMatrix))
    // Grayscale keeps luminance in the red channel, so the mask fits one byte per texel.
    paint.filter = 'grayscale(1)'
    for (const child of mask.source.children) render(child, mask.base, 1, toTexels, false)
    paint.filter = 'none'
    mask.texture = upload(mask.texture, scratch, true, width, height)
  }

  const clipFor = (clips: Clip[]) => {
    const chain = clips.map(clip => clip.source.id).join('|')
    const key = `${shared.clipVersion}:${clips.map(clip => `${clipVersions.get(clip.source) ?? 0}@${clip.owner.matrix.map(value => value.toFixed(2)).join(',')}`).join('|')}`
    const known = clipTextures.get(chain)
    if (known?.key === key && known.texture) return known.texture
    clearRaster(canvas.width, canvas.height)
    paint.save()
    for (const clip of clips) {
      const { shape, rule } = clipPathOf(clip.source, clip.owner.screen)
      paint.clip(shape, rule)
    }
    paint.globalAlpha = 1
    paint.fillStyle = '#fff'
    paint.fillRect(0, 0, canvas.width, canvas.height)
    paint.restore()
    const texture = upload(known?.texture ?? null, scratch, true, canvas.width, canvas.height)
    clipTextures.set(chain, { texture, key })
    return texture
  }
  return { trimRaster, bounds, rasterize, rasterizeMask, clipFor }
}
export type Raster = ReturnType<typeof createRaster>
