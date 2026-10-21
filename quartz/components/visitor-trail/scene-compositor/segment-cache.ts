import type { Gpu } from './gpu-textures'
import type { Box, Draw, Rect, Run, Scope, Segment, Surface, Trace } from './render-model'

// What each draw last showed, and the segments: runs of settled draws composited once into a
// texture covering their canvas extent.
export function createSegmentCache(scope: Scope, gpu: Gpu) {
  const { gl, canvas, shared } = scope
  const { contents, release } = gpu
  const traces = new WeakMap<Run | Surface, Trace>()
  const segments = new Map<Run | Surface, Segment>()
  const segmentHold = 500
  const same = (a: Float32Array, b: Float32Array) => {
    if (a === b) return true
    for (let index = 0; index < 9; index++) if (a[index] !== b[index]) return false
    return true
  }
  const trace = (draw: Draw) => {
    const content = contents.get(draw.texture) ?? 0
    const maskContent = draw.maskTexture ? (contents.get(draw.maskTexture) ?? 0) : 0
    const clipContent = draw.clipTexture ? (contents.get(draw.clipTexture) ?? 0) : 0
    const known = traces.get(draw.item)
    if (
      known &&
      known.texture === draw.texture &&
      known.content === content &&
      known.alpha === draw.alpha &&
      known.maskTexture === draw.maskTexture &&
      known.maskContent === maskContent &&
      known.clipTexture === draw.clipTexture &&
      known.clipContent === clipContent &&
      same(known.placed, draw.placed) &&
      same(known.mask, draw.mask)
    )
      return known
    // A draw that moves again after joining a segment waits longer next time, so
    // intermittent motion cannot rebuild its segment every few frames.
    const hold = !known
      ? segmentHold
      : shared.clock - known.changed >= known.hold
        ? Math.min(8000, known.hold * 2)
        : known.hold
    const next: Trace = {
      texture: draw.texture,
      content,
      placed: draw.placed,
      alpha: draw.alpha,
      mask: draw.mask,
      maskTexture: draw.maskTexture,
      maskContent,
      clipTexture: draw.clipTexture,
      clipContent,
      changed: shared.clock,
      repaint:
        !!known &&
        known.texture === draw.texture &&
        known.alpha === draw.alpha &&
        known.maskTexture === draw.maskTexture &&
        known.maskContent === maskContent &&
        known.clipTexture === draw.clipTexture &&
        known.clipContent === clipContent &&
        same(known.placed, draw.placed) &&
        same(known.mask, draw.mask),
      hold,
    }
    traces.set(draw.item, next)
    return next
  }
  // The canvas pixels a placed unit quad can touch.
  const extent = (p: Float32Array): Rect => {
    const x = p[6] + Math.min(0, p[0]) + Math.min(0, p[3])
    const y = p[7] + Math.min(0, p[1]) + Math.min(0, p[4])
    return [
      Math.floor(x) - 1,
      Math.floor(y) - 1,
      Math.ceil(x + Math.abs(p[0]) + Math.abs(p[3])) + 1,
      Math.ceil(y + Math.abs(p[1]) + Math.abs(p[4])) + 1,
    ]
  }
  // The canvas pixels a rect of a surface's own texels covers.
  const texels = (p: Float32Array, surface: Surface, box: Box): Rect => {
    const { width, height } = surface.canvas
    let left = Infinity
    let top = Infinity
    let right = -Infinity
    let bottom = -Infinity
    for (const [x, y] of [
      [box.left, box.top],
      [box.right, box.top],
      [box.left, box.bottom],
      [box.right, box.bottom],
    ]) {
      const u = x / width
      const v = y / height
      const px = p[0] * u + p[3] * v + p[6]
      const py = p[1] * u + p[4] * v + p[7]
      left = Math.min(left, px)
      top = Math.min(top, py)
      right = Math.max(right, px)
      bottom = Math.max(bottom, py)
    }
    return [Math.floor(left) - 1, Math.floor(top) - 1, Math.ceil(right) + 1, Math.ceil(bottom) + 1]
  }
  const meets = (a: Rect, b: Rect, gap = 0) =>
    a[0] < b[2] + gap && b[0] < a[2] + gap && a[1] < b[3] + gap && b[1] < a[3] + gap
  const dropSegment = (segment: Segment) => {
    gl.deleteFramebuffer(segment.framebuffer)
    release(segment.texture)
  }
  const dropSegments = () => {
    for (const segment of segments.values()) dropSegment(segment)
    segments.clear()
  }
  const mergeRegions = (dirty: Rect[], gap: number) => {
    const regions: Rect[] = []
    for (const [left, top, right, bottom] of dirty) {
      const rect: Rect = [
        Math.max(0, left),
        Math.max(0, top),
        Math.min(canvas.width, right),
        Math.min(canvas.height, bottom),
      ]
      if (rect[2] <= rect[0] || rect[3] <= rect[1]) continue
      // A grown rect can reach regions already passed, so the search restarts after a merge.
      for (let index = 0; index < regions.length; ) {
        const other = regions[index]
        if (!meets(other, rect, gap)) {
          index++
          continue
        }
        rect[0] = Math.min(rect[0], other[0])
        rect[1] = Math.min(rect[1], other[1])
        rect[2] = Math.max(rect[2], other[2])
        rect[3] = Math.max(rect[3], other[3])
        regions.splice(index, 1)
        index = 0
      }
      regions.push(rect)
    }
    return regions
  }
  return { traces, segments, trace, extent, texels, meets, mergeRegions, dropSegment, dropSegments }
}
export type SegmentCache = ReturnType<typeof createSegmentCache>
