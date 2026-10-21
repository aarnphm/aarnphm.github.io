import type { Gpu } from './gpu-textures'
import type { Raster } from './raster-layers'
import type { Draw, Rect, Run, Scope, Segment, Surface, Unit } from './render-model'
import type { SceneTree } from './scene-partition'
import type { SegmentCache } from './segment-cache'
import type { Transitions } from './transitions'
import { formatGrade, IDENTITY_GRADE } from '../../../util/scene-grade'
import { none, place } from './affine-geometry'
import { collectDraws } from './draw-list'

// Composites the scene's textures into the retained canvas, repainting only what changed.
export function createScenePainter(
  scope: Scope,
  gpu: Gpu,
  sceneTree: SceneTree,
  raster: Raster,
  transitions: Transitions,
  cache: SegmentCache,
) {
  const { gl, canvas, coverage, repainted, shared } = scope
  const { uniforms, textures, filter } = gpu
  const { root } = sceneTree
  const { clipFor } = raster
  const { gradeAt, frameUniforms } = transitions
  const {
    traces,
    segments,
    trace,
    extent,
    texels,
    meets,
    mergeRegions,
    dropSegment,
    dropSegments,
  } = cache
  // The scene state the retained canvas shows, so a frame can redraw only what changed.
  let shown = new Map<Run | Surface, { rect: Rect; below: Run | Surface | null }>()
  let shownScene = ''
  // The canvas size the segment textures were built for. They hold untinted pixels, so a
  // daylight tint change keeps them.
  let segmentScene = ''
  let rectBuffer: WebGLBuffer | null = null
  let rectVertices = new Float32Array(4096)
  const segmentMinimum = 4
  const paintScene = () => {
    const grade = gradeAt(shared.clock)
    // Frame-wide uniforms are set by the first draw, so a frame with nothing to repaint
    // makes no WebGL calls.
    let primed = false
    const prime = () => {
      if (primed) return
      primed = true
      frameUniforms(grade)
    }
    const draws = collectDraws(scope, root, clipFor)
    // WebGL validates every call; skip the ones that would not change state.
    const state = {
      alpha: -1,
      mask: null as Float32Array | null,
      unit: -1,
      plain: false,
      bound: [null, null, null, null] as (WebGLTexture | null)[],
    }
    const bind = (unit: number, texture: WebGLTexture | null) => {
      if (state.bound[unit] === texture) return
      if (state.unit !== unit) gl.activeTexture(gl.TEXTURE0 + (state.unit = unit))
      gl.bindTexture(gl.TEXTURE_2D, (state.bound[unit] = texture))
    }
    // Segment textures hold untinted pixels; the grade applies when they reach the canvas.
    let building = false
    const shade = (plain: boolean) => {
      if (state.plain === plain) return
      state.plain = plain
      const current = plain ? IDENTITY_GRADE : grade
      gl.uniformMatrix3fv(uniforms.gradeM, true, current.m)
      gl.uniform3fv(uniforms.gradeB, current.b)
    }
    // Draws a unit quad, or only its share of `first..first + count` rect vertices.
    const paint = (draw: Draw, first = -1, count = 0) => {
      prime()
      shade(building)
      bind(3, coverage.texture ?? gpu.blank)
      bind(1, draw.maskTexture ?? gpu.blank)
      bind(2, draw.clipTexture ?? gpu.blank)
      bind(0, draw.texture)
      if (draw.alpha !== state.alpha) gl.uniform1f(uniforms.alpha, (state.alpha = draw.alpha))
      if (draw.mask !== state.mask)
        gl.uniformMatrix3fv(uniforms.maskPlace, false, (state.mask = draw.mask))
      gl.uniformMatrix3fv(uniforms.place, false, draw.placed)
      if (first < 0) gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      else gl.drawArrays(gl.TRIANGLES, first, count)
    }
    const paintSegment = (segment: Segment, first = -1, count = 0) => {
      prime()
      shade(false)
      bind(3, coverage.texture ?? gpu.blank)
      bind(1, gpu.blank)
      bind(2, gpu.blank)
      bind(0, segment.texture)
      if (state.alpha !== 1) gl.uniform1f(uniforms.alpha, (state.alpha = 1))
      if (state.mask !== none) gl.uniformMatrix3fv(uniforms.maskPlace, false, (state.mask = none))
      gl.uniformMatrix3fv(uniforms.place, false, segment.placed)
      if (first < 0) gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      else gl.drawArrays(gl.TRIANGLES, first, count)
    }
    // Retained pixels and cached segments hold finished draws, so the theme and resize overlays
    // and the load coverage must be inactive.
    const surfaceKey = `${canvas.width}x${canvas.height}@${shared.resolution}`
    const key = `${surfaceKey}|${formatGrade(grade)}`
    const cacheable = !shared.fade && !shared.resizeFrame && !coverage.active
    if (!cacheable || surfaceKey !== segmentScene) {
      dropSegments()
      segmentScene = cacheable ? surfaceKey : ''
    }
    if (!cacheable) {
      shownScene = ''
      shown = new Map()
      repainted.clear()
      gl.clear(gl.COLOR_BUFFER_BIT)
      for (const draw of draws) paint(draw)
      return
    }
    // A new tint changes every pixel: repaint the whole canvas from the segments that remain.
    const retint = key !== shownScene
    shownScene = key
    // What changed since the retained frame: moved or repainted draws at both positions,
    // removed draws, and draws whose neighbour below changed through depth sorting.
    const rects = draws.map(draw => extent(draw.placed))
    const next = new Map<Run | Surface, { rect: Rect; below: Run | Surface | null }>()
    let dirty: Rect[] = []
    const idle = draws.map((draw, index) => {
      const known = trace(draw)
      const rect = rects[index]
      const below = index ? draws[index - 1].item : null
      const before = shown.get(draw.item)
      next.set(draw.item, { rect, below })
      // A surface that names its repainted rects dirties only those.
      const surface = draw.item.kind === 'surface' ? draw.item : null
      const boxes =
        surface && known.repaint && before?.below === below ? repainted.get(surface) : undefined
      if (!before) dirty.push(rect)
      else if (known.changed === shared.clock && surface && boxes)
        for (const box of boxes) dirty.push(texels(draw.placed, surface, box))
      else if (known.changed === shared.clock || before.below !== below)
        dirty.push(rect, before.rect)
      return shared.clock - known.changed >= known.hold
    })
    for (const [item, before] of shown) if (!next.has(item)) dirty.push(before.rect)
    shown = next
    repainted.clear()
    if (retint) dirty = [[0, 0, canvas.width, canvas.height]]
    // Merge nearby regions: one rect costs a draw per layer it crosses.
    dirty = mergeRegions(dirty, 8 * shared.resolution)
    const covered = dirty.reduce((sum, rect) => sum + (rect[2] - rect[0]) * (rect[3] - rect[1]), 0)
    const whole = covered > 0.5 * canvas.width * canvas.height
    if (!dirty.length) return
    // Premultiplied source-over is associative, so a run of draws composited into a
    // transparent texture, then drawn once, matches drawing each of them in turn.
    // A built segment stays while its draws are unchanged; draws that settle beside it
    // form new segments instead of rebuilding it. Area budgets bound texture memory and
    // the offscreen work one frame may add.
    const area = canvas.width * canvas.height
    let memory = 0
    for (const segment of segments.values())
      memory += (segment.rect[2] - segment.rect[0]) * (segment.rect[3] - segment.rect[1])
    let rebuilt = 0
    const units: Unit[] = []
    const used = new Set<Segment>()
    const build = (from: number, to: number) => {
      const rect: Rect = [canvas.width, canvas.height, 0, 0]
      for (let index = from; index < to; index++) {
        rect[0] = Math.min(rect[0], Math.max(0, rects[index][0]))
        rect[1] = Math.min(rect[1], Math.max(0, rects[index][1]))
        rect[2] = Math.max(rect[2], Math.min(canvas.width, rects[index][2]))
        rect[3] = Math.max(rect[3], Math.min(canvas.height, rects[index][3]))
      }
      const width = rect[2] - rect[0]
      const height = rect[3] - rect[1]
      const items = draws.slice(from, to).map(draw => draw.item)
      const known = segments.get(items[0])
      const reuse =
        known &&
        !used.has(known) &&
        known.rect[2] - known.rect[0] === width &&
        known.rect[3] - known.rect[1] === height
      // A resized segment replaces its old texture, so its old area does not count against the cap.
      const reclaimed =
        known && !used.has(known) && !reuse
          ? (known.rect[2] - known.rect[0]) * (known.rect[3] - known.rect[1])
          : 0
      const added = reuse ? 0 : width * height
      if (
        to - from < segmentMinimum ||
        width <= 0 ||
        height <= 0 ||
        width > gpu.maxTexture ||
        height > gpu.maxTexture ||
        memory - reclaimed + added > 3 * area ||
        rebuilt + width * height > area
      ) {
        for (let index = from; index < to; index++)
          units.push({ draw: draws[index], rect: rects[index] })
        return
      }
      let segment = reuse ? known : undefined
      if (segment) gl.bindFramebuffer(gl.FRAMEBUFFER, segment.framebuffer)
      else {
        if (known && !used.has(known)) {
          memory -= (known.rect[2] - known.rect[0]) * (known.rect[3] - known.rect[1])
          dropSegment(known)
          segments.delete(items[0])
        }
        const texture = gl.createTexture()
        const framebuffer = gl.createFramebuffer()
        if (!texture || !framebuffer) {
          gl.deleteTexture(texture)
          gl.deleteFramebuffer(framebuffer)
          for (let index = from; index < to; index++)
            units.push({ draw: draws[index], rect: rects[index] })
          return
        }
        textures.add(texture)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        filter()
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
        segment = { items, rect, texture, framebuffer, placed: none, built: shared.clock }
        memory += added
      }
      rebuilt += width * height
      segment.items = items
      segment.rect = rect
      // The texture holds the target's bottom row first, so the quad is flipped to match.
      segment.placed = place([width, 0, 0, -height, rect[0], rect[3]])
      segment.built = shared.clock
      segments.set(items[0], segment)
      state.unit = -1
      state.bound.fill(null)
      gl.viewport(0, 0, width, height)
      gl.uniform4f(uniforms.target, rect[0], rect[1], width, height)
      gl.clear(gl.COLOR_BUFFER_BIT)
      building = true
      for (let index = from; index < to; index++) paint(draws[index])
      building = false
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.uniform4f(uniforms.target, 0, 0, canvas.width, canvas.height)
      units.push({ segment, rect })
      used.add(segment)
    }
    let pending = -1
    for (let index = 0; index < draws.length; ) {
      const segment = segments.get(draws[index].item)
      const kept =
        segment &&
        !used.has(segment) &&
        index + segment.items.length <= draws.length &&
        segment.items.every(
          (item, offset) =>
            draws[index + offset].item === item &&
            (traces.get(item)?.changed ?? Infinity) < segment.built,
        )
      if (segment && kept) {
        if (pending >= 0) build(pending, index)
        pending = -1
        units.push({ segment, rect: segment.rect })
        used.add(segment)
        index += segment.items.length
        continue
      }
      if (idle[index]) {
        if (pending < 0) pending = index
      } else {
        if (pending >= 0) build(pending, index)
        pending = -1
        units.push({ draw: draws[index], rect: rects[index] })
      }
      index++
    }
    if (pending >= 0) build(pending, draws.length)
    for (const [first, segment] of segments) {
      if (used.has(segment)) continue
      dropSegment(segment)
      segments.delete(first)
    }
    if (whole) {
      gl.clear(gl.COLOR_BUFFER_BIT)
      for (const unit of units)
        if ('segment' in unit) paintSegment(unit.segment)
        else paint(unit.draw)
      return
    }
    // Clear the changed regions, then draw each unit once over just the regions it crosses.
    // Its vertices carry their own quad coordinates, so the shaders sample as for a whole quad.
    gl.enable(gl.SCISSOR_TEST)
    for (const [left, top, right, bottom] of dirty) {
      gl.scissor(left, canvas.height - bottom, right - left, bottom - top)
      gl.clear(gl.COLOR_BUFFER_BIT)
    }
    gl.disable(gl.SCISSOR_TEST)
    const spans: { unit: Unit; first: number; count: number }[] = []
    let length = 0
    for (const unit of units) {
      const p = 'segment' in unit ? unit.segment.placed : unit.draw.placed
      const det = p[0] * p[4] - p[1] * p[3]
      if (!det) continue
      const first = length / 2
      for (const rect of dirty) {
        if (!meets(rect, unit.rect)) continue
        const left = Math.max(rect[0], unit.rect[0])
        const top = Math.max(rect[1], unit.rect[1])
        const right = Math.min(rect[2], unit.rect[2])
        const bottom = Math.min(rect[3], unit.rect[3])
        if (rectVertices.length < length + 12) {
          const grown = new Float32Array(rectVertices.length * 2)
          grown.set(rectVertices)
          rectVertices = grown
        }
        for (const [x, y] of [
          [left, top],
          [right, top],
          [left, bottom],
          [left, bottom],
          [right, top],
          [right, bottom],
        ]) {
          rectVertices[length++] = (p[4] * (x - p[6]) - p[3] * (y - p[7])) / det
          rectVertices[length++] = (p[0] * (y - p[7]) - p[1] * (x - p[6])) / det
        }
      }
      if (length / 2 > first) spans.push({ unit, first, count: length / 2 - first })
    }
    rectBuffer ??= gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, rectBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, rectVertices.subarray(0, length), gl.STREAM_DRAW)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    for (const { unit, first, count } of spans)
      if ('segment' in unit) paintSegment(unit.segment, first, count)
      else paint(unit.draw, first, count)
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.vertexBuffer)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  }
  const lose = () => {
    shownScene = ''
    segmentScene = ''
    rectBuffer = null
  }
  const dispose = () => gl.deleteBuffer(rectBuffer)
  return { paintScene, lose, dispose }
}
export type ScenePainter = ReturnType<typeof createScenePainter>
