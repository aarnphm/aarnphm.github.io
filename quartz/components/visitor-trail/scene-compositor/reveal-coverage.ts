import type { Gpu } from './gpu-textures'
import type { Raster } from './raster-layers'
import type { Box, Group, Run, Scope } from './render-model'
import { trailOpeningCell } from '../scene-reveal'
import { invert, multiply } from './affine-geometry'

// Load coverage: the reveal cells that still wait for a texture, and the order that fills them.
export function createLoadCoverage(scope: Scope, gpu: Gpu, raster: Raster) {
  const { scene, gl, opening, size, coverage, shared } = scope
  const { textures, filter } = gpu
  const { bounds } = raster
  const coverageBounds = new WeakMap<Run | Group, { revision: number; box: Box | null }>()
  const paintBounds = (target: Run | Group): Box | null => {
    const owner = target.kind === 'group' ? target : target.owner
    const layer = target.kind === 'group' ? target.mask : target
    if (!layer) return null
    let matrix = owner.screen
    let box: Box
    if (layer.width && layer.height) {
      matrix = multiply(owner.screen, invert(layer.base))
      box = { left: 0, top: 0, right: layer.width, bottom: layer.height }
    } else if (target.kind === 'run' && target.box) {
      box = target.box
    } else {
      const known = coverageBounds.get(target)
      let next = known?.revision === shared.revision ? known.box : undefined
      if (next === undefined) {
        // Empty runs can share an owner with the whole valley. Measure their own nodes,
        // otherwise an invisible container reserves that owner's entire reveal area.
        next = bounds(
          target.kind === 'run' ? target.nodes : [owner.element],
          owner.matrix,
          target.kind === 'group' || target.nodes[0] === owner.element,
        )
        if (target.kind === 'run') target.box = next
        coverageBounds.set(target, { revision: shared.revision, box: next })
      }
      if (!next) return null
      box = next
    }
    const x = (box.left + box.right) / 2
    const y = (box.top + box.bottom) / 2
    const cx = matrix[0] * x + matrix[2] * y + matrix[4]
    const cy = matrix[1] * x + matrix[3] * y + matrix[5]
    const halfWidth = (box.right - box.left) / 2
    const halfHeight = (box.bottom - box.top) / 2
    const reachX = Math.abs(matrix[0]) * halfWidth + Math.abs(matrix[2]) * halfHeight
    const reachY = Math.abs(matrix[1]) * halfWidth + Math.abs(matrix[3]) * halfHeight
    return { left: cx - reachX, right: cx + reachX, top: cy - reachY, bottom: cy + reachY }
  }
  const paintCoverage = (unfinished: (Run | Group)[]) => {
    if (!unfinished.length && !coverage.active) return
    const cell = trailOpeningCell * shared.resolution
    const width = Math.ceil(size.width / cell)
    const height = Math.ceil(size.height / cell)
    if (coverage.width !== width || coverage.height !== height) {
      coverage.width = width
      coverage.height = height
      coverage.cells = new Uint8Array(width * height)
    }
    coverage.cells.fill(255)
    for (const target of unfinished) {
      const box = paintBounds(target)
      if (!box) continue
      const left = Math.max(0, Math.floor(box.left / cell))
      const right = Math.min(width, Math.ceil(box.right / cell))
      const top = Math.max(0, Math.floor(box.top / cell))
      const bottom = Math.min(height, Math.ceil(box.bottom / cell))
      if (right <= left || bottom <= top) continue
      for (let y = top; y < bottom; y++) coverage.cells.fill(0, y * width + left, y * width + right)
    }
    coverage.active = false
    let available = 0
    for (let index = 0; index < coverage.cells.length; index++) {
      if (!coverage.cells[index]) coverage.active = true
      else available++
    }
    scene.dataset.trailLoadProgress = (available / coverage.cells.length).toFixed(3)
    if (!coverage.active) return
    coverage.texture ??= gl.createTexture()
    if (!coverage.texture) return
    textures.add(coverage.texture)
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D, coverage.texture)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R8,
      width,
      height,
      0,
      gl.RED,
      gl.UNSIGNED_BYTE,
      coverage.cells,
    )
    filter()
  }
  const priorities = new WeakMap<Run | Group, { revision: number; value: number }>()
  const rank = (target: Run | Group) => {
    const owner = target.kind === 'group' ? target : target.owner
    if (target.kind === 'group' || owner.element.closest('.trail-paper, .trail-ground-base'))
      return -Infinity
    if (owner.element.closest('.trail-camp, .trail-fire, .trail-camp-approach')) return -1e9
    // Full-width ground and route rasters cover the lowest reveal cells. Fill them before
    // individual crowns, otherwise those cells hold the whole opening behind the forest.
    if (owner.element.closest('.trail-ground, .trail-steps, .trail-hinterland')) return -1e8
    if (owner.element.closest('.trail-weather')) return Infinity
    const ground = owner.element.closest<SVGElement>('[data-ground-y]')
    // Untranslated paths can still reach the camp. Use their painted extent rather than
    // the owner's transform origin, so bottom-row work cannot sit behind the mountains.
    return -(ground
      ? Number(ground.dataset.groundY) * shared.resolution
      : (paintBounds(target)?.bottom ?? -Infinity))
  }
  const paintPriority = (target: Run | Group) => {
    const known = priorities.get(target)
    if (known?.revision === shared.revision) return known.value
    // A palette sweep keeps its layout. Resolve each layer's ancestor-based order once;
    // changing pixels and animation transforms still update coverage on every frame.
    const value = rank(target)
    priorities.set(target, { revision: shared.revision, value })
    return value
  }
  const coverageCeiling = () => {
    const origin = opening.origin()
    const radius = opening.radius()
    if (!coverage.active) return 1
    let ceiling = 1
    const cell = trailOpeningCell * shared.resolution
    for (let index = 0; index < coverage.cells.length; index++) {
      if (coverage.cells[index]) continue
      const y = Math.floor(index / coverage.width)
      const x = index % coverage.width
      const dx = Math.max(0, Math.abs((x + 0.5) * cell - origin.x * shared.resolution) - cell / 2)
      const dy = Math.max(0, Math.abs((y + 0.5) * cell - origin.y * shared.resolution) - cell / 2)
      ceiling = Math.min(ceiling, Math.hypot(dx, dy) / (radius * shared.resolution + cell))
    }
    return Math.max(0, ceiling - 0.000001)
  }
  return { paintCoverage, paintPriority, coverageCeiling }
}
export type LoadCoverage = ReturnType<typeof createLoadCoverage>
