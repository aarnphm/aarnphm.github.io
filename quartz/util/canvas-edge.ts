/** Canvas node box with a top-left origin, in canvas units. */
export interface CanvasEdgeRect {
  x: number
  y: number
  width: number
  height: number
}

export interface CanvasEdgePoint {
  x: number
  y: number
  side: string
}

export interface CanvasEdgeGeometry {
  p1: CanvasEdgePoint
  ext1: { x: number; y: number }
  cp1: { x: number; y: number }
  cp2: { x: number; y: number }
  ext2: { x: number; y: number }
  p2: CanvasEdgePoint
}

const STRAIGHT_LENGTH = 20

function offsetFromSide(point: { x: number; y: number }, side: string, distance: number) {
  switch (side) {
    case 'top':
      return { x: point.x, y: point.y - distance }
    case 'right':
      return { x: point.x + distance, y: point.y }
    case 'bottom':
      return { x: point.x, y: point.y + distance }
    case 'left':
      return { x: point.x - distance, y: point.y }
    default:
      return { x: point.x, y: point.y }
  }
}

export function canvasNodeEdgePoint(
  node: CanvasEdgeRect,
  side?: string,
  targetX?: number,
  targetY?: number,
): CanvasEdgePoint {
  const cx = node.x + node.width / 2
  const cy = node.y + node.height / 2
  const hw = node.width / 2
  const hh = node.height / 2

  if (side) {
    switch (side) {
      case 'top':
        return { x: cx, y: cy - hh, side: 'top' }
      case 'right':
        return { x: cx + hw, y: cy, side: 'right' }
      case 'bottom':
        return { x: cx, y: cy + hh, side: 'bottom' }
      case 'left':
        return { x: cx - hw, y: cy, side: 'left' }
    }
  }

  if (targetX !== undefined && targetY !== undefined) {
    const dx = targetX - cx
    const dy = targetY - cy

    if (dx === 0 && dy === 0) return { x: cx, y: cy, side: 'right' }

    const tx = Math.abs(dx) > 0 ? hw / Math.abs(dx) : Infinity
    const ty = Math.abs(dy) > 0 ? hh / Math.abs(dy) : Infinity
    const t = Math.min(tx, ty)

    const x = cx + dx * t
    const y = cy + dy * t

    let determinedSide = 'right'
    if (Math.abs(y - (cy - hh)) < 1) determinedSide = 'top'
    else if (Math.abs(y - (cy + hh)) < 1) determinedSide = 'bottom'
    else if (Math.abs(x - (cx - hw)) < 1) determinedSide = 'left'
    else if (Math.abs(x - (cx + hw)) < 1) determinedSide = 'right'

    return { x, y, side: determinedSide }
  }

  return { x: cx, y: cy, side: 'right' }
}

/** Side anchors, a short straight lead-out, then a cubic between the leads (Obsidian's edge shape). */
export function canvasEdgeGeometry(
  source: CanvasEdgeRect,
  target: CanvasEdgeRect,
  fromSide?: string,
  toSide?: string,
): CanvasEdgeGeometry {
  const p1 = canvasNodeEdgePoint(
    source,
    fromSide,
    target.x + target.width / 2,
    target.y + target.height / 2,
  )
  const p2 = canvasNodeEdgePoint(
    target,
    toSide,
    source.x + source.width / 2,
    source.y + source.height / 2,
  )
  const ext1 = offsetFromSide(p1, p1.side, STRAIGHT_LENGTH)
  const ext2 = offsetFromSide(p2, p2.side, STRAIGHT_LENGTH)
  const controlDist = Math.min(Math.hypot(ext2.x - ext1.x, ext2.y - ext1.y) * 0.4, 100)
  return {
    p1,
    ext1,
    cp1: offsetFromSide(ext1, p1.side, controlDist),
    cp2: offsetFromSide(ext2, p2.side, controlDist),
    ext2,
    p2,
  }
}

export function canvasEdgePath(
  { p1, ext1, cp1, cp2, ext2, p2 }: CanvasEdgeGeometry,
  round: (value: number) => number = value => value,
): string {
  const point = ({ x, y }: { x: number; y: number }) => `${round(x)} ${round(y)}`
  return `M ${point(p1)} L ${point(ext1)} C ${point(cp1)}, ${point(cp2)}, ${point(ext2)} L ${point(p2)}`
}

export function canvasEdgeMidpoint({ ext1, cp1, cp2, ext2 }: CanvasEdgeGeometry): {
  x: number
  y: number
} {
  const t = 0.5
  const mt = 1 - t
  return {
    x:
      mt * mt * mt * ext1.x + 3 * mt * mt * t * cp1.x + 3 * mt * t * t * cp2.x + t * t * t * ext2.x,
    y:
      mt * mt * mt * ext1.y + 3 * mt * mt * t * cp1.y + 3 * mt * t * t * cp2.y + t * t * t * ext2.y,
  }
}
