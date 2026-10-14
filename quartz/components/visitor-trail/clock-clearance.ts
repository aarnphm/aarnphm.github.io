import type { CollisionActor } from './actor-spacing'
import type { Point } from './landscape'
import { svgLocalMatrix } from '../../util/svg-geometry'

type ClockRegion = { left: number; right: number; top: number; bottom: number }
type FrameGeometry = {
  time: DocumentTimeline['currentTime']
  width: number
  height: number
  clock: Element | null
  region?: ClockRegion | null
  layers: WeakMap<Element, { transform: string; matrix: DOMMatrix | null }>
  screens: WeakMap<SVGSVGElement, DOMMatrix | null>
}
const frames = new WeakMap<Document, FrameGeometry>()

// All walkers reserve the same fixed corner. Read layout once per frame instead of
// forcing it again after each walker's transform write.
function frameGeometry(document: Document) {
  const view = document.defaultView
  if (!view) return null
  const time = document.timeline.currentTime
  const known = frames.get(document)
  if (known?.time === time && known.clock?.isConnected) return known
  const next: FrameGeometry = {
    time,
    width: view.innerWidth,
    height: view.innerHeight,
    clock: document.querySelector('.trail-local-time'),
    layers: new WeakMap(),
    screens: new WeakMap(),
  }
  frames.set(document, next)
  return next
}

export function trailClockRegion(document: Document) {
  const frame = frameGeometry(document)
  if (!frame) return null
  if (frame.region !== undefined) return frame.region
  const clock = frame.clock?.getBoundingClientRect()
  if (!clock || clock.width <= 0 || clock.height <= 0) return (frame.region = null)
  return (frame.region = {
    left: clock.left - 12,
    right: frame.width,
    top: clock.top - 12,
    bottom: frame.height,
  })
}

export function clockObstacles(
  document: Document,
  offset: Point = { x: 0, y: 0 },
): CollisionActor[] {
  const region = trailClockRegion(document)
  if (!region) return []
  const height = region.bottom - region.top
  const width = region.right - region.left
  return [
    {
      id: 'clock-corner',
      kind: 'trunk',
      x: (region.left + region.right) / 2 + offset.x,
      y: (region.top + region.bottom) / 2 + offset.y,
      radius: Math.hypot(width, height) / 2,
      bounds: {
        left: region.left + offset.x,
        right: region.right + offset.x,
        top: region.top + offset.y,
        bottom: region.bottom + offset.y,
      },
    },
  ]
}

export function readClockObstacles(scene: ParentNode, relativeTo?: SVGGElement) {
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  if (!svg) return []
  const layer = relativeTo?.closest<SVGGElement>('[data-trail-depth]') ?? svg
  const frame = frameGeometry(svg.ownerDocument)
  const transform = `${layer.getAttribute('transform')}:${layer.getAttribute('style')}:${svg.getAttribute('viewBox')}`
  const known = frame?.layers.get(layer)
  let screen = frame?.screens.get(svg)
  if (screen === undefined) {
    const native = svg.getScreenCTM()
    screen = native
      ? new DOMMatrix([native.a, native.b, native.c, native.d, native.e, native.f])
      : null
    frame?.screens.set(svg, screen)
  }
  const matrix =
    known?.transform === transform
      ? known.matrix
      : (screen?.multiply(svgLocalMatrix(layer, svg)) ?? null)
  if (frame && known?.transform !== transform) frame.layers.set(layer, { transform, matrix })
  if (!matrix) return []
  const inverse = matrix.inverse()
  const scale = Math.hypot(matrix.a, matrix.b)
  return clockObstacles(svg.ownerDocument).map(other => {
    const point = (x: number, y: number) => ({
      x: inverse.a * x + inverse.c * y + inverse.e,
      y: inverse.b * x + inverse.d * y + inverse.f,
    })
    const corners = other.bounds
      ? [
          point(other.bounds.left, other.bounds.top),
          point(other.bounds.right, other.bounds.top),
          point(other.bounds.left, other.bounds.bottom),
          point(other.bounds.right, other.bounds.bottom),
        ]
      : []
    return {
      ...other,
      ...point(other.x, other.y),
      radius: other.radius / scale,
      bounds: corners.length
        ? {
            left: Math.min(...corners.map(p => p.x)),
            right: Math.max(...corners.map(p => p.x)),
            top: Math.min(...corners.map(p => p.y)),
            bottom: Math.max(...corners.map(p => p.y)),
          }
        : undefined,
    }
  })
}
