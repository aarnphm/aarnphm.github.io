import { svgPathGeometry } from '../../util/svg-geometry'
import { trailGroundScale, type Landscape, type Point } from './landscape'

function sample(path: SVGPathElement, from: number, to: number) {
  const geometry = svgPathGeometry(path)
  const segments = Math.max(1, Math.ceil(Math.abs(to - from) / 4))
  return Array.from({ length: segments + 1 }, (_, index): Point => {
    const point = geometry.getPointAtLength(from + ((to - from) * index) / segments)
    return { x: point.x, y: point.y }
  })
}

function pathLength(points: Point[]) {
  return points.slice(1).reduce((length, point, index) => {
    const previous = points[index]
    return length + Math.hypot(point.x - previous.x, point.y - previous.y)
  }, 0)
}

function line(points: Point[]) {
  return points.map(point => `L${point.x} ${point.y}`).join('')
}

export function petExpedition(scene: HTMLElement, landscape: Landscape) {
  const roads = [...scene.querySelectorAll<SVGPathElement>('path[data-mountain-track]')]
  const road = roads
    .filter(path => {
      const geometry = svgPathGeometry(path)
      const point = geometry.getPointAtLength(geometry.getTotalLength())
      return point.x >= 40 && point.x <= landscape.width - 40
    })
    .sort((left, right) => {
      return (
        svgPathGeometry(left).getPointAtLength(svgPathGeometry(left).getTotalLength()).y -
        svgPathGeometry(right).getPointAtLength(svgPathGeometry(right).getTotalLength()).y
      )
    })[0]
  if (!road) return null
  const roadGeometry = svgPathGeometry(road)
  const roadLength = roadGeometry.getTotalLength()
  const summit = roadGeometry.getPointAtLength(roadLength)
  const entrance = roadGeometry.getPointAtLength(0)
  const valley = [...scene.querySelectorAll<SVGPathElement>('[data-trail-route]')].find(path => {
    const point = svgPathGeometry(path).getPointAtLength(0)
    return Math.hypot(point.x - entrance.x, point.y - entrance.y) < 1
  })
  if (!valley) return null
  const descent = sample(road, roadLength, 0)
  const valleyGeometry = svgPathGeometry(valley)
  const valleyLength = valleyGeometry.getTotalLength()
  const descentEnd = pathLength(descent)
  const total = descentEnd + valleyLength
  const scaleStops = [
    [0, 0.18],
    [descentEnd / total, trailGroundScale(entrance.y, landscape)],
    ...Array.from({ length: 25 }, (_, index) => {
      const distance = (valleyLength * (index + 1)) / 25
      const point = valleyGeometry.getPointAtLength(distance)
      return [(descentEnd + distance) / total, trailGroundScale(point.y, landscape)]
    }),
  ]
  return {
    d:
      `M${summit.x} ${summit.y}` +
      line(descent.slice(1)) +
      (valley.getAttribute('d') ?? '').replace(/^M/, 'L'),
    scaleStops,
    alpineSpan: [0, descentEnd / total],
    mountainTrack: road.dataset.mountainTrack,
    valleyRoute: valley.dataset.trailRoute,
  }
}
