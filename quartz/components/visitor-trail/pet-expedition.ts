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
  const upper = scene.querySelector<SVGPathElement>('[data-pet-walk-path="0"]')
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
  if (!upper || !road) return null
  const roadGeometry = svgPathGeometry(road)
  const roadLength = roadGeometry.getTotalLength()
  const summit = roadGeometry.getPointAtLength(roadLength)
  const entrance = roadGeometry.getPointAtLength(0)
  const valley = [...scene.querySelectorAll<SVGPathElement>('[data-trail-route]')].find(path => {
    const point = svgPathGeometry(path).getPointAtLength(0)
    return Math.hypot(point.x - entrance.x, point.y - entrance.y) < 1
  })
  if (!valley) return null
  const upperLength = svgPathGeometry(upper).getTotalLength()
  const prefix = sample(upper, 0, upperLength)
  const end = prefix[prefix.length - 1]
  const before = prefix[Math.max(0, prefix.length - 2)]
  const exit = {
    x: landscape.width + 64,
    y: end.y + ((end.y - before.y) * 64) / Math.max(1, end.x - before.x),
  }
  prefix.push(exit)
  const descent = sample(road, roadLength, 0)
  const valleyGeometry = svgPathGeometry(valley)
  const valleyLength = valleyGeometry.getTotalLength()
  const prefixLength = pathLength(prefix)
  const descentEnd = prefixLength + pathLength(descent)
  const total = descentEnd + valleyLength
  const scaleStops = [
    [0, 1],
    [prefixLength / total, 1],
    [prefixLength / total, 0.18],
    [descentEnd / total, trailGroundScale(entrance.y, landscape)],
    ...Array.from({ length: 25 }, (_, index) => {
      const distance = (valleyLength * (index + 1)) / 25
      const point = valleyGeometry.getPointAtLength(distance)
      return [(descentEnd + distance) / total, trailGroundScale(point.y, landscape)]
    }),
  ]
  return {
    d:
      `M${prefix[0].x} ${prefix[0].y}${line(prefix.slice(1))}` +
      `M${summit.x} ${summit.y}` +
      line(descent.slice(1)) +
      (valley.getAttribute('d') ?? '').replace(/^M/, 'L'),
    handoff: { at: prefixLength / total, exit: [exit.x, exit.y], entry: [summit.x, summit.y] },
    scaleStops,
    alpineSpan: [prefixLength / total, descentEnd / total],
    mountainTrack: road.dataset.mountainTrack,
    valleyRoute: valley.dataset.trailRoute,
  }
}
