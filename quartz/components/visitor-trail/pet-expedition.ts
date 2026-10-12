import { trailGroundScale, type Landscape, type Point } from './landscape'

function sample(path: SVGPathElement, from: number, to: number) {
  const segments = Math.max(1, Math.ceil(Math.abs(to - from) / 4))
  return Array.from({ length: segments + 1 }, (_, index): Point => {
    const point = path.getPointAtLength(from + ((to - from) * index) / segments)
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
      const point = path.getPointAtLength(path.getTotalLength())
      return point.x >= 40 && point.x <= landscape.width - 40
    })
    .sort((left, right) => {
      return (
        left.getPointAtLength(left.getTotalLength()).y -
        right.getPointAtLength(right.getTotalLength()).y
      )
    })[0]
  if (!upper || !road) return null
  const roadLength = road.getTotalLength()
  const summit = road.getPointAtLength(roadLength)
  const entrance = road.getPointAtLength(0)
  const valley = [...scene.querySelectorAll<SVGPathElement>('[data-trail-route]')].find(path => {
    const point = path.getPointAtLength(0)
    return Math.hypot(point.x - entrance.x, point.y - entrance.y) < 1
  })
  if (!valley) return null
  const upperLength = upper.getTotalLength()
  let junctionDistance = 0
  let score = Infinity
  for (let distance = 0; distance <= upperLength; distance += 4) {
    const point = upper.getPointAtLength(distance)
    if (point.y < landscape.bottom + 60 || point.x < 40 || point.x > landscape.width - 40) continue
    const candidate = Math.abs(point.x - summit.x)
    if (candidate < score) {
      score = candidate
      junctionDistance = distance
    }
  }
  if (!Number.isFinite(score)) return null
  const prefix = sample(upper, 0, junctionDistance)
  const junction = prefix[prefix.length - 1]
  const fall = summit.y - junction.y
  const connector = `M${junction.x} ${junction.y}C${junction.x} ${junction.y + fall * 0.35} ${summit.x} ${summit.y - fall * 0.3} ${summit.x} ${summit.y}`
  const probe = upper.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'path')
  probe.setAttribute('d', connector)
  const connectorLength = probe.getTotalLength()
  const descent = sample(road, roadLength, 0)
  const valleyLength = valley.getTotalLength()
  const prefixLength = pathLength(prefix)
  const passEnd = prefixLength + connectorLength
  const descentEnd = passEnd + pathLength(descent)
  const total = descentEnd + valleyLength
  const scaleStops = [
    [0, 1],
    [prefixLength / total, 1],
    [passEnd / total, 0.18],
    [descentEnd / total, trailGroundScale(entrance.y, landscape)],
    ...Array.from({ length: 25 }, (_, index) => {
      const distance = (valleyLength * (index + 1)) / 25
      const point = valley.getPointAtLength(distance)
      return [(descentEnd + distance) / total, trailGroundScale(point.y, landscape)]
    }),
  ]
  return {
    d:
      `M${prefix[0].x} ${prefix[0].y}${line(prefix.slice(1))}` +
      connector.replace(/^M[^C]+/, '') +
      line(descent.slice(1)) +
      (valley.getAttribute('d') ?? '').replace(/^M/, 'L'),
    connector,
    scaleStops,
    alpineSpan: [prefixLength / total, descentEnd / total],
    mountainTrack: road.dataset.mountainTrack,
    valleyRoute: valley.dataset.trailRoute,
  }
}
