import type { ActivityCardPoint } from '../../../util/press-kit-activity'

const WIDTH = 960
const HEIGHT = 540

export interface CardRoute {
  segments: { x: number; y: number }[][]
}

export function cardRoute(route: ActivityCardPoint[][]): CardRoute | undefined {
  const origin = route[0]?.[0]
  if (!origin) return undefined
  const project = (point: ActivityCardPoint) => {
    const delta = ((point.lng - origin.lng + 540) % 360) - 180
    const sin = Math.sin((point.lat * Math.PI) / 180)
    return {
      x: (origin.lng + delta + 180) / 360,
      y: 0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI),
    }
  }
  const projected = route.map(segment => segment.map(project))
  const points = projected.flat()
  let left = Infinity,
    right = -Infinity,
    top = Infinity,
    bottom = -Infinity
  for (const point of points) {
    left = Math.min(left, point.x)
    right = Math.max(right, point.x)
    top = Math.min(top, point.y)
    bottom = Math.max(bottom, point.y)
  }
  const cx = (left + right) / 2
  const cy = (top + bottom) / 2
  const scale = Math.min(
    (WIDTH - 96) / Math.max(right - left, 1e-8),
    (HEIGHT - 96) / Math.max(bottom - top, 1e-8),
  )
  return {
    segments: projected.map(segment =>
      segment.map(point => ({
        x: WIDTH / 2 + (point.x - cx) * scale,
        y: HEIGHT / 2 + (point.y - cy) * scale,
      })),
    ),
  }
}

export function routeSvg(route: CardRoute, ink: string, background: string): string {
  const ns = 'http://www.w3.org/2000/svg'
  const svg = document.createElementNS(ns, 'svg')
  svg.setAttribute('xmlns', ns)
  svg.setAttribute('viewBox', `0 0 ${WIDTH} ${HEIGHT}`)
  svg.setAttribute('width', String(WIDTH))
  svg.setAttribute('height', String(HEIGHT))
  const title = document.createElementNS(ns, 'title')
  title.textContent = 'Recorded activity route'
  svg.append(title)
  for (const segment of route.segments) {
    const path = document.createElementNS(ns, 'path')
    path.setAttribute(
      'd',
      segment
        .map(
          (point, index) => `${index === 0 ? 'M' : 'L'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`,
        )
        .join(' '),
    )
    path.setAttribute('fill', 'none')
    path.setAttribute('stroke', ink)
    path.setAttribute('stroke-width', '5')
    path.setAttribute('stroke-linecap', 'round')
    path.setAttribute('stroke-linejoin', 'round')
    svg.append(path)
  }
  const ends = [
    { point: route.segments[0]?.[0], fill: '#668a3a' },
    { point: route.segments.at(-1)?.at(-1), fill: '#c54c3c' },
  ]
  for (const { point, fill } of ends) {
    if (!point) continue
    const circle = document.createElementNS(ns, 'circle')
    circle.setAttribute('cx', String(point.x))
    circle.setAttribute('cy', String(point.y))
    circle.setAttribute('r', '8')
    circle.setAttribute('fill', fill)
    circle.setAttribute('stroke', background)
    circle.setAttribute('stroke-width', '3')
    svg.append(circle)
  }
  return svg.outerHTML
}
