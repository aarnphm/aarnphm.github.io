import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'

type RouteDrawPoint = { x: number; y: number }

const routePath = (route: RouteDrawPoint[]): string => {
  const pad = 6
  const span = 100 - pad * 2
  let d = ''
  route.forEach((p, i) => {
    d += `${i ? 'L' : 'M'} ${(pad + p.x * span).toFixed(2)} ${(pad + (1 - p.y) * span).toFixed(2)} `
  })
  return d
}

const routePointAtDistance = (
  route: StravaActivityDetail['route'],
  distanceKm: number,
): RouteDrawPoint => {
  if (distanceKm <= route[0].d) return route[0]
  for (let index = 1; index < route.length; index++) {
    const previous = route[index - 1]
    const next = route[index]
    if (distanceKm > next.d) continue
    const span = next.d - previous.d
    const fraction = span > 0 ? (distanceKm - previous.d) / span : 1
    return {
      x: previous.x + (next.x - previous.x) * fraction,
      y: previous.y + (next.y - previous.y) * fraction,
    }
  }
  return route[route.length - 1]
}

const selectedRoute = (
  route: StravaActivityDetail['route'],
  range: ActivityAnalysisRange | null,
): RouteDrawPoint[] => {
  if (!range) return []
  const start = Math.max(route[0].d, range.startDistanceKm)
  const end = Math.min(route[route.length - 1].d, range.endDistanceKm)
  if (end <= start) return []
  return [
    routePointAtDistance(route, start),
    ...route.filter(point => point.d > start && point.d < end),
    routePointAtDistance(route, end),
  ]
}

export const buildRoute = <N>(
  f: TriNodeFactory<N>,
  route: StravaActivityDetail['route'],
  range: ActivityAnalysisRange | null = null,
): N => {
  const fig = f.svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  f.add(fig, f.svg('path', { d: routePath(route), class: 'tri-route-path' }))
  const selection = selectedRoute(route, range)
  f.add(
    fig,
    f.svg('path', {
      d: selection.length >= 2 ? routePath(selection) : '',
      class: 'tri-route-selected',
    }),
  )
  f.add(fig, f.svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return fig
}
