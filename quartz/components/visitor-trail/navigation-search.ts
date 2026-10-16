import type { JourneyRoute } from './hiker-journey'
import type { NavigationQuery, NavigationRoute } from './navigation-worker-types'
import { campDepartureSteps, campPatrolRouteSteps } from './camp-routine'
import {
  detourTrailSteps,
  plannedTrail,
  replanOccupiedWaypointsSteps,
  type PlannedTrail,
} from './obstacle-route'

function restoreRoutes(routes: NavigationRoute[]): JourneyRoute[] {
  return routes.flatMap(route => {
    const restored = plannedTrail(
      { ...route, length: 0, point: () => ({ x: 0, y: 0 }) },
      { points: route.points, detours: 0 },
    )
    return restored ? [restored] : []
  })
}

export function* navigationSearch(
  query: NavigationQuery,
): Generator<void, PlannedTrail | null, undefined> {
  if (query.kind === 'trail')
    return yield* detourTrailSteps(query.road, query.radius, query.obstacles, query.bounds)
  const points =
    query.kind === 'repair'
      ? yield* replanOccupiedWaypointsSteps(
          query.actor,
          query.from,
          query.waypoints,
          query.radius,
          query.obstacles,
          query.bounds,
          query.waitingForAnimal,
        )
      : query.kind === 'patrol'
        ? yield* campPatrolRouteSteps(
            query.from,
            query.to,
            restoreRoutes(query.routes),
            query.landscape,
            query.obstacles,
          )
        : yield* campDepartureSteps(
            query.from,
            restoreRoutes(query.routes),
            query.landscape,
            query.obstacles,
          )
  return points ? { points, detours: 0 } : null
}
