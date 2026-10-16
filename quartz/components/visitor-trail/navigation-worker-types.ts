import type { CollisionActor, SpacingBounds } from './actor-spacing'
import type { JourneyRoute } from './hiker-journey'
import type { Landscape, Point } from './landscape'
import type { PlannedTrail } from './obstacle-route'

export type NavigationRoute = Pick<JourneyRoute, 'id' | 'readyAfter'> & { points: Point[] }
export type NavigationObstacle = Omit<CollisionActor, 'element'>
type CollisionQuery = { obstacles: NavigationObstacle[]; bounds: SpacingBounds }
type CampQuery = {
  from: Point
  routes: NavigationRoute[]
  landscape: Landscape
  obstacles: NavigationObstacle[]
}
export type NavigationQuery =
  | (CollisionQuery & { kind: 'trail'; road: Point[]; radius: number })
  | (CollisionQuery & {
      kind: 'repair'
      actor: string
      from: Point
      waypoints: Point[]
      radius: number
      waitingForAnimal?: string | null
    })
  | (CampQuery & { kind: 'patrol'; to: Point })
  | (CampQuery & { kind: 'departure' })
export type NavigationWorkerRequest = { id: number; query: NavigationQuery }
export type NavigationWorkerResponse = { id: number; planned: PlannedTrail | null }
