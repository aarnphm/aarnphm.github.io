import type { Landscape, Point } from './landscape'
import { trailHinterland } from './hinterland'
import { trailMountainBackdrop } from './mountain-backdrop'
import { trailTerraces, trailTerrain } from './terrain'

export type SceneryRequest = {
  landscape: Landscape
  seed: number
  width: number
  camp: Point
  routes: string[]
  backdropClip: string
  entrances: Point[]
}
export type SceneryParts = {
  terraces: string
  hinterland: string
  terrain: string
  backdrop: string
}
export type SceneryPart = keyof SceneryParts
export type SceneryWorkerRequest = SceneryRequest & { parts: SceneryPart[] }
export type SceneryMessage = {
  [K in SceneryPart]: { part: K; value: SceneryParts[K] }
}[SceneryPart]

// The scene inserts the parts in this order.
export const sceneryOrder: SceneryPart[] = ['terraces', 'hinterland', 'terrain', 'backdrop']

export function paintSceneryPart<K extends SceneryPart>(
  part: K,
  request: SceneryRequest,
): SceneryParts[K] {
  const { landscape, seed } = request
  const parts: { [P in SceneryPart]: () => SceneryParts[P] } = {
    terraces: () => trailTerraces(landscape, seed),
    hinterland: () => trailHinterland(request.width, request.camp, seed),
    terrain: () => trailTerrain(landscape, seed, request.routes),
    backdrop: () => trailMountainBackdrop(landscape, seed, request.backdropClip, request.entrances),
  }
  return parts[part]()
}
