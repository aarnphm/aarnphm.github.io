import type { Point } from './landscape'
import { canMove, circleClearance, type CollisionActor, type SpacingBounds } from './actor-spacing'
import { detourTrailSteps } from './obstacle-route'

export type PetTrailFoot = Point & { scale: number; progress: number }
export type PetTrailSearch = Generator<void, PetTrailFoot[] | null, undefined>

// Keep the sampled road's order at crossings. A bypass inherits its two endpoints' perspective
// and journey progress by distance, rather than projecting onto a possibly earlier road leg.
export function detourPetTrail(
  road: readonly PetTrailFoot[],
  body: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
): PetTrailFoot[] | null {
  const search = detourPetTrailSteps(road, body, obstacles, bounds)
  let step = search.next()
  while (!step.done) step = search.next()
  return step.value
}

export function* detourPetTrailSteps(
  road: readonly PetTrailFoot[],
  body: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
): PetTrailSearch {
  if (!road.length) return null
  const feet = [road[0]]
  for (let index = 1; index < road.length; index++) {
    yield
    const from = feet[feet.length - 1]
    let goal = road[index]
    let radius = body * Math.max(from.scale, goal.scale)
    if (canMove(from, goal, radius, obstacles)) {
      feet.push(goal)
      continue
    }
    // Skip the dense samples inside a solid together. The shared visibility graph finds the
    // continuous road around that solid, including any neighbouring camp bodies.
    while (
      index < road.length - 1 &&
      obstacles.some(other => circleClearance(goal, radius, other) < 0.1)
    ) {
      goal = road[++index]
      radius = Math.max(radius, body * goal.scale)
    }
    let planned = yield* detourTrailSteps(
      [from, goal],
      radius,
      obstacles,
      bounds,
      index === road.length - 1,
    )
    while (!planned || !canMove(from, planned.points[0], radius, obstacles)) {
      if (index === road.length - 1) return null
      goal = road[++index]
      radius = Math.max(radius, body * goal.scale)
      planned = yield* detourTrailSteps(
        [from, goal],
        radius,
        obstacles,
        bounds,
        index === road.length - 1,
      )
    }
    const last = planned.points.at(-1)
    if (!last) return null
    // Camp roads end at the hearth. The shared planner chooses a reachable arrival outside
    // that body, reached by the same walk, with the destination's perspective and progress.
    const arrival: PetTrailFoot = { ...goal, ...last }
    const points = [from, ...planned.points]
    let length = 0
    const distances = points.map((point, step) => {
      if (step) length += Math.hypot(point.x - points[step - 1].x, point.y - points[step - 1].y)
      return length
    })
    for (let step = 1; step < points.length; step++) {
      const distance = distances[step] - distances[step - 1]
      if (distance < 0.01) continue
      const samples = Math.ceil(distance / 4)
      for (let sample = 1; sample <= samples; sample++) {
        const along = sample / samples
        const fraction = (distances[step - 1] + distance * along) / Math.max(0.01, length)
        feet.push(
          step === points.length - 1 && sample === samples
            ? arrival
            : {
                x: points[step - 1].x + (points[step].x - points[step - 1].x) * along,
                y: points[step - 1].y + (points[step].y - points[step - 1].y) * along,
                scale: from.scale + (goal.scale - from.scale) * fraction,
                progress: from.progress + (goal.progress - from.progress) * fraction,
              },
        )
      }
    }
  }
  return feet
}
