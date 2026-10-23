import {
  circleClearance,
  readTrunks,
  type CollisionActor,
  type SpacingBounds,
} from '../../visitor-trail/actor-spacing'
import { clockObstacles } from '../../visitor-trail/clock-clearance'
import { detourPetTrailSteps, type PetTrailSearch } from '../../visitor-trail/pet-trail'
import { TRAIL_PET_RADIUS } from '../../visitor-trail/trail-pet'
import {
  edgeY,
  height,
  margin,
  petState,
  trailScreenMatrix,
  viewBottom,
  viewTop,
  type HopPoint,
  type Perch,
  type Point,
  type TrailFoot,
} from './actor-model'
import { coversText, placeOn, type PetObstacles } from './perch-placement'
import { trailPx, trailRoutes } from './trail-geometry'

export const trailPointClear = (p: TrailFoot, path: SVGPathElement, obstacles: PetObstacles) => {
  const at = { x: p.x - window.scrollX, y: p.y - window.scrollY }
  const top = at.y - (edgeY.stand - viewTop) * petState.unit * p.scale
  const right = window.innerWidth - margin - (petState.width * p.scale) / 2
  return (
    at.x >= margin + (petState.width * p.scale) / 2 &&
    at.x <= right &&
    top >= margin &&
    top + height(p.scale) <= window.innerHeight - margin &&
    !coversText({ el: path, kind: 'stand', t: 0, at, score: 0, scale: p.scale }, obstacles)
  )
}

type TrailRun = { path: SVGPathElement; feet: TrailFoot[]; length: number }
export const trailActors = (scene: SVGSVGElement | null): CollisionActor[] => {
  const actors: CollisionActor[] = clockObstacles(document, {
    x: window.scrollX,
    y: window.scrollY,
  })
  const drawing = scene?.closest<HTMLElement>('.trail-scene') ?? scene
  // These bodies use the current screen matrix; the guard checks the live camera, so their
  // radii need no reservation for future camera positions in the scene's coordinate space.
  for (const other of drawing ? readTrunks(drawing, undefined, false) : []) {
    const element = other.element
    const parent = element?.parentElement
    const cached = element?.ownerSVGElement === scene
    const matrix = element && (cached ? trailScreenMatrix(element) : element.getScreenCTM())
    const ground =
      parent instanceof SVGGraphicsElement
        ? cached
          ? trailScreenMatrix(parent)
          : parent.getScreenCTM()
        : null
    if (!matrix || !ground) continue
    actors.push({
      ...other,
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: other.radius * Math.hypot(ground.a, ground.b),
    })
  }
  for (const element of scene?.querySelectorAll<SVGGElement>('[data-trail-collision]') ?? []) {
    const id = element.dataset.collisionId
    const kind = element.dataset.trailCollision
    if (
      !id ||
      id === 'pet-blob' ||
      element.dataset.collisionActive === 'false' ||
      (kind !== 'animal' && kind !== 'traveler')
    )
      continue
    const matrix = trailScreenMatrix(element)
    const parent = element.parentElement
    const ground = parent instanceof SVGGraphicsElement ? trailScreenMatrix(parent) : null
    const radius = Number(element.dataset.collisionRadius)
    if (!matrix || !ground || !Number.isFinite(radius)) continue
    actors.push({
      id,
      kind,
      x: matrix.e + window.scrollX,
      y: matrix.f + window.scrollY,
      radius: radius * Math.hypot(ground.a, ground.b),
      resting: kind === 'animal' && element.dataset.phase === 'resting',
      element,
    })
  }
  return actors
}
export const visibleTrailRun = (obstacles: PetObstacles): TrailRun | null => {
  const text = { ...obstacles, camps: [] }
  for (const { path, feet, key } of trailRoutes()) {
    const actors = trailActors(path.ownerSVGElement)
    const cursor = petState.trailJourney?.key === key ? petState.trailJourney.progress : 0
    const runs: TrailRun[] = []
    let run: TrailFoot[] = []
    let length = 0
    const finish = () => {
      if (length >= 50) runs.push({ path, feet: run, length })
      run = []
      length = 0
    }
    for (const foot of feet) {
      if (
        foot.progress < cursor ||
        !trailPointClear(foot, path, text) ||
        (!run.length &&
          actors.some(
            other =>
              circleClearance(foot, TRAIL_PET_RADIUS * trailPx(path) * foot.scale, other) < 0,
          ))
      ) {
        finish()
        continue
      }
      const previous = run[run.length - 1]
      if (previous) length += Math.hypot(foot.x - previous.x, foot.y - previous.y)
      run.push(foot)
    }
    finish()
    if (runs.length) return runs[0]
  }
  return null
}

export const planTrailWalk = (
  feet: TrailFoot[],
  path: SVGPathElement,
  bodies: readonly CollisionActor[],
  obstacles: PetObstacles,
) => {
  const search = planTrailWalkSteps(feet, path, bodies, obstacles)
  let step = search.next()
  while (!step.done) step = search.next()
  return step.value
}
export function* planTrailWalkSteps(
  feet: TrailFoot[],
  path: SVGPathElement,
  bodies: readonly CollisionActor[],
  obstacles: PetObstacles,
): PetTrailSearch {
  const scale = Math.max(...feet.map(foot => foot.scale))
  const bounds: SpacingBounds = {
    left: window.scrollX + margin + (petState.width * scale) / 2,
    right: window.scrollX + window.innerWidth - margin - (petState.width * scale) / 2,
    top: window.scrollY + margin + (edgeY.stand - viewTop) * petState.unit * scale,
    bottom:
      window.scrollY +
      window.innerHeight -
      margin -
      (viewBottom - edgeY.stand) * petState.unit * scale,
  }
  const planned = yield* detourPetTrailSteps(feet, TRAIL_PET_RADIUS * trailPx(path), bodies, bounds)
  if (!planned) return null
  for (const foot of planned) {
    yield
    if (!trailPointClear(foot, path, obstacles)) return null
  }
  return planned
}

export function hopArc(
  from: Point,
  to: Point,
  fromScale: number,
  toScale: number,
  top: number,
): [HopPoint, HopPoint, HopPoint] {
  const distance = Math.hypot(to.x - from.x, to.y - from.y)
  const peak = Math.min(180, 50 + distance * 0.3, Math.max(0, Math.min(from.y, to.y) - top))
  return [
    { ...from, scale: fromScale },
    { x: (from.x + to.x) / 2, y: Math.min(from.y, to.y) - peak, scale: (fromScale + toScale) / 2 },
    { ...to, scale: toScale },
  ]
}

export function trailHopClear(next: Perch, obstacles: PetObstacles) {
  if (
    !petState.doc ||
    petState.doc.y < window.scrollY ||
    petState.doc.y > window.scrollY + window.innerHeight
  )
    return false
  const target = placeOn(next)
  if (Math.hypot(target.x - petState.doc.x, target.y - petState.doc.y) > 900) return false
  const arc = hopArc(
    petState.doc,
    target,
    petState.perch?.scale ?? 1,
    next.scale ?? 1,
    window.scrollY,
  )
  for (let segment = 1; segment < arc.length; segment++) {
    const from = arc[segment - 1]
    const to = arc[segment]
    const count = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / 6))
    for (let index = 0; index <= count; index++) {
      const fraction = index / count
      const scale = from.scale + (to.scale - from.scale) * fraction
      const at = {
        x: from.x + (to.x - from.x) * fraction + (petState.width * scale) / 2 - window.scrollX,
        y:
          from.y +
          (to.y - from.y) * fraction +
          (edgeY.stand - viewTop) * petState.unit * scale -
          window.scrollY,
      }
      if (coversText({ ...next, kind: 'stand', scale, at, score: 0 }, obstacles)) return false
    }
  }
  return true
}
