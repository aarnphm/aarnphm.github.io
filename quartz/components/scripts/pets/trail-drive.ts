import { canMove, circleClearance, type CollisionActor } from '../../visitor-trail/actor-spacing'
import { pondDistance } from '../../visitor-trail/landscape'
import { TRAIL_PET_RADIUS } from '../../visitor-trail/trail-pet'
import {
  center,
  onClock,
  petState,
  reducedMotion,
  scheduleWander,
  trailScreenMatrix,
  transformFor,
  type Perch,
  type Point,
  type TrailFoot,
} from './actor-model'
import { armQuietTimer, render, syncTrailArt } from './appearance'
import { moveTo, savePosition } from './perch-motion'
import { campSolids, petObstacles, type PetObstacles } from './perch-placement'
import { trailActors, trailHopClear, trailPointClear } from './trail-clearance'
import {
  expeditionPath,
  nearestTrailProgress,
  petTerrain,
  trailGeometry,
  trailKey,
  trailPx,
  trailRoutes,
  trailScale,
} from './trail-geometry'
import { pauseTrail, reserveTrailFoot } from './trail-pose'
import { follow, walkTrail } from './trail-walk'

// Arrow keys walk the pet about the trail: a press hops it onto the walking path if it is not on
// it, then each step goes a short way in the keys' direction, at the folk's ground scale for that
// depth, round trunks, tents, animals and the pond, and stops short of the page's text. The keys
// are taken while the trail is open and nothing else holds the focus, so the page does not scroll.
export const driveKeys = new Set<string>()
const arrowKeys: Record<string, Point> = {
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
}
const driveStride = 18
const driveSpeed = 150
const driveState = {
  // The step under way, so a held key's repeats leave it to finish; the obstacle scan is shared
  // by the steps of one burst (a fresh scan each step made the page stutter).
  driving: null as Animation | null,
  // A tap that lands while a step is under way still gets its own step after it.
  driveQueued: null as Point | null,
  driveScan: null as { at: number; text: PetObstacles; actors: CollisionActor[] } | null,
}
const driveObstacles = (svg: SVGSVGElement) => {
  const now = performance.now()
  if (!driveState.driveScan || now - driveState.driveScan.at > 400) {
    driveState.driveScan = {
      at: now,
      text: { ...petObstacles(), camps: [] },
      actors: [...trailActors(svg), ...campSolids()],
    }
  }
  return driveState.driveScan
}

const driveTakesKeys = (event: KeyboardEvent) => {
  if (!(event.key in arrowKeys) || event.altKey || event.ctrlKey || event.metaKey) return false
  if (!petState.root || petState.carry || reducedMotion() || !expeditionPath()) return false
  const active = document.activeElement
  return (
    !active ||
    active === document.body ||
    !!active.closest('.garden-pet, .homepage-trail-toggle, .homepage-trail-scene')
  )
}

// Hops onto the nearest clear spot of the walking path in view, down in the valley. A stroll or
// hop under way is cut where the pet is, so the hop starts from where it stands.
// Puts the pet on the nearest clear foot of the route and hands over once it stands there.
// Stops the move under way where it is, so the next move starts from there.
export const dropHop = () => {
  if (!petState.root || !petState.hop) return
  const rect = petState.root.getBoundingClientRect()
  petState.hop.cancel()
  petState.hop = null
  petState.override = null
  petState.afterMove = null
  petState.doc = { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
  petState.root.style.transform = transformFor(petState.doc, rect.width / petState.width)
}
const landOnTrail = (path: SVGPathElement, then: () => void) => {
  if (!petState.root) return false
  dropHop()
  const obstacles = petObstacles()
  const text = { ...obstacles, camps: [] }
  const actors = [...trailActors(path.ownerSVGElement), ...campSolids()]
  const here = center()
  const from = { x: here.x + window.scrollX, y: here.y + window.scrollY }
  let start: TrailFoot | null = null
  let near = Infinity
  for (const route of trailRoutes()) {
    if (route.path !== path) continue
    for (const foot of route.feet) {
      if (foot.scale < 0.35 || !trailPointClear(foot, path, text)) continue
      const radius = TRAIL_PET_RADIUS * trailPx(path) * foot.scale
      if (actors.some(other => circleClearance(foot, radius, other) < 0)) continue
      const d = Math.hypot(foot.x - from.x, foot.y - from.y)
      if (d < near) {
        near = d
        start = foot
      }
    }
  }
  if (!start) return false
  const foot = start
  petState.afterMove = () => {
    petState.trailWalk = {
      path,
      geometry: trailGeometry(path),
      feet: [foot],
      duration: 0,
      offsets: [0],
    }
    petState.trailJourney = { key: trailKey(path), progress: foot.progress }
    reserveTrailFoot(path, foot)
    petState.settled = performance.now()
    render()
    then()
  }
  const next: Perch = {
    el: path,
    kind: 'stand',
    t: 0,
    foot,
    scale: foot.scale,
    progress: foot.progress,
  }
  moveTo(next, trailHopClear(next, obstacles) ? null : 'fade')
  return true
}
const boardTrail = (path: SVGPathElement) => landOnTrail(path, driveStep)
// The scene has just opened: the pet appears on the route at once and sets off along it, or, with
// no walkable run in view, lands on the nearest clear foot and leaves the rest to the wander.
export const enterTrail = (obstacles: PetObstacles) => {
  dropHop()
  if (walkTrail(obstacles, onClock() ? 'hop' : 'fade')) return true
  const path = expeditionPath()
  if (!path) return false
  return landOnTrail(path, () => {
    armQuietTimer()
    scheduleWander()
  })
}
const driveStep = () => {
  if (!petState.root || petState.carry || reducedMotion() || document.hidden) return
  const path = expeditionPath()
  if (!path) return
  let dx = 0
  let dy = 0
  for (const key of driveKeys) {
    dx += arrowKeys[key].x
    dy += arrowKeys[key].y
  }
  if (!dx && !dy && driveState.driveQueued) {
    dx = driveState.driveQueued.x
    dy = driveState.driveQueued.y
  }
  driveState.driveQueued = null
  if (!dx && !dy) {
    driveState.driveScan = null
    if (petState.trailWalk && petState.perch?.foot && !petState.hop) {
      // Back on the expedition's clock from the nearest point of its path.
      const progress = nearestTrailProgress(path, petState.perch.foot)
      petState.perch.progress = progress
      petState.trailJourney = { key: trailKey(path), progress }
      reserveTrailFoot(path, { ...petState.perch.foot, scale: petState.perch.scale ?? 1, progress })
      petState.settled = performance.now()
      armQuietTimer()
      scheduleWander()
      savePosition()
    }
    return
  }
  if (petState.hop && petState.hop === driveState.driving) {
    driveState.driveQueued = { x: dx, y: dy }
    return
  }
  // A key mid-stroll takes the walk over from where the pet stands.
  if (petState.hop && petState.trailWalk?.duration) pauseTrail()
  if (petState.hop) return
  if (!petState.trailWalk || !petState.perch?.foot) return boardTrail(path)
  const svg = path.ownerSVGElement
  const matrix = svg && trailScreenMatrix(svg)
  const terrain = petTerrain(path)
  if (!svg || !matrix || !terrain) return
  const px = Math.hypot(matrix.a, matrix.b)
  const from: TrailFoot = {
    ...petState.perch.foot,
    scale: petState.perch.scale ?? 1,
    progress: petState.perch.progress ?? 0,
  }
  const length = Math.hypot(dx, dy)
  const { text, actors } = driveObstacles(svg)
  // The scene's own box, in scene units: the drawing may bleed past it for parallax.
  const drawing = path.closest<HTMLElement>('.trail-scene')
  const width = drawing?.clientWidth || svg.viewBox.baseVal.width
  const height = drawing?.clientHeight || svg.viewBox.baseVal.height
  const inverse = matrix.inverse()
  // The walk keeps below the treeline; where the route itself put the pet higher (its alpine
  // stretch) it may go level or down from there.
  const here = new DOMPoint(from.x - window.scrollX, from.y - window.scrollY).matrixTransform(
    inverse,
  )
  const top = Math.min(terrain.treeline + 6, here.y)
  // The full direction first, then along one axis only, so it slides along what blocks it.
  const tries = [
    { x: dx / length, y: dy / length },
    ...(dx && dy
      ? [
          { x: Math.sign(dx), y: 0 },
          { x: 0, y: Math.sign(dy) },
        ]
      : []),
  ]
  for (const dir of tries) {
    const stride = driveStride * from.scale * px
    const to = { x: from.x + dir.x * stride, y: from.y + dir.y * stride }
    const ground = new DOMPoint(to.x - window.scrollX, to.y - window.scrollY).matrixTransform(
      inverse,
    )
    if (ground.x < 14 || ground.x > width - 14) continue
    if (ground.y < top || ground.y > height - 12) continue
    if (terrain.pond && pondDistance(ground, terrain.pond) < 1.15) continue
    const scale = trailScale(0, ground, [], terrain)
    const foot: TrailFoot = { ...to, scale, progress: from.progress }
    if (!trailPointClear(foot, path, text)) continue
    const radius = TRAIL_PET_RADIUS * px * Math.max(from.scale, scale)
    if (!canMove(from, foot, radius, actors)) continue
    petState.lastMove = performance.now()
    follow([from, foot], path, { speed: driveSpeed, done: driveStep })
    driveState.driving = petState.hop
    return
  }
  // Blocked all round: it still turns to face the way it was asked.
  if (dx) {
    petState.face = dx > 0 ? -1 : 1
    petState.gazeArt?.face(petState.face)
    syncTrailArt()
  }
}
export const onDriveKey = (event: KeyboardEvent) => {
  if (!driveTakesKeys(event)) return
  event.preventDefault()
  driveKeys.add(event.key)
  driveStep()
}
export const onDriveKeyUp = (event: KeyboardEvent) => {
  if (!driveKeys.delete(event.key)) return
  if (!driveKeys.size && !petState.hop) driveStep()
}
