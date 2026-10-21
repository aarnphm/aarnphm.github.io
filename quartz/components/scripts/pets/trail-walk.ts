import { trailHost } from '../../../util/trail-scene'
import {
  animalInWay,
  canMove,
  queuedAnimal,
  yieldPatience,
} from '../../visitor-trail/actor-spacing'
import { TRAIL_PET_RADIUS } from '../../visitor-trail/trail-pet'
import {
  petState,
  reducedMotion,
  scheduleWander,
  trailState,
  transformFor,
  type Perch,
  type TrailFoot,
} from './actor-model'
import { armQuietTimer, render } from './appearance'
import { moveTo, observePerch, savePosition } from './perch-motion'
import { petObstacles, placeOn } from './perch-placement'
import {
  planTrailWalk,
  planTrailWalkSteps,
  trailActors,
  trailHopClear,
  visibleTrailRun,
} from './trail-clearance'
import { currentTrailPath, trailGeometry, trailKey, trailPx } from './trail-geometry'
import { paintTrail, pauseTrail, reserveTrailFoot, stopTrailGuard, trailFootAt } from './trail-pose'

const guardTrail = (animation: Animation, path: SVGPathElement) => {
  stopTrailGuard()
  let blockedSince = 0
  let lastPlan = -Infinity
  let waited: { id: string; since: number } | null = null
  let passing: string | null = null
  const check = () => {
    trailState.trailGuardTimer = 0
    if (!petState.root || petState.hop !== animation || !petState.trailWalk) return stopTrailGuard()
    if (
      !currentTrailPath(path) ||
      petState.trailWalk.geometry !== trailGeometry(path) ||
      document.hidden ||
      reducedMotion() ||
      path.closest<HTMLElement>('.trail-scene')?.dataset.trailRendered !== 'true' ||
      trailHost(path)?.hidden !== false
    )
      return pauseTrail()
    const current = petState.trailWalk
    const time = Number(animation.currentTime)
    const foot = trailFootAt(current, time)
    const next = trailFootAt(current, time + 150)
    const actors = trailActors(path.ownerSVGElement)
    const body = TRAIL_PET_RADIUS * trailPx(path)
    const radius = body * Math.max(foot.scale, next.scale)
    const now = performance.now()
    // As with the trail's walkers, an animal that holds the way past yieldPatience is passed
    // like a trunk. A grazing animal would otherwise keep the pet waiting indefinitely.
    if (
      passing &&
      !actors.some(
        other => other.id === passing && animalInWay(foot, next, radius, [other], passing),
      )
    )
      passing = null
    if (!passing && waited && now - waited.since > yieldPatience) passing = waited.id
    let animal = animalInWay(
      foot,
      next,
      body * foot.scale,
      actors.filter(other => other.id !== passing),
      petState.root.dataset.yieldingTo,
    )
    const blocker = actors.find(
      other =>
        (other.kind !== 'animal' || other.id === passing) && !canMove(foot, next, radius, [other]),
    )
    if (!animal && blocker?.kind === 'traveler') {
      const waitingFor = queuedAnimal('pet-blob', blocker, actors)
      animal = actors.find(other => other.id === waitingFor && other.id !== passing)
    }
    if (!animal) waited = null
    else if (waited?.id !== animal.id) waited = { id: animal.id, since: now }
    if (animal && petState.root.dataset.yieldingTo !== animal.id) trailState.trailYieldEpisode++
    const blocked = !!animal || !!blocker
    if (blocked) blockedSince ||= now
    else blockedSince = 0
    reserveTrailFoot(path, foot)
    // Keep the current foot when replacing the remaining native animation. Fixed obstructions
    // replan at once; a passing person gets time to clear before choosing another road.
    const fixed = blocker?.kind === 'trunk' || (!!passing && blocker?.id === passing)
    if (
      blocked &&
      !trailState.trailPlanFrame &&
      !animal &&
      now - lastPlan >= 750 &&
      (fixed || now - blockedSince >= 500)
    ) {
      lastPlan = now
      const remaining = current.feet.filter(
        (_, index) => current.offsets[index] * current.duration > time,
      )
      const bodies = actors.flatMap(other =>
        other.kind !== 'animal'
          ? [other]
          : other.id === passing
            ? [{ ...other, radius: other.radius + radius * 1.5 + 14 }]
            : [],
      )
      const search = planTrailWalkSteps([foot, ...remaining], path, bodies, petObstacles())
      const advance = () => {
        trailState.trailPlanFrame = 0
        if (
          !petState.root ||
          petState.hop !== animation ||
          animation.playState !== 'paused' ||
          Math.abs(Number(animation.currentTime) - time) > 1
        )
          return
        const deadline = performance.now() + 3
        let result = search.next()
        while (!result.done && performance.now() < deadline) result = search.next()
        if (!result.done) {
          trailState.trailPlanFrame = requestAnimationFrame(advance)
          return
        }
        const planned = result.value
        if (planned && planned.length > 1) {
          petState.root.dataset.trailDetours = String(
            Number(petState.root.dataset.trailDetours ?? 0) + 1,
          )
          follow(planned, path)
        }
      }
      trailState.trailPlanFrame = requestAnimationFrame(advance)
    }
    petState.root.dataset.trailGuard = 'active'
    petState.root.dataset.yieldingTo = animal?.id ?? ''
    if (trailState.trailActor) {
      trailState.trailActor.dataset.motion = blocked ? 'yielding' : 'active'
      trailState.trailActor.dataset.yieldingTo = animal?.id ?? ''
      trailState.trailActor.dataset.yieldEpisode = String(trailState.trailYieldEpisode)
    }
    if (blocked && animation.playState !== 'paused') {
      animation.pause()
      petState.override = { state: 'wait', until: Infinity }
      render()
    } else if (!blocked && animation.playState === 'paused') {
      petState.override = { state: 'walk', until: Infinity }
      animation.play()
      render()
    }
    trailState.trailGuardTimer = window.setTimeout(check, 50)
  }
  check()
}

// Resume the current alpine descent ahead of its cursor.
// Entering from the page it appears on the route (a fade); moves within the trail hop.
export const walkTrail = (obstacles = petObstacles(), arrive: 'hop' | 'fade' = 'hop') => {
  if (!petState.root || petState.hop || petState.carry || reducedMotion() || document.hidden)
    return false
  const best = visibleTrailRun(obstacles)
  if (!best) return false
  const { path, feet } = best
  const route = planTrailWalk(
    feet,
    path,
    trailActors(path.ownerSVGElement).filter(other => other.kind === 'trunk'),
    obstacles,
  )
  if (!route || route.length < 11) return false
  petState.trailWalk = {
    path,
    geometry: trailGeometry(path),
    feet: route,
    duration: 0,
    offsets: [],
  }
  const walk = () => {
    if (
      !currentTrailPath(path) ||
      path.closest<HTMLElement>('.trail-scene')?.dataset.trailRendered !== 'true' ||
      trailHost(path)?.hidden !== false ||
      petState.trailWalk?.geometry !== trailGeometry(path) ||
      reducedMotion() ||
      document.hidden
    ) {
      pauseTrail()
      return
    }
    follow(route, path)
  }
  const start = route[0]
  if (
    petState.perch?.foot &&
    Math.hypot(petState.perch.foot.x - start.x, petState.perch.foot.y - start.y) < 12
  )
    walk()
  else {
    petState.afterMove = walk
    const next: Perch = {
      el: path,
      kind: 'stand',
      t: 0,
      foot: start,
      scale: start.scale,
      progress: start.progress,
    }
    moveTo(next, arrive === 'fade' || !trailHopClear(next, obstacles) ? 'fade' : null)
  }
  return true
}

// Walks a run of trail points at a stroll, ending on the last of them. A driven step (the arrow
// keys) goes at its own pace and hands back to the driver instead of the wander timer.
type Drive = { speed: number; done: () => void }
export function follow(feet: TrailFoot[], el: SVGPathElement, drive?: Drive) {
  if (!petState.root) return
  stopTrailGuard()
  petState.hop?.cancel()
  petState.hop = null
  const to = feet[feet.length - 1]
  petState.perch = { el, kind: 'stand', t: 0, foot: to, scale: to.scale, progress: to.progress }
  petState.doc = placeOn(petState.perch)
  observePerch()
  let duration = 0
  const times = [0]
  for (let k = 1; k < feet.length; k++) {
    const previous = feet[k - 1]
    const point = feet[k]
    const scale = (point.scale + previous.scale) / 2
    duration +=
      (Math.hypot(point.x - previous.x, point.y - previous.y) / ((drive?.speed ?? 55) * scale)) *
      1000
    times.push(duration)
  }
  const offsets = times.map(time => time / Math.max(1, duration))
  petState.trailWalk = { path: el, geometry: trailGeometry(el), feet, duration, offsets }
  petState.override = { state: 'walk', until: Infinity }
  render()
  const anim = petState.root.animate(
    feet.map((foot, index) => ({
      transform: transformFor(
        placeOn({ el, kind: 'stand', t: 0, foot, scale: foot.scale }),
        foot.scale,
      ),
      offset: offsets[index],
    })),
    { duration, easing: 'linear' },
  )
  petState.hop = anim
  guardTrail(anim, el)
  if (petState.hop === anim) paintTrail(anim, el)
  anim.onfinish = () => {
    if (petState.hop !== anim || !petState.root || !petState.doc) return
    stopTrailGuard()
    reserveTrailFoot(el, to)
    petState.root.style.transform = transformFor(petState.doc)
    anim.cancel()
    petState.hop = null
    petState.override = null
    petState.trailJourney = { key: trailKey(el), progress: to.progress }
    petState.trailWalk = {
      path: el,
      geometry: trailGeometry(el),
      feet: [to],
      duration: 0,
      offsets: [0],
    }
    petState.settled = performance.now()
    savePosition()
    render()
    armQuietTimer()
    if (drive) drive.done()
    else scheduleWander()
  }
}
