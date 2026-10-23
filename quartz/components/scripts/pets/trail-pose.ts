import { trailHost } from '../../../util/trail-scene'
import { footprintTransform, impression } from '../../visitor-trail/footprints'
import { TRAIL_PET_RADIUS, trailPetSprite, trailPetTop } from '../../visitor-trail/trail-pet'
import {
  clamp,
  edgeY,
  petState,
  scheduleWander,
  trailScreenMatrix,
  trailState,
  transformFor,
  viewBottom,
  viewTop,
  type TrailFoot,
  type TrailWalk,
} from './actor-model'
import { armQuietTimer, render, syncStorm, syncTrailArt, weatherState } from './appearance'
import { placeOn } from './perch-placement'
import { currentTrailPath, pair, trailData, trailGeometry, trailKey } from './trail-geometry'

const currentTrailProgress = () => {
  const current = petState.trailWalk
  if (!current) return null
  if (!petState.hop || !current.duration)
    return petState.perch?.progress ?? current.feet[0].progress
  return trailFootAt(current, Number(petState.hop.currentTime)).progress
}
export const trailFootAt = (current: TrailWalk, time: number): TrailFoot => {
  const elapsed = clamp(time / Math.max(1, current.duration), 0, 1)
  const index = current.offsets.findIndex(offset => offset >= elapsed)
  if (index <= 0) return current.feet[index === 0 ? 0 : current.feet.length - 1]
  const from = current.feet[index - 1]
  const to = current.feet[index]
  const fraction =
    (elapsed - current.offsets[index - 1]) /
    Math.max(0.000001, current.offsets[index] - current.offsets[index - 1])
  return {
    x: from.x + (to.x - from.x) * fraction,
    y: from.y + (to.y - from.y) * fraction,
    scale: from.scale + (to.scale - from.scale) * fraction,
    progress: from.progress + (to.progress - from.progress) * fraction,
  }
}

export const stopTrailGuard = () => {
  window.clearTimeout(trailState.trailGuardTimer)
  trailState.trailGuardTimer = 0
  cancelAnimationFrame(trailState.trailPaintFrame)
  trailState.trailPaintFrame = 0
  cancelAnimationFrame(trailState.trailPlanFrame)
  trailState.trailPlanFrame = 0
  if (petState.root) {
    petState.root.dataset.trailGuard = 'paused'
    petState.root.dataset.yieldingTo = ''
  }
  if (trailState.trailActor) {
    trailState.trailActor.dataset.yieldingTo = ''
    trailState.trailActor.dataset.motion = 'still'
  }
}
export const clearTrailActor = () => {
  stopTrailGuard()
  trailState.trailActor?.remove()
  trailState.trailActor = null
  trailState.trailArt = null
  trailState.trailHeadY = null
  trailState.trailPrint = null
  petState.root?.removeAttribute('data-trail-paint')
  for (const property of ['--pet-trail-top', '--pet-trail-foot', '--pet-trail-unscale'])
    petState.root?.style.removeProperty(property)
}
export const reserveTrailFoot = (path: SVGPathElement, foot: TrailFoot) => {
  if (!currentTrailPath(path)) return clearTrailActor()
  const scene = path.ownerSVGElement
  const matrix = scene ? trailScreenMatrix(scene) : null
  if (!scene || !matrix || !petState.svg || !petState.root) return
  const drawing = scene.closest<HTMLElement>('.trail-scene')
  const span = trailData(path.dataset.petAlpineSpan)
  const alpine = pair(span) && foot.progress <= span[1]
  let changed = false
  if (trailState.trailActor?.ownerSVGElement !== scene || !trailState.trailActor.isConnected) {
    trailState.trailActor?.remove()
    trailState.trailActor = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailState.trailActor.dataset.trailCollision = 'pet'
    trailState.trailActor.dataset.collisionId = 'pet-blob'
    trailState.trailActor.dataset.trailPet = ''
    trailState.trailActor.setAttribute('aria-hidden', 'true')
    trailState.trailArt = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailState.trailArt.setAttribute('class', 'trail-pet-art')
    // Animation events do not cross the scene's shadow/iframe boundary.
    trailState.trailArt.addEventListener('animationstart', syncStorm, {
      signal: petState.trailSpaceSignal,
    })
    trailState.trailArt.innerHTML = trailPetSprite()
    trailState.trailActor.append(trailState.trailArt)
    scene.append(trailState.trailActor)
    petState.root.dataset.trailPaint = 'scene'
    syncTrailArt()
    changed = true
  }
  if (trailState.trailActor.dataset.petAlpine !== String(alpine)) {
    trailState.trailActor.dataset.petAlpine = String(alpine)
    changed = true
  }
  if (changed) drawing?.dispatchEvent(new Event('trail:depth-change'))
  const parent = trailState.trailActor.parentElement
  const ground = parent instanceof SVGGraphicsElement ? trailScreenMatrix(parent) : matrix
  if (!ground) return
  const point = new DOMPoint(foot.x - window.scrollX, foot.y - window.scrollY).matrixTransform(
    ground.inverse(),
  )
  // Scene units at the ground scale, as the folk on the trail are drawn, carried into the parent
  // the depth order put it in.
  const scale = (foot.scale * Math.hypot(matrix.a, matrix.b)) / Math.hypot(ground.a, ground.b)
  const transform = `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`
  if (trailState.trailActor.getAttribute('transform') !== transform)
    trailState.trailActor.setAttribute('transform', transform)
  const pose = `scale(${(scale * petState.face).toFixed(4)} ${scale.toFixed(4)})`
  if (trailState.trailArt?.getAttribute('transform') !== pose)
    trailState.trailArt?.setAttribute('transform', pose)
  if (trailState.trailActor.dataset.face !== String(petState.face))
    trailState.trailActor.dataset.face = String(petState.face)
  trailState.trailActor.dataset.collisionRadius = String(TRAIL_PET_RADIUS * scale)
  trailState.trailActor.dataset.collisionActive = 'true'
  trailState.trailActor.dataset.groundX = String(point.x)
  trailState.trailActor.dataset.groundY = String(point.y)
  trailState.trailActor.dataset.progress = foot.progress.toFixed(6)
  trailState.trailActor.dataset.scale = String(foot.scale)
  // The speech bubble, in the root's own pixels: the root is scaled by the foot's ground scale with
  // the hidden vector pet's feet on the trail, so the pixel pet's head is its height in scene units
  // above them. The bubble undoes the scale to read at the page's size.
  const head = trailPetTop(weatherState.rain) * Math.hypot(matrix.a, matrix.b)
  const feet = (edgeY.stand - viewTop) * petState.unit
  const gap = 4 / foot.scale
  trailState.trailHeadY = foot.y - head * foot.scale
  const bubble: [string, string][] = [
    [
      '--pet-trail-top',
      `${((viewBottom - edgeY.stand) * petState.unit + head + gap).toFixed(1)}px`,
    ],
    ['--pet-trail-foot', `${(feet + gap).toFixed(1)}px`],
    ['--pet-trail-unscale', (1 / foot.scale).toFixed(4)],
  ]
  for (const [property, value] of bubble)
    if (petState.root.style.getPropertyValue(property) !== value)
      petState.root.style.setProperty(property, value)
  leaveTrailPrint(scene, foot)
}
// One blob print a stride along the walk, on alternating sides; muddy in the rain. A jump longer
// than a few strides (a hop on, a new perch) starts the line again without a print.
const leaveTrailPrint = (scene: SVGSVGElement, foot: TrailFoot) => {
  const layer = scene.querySelector<SVGGElement>('.trail-visitor-tracks')
  const plane = layer ? trailScreenMatrix(layer) : null
  const matrix = trailScreenMatrix(scene)
  if (!layer || !plane || !matrix) return
  if (trailState.trailTracks?.parentNode !== layer) {
    trailState.trailTracks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    trailState.trailTracks.setAttribute('class', 'trail-pet-tracks')
    layer.append(trailState.trailTracks)
    trailState.trailPrint = null
  }
  const at = new DOMPoint(foot.x - window.scrollX, foot.y - window.scrollY).matrixTransform(
    plane.inverse(),
  )
  const size = (foot.scale * Math.hypot(matrix.a, matrix.b)) / Math.hypot(plane.a, plane.b)
  const stride = 9 * size
  if (!trailState.trailPrint) {
    trailState.trailPrint = { x: at.x, y: at.y, side: 1 }
    return
  }
  const step = Math.hypot(at.x - trailState.trailPrint.x, at.y - trailState.trailPrint.y)
  if (step > stride * 4) {
    trailState.trailPrint = { x: at.x, y: at.y, side: trailState.trailPrint.side }
    return
  }
  if (step < stride) return
  const angle = Math.atan2(at.y - trailState.trailPrint.y, at.x - trailState.trailPrint.x)
  const mark = impression(document, 'blob')
  mark.dataset.footprint = 'blob'
  mark.dataset.footprintOwner = 'pet'
  if (weatherState.rain) mark.dataset.wet = ''
  mark.setAttribute(
    'transform',
    `${footprintTransform(at.x, at.y, angle, trailState.trailPrint.side, 3 * size)} scale(${(0.9 * size).toFixed(4)})`,
  )
  trailState.trailTracks.append(mark)
  while (trailState.trailTracks.childElementCount > 48)
    trailState.trailTracks.firstElementChild?.remove()
  trailState.trailPrint = { x: at.x, y: at.y, side: -trailState.trailPrint.side }
}
export const paintTrail = (animation: Animation, path: SVGPathElement) => {
  const paint = () => {
    trailState.trailPaintFrame = 0
    if (petState.hop !== animation || !petState.trailWalk || !petState.root || document.hidden)
      return
    if (!currentTrailPath(path) || trailHost(path)?.hidden !== false) return pauseTrail()
    const time = Number(animation.currentTime)
    const foot = trailFootAt(petState.trailWalk, time)
    const ahead = trailFootAt(petState.trailWalk, time + 120)
    const dx = ahead.x - foot.x
    if (Math.abs(dx) > 0.25 * foot.scale) {
      petState.face = dx > 0 ? -1 : 1
      petState.gazeArt?.face(petState.face)
    }
    reserveTrailFoot(path, foot)
    trailState.trailPaintFrame = requestAnimationFrame(paint)
  }
  paint()
}

export const rememberTrail = () => {
  const progress = currentTrailProgress()
  if (petState.trailWalk && progress !== null)
    petState.trailJourney = { key: trailKey(petState.trailWalk.path), progress }
}

const stopTrail = (mode: 'pause' | 'leave') => {
  if (!petState.trailWalk) return
  const rect = petState.root?.isConnected ? petState.root.getBoundingClientRect() : null
  const scale = rect ? rect.width / petState.width : (petState.perch?.scale ?? 1)
  const position = rect
    ? { x: rect.left + window.scrollX, y: rect.top + window.scrollY }
    : petState.doc
  const progress = currentTrailProgress() ?? 0
  const foot: TrailFoot | undefined =
    rect && position
      ? {
          x: position.x + rect.width / 2,
          y: position.y + (edgeY.stand - viewTop) * petState.unit * scale,
          scale,
          progress,
        }
      : petState.perch?.foot
        ? { ...petState.perch.foot, scale, progress }
        : undefined
  const next =
    mode === 'leave' && foot ? placeOn({ el: null, kind: 'stand', t: 0, foot }) : position
  const transform = next ? transformFor(next, mode === 'leave' ? 1 : scale) : null
  rememberTrail()
  stopTrailGuard()
  const path = petState.trailWalk.path
  petState.hop?.cancel()
  petState.hop = null
  petState.afterMove = null
  petState.override = null
  petState.resizeObserver?.disconnect()
  cancelAnimationFrame(petState.anchorFrame)
  if (mode === 'leave') {
    // Leaving removes the trail twin directly; painting an intermediate paused pose forces layout.
    clearTrailActor()
    petState.trailWalk = null
    petState.perch = null
  } else if (petState.root && rect && foot) {
    petState.perch = { el: path, kind: 'stand', t: 0, foot, scale, progress: foot.progress }
    petState.trailWalk = {
      path,
      geometry: trailGeometry(path),
      feet: [foot],
      duration: 0,
      offsets: [0],
    }
  }
  petState.doc = next
  if (petState.root && transform) petState.root.style.transform = transform
  if (mode === 'pause' && petState.root && rect && foot) reserveTrailFoot(path, foot)
  petState.settled = performance.now()
  render()
  armQuietTimer()
  scheduleWander()
  return { position, scale }
}

export const pauseTrail = () => stopTrail('pause')
export const leaveTrail = () => stopTrail('leave')
