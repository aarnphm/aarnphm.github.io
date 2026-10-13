import assert from 'node:assert/strict'
import { test } from 'node:test'
import { circleClearance, type CollisionActor } from './actor-spacing'
import { detourPetTrail, type PetTrailFoot } from './pet-trail'

const bounds = { left: 0, right: 320, top: 0, bottom: 260 }
const body = 25.2

function clearSegments(feet: readonly PetTrailFoot[], obstacles: readonly CollisionActor[]) {
  for (let index = 1; index < feet.length; index++) {
    const from = feet[index - 1]
    const to = feet[index]
    const radius = body * Math.max(from.scale, to.scale)
    for (const obstacle of obstacles) {
      const dx = to.x - from.x
      const dy = to.y - from.y
      const fraction = Math.max(
        0,
        Math.min(
          1,
          ((obstacle.x - from.x) * dx + (obstacle.y - from.y) * dy) / (dx * dx + dy * dy || 1),
        ),
      )
      const point = { x: from.x + dx * fraction, y: from.y + dy * fraction }
      assert.ok(circleClearance(point, radius, obstacle) >= -0.01)
    }
    assert.ok(to.progress >= from.progress)
  }
}

test('a growing blob bypasses dense camp and tree waypoints and reaches the same destination', () => {
  const road = Array.from(
    { length: 66 },
    (_, index): PetTrailFoot => ({
      x: 20 + index * 4,
      y: 130,
      scale: 0.18 + index / 65,
      progress: index / 65,
    }),
  )
  const obstacles = [
    { id: 'hearth', kind: 'trunk', x: 100, y: 130, radius: 16 },
    { id: 'tent', kind: 'trunk', x: 127, y: 106, radius: 16 },
    { id: 'tree', kind: 'trunk', x: 220, y: 130, radius: 7 },
  ] satisfies CollisionActor[]
  const feet = detourPetTrail(road, body, obstacles, bounds)
  assert.ok(feet)
  assert.deepEqual(feet[0], road[0])
  assert.deepEqual(feet.at(-1), road.at(-1))
  assert.ok(feet.some(foot => Math.abs(foot.y - 130) > 20))
  clearSegments(feet, obstacles)
})

test('a detour starts at the current animated foot without a jump or a scale reset', () => {
  const road: PetTrailFoot[] = [
    { x: 60, y: 150, scale: 0.7, progress: 0.41 },
    { x: 100, y: 150, scale: 0.85, progress: 0.51 },
    { x: 190, y: 150, scale: 1, progress: 0.7 },
  ]
  const obstacles = [
    { id: 'rock', kind: 'trunk', x: 110, y: 150, radius: 10 },
  ] satisfies CollisionActor[]
  const feet = detourPetTrail(road, body, obstacles, bounds)
  assert.ok(feet)
  assert.deepEqual(feet[0], road[0])
  assert.deepEqual(feet.at(-1), road.at(-1))
  for (const foot of feet) assert.ok(foot.scale >= 0.7 && foot.scale <= 1)
  clearSegments(feet, obstacles)
})

test('a crossing keeps the later leg and its perspective instead of snapping to an earlier leg', () => {
  const road: PetTrailFoot[] = [
    { x: 100, y: 130, scale: 0.18, progress: 0 },
    { x: 220, y: 130, scale: 0.3, progress: 0.2 },
    { x: 220, y: 210, scale: 0.5, progress: 0.4 },
    { x: 80, y: 210, scale: 0.7, progress: 0.6 },
    { x: 80, y: 130, scale: 0.8, progress: 0.7 },
    { x: 100, y: 130, scale: 0.9, progress: 0.8 },
    { x: 210, y: 130, scale: 1, progress: 1 },
  ]
  const obstacles = [
    { id: 'reader', kind: 'traveler', x: 150, y: 130, radius: 10 },
  ] satisfies CollisionActor[]
  const feet = detourPetTrail(road, body, obstacles, bounds)
  assert.ok(feet)
  assert.ok(feet.some(foot => foot.progress > 0.8 && foot.scale > 0.9))
  assert.deepEqual(feet.at(-1), road.at(-1))
  clearSegments(feet, obstacles)
})

test('a closed corridor pauses without cutting through the rock', () => {
  const road: PetTrailFoot[] = [
    { x: 20, y: 130, scale: 1, progress: 0.4 },
    { x: 280, y: 130, scale: 1, progress: 0.8 },
  ]
  const feet = detourPetTrail(
    road,
    body,
    [{ id: 'rock', kind: 'trunk', x: 150, y: 130, radius: 28 }],
    { ...bounds, top: 100, bottom: 160 },
  )
  assert.equal(feet, null)
})

test('the blob arrives beside an occupied camp destination on a continuous road', () => {
  const road: PetTrailFoot[] = [
    { x: 30, y: 130, scale: 0.8, progress: 0.5 },
    { x: 150, y: 130, scale: 1, progress: 1 },
  ]
  const hearth: CollisionActor = { id: 'hearth', kind: 'trunk', x: 150, y: 130, radius: 15 }
  const feet = detourPetTrail(road, body, [hearth], bounds)
  assert.ok(feet)
  assert.deepEqual(feet[0], road[0])
  const arrival = feet.at(-1)
  assert.ok(arrival)
  assert.equal(arrival.progress, 1)
  assert.equal(arrival.scale, 1)
  assert.ok(Math.hypot(arrival.x - hearth.x, arrival.y - hearth.y) < 80)
  clearSegments(feet, [hearth])
})
