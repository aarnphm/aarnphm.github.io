import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { CollisionActor } from './actor-spacing'
import type { Point } from './landscape'
import { detourTrail, replanOccupiedWaypoints } from './obstacle-route'

function sweptClearance(from: Point, to: Point, obstacle: CollisionActor, radius: number) {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const fraction = Math.max(
    0,
    Math.min(
      1,
      ((obstacle.x - from.x) * dx + (obstacle.y - from.y) * dy) / (dx * dx + dy * dy || 1),
    ),
  )
  return (
    Math.hypot(from.x + fraction * dx - obstacle.x, from.y + fraction * dy - obstacle.y) -
    obstacle.radius -
    radius
  )
}

test('walk around overlapping camp bodies and reconnect on the far side', () => {
  const obstacles = [
    { id: 'hearth', kind: 'trunk', x: 95, y: 95, radius: 16 },
    { id: 'tent', kind: 'trunk', x: 118, y: 78, radius: 16 },
  ] satisfies CollisionActor[]
  const road = Array.from({ length: 19 }, (_, index) => ({ x: 10 + index * 10, y: 95 }))
  const bounds = { left: 0, right: 210, top: 0, bottom: 180 }
  const route = detourTrail(road, 18, obstacles, bounds)
  assert.ok(route)
  assert.ok(route.detours > 0)
  assert.deepEqual(route.points[0], road[0])
  assert.deepEqual(route.points.at(-1), road.at(-1))
  assert.ok(route.points.some(point => Math.abs(point.y - 95) > 25))
  for (const [index, point] of route.points.entries()) {
    assert.ok(point.x >= bounds.left && point.x <= bounds.right)
    assert.ok(point.y >= bounds.top && point.y <= bounds.bottom)
    if (!index) continue
    for (const obstacle of obstacles)
      assert.ok(sweptClearance(route.points[index - 1], point, obstacle, 18) >= 1.99)
  }
})

test('include a neighbouring obstruction beyond the direct road when planning a detour', () => {
  const obstacles = [50, 130, 210, 290].map(
    (y, index): CollisionActor => ({ id: `tree-${index}`, kind: 'trunk', x: 100, y, radius: 30 }),
  )
  const from = { x: 10, y: 130 }
  const to = { x: 200, y: 130 }
  const route = detourTrail([from, to], 25, obstacles, { left: 0, right: 240, top: 0, bottom: 400 })
  assert.ok(route)
  assert.deepEqual(route.points[0], from)
  assert.deepEqual(route.points.at(-1), to)
  assert.ok(route.points.some(point => point.y > 340))
  for (let index = 1; index < route.points.length; index++)
    for (const obstacle of obstacles)
      assert.ok(sweptClearance(route.points[index - 1], route.points[index], obstacle, 25) >= 1.99)
})

test('arrive at a reachable edge when the destination clearing is enclosed', () => {
  const obstacles = Array.from(
    { length: 8 },
    (_, index): CollisionActor => ({
      id: `tree-${index}`,
      kind: 'trunk',
      x: 100 + Math.cos((index * Math.PI) / 4) * 40,
      y: 130 + Math.sin((index * Math.PI) / 4) * 40,
      radius: 15,
    }),
  )
  const goal = { x: 100, y: 130 }
  const route = detourTrail([{ x: 10, y: 130 }, goal], 12, obstacles, {
    left: 0,
    right: 240,
    top: 0,
    bottom: 260,
  })
  assert.ok(route)
  const arrival = route.points.at(-1)
  assert.ok(arrival)
  assert.ok(Math.hypot(arrival.x - goal.x, arrival.y - goal.y) <= 96)
  assert.ok(Math.hypot(arrival.x - goal.x, arrival.y - goal.y) > 60)
  for (let index = 1; index < route.points.length; index++)
    for (const obstacle of obstacles)
      assert.ok(sweptClearance(route.points[index - 1], route.points[index], obstacle, 12) >= 1.99)
})

test('report a closed corridor instead of cutting through a camp body', () => {
  const obstacles = [
    { id: 'tent', kind: 'trunk', x: 100, y: 50, radius: 28 },
  ] satisfies CollisionActor[]
  const route = detourTrail(
    [
      { x: 10, y: 50 },
      { x: 190, y: 50 },
    ],
    18,
    obstacles,
    { left: 0, right: 200, top: 20, bottom: 80 },
  )
  assert.equal(route, null)
})

test('bypass an enclosed intermediate road point and continue to the destination', () => {
  const obstacles = Array.from(
    { length: 8 },
    (_, index): CollisionActor => ({
      id: `tree-${index}`,
      kind: 'trunk',
      x: 100 + Math.cos((index * Math.PI) / 4) * 40,
      y: 100 + Math.sin((index * Math.PI) / 4) * 40,
      radius: 15,
    }),
  )
  const road = [
    { x: 10, y: 100 },
    { x: 100, y: 100 },
    { x: 190, y: 100 },
  ]
  const route = detourTrail(road, 18, obstacles, { left: 0, right: 200, top: 0, bottom: 200 })
  assert.ok(route)
  assert.deepEqual(route.points[0], road[0])
  assert.deepEqual(route.points.at(-1), road.at(-1))
  assert.ok(route.points.some(point => Math.abs(point.y - 100) > 60))
  for (let index = 1; index < route.points.length; index++)
    for (const obstacle of obstacles)
      assert.ok(sweptClearance(route.points[index - 1], route.points[index], obstacle, 18) >= 1.99)
})

test('walk around a pet covering the next waypoint without entering its body', () => {
  const from = { x: 1078.39, y: 1080.99 }
  const waypoints = [
    { x: 1067.34, y: 1067.48 },
    { x: 1045.3, y: 980 },
  ]
  const pet: CollisionActor = { id: 'pet-blob', kind: 'pet', x: 1045.3, y: 1053.06, radius: 25.2 }
  const points = replanOccupiedWaypoints('builder-0-0', from, waypoints, 12, [pet], {
    left: 18,
    right: 1421,
    top: 765,
    bottom: 1479,
  })
  assert.ok(points)
  assert.deepEqual(points.at(-1), waypoints.at(-1))
  const road = [from, ...points]
  for (let index = 1; index < road.length; index++)
    assert.ok(sweptClearance(road[index - 1], road[index], pet, 12) >= 1.99)
})

test('replan a blocked camp passage when the next waypoint is clear', () => {
  const from = { x: 10, y: 95 }
  const waypoints = [
    { x: 190, y: 95 },
    { x: 190, y: 150 },
  ]
  const obstacles = [
    { id: 'hearth', kind: 'trunk', x: 95, y: 95, radius: 16 },
    { id: 'tent', kind: 'trunk', x: 118, y: 78, radius: 16 },
  ] satisfies CollisionActor[]
  const bounds = { left: 0, right: 220, top: 0, bottom: 200 }
  const points = replanOccupiedWaypoints('park-ranger-0', from, waypoints, 18, obstacles, bounds)
  assert.ok(points)
  assert.deepEqual(points.at(-1), waypoints.at(-1))
  assert.ok(points.some(point => Math.abs(point.y - from.y) > 25))
  const road = [from, ...points]
  for (let index = 1; index < road.length; index++)
    for (const obstacle of obstacles)
      assert.ok(sweptClearance(road[index - 1], road[index], obstacle, 18) >= 1.99)
})

test('walk around a stationary traveler between clear waypoints', () => {
  const from = { x: 20, y: 100 }
  const goal = { x: 180, y: 100 }
  const reader: CollisionActor = { id: 'reader-0', kind: 'traveler', x: 100, y: 100, radius: 12 }
  const points = replanOccupiedWaypoints('park-ranger-0', from, [goal], 18, [reader], {
    left: 0,
    right: 200,
    top: 0,
    bottom: 200,
  })
  assert.ok(points)
  assert.deepEqual(points.at(-1), goal)
  const road = [from, ...points]
  for (let index = 1; index < road.length; index++)
    assert.ok(sweptClearance(road[index - 1], road[index], reader, 18) >= 1.99)
})

test('retain animal right-of-way when an animal covers the next waypoint', () => {
  const animal: CollisionActor = { id: 'animal-cow-0', kind: 'animal', x: 100, y: 100, radius: 18 }
  assert.equal(
    replanOccupiedWaypoints(
      'builder-0-0',
      { x: 50, y: 100 },
      [
        { x: 100, y: 100 },
        { x: 150, y: 100 },
      ],
      12,
      [animal],
      { left: 0, right: 200, top: 0, bottom: 200 },
    ),
    null,
  )
  assert.equal(
    replanOccupiedWaypoints(
      'park-ranger-0',
      { x: 20, y: 100 },
      [{ x: 180, y: 100 }],
      18,
      [animal],
      { left: 0, right: 200, top: 0, bottom: 200 },
    ),
    null,
  )
})

test('a waiting traveler gives a resting animal room on an alternative road', () => {
  const animal: CollisionActor = {
    id: 'animal-cow-0',
    kind: 'animal',
    x: 100,
    y: 100,
    radius: 18,
    resting: true,
  }
  const from = { x: 50, y: 100 }
  const points = replanOccupiedWaypoints(
    'reader-3',
    from,
    [
      { x: 100, y: 100 },
      { x: 190, y: 100 },
    ],
    12,
    [animal],
    { left: 0, right: 220, top: 0, bottom: 220 },
    animal.id,
  )
  assert.ok(points)
  assert.deepEqual(points.at(-1), { x: 190, y: 100 })
  assert.ok(points[0].x < from.x)
  const road = [from, ...points]
  for (let index = 1; index < road.length; index++)
    assert.ok(sweptClearance(road[index - 1], road[index], animal, 12) >= 1.99)
})

test('a waiting traveler continues yielding while the animal is crossing', () => {
  const animal: CollisionActor = {
    id: 'animal-cow-0',
    kind: 'animal',
    x: 100,
    y: 100,
    radius: 18,
    resting: false,
  }
  assert.equal(
    replanOccupiedWaypoints(
      'reader-3',
      { x: 50, y: 100 },
      [
        { x: 100, y: 100 },
        { x: 190, y: 100 },
      ],
      12,
      [animal],
      { left: 0, right: 220, top: 0, bottom: 220 },
      animal.id,
    ),
    null,
  )
})
