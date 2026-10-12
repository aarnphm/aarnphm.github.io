import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { CollisionActor } from './actor-spacing'
import type { Point } from './landscape'
import { detourTrail } from './obstacle-route'

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
