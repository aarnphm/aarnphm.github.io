import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  actorSpacing,
  canMove,
  spacedPosition,
  spacedStep,
  yieldPatience,
  type CollisionActor,
} from './actor-spacing'

test('check the full swept segment, including tangency and diagonal crossings', () => {
  const obstacles = [
    { id: 'distant-tree', kind: 'trunk', x: 900, y: 900, radius: 3 },
    { id: 'tree', kind: 'trunk', x: 0, y: 0, radius: 3 },
  ] satisfies CollisionActor[]
  assert.equal(canMove({ x: -20, y: 6.98 }, { x: 20, y: 6.98 }, 2, obstacles), false)
  assert.equal(canMove({ x: -20, y: 7 }, { x: 20, y: 7 }, 2, obstacles), true)
  assert.equal(canMove({ x: -20, y: -20 }, { x: 20, y: 20 }, 2, obstacles), false)
  assert.equal(canMove({ x: 20, y: 20 }, { x: -20, y: -20 }, 2, obstacles), false)
  assert.equal(canMove({ x: 20, y: -20 }, { x: 20, y: 20 }, 2, obstacles), true)
})

test('allow an overlapping actor to escape only while separation increases', () => {
  const obstacles = [
    { id: 'tree', kind: 'trunk', x: 0, y: 0, radius: 3 },
  ] satisfies CollisionActor[]
  const from = { x: 3, y: 0 }
  assert.equal(canMove(from, { x: 4, y: 0 }, 2, obstacles), true)
  assert.equal(canMove(from, { x: 3, y: 1 }, 2, obstacles), true)
  assert.equal(canMove(from, { x: 2, y: 0 }, 2, obstacles), false)
  assert.equal(canMove(from, { x: -10, y: 0 }, 2, obstacles), false)
  assert.equal(canMove(from, from, 2, obstacles), false)
  assert.equal(canMove({ x: 8, y: 0 }, { x: 8, y: 0 }, 2, obstacles), true)
})

test('recover a displaced reader to its endpoint with bounded camp clearance', () => {
  const goal = { x: 865.2657702326774, y: 1135 }
  const spacing = actorSpacing()
  spacing.offset = { x: 928.47 - goal.x, y: 1055.5 - goal.y }
  const bodies = [
    { id: 'tent-0-0', kind: 'trunk', x: 896.2657702326774, y: 1028, radius: 28 },
    { id: 'hearth-0', kind: 'trunk', x: 951.2657702326774, y: 1079, radius: 16 },
  ] satisfies CollisionActor[]
  const budget = 30.7857 / 24
  for (let frame = 0; frame < 600; frame++) {
    const before = spacedPosition(goal, spacing)
    spacedStep(
      'reader-0',
      goal,
      goal,
      12,
      budget,
      spacing,
      bodies,
      { left: 12, right: 1268, top: 706, bottom: 1141 },
      true,
      goal,
      true,
    )
    const after = spacedPosition(goal, spacing)
    assert.ok(Math.hypot(after.x - before.x, after.y - before.y) <= budget + 0.00001)
    for (const body of bodies)
      assert.ok(Math.hypot(after.x - body.x, after.y - body.y) - 12 - body.radius >= 1.99)
    if (Math.hypot(after.x - goal.x, after.y - goal.y) <= 3) break
  }
  const location = spacedPosition(goal, spacing)
  assert.ok(Math.hypot(location.x - goal.x, location.y - goal.y) <= 3)
})

test('a fixed rectangular corner keeps the walking lane above it open', () => {
  const corner = {
    id: 'clock-corner',
    kind: 'trunk',
    x: 915,
    y: 698,
    radius: Math.hypot(130, 44) / 2,
    bounds: { left: 850, right: 980, top: 676, bottom: 720 },
  } satisfies CollisionActor
  assert.equal(canMove({ x: 820, y: 655 }, { x: 1000, y: 655 }, 18, [corner]), true)
  assert.equal(canMove({ x: 820, y: 698 }, { x: 1000, y: 698 }, 18, [corner]), false)
})

test('an actor swept into a fixed corner can leave without stepping deeper inside', () => {
  const corner = {
    id: 'clock-corner',
    kind: 'trunk',
    x: 915,
    y: 698,
    radius: Math.hypot(130, 44) / 2,
    bounds: { left: 850, right: 980, top: 676, bottom: 720 },
  } satisfies CollisionActor
  assert.equal(canMove({ x: 928, y: 670 }, { x: 928, y: 660 }, 18, [corner]), true)
  assert.equal(canMove({ x: 928, y: 670 }, { x: 928, y: 672 }, 18, [corner]), false)
  assert.equal(canMove({ x: 928, y: 670 }, { x: 900, y: 670 }, 18, [corner]), true)
  assert.equal(canMove({ x: 928, y: 698 }, { x: 928, y: 760 }, 18, [corner]), true)
})

test('a walker stops yielding to an animal that rests across its path past the patience', () => {
  const goose = {
    id: 'animal-goose-0',
    kind: 'animal',
    x: 40,
    y: 8,
    radius: 14,
    resting: true,
  } satisfies CollisionActor
  const spacing = actorSpacing()
  const walker = { x: 0, y: 0 }
  // The owner moves the walker when a step advances, as the journeys do.
  const step = () => {
    const next = { x: walker.x + 2, y: walker.y }
    const advanced = spacedStep('walker', walker, next, 18, 2, spacing, [goose])
    if (advanced) Object.assign(walker, next)
    return advanced
  }
  assert.equal(step(), false)
  assert.equal(spacing.yieldingTo, goose.id)
  assert.equal(step(), false)
  assert.equal(spacing.yieldingTo, goose.id)
  assert.ok(spacing.waited)
  spacing.waited.since -= yieldPatience + 1
  for (let index = 0; index < 60; index++) {
    step()
    assert.equal(spacing.yieldingTo, null)
    assert.notEqual(spacing.status, 'yielding')
  }
  const point = spacedPosition(walker, spacing)
  assert.ok(point.x >= 20, `walker reached x=${point.x.toFixed(1)}`)
  assert.ok(
    canMove(point, point, 18, [goose]) || Math.hypot(point.x - goose.x, point.y - goose.y) >= 32,
    'walker is not inside the goose',
  )
})
