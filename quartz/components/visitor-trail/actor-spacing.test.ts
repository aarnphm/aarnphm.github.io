import assert from 'node:assert/strict'
import { test } from 'node:test'
import { actorSpacing, spacedPosition, spacedStep, type CollisionActor } from './actor-spacing'

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
