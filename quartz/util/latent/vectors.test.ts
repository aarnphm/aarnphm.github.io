import assert from 'node:assert/strict'
import { test } from 'node:test'
import { VectorEngine } from './vectors'

// 6 rows × 4 dims, unit-ish int8 rows
const rows = Int8Array.from([
  127, 0, 0, 0, 90, 90, 0, 0, 0, 127, 0, 0, 0, 0, 127, 0, 0, 0, 90, 90, -127, 0, 0, 0,
])

function engine() {
  return new VectorEngine(async () => rows.buffer.slice(0))
}

test('fuse ranks by Σ weight × channel and reports the base channel', async () => {
  const e = engine()
  await e.handle({ type: 'load', seq: 1, url: 'm', rows: 6, dims: 4 })
  const base = Float32Array.from([1, 0, 0, 0])
  const toward = Float32Array.from([0, 1, 0, 0])
  e.makeChannel('m', 'base', base, 0, 1, false, 127)
  e.makeChannel('m', 'toward', toward, 0, 1, true, 127)
  const plain = e.fuse('m', [{ key: 'base', weight: 1 }], 'base', [0], 3)
  assert.deepEqual([...plain.hits], [1, 2, 3])
  const steered = e.fuse(
    'm',
    [
      { key: 'base', weight: 1 },
      { key: 'toward', weight: 3 },
    ],
    'base',
    [0],
    3,
  )
  assert.deepEqual([...steered.hits], [2, 1, 3])
  // the reported score is the base cosine, not the fused one
  assert.equal(steered.scores[0], 0)
  const masked = e.fuse(
    'm',
    [{ key: 'base', weight: 1 }],
    'base',
    [],
    3,
    Uint8Array.from([0, 0, 1, 1, 0, 0]),
  )
  assert.deepEqual([...masked.hits], [2, 3])
})

test('handle answers as the worker did, errors included', async () => {
  const e = engine()
  const missing = await e.handle({ type: 'row', seq: 7, url: 'm', row: 0, scale: 127 })
  assert.equal(missing.response.type, 'error')
  await e.handle({ type: 'load', seq: 1, url: 'm', rows: 6, dims: 4 })
  const row = await e.handle({ type: 'row', seq: 8, url: 'm', row: 1, scale: 127 })
  assert.equal(row.response.type, 'row')
  if (row.response.type === 'row') assert.ok(Math.abs(row.response.vector[0] - Math.SQRT1_2) < 1e-6)
  const projected = await e.handle({
    type: 'project',
    seq: 9,
    url: 'm',
    dirs: Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    mean: new Float32Array(4),
    scale: 127,
  })
  assert.equal(projected.response.type, 'projected')
  if (projected.response.type === 'projected') assert.equal(projected.response.extent, 127)
})
