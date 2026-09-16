import assert from 'node:assert/strict'
import test from 'node:test'
import { parseArenaCardPage } from './channel-data'
import { arenaRowAt, arenaRowOffsets, arenaVisibleRows } from './channel-window'

test('virtual rows preserve the full scroll extent with variable heights and a partial final row', () => {
  const offsets = arenaRowOffsets(
    10,
    3,
    80,
    new Map([
      [1, 120],
      [3, 95],
    ]),
  )
  assert.deepEqual(offsets, [0, 80, 200, 280, 375])
  assert.equal(arenaRowAt(offsets, 79), 0)
  assert.equal(arenaRowAt(offsets, 80), 1)
  assert.equal(arenaRowAt(offsets, 374), 3)
  assert.equal(arenaRowAt(offsets, 900), 3)
})

test('scrolling replaces the mounted window and excludes sections outside the buffer', () => {
  const offsets = arenaRowOffsets(3000, 3, 80, new Map())
  assert.deepEqual(arenaVisibleRows(offsets, 0, 160, 80), [0, 1, 2, 3])
  assert.deepEqual(arenaVisibleRows(offsets, 8000, 160, 80), [99, 100, 101, 102, 103])
  assert.deepEqual(arenaVisibleRows(offsets, -400, 160, 80), [])
  assert.deepEqual(arenaVisibleRows(offsets, 80200, 160, 80), [])
  assert.deepEqual(arenaVisibleRows([0], 0, 160, 80), [])
})

test('grid and list windows cover the same final block without growing with channel size', () => {
  for (const columns of [1, 2, 3, 4]) {
    const offsets = arenaRowOffsets(724, columns, 80, new Map())
    const total = offsets.at(-1) ?? 0
    const rows = arenaVisibleRows(offsets, total - 800, 800, 600)
    assert.ok(rows.length <= 19)
    assert.equal(rows.at(-1), Math.ceil(724 / columns) - 1)
  }
})

test('card page validation rejects malformed random-access ranges', () => {
  assert.deepEqual(parseArenaCardPage({ html: '<div></div>', offset: 24, total: 30 }), {
    html: '<div></div>',
    offset: 24,
    total: 30,
  })
  for (const offset of [-24, 1, 25, 48, NaN]) {
    assert.throws(() => parseArenaCardPage({ html: '', offset, total: 30 }))
  }
})
