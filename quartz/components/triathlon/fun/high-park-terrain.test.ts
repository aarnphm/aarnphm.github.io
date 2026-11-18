import assert from 'node:assert/strict'
import test from 'node:test'
import { heightAt, tracePoints } from './high-park-terrain'

test('lap traces decode to absolute points, run by run', () => {
  assert.deepEqual(tracePoints('M1 2l3-4 -5 6M10 10l1 1'), [
    [1, 2],
    [4, -2],
    [-1, 4],
    [10, 10],
    [11, 11],
  ])
})

test('terrain heights interpolate between cells and clamp at the edge', () => {
  // 2x2 cells, 10 m apart: 0 and 100 on the top row, 200 and 255 below; 0.1 m per byte.
  const terrain = {
    x0: 0,
    y0: 0,
    step: 10,
    cols: 2,
    rows: 2,
    min: 80,
    scale: 0.1,
    h: Buffer.from([0, 100, 200, 255]).toString('base64'),
  }
  const bytes = Uint8Array.from([0, 100, 200, 255])
  assert.equal(heightAt(terrain, bytes, 0, 0), 80)
  assert.ok(Math.abs(heightAt(terrain, bytes, 5, 0) - 85) < 1e-9)
  assert.ok(Math.abs(heightAt(terrain, bytes, -50, -50) - 80) < 1e-9)
  assert.ok(heightAt(terrain, bytes, 99, 99) > 105)
})
