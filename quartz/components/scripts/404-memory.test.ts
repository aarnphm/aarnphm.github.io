import assert from 'node:assert/strict'
import test from 'node:test'
import { type Crown, type Shed, stepShed } from './404-boughs'
import { type Voices, ageBubbles, say } from './404-crew'
import { type Grid, type Tones, interior, mix, paintBlob, putMix, putRgb } from './404-pixel'

function makeGrid(bw: number, bh: number): Grid {
  const sx = new Float32Array(bw)
  const sy = new Float32Array(bh)
  for (let i = 0; i < bw; i++) sx[i] = i
  for (let j = 0; j < bh; j++) sy[j] = j
  return {
    bw,
    bh,
    sx,
    sy,
    px: 1,
    toBuffer: { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 } as DOMMatrix,
    bottom: bh,
  }
}

const TONES: Tones = {
  ink: [16, 15, 15],
  light: [240, 235, 220],
  plate: [180, 175, 160],
  shade: [110, 105, 95],
}

test('mix, putRgb, and putMix compute exact channels without Array.prototype.map allocations', () => {
  let mapCalls = 0
  const origMap = Array.prototype.map
  Array.prototype.map = function (this: unknown[], ...args: Parameters<typeof origMap>) {
    mapCalls++
    return origMap.apply(this, args)
  } as typeof Array.prototype.map
  try {
    const c = mix([10, 20, 30], [110, 220, 130], 0.25)
    assert.deepEqual(c, [35, 70, 55])
    assert.equal(mapCalls, 0, 'mix() must not call Array.prototype.map')

    const buf = new Uint8ClampedArray(8)
    putRgb(buf, 0, 12, 34, 56, 200)
    assert.deepEqual(Array.from(buf.slice(0, 4)), [12, 34, 56, 200])

    putMix(buf, 4, 10, 20, 30, [110, 220, 130], 0.25)
    assert.deepEqual(Array.from(buf.slice(4, 8)), [35, 70, 55, 255])
    assert.equal(mapCalls, 0, 'putMix() must not call Array.prototype.map')
  } finally {
    Array.prototype.map = origMap
  }
})

test('paintBlob reuses scratch mask without stale bits bleeding from larger blobs into smaller blobs', () => {
  const g = makeGrid(40, 40)
  const d1 = new Uint8ClampedArray(40 * 40 * 4)
  const d2 = new Uint8ClampedArray(40 * 40 * 4)

  // First paint a large filled disc of radius 14 centered at (20, 20)
  const large = paintBlob(d1, g, (x, y) => Math.hypot(x - 20, y - 20) - 14, [4, 4, 36, 36], TONES)
  assert(large, 'expected large blob to paint')

  // Next paint a small blob over the same box where only the top-left corner is inside,
  // verifying that bits set by `large` in the shared scratch buffer do not bleed into `small`.
  const small = paintBlob(d2, g, (x, y) => Math.hypot(x - 10, y - 10) - 4, [4, 4, 20, 20], TONES)
  assert(small, 'expected small blob to paint')
  assert.equal(
    small.mask.buffer,
    large.mask.buffer,
    'expected paintBlob to reuse the underlying scratch ArrayBuffer',
  )

  const visited: [number, number][] = []
  interior(small, (i, j) => visited.push([i, j]))
  for (const [i, j] of visited) {
    assert(Math.hypot(i - 10, j - 10) < 4, `stale mask bit leaked into interior at (${i}, ${j})`)
  }
})

test('stepShed and ageBubbles compact in-place without flatMap or filter allocations', () => {
  let arrayAllocs = 0
  const origFilter = Array.prototype.filter
  const origFlatMap = Array.prototype.flatMap
  Array.prototype.filter = function (this: unknown[], ...args: Parameters<typeof origFilter>) {
    arrayAllocs++
    return origFilter.apply(this, args)
  } as typeof Array.prototype.filter
  Array.prototype.flatMap = function (this: unknown[], ...args: Parameters<typeof origFlatMap>) {
    arrayAllocs++
    return origFlatMap.apply(this, args)
  } as typeof Array.prototype.flatMap
  try {
    const crowns: Crown[] = [
      {
        segs: [],
        tips: [
          [10, 10, 0.5],
          [20, 15, 0.8],
        ],
      },
      { segs: [], tips: [[30, 12, 0.4]] },
    ]
    const shed: Shed = {
      leaves: [
        { x: 15, y: 20, vx: 1, vy: 10, spin: 0, tone: 0 },
        { x: 15, y: 250, vx: 1, vy: 10, spin: 0, tone: 1 },
      ],
      spawn: 0.9,
    }
    const leavesRef = shed.leaves
    stepShed(
      shed,
      crowns,
      { leaves: 1, blossom: 0, wind: 0.5, shake: null, box: [0, 0, 100, 100] },
      0.1,
    )
    assert.equal(shed.leaves, leavesRef, 'stepShed should compact shed.leaves in-place')
    assert(
      shed.leaves.every(l => l.y < 100),
      'out-of-bounds leaf should be removed',
    )

    const voices: Voices = {
      bubbles: [
        { who: 'kid', glyph: '!', age: 0.2 },
        { who: 'pointer', glyph: '?', age: 1.75 },
      ],
      heads: {},
    }
    const bubblesRef = voices.bubbles
    ageBubbles(voices, 0.1)
    assert.equal(voices.bubbles, bubblesRef, 'ageBubbles should compact bubbles in-place')
    assert.equal(voices.bubbles.length, 1)
    assert.equal(voices.bubbles[0].who, 'kid')

    say(voices, 'kid', '?')
    assert.equal(voices.bubbles, bubblesRef, 'say should update bubbles in-place')
    assert.equal(voices.bubbles.length, 1)
    assert.equal(voices.bubbles[0].glyph, '?')

    assert.equal(arrayAllocs, 0, 'stepShed, ageBubbles, and say must not call filter or flatMap')
  } finally {
    Array.prototype.filter = origFilter
    Array.prototype.flatMap = origFlatMap
  }
})
