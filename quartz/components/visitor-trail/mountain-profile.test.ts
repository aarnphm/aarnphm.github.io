import assert from 'node:assert/strict'
import test from 'node:test'
import type { Alpine } from './landscape'
import { buildMountainRanges, mountainRangesAt, mountainTerrain } from './mountain-profile'

const alpine = (width: number): Alpine => ({ left: 0, top: 500, width, height: 206, base: 706 })
const seeds = [0, 1, 43, 8192, 2701974569, 4294967295]

test('wider mountain views retain the geometry of existing ranges in every tier', () => {
  for (const seed of seeds) {
    const views = [1280, 1439, 1920, 2880, 3840, 5760].map(width =>
      buildMountainRanges(alpine(width), seed),
    )
    for (let index = 1; index < views.length; index++) {
      const previous = views[index - 1]
      const current = views[index]
      for (const tier of ['background', 'middle', 'foreground']) {
        const before = previous.filter(range => range.tier === tier)
        const after = current.filter(range => range.tier === tier)
        assert(after.length >= before.length)
        const geometry = (ranges: typeof before) =>
          ranges.map(({ index: _index, ...range }) => range)
        assert.deepEqual(geometry(after.slice(0, before.length)), geometry(before))
      }
    }
  }
})

test('wide ranges preserve their authored proportions and cover every tier without gaps', () => {
  for (const seed of seeds) {
    for (const width of [800, 1280, 1439, 1920, 2880, 3840, 5760]) {
      const bounds = alpine(width)
      const ranges = buildMountainRanges(bounds, seed)
      for (const range of ranges) {
        const aspect = mountainTerrain(range).aspect
        assert(range.width / range.height >= aspect / 1.06 - 1e-9)
        assert(range.width / range.height <= aspect / 0.82 + 1e-9)
        assert(range.top >= bounds.top)
        assert(range.base <= bounds.base)
      }
      for (let x = bounds.left; x <= width; x += 8) {
        const visible = mountainRangesAt(x, ranges)
        for (const tier of ['background', 'middle', 'foreground'])
          assert(
            visible.some(range => range.tier === tier),
            `${tier} has a gap at ${x}`,
          )
      }
      assert(buildMountainRanges(alpine(width * 2), seed).length > ranges.length)
    }
  }
})

test('narrow views retain three distinct terrain tiers within their elevation band', () => {
  for (const seed of seeds) {
    const ranges = buildMountainRanges(alpine(390), seed)
    assert.equal(ranges.length, 3)
    assert.equal(new Set(ranges.map(range => range.form)).size, 3)
    assert(ranges.every(range => range.top >= 500 && range.base <= 706))
  }
})
