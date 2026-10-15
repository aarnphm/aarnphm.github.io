import assert from 'node:assert/strict'
import test from 'node:test'
import { creekDistance, creekSection, pondDistance, trailLandscape } from './landscape'

const seeds = [0, 1, 43, 8192, 2701974569, 4294967295]
const widths = [320, 390, 700, 1280, 1920, 2880, 5760]

test('the ravine enters through the central foothills and stays connected to its runoff', () => {
  for (const width of widths)
    for (const seed of seeds) {
      const landscape = trailLandscape(width, 1497, 380, { x: width * 0.72, y: 1360 }, seed)
      const head = landscape.creek[0][0]
      const runoff = landscape.stream.at(-1)
      assert(runoff)
      assert(head.x > width * 0.25 && head.x < width * 0.75)
      assert(Math.hypot(head.x - runoff.x, head.y - runoff.y) < 3)
      const main = landscape.creek[0]
      const outlet = main.at(-1)
      assert(outlet && outlet.x >= width)
      for (let index = 1; index < main.length; index++) {
        const from = main[index - 1]
        const to = main[index]
        assert(to.x >= from.x - width * 0.01, `Creek doubles back at ${width}, seed ${seed}`)
        assert(Math.hypot(to.x - from.x, to.y - from.y) < 20)
        assert(to.y >= landscape.alpine.base - 2)
        assert(to.y < landscape.camp.y - 40)
      }
      for (let index = 1; index < landscape.stream.length; index++)
        assert(landscape.stream[index].y >= landscape.stream[index - 1].y)
      for (const line of landscape.creek)
        for (const point of line)
          assert(pondDistance(point, landscape.pond) > 1.2, 'The pond must remain separate')
    }
})

test('valley relief fades at flat ground and beside the creek instead of forming full-width ledges', () => {
  for (const width of widths)
    for (const seed of seeds) {
      const landscape = trailLandscape(width, 1497, 380, { x: width * 0.72, y: 1360 }, seed)
      let flat = 0
      for (const { crest } of landscape.rises)
        for (const column of crest) {
          assert(column.y <= column.brow && column.brow <= column.foot)
          if (column.form === null) {
            flat++
            assert(column.foot - column.brow <= 8)
          }
          if (creekDistance(column, landscape) < 8)
            assert(column.foot - column.brow <= 8, `Cliff beside water at ${width}, seed ${seed}`)
        }
      assert(flat > 0)
    }
})

test('the channel cross-section narrows in the distance and includes asymmetric shallow banks', () => {
  const landscape = trailLandscape(1439, 1497, 380, { x: 1036, y: 1360 }, 43)
  const far = creekSection(landscape.creek[0][0], landscape)
  const near = creekSection({ x: 900, y: landscape.camp.y - 60 }, landscape)
  assert(near.waterHalf > far.waterHalf)
  assert(near.nearBank > far.nearBank)
  assert(near.nearBank > near.farBank)
  for (const line of landscape.creek)
    for (const point of line) {
      const section = creekSection(point, landscape)
      assert(Object.values(section).every(value => Number.isFinite(value) && value > 0))
      assert(section.waterHalf + section.nearBank + section.shoulder < 40)
    }
})

test('each valley channel has both confined sections and wider water with varying banks', () => {
  for (const width of widths)
    for (const seed of seeds) {
      const landscape = trailLandscape(width, 1497, 380, { x: width * 0.72, y: 1360 }, seed)
      const sections = landscape.creek[0]
        .filter(point => point.y > landscape.meadowTop + 80)
        .map(point => creekSection(point, landscape))
      const water = sections.map(section => section.waterHalf)
      const bank = sections.map(section => section.nearBank)
      assert(Math.max(...water) / Math.min(...water) > 2, `Uniform water at ${width}, ${seed}`)
      assert(Math.max(...bank) / Math.min(...bank) > 1.5, `Uniform bank at ${width}, ${seed}`)
    }
})
