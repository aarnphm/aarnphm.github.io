import assert from 'node:assert/strict'
import test from 'node:test'
import type { Alpine } from './landscape'
import { buildClimbingRoutes, climbingPoint } from './climbing-route'
import {
  buildMountainRanges,
  heightAt,
  mountainProfileX,
  mountainRangesAt,
  mountainTerrain,
} from './mountain-profile'

const seeds = [0, 1, 43, 8192, 2701974569, 4294967295]
const widths = [320, 390, 700, 1439, 2880, 5760]
const bounds = (width: number): Alpine => ({ left: 0, top: 500, width, height: 206, base: 706 })

test('climbing routes keep the entire sprite on a visible alpine face across seeds and widths', () => {
  for (const width of widths)
    for (const seed of seeds) {
      const alpine = bounds(width)
      const ranges = buildMountainRanges(alpine, seed)
      const routes = buildClimbingRoutes(alpine, seed, 6)
      assert(routes.length > 0, `No climbable face at ${width}, seed ${seed}`)
      for (const route of routes) {
        const range = ranges.find(candidate => candidate.index === route.rangeIndex)
        assert(range && range.tier !== 'foreground')
        const projected = (y: number) => 350 + ((y - range.base) * 350) / range.height
        const terrain = mountainTerrain(range)
        assert(
          projected(route.anchor.y) >=
            heightAt(terrain.crest, mountainProfileX(route.anchor.x, range)),
        )
        for (let sample = 0; sample <= 64; sample++) {
          const point = climbingPoint(route, sample / 64)
          assert(point.x >= 0 && point.x <= width)
          assert(route.anchor.y < point.y - 9 * route.scale)
          for (const x of [point.x - 11 * route.scale, point.x, point.x + 11 * route.scale]) {
            const profileX = mountainProfileX(x, range)
            assert(projected(point.y - 30 * route.scale) >= heightAt(terrain.crest, profileX))
            assert(projected(point.y) <= heightAt(terrain.floor, profileX))
            const cover = mountainRangesAt(x, ranges).slice(
              0,
              mountainRangesAt(x, ranges).findIndex(candidate => candidate.index === range.index),
            )
            for (const front of cover) {
              const crest =
                front.base -
                ((350 - heightAt(mountainTerrain(front).crest, mountainProfileX(x, front))) *
                  front.height) /
                  350
              assert(point.y < crest, `Climber hidden by ${front.form} at ${width}, seed ${seed}`)
            }
          }
        }
        assert(climbingPoint(route, 1).y < climbingPoint(route, 0).y)
      }
    }
})

test('climber counts are bounded and party routes stay separated', () => {
  for (const seed of seeds) {
    const alpine = bounds(1439)
    assert.deepEqual(buildClimbingRoutes(alpine, seed, 0), [])
    for (const count of [1, 2, 6]) {
      const routes = buildClimbingRoutes(alpine, seed, count)
      assert.equal(routes.length, count)
      assert.deepEqual(routes, buildClimbingRoutes(alpine, seed, count))
      for (let index = 0; index < routes.length; index++)
        for (const other of routes.slice(index + 1))
          assert(Math.abs(routes[index].anchor.x - other.anchor.x) >= 32)
    }
  }
})
