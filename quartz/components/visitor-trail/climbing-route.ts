import type { Alpine, Point } from './landscape'
import {
  buildMountainRanges,
  heightAt,
  mountainProfileX,
  mountainRangesAt,
  mountainTerrain,
  type MountainRange,
} from './mountain-profile'
import { trailRandom } from './random'

export type ClimbingRoute = {
  id: string
  points: Point[]
  anchor: Point
  rangeIndex: number
  scale: number
}

function surfaceAt(x: number, range: MountainRange, boundary: 'crest' | 'floor') {
  return (
    range.base -
    ((350 - heightAt(mountainTerrain(range)[boundary], mountainProfileX(x, range))) *
      range.height) /
      350
  )
}

export function climbingPoint(route: ClimbingRoute, progress: number): Point {
  const distance = Math.max(0, Math.min(1, progress)) * (route.points.length - 1)
  const index = Math.min(route.points.length - 2, Math.floor(distance))
  const from = route.points[index]
  const to = route.points[index + 1]
  const blend = distance - index
  return { x: from.x + (to.x - from.x) * blend, y: from.y + (to.y - from.y) * blend }
}

export function buildClimbingRoutes(
  alpine: Alpine,
  seed: number,
  count: number,
  ranges: readonly MountainRange[] = buildMountainRanges(alpine, seed),
): ClimbingRoute[] {
  if (count <= 0) return []
  const candidates: { route: ClimbingRoute; score: number; rank: number; rangeKey: string }[] = []
  const ordinals = { background: 0, middle: 0 }
  const columns = new Map<number, MountainRange[]>()
  const surfaces = new Map<string, number>()
  const boundary = (x: number, range: MountainRange, edge: 'crest' | 'floor') => {
    const key = `${range.index}|${x}|${edge}`
    const cached = surfaces.get(key)
    if (cached !== undefined) return cached
    const height = surfaceAt(x, range, edge)
    surfaces.set(key, height)
    return height
  }
  for (const range of ranges) {
    if (range.tier === 'foreground') continue
    const ordinal = ordinals[range.tier]++
    const rank = ordinal * 2 + (range.tier === 'middle' ? 1 : 0)
    const rangeKey = `${range.tier}-${ordinal}`
    for (let site = 0; site < 40; site++) {
      const fraction = 0.06 + site * 0.022
      const x = range.left + (range.mirrored ? 1 - fraction : fraction) * range.width
      if (x < alpine.left + 16 || x > alpine.left + alpine.width - 16) continue
      const random = trailRandom(
        seed ^ Math.imul(rank + 1, 0x675ab943) ^ Math.imul(site + 1, 0x27d4eb2d),
      )
      const scale = range.tier === 'background' ? 0.3 : 0.36
      const drift = 3 + random() * 3
      const padding = 11 * scale + drift + 2
      let top = -Infinity
      let bottom = Infinity
      // Reserve the full swept sprite, including its helmet, against every overlapping face.
      for (let offset = -padding; offset <= padding + 1; offset += 1) {
        const column = Math.round(x + Math.min(padding, offset))
        let stack = columns.get(column)
        if (!stack) {
          stack = mountainRangesAt(column, ranges)
          columns.set(column, stack)
        }
        const index = stack.findIndex(candidate => candidate.index === range.index)
        if (index < 0) {
          bottom = -Infinity
          break
        }
        top = Math.max(top, boundary(column, range, 'crest') + 30 * scale + 4)
        bottom = Math.min(bottom, boundary(column, range, 'floor') - 3)
        for (const front of stack.slice(0, index))
          bottom = Math.min(bottom, boundary(column, front, 'crest') - 4)
      }
      if (!(bottom - top >= 20)) continue
      const lean = random() > 0.5 ? 1 : -1
      const points = [0, 1, 2, 3].map(step => ({
        x: x + lean * drift * [0.6, -0.3, 0.25, 0][step],
        y: bottom - ((bottom - top) * step) / 3,
      }))
      candidates.push({
        route: {
          id: `${rangeKey}-${site}`,
          points,
          anchor: { x, y: top - 30 * scale - 1 },
          rangeIndex: range.index,
          scale,
        },
        score: random(),
        rank,
        rangeKey,
      })
    }
  }
  candidates.sort((a, b) => a.rank - b.rank || b.score - a.score)
  const routes: ClimbingRoute[] = []
  const occupied = new Set<string>()
  for (const dispersed of [true, false]) {
    for (const { route, rangeKey } of candidates) {
      if (dispersed && occupied.has(rangeKey)) continue
      if (!routes.every(other => Math.abs(route.anchor.x - other.anchor.x) >= 32)) continue
      routes.push(route)
      occupied.add(rangeKey)
      if (routes.length >= count) return routes
    }
  }
  return routes
}
