import type { Alpine } from './landscape'
import {
  mountainCrestAt,
  mountainGroundAt,
  type MountainRange,
  type MountainTrack,
} from './mountain-profile'

// Ranked so a later feature never paints under an earlier one: the cut bench wins over the
// contour it crosses, and a step riser wins over the tread it crosses.
const cutTones = ['contour', 'spoil', 'bank', 'wear', 'tread', 'stone', 'riser'] as const
type CutTone = (typeof cutTones)[number]
type Sample = { x: number; y: number; dx: number; dy: number; distance: number }

// Half the 4 px mountain pixel, the same grid the hikers are drawn on.
const cell = 2

function coordinate(value: number) {
  return Math.round(value * 100) / 100
}

function noise(column: number, row: number, salt: number) {
  let value = Math.imul(column, 0x27d4eb2d) ^ Math.imul(row, 0x165667b1) ^ salt
  value = Math.imul(value ^ (value >>> 15), 0x85ebca6b)
  value ^= value >>> 13
  return ((value >>> 0) % 1024) / 1024
}

function resample(points: readonly { x: number; y: number }[]) {
  const samples: Sample[] = []
  let distance = 0
  for (let index = 1; index < points.length; index++) {
    const from = points[index - 1]
    const to = points[index]
    const length = Math.hypot(to.x - from.x, to.y - from.y)
    if (!length) continue
    const steps = Math.max(1, Math.ceil(length / (cell / 2)))
    for (let step = 0; step < steps; step++) {
      samples.push({
        x: from.x + ((to.x - from.x) * step) / steps,
        y: from.y + ((to.y - from.y) * step) / steps,
        dx: (to.x - from.x) / length,
        dy: (to.y - from.y) / length,
        distance: distance + (length * step) / steps,
      })
    }
    distance += length
  }
  return samples
}

// A switchback turns where the route reverses across the slope.
function turns(samples: readonly Sample[]) {
  const result: Sample[] = []
  let heading = 0
  let last = -Infinity
  for (const sample of samples) {
    if (Math.abs(sample.dx) < 0.3) continue
    const sign = Math.sign(sample.dx)
    if (heading && sign !== heading && sample.distance - last > 24) {
      result.push(sample)
      last = sample.distance
    }
    heading = sign
  }
  return result
}

// Paints the mountain routes as benches cut into the slope, in the pixel grid of the hikers:
// a shaded cutbank above a worn tread, spoil below, steps on steep pitches, walled landings at
// the switchbacks, and the contours the route crosses.
export function carvedTrails(
  alpine: Alpine,
  ranges: readonly MountainRange[],
  tracks: readonly MountainTrack[],
): string {
  const columns = Math.ceil(alpine.width / cell)
  const rows = Math.floor(alpine.height / cell)
  // One byte per cell: 0 is bare rock, otherwise the index of the tone plus one.
  const cells = new Uint8Array(columns * rows)
  const crests = new Float64Array(columns).fill(Number.NaN)
  const crestAt = (column: number) => {
    if (Number.isNaN(crests[column]))
      crests[column] = mountainCrestAt(alpine.left + (column + 0.5) * cell, ranges) ?? alpine.base
    return crests[column]
  }
  const paint = (column: number, row: number, tone: CutTone) => {
    if (column < 0 || column >= columns || row < 0 || row >= rows) return
    if (alpine.top + row * cell < crestAt(column) + 2) return
    const rank = cutTones.indexOf(tone) + 1
    const key = row * columns + column
    if (cells[key] < rank) cells[key] = rank
  }
  // Narrow screens draw lower mountains on the same grid, so they carry no contours or spoil.
  const wide = alpine.width >= 700
  // Distance in cells from the route being painted, capped at the contour band.
  const band = 6
  const reach = new Uint8Array(columns * rows)
  const disc: [number, number, number][] = []
  for (let dy = -band; dy <= band; dy++)
    for (let dx = -band; dx <= band; dx++) {
      const distance = Math.round(Math.hypot(dx, dy))
      if (distance <= band) disc.push([dx, dy, distance])
    }
  const columnOf = (x: number) => Math.floor((x - alpine.left) / cell)
  const rowOf = (y: number) => Math.floor((y - alpine.top) / cell)

  for (const track of tracks) {
    const samples = resample(track.points)
    if (!samples.length) continue
    const salt = Math.imul(track.id + 1, 0x9e3779b9)
    const xs = track.points.map(point => point.x)
    const ys = track.points.map(point => point.y)
    const west = Math.min(...xs)
    const east = Math.max(...xs)
    const north = Math.min(...ys)
    const south = Math.max(...ys)

    // Contours show only where the route crosses them: each line fades out a few cells either
    // side of the bench, so the rest of the range keeps its rock. A line follows the skyline,
    // smoothed more the further down the slope it lies, as erosion rounds the lower ground.
    // The narrow mountains are too low to carry contours without crowding the routes.
    if (wide) {
      reach.fill(255)
      for (let index = 0; index < samples.length; index += 4) {
        const column = columnOf(samples[index].x)
        const row = rowOf(samples[index].y)
        for (const [dx, dy, distance] of disc) {
          const x = column + dx
          const y = row + dy
          if (x < 0 || x >= columns || y < 0 || y >= rows) continue
          const key = y * columns + x
          if (distance < reach[key]) reach[key] = distance
        }
      }
      const levels = Math.max(3, Math.round((south - north) / 18))
      const first = Math.max(0, columnOf(west) - band)
      const span = Math.min(columns, columnOf(east) + band + 1) - first
      const skyline = new Float64Array(span + 1)
      const counted = new Float64Array(span + 1)
      const floors = new Float64Array(span)
      for (let index = 0; index < span; index++) {
        const x = alpine.left + (first + index + 0.5) * cell
        const top = mountainGroundAt(x, 0, ranges)
        const bottom = mountainGroundAt(x, 1, ranges)
        const valid = top !== undefined && bottom !== undefined
        skyline[index + 1] = skyline[index] + (valid ? top : 0)
        counted[index + 1] = counted[index] + (valid ? 1 : 0)
        floors[index] = valid ? bottom : Number.NaN
      }
      for (let level = 1; level <= levels; level++) {
        const depth = level / (levels + 1)
        const radius = Math.round(1 + depth * 9)
        let before: number | undefined
        for (let index = 0; index < span; index++) {
          const column = first + index
          const from = Math.max(0, index - radius)
          const to = Math.min(span, index + radius + 1)
          const count = counted[to] - counted[from]
          const crest = (skyline[to] - skyline[from]) / count
          const ground = crest + (floors[index] - crest) * depth
          if (!count || Number.isNaN(ground)) {
            before = undefined
            continue
          }
          const row = rowOf(ground)
          const previous = before
          before = row
          // Close the stair between neighbouring columns so a steep contour stays one line.
          const start = previous === undefined ? row : previous + Math.sign(row - previous)
          for (let step = Math.min(start, row); step <= Math.max(start, row); step++) {
            if (step < 0 || step >= rows) continue
            const distance = reach[step * columns + column]
            if (distance > band || noise(column, step, salt) < distance / band) continue
            paint(column, step, 'contour')
          }
        }
      }
    }

    for (const sample of samples) {
      const near = Math.max(0, Math.min(1, (sample.y - alpine.top) / alpine.height))
      // High on the slope the bench recedes: the cutbank breaks up and the tread thins out.
      const recede = near < 0.4
      const column = columnOf(sample.x)
      const row = rowOf(sample.y)
      // Water bars and steps cross the tread at a walking rhythm that tightens with the grade.
      const grade = Math.abs(sample.dy) / Math.max(0.05, Math.abs(sample.dx))
      const spacing = grade > 0.9 ? 12 : grade > 0.35 ? 34 : 0
      const bar = spacing > 0 && sample.distance % spacing < cell / 2
      if (grade > 1.4) {
        // A fall-line pitch: stone steps beside a low wall.
        paint(column, row, row % 2 ? 'tread' : 'riser')
        if (noise(column - 1, row, salt) < 0.4) paint(column - 1, row, 'bank')
        continue
      }
      // A traverse: the bench is level across the slope, so the uphill face shows as a shaded
      // cutbank and the downhill edge sheds a little spoil.
      if (!recede || noise(column, row - 1, salt) < 0.5) paint(column, row - 1, 'bank')
      if (!recede || noise(column, row, salt) < 0.7) paint(column, row, 'tread')
      // A water bar is a log set across the climbing tread; divots are boot-worn pits.
      if (bar) paint(column, row, 'riser')
      else if (!recede && noise(column, row, salt ^ 0x5bd1e995) < 0.06) paint(column, row, 'wear')
      if (wide && !recede && noise(column, row + 1, salt) < 0.2) paint(column, row + 1, 'spoil')
    }

    // Landings: each switchback flattens into a platform held up by a dry-stone wall, with a
    // cairn on every other turn.
    for (const [index, turn] of turns(samples).entries()) {
      const near = Math.max(0, Math.min(1, (turn.y - alpine.top) / alpine.height))
      const reach = 2
      const column = columnOf(turn.x)
      const top = rowOf(turn.y)
      const outward = turn.dx > 0 ? -1 : 1
      for (let offset = -reach; offset <= reach; offset++) {
        const x = column + offset
        paint(x, top - 1, 'bank')
        paint(x, top, 'tread')
        // The wall runs under the outside of the turn, coursed like dry-stone work.
        if (offset * outward >= 0) paint(x, top + 1, x % 3 === 0 ? 'riser' : 'stone')
      }
      if (index % 2 === 0 && near >= 0.4) {
        const base = top - 2
        const side = column - outward * (reach + 1)
        for (const [lift, stones] of [
          [0, [-1, 0, 1]],
          [1, [-1, 0]],
          [2, [0]],
        ] as const)
          for (const stone of stones)
            paint(side + stone * outward, base - lift, stone * outward > 0 ? 'riser' : 'stone')
      }
    }
  }

  // Horizontal runs keep each tone in one SVG path, as in the mountain paint.
  const marks: string[][] = cutTones.map(() => [])
  for (let row = 0; row < rows; row++) {
    const y = coordinate(alpine.top + row * cell)
    for (let column = 0; column < columns; ) {
      const rank = cells[row * columns + column]
      let run = 1
      while (column + run < columns && cells[row * columns + column + run] === rank) run++
      if (rank)
        marks[rank - 1].push(
          `M${coordinate(alpine.left + column * cell)} ${y}h${run * cell}v${cell}h-${run * cell}Z`,
        )
      column += run
    }
  }
  return cutTones
    .map((tone, index) =>
      marks[index].length
        ? `<path class="trail-mountain-cut trail-mountain-cut-${tone}" d="${marks[index].join('')}"/>`
        : '',
    )
    .join('')
}
