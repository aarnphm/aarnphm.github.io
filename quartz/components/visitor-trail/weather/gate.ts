import type { Landscape } from '../landscape'
import { mountainCrestAt, type MountainRange } from '../mountain-profile'
import { trailRandom } from '../random'

// The 404's gate, seen from the trail: a stone ring on the flattest stretch of the right-hand
// ranges' skyline, its portal dark and its hands keeping the visitor's time. It is drawn in front
// of the ranges with its cells cut along the crest, so the ridge buries its foot the way the hill
// buries the gate on the 404; behind the ranges it floated, because the crest band is near-paper
// rock. Drawn in 2 px cells, redrawn with the sky each minute. The scene lays a link to the 404
// over the ring.
export type FarGateSite = { cx: number; cy: number; radius: number; cell: number }

export function farGateSite(landscape: Landscape, ranges: readonly MountainRange[]): FarGateSite {
  const { width, alpine } = landscape
  const cell = 2
  const radius = width < 700 ? 6 : 8
  // The flattest window three ring widths wide between 72% and 90% of the scene: the smallest
  // crest rise and fall, the higher stretch on a tie, so the ring stands on a shelf or a saddle
  // instead of a peak.
  const half = radius * cell * 3
  const step = 4
  let x = Math.round(width * 0.8)
  let crest = mountainCrestAt(x, ranges) ?? alpine.base
  let flattest = Infinity
  for (let centre = Math.round(width * 0.72); centre <= width * 0.9; centre += step) {
    let low = Infinity
    let high = -Infinity
    for (let probe = centre - half; probe <= centre + half; probe += step) {
      const y = mountainCrestAt(probe, ranges)
      if (y === undefined) {
        low = Infinity
        break
      }
      low = Math.min(low, y)
      high = Math.max(high, y)
    }
    if (low === Infinity) continue
    const spread = high - low
    const y = mountainCrestAt(centre, ranges) ?? alpine.base
    if (spread < flattest - 0.5 || (Math.abs(spread - flattest) <= 0.5 && y < crest))
      [x, crest, flattest] = [centre, y, spread]
  }
  const cx = Math.round(x / cell) * cell
  // The ranges' tops dissolve into dither, so the ring sinks four cells to stand in solid rock.
  const cy = Math.round((crest - (radius - 4) * cell) / cell) * cell
  return { cx, cy, radius, cell }
}

export function farGate(site: FarGateSite, ranges: readonly MountainRange[], now: number) {
  const { cx, cy, radius, cell } = site
  // Later tones paint over earlier ones: the hour hand sits above the minute hand where they cross.
  const tones: Record<
    'stone' | 'joint' | 'portal' | 'star' | 'tick' | 'minute' | 'hour',
    string[]
  > = { stone: [], joint: [], portal: [], star: [], tick: [], minute: [], hour: [] }
  // A cell whose centre lies under the skyline is rock, so the ring's foot stays buried.
  const buried = (i: number, j: number) =>
    cy + (j + 0.5) * cell > (mountainCrestAt(cx + (i + 0.5) * cell, ranges) ?? Infinity)
  const plot = (tone: keyof typeof tones, i: number, j: number) => {
    if (buried(i, j)) return
    tones[tone].push(`M${cx + i * cell} ${cy + j * cell}h${cell}v${cell}h${-cell}Z`)
  }
  const random = trailRandom(0x404)
  for (let j = -radius; j < radius; j++)
    for (let i = -radius; i < radius; i++) {
      const distance = Math.hypot(i + 0.5, j + 0.5)
      if (distance > radius) continue
      if (distance > radius - 2.2) {
        // Twelve dressed stones with a dark joint between each.
        const turn = (Math.atan2(j + 0.5, i + 0.5) / (2 * Math.PI) + 1) * 12
        plot(turn % 1 < 0.14 ? 'joint' : 'stone', i, j)
      } else plot(random() < 0.06 ? 'star' : 'portal', i, j)
    }
  // A face to read the hands against: one cell inside the ring at each hour, or only the quarters
  // on the small ring, where twelve marks would close up into a second ring.
  const marks = radius >= 8 ? 12 : 4
  for (let mark = 0; mark < marks; mark++) {
    const angle = (mark / marks) * 2 * Math.PI
    const reach = radius - 2.9
    plot(
      'tick',
      Math.round(Math.sin(angle) * reach - 0.5),
      Math.round(-Math.cos(angle) * reach - 0.5),
    )
  }
  const time = new Date(now)
  const minute = time.getMinutes() + time.getSeconds() / 60
  const hour = (time.getHours() % 12) + minute / 60
  const hand = (tone: 'minute' | 'hour', turns: number, length: number) => {
    const angle = turns * 2 * Math.PI
    const seen = new Set<string>()
    for (let t = 0; t <= length; t += 0.5) {
      const i = Math.round(Math.sin(angle) * t - 0.5)
      const j = Math.round(-Math.cos(angle) * t - 0.5)
      if (seen.has(`${i},${j}`)) continue
      seen.add(`${i},${j}`)
      plot(tone, i, j)
    }
  }
  hand('minute', minute / 60, radius - 3)
  hand('hour', hour / 12, (radius - 3) * 0.6)
  const paths = Object.entries(tones)
    .map(([tone, runs]) => `<path class="trail-far-gate-${tone}" d="${runs.join('')}"/>`)
    .join('')
  return `<g class="trail-far-gate" data-far-gate="${time.getHours()}:${String(time.getMinutes()).padStart(2, '0')}">${paths}</g>`
}
