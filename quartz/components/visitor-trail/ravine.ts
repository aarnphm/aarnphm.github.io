import { bayerThreshold } from '../../util/bayer'
import {
  creekSection,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { ravineArt } from './ravine-art'
import { trailRelief } from './relief'

export { contourFaces, faceAt, onSteepGround, type Face } from './relief'

type Runs = Map<string, string[]>

const threshold = bayerThreshold

function run(runs: Runs, tone: string, x: number, y: number, width: number, height = 2) {
  if (width <= 0) return
  const list = runs.get(tone) ?? []
  list.push(`M${x} ${y}h${width}v${height}h${-width}Z`)
  runs.set(tone, list)
}

function paths(runs: Runs) {
  return [...runs.entries()]
    .map(([tone, list]) => `<path class="trail-ravine-${tone}" d="${list.join('')}"/>`)
    .join('')
}

// A shallow cross-section has a sloping near bank, a short far bank and broken grassy shoulders.
// Quantized cells merge into a few paths; the water and its crossings use the same section.
function creekBed(landscape: Landscape, random: () => number) {
  const cells = new Map<string, { tone: string; rank: number }>()
  const paint = (x: number, y: number, tone: string, rank: number) => {
    const key = `${Math.round(x / 2) * 2},${Math.round(y / 2) * 2}`
    if ((cells.get(key)?.rank ?? -1) < rank) cells.set(key, { tone, rank })
  }
  landscape.creek.forEach(line => {
    for (let index = 1; index < line.length; index++) {
      const a = line[index - 1]
      const b = line[index]
      const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
      let nx = -(b.y - a.y) / length
      let ny = (b.x - a.x) / length
      // The normal points to the far bank, up the screen.
      if (ny > 0 || (ny === 0 && nx > 0)) {
        nx = -nx
        ny = -ny
      }
      for (let k = 0; k < length; k++) {
        const x = a.x + ((b.x - a.x) * k) / length
        const y = a.y + ((b.y - a.y) * k) / length
        if (y > landscape.height + 8 || x > landscape.width + 8) continue
        const section = creekSection({ x, y }, landscape)
        const far = section.waterHalf + section.farBank
        const near = section.waterHalf + section.nearBank
        for (let side = -near - section.shoulder; side <= far + section.shoulder; side += 1) {
          const bank = side > 0 ? section.farBank : section.nearBank
          const inland = Math.abs(side) - section.waterHalf
          const texture = threshold(Math.round(x / 2) + Math.round(side), Math.round(y / 2))
          let tone: string
          let rank: number
          if (inland <= 0) {
            tone = Math.abs(side) < section.waterHalf * 0.4 && texture < 0.55 ? 'deep' : 'water'
            rank = 4
          } else if (inland <= bank) {
            // Gravel sits beside the water; the upper slope mixes into the grass.
            tone = inland < bank * 0.3 ? 'sand' : texture < inland / bank ? 'lit' : 'shade'
            rank = 3
          } else {
            const fade = 1 - (inland - bank) / section.shoulder
            if (texture > fade * 0.6) continue
            tone = 'shoulder'
            rank = 1
          }
          paint(x + nx * side, y + ny * side * 0.6, tone, rank)
        }
        const roll = random()
        if (roll < 0.035) paint(x, y, 'foam', 5)
        else if (roll < 0.045) paint(x, y, 'stone', 5)
      }
    }
  })
  const runs: Runs = new Map()
  for (const [key, { tone }] of cells) {
    const [x, y] = key.split(',').map(Number)
    run(runs, tone, x, y, 2)
  }
  return paths(runs)
}

// Sample by travelled distance rather than mesh segments: alpine segments are only two pixels
// long. A glint follows several of them at a visible speed while the bed keeps its texture.
function current(line: readonly Point[], landscape: Landscape, spacing: number) {
  const distances = [0]
  for (let index = 1; index < line.length; index++)
    distances.push(
      distances[index - 1] +
        Math.hypot(line[index].x - line[index - 1].x, line[index].y - line[index - 1].y),
    )
  const total = distances.at(-1) ?? 0
  const interval = Math.max(spacing, total / 18)
  const pointAt = (distance: number) => {
    let index = 1
    while (index < line.length - 1 && distances[index] < distance) index++
    const a = line[index - 1]
    const b = line[index]
    const share = (distance - distances[index - 1]) / (distances[index] - distances[index - 1] || 1)
    return { x: a.x + (b.x - a.x) * share, y: a.y + (b.y - a.y) * share }
  }
  const glints: string[] = []
  for (
    let distance = interval * 0.2;
    distance < total - 8 && glints.length < 18;
    distance += interval
  ) {
    const reach = Math.min(total - distance, spacing * 0.8, 42)
    const points = Array.from({ length: 5 }, (_, index) => pointAt(distance + (reach * index) / 4))
    const a = points[0]
    if (points[4].x > landscape.width - 8) continue
    const depth = trailGroundScale(a.y, landscape)
    const duration = reach / (12 + depth * 10)
    const motion = points
      .map((point, index) => {
        const from = points[Math.max(0, index - 1)]
        const to = points[Math.min(4, index + 1)]
        const angle = (Math.atan2(to.y - from.y, to.x - from.x) * 180) / Math.PI
        return `--current-x${index}:${(point.x - a.x).toFixed(2)}px;--current-y${index}:${(point.y - a.y).toFixed(2)}px;--current-angle${index}:${angle.toFixed(1)}deg`
      })
      .join(';')
    const span = 2 + depth * 3
    glints.push(
      `<g class="trail-river-current" transform="translate(${a.x.toFixed(1)} ${a.y.toFixed(1)})" style="${motion};--current-duration:${duration.toFixed(2)}s;--current-delay:-${((glints.length * 0.73) % duration).toFixed(2)}s"><g class="trail-river-glint"><path class="trail-ravine-foam" d="M0 -0.75h${span.toFixed(1)}v1.5H0Z"/></g></g>`,
    )
  }
  return glints.join('')
}

function creekCurrent(landscape: Landscape) {
  return landscape.creek.map(line => current(line, landscape, 62)).join('')
}

function creekRocks(landscape: Landscape, routes: readonly string[]) {
  const walks = routes.flatMap(routePoints)
  const rocks: string[] = []
  let travelled = 0
  const line = landscape.creek[0] ?? []
  for (let index = 1; index < line.length && rocks.length < 10; index++) {
    const a = line[index - 1]
    const b = line[index]
    const length = Math.hypot(b.x - a.x, b.y - a.y)
    travelled += length
    if (travelled < 130 || length < 1 || b.x > landscape.width - 12) continue
    const section = creekSection(b, landscape)
    if (
      section.waterHalf < 1.8 ||
      walks.some(point => Math.hypot(point.x - b.x, point.y - b.y) < 24)
    )
      continue
    travelled = 0
    const size = 0.65 + section.depth * 0.45
    const x = b.x - ((b.y - a.y) / length) * section.waterHalf * 0.7
    const y = b.y + ((b.x - a.x) / length) * section.waterHalf * 0.4
    rocks.push(
      `<g class="trail-river-rock" transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${size.toFixed(3)})"><path class="trail-ravine-stone" d="M-3 -3h5v1h2v2h-7Z"/><path class="trail-ravine-stone-lit" d="M-2 -4h3v2h-3Z"/><path class="trail-ravine-foam" d="M-4 1h3v1h-3ZM2 1h3v1H2Z"/></g>`,
    )
  }
  return rocks.join('')
}

// The stream down the face: it runs along the flank and foams where it tumbles, with a dithered
// cut below it in the rock. Its narrow runoff meets the meadow without a broad spray apron.
function mountainStream(landscape: Landscape) {
  const line = landscape.stream
  if (line.length < 2) return ''
  const runs: Runs = new Map()
  const seen = new Set<string>()
  const length = line.length - 1
  for (let index = 1; index <= length; index++) {
    const a = line[index - 1]
    const b = line[index]
    const steep = Math.abs(b.y - a.y) > Math.abs(b.x - a.x) * 1.8
    const samples = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y)))
    for (let k = 0; k < samples; k++) {
      const x = Math.round((a.x + ((b.x - a.x) * k) / samples) / 2) * 2
      const y = Math.round((a.y + ((b.y - a.y) * k) / samples) / 2) * 2
      if (seen.has(`${x},${y}`)) continue
      seen.add(`${x},${y}`)
      const t = index / length
      const width = t < 0.55 ? 2 : 4
      const left = x - width / 2
      if (steep) {
        for (const side of [left - 2, left + width])
          if (threshold(side / 2, y / 2) < 0.45) run(runs, 'gully', side, y, 2)
        if (threshold(x / 2 + 1, y / 2) < t * 0.12) run(runs, 'mist', left + width, y, 2)
      } else if (threshold(x / 2, y / 2 + 1) < 0.7) run(runs, 'gully', left, y + 2, width)
      run(runs, 'water', left, y, width)
      if (steep && threshold(x / 2, y / 2) < 0.4) run(runs, 'foam', x, y, 2)
    }
  }
  const foot = line[length]
  return `<g class="trail-ravine-stream" data-ground-y="${foot.y}">${paths(runs)}${current(line, landscape, 30)}</g>`
}

// Samples a route's `M x y C …` path into a polyline.
export function routePoints(d: string) {
  const numbers = (d.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number)
  if (numbers.length < 8) return []
  const points: Point[] = [{ x: numbers[0], y: numbers[1] }]
  for (let index = 2; index + 6 <= numbers.length; index += 6) {
    const from = points[points.length - 1]
    const [ax, ay, bx, by, x, y] = numbers.slice(index, index + 6)
    for (let k = 1; k <= 24; k++) {
      const t = k / 24
      const u = 1 - t
      points.push({
        x: u * u * u * from.x + 3 * u * u * t * ax + 3 * u * t * t * bx + t * t * t * x,
        y: u * u * u * from.y + 3 * u * u * t * ay + 3 * u * t * t * by + t * t * t * y,
      })
    }
  }
  return points
}

export function crossing(a: Point, b: Point, c: Point, d: Point) {
  const denominator = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x)
  if (Math.abs(denominator) < 1e-9) return null
  const t = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / denominator
  const u = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / denominator
  return t >= 0 && t <= 1 && u >= 0 && u <= 1
    ? { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }
    : null
}

// Where a trail meets the creek in the meadow, walkers cross on stepping stones laid along the
// trail across the cut, with foam breaking on their upstream side.
export function trailFords(landscape: Landscape, routes: readonly string[]) {
  const runs: Runs = new Map()
  const fords: Point[] = []
  for (const creek of landscape.creek)
    for (const route of routes) {
      const points = routePoints(route)
      for (let index = 1; index < points.length; index++) {
        const a = points[index - 1]
        const b = points[index]
        for (let k = 1; k < creek.length; k++) {
          const c = creek[k - 1]
          const d = creek[k]
          if (Math.max(a.y, b.y) < Math.min(c.y, d.y) || Math.min(a.y, b.y) > Math.max(c.y, d.y))
            continue
          const point = crossing(a, b, c, d)
          if (
            !point ||
            point.y < landscape.alpine.base + 4 ||
            pondDistance(point, landscape.pond) < 1.2 ||
            fords.some(other => Math.hypot(other.x - point.x, other.y - point.y) < 16)
          )
            continue
          fords.push(point)
          const along = Math.hypot(b.x - a.x, b.y - a.y) || 1
          const rx = (b.x - a.x) / along
          const ry = (b.y - a.y) / along
          const flow = Math.hypot(d.x - c.x, d.y - c.y) || 1
          const cx = (d.x - c.x) / flow
          const cy = (d.y - c.y) / flow
          // The bed's cross-sections are squashed by 0.6 up the screen, so a sideways run is a
          // thin band. Its half-width across the flow, measured along the trail, sets the run.
          const depth = trailGroundScale(point.y, landscape)
          const section = creekSection(point, landscape)
          const half =
            (section.waterHalf + Math.max(section.nearBank, section.farBank)) *
            (cy * cy + 0.6 * cx * cx)
          const span = half / Math.max(0.4, Math.abs(cx * ry - cy * rx))
          // Flat-topped boulders, lit on the left, with a wet line where they meet the water.
          const size = depth > 0.5 ? 10 : 8
          const extent = Math.abs(rx) * size + Math.abs(ry) * 6 + 2
          const count = Math.max(3, Math.min(7, Math.ceil((span * 2.2) / extent) + 1))
          for (let stone = 0; stone < count; stone++) {
            const offset = (stone - (count - 1) / 2) * extent
            // Stones step from side to side along a sideways run, like footfalls.
            const stagger = (stone % 2 ? 4 : -4) * Math.abs(cx)
            const x = Math.round((point.x + rx * offset + stagger - size / 2) / 2) * 2
            const y = Math.round((point.y + ry * offset - 5) / 2) * 2
            run(runs, 'stone-lit', x + 4, y, size - 8)
            run(runs, 'stone-lit', x + 2, y + 2, size / 2 - 2)
            run(runs, 'stone', x + size / 2, y + 2, size / 2 - 2)
            run(runs, 'stone-lit', x, y + 4, 4)
            run(runs, 'stone', x + 4, y + 4, size - 4)
            run(runs, 'stone', x, y + 6, size)
            run(runs, 'rim', x + 2, y + 8, size - 4)
            run(runs, 'foam', x - 2, y + 6, 2)
            run(runs, 'foam', x + size, y + 4, 2)
            run(runs, 'foam', x + size - 2, y + 8, 4)
          }
        }
      }
    }
  return fords.length
    ? `<g class="trail-ravine trail-ravine-fords" aria-hidden="true">${paths(runs)}</g>`
    : ''
}

const beaverTones = {
  B: 'trail-beaver-coat',
  E: 'trail-beaver-eye',
  S: 'trail-under-wood',
  K: 'trail-under-knot',
  W: 'trail-pond-ripple',
}

// The lodge and swimming route belong to the pond, independently of the river.
export function trailBeavers(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x68e31da4)
  const { pond } = landscape
  const cell = 2
  const toward = Math.PI * 0.16 + (random() - 0.5) * 0.4
  const away = toward + Math.PI
  const spread = (random() - 0.5) * 0.8
  const lodge = {
    x: Math.round(pond.x + Math.cos(away + spread) * pond.rx * 0.55),
    y: Math.round(pond.y + Math.sin(away + spread) * pond.ry * 0.5),
  }
  const lodgeRows = ['....SKSS....', '..SSKSSKSS..', '.SKSSSSKSSS.', 'SSSKSSSSKSSS', 'WWW.WWWW.WWW']
  const lodgeMarkup = `<g class="trail-beaver-lodge" transform="translate(${lodge.x - 12} ${lodge.y - 8})">${pixels(lodgeRows, beaverTones, cell)}</g>`
  const end = {
    x: pond.x + Math.cos(toward) * pond.rx * 0.78,
    y: pond.y + Math.sin(toward) * pond.ry * 0.7,
  }
  const start = { x: lodge.x + (end.x > lodge.x ? 14 : -14), y: lodge.y + 2 }
  const face = end.x >= start.x ? 1 : -1
  const swimmer = ['..SSSS..', '.BBBE...', 'BBBBBB..', 'W.WW.W..']
  const beaver = `<g class="trail-beaver" transform="translate(${Math.round(start.x)} ${Math.round(start.y)})" style="--beaver-dx: ${(end.x - start.x).toFixed(1)}px; --beaver-dy: ${(end.y - start.y).toFixed(1)}px; --beaver-face: ${face}"><g class="trail-beaver-swim"><g transform="translate(-8 -6)">${pixels(swimmer.slice(1), beaverTones, cell)}<g class="trail-beaver-stick" transform="translate(0 -${cell})">${pixels([swimmer[0]], beaverTones, cell)}</g></g></g></g>`
  return lodgeMarkup + beaver
}

function bankDetails(landscape: Landscape, random: () => number, routes: readonly string[]) {
  const walks = routes.flatMap(routePoints)
  const tones = {
    L: 'trail-ravine-stone-lit',
    S: 'trail-ravine-stone',
    G: 'trail-ravine-grass',
    D: 'trail-ravine-grass-dark',
  }
  const details: string[] = []
  for (const [lineIndex, line] of landscape.creek.entries()) {
    let spacing = 0
    for (let index = 1; index < line.length; index++) {
      const a = line[index - 1]
      const b = line[index]
      const length = Math.hypot(b.x - a.x, b.y - a.y)
      spacing += length
      if (spacing < 36 + random() * 30 || length < 0.1) continue
      spacing = 0
      const section = creekSection(b, landscape)
      const side = random() < 0.6 ? -1 : 1
      const bank = side < 0 ? section.nearBank : section.farBank
      const offset = side * (section.waterHalf + bank + section.shoulder * 0.5)
      const normal = b.x >= a.x ? -1 : 1
      const x = b.x - ((b.y - a.y) / length) * normal * offset
      const y = b.y + ((b.x - a.x) / length) * normal * offset * 0.6
      if (
        x < 6 ||
        x > landscape.width - 8 ||
        pondDistance({ x, y }, landscape.pond) < 1.4 ||
        walks.some(point => Math.hypot(point.x - x, point.y - y) < 20)
      )
        continue
      const size = (0.6 + section.depth * 0.7) * (0.8 + random() * 0.5)
      const rows =
        random() < 0.4 ? ravineArt.stone : random() < 0.5 ? ravineArt.fern : ravineArt.sedge
      details.push(
        `<g data-ravine-bank="${lineIndex}" transform="translate(${(x - (rows[0].length * size) / 2).toFixed(1)} ${(y - rows.length * size).toFixed(1)})">${pixels(rows, tones, size)}</g>`,
      )
    }
  }
  return details.join('')
}

export function trailCreek(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x1b873593)
  return `<g class="trail-ravine" aria-hidden="true">${creekBed(landscape, random)}${bankDetails(landscape, random, [])}${creekCurrent(landscape)}${creekRocks(landscape, [])}</g>`
}

export function trailRavine(landscape: Landscape, seed: number, routes: readonly string[] = []) {
  const random = trailRandom(seed ^ 0x1b873593)
  return (
    trailRelief(landscape, seed) +
    `<g class="trail-ravine" aria-hidden="true">${creekBed(landscape, random)}${bankDetails(landscape, random, routes)}${creekCurrent(landscape)}${creekRocks(landscape, routes)}</g>` +
    mountainStream(landscape)
  )
}
