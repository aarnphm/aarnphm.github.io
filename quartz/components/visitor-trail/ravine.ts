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

type Runs = Map<string, string[]>

const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const threshold = (x: number, y: number) => (bayer[(y & 3) * 4 + (x & 3)] + 0.5) / 16

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
  landscape.creek.forEach((line, lineIndex) => {
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
        if (lineIndex && pondDistance({ x, y }, landscape.pond) < 1) continue
        const section = creekSection({ x, y }, landscape, lineIndex > 0)
        const far = section.waterHalf + section.farBank
        const near = section.waterHalf + section.nearBank
        for (let side = -near - section.shoulder; side <= far + section.shoulder; side += 1) {
          const bank = side > 0 ? section.farBank : section.nearBank
          const inland = Math.abs(side) - section.waterHalf
          const texture = threshold(Math.round(x / 2) + Math.round(side), Math.round(y / 2))
          let tone: string
          let rank: number
          if (inland <= 0) {
            tone = 'water'
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

// The stream down the face: it runs along the flank and foams where it tumbles, with a dithered
// cut below it in the rock. Its narrow runoff meets the meadow without a broad spray apron.
function mountainStream(landscape: Landscape) {
  const line = landscape.stream
  if (line.length < 2) return ''
  const runs: Runs = new Map()
  const seen = new Set<string>()
  const centre: Point[] = []
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
      const width = 2
      if (steep) {
        for (const side of [x - 2, x + width])
          if (threshold(side / 2, y / 2) < 0.45) run(runs, 'gully', side, y, 2)
        if (threshold(x / 2 + 1, y / 2) < t * 0.12) run(runs, 'mist', x + width, y, 2)
      } else if (threshold(x / 2, y / 2 + 1) < 0.7) run(runs, 'gully', x, y + 2, width)
      run(runs, 'water', x, y, width)
      if (steep && threshold(x / 2, y / 2) < 0.4) run(runs, 'foam', x, y, 2)
      centre.push({ x: x + width / 2, y })
    }
  }
  const foot = line[length]
  const d = centre
    .filter((_, index) => index % 2 === 0)
    .map((point, index) => `${index ? 'L' : 'M'}${point.x} ${point.y}`)
    .join('')
  return `<g class="trail-ravine-stream" data-ground-y="${foot.y}">${paths(runs)}<path class="trail-ravine-flow" d="${d}"/></g>`
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
          const section = creekSection(point, landscape, creek !== landscape.creek[0])
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
  M: 'trail-beaver-mud',
  W: 'trail-pond-ripple',
}

// Beavers have dammed the outlet: a lip of sticks and mud across the channel where it leaves the
// pond, a lodge out in the water, and one beaver ferrying sticks from the lodge to the dam.
export function trailBeavers(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x68e31da4)
  const { pond } = landscape
  const line = landscape.creek[1]
  if (!line) return ''
  const outlet = line.find(point => pondDistance(point, pond) > 1.12)
  if (!outlet) return ''
  const depth = trailGroundScale(outlet.y, landscape)
  const cell = 2
  const dam = ['.SKS..SKS.KS.', 'SSKSSKSSSKSSK', 'KSSSKSSSKSSSS', 'MMMMMMMMMMMMM']
  const damWidth = dam[0].length * cell
  const damMarkup = `<g class="trail-beaver-dam" data-ground-y="${Math.round(outlet.y)}" transform="translate(${Math.round(outlet.x)} ${Math.round(outlet.y)}) scale(${(0.7 + depth * 0.4).toFixed(2)}) translate(${-damWidth / 2} -6)">${pixels(dam, beaverTones, cell)}</g>`
  // The lodge sits off the far shore from the outlet.
  const away = Math.atan2((outlet.y - pond.y) / pond.ry, (outlet.x - pond.x) / pond.rx) + Math.PI
  const spread = (random() - 0.5) * 0.8
  const lodge = {
    x: Math.round(pond.x + Math.cos(away + spread) * pond.rx * 0.55),
    y: Math.round(pond.y + Math.sin(away + spread) * pond.ry * 0.5),
  }
  const lodgeRows = ['....SKSS....', '..SSKSSKSS..', '.SKSSSSKSSS.', 'SSSKSSSSKSSS', 'WWW.WWWW.WWW']
  const lodgeMarkup = `<g class="trail-beaver-lodge" transform="translate(${lodge.x - 12} ${lodge.y - 8})">${pixels(lodgeRows, beaverTones, cell)}</g>`
  const toward = Math.atan2((outlet.y - pond.y) / pond.ry, (outlet.x - pond.x) / pond.rx)
  const end = {
    x: pond.x + Math.cos(toward) * pond.rx * 0.78,
    y: pond.y + Math.sin(toward) * pond.ry * 0.7,
  }
  const start = { x: lodge.x + (end.x > lodge.x ? 14 : -14), y: lodge.y + 2 }
  const face = end.x >= start.x ? 1 : -1
  const swimmer = ['..SSSS..', '.BBBE...', 'BBBBBB..', 'W.WW.W..']
  const beaver = `<g class="trail-beaver" transform="translate(${Math.round(start.x)} ${Math.round(start.y)})" style="--beaver-dx: ${(end.x - start.x).toFixed(1)}px; --beaver-dy: ${(end.y - start.y).toFixed(1)}px; --beaver-face: ${face}"><g class="trail-beaver-swim"><g transform="translate(-8 -6)">${pixels(swimmer.slice(1), beaverTones, cell)}<g class="trail-beaver-stick" transform="translate(0 -${cell})">${pixels([swimmer[0]], beaverTones, cell)}</g></g></g></g>`
  return lodgeMarkup + damMarkup + beaver
}

export type Face = { x: number; y: number; span: number; tall: number; turn: number }

// One face per crest column of each landform, from its crest down to its foot. `turn` > 0 where
// the crest drops to the right, so the face looks away from the light.
export function contourFaces(landscape: Landscape): Face[][] {
  return landscape.rises.map(({ crest }) => {
    const yAt = (k: number) => crest[Math.max(0, Math.min(crest.length - 1, k))].y
    return crest.map(({ x, y, foot }, k) => ({
      x,
      y,
      span: 4,
      tall: foot - y,
      // The slope over a few columns, so one jagged step does not flip a column's shade.
      turn: Math.max(-0.6, Math.min(0.6, (yAt(k + 6) - yAt(k - 6)) / 48)),
    }))
  })
}

// The nearest face whose slope contains the point, with how far below its crest the point lies.
export function faceAt(faces: readonly Face[][], point: Point) {
  let best: (Face & { below: number }) | null = null
  for (const contour of faces)
    for (const face of contour) {
      if (point.x < face.x || point.x >= face.x + face.span) continue
      const below = point.y - face.y
      if (below >= 0 && below < face.tall && (!best || below < best.below))
        best = { ...face, below }
    }
  return best
}

// Individual hills paint back to front. Flat columns have no wall; broken rims and sparse shade
// let the upper slopes meet the surrounding meadow without a continuous shelf.
function relief(landscape: Landscape, random: () => number) {
  const faces = contourFaces(landscape)
  return landscape.rises
    .map(({ crest }, index) => {
      const runs: Runs = new Map()
      let side = 'face'
      faces[index].forEach(({ x, y, tall, turn }, k) => {
        if (tall <= 0) return
        const { form, brow } = crest[k]
        const next = faces[index][k + 1]
        const nextBrow = crest[k + 1]?.brow
        // Three flat planes, as in isometric pixel art: the top is lightest, a face turned left
        // toward the light is mid tone, and a face turned right is darkest.
        // A plane keeps its tone until the slope clearly turns, so faces do not flicker.
        if (turn < -0.1) side = 'face-lit'
        else if (turn > 0.1) side = 'face-dark'
        else if (Math.abs(turn) < 0.03) side = 'face'
        run(runs, side, x, brow, 4, tall - (brow - y))
        if (brow > y) run(runs, 'top', x, y, 4, brow - y)
        const rim = threshold(x / 4, y / 2) < 0.25
        if (rim && form) run(runs, 'outline', x, y, 4)
        if (rim && form && next && next.y !== y) {
          const top = Math.min(y, next.y)
          run(runs, 'outline', x + 4 - (next.y > y ? 2 : 0), top, 2, Math.abs(next.y - y) + 2)
        }
        // A highlight runs along the brow, the edge where the top turns down into the face.
        if (brow > y) {
          run(runs, 'crest', x, brow, 4)
          if (nextBrow !== undefined && nextBrow !== brow && nextBrow > (next?.y ?? 0)) {
            const top = Math.min(brow, nextBrow)
            run(
              runs,
              'crest',
              x + 4 - (nextBrow > brow ? 2 : 0),
              top,
              2,
              Math.abs(nextBrow - brow) + 2,
            )
          }
        }
        const face = tall - (brow - y)
        for (let row = 4; row < face; row += 2) {
          const fall = row / face
          const at = brow + row
          if (form === 'butte' && fall < 0.6 && row % 6 === 4) {
            if (threshold(x / 4, row / 6) < 0.9) run(runs, 'strata', x, at, 4)
            continue
          }
          if (form === 'knob' && fall < 0.55) {
            if (threshold(x / 2, at / 2) < 0.18) run(runs, 'lip', x, at, 4)
            else if (turn < -0.06) run(runs, 'rock-lit', x, at, 4)
            else run(runs, 'rock', x, at, 4)
            continue
          }
        }
      })
      // A few boulders rest on the taller faces.
      for (let boulder = 0; boulder < 4; boulder++) {
        const column = crest[Math.floor(random() * crest.length)]
        const tall = column.foot - column.y
        if (tall < 24 || !column.form) continue
        const size = random() < 0.5 ? 8 : 6
        const x = column.x
        const y = Math.round((column.y + 8 + random() * (tall - 16)) / 2) * 2
        run(runs, 'rock-lit', x + 2, y, size - 4)
        run(runs, 'rock', x, y + 2, size, 2)
        run(runs, 'rock', x + size / 2, y, size / 2)
        run(runs, 'lip', x, y + 4, size)
      }
      return paths(runs)
    })
    .join('')
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
      const section = creekSection(b, landscape, lineIndex > 0)
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
      const size = 0.6 + section.depth * 0.7
      const rows =
        random() < 0.3 ? ravineArt.stone : random() < 0.5 ? ravineArt.fern : ravineArt.sedge
      details.push(
        `<g data-ravine-bank="${lineIndex}" transform="translate(${(x - (rows[0].length * size) / 2).toFixed(1)} ${(y - rows.length * size).toFixed(1)})">${pixels(rows, tones, size)}</g>`,
      )
    }
  }
  return details.join('')
}

export function trailCreek(landscape: Landscape, seed: number) {
  const random = trailRandom(seed ^ 0x1b873593)
  return `<g class="trail-ravine" aria-hidden="true">${creekBed(landscape, random)}${bankDetails(landscape, random, [])}</g>`
}

export function trailRavine(landscape: Landscape, seed: number, routes: readonly string[] = []) {
  const random = trailRandom(seed ^ 0x1b873593)
  return (
    `<g class="trail-relief" aria-hidden="true">${relief(landscape, random)}</g>` +
    `<g class="trail-ravine" aria-hidden="true">${creekBed(landscape, random)}${bankDetails(landscape, random, routes)}</g>` +
    mountainStream(landscape)
  )
}
