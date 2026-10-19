import { buildMountainRanges, mountainCrestAt } from './mountain-profile'
import { trailRandom } from './random'

export type Point = { x: number; y: number }
export type Pond = Point & { rx: number; ry: number }
export type Grove = Point & { rx: number; ry: number }
export type Alpine = { left: number; top: number; width: number; height: number; base: number }
export type Landscape = {
  width: number
  height: number
  bottom: number
  alpine: Alpine
  treeline: number
  meadowTop: number
  camp: Point
  pond: Pond
  groves: Grove[]
  contours: Point[][]
  // Mountain runoff follows the valley; the pond is a separate basin.
  creek: Point[][]
  stream: Point[]
  // Rolling ridgelines across the meadow, back to front. Each follows a contour and swells into
  // landforms along its length; every crest column carries its own foot and the form it is part of.
  rises: Rise[]
  relief: { phase: number; amplitude: number }
}
export type RiseKind = 'ridge' | 'butte' | 'hill' | 'knob'
// `brow` is the front edge of a swell's top: seen from above, the ground between the crest and the
// brow is the lit top surface, and the face falls from the brow to the foot.
export type Rise = { crest: (Point & { brow: number; foot: number; form: RiseKind | null })[] }

export function groundOffset(x: number, landscape: Landscape, y = landscape.meadowTop) {
  const { width, relief } = landscape
  const depth = (y - landscape.meadowTop) / Math.max(1, landscape.camp.y - landscape.meadowTop)
  return (
    Math.sin((x / width) * Math.PI * 2 + relief.phase + depth * 2.4) * relief.amplitude +
    Math.sin((x / width) * Math.PI * 5 - relief.phase - depth * 3.1) * relief.amplitude * 0.45 +
    Math.sin(depth * Math.PI * 2.3 + relief.phase) * relief.amplitude * 0.25
  )
}

// The ranges meet the valley along a rolling foot, up to 16 px either side of the alpine base, so
// the backdrop haze and the snow blanket share one contact line instead of a level edge.
export function alpineFoot(x: number, landscape: Landscape) {
  const { alpine, relief } = landscape
  const swell = Math.sin(x / 83 + relief.phase) * 11 + Math.sin(x / 29 + relief.phase * 2.3) * 5
  return alpine.base + Math.round(swell / 2) * 2
}

// Distance from a point to the creek's centre line.
export function creekDistance(point: Point, landscape: Landscape) {
  let best = Infinity
  for (const line of landscape.creek)
    for (let index = 1; index < line.length; index++) {
      const a = line[index - 1]
      const b = line[index]
      if (
        point.x + best <= Math.min(a.x, b.x) ||
        point.x - best >= Math.max(a.x, b.x) ||
        point.y + best <= Math.min(a.y, b.y) ||
        point.y - best >= Math.max(a.y, b.y)
      )
        continue
      const dx = b.x - a.x
      const dy = b.y - a.y
      const t = Math.max(
        0,
        Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy || 1)),
      )
      best = Math.min(best, Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy))
    }
  return best
}

export function pondDistance(point: Point, pond: Pond) {
  return Math.hypot((point.x - pond.x) / pond.rx, (point.y - pond.y) / pond.ry)
}

// Grass and low plants grow on the middle and far meadow and give out toward the viewer, so the
// foreground stays open. True keeps a low plant at `y`; `roll` is a uniform draw in [0, 1).
export function grassGrows(y: number, landscape: Landscape, roll: number) {
  const near = (y - landscape.meadowTop) / Math.max(1, landscape.height - landscape.meadowTop)
  return roll >= (near - 0.45) / 0.2
}

export function trailGroundScale(y: number, landscape: Landscape) {
  const depth = (y - landscape.treeline) / (landscape.meadowTop + 160 - landscape.treeline)
  return 0.35 + 0.65 * Math.max(0, Math.min(1, depth))
}

// Rendering and stepping-stone crossings share the same banks. Perspective continues through
// the valley, with pools and gravel bars varying the section instead of a fixed-width ribbon.
export function creekSection(point: Point, landscape: Landscape) {
  const distance = Math.max(
    0,
    Math.min(
      1,
      (point.y - landscape.alpine.base) / Math.max(1, landscape.camp.y - landscape.alpine.base),
    ),
  )
  const depth = 0.24 + Math.sqrt(distance) * 0.76
  const phase = landscape.relief.phase
  const pool = (1 + Math.sin(distance * Math.PI * 7 + phase)) / 2
  const cut = (1 + Math.sin(distance * Math.PI * 5 - phase * 0.7)) / 2
  const scale = 0.8 + 0.2 * Math.min(1, landscape.width / 700)
  return {
    waterHalf: (0.8 + depth * 7.2) * (0.35 + pool * pool * 1.1) * scale,
    farBank: (0.8 + depth * 3.4) * (0.6 + cut * 0.9) * scale,
    nearBank: (1.4 + depth * 6.2) * (0.6 + cut * 0.9) * scale,
    shoulder: (1.5 + depth * 4) * (0.65 + pool * 0.6) * scale,
    depth,
  }
}

export function trailLandscape(
  width: number,
  height: number,
  bottom: number,
  camp: Point,
  seed: number,
): Landscape {
  const random = trailRandom(seed ^ 0x51ed270b)
  const alpineHeight = Math.min(width < 700 ? 112 : 206, width * 0.27)
  const alpineTop = bottom + 112
  const alpine: Alpine = {
    left: 0,
    top: alpineTop,
    width,
    height: alpineHeight,
    base: alpineTop + alpineHeight,
  }
  const meadowTop = alpine.base + 24
  const clearing = camp.y - meadowTop
  const landscape: Landscape = {
    width,
    height,
    bottom,
    alpine,
    treeline: alpineTop + alpineHeight * 0.84,
    meadowTop,
    camp,
    relief: { phase: random() * Math.PI * 2, amplitude: Math.min(42, clearing * 0.095) },
    pond: {
      x: width * (0.25 + random() * 0.1),
      y: meadowTop + clearing * 0.64,
      rx: Math.min(170, width * 0.11),
      ry: Math.min(58, clearing * 0.088),
    },
    groves: [],
    contours: [],
    creek: [],
    stream: [],
    rises: [],
  }
  landscape.pond.y += groundOffset(landscape.pond.x, landscape, landscape.pond.y)
  const groves = [
    { x: 0.12, y: 0.36 },
    { x: 0.44, y: 0.31 },
    { x: 0.68, y: 0.57 },
    { x: 0.87, y: 0.78 },
  ]
  landscape.groves = groves.map(point => {
    const x = width * (point.x + (random() - 0.5) * 0.04)
    const y = meadowTop + 64 + Math.max(1, clearing - 100) * point.y
    return { x, y, rx: Math.min(112, width * 0.1), ry: Math.min(70, clearing * 0.13) }
  })
  landscape.contours = [0.16, 0.39, 0.64, 0.91].map((level, index) => {
    const points: Point[] = []
    const step = width < 700 ? 8 : 16
    for (let x = 0; x < width + step; x += step) {
      const edge = Math.min(width, x)
      const base = meadowTop + clearing * level
      const y = base + groundOffset(edge + index * 36, landscape, base)
      points.push({ x: edge, y: Math.round(y / 4) * 4 })
    }
    return points
  })
  const flow = trailRandom(seed ^ 0x2545f491)
  // A mountain stream rises on the flank of a peak near the chosen spot and runs down the face
  // toward it, hugging the ridge line at first and steepening as it nears the foot. The
  // background tier fades into the sky, so the stream keeps to the nearer ridges.
  const ranges = buildMountainRanges(alpine, seed).filter(range => range.tier !== 'background')
  const crestAt = (x: number) => mountainCrestAt(x, ranges) ?? alpine.base
  // Averaged over a few columns so the stream bends with the ridge without copying every jag.
  const ridgeAt = (x: number) => {
    let sum = 0
    for (let dx = -12; dx <= 12; dx += 4) sum += crestAt(x + dx)
    return sum / 7
  }
  const aim = width * (0.36 + flow() * 0.08)
  const span = Math.max(60, Math.min(150, width * 0.1))
  const peakOn = (from: number, to: number) => {
    let best = { x: aim, y: alpine.base }
    for (let x = Math.max(12, from); x <= Math.min(width - 12, to); x += 4)
      if (ridgeAt(x) < best.y) best = { x, y: ridgeAt(x) }
    return best
  }
  // Prefer a peak upstream, to the left, so the water keeps flowing toward the right.
  const left = peakOn(aim - span, aim - span * 0.3)
  const peak = left.y < alpine.base - 8 ? left : peakOn(aim + span * 0.3, aim + span)
  const toward = Math.sign(aim - peak.x) || 1
  const source = peak.x + toward * 8
  const run = Math.abs(aim - source)
  const bend = Math.min(32, run * (0.24 + flow() * 0.08))
  const stream: Point[] = []
  const headwater = Math.min(alpine.base - 4, ridgeAt(source) + alpine.height * 0.18)
  for (
    let step = 0, steps = Math.max(8, Math.ceil((run + alpine.base - headwater) / 2));
    step <= steps;
    step++
  ) {
    const t = step / steps
    const x =
      source +
      (aim - source) * t +
      toward * Math.sin(t * Math.PI * 2) * Math.sin(t * Math.PI) * bend
    const ridge = Math.min(ridgeAt(x), alpine.base)
    // Keep runoff below the visible ridge and carry it downhill into the same valley channel.
    const y = Math.min(
      alpine.base,
      Math.max(stream.at(-1)?.y ?? headwater, ridge + 4, headwater + (alpine.base - headwater) * t),
    )
    stream.push({ x: Math.round(x), y: Math.round(y) })
  }
  landscape.stream = stream
  // Broad bends descend through the middle before turning toward the edge. Close bend handles
  // are bounded so the spline cannot double back at the narrow mountain-to-meadow join.
  const bends = [
    { x: aim, y: alpine.base },
    { x: width * 0.46, y: meadowTop + clearing * 0.08 },
    { x: width * (0.49 + flow() * 0.04), y: meadowTop + clearing * (0.29 + flow() * 0.03) },
    { x: width * (0.67 + flow() * 0.04), y: meadowTop + clearing * 0.44 },
    { x: width * (0.81 + flow() * 0.03), y: meadowTop + clearing * 0.56 },
    { x: width + 16, y: meadowTop + clearing * 0.68 },
  ]
  const course: Point[] = []
  const at = (index: number) => bends[Math.max(0, Math.min(bends.length - 1, index))]
  for (let index = 0; index < bends.length - 1; index++) {
    const [before, from, to, after] = [at(index - 1), at(index), at(index + 1), at(index + 2)]
    const steps = Math.max(2, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / 8))
    for (let step = index ? 1 : 0; step <= steps; step++) {
      const t = step / steps
      const u = 1 - t
      const blend = (key: 'x' | 'y') => {
        const mid = (from[key] + to[key]) / 2
        const first = Math.max(from[key], Math.min(mid, from[key] + (to[key] - before[key]) / 6))
        const second = Math.max(mid, Math.min(to[key], to[key] - (after[key] - from[key]) / 6))
        return (
          u ** 3 * from[key] + 3 * u * u * t * first + 3 * u * t * t * second + t ** 3 * to[key]
        )
      }
      course.push({ x: blend('x'), y: blend('y') })
    }
  }
  landscape.creek = [course]
  landscape.rises = trailRises(landscape, seed)
  return landscape
}

// Each contour is a continuous ridgeline. Along it the ground swells into a few landforms: long
// rolling ridges, rounded hills, flat-topped mesas with soft shoulders and rocky knobs. The swells
// add to the line's height and blend where they overlap. Ground stays low around the pond and
// the camp, and the creek cuts through whatever it crosses.
function trailRises(landscape: Landscape, seed: number) {
  const { width, meadowTop, camp, pond } = landscape
  const clearing = camp.y - meadowTop
  const random = trailRandom(seed ^ 0x3c6ef372)
  const shapes: Record<RiseKind, { half: [number, number]; tall: [number, number] }> = {
    ridge: { half: [180, 320], tall: [32, 60] },
    hill: { half: [90, 170], tall: [28, 54] },
    butte: { half: [70, 130], tall: [24, 44] },
    knob: { half: [36, 64], tall: [26, 46] },
  }
  const order: RiseKind[] = ['ridge', 'hill', 'butte', 'hill', 'knob', 'ridge', 'hill', 'knob']
  const within = ([low, high]: [number, number]) => low + random() * (high - low)
  const levels = [0.16, 0.39, 0.64, 0.91]
  let next = Math.floor(random() * order.length)
  return levels.map((level, index) => {
    const count = width < 700 ? 1 : 2 + Math.floor(random() * 2)
    const swells = Array.from({ length: count }, (_, k) => {
      const kind = order[next++ % order.length]
      return {
        kind,
        x: width * ((k + 0.2 + random() * 0.6) / count),
        half: within(shapes[kind].half),
        tall: within(shapes[kind].tall),
        cap: 0.35 + random() * 0.25,
        phase: random() * Math.PI * 2,
        skew: (random() - 0.5) * 0.5,
      }
    })
    const lift = (kind: RiseKind, v: number, cap: number, phase: number) => {
      const bell = Math.cos((Math.min(1, Math.abs(v)) * Math.PI) / 2) ** 2
      if (kind === 'hill') return bell
      if (kind === 'knob') return bell ** 2.2
      if (kind === 'ridge') return bell ** 0.6 * (0.72 + 0.28 * Math.sin(v * 4.6 + phase))
      // A bench: a cap that is nearly level, crowned a little toward its middle, and shoulders
      // that bell out to the ground, so no edge of it runs as a rule across the meadow.
      const out = Math.max(0, (Math.abs(v) - cap) / (1 - cap))
      if (out === 0) return 1 - 0.08 * (Math.abs(v) / cap) ** 2
      return 0.92 * Math.cos((Math.min(1, out) * Math.PI) / 2) ** 2
    }
    const crest: Rise['crest'] = []
    const ground = meadowTop + clearing * level
    for (let x = 0; x < width + 4; x += 4) {
      const base = ground + groundOffset(x + index * 36, landscape, ground)
      const depth = trailGroundScale(base, landscape)
      let height = 0
      let form: RiseKind | null = null
      let strongest = 0
      for (const swell of swells) {
        const u = (x - swell.x) / swell.half
        if (Math.abs(u) >= 1) continue
        const v = u + swell.skew * (1 - u * u)
        const value = lift(swell.kind, v, swell.cap, swell.phase) * swell.tall
        height += value
        if (value > strongest) {
          strongest = value
          form = swell.kind
        }
      }
      // Hills recede into the creek's open banks instead of carrying a cliff across the water.
      const calm = Math.min(
        1,
        Math.max(0, pondDistance({ x, y: base }, pond) - 1.4) / 1.2,
        Math.max(0, Math.hypot((x - camp.x) / 2, base - camp.y) - 70) / 90,
        Math.max(0, creekDistance({ x, y: base }, landscape) - 16) / 48,
      )
      height *= calm * (0.5 + depth * 0.4)
      const below = height * 0.12
      const y = Math.round((base - height) / 2) * 2
      const foot = Math.round((base + below) / 2) * 2
      crest.push({
        x,
        y,
        brow: Math.min(
          foot,
          y +
            Math.round((height * (form === 'knob' ? 0.52 : form === 'butte' ? 0.58 : 0.78)) / 2) *
              2,
        ),
        foot,
        form: strongest * calm > 6 ? form : null,
      })
    }
    // Isometric slopes run on two-step lines: over each 4 px column a crest climbs or falls by at
    // most 2 px, so hills, ridges and benches rise as 2:1 planes. Knobs keep sheer rock sides.
    const slope = (k: number, from: number) => {
      const column = crest[k]
      if (column.form === 'knob') return
      const limit = Math.min(column.foot, crest[from].y - 2)
      if (column.y < limit) {
        column.brow = Math.min(column.foot, column.brow + limit - column.y)
        column.y = limit
      }
    }
    for (let k = 1; k < crest.length; k++) slope(k, k - 1)
    for (let k = crest.length - 2; k >= 0; k--) slope(k, k + 1)
    return { crest }
  })
}

export function onDryLand(point: Point, landscape: Landscape, margin = 1.5): Point {
  const { width, height, meadowTop, pond } = landscape
  let { x, y } = point
  const distance = pondDistance(point, pond)
  if (distance < margin) {
    const angle = Math.atan2((y - pond.y) / pond.ry, (x - pond.x) / pond.rx)
    x = pond.x + Math.cos(angle) * pond.rx * margin
    y = pond.y + Math.sin(angle) * pond.ry * margin
  }
  return {
    x: Math.max(26, Math.min(width - 30, x)),
    y: Math.max(meadowTop + 64, Math.min(height - 28, y)),
  }
}

export function followTerrain(points: Point[], landscape: Landscape) {
  const { pond, bottom, camp } = landscape
  const adjusted = points.map((point, index) => {
    if (point.y <= bottom || index === 0 || index === points.length - 1) return point
    const weight = Math.min(1, (point.y - bottom) / Math.max(1, camp.y - bottom))
    const raised = { x: point.x, y: point.y + groundOffset(point.x, landscape, point.y) * weight }
    if (pondDistance(raised, pond) >= 1.8) return raised
    const angle = Math.atan2((raised.y - pond.y) / pond.ry, (raised.x - pond.x) / pond.rx)
    return {
      x: pond.x + Math.cos(angle) * pond.rx * 1.8,
      y: pond.y + Math.sin(angle) * pond.ry * 1.8,
    }
  })
  const routed = [adjusted[0]]
  for (let index = 1; index < adjusted.length; index++) {
    const from = adjusted[index - 1]
    const to = adjusted[index]
    const a = { x: (from.x - pond.x) / pond.rx, y: (from.y - pond.y) / pond.ry }
    const b = { x: (to.x - pond.x) / pond.rx, y: (to.y - pond.y) / pond.ry }
    const dx = b.x - a.x
    const dy = b.y - a.y
    const fraction = Math.max(0, Math.min(1, -(a.x * dx + a.y * dy) / (dx * dx + dy * dy || 1)))
    if (Math.hypot(a.x + fraction * dx, a.y + fraction * dy) < 1.7) {
      const start = Math.atan2(a.y, a.x)
      let sweep = Math.atan2(b.y, b.x) - start
      if (sweep > Math.PI) sweep -= Math.PI * 2
      if (sweep < -Math.PI) sweep += Math.PI * 2
      const steps = Math.max(2, Math.ceil(Math.abs(sweep) / (Math.PI / 6)))
      // An arc outside the bank leaves room for spline handles and walking marks.
      for (let step = 0; step <= steps; step++) {
        const angle = start + (sweep * step) / steps
        routed.push({
          x: pond.x + Math.cos(angle) * pond.rx * 1.8,
          y: pond.y + Math.sin(angle) * pond.ry * 1.8,
        })
      }
    }
    routed.push(to)
  }
  return routed
}
