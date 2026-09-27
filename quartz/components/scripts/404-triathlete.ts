// Now and then a triathlete comes through, the one figure on the bank too busy racing to look at
// the clock. They swim in from the left edge along the bank, wade out at a rack where the bike
// waits, ride the strand tucked over the aerobars on a disc wheel (swinging up the bank behind the
// camp), drop the bike in the grass and run out past the cat with a number on their chest. It is
// an Inferno, so the race goes on up the mountain: they come out from behind the knoll onto the
// switchbacks on the right ridge, run up to a flag on the crest, and take the mountain bike waiting
// there back down. The course is laid out on whatever stretch of bank the viewport shows, clear of
// the camp, and every race brings a new kit.

import { type Grid, type RGB, col, dot, line, mix, row, stamp } from './404-pixel'

type Leg =
  | 'wait'
  | 'swim'
  | 'wade'
  | 't1'
  | 'bike'
  | 't2'
  | 'run'
  | 'climb'
  | 't3'
  | 'descend'
  | 'rest'
// Where the swim starts, the rack, the spot beside it where they strip, where the bike is dropped,
// where the run leaves, the swim lane, and whether the ridge trail is on screen for the mountain.
type Course = {
  start: number
  t1: number
  stand: number
  t2: number
  end: number
  lane: number
  peak: boolean
}

export type Triathlete = {
  leg: Leg
  x: number
  // Seconds into the current leg; strokes, cadence and stride run off it.
  t: number
  wait: number
  kit: number
  course: Course
}

export type TriInk = {
  skin: RGB
  hair: RGB
  shoes: RGB
  paper: RGB
  suits: RGB[]
  caps: RGB[]
  disc: RGB
  rack: RGB
  deep: RGB
  foam: RGB
  wake: RGB
  flag: RGB
  dust: RGB
}

// A point along the ridge trail, from its foot (0) to the crest (1), and which way that stretch runs.
export type Trail = (u: number) => { x: number; y: number; dir: number }

// Scene units per second. Real speeds run about 1 : 8 : 3; the swim is quickened so it does not
// take the whole visit.
const SWIM = 36
const WADE = 24
const BIKE = 120
const RUN = 60
// They stand up out of the lane this far short of the spot by the rack.
const WADE_IN = 36
// Seconds up the switchbacks, on the crest, and back down on the mountain bike.
const CLIMB = 40
const TOP = 2
const DESCENT = 11

// Sprites face right: cap (C), hair (H), skin (S), suit (T), race number (W), shoes (K), frame (F).
const STAND = ['.CC.', '.SS.', 'TTTT', 'STWS', '.TT.', '.TT.', '.SS.', '.SS.', '.KK.']
// Cap off overhead, the first thing out of the water.
const STRIP = ['SCCS', 'S..S', '.HS.', '.TT.', '.TW.', '.TT.', '.SS.', '.SS.', '.KK.']
const STRIDE = [
  '...CC.',
  '...SS.',
  '..TTT.',
  '.S.TWS',
  '...TT.',
  '..TTT.',
  '.S..S.',
  'K....S',
  '.....K',
]
const PASS = [
  '...CC.',
  '...SS.',
  '..TTT.',
  '..STW.',
  '...TS.',
  '..TTT.',
  '...SS.',
  '..KS..',
  '....K.',
]
const LYING = ['.KK....KK.', 'K..KFFK..K', '.KK....KK.']

// The swimmer's arm over one stroke, relative to the front of the cap: out at the hip, high over
// the back, reaching, in.
const ARM = [
  [
    [-4, -1],
    [-5, -2],
  ],
  [
    [-3, -2],
    [-2, -3],
    [-1, -3],
  ],
  [
    [1, -2],
    [2, -2],
  ],
  [
    [2, -1],
    [2, 0],
  ],
]

// The bike, relative to the rear wheel's left column and the ground row: wheel rims, the hip on the
// saddle, and the pedal round the bottom bracket.
const RIM = [
  [1, -3],
  [2, -3],
  [0, -2],
  [3, -2],
  [0, -1],
  [3, -1],
  [1, 0],
  [2, 0],
]
const HIP = [3, -6]
const PEDAL = [
  [5, -1],
  [4, 0],
  [3, -1],
  [4, -2],
]

export function makeTriathlete(reduce: boolean): Triathlete {
  return {
    // Without motion they stand at the rack; otherwise the first race starts a few seconds in.
    leg: reduce ? 'rest' : 'wait',
    x: 0,
    t: 0,
    wait: 3,
    // Blue suit, rose cap, magenta frame, for the still pose.
    kit: 2,
    course: { start: -80, t1: 420, stand: 380, t2: 1180, end: 1680, lane: 900, peak: false },
  }
}

// The rack a quarter of the way along the visible bank, the bike dropped three quarters of the way,
// and the swim and the run starting and ending off the canvas. The canvases overhang the viewport by
// 48 CSS px, so the marks keep 60 units in from their edges. The lane runs in open water, a third of
// the way out, clear of the bank's keyline. The transition takes the bank from where they wade out
// to the rack's far post; when that stretch would reach into the camp (`camp`, the extent of its
// sprites), the rack moves past the nearer end of the camp, or past the right end when the left
// would push the wade-out off screen.
export function layCourse(
  a: Triathlete,
  g: Grid,
  water: number,
  camp: readonly [number, number],
  peak: boolean,
) {
  const left = g.sx[0]
  const right = g.sx[g.bw - 1]
  const lo = left + 60
  const hi = right - 60
  const behind = 12 * g.px + WADE_IN
  const ahead = 6 * g.px
  let t1 = lo + (hi - lo) * 0.26
  if (t1 + ahead > camp[0] && t1 - behind < camp[1]) {
    const before = camp[0] - ahead
    const after = camp[1] + behind
    t1 = t1 - before < after - t1 && before - behind >= lo ? before : after
  }
  a.course = {
    start: left - 30,
    t1,
    stand: t1 - 9 * g.px,
    t2: lo + (hi - lo) * 0.74,
    end: right + 30,
    lane: water + Math.min(30, (g.bottom - water) * 0.4),
    peak,
  }
}

export function stepTriathlete(a: Triathlete, dt: number) {
  const c = a.course
  const go = (leg: Leg, x = a.x) => {
    a.leg = leg
    a.x = x
    a.t = 0
  }
  a.t += dt
  switch (a.leg) {
    case 'wait':
      if ((a.wait -= dt) > 0) break
      a.kit = Math.floor(Math.random() * 12)
      go('swim', c.start)
      break
    case 'swim':
      a.x += SWIM * dt
      if (a.x >= c.stand - WADE_IN) go('wade')
      break
    case 'wade':
      a.x += WADE * dt
      if (a.x >= c.stand) go('t1', c.stand)
      break
    case 't1':
      if (a.t > 1.2) go('bike', c.t1)
      break
    case 'bike':
      a.x += BIKE * dt
      if (a.x >= c.t2) go('t2', c.t2 + 10)
      break
    case 't2':
      if (a.t > 0.5) go('run')
      break
    case 'run':
      a.x += RUN * dt
      if (a.x < c.end) break
      if (c.peak) {
        go('climb')
        break
      }
      a.wait = 20 + Math.random() * 15
      go('wait')
      break
    case 'climb':
      if (a.t > CLIMB) go('t3')
      break
    case 't3':
      if (a.t > TOP) go('descend')
      break
    case 'descend':
      if (a.t < DESCENT) break
      a.wait = 10 + Math.random() * 10
      go('wait')
      break
  }
}

// Where their head is, for anyone who wants to watch them go by.
export function sighting(a: Triathlete, strand: (x: number) => number) {
  if (!['swim', 'wade', 't1', 'bike', 't2', 'run'].includes(a.leg)) return null
  return { x: a.x, y: a.leg === 'swim' ? a.course.lane : strand(a.x) - 28 }
}

// The top-middle pixel of their head on the bank, for a speech bubble, or null while they are away
// on the mountain or not racing.
export function headPixel(
  a: Triathlete,
  g: Grid,
  strand: (x: number) => number,
): [number, number] | null {
  const { stand, lane } = a.course
  switch (a.leg) {
    case 'swim':
      return [col(g, a.x) - 1, row(g, lane) - 2]
    case 'wade': {
      const k = Math.min(1, (a.x - (stand - WADE_IN)) / 30)
      return [col(g, a.x), row(g, lane + (strand(stand) - lane) * k) - 9]
    }
    case 't1':
    case 'rest':
      return [col(g, stand), row(g, strand(stand)) - 9]
    case 'bike':
      return [col(g, a.x) + 2, row(g, strand(a.x)) - 9]
    case 't2':
    case 'run':
      return [col(g, a.x), row(g, strand(a.x)) - 9]
    default:
      return null
  }
}

export function paintTriathlete(
  d: Uint8ClampedArray,
  g: Grid,
  a: Triathlete,
  ink: TriInk,
  strand: (x: number) => number,
  water: number,
) {
  const { t1, stand, t2, lane } = a.course
  const suit = ink.suits[a.kit % ink.suits.length]
  const cap = ink.caps[a.kit % ink.caps.length]
  const frame = ink.suits[(a.kit + 3) % ink.suits.length]
  const letter = (ch: string) =>
    ch === 'C'
      ? cap
      : ch === 'H'
        ? ink.hair
        : ch === 'S'
          ? ink.skin
          : ch === 'T'
            ? suit
            : ch === 'W'
              ? ink.paper
              : ch === 'K'
                ? ink.shoes
                : ch === 'F'
                  ? frame
                  : null
  // Stands the sprite with its last row on the pixel above `feet`, drawing only rows above `clip`.
  const figure = (rows: readonly string[], x: number, feet: number, clip = Infinity) => {
    const w = rows[0].length
    const i0 = col(g, x) - (w >> 1)
    const j0 = row(g, feet) - rows.length
    rows.forEach((text, r) => {
      if (j0 + r >= clip) return
      for (let c = 0; c < w; c++) {
        const color = letter(text[c])
        if (color) dot(d, g, i0 + c, j0 + r, color)
      }
    })
  }
  const shade = (c: RGB) => mix(c, ink.shoes, 0.35)

  // The bike, from the rear wheel's left column `i` and the ground row `jf`, with or without its rider.
  const bike = (i: number, jf: number, phase: number | null) => {
    const px = (p: readonly number[], c: RGB) => dot(d, g, i + p[0], jf + p[1], c)
    const ln = (p: readonly number[], q: readonly number[], c: RGB) =>
      line(d, g, i + p[0], jf + p[1], i + q[0], jf + q[1], c)
    const leg = (n: number, thigh: RGB, shin: RGB) => {
      const p = PEDAL[n & 3]
      const knee = [Math.round((HIP[0] + p[0]) / 2 + 1.5), Math.round((HIP[1] + p[1]) / 2 - 0.5)]
      ln(HIP, knee, thigh)
      ln(knee, p, shin)
      px(p, ink.shoes)
    }
    if (phase !== null) leg(phase + 2, shade(suit), shade(ink.skin))
    for (const [p, q] of [
      [
        [4, -1],
        [3, -4],
      ],
      [
        [3, -4],
        [7, -4],
      ],
      [
        [4, -1],
        [7, -4],
      ],
      [
        [3, -4],
        [2, -2],
      ],
      [
        [2, -2],
        [4, -1],
      ],
      [
        [7, -4],
        [7, -2],
      ],
    ])
      ln(p, q, frame)
    for (const p of [
      [2, -5],
      [3, -5],
      [7, -5],
      [8, -5],
      [9, -5],
    ])
      px(p, ink.shoes)
    // A disc behind, spokes in front.
    for (const [di, dj] of RIM) {
      px([di, dj], ink.shoes)
      px([di + 6, dj], ink.shoes)
    }
    for (const p of [
      [1, -2],
      [2, -2],
      [1, -1],
      [2, -1],
    ])
      px(p, ink.disc)
    if (phase === null) return
    leg(phase, suit, ink.skin)
    // Tucked over the aerobars.
    for (const p of [
      [3, -6],
      [4, -6],
      [5, -7],
      [6, -7],
    ])
      px(p, suit)
    for (const p of [
      [6, -8],
      [7, -8],
      [8, -8],
    ])
      px(p, cap)
    for (const p of [
      [7, -7],
      [8, -7],
      [7, -6],
      [8, -6],
    ])
      px(p, ink.skin)
  }

  // The rack, and the bike on it until the rider takes it.
  const rackFeet = row(g, strand(t1))
  const rc = col(g, t1)
  line(d, g, rc - 6, rackFeet - 1, rc - 6, rackFeet - 7, ink.rack)
  line(d, g, rc + 6, rackFeet - 1, rc + 6, rackFeet - 7, ink.rack)
  line(d, g, rc - 6, rackFeet - 7, rc + 6, rackFeet - 7, ink.rack)
  const racked =
    a.leg === 'wait' || a.leg === 'swim' || a.leg === 'wade' || a.leg === 't1' || a.leg === 'rest'
  if (racked) bike(rc - 5, rackFeet - 1, null)

  const beat = Math.floor(a.t * 5)
  switch (a.leg) {
    case 'swim': {
      const i = col(g, a.x)
      const j = row(g, lane)
      const p = beat & 3
      // The wake: two lines opening behind, flickering on the beat.
      for (let s = 2; s <= 14; s++) {
        if ((s + Math.floor(a.t * 10)) % 3 === 0) continue
        dot(d, g, i - 5 - s, j + 1 + (s >> 2), ink.wake)
        if (s <= 9) dot(d, g, i - 5 - s, j - 1 - (s >> 3), ink.wake)
      }
      // The body under the surface, kicking white at the tail.
      for (let s = 1; s <= 6; s++) dot(d, g, i - s, j, ink.deep)
      for (let s = 2; s <= 5; s++) dot(d, g, i - s, j + 1, ink.deep)
      dot(d, g, i - 7 - (p & 1), j - (p & 1), ink.foam)
      dot(d, g, i - 8 + (p & 1), j, ink.foam)
      // The cap, with the face turned out to breathe every other stroke.
      const breathe = p === 1 && (beat >> 2) % 2 === 0
      dot(d, g, i - 1, j - 1, cap)
      dot(d, g, i, j - 1, cap)
      dot(d, g, i - 1, j, cap)
      dot(d, g, i, j, breathe ? ink.skin : cap)
      for (const [di, dj] of ARM[p]) dot(d, g, i + di, j + dj, ink.skin)
      if (p === 0) dot(d, g, i - 4, j, ink.foam)
      if (p === 3) dot(d, g, i + 3, j - 1, ink.foam)
      break
    }
    case 'wade': {
      // Standing up out of the lane and onto the bank, legs hidden until they clear the water.
      const k = Math.min(1, (a.x - (stand - WADE_IN)) / 30)
      figure(beat & 1 ? PASS : STRIDE, a.x, lane + (strand(stand) - lane) * k, row(g, water))
      break
    }
    case 't1':
    case 'rest':
      figure(a.leg === 't1' && a.t < 0.6 ? STRIP : STAND, stand, strand(stand))
      break
    case 'bike':
      bike(col(g, a.x) - 5, row(g, strand(a.x)) - 1, beat & 3)
      break
    case 't2':
    case 'run': {
      stamp(d, g, LYING, col(g, t2) - 5, row(g, strand(t2)) - 3, false, letter)
      const flight = a.leg === 'run' && beat & 1
      const rows = a.leg === 't2' ? STAND : flight ? PASS : STRIDE
      figure(rows, a.x, strand(a.x) - (flight ? g.px : 0))
      break
    }
    case 'climb':
    case 't3':
    case 'descend':
      stamp(d, g, LYING, col(g, t2) - 5, row(g, strand(t2)) - 3, false, letter)
      break
  }
}

// The mountain, on the far ridge at a third of the size, where people are a pixel wide: a flag on
// the crest with the mountain bike racked beside it, the climb with the head leaning into the grade,
// arms up on top, and the ride down standing on the pedals, weight back, dust off the rear wheel.
// Returns the top of their head while they are up there, for a speech bubble.
export function paintAscent(
  d: Uint8ClampedArray,
  g: Grid,
  a: Triathlete,
  ink: TriInk,
  clock: number,
  trail: Trail,
): [number, number] | null {
  if (!a.course.peak) return null
  const suit = ink.suits[a.kit % ink.suits.length]
  const cap = ink.caps[a.kit % ink.caps.length]
  const frame = ink.suits[(a.kit + 3) % ink.suits.length]
  const top = trail(1)
  const si = col(g, top.x)
  const sj = row(g, top.y)
  const flap = Math.floor(clock * 3) & 1
  line(d, g, si + 6, sj - 1, si + 6, sj - 5, ink.rack)
  dot(d, g, si + 7, sj - 5, ink.flag)
  dot(d, g, si + 7, sj - 4, ink.flag)
  dot(d, g, si + 8, sj - 5 + flap, ink.flag)
  // Racked facing downhill, top tube and bars over two wheels.
  if (a.leg !== 'descend') {
    dot(d, g, si + 2, sj - 1, ink.shoes)
    dot(d, g, si + 3, sj - 1, frame)
    dot(d, g, si + 4, sj - 1, ink.shoes)
    dot(d, g, si + 3, sj - 2, frame)
    dot(d, g, si + 2, sj - 2, ink.shoes)
  }

  const beat = Math.floor(a.t * 5)
  if (a.leg === 'climb') {
    const at = trail(Math.min(1, a.t / CLIMB))
    const i = col(g, at.x)
    const j = row(g, at.y)
    dot(d, g, i + at.dir, j - 3, cap)
    dot(d, g, i, j - 2, suit)
    if (Math.floor(a.t * 3) & 1) {
      dot(d, g, i - 1, j - 1, ink.shoes)
      dot(d, g, i + 1, j - 1, ink.shoes)
    } else dot(d, g, i, j - 1, ink.shoes)
    return [i + at.dir, j - 3]
  } else if (a.leg === 't3') {
    dot(d, g, si, sj - 3, cap)
    dot(d, g, si, sj - 2, suit)
    dot(d, g, si, sj - 1, ink.shoes)
    dot(d, g, si - 1, sj - 4, ink.skin)
    dot(d, g, si + 1, sj - 4, ink.skin)
    return [si, sj - 3]
  } else if (a.leg === 'descend') {
    const at = trail(Math.max(0, 1 - a.t / DESCENT))
    const face = -at.dir
    const i = col(g, at.x)
    const j = row(g, at.y)
    // The odd root or rock jolts the rider a pixel off the saddle.
    const jolt = beat % 7 === 3 ? 1 : 0
    dot(d, g, i - 1, j - 1, ink.shoes)
    dot(d, g, i, j - 1, frame)
    dot(d, g, i + 1, j - 1, ink.shoes)
    dot(d, g, i + face, j - 2 - jolt, ink.shoes)
    dot(d, g, i, j - 2 - jolt, suit)
    dot(d, g, i - face, j - 3 - jolt, suit)
    dot(d, g, i, j - 3 - jolt, ink.skin)
    dot(d, g, i - face, j - 4 - jolt, cap)
    if (beat & 1) dot(d, g, i - 2 * face, j - 1, ink.dust)
    else dot(d, g, i - 3 * face, j - 2, ink.dust)
    return [i - face, j - 4 - jolt]
  }
  return null
}
