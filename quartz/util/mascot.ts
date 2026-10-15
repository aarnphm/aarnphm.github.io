// The garden's mascot in pen strokes: a hand-drawn circle with two short upright eyes shifted off to
// the left, traced from aarnphm's Freeform sketch. Each line is a filled ribbon around a wobbly
// centre line whose width swells and thins as pen pressure would, so it reads the same rasterised
// into the site icon (quartz/scripts/brand-art.ts) and live in the page as the pet. Seeds are fixed,
// so every caller draws the same wobble.

import { mulberry } from '../components/scripts/404-pixel'

export type Pt = readonly [number, number]

export const PAPER = '#fffcf0'
export const INK = '#1c1b1a'
export const PINK = '#fdb2a2'
export const SAGE = '#cdd597'

export const poly = (pts: readonly Pt[]) =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(3)}`).join('') + 'Z'

// A few slow sines with seeded phases, summing to roughly [-1, 1].
export function drift(seed: number, ks: readonly number[]) {
  const rand = mulberry(seed)
  const waves = ks.map(k => ({ k, phase: rand() * 2 * Math.PI, a: 1 / Math.sqrt(k) }))
  const norm = waves.reduce((s, w) => s + w.a, 0)
  return (t: number) => waves.reduce((s, w) => s + w.a * Math.sin(w.k * t + w.phase), 0) / norm
}

// A pen stroke along `pts`: the centre line offset both ways by half of `width(t)`, closed with a
// round end at each tip so the pen's landing and lift read as dots of ink.
export function stroke(pts: readonly Pt[], width: (t: number) => number, ink = INK) {
  const n = pts.length - 1
  const left: Pt[] = []
  const right: Pt[] = []
  for (let i = 0; i <= n; i++) {
    const [ax, ay] = pts[Math.max(0, i - 1)]
    const [bx, by] = pts[Math.min(n, i + 1)]
    const len = Math.hypot(bx - ax, by - ay) || 1
    const h = width(i / n) / 2
    const [x, y] = pts[i]
    left.push([x - ((by - ay) / len) * h, y + ((bx - ax) / len) * h])
    right.push([x + ((by - ay) / len) * h, y - ((bx - ax) / len) * h])
  }
  const tip = ([x, y]: Pt, r: number) =>
    `<circle cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="${r.toFixed(3)}"/>`
  return `<g style="fill:${ink}"><path d="${poly([...left, ...right.reverse()])}"/>${tip(pts[0], width(0) / 2)}${tip(pts[n], width(1) / 2)}</g>`
}

// Pressure along a stroke: thin where the pen lands and lifts, swelling in a few places between.
export const pressure = (seed: number, base: number, swell = 0.45) => {
  const wave = drift(seed, [3, 5, 8])
  return (t: number) =>
    base * (0.45 + 0.55 * Math.sin(Math.PI * t) ** 0.35) * (1 + swell * wave(t * 6))
}

// A circle drawn by hand, radius 1 around the origin: the radius drifts, the circle leans a little
// egg-ward, and the pen runs past where it started so the line overlaps itself near the top.
export function loop(seed: number, wobble = 0.06, overrun = 0.03, n = 220): Pt[] {
  const r = drift(seed, [2, 3, 5, 7])
  const start = -2.1
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const a = start + t * (2 * Math.PI + 0.3)
    const out = 1 + wobble * r(a) + overrun * Math.max(0, t - 0.9) * 10
    return [out * Math.cos(a) * 1.02, out * Math.sin(a) * (0.98 + 0.03 * Math.sin(a))] as const
  })
}

// Points along a polyline, about `step` apart, so a hand wobble has somewhere to live.
export function densify(pts: readonly Pt[], step: number): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i]
    const [bx, by] = pts[i + 1]
    const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / step))
    for (let s = 0; s < n; s++) out.push([ax + ((bx - ax) * s) / n, ay + ((by - ay) * s) / n])
  }
  out.push(pts[pts.length - 1])
  return out
}

// A polyline as the hand draws it: densified, then nudged sideways by a slow seeded drift.
export function hand(pts: readonly Pt[], step: number, amp: number, seed: number): Pt[] {
  const line = densify(pts, step)
  const wave = drift(seed, [3, 5, 9])
  const n = line.length - 1
  return line.map(([x, y], i) => {
    const [ax, ay] = line[Math.max(0, i - 1)]
    const [bx, by] = line[Math.min(n, i + 1)]
    const len = Math.hypot(bx - ax, by - ay) || 1
    const o = amp * wave((i / n) * 2 * Math.PI)
    return [x - ((by - ay) / len) * o, y + ((bx - ax) / len) * o] as const
  })
}

// A rounded rectangle's outline, clockwise from the top left, closed back on itself.
export function rounded(cx: number, cy: number, w: number, h: number, r: number): Pt[] {
  const corners: [number, number, number][] = [
    [cx + w / 2 - r, cy - h / 2 + r, -Math.PI / 2],
    [cx + w / 2 - r, cy + h / 2 - r, 0],
    [cx - w / 2 + r, cy + h / 2 - r, Math.PI / 2],
    [cx - w / 2 + r, cy - h / 2 + r, Math.PI],
  ]
  const pts: Pt[] = []
  for (const [x, y, a0] of corners)
    for (let i = 0; i <= 6; i++) {
      const a = a0 + (i / 6) * (Math.PI / 2)
      pts.push([x + r * Math.cos(a), y + r * Math.sin(a)])
    }
  pts.push(pts[0])
  return pts
}

// An upright eye: a short stroke that bulges in the middle like a pressed pen.
function eye(x: number, base: number, ink: string): string {
  const pts: Pt[] = Array.from({ length: 13 }, (_, i) => {
    const t = i / 12
    return [x + 0.025 * t + 0.012 * Math.sin(Math.PI * t), -0.36 + 0.32 * t] as const
  })
  return stroke(pts, t => base * (0.7 + 0.5 * Math.sin(Math.PI * t)), ink)
}

// Where the eyes rest, as in aarnphm's sketch: both shifted left of centre.
const EYES = [-0.6, 0.24] as const

type MascotOptions = { line: number; seed: number; paper?: string; ink?: string; points?: number }

// The mascot in its own units, radius about 1 around the origin, filled with paper inside the line
// so it reads on a dark background.
export function mascot({ line, seed, paper = PAPER, ink = INK, points = 220 }: MascotOptions) {
  const outline = loop(seed, 0.06, 0.03, points)
  const eyes = EYES.map(x => eye(x, line * 1.7, ink)).join('')
  return `<path d="${poly(outline)}" style="fill:${paper}"/>${stroke(outline, pressure(seed + 1, line), ink)}${eyes}`
}

// A quadratic curve from `a` to `b` pulled toward `c`, as `n + 1` points.
const curve = (a: Pt, c: Pt, b: Pt, n = 12): Pt[] =>
  Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const u = 1 - t
    return [
      u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
      u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
    ] as const
  })

// Thin at both tips, full in the middle: for short marks where seeded pressure would be noise.
const even = (base: number) => (t: number) => base * (0.7 + 0.4 * Math.sin(Math.PI * t))

// A small hand circle of radius `r` at (x, y).
const ring = (seed: number, x: number, y: number, r: number): Pt[] =>
  loop(seed, 0.1, 0.1, 36).map(([px, py]) => [x + r * px, y + r * py] as const)

const part = (name: string, body: string, cls = '') =>
  `<g data-part="${name}"${cls ? ` class="${cls}"` : ''}>${body}</g>`

// The mascot's bucket hat, in body radii: a crown in `fill`, a band, a brim. Pen widths scale with
// `u`, one pixel on the brand cards, where the mascot's own line is 3.2u.
export function bucketHat(u: number, fill: string, ink = INK, band = PINK) {
  const crown = hand(
    [
      [-0.62, -0.78],
      [-0.52, -1.32],
      [-0.12, -1.52],
      [0.3, -1.48],
      [0.6, -1.24],
      [0.66, -0.78],
      [-0.62, -0.78],
    ],
    0.04,
    0.015,
    67,
  )
  const ribbon = hand(
    [
      [-0.58, -0.9],
      [0.62, -0.9],
    ],
    0.04,
    0.01,
    69,
  )
  const brim = hand(
    [
      [-1.08, -0.7],
      [-0.7, -0.8],
      [0, -0.82],
      [0.7, -0.8],
      [1.08, -0.68],
    ],
    0.04,
    0.015,
    70,
  )
  return `<path d="${poly(crown)}" style="fill:${fill}"/>${stroke(crown, pressure(68, 2.6 * u, 0.35), ink)}${stroke(ribbon, () => 4 * u, band)}${stroke(brim, pressure(71, 3.4 * u, 0.4), ink)}`
}

// A closed band around a centre line, `half(t)` wide to each side, for a ribbon, a loaf or a quill.
export function ribbon(centre: readonly Pt[], half: (t: number) => number): Pt[] {
  const n = centre.length - 1
  const side = (sign: number) =>
    centre.map(([x, y], i) => {
      const [ax, ay] = centre[Math.max(0, i - 1)]
      const [bx, by] = centre[Math.min(n, i + 1)]
      const len = Math.hypot(bx - ax, by - ay) || 1
      const h = sign * half(i / n)
      return [x - ((by - ay) / len) * h, y + ((bx - ax) / len) * h] as const
    })
  return [...side(1), ...side(-1).reverse(), side(1)[0]]
}

// A closed shape in one colour inside a pen line.
const filled = (pts: readonly Pt[], fill: string, width: (t: number) => number, ink: string) =>
  `<path d="${poly(pts)}" style="fill:${fill}"/>${stroke(pts, width, ink)}`

export const CRUST = '#fcc192'

// Section outfits, in body radii like the mascot, drawn with a pen `w` wide (0.042 on the site
// icons, where the body's own line is 0.055). The brand icons and the pet share them.

// Triathlon: a sweatband round the brow, tied at the back with its tails flying.
export function sweatband(w: number, ink = INK, band = PINK) {
  const brow = Array.from({ length: 25 }, (_, i) => {
    const x = -0.8 + (1.7 * i) / 24
    return [x, -0.6 + 0.13 * (1 - x * x) + 0.05 * x] as const
  })
  const tails = (
    [
      [
        [0.9, -0.57],
        [1.18, -0.86],
        [1.44, -0.8],
      ],
      [
        [0.9, -0.5],
        [1.2, -0.44],
        [1.4, -0.24],
      ],
    ] as const
  )
    .map(([a, c, b], i) =>
      filled(
        hand(
          ribbon(curve(a, c, b, 24), t => 0.075 * (1 - 0.55 * t)),
          0.04,
          0.02,
          203 + i,
        ),
        band,
        pressure(205 + i, w, 0.3),
        ink,
      ),
    )
    .join('')
  const knot = filled(
    hand(rounded(0.9, -0.54, 0.16, 0.2, 0.07), 0.03, 0.01, 207),
    band,
    pressure(208, w, 0.3),
    ink,
  )
  const strip = hand(
    ribbon(brow, () => 0.1),
    0.04,
    0.012,
    201,
  )
  return `${tails}${filled(strip, band, pressure(202, w, 0.35), ink)}${knot}`
}

// French: a beret slumped to the back of the head, stalk and all.
export function beret(w: number, fill = INK, ink = INK) {
  const brim: Pt[] = Array.from({ length: 49 }, (_, i) => {
    const a = (i / 48) * 2 * Math.PI
    const ry = Math.sin(a) < 0 ? 0.3 : 0.11
    const c = Math.cos(0.24)
    const s = Math.sin(0.24)
    const x = 0.74 * Math.cos(a)
    const y = ry * Math.sin(a)
    return [0.12 + x * c - y * s, -0.86 + x * s + y * c] as const
  })
  const stalk = stroke(
    hand(
      [
        [0.18, -1.14],
        [0.13, -1.29],
      ],
      0.03,
      0.008,
      221,
    ),
    t => w * (1.6 - 0.6 * t),
    ink,
  )
  return `${filled(brim, fill, pressure(222, w, 0.3), ink)}${stalk}`
}

// A baguette from `a` to `b`, pointed at both ends, with slashes skewed along its crust.
export function baguette(a: Pt, b: Pt, seed: number, w: number, ink = INK, crust = CRUST) {
  const outline = hand(
    ribbon(densify([a, b], 0.04), t => 0.15 * Math.sin(Math.PI * t) ** 0.45),
    0.04,
    0.012,
    seed,
  )
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const len = Math.hypot(dx, dy)
  const [ux, uy] = [dx / len, dy / len]
  const cuts = [0.24, 0.41, 0.58, 0.75]
    .map(t => {
      const [cx, cy] = [a[0] + dx * t, a[1] + dy * t]
      const [sx, sy] = [0.09 * ux - 0.08 * uy, 0.09 * uy + 0.08 * ux]
      return stroke(
        densify(
          [
            [cx - sx, cy - sy],
            [cx + sx, cy + sy],
          ],
          0.03,
        ),
        t => w * 0.7 * (0.6 + 0.6 * Math.sin(Math.PI * t)),
        ink,
      )
    })
    .join('')
  return `${filled(outline, crust, pressure(seed + 1, w, 0.3), ink)}${cuts}`
}

// Legal pages: the lawyer. A barrister's wig in `paper`, its crown in rows of curls with two rolls
// over each ear, a tie under the chin and a briefcase at its back hand.
export function lawyer(w: number, paper = PAPER, ink = INK, tie = PINK, leather = CRUST) {
  const cap = hand(
    [
      ...curve([-0.92, -0.5], [-0.62, -1.38], [0.12, -1.3], 18),
      ...curve([0.12, -1.3], [0.78, -1.26], [0.98, -0.5], 18).slice(1),
      ...curve([0.98, -0.5], [0.1, -0.86], [-0.92, -0.5], 18).slice(1),
    ],
    0.03,
    0.012,
    271,
  )
  // Curls: a prolate cycloid along each row, looping at the bottom.
  const curls = (
    [
      [-0.46, 0.54, -1.16],
      [-0.68, 0.76, -1.0],
      [-0.8, 0.86, -0.84],
    ] as const
  )
    .map(([x0, x1, y0], i) => {
      const turns = (x1 - x0) / 0.15
      const pts = Array.from({ length: 80 }, (_, k) => {
        const t = (k / 79) * turns * 2 * Math.PI
        return [
          x0 + (0.15 * t) / (2 * Math.PI) - 0.05 * Math.sin(t),
          y0 + 0.06 * (1 - Math.cos(t)),
        ] as const
      })
      return stroke(hand(pts, 0.02, 0.006, 272 + i), () => w * 0.62, ink)
    })
    .join('')
  const rolls = (
    [
      [-1.02, -0.42],
      [-1.06, -0.18],
      [1.06, -0.4],
      [1.1, -0.16],
    ] as const
  )
    .map(([x, y], i) => {
      const roll = hand(rounded(x, y, 0.34, 0.22, 0.1), 0.03, 0.01, 276 + i)
      const turn = Array.from({ length: 25 }, (_, k) => {
        const a = (k / 24) * 1.7 * Math.PI
        const r = 0.075 * (1 - k / 32)
        return [x + r * Math.cos(a) * 1.3, y + r * Math.sin(a)] as const
      })
      return `${filled(roll, paper, pressure(280 + i, w, 0.3), ink)}${stroke(turn, () => w * 0.6, ink)}`
    })
    .join('')
  const wig = `${filled(cap, paper, pressure(270, w, 0.3), ink)}${curls}${rolls}`
  const mx = (EYES[0] + EYES[1]) / 2
  const knot = hand(
    [
      [mx - 0.09, 0.38],
      [mx + 0.09, 0.38],
      [mx + 0.06, 0.52],
      [mx - 0.06, 0.52],
      [mx - 0.09, 0.38],
    ],
    0.02,
    0.006,
    285,
  )
  const blade = hand(
    [
      [mx - 0.05, 0.52],
      [mx + 0.05, 0.52],
      [mx + 0.13, 0.88],
      [mx, 1.0],
      [mx - 0.13, 0.88],
      [mx - 0.05, 0.52],
    ],
    0.02,
    0.008,
    286,
  )
  const neck = `${filled(densify(blade, 0.04), tie, pressure(287, w, 0.25), ink)}${filled(densify(knot, 0.03), tie, pressure(288, w, 0.25), ink)}`
  const [bx, by, bw, bh] = [0.98, 0.98, 0.7, 0.5]
  const caseBody = hand(rounded(bx, by, bw, bh, 0.07), 0.03, 0.01, 289)
  const handle = hand(
    curve([bx - 0.13, by - bh / 2], [bx, by - bh / 2 - 0.26], [bx + 0.13, by - bh / 2], 12),
    0.02,
    0.006,
    290,
  )
  const seam = hand(
    [
      [bx - bw / 2 + 0.03, by - 0.06],
      [bx + bw / 2 - 0.03, by - 0.06],
    ],
    0.02,
    0.006,
    291,
  )
  const clasp = hand(rounded(bx, by - 0.06, 0.1, 0.08, 0.02), 0.01, 0.004, 292)
  const briefcase = `${stroke(handle, () => w * 1.3, ink)}${filled(caseBody, leather, pressure(293, w, 0.3), ink)}${stroke(seam, () => w * 0.6, ink)}${filled(clasp, ink, () => w * 0.5, ink)}`
  return { wig, neck, briefcase }
}

// Quotes and images: the artist. A wooden palette in front with dabs of paint and a thumb hole, a
// brush behind its shoulder, bristles dipped in `paint`, and a smudge of it on the brow.
export function artist(w: number, paper = PAPER, ink = INK, paint = PINK, wood = CRUST) {
  const board = hand(
    Array.from({ length: 49 }, (_, i) => {
      const a = (i / 48) * 2 * Math.PI
      // A kidney: an ellipse pinched in on its upper right, where the thumb goes through.
      const d = Math.atan2(Math.sin(a - 5.4), Math.cos(a - 5.4))
      const pinch = 0.16 * Math.exp(-3 * d * d)
      const r = 1 - pinch
      return [-0.5 + 0.56 * r * Math.cos(a), 0.86 + 0.34 * r * Math.sin(a)] as const
    }),
    0.03,
    0.012,
    301,
  )
  const hole = ring(302, -0.18, 0.72, 0.07)
  const dabs = (
    [
      [-0.82, 0.8, 0.09, paint],
      [-0.6, 1.02, 0.08, SAGE],
      [-0.32, 1.04, 0.075, ink],
      [-0.66, 0.66, 0.065, paper],
    ] as const
  )
    .map(([x, y, r, fill], i) => filled(ring(303 + i, x, y, r), fill, () => w * 0.55, ink))
    .join('')
  const palette = `${filled(board, wood, pressure(308, w, 0.3), ink)}${filled(hole, paper, () => w * 0.6, ink)}${dabs}`
  const handle = curve([0.5, 0.62], [0.95, 0.1], [1.32, -0.62], 16)
  const brush = `${stroke(hand(handle, 0.03, 0.006, 309), t => w * (1.5 - 0.5 * t), ink)}${filled(
    hand(rounded(1.38, -0.74, 0.14, 0.2, 0.04), 0.02, 0.006, 310),
    paper,
    () => w * 0.6,
    ink,
  )}${filled(
    hand(
      ribbon(curve([1.42, -0.84], [1.5, -1.0], [1.58, -1.18], 10), t => 0.07 * (1 - t) ** 0.7),
      0.02,
      0.008,
      311,
    ),
    paint,
    () => w * 0.6,
    ink,
  )}`
  const smudge = `<path d="${poly(hand(ring(312, -0.18, -0.66, 0.1), 0.02, 0.02, 313))}" style="fill:${paint}"/>`
  return { palette, brush, smudge }
}

// Bases: the librarian. Round glasses round the eyes, a temple arm back over the ear, and a stack
// of three books at its side in the site's colours.
export function librarian(w: number, paper = PAPER, ink = INK, spines = [PINK, SAGE, CRUST]) {
  const lenses = EYES.map((x, i) => ring(321 + i, x + 0.02, -0.2, 0.28))
  const bridge = hand(
    curve(
      [EYES[0] + 0.3, -0.24],
      [(EYES[0] + EYES[1]) / 2 + 0.01, -0.34],
      [EYES[1] - 0.26, -0.24],
      10,
    ),
    0.02,
    0.006,
    323,
  )
  const arm = hand(curve([EYES[1] + 0.3, -0.24], [0.68, -0.3], [0.94, -0.2], 10), 0.02, 0.006, 324)
  const glasses = `${lenses.map((l, i) => stroke(l, pressure(325 + i, w * 0.9, 0.2), ink)).join('')}${stroke(bridge, () => w * 0.8, ink)}${stroke(arm, () => w * 0.8, ink)}`
  const books = (
    [
      [1.02, 1.2, 0.78, 0.18, 0.04],
      [0.98, 1.01, 0.66, 0.2, -0.05],
      [1.05, 0.81, 0.72, 0.17, 0.06],
    ] as const
  )
    .map(([x, y, bw, bh, tilt], i) => {
      const c = Math.cos(tilt)
      const sn = Math.sin(tilt)
      const turn = (pts: Pt[]) =>
        pts.map(
          ([px, py]) =>
            [x + (px - x) * c - (py - y) * sn, y + (px - x) * sn + (py - y) * c] as const,
        )
      const cover = hand(turn(rounded(x, y, bw, bh, 0.03)), 0.03, 0.008, 330 + i)
      const pages = hand(
        turn([
          [x + bw / 2 - 0.1, y - bh / 2 + 0.04],
          [x + bw / 2 - 0.1, y + bh / 2 - 0.04],
        ]),
        0.02,
        0.004,
        334 + i,
      )
      const band = hand(
        turn([
          [x - bw / 2 + 0.12, y - bh / 2 + 0.03],
          [x - bw / 2 + 0.12, y + bh / 2 - 0.03],
        ]),
        0.02,
        0.004,
        338 + i,
      )
      return `${filled(cover, spines[i % spines.length], pressure(342 + i, w, 0.25), ink)}${stroke(pages, () => w * 0.5, ink)}${stroke(band, () => w * 0.7, paper)}`
    })
    .join('')
  return { glasses, books }
}

// Slides: the presenter. A bow tie under the chin and a pointer held out toward the slide. The
// site icon adds the slide itself, a small screen with a bar chart, for the pointer to touch.
export function presenter(w: number, paper = PAPER, ink = INK, tie = PINK) {
  const mx = (EYES[0] + EYES[1]) / 2
  const wing = (side: number) =>
    hand(
      [
        [mx, 0.44],
        [mx + side * 0.22, 0.32],
        [mx + side * 0.24, 0.58],
        [mx, 0.46],
      ],
      0.02,
      0.008,
      351 + side,
    )
  const bow = `${[-1, 1].map(side => filled(densify(wing(side), 0.03), tie, pressure(353 + side, w, 0.25), ink)).join('')}${filled(hand(rounded(mx, 0.45, 0.09, 0.11, 0.03), 0.01, 0.004, 355), tie, () => w * 0.7, ink)}`
  const stick = hand(
    [
      [-0.86, 0.42],
      [-1.66, -0.74],
    ],
    0.04,
    0.006,
    356,
  )
  const pointer = `${stroke(stick, t => w * (1.5 - 0.8 * t), ink)}${filled(ring(357, -1.68, -0.77, 0.05), paper, () => w * 0.6, ink)}`
  const frame = hand(rounded(-1.66, -1.02, 1.0, 0.72, 0.03), 0.03, 0.01, 358)
  const bars = (
    [
      [-1.94, 0.22, SAGE],
      [-1.72, 0.36, CRUST],
      [-1.5, 0.5, tie],
    ] as const
  )
    .map(([x, h, fill], i) =>
      filled(
        hand(rounded(x, -0.72 - h / 2, 0.14, h, 0.01), 0.02, 0.004, 360 + i),
        fill,
        () => w * 0.55,
        ink,
      ),
    )
    .join('')
  const legs = (
    [
      [
        [-1.94, -0.66],
        [-2.08, -0.2],
      ],
      [
        [-1.38, -0.66],
        [-1.24, -0.2],
      ],
    ] as Pt[][]
  )
    .map((pts, i) => stroke(hand(pts, 0.03, 0.006, 364 + i), () => w * 0.9, ink))
    .join('')
  const screen = `${legs}${filled(frame, paper, pressure(366, w, 0.3), ink)}${bars}`
  return { bow, pointer, screen }
}

// Stream: the reader. The librarian's glasses on, and a sheet held up in front at a tilt, a stubby
// hand at each side, its top corner dog-eared, four lines of text on it, one marked in `mark`.
export function reader(w: number, paper = PAPER, ink = INK, mark = PINK) {
  const [cx, cy, sw, sh, tilt] = [-0.42, 0.74, 1.02, 0.72, -0.12]
  const c = Math.cos(tilt)
  const sn = Math.sin(tilt)
  const turn = (pts: readonly Pt[]): Pt[] =>
    pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c])
  const [l, r, t, b] = [cx - sw / 2, cx + sw / 2, cy - sh / 2, cy + sh / 2]
  const ear = 0.15
  const outline = hand(
    turn([
      [l, t],
      [r - ear, t],
      [r, t + ear],
      [r, b],
      [l, b],
      [l, t],
    ]),
    0.03,
    0.01,
    391,
  )
  const fold = hand(
    turn([
      [r - ear, t],
      [r - ear + 0.02, t + ear - 0.02],
      [r, t + ear],
    ]),
    0.02,
    0.004,
    392,
  )
  const rows = [0.74, 0.62, 0.8, 0.46].map((len, i) => {
    const y = t + 0.16 + i * 0.13
    return hand(
      turn([
        [l + 0.11, y],
        [l + 0.11 + (sw - 0.22) * len, y],
      ]),
      0.03,
      0.008,
      393 + i,
    )
  })
  const text = rows
    .map(
      (row, i) =>
        `${i === 1 ? stroke(row, () => w * 2.6, mark) : ''}${stroke(row, () => w * 0.55, ink)}`,
    )
    .join('')
  const paws = [l + 0.02, r - 0.02]
    .map((x, i) =>
      filled(
        hand(turn(rounded(x, cy + 0.08, 0.2, 0.17, 0.08)), 0.02, 0.006, 398 + i),
        paper,
        pressure(400 + i, w, 0.2),
        ink,
      ),
    )
    .join('')
  const sheet = `${filled(outline, paper, pressure(402, w, 0.3), ink)}${stroke(fold, () => w * 0.7, ink)}${text}${paws}`
  return { glasses: librarian(w, paper, ink).glasses, sheet }
}

// Courses: the student. A mortarboard with a pink tassel, a spiral notepad held up in front with a
// line of handwriting on its ruled lines, and a pencil in its other hand, its point at the end of
// that line, as if mid-word.
export function student(w: number, paper = PAPER, ink = INK, eraser = PINK, wood = CRUST) {
  const skull = hand(
    [
      [-0.64, -0.8],
      [-0.6, -1.06],
      [0.02, -1.14],
      [0.62, -1.04],
      [0.64, -0.8],
      [0.02, -0.88],
      [-0.64, -0.8],
    ],
    0.03,
    0.01,
    431,
  )
  const board = hand(
    [
      [-1.06, -1.1],
      [0.06, -1.34],
      [1.1, -1.12],
      [0.0, -0.92],
      [-1.06, -1.1],
    ],
    0.03,
    0.01,
    432,
  )
  const cord = hand(curve([0.04, -1.12], [0.9, -1.16], [1.0, -0.7], 14), 0.03, 0.006, 433)
  const tassel = hand(
    ribbon(
      densify(
        [
          [1.0, -0.72],
          [1.03, -0.42],
        ],
        0.03,
      ),
      t => 0.035 + 0.05 * t,
    ),
    0.02,
    0.006,
    434,
  )
  const cap = `${filled(skull, ink, pressure(435, w, 0.25), ink)}${filled(board, ink, pressure(436, w, 0.25), ink)}${stroke(cord, () => w * 0.7, eraser)}${filled(tassel, eraser, () => w * 0.5, ink)}${filled(ring(437, 0.04, -1.12, 0.05), eraser, () => w * 0.4, ink)}`

  const [cx, cy, pw, ph, tilt] = [-0.46, 0.8, 0.96, 0.68, 0.06]
  const c = Math.cos(tilt)
  const sn = Math.sin(tilt)
  const turn = (pts: readonly Pt[]): Pt[] =>
    pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c])
  const [l, r, t] = [cx - pw / 2, cx + pw / 2, cy - ph / 2]
  const sheet = hand(turn(rounded(cx, cy, pw, ph, 0.04)), 0.03, 0.01, 441)
  const rules = [0.22, 0.36, 0.5]
    .map((d, i) =>
      stroke(
        hand(
          turn([
            [l + 0.08, t + d + 0.04],
            [r - 0.08, t + d + 0.04],
          ]),
          0.03,
          0.006,
          442 + i,
        ),
        () => w * 0.35,
        ink,
      ),
    )
    .join('')
  const spiral = Array.from({ length: 7 }, (_, i) => {
    const [x, y] = turn([[l + 0.12 + (i * (pw - 0.24)) / 6, t]])[0]
    return stroke(ring(446 + i, x, y, 0.045), () => w * 0.55, ink)
  }).join('')
  // Handwriting on the first rule: two words of small humps, the second ending where the pencil is.
  const x0 = l + 0.1
  const x1 = cx + 0.18
  const gap = 0.07
  const split = x0 + (x1 - x0 - gap) * 0.45
  const writing = (
    [
      [x0, split],
      [split + gap, x1],
    ] as const
  )
    .map(([a, z], i) => {
      const humps = Math.max(2, Math.round((z - a) / 0.06))
      const word = Array.from({ length: humps * 8 + 1 }, (_, k) => {
        const u = k / (humps * 8)
        return [a + (z - a) * u, t + 0.25 - 0.05 * Math.abs(Math.sin(Math.PI * humps * u))] as Pt
      })
      return stroke(hand(turn(word), 0.02, 0.004, 454 + i), () => w * 0.5, ink)
    })
    .join('')
  const pad = `${filled(sheet, paper, pressure(455, w, 0.3), ink)}${rules}${writing}${spiral}${filled(
    hand(turn(rounded(l + 0.02, cy + 0.12, 0.2, 0.17, 0.08)), 0.02, 0.006, 456),
    paper,
    pressure(457, w, 0.2),
    ink,
  )}`

  // The pencil, from its point at the end of the line up to the eraser, held near the top.
  const [tip] = turn([[x1, t + 0.22]])
  const top: Pt = [tip[0] + 0.56, tip[1] - 0.66]
  const len = Math.hypot(top[0] - tip[0], top[1] - tip[1])
  const along = (d: number): Pt => [
    tip[0] + ((top[0] - tip[0]) * d) / len,
    tip[1] + ((top[1] - tip[1]) * d) / len,
  ]
  const band = (from: number, to: number, half: number, fill: string, seed: number) =>
    filled(
      hand(
        ribbon(densify([along(from), along(to)], 0.03), () => half),
        0.02,
        0.004,
        seed,
      ),
      fill,
      () => w * 0.6,
      ink,
    )
  const point = filled(
    hand(
      ribbon(densify([along(0.01), along(0.2)], 0.02), u => 0.075 * u),
      0.01,
      0.003,
      458,
    ),
    paper,
    () => w * 0.55,
    ink,
  )
  const lead = filled(
    ribbon(densify([along(0), along(0.07)], 0.01), u => 0.03 * u),
    ink,
    () => w * 0.3,
    ink,
  )
  const grip = along(len - 0.4)
  const pencil = `${band(0.2, len - 0.18, 0.075, wood, 459)}${band(len - 0.18, len - 0.1, 0.08, paper, 460)}${band(len - 0.1, len, 0.075, eraser, 461)}${point}${lead}${filled(
    hand(rounded(grip[0] + 0.04, grip[1] + 0.02, 0.22, 0.19, 0.09), 0.02, 0.006, 462),
    paper,
    pressure(463, w, 0.2),
    ink,
  )}`
  return { cap, pad, pencil }
}

// University notes: the engineer. A hard hat with a ridge down its crown and a brim longer at the
// front, an Erlenmeyer flask held up by its neck, `liquid` in it with bubbles rising out, and the
// iron ring on the little finger of the hand that holds it. `bubbles` are returned apart, so the pet
// can float them.
export function engineer(
  w: number,
  paper = PAPER,
  ink = INK,
  liquid = PINK,
  hat = CRUST,
  iron = '#878580',
) {
  const dome = hand(
    [
      ...Array.from({ length: 25 }, (_, i) => {
        const a = Math.PI + (i / 24) * Math.PI
        return [0.02 + 0.8 * Math.cos(a), -0.84 + 0.6 * Math.sin(a)] as const
      }),
      [-0.78, -0.84],
    ],
    0.03,
    0.01,
    471,
  )
  const ridge = [-0.12, 0.14]
    .map((x, i) =>
      stroke(
        hand(curve([x, -0.84], [x * 0.6 + 0.02, -1.36], [0.02, -1.44], 10), 0.03, 0.006, 472 + i),
        () => w * 0.6,
        ink,
      ),
    )
    .join('')
  const brim = hand(
    [
      [-1.16, -0.86],
      [-0.4, -0.94],
      [0.5, -0.92],
      [0.96, -0.86],
      [0.94, -0.76],
      [0.4, -0.8],
      [-0.5, -0.8],
      [-1.12, -0.76],
      [-1.16, -0.86],
    ],
    0.03,
    0.008,
    474,
  )
  const hardhat = `${filled(dome, hat, pressure(475, w, 0.3), ink)}${ridge}${filled(brim, hat, pressure(476, w, 0.3), ink)}`

  const [nl, nr, nt, nb, bl, br, bb] = [-1.0, -0.76, 0.18, 0.48, -1.36, -0.4, 1.2]
  const outline: Pt[] = [
    [nl, nt],
    [nl, nb],
    [bl + 0.06, bb - 0.1],
    ...curve([bl + 0.06, bb - 0.1], [bl, bb], [bl + 0.12, bb], 4).slice(1),
    [br - 0.12, bb],
    ...curve([br - 0.12, bb], [br, bb], [br - 0.06, bb - 0.1], 4).slice(1),
    [nr, nb],
    [nr, nt],
  ]
  // The glass's side at height y, for the liquid's surface.
  const side = (y: number, s: number) => {
    const u = (y - nb) / (bb - 0.1 - nb)
    return s < 0 ? nl + (bl + 0.06 - nl) * u : nr + (br - 0.06 - nr) * u
  }
  const surface = 0.78
  const fluid = hand(
    [
      [side(surface, -1) + 0.03, surface],
      [side(surface, 1) - 0.03, surface + 0.02],
      [br - 0.1, bb - 0.04],
      [bl + 0.1, bb - 0.04],
      [side(surface, -1) + 0.03, surface],
    ],
    0.03,
    0.008,
    477,
  )
  const glass = hand(densify(outline, 0.04), 0.03, 0.008, 478)
  const lip = hand(rounded((nl + nr) / 2, nt - 0.02, nr - nl + 0.1, 0.07, 0.03), 0.02, 0.004, 479)
  const ticks = [0.86, 0.98]
    .map((y, i) =>
      stroke(
        hand(
          [
            [side(y, 1) - 0.04, y],
            [side(y, 1) - 0.16, y],
          ],
          0.02,
          0.004,
          480 + i,
        ),
        () => w * 0.45,
        ink,
      ),
    )
    .join('')
  const inside = [
    [-1.0, 1.0, 0.04],
    [-0.74, 0.9, 0.03],
    [-0.84, 1.08, 0.025],
  ]
    .map(([x, y, r], i) => stroke(ring(482 + i, x, y, r), () => w * 0.4, ink))
    .join('')
  const flask = `<path d="${poly(glass)}" style="fill:${paper}"/><path d="${poly(fluid)}" style="fill:${liquid}"/>${stroke(glass, pressure(485, w, 0.3), ink)}${filled(lip, paper, () => w * 0.7, ink)}${ticks}${inside}`
  const bubbles = [
    [-0.86, -0.02, 0.06],
    [-1.02, -0.26, 0.045],
    [-0.9, -0.48, 0.035],
  ]
    .map(
      ([x, y, r], i) =>
        `<g class="pet-bubble">${filled(ring(486 + i, x, y, r), paper, () => w * 0.5, ink)}</g>`,
    )
    .join('')
  // The hand round the neck, the iron ring on its little finger.
  const paw = hand(rounded(nr + 0.04, nb - 0.06, 0.24, 0.2, 0.09), 0.02, 0.006, 490)
  const ironRing = stroke(
    hand(
      [
        [nr + 0.08, nb + 0.0],
        [nr + 0.13, nb + 0.03],
      ],
      0.01,
      0.003,
      491,
    ),
    () => w * 1.5,
    iron,
  )
  const grip = `${filled(paw, paper, pressure(492, w, 0.2), ink)}${ironRing}`
  return { hardhat, flask: `${flask}${grip}`, bubbles }
}

// Cinema: the film nerd. Thick square specs, and a striped popcorn bucket held up, heaped over the
// rim. `kernel` is one more piece apart from the heap, so the pet can pop it up to its mouth.
export function cinephile(w: number, paper = PAPER, ink = INK, stripe = PINK, butter = CRUST) {
  const frames = EYES.map((x, i) =>
    hand(rounded(x + 0.02, -0.2, 0.52, 0.42, 0.09), 0.02, 0.006, 701 + i),
  )
  const bridge = hand(
    [
      [EYES[0] + 0.28, -0.26],
      [EYES[1] - 0.24, -0.26],
    ],
    0.02,
    0.006,
    703,
  )
  const temple = hand(
    curve([EYES[1] + 0.28, -0.26], [0.7, -0.3], [0.96, -0.2], 10),
    0.02,
    0.006,
    704,
  )
  const specs = `${frames.map((f, i) => stroke(f, pressure(705 + i, w * 1.6, 0.15), ink)).join('')}${stroke(bridge, () => w * 1.4, ink)}${stroke(temple, () => w * 1.1, ink)}`

  const [tl, tr, top, bl, br, bottom] = [-1.26, -0.3, 0.44, -1.1, -0.46, 1.22]
  const at = (u: number, y: number) => {
    const v = (y - top) / (bottom - top)
    return tl + (bl - tl) * v + (tr - tl + (br - bl - (tr - tl)) * v) * u
  }
  const bucket = hand(
    [
      [tl, top],
      [bl, bottom],
      [br, bottom],
      [tr, top],
      [tl, top],
    ],
    0.03,
    0.01,
    710,
  )
  const stripes = [0.12, 0.44, 0.76]
    .map((u, i) => {
      const quad = hand(
        [
          [at(u, top), top],
          [at(u, bottom), bottom],
          [at(u + 0.16, bottom), bottom],
          [at(u + 0.16, top), top],
        ],
        0.03,
        0.006,
        711 + i,
      )
      return `<path d="${poly(quad)}" style="fill:${stripe}"/>`
    })
    .join('')
  const rim = hand(rounded((tl + tr) / 2, top + 0.02, tr - tl + 0.08, 0.1, 0.04), 0.02, 0.006, 715)
  // A kernel: three puffed lobes round a centre.
  const puff = (x: number, y: number, r: number, seed: number) =>
    hand(
      Array.from({ length: 37 }, (_, i) => {
        const a = (i / 36) * 2 * Math.PI
        const k = r * (0.82 + 0.22 * Math.abs(Math.sin(1.5 * a + seed)))
        return [x + k * Math.cos(a), y + k * Math.sin(a)] as const
      }),
      0.02,
      0.006,
      seed,
    )
  const heap = (
    [
      [-1.14, 0.36, 0.12],
      [-0.48, 0.34, 0.12],
      [-0.96, 0.28, 0.13],
      [-0.64, 0.24, 0.13],
      [-0.8, 0.3, 0.12],
      [-0.88, 0.14, 0.12],
      [-0.68, 0.1, 0.11],
    ] as const
  )
    .map(([x, y, r], i) =>
      filled(
        puff(x, y, r, 720 + i),
        i % 3 === 1 ? butter : paper,
        pressure(730 + i, w * 0.8, 0.25),
        ink,
      ),
    )
    .join('')
  // The hand round the bucket's far side.
  const paw = hand(rounded(tr + 0.02, 0.74, 0.24, 0.22, 0.09), 0.02, 0.006, 740)
  const popcorn = `<path d="${poly(bucket)}" style="fill:${paper}"/>${stripes}${stroke(bucket, pressure(741, w, 0.3), ink)}${heap}${filled(rim, paper, pressure(742, w * 0.9, 0.2), ink)}${filled(paw, paper, pressure(743, w, 0.2), ink)}`
  const kernel = filled(puff(-0.78, 0.06, 0.1, 745), paper, pressure(746, w * 0.8, 0.25), ink)
  return { specs, popcorn, kernel }
}

// Cold weather: a knit toque pulled down to the brow, ribbed cuff and pom-pom, and a striped scarf
// wound under the chin with one end hanging. `puffs` are three breaths, drawn apart so the pet can
// let them out one at a time.
export function winter(w: number, paper = PAPER, ink = INK, knit = SAGE, stripe = PINK) {
  // The crown's line, x across the brow at height y of the cuff top.
  const crown: Pt[] = Array.from({ length: 31 }, (_, i) => {
    const a = Math.PI + (i / 30) * Math.PI
    return [0.02 + 0.9 * Math.cos(a), -0.66 + 0.58 * Math.sin(a)] as const
  })
  const dome = hand([...crown, [-0.88, -0.66]], 0.03, 0.01, 601)
  const ribs = [-0.42, 0.04, 0.48]
    .map((x, i) =>
      stroke(
        hand(
          curve([x, -0.68], [x * 0.8 + 0.01, -0.98], [x * 0.35 + 0.02, -1.2], 10),
          0.03,
          0.006,
          602 + i,
        ),
        () => w * 0.5,
        ink,
      ),
    )
    .join('')
  const sag = (x: number) => 0.05 * (1 - x * x)
  const cuffEdge = (y: number, from: number, to: number) =>
    Array.from({ length: 21 }, (_, i) => {
      const x = from + ((to - from) * i) / 20
      return [x, y + sag(x)] as const
    })
  const cuff = hand(
    [...cuffEdge(-0.74, -0.96, 0.98), ...cuffEdge(-0.5, 0.98, -0.96), [-0.96, -0.74]],
    0.03,
    0.008,
    606,
  )
  const cuffRibs = Array.from({ length: 13 }, (_, i) => -0.84 + i * 0.14)
    .map((x, i) =>
      stroke(
        hand(
          [
            [x, -0.7 + sag(x)],
            [x + 0.01, -0.54 + sag(x)],
          ],
          0.02,
          0.004,
          610 + i,
        ),
        () => w * 0.4,
        ink,
      ),
    )
    .join('')
  // The pom-pom: a tuft, its edge spiky with loose yarn.
  const tuft: Pt[] = Array.from({ length: 37 }, (_, i) => {
    const a = (i / 36) * 2 * Math.PI
    const r = 0.17 * (1 + (i % 2 ? 0.16 : -0.06))
    return [0.06 + r * Math.cos(a), -1.3 + r * Math.sin(a)] as const
  })
  const pom = filled(hand(tuft, 0.02, 0.008, 630), stripe, pressure(631, w * 0.8, 0.3), ink)
  const toque = `${filled(dome, knit, pressure(632, w, 0.3), ink)}${ribs}${filled(cuff, knit, pressure(633, w, 0.3), ink)}${cuffRibs}${pom}`

  // The wrap, below the mouth: it dips at the middle, as cloth over a round chin does.
  const top = (x: number) => 0.36 + 0.06 * (1 - x * x)
  const bottom = (x: number) => 0.64 + 0.08 * (1 - x * x)
  const along = (f: (x: number) => number, from: number, to: number) =>
    Array.from({ length: 21 }, (_, i) => {
      const x = from + ((to - from) * i) / 20
      return [x, f(x)] as const
    })
  const wrap = hand(
    [...along(top, -0.96, 0.96), ...along(bottom, 0.9, -0.9), [-0.96, top(-0.96)]],
    0.03,
    0.01,
    640,
  )
  const band = (x0: number, x1: number, seed: number) =>
    `<path d="${poly(hand([...along(top, x0, x1), ...along(bottom, x1, x0)], 0.03, 0.006, seed))}" style="fill:${stripe}"/>`
  // The end hangs from a knot at the back of the neck, stripes and fringe and all.
  const end = curve([0.56, 0.6], [0.78, 0.88], [0.7, 1.18], 16)
  const tail = hand(
    ribbon(end, t => 0.13 - 0.02 * t),
    0.03,
    0.008,
    645,
  )
  const tailBand = (t0: number, t1: number, seed: number) => {
    const piece = end.slice(Math.round(t0 * 16), Math.round(t1 * 16) + 1)
    return `<path d="${poly(
      hand(
        ribbon(piece, () => 0.12),
        0.03,
        0.006,
        seed,
      ),
    )}" style="fill:${stripe}"/>`
  }
  const fringe = [-0.09, -0.03, 0.03, 0.09]
    .map((dx, i) =>
      stroke(
        hand(
          [
            [0.7 + dx, 1.18],
            [0.69 + dx * 1.2, 1.3],
          ],
          0.02,
          0.006,
          650 + i,
        ),
        () => w * 0.5,
        ink,
      ),
    )
    .join('')
  const knot = filled(
    hand(rounded(0.6, 0.52, 0.26, 0.26, 0.1), 0.02, 0.008, 655),
    knit,
    pressure(656, w, 0.3),
    ink,
  )
  const outline = (pts: Pt[], seed: number) => stroke(pts, pressure(seed, w, 0.3), ink)
  const scarf =
    `<path d="${poly(wrap)}" style="fill:${knit}"/>${band(-0.5, -0.36, 641)}${band(0.2, 0.34, 642)}${outline(wrap, 643)}` +
    `<path d="${poly(tail)}" style="fill:${knit}"/>${tailBand(0.45, 0.62, 646)}${tailBand(0.78, 0.92, 647)}${outline(tail, 648)}${fringe}${knot}`

  // Breath in the cold, out of the mouth and off to the side it faces.
  const puffs = (
    [
      [-0.98, 0.16, 0.09],
      [-1.22, 0.04, 0.12],
      [-1.5, -0.1, 0.1],
    ] as const
  )
    .map(([x, y, r], i) => {
      const cloud = hand(rounded(x, y, r * 2.2, r * 1.6, r * 0.7), 0.02, 0.008, 660 + i)
      return `<g class="pet-puff">${filled(cloud, paper, () => w * 0.45, ink)}</g>`
    })
    .join('')
  return { toque, scarf, puffs }
}

// Triathlon in the cold: the wind vest a rider zips on for a cold descent. `body` is the pet's own
// outline, so the shell's fill tucks under its line from the mouth down. A stand-up collar, a zip
// down the front (the left, where it faces) with its pull, a reflective band across the chest, and
// the back panel caught by the wind: `flaps` are two drawings of it to swap between.
export function windVest(body: readonly Pt[], w: number, ink = INK, shell = SAGE, tape = PINK) {
  // A V neck from shoulder to shoulder, the arms coming out above it, down to where the zip stops
  // under the mouth (it faces left, so the V sits left of centre); piping a little inside it.
  const zx = -0.16
  const top = (x: number) => 0.43 - 0.48 * ((x - zx) / (x < zx ? 0.74 : 1.06)) ** 2
  const across = (f: (x: number) => number, from: number, to: number) =>
    Array.from({ length: 33 }, (_, i) => {
      const x = from + ((to - from) * i) / 32
      return [x, f(x)] as const
    })
  // Half the body's width at height y, a little inside its line.
  const edge = (y: number) => 0.95 * Math.sqrt(Math.max(0, 1 - y * y))
  const inside = (pts: Pt[]) => pts.filter(([x, y]) => Math.abs(x) < edge(y))
  const lower = body.map(([x, y]) => [x * 0.975, y * 0.975] as const).filter(([x, y]) => y > top(x))
  const shape: Pt[] = [...lower, ...across(top, lower[lower.length - 1][0], lower[0][0])]
  // The silhouette's lines follow the theme's ink; the marks on the shell stay dark on its fill.
  const line = (pts: Pt[], seed: number, k = 1, color = INK) =>
    stroke(pts, pressure(seed, w * k, 0.3), color)
  const collar = line(hand(inside(across(top, -1, 1)), 0.03, 0.008, 701), 702, 1, ink)
  const crease = line(hand(inside(across(x => top(x) + 0.11, -1, 1)), 0.03, 0.008, 703), 704, 0.55)
  const mid = (x: number) => 0.7 + 0.04 * (1 - x * x)
  const tapeEdge = (d: number, from: number, to: number) => across(x => mid(x) + d, from, to)
  const reach = edge(0.78)
  const band = hand(
    [
      ...tapeEdge(-0.05, -reach, reach),
      ...tapeEdge(0.05, reach, -reach),
      [-reach, mid(-reach) - 0.05],
    ],
    0.03,
    0.006,
    705,
  )
  const zipLine = curve([zx, top(zx)], [zx - 0.1, 0.68], [zx - 0.14, 0.94], 16)
  const zip = line(hand(zipLine, 0.03, 0.006, 706), 707, 0.75)
  const teeth = zipLine
    .filter((_, i) => i % 2 && i < 15)
    .map(([x, y]) =>
      stroke(
        [
          [x - 0.035, y],
          [x + 0.035, y],
        ],
        () => w * 0.35,
        INK,
      ),
    )
    .join('')
  const pull = filled(
    hand(rounded(zx + 0.01, top(zx) + 0.1, 0.07, 0.15, 0.03), 0.02, 0.004, 708),
    INK,
    () => w * 0.3,
    INK,
  )
  const vest =
    `<path d="${poly(shape)}" style="fill:${shell}"/>${crease}` +
    `<path d="${poly(band)}" style="fill:${tape}"/>${stroke(band, () => w * 0.45, INK)}${zip}${teeth}${collar}${pull}`

  // The back panel, out from the body's own line behind it (on its right) and back, its hem
  // rippling; the wind's streaks trail off behind.
  const back = body.filter(([x, y]) => {
    const a = Math.atan2(y, x)
    return a > 0.35 && a < 1.25
  })
  const flaps = [
    [0.12, 0.25, 0],
    [0.17, 0.2, 2.2],
  ].map(([amp, ripple, phase], k) => {
    const n = back.length - 1
    const out = back.map(([x, y], i) => {
      const t = i / n
      const s =
        1 + amp * Math.sin(Math.PI * t) ** 0.8 * (1 + ripple * Math.sin(4 * Math.PI * t + phase))
      return [x * s, y * s] as const
    })
    const panel = [...out, ...back.map(([x, y]) => [x * 0.92, y * 0.92] as const).reverse()]
    const streaks = [
      [1.2 + 0.05 * k, 0.36, 0.34],
      [1.12 - 0.03 * k, 0.74, 0.26],
    ]
      .map(([x, y, len], i) =>
        stroke(
          hand(
            curve([x, y], [x + len / 2, y + 0.04], [x + len, y - 0.02], 10),
            0.03,
            0.006,
            710 + 2 * k + i,
          ),
          even(w * 0.6),
          ink,
        ),
      )
      .join('')
    return `<path d="${poly(panel)}" style="fill:${shell}"/>${line(out, 714 + k, 1, ink)}${streaks}`
  })
  return { vest, flaps }
}

// Rain: a little umbrella held up in the paw on its right, the canopy tipped forward over its head,
// pink with sage panels between the ribs and a scalloped hem. `held` has it open and, hidden, blown
// inside out for storms. `rain` is two drawings of the streaks falling past it and the drips off the
// rib tips, to swap between.
export function umbrella(w: number, paper = PAPER, ink = INK, cloth = PINK, panel = SAGE) {
  const tilt = 0.2
  const hold: Pt = [0.84, 0.2]
  const up: Pt = [-Math.sin(tilt), -Math.cos(tilt)]
  const along: Pt = [Math.cos(tilt), -Math.sin(tilt)]
  const centre: Pt = [hold[0] + 1.9 * up[0], hold[1] + 1.9 * up[1]]
  const half = 0.74
  const dome = 0.4
  // Canopy points: u across the hem (-1 to 1), v up the shaft.
  const at = (u: number, v: number): Pt => [
    centre[0] + u * half * along[0] + v * up[0],
    centre[1] + u * half * along[1] + v * up[1],
  ]
  const tips = [-1, -0.5, 0, 0.5, 1]
  const rib = (u: number) =>
    curve([u, 0], [u * 0.92, dome * 0.86], [0, dome], 12).map(([a, b]) => at(a, b))
  const hem = (u0: number, u1: number) =>
    Array.from({ length: 9 }, (_, i) => {
      const t = i / 8
      return at(u0 + (u1 - u0) * t, 0.08 * Math.sin(Math.PI * t))
    })
  const panelShape = (k: number) => [
    ...hem(tips[k], tips[k + 1]),
    ...rib(tips[k + 1]).slice(1),
    ...rib(tips[k]).reverse(),
  ]
  const canopy = hand(
    [
      ...tips.slice(0, -1).flatMap((u, k) => hem(u, tips[k + 1]).slice(k ? 1 : 0)),
      ...rib(1).slice(1),
      ...rib(-1).reverse(),
    ],
    0.03,
    0.008,
    801,
  )
  const panels = [1, 3]
    .map(
      k => `<path d="${poly(hand(panelShape(k), 0.03, 0.006, 802 + k))}" style="fill:${panel}"/>`,
    )
    .join('')
  const ribs = [-0.5, 0, 0.5]
    .map((u, i) => stroke(hand(rib(u), 0.03, 0.006, 806 + i), () => w * 0.5, INK))
    .join('')
  const foot: Pt = [hold[0] - 0.22 * up[0], hold[1] - 0.22 * up[1]]
  const shaft = stroke(hand([centre, foot], 0.03, 0.006, 810), pressure(811, w * 0.9, 0.2), ink)
  const crook = stroke(
    curve(foot, [foot[0] - 0.04, foot[1] + 0.2], [foot[0] - 0.2, foot[1] + 0.06], 10),
    pressure(812, w * 0.9, 0.2),
    ink,
  )
  const top = at(0, dome)
  const ferrule = stroke([top, [top[0] + 0.1 * up[0], top[1] + 0.1 * up[1]]], () => w * 0.9, ink)
  const paw = filled(
    hand(rounded(hold[0], hold[1], 0.24, 0.2, 0.09), 0.02, 0.006, 813),
    paper,
    pressure(814, w, 0.2),
    ink,
  )
  const open =
    `<path d="${poly(canopy)}" style="fill:${cloth}"/>${panels}${ribs}` +
    `${stroke(canopy, pressure(815, w, 0.3), ink)}${ferrule}`

  // Inside out in a storm: the ribs bent up past the top of the shaft into a cup open to the sky, the
  // bare wire tips poking past the hem, and the end rib on the right snapped so its corner droops.
  // Two drawings of the cloth flapping, to swap between.
  const shaftTop = stroke(hand([centre, at(0, dome + 0.04)], 0.03, 0.004, 830), () => w * 0.9, ink)
  const inverted = [0, 1].map(k => {
    const tip = (u: number): Pt =>
      u === 1
        ? [1, dome + 0.06 + 0.05 * k]
        : [u, dome + 0.52 + 0.07 * (k ? u : -u) - 0.13 * Math.sqrt(1 - u * u)]
    const bend = (u: number): Pt => [tip(u)[0] * 0.62, dome + 0.05]
    const rib = (u: number) => curve([0, dome], bend(u), tip(u), 10)
    const rim = (u0: number, u1: number) => {
      const [a, b] = [tip(u0), tip(u1)]
      return Array.from({ length: 9 }, (_, i): Pt => {
        const t = i / 8
        const dip = (0.07 + 0.03 * k) * Math.sin(Math.PI * t)
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t - dip]
      })
    }
    // The far side of the hem, seen over the near one: the inside of the cup.
    const far = Array.from({ length: 17 }, (_, i): Pt => {
      const t = i / 16
      const [a, b] = [tip(-1), tip(1)]
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t + 0.14 * Math.sin(Math.PI * t)]
    })
    const near = tips.slice(0, -1).flatMap((u, j) => rim(u, tips[j + 1]).slice(j ? 1 : 0))
    const xy = (pts: readonly Pt[]) => pts.map(([a, b]) => at(a, b))
    const inside = hand(xy([...near, ...far.slice().reverse()]), 0.03, 0.006, 832 + k)
    const outside = hand(
      xy([...rib(-1), ...near.slice(1), ...rib(1).reverse().slice(1)]),
      0.03,
      0.008,
      834 + k,
    )
    const cups = [1, 3]
      .map(j => {
        const shape = [
          ...rim(tips[j], tips[j + 1]),
          ...rib(tips[j + 1])
            .reverse()
            .slice(1),
          ...rib(tips[j]).slice(1),
        ]
        return `<path d="${poly(hand(xy(shape), 0.03, 0.006, 836 + j + k))}" style="fill:${panel}"/>`
      })
      .join('')
    const wires = [-0.5, 0, 0.5]
      .map((u, i) => stroke(hand(xy(rib(u)), 0.03, 0.006, 840 + i + k), () => w * 0.5, INK))
      .join('')
    const bare = tips
      .map(u => {
        const [t, b] = [tip(u), bend(u)]
        const len = Math.hypot(t[0] - b[0], t[1] - b[1])
        const end: Pt = [t[0] + ((t[0] - b[0]) / len) * 0.1, t[1] + ((t[1] - b[1]) / len) * 0.1]
        return stroke(xy([t, end]), () => w * 0.6, ink)
      })
      .join('')
    return (
      `<path d="${poly(inside)}" style="fill:color-mix(in srgb, ${cloth} 80%, ${ink})"/>` +
      `${stroke(hand(xy(far), 0.03, 0.006, 846 + k), pressure(847, w * 0.8, 0.3), ink)}` +
      `<path d="${poly(outside)}" style="fill:${cloth}"/>${cups}${wires}` +
      `${stroke(outside, pressure(848 + k, w, 0.3), ink)}${bare}`
    )
  })
  const held =
    `<g class="pet-brolly">${shaft}${crook}<g class="pet-brolly-open">${open}</g>` +
    `<g class="pet-brolly-out">${shaftTop}${inverted.map(f => `<g class="pet-out-frame">${f}</g>`).join('')}</g>` +
    `${paw}</g>`

  // Streaks slanting a little, clear of the canopy, the face and the breath; each drawing has them
  // a step further down, and a drop hanging off each end rib that falls the next.
  const streaks = [
    [-1.3, -1.5],
    [-0.85, -2.05],
    [-1.55, -0.85],
    [1.62, -1.35],
    [1.68, -0.6],
    [0.95, -2.3],
  ] as const
  const rain = [0, 1].map(k => {
    const lines = streaks
      .map(([x, y]) => {
        const [sx, sy] = [x - 0.04 * k, y + 0.24 * k]
        return stroke(
          [
            [sx, sy],
            [sx - 0.05, sy + 0.18],
          ],
          even(w * 0.55),
          ink,
        )
      })
      .join('')
    const drips = [at(-1, 0), at(1, 0)]
      .map(([x, y]) => {
        const d = 0.1 + 0.2 * k
        return filled(
          hand(rounded(x, y + d, 0.06, 0.09, 0.03), 0.02, 0.004, 820 + k),
          paper,
          () => w * 0.45,
          ink,
        )
      })
      .join('')
    return `${lines}${drips}`
  })
  return { held, rain }
}

// Heat: sunglasses, two dark lenses in the theme's ink rims with a glint across each, a bridge,
// and an arm back over the side of the head.
export function sunglasses(w: number, ink = INK, lens = INK, glint = PINK) {
  const lenses = EYES.map((x, i) =>
    hand(
      [
        [x - 0.24, -0.38],
        [x + 0.28, -0.38],
        [x + 0.24, -0.12],
        [x + 0.12, -0.02],
        [x - 0.12, -0.02],
        [x - 0.22, -0.14],
        [x - 0.24, -0.38],
      ],
      0.02,
      0.006,
      850 + i,
    ),
  )
  const shine = EYES.map(x =>
    stroke(
      [
        [x - 0.12, -0.18],
        [x + 0.04, -0.32],
      ],
      () => w * 0.8,
      glint,
    ),
  )
  const bridge = stroke(
    hand(
      curve([EYES[0] + 0.28, -0.34], [(EYES[0] + EYES[1]) / 2, -0.42], [EYES[1] - 0.24, -0.34], 8),
      0.02,
      0.006,
      852,
    ),
    () => w * 1.3,
    ink,
  )
  const temple = stroke(
    hand(curve([EYES[1] + 0.28, -0.34], [0.7, -0.36], [0.96, -0.26], 10), 0.02, 0.006, 853),
    () => w * 1.1,
    ink,
  )
  return (
    lenses
      .map((l, i) => `${filled(l, lens, pressure(854 + i, w * 1.4, 0.15), ink)}${shine[i]}`)
      .join('') +
    bridge +
    temple
  )
}

// A sun lounger for the hottest afternoons: a striped cushion along a flat bed and a backrest
// raised behind it (on its right, as it faces left), on a pen-line frame with short legs to the
// ground at y = 1.4. The pet lies back in it with its body turned 30°.
export function lounger(w: number, paper = PAPER, ink = INK, stripe = PINK, wood = CRUST) {
  const hinge: Pt = [0.55, 0.98]
  const head: Pt = [1.55, -0.55]
  const foot: Pt = [-1.62, 0.98]
  const line = (pts: readonly Pt[], seed: number, k = 1, color = ink) =>
    stroke(hand(pts, 0.03, 0.006, seed), pressure(seed + 1, w * k, 0.2), color)
  // The frame under the cushion, and its legs.
  const frame =
    line([foot, hinge, head], 860, 1.3) +
    (
      [
        [
          [-1.45, 1.0],
          [-1.5, 1.4],
        ],
        [
          [0.42, 1.0],
          [0.48, 1.4],
        ],
        [
          [1.15, 0.32],
          [1.05, 1.4],
        ],
      ] as Pt[][]
    )
      .map((pts, i) => line(pts, 862 + 2 * i, 1.1))
      .join('')
  // The cushion: a band laid along bed and backrest, its top face where the pet lies.
  const spine = [...densify([foot, hinge], 0.05), ...densify([hinge, head], 0.05).slice(1)]
  const pad = hand(
    ribbon(
      spine.map(([x, y]) => [x, y - 0.07] as const),
      () => 0.1,
    ),
    0.03,
    0.006,
    870,
  )
  const n = spine.length - 1
  const stripes = Array.from({ length: 7 }, (_, k) => {
    const piece = spine.slice(
      Math.round(((2 * k + 0.5) / 14) * n),
      Math.round(((2 * k + 1.5) / 14) * n) + 1,
    )
    return `<path d="${poly(
      ribbon(
        piece.map(([x, y]) => [x, y - 0.07] as const),
        () => 0.095,
      ),
    )}" style="fill:${stripe}"/>`
  }).join('')
  const cushion = `<path d="${poly(pad)}" style="fill:${paper}"/>${stripes}${stroke(pad, pressure(871, w, 0.3), ink)}`
  // A tall iced drink on the ground by its feet, with a straw and a slice of lime.
  const glass = hand(
    [
      [-1.86, 0.98],
      [-1.62, 0.98],
      [-1.65, 1.4],
      [-1.83, 1.4],
      [-1.86, 0.98],
    ],
    0.02,
    0.004,
    872,
  )
  const drink =
    `<path d="${poly(glass)}" style="fill:${wood}"/>${stroke(glass, pressure(873, w * 0.8, 0.2), ink)}` +
    stroke(
      [
        [-1.72, 1.2],
        [-1.68, 0.86],
        [-1.58, 0.8],
      ],
      () => w * 0.6,
      ink,
    ) +
    filled(
      Array.from({ length: 13 }, (_, i) => {
        const a = Math.PI + (i / 12) * Math.PI
        return [-1.86 + 0.11 * Math.cos(a), 0.99 + 0.11 * Math.sin(a)] as const
      }),
      SAGE,
      () => w * 0.6,
      ink,
    )
  return { frame, cushion, drink }
}

// Fog, dusk and night: a little hurricane lantern hanging from the paw it holds out in front (on
// its left, the way it faces, or with `side` 1 on its right). A cap and base, a glass lit warm with wire guards, and a flame
// drawn twice to flicker between; `glow` is three faint rings round the flame, under everything.
export function lantern(
  w: number,
  paper = PAPER,
  ink = INK,
  glass = CRUST,
  fire = PINK,
  side: 1 | -1 = -1,
) {
  const x = 1.22 * side
  const [top, bottom] = [0.42, 0.8]
  const glow = [0.86, 0.62, 0.4]
    .map(
      (r, i) =>
        `<circle cx="${x}" cy="0.6" r="${r}" style="fill:${glass};opacity:${[0.14, 0.18, 0.26][i]}"/>`,
    )
    .join('')
  const line = (pts: readonly Pt[], seed: number, k = 1, color = ink) =>
    stroke(hand(pts, 0.02, 0.004, seed), pressure(seed + 1, w * k, 0.2), color)
  const bail = line(curve([x - 0.13, top - 0.04], [x, -0.04], [x + 0.13, top - 0.04], 14), 830, 0.7)
  const cap = filled(
    hand(
      [
        [x - 0.07, 0.3],
        [x + 0.07, 0.3],
        [x + 0.18, top],
        [x - 0.18, top],
        [x - 0.07, 0.3],
      ],
      0.02,
      0.004,
      832,
    ),
    paper,
    pressure(833, w, 0.2),
    ink,
  )
  const ring = stroke(
    loop(834, 0.1, 0.05, 24).map(([a, b]) => [x + 0.045 * a, 0.25 + 0.045 * b] as const),
    () => w * 0.6,
    ink,
  )
  const globe = hand(rounded(x, (top + bottom) / 2, 0.3, bottom - top, 0.12), 0.02, 0.004, 835)
  const flames = [0, 1].map(k => {
    const h = k ? 0.17 : 0.21
    const lean = k ? 0.025 : -0.015
    const drop = curve([x, 0.7], [x - 0.12, 0.66], [x + lean, 0.7 - h], 10).concat(
      curve([x + lean, 0.7 - h], [x + 0.12, 0.66], [x, 0.7], 10).slice(1),
    )
    return filled(drop, fire, () => w * 0.45, INK)
  })
  const guards = [x - 0.08, x + 0.08]
    .map((gx, i) =>
      line(
        [
          [gx, top + 0.01],
          [gx + (i ? 0.03 : -0.03), (top + bottom) / 2],
          [gx, bottom - 0.01],
        ],
        836 + 2 * i,
        0.45,
        INK,
      ),
    )
    .join('')
  const base = filled(
    hand(rounded(x, bottom + 0.03, 0.36, 0.08, 0.03), 0.02, 0.004, 840),
    paper,
    pressure(841, w, 0.2),
    ink,
  )
  const body = `<path d="${poly(globe)}" style="fill:${glass}"/>`
  const glassLine = stroke(globe, pressure(842, w * 0.8, 0.2), ink)
  return { glow, bail, body, flames, front: `${guards}${glassLine}${cap}${ring}${base}` }
}

// Snow: a pink knit mitten with a sage ribbed cuff over the wrist, on an arm that reaches from
// `from` to `tip`; `side` puts the thumb on one side of it or the other.
export function mitten(
  from: Pt,
  tip: Pt,
  side: 1 | -1,
  w: number,
  ink = INK,
  wool = PINK,
  cuff = SAGE,
  seed = 900,
) {
  const a = Math.atan2(tip[0] - from[0], from[1] - tip[1])
  const [c, s] = [Math.cos(a), Math.sin(a)]
  // Drawn pointing up (-y), then turned to lie along the arm.
  const place = (pts: readonly Pt[]) =>
    pts.map(([x, y]) => [tip[0] + x * c - y * s, tip[1] + x * s + y * c] as const)
  const shape = (pts: Pt[], fill: string, k: number) =>
    filled(place(hand(pts, 0.02, 0.005, seed + k)), fill, pressure(seed + 5 + k, w, 0.2), ink)
  const ribs = [-0.07, -0.024, 0.024, 0.07]
    .map(x =>
      stroke(
        place([
          [x, -0.04],
          [x, 0.03],
        ]),
        () => w * 0.4,
        INK,
      ),
    )
    .join('')
  return (
    shape(rounded(side * 0.12, -0.12, 0.1, 0.14, 0.05), wool, 0) +
    shape(rounded(0, -0.19, 0.22, 0.28, 0.1), wool, 1) +
    shape(rounded(0, -0.005, 0.25, 0.09, 0.03), cuff, 2) +
    ribs
  )
}

// A snowflake: three strokes crossed at its centre, with a fork at each tip.
export function flake(x: number, y: number, r: number, w: number, ink = INK) {
  return [0, 1, 2]
    .map(k => {
      const a = (k * Math.PI) / 3 + 0.3
      const [dx, dy] = [Math.cos(a) * r, Math.sin(a) * r]
      const arm = stroke(
        [
          [x - dx, y - dy],
          [x + dx, y + dy],
        ],
        () => w,
        ink,
      )
      const forks = [1, -1]
        .flatMap(e => [0.5, -0.5].map(t => [e, t] as const))
        .map(([e, t]) => {
          const [bx, by] = [x + e * dx * 0.6, y + e * dy * 0.6]
          const b = a + (e > 0 ? 0 : Math.PI) + t * 1.6
          return stroke(
            [
              [bx, by],
              [bx + Math.cos(b) * r * 0.35, by + Math.sin(b) * r * 0.35],
            ],
            () => w * 0.7,
            ink,
          )
        })
        .join('')
      return arm + forks
    })
    .join('')
}

// Philosophy: hair gone wild after a long night with the questions. Two dozen strands spring off
// the crown, most of them up, the rest out over the ears, each a tapered line that kinks back and
// forth as it goes. `k` picks one of its drawings, so the hair can frizz between them.
export function wildHair(w: number, ink = INK, k = 0) {
  const n = 26
  const noise = (i: number) => {
    const v = Math.sin((i + 1) * 91.7 + k * 13.1) * 43758.5453
    return v - Math.floor(v)
  }
  return Array.from({ length: n }, (_, i) => {
    // Roots from just above one ear over the crown to the other, a little denser on top.
    const u = i / (n - 1)
    const a =
      -Math.PI * (0.04 + 0.92 * (u + 0.08 * Math.sin(2 * Math.PI * u))) + (noise(i) - 0.5) * 0.08
    const root: Pt = [0.9 * Math.cos(a), 0.9 * Math.sin(a)]
    // Longest on top, short over the ears.
    const len = (0.3 + 0.32 * Math.sin(Math.PI * u)) * (0.75 + 0.5 * noise(i + 40))
    const dir = a + (noise(i + 80) - 0.5) * 0.7
    const [tx, ty] = [Math.cos(dir), Math.sin(dir)]
    const waves = 1.5 + 1.5 * noise(i + 120)
    const phase = noise(i + 160) * 2 * Math.PI
    const pts = Array.from({ length: 17 }, (_, j) => {
      const t = j / 16
      const off = 0.07 * t * Math.sin(2 * Math.PI * waves * t + phase)
      return [root[0] + tx * len * t - ty * off, root[1] + ty * len * t + tx * off] as const
    })
    return stroke(hand(pts, 0.03, 0.006, 410 + i + k * 40), t => w * (1.3 - 0.8 * t), ink)
  }).join('')
}

// Posts: a quill from its nib up to its tip, bowed toward `c`, its vane in `fill` with three barbs.
// Returns the drawing and the shaft, for whatever holds it.
export function quill(nib: Pt, c: Pt, tip: Pt, w: number, ink = INK, fill = PINK) {
  const shaft = curve(nib, c, tip, 32)
  const feather = hand(
    ribbon(shaft.slice(10), t => 0.21 * Math.sin(Math.PI * Math.min(1, t * 1.12)) ** 0.6),
    0.04,
    0.014,
    241,
  )
  const barbs = [0.42, 0.58, 0.74]
    .map(t => {
      const [x, y] = shaft[Math.round(t * 32)]
      return stroke(
        densify(
          [
            [x, y],
            [x - 0.16, y - 0.1],
          ],
          0.03,
        ),
        () => w * 0.6,
        ink,
      )
    })
    .join('')
  const svg = `${filled(feather, fill, pressure(242, w, 0.3), ink)}${stroke(
    hand(shaft, 0.04, 0.006, 243),
    t => w * (1.3 - 0.7 * t),
    ink,
  )}${barbs}`
  return { svg, shaft }
}

// The pet's friend on the friendship page: a sprout from the top of its head, a bent stem with a
// leaf to each side, in body radii.
export function sprout(w: number, ink = INK, leaf = SAGE) {
  const stem = hand(curve([-0.12, -0.96], [-0.22, -1.3], [0.02, -1.5]), 0.03, 0.01, 371)
  const blade = (base: Pt, tip: Pt, bend: number, seed: number) => {
    const mid = [(base[0] + tip[0]) / 2, (base[1] + tip[1]) / 2]
    const nx = base[1] - tip[1]
    const ny = tip[0] - base[0]
    const side = (k: number) =>
      curve(base, [mid[0] + nx * bend * k, mid[1] + ny * bend * k], tip, 10)
    const outline = [...side(1), ...side(-1).reverse()]
    return filled(hand(outline, 0.02, 0.006, seed), leaf, pressure(seed + 1, w * 0.9, 0.25), ink)
  }
  return `${stroke(stem, pressure(372, w * 1.1, 0.2), ink)}${blade([0.0, -1.42], [0.34, -1.62], 0.32, 373)}${blade([-0.08, -1.28], [-0.4, -1.42], 0.32, 375)}`
}

// A heart in pen, about 1 across, centred on (0, 0): it rises between the two pets as they hug.
export function heart(w: number, ink = INK, fill = PINK) {
  const pts = Array.from({ length: 49 }, (_, i) => {
    const t = (i / 48) * 2 * Math.PI
    return [
      (16 * Math.sin(t) ** 3) / 34,
      -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 34,
    ] as const
  })
  return filled(hand(pts, 0.03, 0.01, 381), fill, pressure(382, w, 0.3), ink)
}

// The pet: the mascot with every look the old rocky spritesheet had, each pre-drawn as a part. The
// SVG's data-state picks which parts show and how they move (quartz/components/styles/pet.scss),
// so changing a look is one attribute write. Colours are theme variables; the pet faces left as
// drawn and is mirrored to face right. `seed` picks the outline's wobble, so a friend is its own circle.
export function petSvg(paper = 'var(--light)', ink = 'var(--dark)', pink = PINK, seed = 7): string {
  const L = 0.08
  const outline = loop(seed, 0.06, 0.03, 140)
  const pen = (pts: readonly Pt[], w: (t: number) => number, color = ink) => stroke(pts, w, color)

  const leg = (hip: Pt, foot: Pt, toe: number, seed: number) =>
    `<g class="pet-leg">${pen(densify([hip, foot, [foot[0] + toe, foot[1] + 0.02]], 0.05), pressure(seed, L * 1.15, 0.2))}</g>`
  const legs = part(
    'legs',
    leg([-0.34, 0.8], [-0.38, 1.36], -0.17, 41) + leg([0.3, 0.84], [0.33, 1.38], -0.16, 43),
    'pet-legs',
  )

  // In the snow every hand it shows wears a mitten (pet.scss shows them with data-snow).
  const mitt = (pts: readonly Pt[], side: 1 | -1, seed: number) =>
    `<g class="pet-mitten">${mitten(pts[pts.length - 2], pts[pts.length - 1], side, L * 0.76, ink, pink, SAGE, seed)}</g>`
  const arm = (side: 'l' | 'r', pts: Pt[], seed: number) =>
    part(
      `arm-${side}`,
      pen(densify(pts, 0.05), pressure(seed, L * 1.1, 0.2)) +
        mitt(pts, side === 'l' ? 1 : -1, 900 + seed),
      'pet-arm',
    )
  const arms =
    arm(
      'l',
      [
        [-0.9, -0.08],
        [-1.2, -0.42],
        [-1.3, -0.82],
      ],
      45,
    ) +
    arm(
      'r',
      [
        [0.92, -0.08],
        [1.22, -0.44],
        [1.3, -0.84],
      ],
      47,
    )
  // Both hands up on a ledge the pet hangs from; the ledge runs through y = -1.45.
  const grip = part(
    'arms-hang',
    (
      [
        [
          [-0.62, -0.78],
          [-0.68, -1.12],
          [-0.56, -1.45],
        ],
        [
          [0.5, -0.86],
          [0.57, -1.16],
          [0.44, -1.45],
        ],
      ] as Pt[][]
    )
      .map(
        (pts, i) =>
          pen(densify(pts, 0.05), pressure(48 + i, L * 1.1, 0.2)) +
          mitt(pts, i ? -1 : 1, 960 + 10 * i),
      )
      .join(''),
  )

  // Clinging to an upright pole on its right (the scroll rail), one hand above the other; the pole
  // runs through x = 1.32.
  const pole = part(
    'arms-pole',
    pen(
      hand(
        [
          [1.32, -1.9],
          [1.33, 1.6],
        ],
        0.02,
        0.01,
        52,
      ),
      pressure(53, L * 0.9, 0.2),
    ) +
      (
        [
          [
            [0.7, -0.62],
            [1.04, -0.96],
            [1.32, -1.06],
          ],
          [
            [0.96, -0.02],
            [1.16, -0.38],
            [1.32, -0.5],
          ],
        ] as Pt[][]
      )
        .map(
          (pts, i) =>
            pen(densify(pts, 0.05), pressure(54 + i, L * 1.1, 0.2)) + mitt(pts, 1, 980 + 10 * i),
        )
        .join(''),
  )

  const body = `<path d="${poly(outline)}" style="fill:${paper}"/>${pen(outline, pressure(8, L))}`

  const dash = part(
    'eyes-dash',
    EYES.map(x => `<g class="pet-eye">${eye(x, L * 1.7, ink)}</g>`).join(''),
  )
  const happy = part(
    'eyes-happy',
    EYES.map(x =>
      pen(curve([x - 0.13, -0.1], [x + 0.02, -0.5], [x + 0.17, -0.1]), even(L * 1.35)),
    ).join(''),
  )
  const wide = part(
    'eyes-wide',
    EYES.map((x, i) => pen(ring(21 + i, x + 0.02, -0.2, 0.13), pressure(23 + i, L, 0.25))).join(''),
    'pet-blinkless',
  )
  const wince = part(
    'eyes-wince',
    pen(
      densify(
        [
          [-0.72, -0.34],
          [-0.52, -0.2],
          [-0.72, -0.06],
        ],
        0.03,
      ),
      even(L * 1.3),
    ) +
      pen(
        densify(
          [
            [0.38, -0.34],
            [0.18, -0.2],
            [0.38, -0.06],
          ],
          0.03,
        ),
        even(L * 1.3),
      ),
    'pet-blinkless',
  )

  const mx = (EYES[0] + EYES[1]) / 2
  const mouthO = part('mouth-o', pen(ring(31, mx, 0.22, 0.09), pressure(33, L * 0.95, 0.25)))
  const smile = part(
    'mouth-smile',
    pen(curve([mx - 0.15, 0.12], [mx, 0.3], [mx + 0.15, 0.12]), even(L * 1.1)),
  )
  const grinPts = [
    ...densify(
      [
        [mx - 0.24, 0.1],
        [mx + 0.24, 0.1],
      ],
      0.04,
    ),
    ...curve([mx + 0.24, 0.1], [mx, 0.58], [mx - 0.24, 0.1]).slice(1),
  ]
  const grin = part(
    'mouth-grin',
    `<path d="${poly(grinPts)}" style="fill:${pink}"/>${pen(grinPts, pressure(35, L, 0.2))}`,
  )
  const blush = part(
    'blush',
    [EYES[0] - 0.18, EYES[1] + 0.2]
      .map(
        x => `<ellipse cx="${x.toFixed(3)}" cy="0.07" rx="0.15" ry="0.075" style="fill:${pink}"/>`,
      )
      .join(''),
  )

  const sweat = part(
    'sweat',
    (
      [
        [
          [0.8, -0.98],
          [0.9, -1.22],
        ],
        [
          [1.0, -0.74],
          [1.2, -0.92],
        ],
        [
          [1.13, -0.44],
          [1.38, -0.52],
        ],
      ] as Pt[][]
    )
      .map(([a, b]) => pen(densify([a, b], 0.04), even(L)))
      .join(''),
  )
  const dots = part(
    'dots',
    (
      [
        [0.55, -1.3],
        [0.85, -1.42],
        [1.15, -1.54],
      ] as Pt[]
    )
      .map(
        ([x, y]) => `<circle class="pet-dot" cx="${x}" cy="${y}" r="0.075" style="fill:${ink}"/>`,
      )
      .join(''),
  )
  // A thought: a paper cloud up to the right, reached by two bubbles, with the three dots inside.
  const cloud = ring(57, 1.3, -2.2, 0.46).map(([x, y]) => [1.3 + (x - 1.3) * 1.5, y] as Pt)
  const thought = part(
    'thought',
    `<path d="${poly(cloud)}" style="fill:${paper}"/>${pen(cloud, pressure(58, L * 0.9, 0.25))}` +
      (
        [
          [0.66, -1.24, 0.07],
          [0.88, -1.54, 0.11],
        ] as const
      )
        .map(([x, y, r], i) => pen(ring(59 + i, x, y, r), even(L * 0.8)))
        .join('') +
      [0.95, 1.3, 1.65]
        .map(x => `<circle class="pet-dot" cx="${x}" cy="-2.2" r="0.08" style="fill:${ink}"/>`)
        .join(''),
  )
  // A parachute for coming down off the scroll rail, drawn as a kid would sew one: a paper canopy
  // leaning off true over uneven scallops, a hatched patch, a knob on top, and four lines to the
  // raised hands, the middle two crossed. It is drawn three times with different wobbles; pet.scss
  // flicks between them so the line boils, and puffs the canopy as it fills.
  const hem: Pt[] = (
    [
      [-1.78, -2.92],
      [-0.66, -3.06],
      [0.38, -2.98],
      [1.6, -2.8],
    ] as const
  ).map(([x, y]) => [x, y])
  const hands: Pt[] = [
    [-0.56, -1.45],
    [0.44, -1.45],
  ]
  const chuteFrame = (k: number) => {
    const seed = 140 + k * 11
    // The dome is two halves meeting at a crown off centre, so the crown is on the canopy line and
    // the gores and knob can end on it. Each drawing nudges the crown so the silhouette boils too.
    const crown: Pt = [0.12 + 0.05 * (k - 1), -4.6 + 0.04 * ((k * 2) % 3)]
    const left = curve(hem[0], [-1.72, crown[1] - 0.02], crown, 11)
    const right = curve(crown, [1.62, crown[1] + 0.08], hem[3], 11)
    const dome = [...left, ...right.slice(1)]
    const scallops = [3, 2, 1].flatMap(i =>
      curve(
        hem[i],
        [(hem[i][0] + hem[i - 1][0]) / 2, Math.max(hem[i][1], hem[i - 1][1]) + 0.2 + 0.07 * i],
        hem[i - 1],
        7,
      ).slice(1),
    )
    const canopy = hand([...dome, ...scallops], 0.08, 0.085, seed)
    const gores = [left[8], right[3]]
      .map((top, i) =>
        pen(
          hand(
            curve(
              hem[i + 1],
              [hem[i + 1][0] * 0.82 + top[0] * 0.18 + (i ? 0.16 : -0.16), -3.8],
              top,
              9,
            ),
            0.08,
            0.05,
            seed + i,
          ),
          even(L * 0.65),
        ),
      )
      .join('')
    const patch = hand(rounded(-0.98, -3.72, 0.44, 0.36, 0.07), 0.04, 0.012, seed + 3)
    const hatch = [-0.1, 0.04, 0.18]
      .map(d =>
        pen(
          densify(
            [
              [-1.12 + d, -3.6],
              [-0.98 + d, -3.84],
            ],
            0.05,
          ),
          even(L * 0.45),
        ),
      )
      .join('')
    const knob = ring(seed + 4, crown[0] + 0.02, crown[1] - 0.15, 0.13)
    // Outer lines to their own hands; the middle two cross on the way down.
    const lines = (
      [
        [hem[0], hands[0], -0.08],
        [hem[1], hands[1], 0.06],
        [hem[2], hands[0], -0.06],
        [hem[3], hands[1], 0.1],
      ] as const
    )
      .map(([a, b, bow], i) =>
        pen(
          hand(
            curve(a, [(a[0] + b[0]) / 2 + bow, (a[1] + b[1]) / 2], b, 10),
            0.08,
            0.04,
            seed + 5 + i,
          ),
          even(L * 0.55),
        ),
      )
      .join('')
    return `<g class="pet-canopy-frame"><path d="${poly(canopy)}" style="fill:${paper}"/>${pen(canopy, pressure(seed + 9, L, 0.3))}${gores}<path d="${poly(patch)}" style="fill:${paper}"/>${pen(patch, even(L * 0.55))}${hatch}<path d="${poly(knob)}" style="fill:${paper}"/>${pen(knob, even(L * 0.6))}${lines}</g>`
  }
  // Worn for walks on the home page's trail, as on the brand card; its pen widths match the pet's
  // line, which is 3.2 hat units on the card.
  const hat = part('hat', bucketHat(L / 3.2, paper, ink, pink), 'pet-hat')
  // Section outfits (quartz/util/section-icon.ts), as on the site icons: the sweatband on
  // triathlon pages, the beret and baguettes on French ones, a quill tucked behind the head on
  // posts, the wig, tie and briefcase on the terms and privacy pages, the palette and brush on
  // quotes and images, glasses and books on bases, a bow tie and pointer on slides, a sheet to read
  // on the stream, a mortarboard, notepad and pencil on courses. `-back` parts sit behind the body.
  const w = (L * 0.042) / 0.055
  const outfit = {
    front: [
      part('outfit-triathlon', sweatband(w, ink, pink)),
      part('outfit-fr', `${baguette([-0.62, 1.2], [1.4, 0.3], 227, w, ink)}${beret(w, ink, ink)}`),
      part('outfit-legal', Object.values(lawyer(w, paper, ink, pink)).join('')),
      part(
        'outfit-art',
        (({ palette, smudge }) => `${smudge}${palette}`)(artist(w, paper, ink, pink)),
      ),
      part('outfit-library', Object.values(librarian(w, paper, ink)).join('')),
      part(
        'outfit-slides',
        Object.values(presenter(w, paper, ink, pink))
          .filter((_, i) => i < 2)
          .join(''),
      ),
      // The glasses are the shared `glasses` part below.
      part('outfit-stream', reader(w, paper, ink, pink).sheet),
      part(
        'outfit-courses',
        (({ cap, pad, pencil }) => `${cap}${pad}<g class="pet-pencil">${pencil}</g>`)(
          student(w, paper, ink, pink),
        ),
      ),
      part(
        'outfit-cinema',
        (({ specs, popcorn, kernel }) => `${specs}${popcorn}<g class="pet-kernel">${kernel}</g>`)(
          cinephile(w, paper, ink, pink),
        ),
      ),
      part(
        'outfit-engineering',
        (({ hardhat, flask, bubbles }) => `${hardhat}${flask}${bubbles}`)(
          engineer(w, paper, ink, pink),
        ),
      ),
    ].join(''),
    back: [
      part('outfit-fr-back', baguette([0.3, 1.02], [1.46, -0.2], 223, w, ink)),
      part('outfit-posts-back', quill([0.42, 0.25], [0.95, -0.5], [1.32, -1.6], w, ink, pink).svg),
      part('outfit-art-back', artist(w, paper, ink, pink).brush),
    ].join(''),
  }
  // Reading glasses for reader mode, the stream and technical pages: the librarian's pair.
  const glasses = part('glasses', librarian(w, paper, ink).glasses)
  // Wild hair on philosophy pages, drawn twice to frizz; behind the body, so the roots tuck in.
  const hair = part(
    'hair',
    [0, 1].map(k => `<g class="pet-hair-frame">${wildHair(w * 1.1, ink, k)}</g>`).join(''),
  )
  // Wrapped up against the visitor's cold (pet.inline.ts): the scarf, a toque and breath that shows.
  const cold = winter(w, paper, ink, SAGE, pink)
  const scarf = part('scarf', cold.scarf)
  const toque = part('toque', cold.toque)
  const breath = part('breath', cold.puffs)
  // On triathlon pages it goes out in its wind vest instead, the back panel flapping.
  const gilet = windVest(outline, w, ink, SAGE, pink)
  // In the snow, at rest, it holds both mittens up to catch the flakes falling round it; one comes
  // down into the front mitten and melts there.
  const reach: Pt[][] = [
    [
      [-0.9, -0.08],
      [-1.22, -0.3],
      [-1.36, -0.62],
    ],
    [
      [0.92, -0.08],
      [1.24, -0.32],
      [1.36, -0.64],
    ],
  ]
  const caught = part(
    'catch',
    reach
      .map(
        (pts, i) =>
          `<g class="pet-catch-arm">${pen(densify(pts, 0.05), pressure(61 + i, L * 1.1, 0.2))}<g class="pet-catch-mitt">${mitten(pts[1], pts[2], i ? -1 : 1, L * 0.76, ink, pink, SAGE, 940 + 10 * i)}</g></g>`,
      )
      .join('') + `<g class="pet-catch">${flake(-1.5, -0.98, 0.09, L * 0.45, ink)}</g>`,
  )
  // Falling where it stands: each flake drifts down its own distance with a sway.
  const snowfall = part(
    'snowfall',
    (
      [
        [-1.62, -1.9, 1.5, 0.12],
        [-1.0, -2.15, 1.0, -0.1],
        [-0.25, -2.25, 0.7, 0.08],
        [0.55, -2.1, 0.6, -0.08],
        [1.12, -1.95, 0.9, 0.1],
        [1.66, -1.7, 1.6, -0.12],
      ] as const
    )
      .map(
        ([x, y, fall, sway], i) =>
          `<g class="pet-flake" style="--fall:${fall}px;--sway:${sway}px;animation-delay:${(-0.7 * i).toFixed(1)}s">${flake(x, y, 0.07 + 0.015 * (i % 2), L * 0.4, ink)}</g>`,
      )
      .join(''),
  )
  // In fog, at dusk and at night it holds out its little lantern on the side it faces, the flame
  // flickering between two drawings and its light falling faintly on the body. Outfits with a prop
  // in that paw (pet.scss) hold it out behind instead.
  const lamp = (side: 1 | -1, name: string, seed: number) => {
    const l = lantern(w, paper, ink, CRUST, pink, side)
    const reach: Pt[] = [
      [0.9 * side, -0.02],
      [1.08 * side, 0.02],
      [1.22 * side, 0.1],
    ]
    return part(
      name,
      `<g class="pet-glow">${l.glow}</g>${l.bail}${l.body}` +
        l.flames.map(f => `<g class="pet-flame-frame">${f}</g>`).join('') +
        l.front +
        pen(densify(reach, 0.05), pressure(seed, L * 1.1, 0.2)) +
        filled(
          hand(rounded(1.22 * side, 0.1, 0.2, 0.17, 0.08), 0.02, 0.006, seed + 1),
          paper,
          pressure(seed + 2, w, 0.2),
          ink,
        ) +
        mitt(reach, side === -1 ? 1 : -1, 990 + seed),
    )
  }
  const lit = lamp(-1, 'lantern', 65) + lamp(1, 'lantern-back', 68)
  // In the wind it holds its hat on with the hand on its right, reaching up over its head; hats
  // sit at different heights (pet.scss picks the reach), and gusts stream past in two drawings.
  const holdHat = (name: string, y: number, seed: number) => {
    const reach: Pt[] = [
      [0.92, -0.12],
      ...curve([1.26, -0.56], [1.3, y - 0.1], [0.62, y - 0.02], 10),
    ]
    return part(
      name,
      pen(densify(reach, 0.05), pressure(seed, L * 1.1, 0.2)) +
        filled(
          hand(rounded(0.46, y, 0.26, 0.18, 0.08), 0.02, 0.006, seed + 1),
          paper,
          pressure(seed + 2, w, 0.2),
          ink,
        ) +
        `<g class="pet-mitten">${mitten([0.72, y - 0.04], [0.46, y], -1, L * 0.76, ink, pink, SAGE, 995 + seed)}</g>`,
    )
  }
  const holds =
    holdHat('hold-high', -1.42, 74) +
    holdHat('hold-mid', -1.24, 77) +
    holdHat('hold-low', -1.08, 80)
  const gust = part(
    'gust',
    [0, 1]
      .map(k => {
        const lines = (
          [
            [-1.78, -1.2, 0.5],
            [-1.72, -0.5, 0.38],
            [1.38, -1.08, 0.42],
          ] as const
        )
          .map(([x, y, len], i) => {
            const x0 = x + 0.12 * k
            return stroke(
              hand(
                curve([x0, y], [x0 + len * 0.5, y - 0.1 + 0.06 * k], [x0 + len, y + 0.02], 10),
                0.03,
                0.006,
                1000 + 3 * k + i,
              ),
              even(w * 0.6),
              ink,
            )
          })
          .join('')
        return `<g class="pet-gust-frame">${lines}</g>`
      })
      .join(''),
  )
  // In the heat: sunglasses, a tan that deepens while it lies out (pet.inline.ts sets how deep),
  // and on sunny afternoons a sun lounger to lie back on, legs out along the cushion.
  const shades = part('sunglasses', sunglasses(w, ink, INK, pink))
  const tan = part(
    'tan',
    `<path d="${poly(outline.map(([x, y]) => [x * 0.97, y * 0.97] as const))}" style="fill:${CRUST}"/>`,
    'pet-tan',
  )
  const bed = lounger(w, paper, ink, pink, CRUST)
  const lounge = part(
    'lounger',
    bed.frame +
      bed.cushion +
      bed.drink +
      (
        [
          [
            [-0.5, 0.72],
            [-1.3, 0.8],
            [-1.42, 0.66],
          ],
          [
            [-0.2, 0.86],
            [-1.1, 0.84],
            [-1.2, 0.7],
          ],
        ] as Pt[][]
      )
        .map((pts, i) => pen(densify(pts, 0.05), pressure(70 + i, L * 1.15, 0.2)))
        .join(''),
  )
  // In the rain it holds up its little umbrella, the rain falling past in two drawings.
  const brolly = umbrella(w, paper, ink, pink, SAGE)
  const shelter = part(
    'umbrella',
    `${brolly.rain.map(r => `<g class="pet-rain-frame">${r}</g>`).join('')}${brolly.held}`,
  )
  const vest = part(
    'vest',
    `${gilet.flaps.map(f => `<g class="pet-flap-frame">${f}</g>`).join('')}${gilet.vest}`,
  )
  const chute = part('chute', `<g class="pet-canopy">${[0, 1, 2].map(chuteFrame).join('')}</g>`)
  // A jetpack for flying up to the scroll rail: two fat tanks on its back (behind the body, on the
  // right as drawn, since it faces left), a strap round the front so it reads as worn, and below
  // each nozzle a flame of three tongues with exhaust puffs, drawn twice so they flicker.
  const tanks = [
    [1.0, -0.06, 0.4, 1.02, 251],
    [1.38, 0.02, 0.34, 0.9, 253],
  ] as const
  const pack = tanks
    .map(([x, y, tw, th, seed]) => {
      const tank = hand(rounded(x, y, tw, th, tw / 2), 0.04, 0.01, seed)
      const band = (k: number, s: number) =>
        hand(
          [
            [x - tw / 2 + 0.03, y + th * k],
            [x + tw / 2 - 0.03, y + th * k],
          ],
          0.03,
          0.006,
          s,
        )
      const bottom = y + th / 2
      const nozzle: Pt[] = [
        [x - 0.08, bottom - 0.02],
        [x + 0.08, bottom - 0.02],
        [x + 0.12, bottom + 0.11],
        [x - 0.12, bottom + 0.11],
        [x - 0.08, bottom - 0.02],
      ]
      return `${filled(tank, paper, pressure(seed + 2, L * 0.95, 0.25), ink)}${pen(band(-0.2, seed + 1), () => L * 1.7, pink)}${pen(band(0.16, seed + 3), even(L * 0.6))}${filled(densify(nozzle, 0.04), ink, () => L * 0.6, ink)}`
    })
    .join('')
  const strap = part(
    'jetpack',
    pen(
      hand(curve([0.9, -0.5], [0.52, 0.02], [0.84, 0.6]), 0.03, 0.01, 257),
      pressure(258, L * 0.85, 0.3),
    ),
  )
  const tongue = (x: number, top: number, len: number, lean: number, w: number, seed: number) =>
    hand(
      ribbon(
        densify(
          [
            [x, top],
            [x + lean, top + len],
          ],
          0.03,
        ),
        t => w * (1 - t) ** 0.9 * Math.min(1, 0.5 + t * 6),
      ),
      0.03,
      0.012,
      seed,
    )
  const flame = (x: number, top: number, len: number, k: number, seed: number) => {
    const sides = [-1, 1]
      .map(side =>
        filled(
          tongue(
            x + side * 0.05,
            top,
            len * (0.5 + 0.12 * ((k + side + 2) % 2)),
            side * 0.12,
            0.07,
            seed + side,
          ),
          pink,
          () => L * 0.45,
          ink,
        ),
      )
      .join('')
    const middle = tongue(x, top, len, k ? 0.04 : -0.04, 0.1, seed + 3)
    const core = tongue(x, top, len * 0.5, 0, 0.05, seed + 4)
    // Puffs drift off below the tip, at other spots in each drawing.
    const puffs = [0, 1]
      .map(j => {
        const r = 0.07 + 0.04 * ((j + k) % 2)
        const ring_ = ring(
          seed + 6 + j,
          x + (j ? 0.1 : -0.08) * (k ? -1 : 1),
          top + len + 0.08 + j * 0.16,
          r,
        )
        return filled(ring_, paper, () => L * 0.5, ink)
      })
      .join('')
    return `${sides}${filled(middle, pink, () => L * 0.5, ink)}<path d="${poly(core)}" style="fill:${CRUST}"/>${puffs}`
  }
  const flames = [0, 1]
    .map(
      k =>
        `<g class="pet-flame">${tanks
          .map(([x, y, , th], i) =>
            flame(
              x,
              y + th / 2 + 0.12,
              k ? 0.42 + 0.06 * i : 0.56 - 0.08 * i,
              k,
              260 + k * 10 + i * 20,
            ),
          )
          .join('')}</g>`,
    )
    .join('')
  const jetpack = part('jetpack', `${flames}${pack}`)
  const speed = part(
    'speed',
    (
      [
        [1.24, -0.28, 1.62],
        [1.32, 0.04, 1.76],
        [1.22, 0.36, 1.56],
      ] as const
    )
      .map(([a, y, b]) =>
        pen(
          densify(
            [
              [a, y],
              [b, y + 0.02],
            ],
            0.04,
          ),
          even(L * 0.9),
        ),
      )
      .join(''),
    'pet-speed',
  )
  const screen = hand(rounded(-0.18, 0.4, 1.3, 0.76, 0.07), 0.04, 0.012, 51)
  const base = hand(rounded(-0.18, 0.84, 1.66, 0.12, 0.05), 0.04, 0.008, 53)
  const laptop = part(
    'laptop',
    `<path d="${poly(screen)}" style="fill:${paper}"/>${pen(screen, pressure(52, L * 0.95, 0.25))}` +
      `<circle cx="-0.18" cy="0.4" r="0.075" style="fill:${pink}"/>` +
      `<path d="${poly(base)}" style="fill:${paper}"/>${pen(base, pressure(54, L * 0.9, 0.2))}` +
      (
        [
          [-1.08, -0.07],
          [0.76, 0.07],
        ] as const
      )
        .map(
          ([x, dx]) =>
            `<g class="pet-key">${pen(
              densify(
                [
                  [x, 0.7],
                  [x + dx, 0.54],
                ],
                0.03,
              ),
              even(L * 0.8),
            )}</g>`,
        )
        .join(''),
  )

  // A short line of its own to stand on, for when no box nearby has an edge.
  const ground = part(
    'ground',
    pen(
      hand(
        [
          [-1.2, 1.43],
          [1.2, 1.45],
        ],
        0.05,
        0.015,
        57,
      ),
      pressure(58, L * 0.8, 0.3),
    ),
  )

  return `<svg class="pet-svg" viewBox="-1.8 -2 3.6 3.8" aria-hidden="true" focusable="false" data-state="idle"><g class="pet-flip">${ground}${snowfall}<g class="pet-hop"><g class="pet-lean">${chute}${legs}${lounge}${speed}<g class="pet-body">${jetpack}${hair}${outfit.back}${body}${tan}${vest}${strap}<g class="pet-look">${dash}${happy}${wide}${wince}${mouthO}${smile}${grin}${blush}</g>${scarf}${outfit.front}${glasses}${shades}${toque}${hat}${holds}${shelter}${breath}${arms}${caught}${lit}${grip}${pole}${laptop}</g>${sweat}${dots}${gust}${thought}</g></g></g></svg>`
}
