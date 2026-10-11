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

// The pet: the mascot with every look the old rocky spritesheet had, each pre-drawn as a part. The
// SVG's data-state picks which parts show and how they move (quartz/components/styles/pet.scss),
// so changing a look is one attribute write. Colours are theme variables; the pet faces left as
// drawn and is mirrored to face right.
export function petSvg(paper = 'var(--light)', ink = 'var(--dark)', pink = PINK): string {
  const L = 0.08
  const outline = loop(7, 0.06, 0.03, 140)
  const pen = (pts: readonly Pt[], w: (t: number) => number, color = ink) => stroke(pts, w, color)

  const leg = (hip: Pt, foot: Pt, toe: number, seed: number) =>
    `<g class="pet-leg">${pen(densify([hip, foot, [foot[0] + toe, foot[1] + 0.02]], 0.05), pressure(seed, L * 1.15, 0.2))}</g>`
  const legs = part(
    'legs',
    leg([-0.34, 0.8], [-0.38, 1.36], -0.17, 41) + leg([0.3, 0.84], [0.33, 1.38], -0.16, 43),
    'pet-legs',
  )

  const arm = (side: 'l' | 'r', pts: Pt[], seed: number) =>
    part(`arm-${side}`, pen(densify(pts, 0.05), pressure(seed, L * 1.1, 0.2)), 'pet-arm')
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
      .map((pts, i) => pen(densify(pts, 0.05), pressure(48 + i, L * 1.1, 0.2)))
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

  return `<svg class="pet-svg" viewBox="-1.8 -2 3.6 3.8" aria-hidden="true" focusable="false" data-state="idle"><g class="pet-flip"><g class="pet-hop"><g class="pet-lean">${legs}${speed}<g class="pet-body">${body}<g class="pet-look">${dash}${happy}${wide}${wince}${mouthO}${smile}${grin}${blush}</g>${arms}${grip}${laptop}</g>${sweat}${dots}</g></g></g></svg>`
}
