// Draws the garden's mascot into the site icon and the default social card: a hand-drawn circle
// with two short upright eyes shifted off to the left, traced from aarnphm's Freeform sketch. Each
// line is a pen stroke rather than an SVG stroke: a filled ribbon around a wobbly centre line whose
// width swells and thins as pen pressure would, tapering where the pen lands and lifts. On the card
// the mascot walks over a small planet under a pink sun drawn the same way. Seeds are fixed, so
// each run draws the same wobble.
//
//   pnpm exec tsx quartz/scripts/brand-art.ts              writes the static assets
//   pnpm exec tsx quartz/scripts/brand-art.ts --preview D  writes previews (and favicon sizes) to D

import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { mulberry } from '../components/scripts/404-pixel'

const ROOT = path.resolve(import.meta.dirname, '../..')

const PAPER = '#fffcf0'
const INK = '#1c1b1a'
const PINK = '#fdb2a2'
const SAGE = '#cdd597'

type Pt = readonly [number, number]

const poly = (pts: readonly Pt[]) =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(3)}`).join('') + 'Z'

// A few slow sines with seeded phases, summing to roughly [-1, 1].
function drift(seed: number, ks: readonly number[]) {
  const rand = mulberry(seed)
  const waves = ks.map(k => ({ k, phase: rand() * 2 * Math.PI, a: 1 / Math.sqrt(k) }))
  const norm = waves.reduce((s, w) => s + w.a, 0)
  return (t: number) => waves.reduce((s, w) => s + w.a * Math.sin(w.k * t + w.phase), 0) / norm
}

// A pen stroke along `pts`: the centre line offset both ways by half of `width(t)`, closed with a
// round end at each tip so the pen's landing and lift read as dots of ink.
function stroke(pts: readonly Pt[], width: (t: number) => number) {
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
  return `<g fill="${INK}"><path d="${poly([...left, ...right.reverse()])}"/>${tip(pts[0], width(0) / 2)}${tip(pts[n], width(1) / 2)}</g>`
}

// Pressure along a stroke: thin where the pen lands and lifts, swelling in a few places between.
const pressure = (seed: number, base: number, swell = 0.45) => {
  const wave = drift(seed, [3, 5, 8])
  return (t: number) =>
    base * (0.45 + 0.55 * Math.sin(Math.PI * t) ** 0.35) * (1 + swell * wave(t * 6))
}

// A circle drawn by hand, radius 1 around the origin: the radius drifts, the circle leans a little
// egg-ward, and the pen runs past where it started so the line overlaps itself near the top.
function loop(seed: number, wobble = 0.06, overrun = 0.03): Pt[] {
  const r = drift(seed, [2, 3, 5, 7])
  const start = -2.1
  const n = 220
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const a = start + t * (2 * Math.PI + 0.3)
    const out = 1 + wobble * r(a) + overrun * Math.max(0, t - 0.9) * 10
    return [out * Math.cos(a) * 1.02, out * Math.sin(a) * (0.98 + 0.03 * Math.sin(a))] as const
  })
}

// An upright eye: a short stroke that bulges in the middle like a pressed pen.
function eye(x: number, base: number): string {
  const pts: Pt[] = Array.from({ length: 13 }, (_, i) => {
    const t = i / 12
    return [x + 0.025 * t + 0.012 * Math.sin(Math.PI * t), -0.36 + 0.32 * t] as const
  })
  return stroke(pts, t => base * (0.7 + 0.5 * Math.sin(Math.PI * t)))
}

// The mascot in its own units, radius about 1 around the origin, filled with paper inside the line
// so it reads on a dark tab bar.
function mascot(line: number, seed: number) {
  const outline = loop(seed)
  return `<path d="${poly(outline)}" fill="${PAPER}"/>
  ${stroke(outline, pressure(seed + 1, line))}
  ${eye(-0.6, line * 1.7)}${eye(0.24, line * 1.7)}`
}

const svg = (w: number, h: number, body: string) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${body}</svg>`,
  )

const icon = () =>
  svg(1024, 1024, `<g transform="translate(512 512) scale(410)">${mascot(0.055, 7)}</g>`)

// Points along a polyline, about `step` apart, so a hand wobble has somewhere to live.
function densify(pts: readonly Pt[], step: number): Pt[] {
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
function hand(pts: readonly Pt[], step: number, amp: number, seed: number): Pt[] {
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
function rounded(cx: number, cy: number, w: number, h: number, r: number): Pt[] {
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

// A closed shape in one colour inside a pen line.
const shape = (pts: readonly Pt[], fill: string, width: (t: number) => number) =>
  `<path d="${poly(pts)}" fill="${fill}"/>${stroke(pts, width)}`

// The card: the mascot climbs the long right shoulder of a peak, a pink pack and a sage bedroll on
// its back and a bucket hat on its head, toward a flag on the summit. Its
// eyes already look left, so the slope rises to the left. A sage range stands behind with the pink
// sun coming up between its peaks. The whole scene sits 80 lower than drawn, for sky.
function card() {
  const W = 1200
  const H = 630
  const near: Pt[] = [
    [-20, 520],
    [110, 430],
    [200, 455],
    [300, 250],
    [380, 140],
    [520, 250],
    [640, 318],
    [770, 392],
    [860, 418],
    [970, 318],
    [1060, 352],
    [1130, 330],
    [W + 20, 362],
  ]
  const far: Pt[] = [
    [-20, 330],
    [90, 262],
    [200, 318],
    [470, 300],
    [600, 262],
    [720, 190],
    [810, 246],
    [930, 150],
    [1060, 240],
    [W + 20, 214],
  ]
  // The near ridge's height at x, before the hand wobble, for setting feet on it.
  const ridge = (x: number) => {
    for (let i = 0; i < near.length - 1; i++) {
      const [ax, ay] = near[i]
      const [bx, by] = near[i + 1]
      if (x >= ax && x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
    }
    return H
  }
  const range = (pts: Pt[], seed: number, fill: string, base: number) => {
    const line = hand(pts, 6, 3, seed)
    return `<path d="${poly([...line, [W + 20, H + 20], [-20, H + 20]])}" fill="${fill}"/>${stroke(line, pressure(seed + 1, base, 0.5))}`
  }
  const sun = loop(3, 0.07).map(([x, y]) => [845 + x * 62, 200 + y * 62] as const)
  const thin = (seed: number, base = 2.6) => pressure(seed, base, 0.35)

  const snow = hand(
    [
      [334, 192],
      [356, 214],
      [374, 196],
      [396, 224],
      [418, 198],
      [446, 214],
      [454, 196],
    ],
    4,
    1.2,
    43,
  )

  // The climber: body frame in mascot units, leaning into the hill.
  const r = 48
  const tilt = (-16 * Math.PI) / 180
  const bx = 655
  const by = ridge(655) - r - 40
  const world = ([u, v]: Pt): Pt => [
    bx + r * (u * Math.cos(tilt) - v * Math.sin(tilt)),
    by + r * (u * Math.sin(tilt) + v * Math.cos(tilt)),
  ]
  // Legs in world space so the feet land on the slope: the front one planted uphill, the back one
  // pushing off with its heel up.
  const leg = (hip: Pt, foot: number, lift: number, toe: number, seed: number) => {
    const h = world(hip)
    const f: Pt = [foot, ridge(foot) - lift]
    const knee: Pt = [(h[0] + f[0]) / 2 - 6, (h[1] + f[1]) / 2]
    return stroke(densify([h, knee, f, [f[0] + toe, f[1] + 1]], 3), thin(seed, 2.8))
  }
  const front = world([-0.3, 0.9])
  const back = world([0.32, 0.9])
  const legs =
    leg([-0.3, 0.9], front[0] - 10, 0, -7, 51) + leg([0.32, 0.9], back[0] + 12, 2, -6, 52)

  const u = 1 / r
  const pack = shape(
    hand(rounded(1.14, 0.06, 0.66, 1.06, 0.24), 0.05, 0.02, 61),
    PINK,
    pressure(62, 2.8 * u, 0.35),
  )
  const roll = shape(
    hand(rounded(1.14, -0.58, 0.74, 0.32, 0.16), 0.05, 0.015, 63),
    PAPER,
    pressure(64, 2.6 * u, 0.35),
  )
  const crown = shape(
    hand(
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
    ),
    PAPER,
    pressure(68, 2.6 * u, 0.35),
  )
  const band = stroke(
    hand(
      [
        [-0.58, -0.9],
        [0.62, -0.9],
      ],
      0.04,
      0.01,
      69,
    ),
    () => 4 * u,
  ).replace(INK, PINK)
  const brim = stroke(
    hand(
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
    ),
    pressure(71, 3.4 * u, 0.4),
  )

  const flag = `${stroke(
    densify(
      [
        [380, 142],
        [381, 84],
      ],
      3,
    ),
    thin(72, 2.8),
  )}${shape(
    hand(
      [
        [381, 86],
        [428, 98],
        [381, 112],
        [381, 86],
      ],
      4,
      1,
      73,
    ),
    PINK,
    thin(74, 2.2),
  )}`
  const birds = [
    [640, 110, 1.1],
    [700, 86, 0.9],
  ]
    .map(([x, y, k], i) =>
      [-1, 1]
        .map(side =>
          stroke(
            densify(
              [
                [x, y],
                [x + side * 9 * k, y - 7 * k],
                [x + side * 18 * k, y - 3 * k],
              ],
              2,
            ),
            thin(80 + i * 2 + side, 2.2),
          ),
        )
        .join(''),
    )
    .join('')

  return svg(
    W,
    H,
    `<rect width="${W}" height="${H}" fill="${PAPER}"/>
  <g transform="translate(0 80)">
  ${shape(sun, PINK, pressure(4, 3))}
  ${range(far, 31, SAGE, 3)}
  ${range(near, 33, PAPER, 4.2)}
  ${stroke(snow, thin(44, 2.4))}
  ${flag}
  ${birds}
  ${legs}
  <g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">
    ${pack}${roll}
    ${mascot(3.2 * u, 11)}
    ${crown}${band}${brim}
  </g>
  </g>`,
  )
}

async function main() {
  const at = process.argv.indexOf('--preview')
  const preview = at > 0 ? path.resolve(process.argv[at + 1]) : null
  const ic = sharp(icon())
  const og = sharp(card())
  // Touch icons and the jpeg have no alpha, so they get the paper behind them.
  const flat = () => ic.clone().flatten({ background: PAPER })

  if (preview) {
    await fs.mkdir(preview, { recursive: true })
    await ic.clone().png().toFile(path.join(preview, 'icon.png'))
    await og.clone().png().toFile(path.join(preview, 'og-image.png'))
    // The Favicon emitter's own resize on a grey tab bar, blown back up to be looked at.
    for (const s of [16, 32, 48]) {
      const small = await ic.clone().resize(s, s).png().toBuffer()
      await sharp(small)
        .flatten({ background: '#cfcfcf' })
        .resize(s * 8, s * 8, { kernel: 'nearest' })
        .toFile(path.join(preview, `favicon-${s}.png`))
    }
    return
  }

  const stat = (f: string) => path.join(ROOT, 'quartz/static', f)
  await ic.clone().png({ compressionLevel: 9 }).toFile(stat('icon.png'))
  await ic.clone().webp({ quality: 92 }).toFile(stat('icon.webp'))
  await flat().jpeg({ quality: 92 }).toFile(stat('icon.jpeg'))
  await og.clone().png({ compressionLevel: 9 }).toFile(stat('og-image.png'))
  await og.clone().webp({ quality: 92 }).toFile(stat('og-image.webp'))
  await og.clone().jpeg({ quality: 92 }).toFile(stat('og-image.jpeg'))
  for (const [f, s] of [
    ['apple-touch-icon.png', 180],
    ['android-chrome-192x192.png', 192],
    ['android-chrome-512x512.png', 512],
  ] as const)
    await flat()
      .resize(s, s)
      .png()
      .toFile(path.join(ROOT, 'content', f))
}

await main()
