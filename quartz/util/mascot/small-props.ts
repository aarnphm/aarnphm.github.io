import {
  CRUST,
  curve,
  densify,
  filled,
  hand,
  INK,
  PINK,
  poly,
  pressure,
  ribbon,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from './pen'

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
