import {
  CRUST,
  curve,
  EYE,
  EYES,
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  poly,
  pressure,
  ribbon,
  ring,
  rounded,
  SAGE,
  stroke,
} from './pen'

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

// Cinema: the film nerd. Thick square specs, and a striped popcorn bucket held up, heaped over the
// rim. `kernel` is one more piece apart from the heap, so the pet can pop it up to its mouth.
export function cinephile(w: number, paper = PAPER, ink = INK, stripe = PINK, butter = CRUST) {
  // Each frame boxes the eye (pen.ts EYE) with room for its happy and surprised poses.
  const frames = EYES.map((x, i) =>
    hand(rounded(x + 0.02, EYE.y, 0.56, 0.6, 0.1), 0.02, 0.006, 701 + i),
  )
  const bridge = hand(
    [
      [EYES[0] + 0.3, -0.28],
      [EYES[1] - 0.26, -0.28],
    ],
    0.02,
    0.006,
    703,
  )
  const temple = hand(
    curve([EYES[1] + 0.3, -0.28], [0.7, -0.32], [0.96, -0.22], 10),
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
