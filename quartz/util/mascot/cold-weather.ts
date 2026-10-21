import {
  curve,
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from './pen'

// Cold weather: a knit toque pulled down to the brow, ribbed cuff and pom-pom. `puffs` are three
// breaths, drawn apart so the pet can let them out one at a time.
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
  return { toque, puffs }
}

// Deep cold: a knit jumper pulled on over the body from the chin down. `body` is the pet's own
// outline, so the knit tucks under its line as the wind vest does. A crew collar and a hem in rib,
// and a row of cream stitches across the chest, cream in either theme.
export function jumper(body: readonly Pt[], w: number, ink = INK, knit = PINK, stitch = PAPER) {
  // The neckline dips at the middle, as cloth over a round chin does.
  const top = (x: number) => 0.32 + 0.08 * (1 - x * x)
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
  const line = (pts: Pt[], seed: number, k = 1, color = INK) =>
    stroke(pts, pressure(seed, w * k, 0.3), color)
  const tick = (a: Pt, b: Pt) => stroke([a, b], () => w * 0.35, INK)
  // The collar: the neckline in the theme's ink, a seam under it, and ribs between.
  const rib = 0.11
  const collar = line(hand(inside(across(top, -1, 1)), 0.03, 0.008, 721), 722, 1, ink)
  const seam = line(hand(inside(across(x => top(x) + rib, -1, 1)), 0.03, 0.008, 723), 724, 0.55)
  const collarRibs = Array.from({ length: 15 }, (_, i) => -0.84 + i * 0.12)
    .filter(x => Math.abs(x) < edge(top(x) + rib) - 0.04)
    .map(x => tick([x, top(x) + 0.02], [x, top(x) + rib - 0.02]))
    .join('')
  // The hem: a seam across the bottom of the body, ribs down from it to the body's edge.
  const hem = (x: number) => 0.8 + 0.04 * (1 - x * x)
  const hemSeam = line(hand(inside(across(hem, -1, 1)), 0.03, 0.008, 725), 726, 0.55)
  const hemRibs = Array.from({ length: 13 }, (_, i) => -0.6 + i * 0.1)
    .filter(x => edge(hem(x)) > Math.abs(x) && 0.95 * Math.sqrt(1 - x * x) - hem(x) > 0.08)
    .map(x => tick([x, hem(x) + 0.02], [x, 0.95 * Math.sqrt(1 - x * x) - 0.03]))
    .join('')
  // Cream stitches across the chest, a row of small v's.
  const row = 0.6
  const stitches = Array.from({ length: 13 }, (_, i) => -0.72 + i * 0.12)
    .filter(x => Math.abs(x) < edge(row) - 0.1)
    .map(x =>
      stroke(
        [
          [x - 0.04, row - 0.035],
          [x, row + 0.035],
          [x + 0.04, row - 0.035],
        ],
        () => w * 0.45,
        stitch,
      ),
    )
    .join('')
  return `<path d="${poly(shape)}" style="fill:${knit}"/>${seam}${collarRibs}${hemSeam}${hemRibs}${stitches}${collar}`
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
