import {
  CRUST,
  curve,
  filled,
  hand,
  INK,
  loop,
  PAPER,
  PINK,
  poly,
  pressure,
  rounded,
  stroke,
  type Pt,
} from './pen'

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
