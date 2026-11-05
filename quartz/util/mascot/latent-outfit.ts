import {
  densify,
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
  stroke,
  type Pt,
} from './pen'

// Embeddings: the surveyor of the latent space. A three-draw spyglass held to its near eye,
// trained on a small constellation past its objective: five points joined to their nearest
// neighbours, the query point pink, as the latent map draws them. A rolled chart is tucked over
// the other shoulder. `stars` are returned apart, so the pet can twinkle them.
export function surveyor(w: number, paper = PAPER, ink = INK, pink = PINK) {
  // The spyglass axis, from the eyepiece at the near eye to the objective, in body radii.
  const eye: Pt = [-0.8, -0.22]
  const far: Pt = [-1.4, -0.82]
  const len = Math.hypot(far[0] - eye[0], far[1] - eye[1])
  const along = (d: number): Pt => [
    eye[0] + ((far[0] - eye[0]) * d) / len,
    eye[1] + ((far[1] - eye[1]) * d) / len,
  ]
  // Each draw is a band along the axis, wider toward the objective, drawn far end first so the
  // nearer draws overlap it.
  const draw = (from: number, to: number, half: number, seed: number) =>
    filled(
      hand(
        ribbon(densify([along(from), along(to)], 0.03), () => half),
        0.02,
        0.005,
        seed,
      ),
      paper,
      pressure(seed + 1, w, 0.25),
      ink,
    )
  const tube = `${draw(0.5, len, 0.115, 501)}${draw(0.25, 0.52, 0.095, 503)}${draw(0.04, 0.27, 0.078, 505)}`
  // The objective's glass and the pink eyepiece ring.
  const glass = stroke(
    hand(
      ribbon(densify([along(len - 0.04), along(len)], 0.01), () => 0.09),
      0.01,
      0.003,
      507,
    ),
    () => w * 0.5,
    ink,
  )
  const band = filled(
    hand(
      ribbon(densify([along(0.01), along(0.08)], 0.01), () => 0.082),
      0.01,
      0.003,
      508,
    ),
    pink,
    () => w * 0.5,
    ink,
  )
  // The paw holds the middle draw from below.
  const [gx, gy] = along(0.4)
  const paw = filled(
    hand(rounded(gx + 0.03, gy + 0.13, 0.23, 0.19, 0.09), 0.02, 0.006, 509),
    paper,
    pressure(510, w, 0.2),
    ink,
  )
  const spyglass = `${tube}${glass}${band}${paw}`

  // The constellation: points past the objective, up and to the left, and the links between
  // nearest neighbours. The first point is the query.
  const points: readonly Pt[] = [
    [-1.4, -1.22],
    [-1.1, -1.54],
    [-1.5, -1.58],
    [-0.96, -1.16],
    [-1.24, -0.96],
  ]
  const links = (
    [
      [0, 1],
      [0, 2],
      [1, 3],
      [0, 4],
    ] as const
  )
    .map(([a, b], i) =>
      stroke(hand([points[a], points[b]], 0.02, 0.005, 512 + i), () => w * 0.35, ink),
    )
    .join('')
  const stars = `${links}${points
    .map(
      ([x, y], i) =>
        `<g class="pet-star">${filled(ring(520 + i, x, y, i ? 0.05 : 0.065), i ? paper : pink, () => w * 0.5, ink)}</g>`,
    )
    .join('')}`

  // The chart, rolled, over the far shoulder: a tube tied with a pink band, its end showing the
  // roll.
  const a: Pt = [0.46, -0.9]
  const b: Pt = [1.18, -0.42]
  const roll = hand(
    ribbon(densify([a, b], 0.03), () => 0.115),
    0.02,
    0.006,
    530,
  )
  const end = ring(531, b[0], b[1], 0.115)
  const tie = filled(
    hand(
      ribbon(
        densify(
          [
            [0.84, -0.66],
            [0.92, -0.58],
          ],
          0.01,
        ),
        () => 0.125,
      ),
      0.01,
      0.003,
      532,
    ),
    pink,
    () => w * 0.5,
    ink,
  )
  const curl = stroke(
    hand(ring(533, b[0] - 0.01, b[1] + 0.01, 0.05).slice(0, 26), 0.01, 0.003, 534),
    () => w * 0.4,
    ink,
  )
  const chart = `<path d="${poly(roll)}" style="fill:${paper}"/>${stroke(roll, pressure(535, w, 0.25), ink)}${filled(end, paper, () => w * 0.6, ink)}${curl}${tie}`
  return { spyglass, stars, chart }
}
