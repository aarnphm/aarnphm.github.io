import {
  CRUST,
  curve,
  densify,
  EYES,
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  poly,
  pressure,
  ribbon,
  SAGE,
  stroke,
  type Pt,
} from './pen'

// Heat: sunglasses, two dark lenses in the theme's ink rims with a glint across each, a bridge,
// and an arm back over the side of the head.
export function sunglasses(w: number, ink = INK, lens = INK, glint = PINK) {
  const lenses = EYES.map((x, i) =>
    hand(
      [
        [x - 0.26, -0.5],
        [x + 0.3, -0.5],
        [x + 0.27, -0.16],
        [x + 0.13, 0.04],
        [x - 0.13, 0.04],
        [x - 0.24, -0.18],
        [x - 0.26, -0.5],
      ],
      0.02,
      0.006,
      850 + i,
    ),
  )
  const shine = EYES.map(x =>
    stroke(
      [
        [x - 0.12, -0.2],
        [x + 0.06, -0.4],
      ],
      () => w * 0.8,
      glint,
    ),
  )
  const bridge = stroke(
    hand(
      curve([EYES[0] + 0.3, -0.44], [(EYES[0] + EYES[1]) / 2, -0.52], [EYES[1] - 0.26, -0.44], 8),
      0.02,
      0.006,
      852,
    ),
    () => w * 1.3,
    ink,
  )
  const temple = stroke(
    hand(curve([EYES[1] + 0.3, -0.44], [0.7, -0.44], [0.96, -0.32], 10), 0.02, 0.006, 853),
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
// ground at y = 1.08, where the pet's base rests. The pet lies back in it with its body turned 30°.
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
          [-1.48, 1.08],
        ],
        [
          [0.42, 1.0],
          [0.44, 1.08],
        ],
        [
          [1.15, 0.32],
          [1.06, 1.08],
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
      [-1.86, 0.66],
      [-1.62, 0.66],
      [-1.65, 1.08],
      [-1.83, 1.08],
      [-1.86, 0.66],
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
