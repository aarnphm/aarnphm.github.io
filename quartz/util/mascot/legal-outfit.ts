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
  pressure,
  rounded,
  stroke,
} from './pen'

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
