import {
  CRUST,
  curve,
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
