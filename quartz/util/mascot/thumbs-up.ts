import { densify, filled, hand, part, pressure, rounded, type PetPen, type Pt } from './pen'

// A thumbs up from the hand on its right (the far side as it faces left): the arm bends up from
// the shoulder to a fist, a paper box wider than tall with the curled fingers as two short lines
// across its front, and a thick round thumb rising off its top. In the snow the mitten lies over
// the fist with its own thumb up.
export function thumbsUp(s: PetPen) {
  const { L, w, paper, ink, pen, mitt } = s
  const arm: Pt[] = [
    [0.9, 0.04],
    [1.24, 0.14],
    [1.36, -0.18],
  ]
  const fist = hand(rounded(1.42, -0.36, 0.4, 0.3, 0.1), 0.02, 0.005, 61)
  const fingers = [-0.4, -0.3].map(y =>
    pen(
      densify(
        [
          [1.3, y],
          [1.56, y + 0.01],
        ],
        0.04,
      ),
      () => w * 0.6,
    ),
  )
  const thumb = hand(
    [
      [1.36, -0.48],
      [1.35, -0.64],
      [1.31, -0.8],
    ],
    0.03,
    0.004,
    64,
  )
  const palm: Pt[] = [
    [1.0, -0.36],
    [1.22, -0.36],
  ]
  return part(
    'arm-thumb',
    pen(densify(arm, 0.05), pressure(62, L * 1.1, 0.2)) +
      filled(fist, paper, pressure(63, L * 1.05, 0.2), ink) +
      fingers.join('') +
      pen(thumb, () => L * 2.1) +
      mitt(palm, -1, 1010),
  )
}
