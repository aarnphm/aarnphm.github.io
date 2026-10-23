import {
  curve,
  densify,
  even,
  eye,
  EYES,
  hand,
  part,
  poly,
  pressure,
  ring,
  type PetPen,
  type Pt,
} from './pen'

export function petFace(s: PetPen) {
  const { L, paper, ink, pen } = s
  const dash = part(
    'eyes-dash',
    EYES.map(x => `<g class="pet-eye">${eye(x, L * 1.7, ink)}</g>`).join(''),
  )
  const wide = part(
    'eyes-wide',
    EYES.map((x, i) => pen(ring(21 + i, x + 0.02, -0.2, 0.13), pressure(23 + i, L, 0.25))).join(''),
    'pet-blinkless',
  )
  const wince = part(
    'eyes-wince',
    pen(
      densify(
        [
          [-0.72, -0.34],
          [-0.52, -0.2],
          [-0.72, -0.06],
        ],
        0.03,
      ),
      even(L * 1.3),
    ) +
      pen(
        densify(
          [
            [0.38, -0.34],
            [0.18, -0.2],
            [0.38, -0.06],
          ],
          0.03,
        ),
        even(L * 1.3),
      ),
    'pet-blinkless',
  )

  const mx = (EYES[0] + EYES[1]) / 2
  const mouthO = part('mouth-o', pen(ring(31, mx, 0.22, 0.09), pressure(33, L * 0.95, 0.25)))
  // One stroke, thin where the pen lands and lifts, wide enough to read at the pet's size.
  const smile = part(
    'mouth-smile',
    pen(
      hand(curve([mx - 0.2, 0.1], [mx, 0.36], [mx + 0.2, 0.1], 14), 0.03, 0.004, 34),
      pressure(35, L * 1.1, 0.15),
    ),
  )

  const sweat = part(
    'sweat',
    (
      [
        [
          [0.8, -0.98],
          [0.9, -1.22],
        ],
        [
          [1.0, -0.74],
          [1.2, -0.92],
        ],
        [
          [1.13, -0.44],
          [1.38, -0.52],
        ],
      ] as Pt[][]
    )
      .map(([a, b]) => pen(densify([a, b], 0.04), even(L)))
      .join(''),
  )
  const dots = part(
    'dots',
    (
      [
        [0.55, -1.3],
        [0.85, -1.42],
        [1.15, -1.54],
      ] as Pt[]
    )
      .map(
        ([x, y]) => `<circle class="pet-dot" cx="${x}" cy="${y}" r="0.075" style="fill:${ink}"/>`,
      )
      .join(''),
  )
  // A thought: a paper cloud up to the right, reached by two bubbles, with the three dots inside.
  const cloud = ring(57, 1.3, -2.2, 0.46).map(([x, y]) => [1.3 + (x - 1.3) * 1.5, y] as Pt)
  const thought = part(
    'thought',
    `<path d="${poly(cloud)}" style="fill:${paper}"/>${pen(cloud, pressure(58, L * 0.9, 0.25))}` +
      (
        [
          [0.66, -1.24, 0.07],
          [0.88, -1.54, 0.11],
        ] as const
      )
        .map(([x, y, r], i) => pen(ring(59 + i, x, y, r), even(L * 0.8)))
        .join('') +
      [0.95, 1.3, 1.65]
        .map(x => `<circle class="pet-dot" cx="${x}" cy="-2.2" r="0.08" style="fill:${ink}"/>`)
        .join(''),
  )
  return { dash, wide, wince, mouthO, smile, sweat, dots, thought }
}
