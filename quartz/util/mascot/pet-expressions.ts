import {
  caretEye,
  densify,
  even,
  EYE,
  EYE_POSES,
  EYES,
  part,
  pillEye,
  poly,
  poseEye,
  pressure,
  ring,
  type PetPen,
  type Pt,
} from './pen'

// The pet's faces, all in its eyes as blobatar's are (pen.ts EYE_POSES): no mouth, no brows. Each
// is a [data-part] the state map in pet/outfits.scss shows.
export function petFace(s: PetPen) {
  const { L, paper, ink, pen } = s
  const { y, h, w } = EYE
  const look = (name: keyof typeof EYE_POSES) =>
    part(
      `eyes-${name}`,
      EYES.map(
        (x, i) =>
          `<g class="pet-eye">${poseEye(x, y, h, w, EYE_POSES[name], i as 0 | 1, i, ink)}</g>`,
      ).join(''),
    )
  const dash = part(
    'eyes-dash',
    EYES.map((x, i) => `<g class="pet-eye">${pillEye(x, y, h, w, 0.2, i, ink)}</g>`).join(''),
  )
  // Happy is a pair of carets (^ ^), set a little high as blobatar lifts its happy eyes.
  const happy = part(
    'eyes-happy',
    EYES.map(
      (x, i) => `<g class="pet-eye">${caretEye(x, y - 0.12, 0.34, w * 0.8, i, ink)}</g>`,
    ).join(''),
  )
  const surprised = look('surprised')
  const scared = look('scared')
  const thinking = look('thinking')

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
  return { dash, happy, surprised, scared, thinking, sweat, dots, thought }
}
