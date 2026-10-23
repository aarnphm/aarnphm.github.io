import { mitten } from './mascot/cold-weather'
import {
  densify,
  even,
  eye,
  EYES,
  hand,
  INK,
  loop,
  PAPER,
  part,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type PetPen,
  type Pt,
} from './mascot/pen'
import { petFace } from './mascot/pet-expressions'
import { petChute, petJetpack } from './mascot/pet-flight'
import { petOutfits } from './mascot/pet-outfits'
import { petWeather } from './mascot/pet-weather'
import { thumbsUp } from './mascot/thumbs-up'
export {
  PAPER,
  PINK,
  SAGE,
  densify,
  hand,
  loop,
  poly,
  pressure,
  rounded,
  stroke,
} from './mascot/pen'
export type { Pt } from './mascot/pen'
export { baguette, beret, bucketHat, heart, quill, sprout, sweatband } from './mascot/small-props'
export { lawyer } from './mascot/legal-outfit'
export { artist, cinephile } from './mascot/creative-outfits'
export { librarian, presenter, reader } from './mascot/reading-outfits'
export { engineer, student } from './mascot/academic-outfits'

type MascotOptions = { line: number; seed: number; paper?: string; ink?: string; points?: number }

// The mascot in its own units, radius about 1 around the origin, filled with paper inside the line
// so it reads on a dark background.
export function mascot({ line, seed, paper = PAPER, ink = INK, points = 220 }: MascotOptions) {
  const outline = loop(seed, 0.06, 0.03, points)
  const eyes = EYES.map(x => eye(x, line * 1.7, ink)).join('')
  return `<path d="${poly(outline)}" style="fill:${paper}"/>${stroke(outline, pressure(seed + 1, line), ink)}${eyes}`
}

// The pet: the mascot with every look the old rocky spritesheet had, each pre-drawn as a part. The
// SVG's data-pet-state picks which parts show and how they move (quartz/components/styles/pet.scss),
// so changing a look is one attribute write. Colours are theme variables; the pet faces left as
// drawn and is mirrored to face right. `seed` picks the outline's wobble, so a friend is its own circle.
export function petSvg(paper = 'var(--light)', ink = 'var(--dark)', pink = PINK, seed = 7): string {
  const L = 0.08
  const outline = loop(seed, 0.06, 0.03, 140)
  const pen = (pts: readonly Pt[], w: (t: number) => number, color = ink) => stroke(pts, w, color)

  const leg = (hip: Pt, foot: Pt, toe: number, seed: number) =>
    `<g class="pet-leg">${pen(densify([hip, foot, [foot[0] + toe, foot[1] + 0.02]], 0.05), pressure(seed, L * 1.15, 0.2))}</g>`
  const legs = part(
    'legs',
    leg([-0.34, 0.8], [-0.38, 1.36], -0.17, 41) + leg([0.3, 0.84], [0.33, 1.38], -0.16, 43),
    'pet-legs',
  )

  // In the snow every hand it shows wears a mitten (pet.scss shows them with data-snow).
  const mitt = (pts: readonly Pt[], side: 1 | -1, seed: number) =>
    `<g class="pet-mitten">${mitten(pts[pts.length - 2], pts[pts.length - 1], side, L * 0.76, ink, pink, SAGE, seed)}</g>`
  const arm = (side: 'l' | 'r', pts: Pt[], seed: number) =>
    part(
      `arm-${side}`,
      pen(densify(pts, 0.05), pressure(seed, L * 1.1, 0.2)) +
        mitt(pts, side === 'l' ? 1 : -1, 900 + seed),
      'pet-arm',
    )
  const arms =
    arm(
      'l',
      [
        [-0.9, -0.08],
        [-1.2, -0.42],
        [-1.3, -0.82],
      ],
      45,
    ) +
    arm(
      'r',
      [
        [0.92, -0.08],
        [1.22, -0.44],
        [1.3, -0.84],
      ],
      47,
    )
  // Both hands up on a ledge the pet hangs from; the ledge runs through y = -1.45.
  const grip = part(
    'arms-hang',
    (
      [
        [
          [-0.62, -0.78],
          [-0.68, -1.12],
          [-0.56, -1.45],
        ],
        [
          [0.5, -0.86],
          [0.57, -1.16],
          [0.44, -1.45],
        ],
      ] as Pt[][]
    )
      .map(
        (pts, i) =>
          pen(densify(pts, 0.05), pressure(48 + i, L * 1.1, 0.2)) +
          mitt(pts, i ? -1 : 1, 960 + 10 * i),
      )
      .join(''),
  )

  // Clinging to an upright pole on its right (the scroll rail), one hand above the other; the pole
  // runs through x = 1.32.
  const pole = part(
    'arms-pole',
    pen(
      hand(
        [
          [1.32, -1.9],
          [1.33, 1.6],
        ],
        0.02,
        0.01,
        52,
      ),
      pressure(53, L * 0.9, 0.2),
    ) +
      (
        [
          [
            [0.7, -0.62],
            [1.04, -0.96],
            [1.32, -1.06],
          ],
          [
            [0.96, -0.02],
            [1.16, -0.38],
            [1.32, -0.5],
          ],
        ] as Pt[][]
      )
        .map(
          (pts, i) =>
            pen(densify(pts, 0.05), pressure(54 + i, L * 1.1, 0.2)) + mitt(pts, 1, 980 + 10 * i),
        )
        .join(''),
  )

  const body = `<path d="${poly(outline)}" style="fill:${paper}"/>${pen(outline, pressure(8, L))}`
  const w = (L * 0.042) / 0.055
  const s: PetPen = { L, w, paper, ink, pink, outline, pen, mitt }

  const { dash, wide, wince, mouthO, smile, sweat, dots, thought } = petFace(s)
  const thumb = thumbsUp(s)
  const chute = petChute(s)
  const { hat, outfit, glasses, hair } = petOutfits(s)
  const {
    knit,
    toque,
    breath,
    caught,
    snowfall,
    lit,
    holds,
    gust,
    shades,
    tan,
    lounge,
    shelter,
    vest,
  } = petWeather(s)
  const { jetpack, strap } = petJetpack(s)
  const speed = part(
    'speed',
    (
      [
        [1.24, -0.28, 1.62],
        [1.32, 0.04, 1.76],
        [1.22, 0.36, 1.56],
      ] as const
    )
      .map(([a, y, b]) =>
        pen(
          densify(
            [
              [a, y],
              [b, y + 0.02],
            ],
            0.04,
          ),
          even(L * 0.9),
        ),
      )
      .join(''),
    'pet-speed',
  )
  const screen = hand(rounded(-0.18, 0.4, 1.3, 0.76, 0.07), 0.04, 0.012, 51)
  const base = hand(rounded(-0.18, 0.84, 1.66, 0.12, 0.05), 0.04, 0.008, 53)
  const laptop = part(
    'laptop',
    `<path d="${poly(screen)}" style="fill:${paper}"/>${pen(screen, pressure(52, L * 0.95, 0.25))}` +
      `<circle cx="-0.18" cy="0.4" r="0.075" style="fill:${pink}"/>` +
      `<path d="${poly(base)}" style="fill:${paper}"/>${pen(base, pressure(54, L * 0.9, 0.2))}` +
      (
        [
          [-1.08, -0.07],
          [0.76, 0.07],
        ] as const
      )
        .map(
          ([x, dx]) =>
            `<g class="pet-key">${pen(
              densify(
                [
                  [x, 0.7],
                  [x + dx, 0.54],
                ],
                0.03,
              ),
              even(L * 0.8),
            )}</g>`,
        )
        .join(''),
  )

  // A short line of its own to stand on, for when no box nearby has an edge.
  const ground = part(
    'ground',
    pen(
      hand(
        [
          [-1.2, 1.43],
          [1.2, 1.45],
        ],
        0.05,
        0.015,
        57,
      ),
      pressure(58, L * 0.8, 0.3),
    ),
  )

  return `<svg class="pet-svg" viewBox="-1.8 -2 3.6 3.8" aria-hidden="true" focusable="false" data-pet-state="idle"><defs><filter id="pet-rim" x="-20%" y="-20%" width="140%" height="140%"><feMorphology in="SourceAlpha" operator="dilate" radius="0.06" result="edge"/><feFlood style="flood-color:var(--dark)"/><feComposite in2="edge" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><g class="pet-flip">${ground}${snowfall}<g class="pet-hop"><g class="pet-lean">${chute}${legs}${lounge}${speed}<g class="pet-body">${jetpack}${hair}${outfit.back}<g class="pet-skin">${body}${tan}${vest}${strap}<g class="pet-look">${dash}${wide}${wince}${mouthO}${smile}</g>${knit}${outfit.front}${glasses}${shades}${toque}${hat}</g>${holds}${shelter}${breath}${arms}${thumb}${caught}${lit}${grip}${pole}${laptop}</g>${sweat}${dots}${gust}${thought}</g></g></g></svg>`
}
