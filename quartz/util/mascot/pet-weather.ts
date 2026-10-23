import { flake, jumper, mitten, winter } from './cold-weather'
import { lantern } from './lantern'
import {
  CRUST,
  curve,
  densify,
  even,
  filled,
  hand,
  INK,
  part,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type PetPen,
  type Pt,
} from './pen'
import { umbrella } from './rain-gear'
import { lounger, sunglasses } from './sun-gear'
import { windVest } from './wind-vest'

export function petWeather(s: PetPen) {
  const { L, w, paper, ink, pink, outline, pen, mitt } = s
  // Wrapped up against the visitor's cold (pet.inline.ts): a jumper, a toque and breath that shows.
  const cold = winter(w, paper, ink, SAGE, pink)
  const knit = part('jumper', jumper(outline, w, ink, pink))
  const toque = part('toque', cold.toque)
  const breath = part('breath', cold.puffs)
  // In the cool it zips on its wind vest; on triathlon pages it trains in it in the cold too, the back
  // panel flapping.
  const gilet = windVest(outline, w, ink, SAGE, pink)
  // In the snow, at rest, it holds both mittens up to catch the flakes falling round it; one comes
  // down into the front mitten and melts there.
  const reach: Pt[][] = [
    [
      [-0.9, -0.08],
      [-1.22, -0.3],
      [-1.36, -0.62],
    ],
    [
      [0.92, -0.08],
      [1.24, -0.32],
      [1.36, -0.64],
    ],
  ]
  const caught = part(
    'catch',
    reach
      .map(
        (pts, i) =>
          `<g class="pet-catch-arm">${pen(densify(pts, 0.05), pressure(61 + i, L * 1.1, 0.2))}<g class="pet-catch-mitt">${mitten(pts[1], pts[2], i ? -1 : 1, L * 0.76, ink, pink, SAGE, 940 + 10 * i)}</g></g>`,
      )
      .join('') + `<g class="pet-catch">${flake(-1.5, -0.98, 0.09, L * 0.45, ink)}</g>`,
  )
  // Falling where it stands: each flake drifts down its own distance with a sway.
  const snowfall = part(
    'snowfall',
    (
      [
        [-1.62, -1.9, 1.5, 0.12],
        [-1.0, -2.15, 1.0, -0.1],
        [-0.25, -2.25, 0.7, 0.08],
        [0.55, -2.1, 0.6, -0.08],
        [1.12, -1.95, 0.9, 0.1],
        [1.66, -1.7, 1.6, -0.12],
      ] as const
    )
      .map(
        ([x, y, fall, sway], i) =>
          `<g class="pet-flake" style="--fall:${fall}px;--sway:${sway}px;animation-delay:${(-0.7 * i).toFixed(1)}s">${flake(x, y, 0.07 + 0.015 * (i % 2), L * 0.4, ink)}</g>`,
      )
      .join(''),
  )
  // In fog, at dusk and at night it holds out its little lantern on the side it faces, the flame
  // and the rays off its globe each flickering between two drawings. Outfits with a prop in that
  // paw (pet.scss) hold it out behind instead.
  const lamp = (side: 1 | -1, name: string, seed: number) => {
    const l = lantern(w, paper, ink, CRUST, pink, side)
    const reach: Pt[] = [
      [0.9 * side, -0.02],
      [1.08 * side, 0.02],
      [1.22 * side, 0.1],
    ]
    return part(
      name,
      l.rays.map(r => `<g class="pet-ray-frame">${r}</g>`).join('') +
        `${l.bail}${l.body}` +
        l.flames.map(f => `<g class="pet-flame-frame">${f}</g>`).join('') +
        l.front +
        pen(densify(reach, 0.05), pressure(seed, L * 1.1, 0.2)) +
        filled(
          hand(rounded(1.22 * side, 0.1, 0.2, 0.17, 0.08), 0.02, 0.006, seed + 1),
          paper,
          pressure(seed + 2, w, 0.2),
          ink,
        ) +
        mitt(reach, side === -1 ? 1 : -1, 990 + seed),
    )
  }
  const lit = lamp(-1, 'lantern', 65) + lamp(1, 'lantern-back', 68)
  // In the wind it holds its hat on with the hand on its right, reaching up over its head; hats
  // sit at different heights (pet.scss picks the reach), and gusts stream past in two drawings.
  const holdHat = (name: string, y: number, seed: number) => {
    const reach: Pt[] = [
      [0.92, -0.12],
      ...curve([1.26, -0.56], [1.3, y - 0.1], [0.62, y - 0.02], 10),
    ]
    return part(
      name,
      pen(densify(reach, 0.05), pressure(seed, L * 1.1, 0.2)) +
        filled(
          hand(rounded(0.46, y, 0.26, 0.18, 0.08), 0.02, 0.006, seed + 1),
          paper,
          pressure(seed + 2, w, 0.2),
          ink,
        ) +
        `<g class="pet-mitten">${mitten([0.72, y - 0.04], [0.46, y], -1, L * 0.76, ink, pink, SAGE, 995 + seed)}</g>`,
    )
  }
  const holds =
    holdHat('hold-high', -1.42, 74) +
    holdHat('hold-mid', -1.24, 77) +
    holdHat('hold-low', -1.08, 80)
  const gust = part(
    'gust',
    [0, 1]
      .map(k => {
        const lines = (
          [
            [-1.78, -1.2, 0.5],
            [-1.72, -0.5, 0.38],
            [1.38, -1.08, 0.42],
          ] as const
        )
          .map(([x, y, len], i) => {
            const x0 = x + 0.12 * k
            return stroke(
              hand(
                curve([x0, y], [x0 + len * 0.5, y - 0.1 + 0.06 * k], [x0 + len, y + 0.02], 10),
                0.03,
                0.006,
                1000 + 3 * k + i,
              ),
              even(w * 0.6),
              ink,
            )
          })
          .join('')
        return `<g class="pet-gust-frame">${lines}</g>`
      })
      .join(''),
  )
  // In the heat: sunglasses, a tan that deepens while it lies out (pet.inline.ts sets how deep),
  // and on sunny afternoons a sun lounger to lie back on, legs out along the cushion.
  const shades = part('sunglasses', sunglasses(w, ink, INK, pink))
  const tan = part(
    'tan',
    `<path d="${poly(outline.map(([x, y]) => [x * 0.97, y * 0.97] as const))}" style="fill:${CRUST}"/>`,
    'pet-tan',
  )
  const bed = lounger(w, paper, ink, pink, CRUST)
  const lounge = part(
    'lounger',
    bed.frame +
      bed.cushion +
      bed.drink +
      (
        [
          [
            [-0.5, 0.72],
            [-1.3, 0.8],
            [-1.42, 0.66],
          ],
          [
            [-0.2, 0.86],
            [-1.1, 0.84],
            [-1.2, 0.7],
          ],
        ] as Pt[][]
      )
        .map((pts, i) => pen(densify(pts, 0.05), pressure(70 + i, L * 1.15, 0.2)))
        .join(''),
  )
  // In the rain it holds up its little umbrella, the rain falling past in two drawings.
  const brolly = umbrella(w, paper, ink, pink, SAGE)
  const shelter = part(
    'umbrella',
    `${brolly.rain.map(r => `<g class="pet-rain-frame">${r}</g>`).join('')}${brolly.held}`,
  )
  const vest = part(
    'vest',
    `${gilet.flaps.map(f => `<g class="pet-flap-frame">${f}</g>`).join('')}${gilet.vest}`,
  )
  return {
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
  }
}
