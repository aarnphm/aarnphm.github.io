import {
  CRUST,
  curve,
  densify,
  even,
  filled,
  hand,
  part,
  poly,
  pressure,
  ribbon,
  ring,
  rounded,
  type PetPen,
  type Pt,
} from './pen'

export function petChute(s: PetPen) {
  const { L, paper, pen } = s
  // A parachute for coming down off the scroll rail, drawn as a kid would sew one: a paper canopy
  // leaning off true over uneven scallops, a hatched patch, a knob on top, and four lines to the
  // raised hands, the middle two crossed. It is drawn three times with different wobbles; pet.scss
  // flicks between them so the line boils, and puffs the canopy as it fills.
  const hem: Pt[] = (
    [
      [-1.78, -2.92],
      [-0.66, -3.06],
      [0.38, -2.98],
      [1.6, -2.8],
    ] as const
  ).map(([x, y]) => [x, y])
  const hands: Pt[] = [
    [-0.56, -1.45],
    [0.44, -1.45],
  ]
  const chuteFrame = (k: number) => {
    const seed = 140 + k * 11
    // The dome is two halves meeting at a crown off centre, so the crown is on the canopy line and
    // the gores and knob can end on it. Each drawing nudges the crown so the silhouette boils too.
    const crown: Pt = [0.12 + 0.05 * (k - 1), -4.6 + 0.04 * ((k * 2) % 3)]
    const left = curve(hem[0], [-1.72, crown[1] - 0.02], crown, 11)
    const right = curve(crown, [1.62, crown[1] + 0.08], hem[3], 11)
    const dome = [...left, ...right.slice(1)]
    const scallops = [3, 2, 1].flatMap(i =>
      curve(
        hem[i],
        [(hem[i][0] + hem[i - 1][0]) / 2, Math.max(hem[i][1], hem[i - 1][1]) + 0.2 + 0.07 * i],
        hem[i - 1],
        7,
      ).slice(1),
    )
    const canopy = hand([...dome, ...scallops], 0.08, 0.085, seed)
    const gores = [left[8], right[3]]
      .map((top, i) =>
        pen(
          hand(
            curve(
              hem[i + 1],
              [hem[i + 1][0] * 0.82 + top[0] * 0.18 + (i ? 0.16 : -0.16), -3.8],
              top,
              9,
            ),
            0.08,
            0.05,
            seed + i,
          ),
          even(L * 0.65),
        ),
      )
      .join('')
    const patch = hand(rounded(-0.98, -3.72, 0.44, 0.36, 0.07), 0.04, 0.012, seed + 3)
    const hatch = [-0.1, 0.04, 0.18]
      .map(d =>
        pen(
          densify(
            [
              [-1.12 + d, -3.6],
              [-0.98 + d, -3.84],
            ],
            0.05,
          ),
          even(L * 0.45),
        ),
      )
      .join('')
    const knob = ring(seed + 4, crown[0] + 0.02, crown[1] - 0.15, 0.13)
    // Outer lines to their own hands; the middle two cross on the way down.
    const lines = (
      [
        [hem[0], hands[0], -0.08],
        [hem[1], hands[1], 0.06],
        [hem[2], hands[0], -0.06],
        [hem[3], hands[1], 0.1],
      ] as const
    )
      .map(([a, b, bow], i) =>
        pen(
          hand(
            curve(a, [(a[0] + b[0]) / 2 + bow, (a[1] + b[1]) / 2], b, 10),
            0.08,
            0.04,
            seed + 5 + i,
          ),
          even(L * 0.55),
        ),
      )
      .join('')
    return `<g class="pet-canopy-frame"><path d="${poly(canopy)}" style="fill:${paper}"/>${pen(canopy, pressure(seed + 9, L, 0.3))}${gores}<path d="${poly(patch)}" style="fill:${paper}"/>${pen(patch, even(L * 0.55))}${hatch}<path d="${poly(knob)}" style="fill:${paper}"/>${pen(knob, even(L * 0.6))}${lines}</g>`
  }
  return part('chute', `<g class="pet-canopy">${[0, 1, 2].map(chuteFrame).join('')}</g>`)
}

export function petJetpack(s: PetPen) {
  const { L, paper, ink, pink, pen } = s
  // A jetpack for flying up to the scroll rail: two fat tanks on its back (behind the body, on the
  // right as drawn, since it faces left), a strap round the front so it reads as worn, and below
  // each nozzle a flame of three tongues with exhaust puffs, drawn twice so they flicker.
  const tanks = [
    [1.0, -0.06, 0.4, 1.02, 251],
    [1.38, 0.02, 0.34, 0.9, 253],
  ] as const
  const pack = tanks
    .map(([x, y, tw, th, seed]) => {
      const tank = hand(rounded(x, y, tw, th, tw / 2), 0.04, 0.01, seed)
      const band = (k: number, s: number) =>
        hand(
          [
            [x - tw / 2 + 0.03, y + th * k],
            [x + tw / 2 - 0.03, y + th * k],
          ],
          0.03,
          0.006,
          s,
        )
      const bottom = y + th / 2
      const nozzle: Pt[] = [
        [x - 0.08, bottom - 0.02],
        [x + 0.08, bottom - 0.02],
        [x + 0.12, bottom + 0.11],
        [x - 0.12, bottom + 0.11],
        [x - 0.08, bottom - 0.02],
      ]
      return `${filled(tank, paper, pressure(seed + 2, L * 0.95, 0.25), ink)}${pen(band(-0.2, seed + 1), () => L * 1.7, pink)}${pen(band(0.16, seed + 3), even(L * 0.6))}${filled(densify(nozzle, 0.04), ink, () => L * 0.6, ink)}`
    })
    .join('')
  const strap = part(
    'jetpack',
    pen(
      hand(curve([0.9, -0.5], [0.52, 0.02], [0.84, 0.6]), 0.03, 0.01, 257),
      pressure(258, L * 0.85, 0.3),
    ),
  )
  const tongue = (x: number, top: number, len: number, lean: number, w: number, seed: number) =>
    hand(
      ribbon(
        densify(
          [
            [x, top],
            [x + lean, top + len],
          ],
          0.03,
        ),
        t => w * (1 - t) ** 0.9 * Math.min(1, 0.5 + t * 6),
      ),
      0.03,
      0.012,
      seed,
    )
  const flame = (x: number, top: number, len: number, k: number, seed: number) => {
    const sides = [-1, 1]
      .map(side =>
        filled(
          tongue(
            x + side * 0.05,
            top,
            len * (0.5 + 0.12 * ((k + side + 2) % 2)),
            side * 0.12,
            0.07,
            seed + side,
          ),
          pink,
          () => L * 0.45,
          ink,
        ),
      )
      .join('')
    const middle = tongue(x, top, len, k ? 0.04 : -0.04, 0.1, seed + 3)
    const core = tongue(x, top, len * 0.5, 0, 0.05, seed + 4)
    // Puffs drift off below the tip, at other spots in each drawing.
    const puffs = [0, 1]
      .map(j => {
        const r = 0.07 + 0.04 * ((j + k) % 2)
        const ring_ = ring(
          seed + 6 + j,
          x + (j ? 0.1 : -0.08) * (k ? -1 : 1),
          top + len + 0.08 + j * 0.16,
          r,
        )
        return filled(ring_, paper, () => L * 0.5, ink)
      })
      .join('')
    return `${sides}${filled(middle, pink, () => L * 0.5, ink)}<path d="${poly(core)}" style="fill:${CRUST}"/>${puffs}`
  }
  const flames = [0, 1]
    .map(
      k =>
        `<g class="pet-flame">${tanks
          .map(([x, y, , th], i) =>
            flame(
              x,
              y + th / 2 + 0.12,
              k ? 0.42 + 0.06 * i : 0.56 - 0.08 * i,
              k,
              260 + k * 10 + i * 20,
            ),
          )
          .join('')}</g>`,
    )
    .join('')
  const jetpack = part('jetpack', `${flames}${pack}`)
  return { jetpack, strap }
}
