// The card at night: the same ranges under the moon, the flag still on the summit, and the mascot
// camped on a shelf below the peak it climbed. It sits facing a fire with a marshmallow on a stick,
// a sage tent behind it and the pink pack leant on the tent. The camp stands where the day card's
// climber is, so the press kit's crops keep it in view. The press kit lays this card on the encre
// ground, so the sky is that ground exactly (quartz.config.ts darkMode light).

import { mulberry } from '../../components/scripts/404-pixel'
import { bucketHat, mascot } from '../../util/mascot'
import {
  blob,
  densify,
  hand,
  loop,
  PAPER,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from '../../util/mascot/pen'
import { fire } from './camp'
import { haze, scrubField } from './ground'
import { facets, shading } from './relief'
import {
  BACK,
  band,
  FAR,
  FLANK,
  flag,
  grain,
  H,
  mix,
  NEAR_L,
  NEAR_R,
  pines,
  range,
  ridge,
  ridgeAt,
  shadeFaces,
  shape,
  snowCap,
  SUMMIT,
  svg,
  W,
  type Tone,
} from './scene'

const NIGHT = '#100f0f'
const MOONLIGHT = '#cecdc3'
const DUSK = '#282726'
const STONE = '#1c1b1a'
const EMBER = '#fcc192'

// The right shoulder drops to a shelf wide enough for the camp, then rises to the lower top.
const SHELF: Pt[] = [
  ...FLANK,
  [470, 248],
  [520, 276],
  [560, 330],
  [596, 384],
  [630, 404],
  [720, 407],
  [830, 409],
  [930, 411],
  [980, 392],
  [1030, 356],
  [1090, 374],
  [1150, 364],
  [W + 20, 388],
]

const lit = (t: number) => mix(NIGHT, MOONLIGHT, t)
const TONES: Record<'far' | 'back' | 'peak' | 'near', Tone> = {
  far: { fill: lit(0.14), shade: lit(0.09), ink: lit(0.36), pen: 1.2 },
  back: { fill: mix(NIGHT, SAGE, 0.24), shade: mix(NIGHT, SAGE, 0.14), ink: lit(0.5), pen: 2 },
  peak: { fill: lit(0.17), shade: lit(0.1), ink: MOONLIGHT, pen: 3.2 },
  near: { fill: mix(NIGHT, SAGE, 0.11), shade: mix(NIGHT, SAGE, 0.05), ink: lit(0.72), pen: 4.4 },
}

// A crescent just inside the body's outline between a0..a1 (y down), `d` deep in the middle.
function crescent(outline: readonly Pt[], a0: number, a1: number, d: number, inset = 0.95): Pt[] {
  const arc = outline.filter(([x, y]) => {
    const a = (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI)
    return a >= a0 && a <= a1
  })
  const n = arc.length - 1
  const inner = arc.map(([x, y], i) => {
    const k = inset - d * Math.sin((Math.PI * i) / n)
    return [x * k, y * k] as const
  })
  return [...arc.map(([x, y]) => [x * inset, y * inset] as const), ...inner.reverse()]
}

export function night() {
  const far = ridge(FAR, 1.5, 31)
  const back = ridge(BACK, 2, 32)
  const shelf = ridge(SHELF, 3, 34)
  const nearL = ridge(NEAR_L, 3, 35)
  const nearR = ridge(NEAR_R, 3, 36)
  const ground = (x: number) => ridgeAt(shelf, x)
  const moon: Pt = [700, 188]
  const faces = shadeFaces(SHELF, shelf, moon[0])
  const thin = (seed: number, base = 2.6) => pressure(seed, base, 0.35)
  const line = (pts: Pt[], seed: number, base = 2.6, ink = MOONLIGHT) =>
    stroke(densify(pts, 3), thin(seed, base), ink)

  // The moon over the camp, low enough that the press kit's band crops keep all of it, with seas.
  const mr = 50
  const disc = loop(5, 0.05).map(([x, y]) => [moon[0] + x * mr, moon[1] + y * mr] as const)
  const seas = (
    [
      [-0.32, -0.2, 0.22, 91],
      [0.28, 0.12, 0.16, 92],
      [-0.06, 0.44, 0.12, 93],
    ] as const
  )
    .map(([x, y, r, seed]) => {
      const pts = loop(seed, 0.12, 0.05, 48).map(
        ([px, py]) => [moon[0] + (x + r * px) * mr, moon[1] + (y + r * py) * mr] as const,
      )
      return `<path d="${poly(pts)}" fill="${mix(MOONLIGHT, NIGHT, 0.16)}"/>`
    })
    .join('')

  // Stars over the ranges and clear of the moon: mostly dots, now and then a four-point glint.
  const rand = mulberry(97)
  const stars: string[] = []
  for (let i = 0; stars.length < 46 && i < 600; i++) {
    const x = 12 + rand() * (W - 24)
    const y = 14 + rand() * 300
    const k = rand()
    if (Math.hypot(x - moon[0], y - moon[1]) < mr + 30) continue
    if (y > ridgeAt(FAR, x) - 26) continue
    if (k > 0.9) {
      const s = 4 + 4 * rand()
      stars.push(
        line(
          [
            [x - s, y],
            [x + s, y],
          ],
          120 + i,
          1.4,
        ) +
          line(
            [
              [x, y - s],
              [x, y + s],
            ],
            220 + i,
            1.4,
          ),
      )
    } else
      stars.push(
        `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(0.9 + 1.5 * k).toFixed(2)}" fill="${MOONLIGHT}" opacity="${(0.45 + 0.55 * k).toFixed(2)}"/>`,
      )
  }
  const comet = stroke(
    densify(
      [
        [96, 50],
        [214, 94],
      ],
      3,
    ),
    t => 2.6 * t ** 1.4,
    MOONLIGHT,
  )

  const MEADOW: Pt[] = [
    [150, 436],
    [300, 426],
    [450, 442],
    [620, 418],
    [760, 432],
    [900, 424],
    [W + 20, 436],
  ]
  // The rock's form under the moon, then the meadows below it.
  const form = facets(
    SHELF,
    shelf,
    SUMMIT[0],
    600,
    MEADOW,
    { lit: lit(0.3), dark: lit(0.04), glow: lit(0.36), seam: lit(0.5), shade: lit(0.1) },
    81,
    'form',
  )
  const meadow =
    band(MEADOW, 28, mix(NIGHT, SAGE, 0.14)) +
    band(
      [
        [150, 470],
        [300, 462],
        [450, 480],
        [620, 452],
        [760, 468],
        [900, 456],
        [W + 20, 470],
      ],
      27,
      mix(NIGHT, SAGE, 0.18),
    )
  const peakOver = `${shading(faces, 'shelf-hatch', NIGHT, mix(lit(0.1), NIGHT, 0.5), 8, 23)}${snowCap(shelf, faces, 356, 504, lit(0.46), lit(0.24), lit(0.7), 44)}`

  // The fire, on the level of the shelf, left of the camper.
  const fx = 652
  const fg = ground(fx)
  const camp = fire(fx, fg, shelf, { ember: EMBER, pink: PINK, moonlight: MOONLIGHT, night: NIGHT })

  // The camper, sat on the shelf facing the fire, lit warm on that side.
  const r = 44
  const u = 1 / r
  const tilt = (-4 * Math.PI) / 180
  const bx = 748
  const by = ground(bx) - r * 0.96
  const world = ([a, b]: Pt): Pt => [
    bx + r * (a * Math.cos(tilt) - b * Math.sin(tilt)),
    by + r * (a * Math.sin(tilt) + b * Math.cos(tilt)),
  ]
  const outline = blob(11, 0.25, 0.06, 0.03)
  const warm = `<path d="${poly(crescent(outline, 0.5 * Math.PI, 1.12 * Math.PI, 0.14))}" fill="${mix(DUSK, EMBER, 0.42)}"/>`
  const seat = `<path d="${poly(
    hand(
      [
        [bx - r * 1.1, ground(bx) + 1],
        [bx - r * 0.2, ground(bx) - 6],
        [bx + r * 0.9, ground(bx) - 4],
        [bx + r * 1.1, ground(bx) + 1],
      ],
      5,
      0.8,
      117,
    ),
  )}" fill="${NIGHT}" opacity="0.6"/>`
  // The stick runs from a paw out over the flame; the marshmallow is browned on the fire side.
  const paw = world([-1.02, 0.2])
  const tip: Pt = [fx + 14, fg - 82]
  const arm = line([world([-0.82, 0.12]), paw], 113, 2.8)
  const stick = line([paw, tip], 114, 2.8)
  const mallow = shape(
    hand(rounded(tip[0] - 5, tip[1] - 1, 19, 14, 5), 2, 0.4, 115),
    '#f4e6c8',
    thin(116, 2),
    MOONLIGHT,
  )

  // The tent: a sage A-frame, its fire side warmed, its door open on the dark inside, a pole tip
  // and a guy line out to a peg; the pink pack leans on its far side.
  const tg = (x: number) => ground(x) + 1
  const apex: Pt = [864, tg(864) - 116]
  const tentPts = hand([[806, tg(806)], apex, [926, tg(926)], [806, tg(806)]], 4, 1.4, 121)
  const tent = `<path d="${poly(tentPts)}" fill="${mix(NIGHT, SAGE, 0.7)}"/><path d="${poly([[806, tg(806)], apex, [apex[0] + 2, tg(apex[0] + 2)]])}" fill="${mix(mix(NIGHT, SAGE, 0.7), EMBER, 0.3)}"/>${stroke(tentPts, thin(122, 3), MOONLIGHT)}`
  const door = shape(
    hand(
      [
        [apex[0] - 3, apex[1] + 22],
        [844, tg(844)],
        [874, tg(874)],
        [apex[0] - 3, apex[1] + 22],
      ],
      3,
      0.8,
      123,
    ),
    STONE,
    thin(124, 2),
    MOONLIGHT,
  )
  const pole = line([apex, [apex[0] + 2, apex[1] - 14]], 125, 2.8)
  const guy = line(
    [
      [apex[0] + 6, apex[1] + 8],
      [960, ground(960) - 1],
    ],
    126,
    1.4,
  )
  const peg = line(
    [
      [956, ground(960) - 9],
      [962, ground(960) + 2],
    ],
    127,
    2.6,
  )
  const pack = `<g transform="translate(938 ${tg(938) - 24}) rotate(-12)">${shape(hand(rounded(0, 0, 32, 46, 9), 3, 0.8, 128), PINK, thin(129, 2.6), MOONLIGHT)}${line(
    [
      [-14, -8],
      [14, -8],
    ],
    130,
    2,
  )}</g>`

  const woods = (line: Pt[], at: (readonly [number, number])[], seed: number) =>
    pines(line, at, mix(NIGHT, SAGE, 0.07), lit(0.4), 1.9, seed)
  const floor = (top: Pt[], seed: number) => band(top, seed, mix(NIGHT, SAGE, 0.04))
  const tufts = (line: Pt[], xa: number, xb: number, n: number, seed: number) =>
    scrubField(line, xa, xb, [20, 58], n, lit(0.3), seed)

  return svg(
    W,
    H,
    `<rect width="${W}" height="${H}" fill="${NIGHT}"/>${stars.join('')}${comet}
<path d="${poly(disc)}" fill="${MOONLIGHT}"/>${seas}${stroke(disc, pressure(6, 2.4), PAPER)}
${range(FAR, far, TONES.far, 31, moon[0])}
${haze(far, FAR, 304, MOONLIGHT, 0.16, 25, 'mist-far')}
${range(BACK, back, TONES.back, 32, moon[0])}
${haze(back, BACK, 374, MOONLIGHT, 0.12, 26, 'mist-back')}
${range(SHELF, shelf, TONES.peak, 34, moon[0], form + meadow, peakOver, false)}
${haze(shelf, SHELF, 530, MOONLIGHT, 0.07, 29, 'mist-peak')}
${flag(MOONLIGHT, PINK, mix(PINK, NIGHT, 0.2), thin)}
${camp.glow}${tent}${door}${pole}${guy}${peg}${pack}
${camp.back}${camp.body}${camp.front}
${seat}${arm}
<g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">${mascot({ line: 3.2 * u, seed: 11, paper: DUSK, ink: MOONLIGHT })}${warm}${bucketHat(u, DUSK, MOONLIGHT)}</g>
${stick}${mallow}
${range(
  NEAR_L,
  nearL,
  TONES.near,
  35,
  moon[0],
  '',
  floor(
    [
      [-20, 548],
      [160, 536],
      [340, 596],
      [520, 650],
    ],
    37,
  ) + tufts(NEAR_L, -10, 480, 15, 140),
)}${woods(
      NEAR_L,
      [
        [30, 70],
        [78, 98],
        [120, 60],
        [172, 84],
        [214, 52],
        [262, 90],
        [318, 58],
      ],
      90,
    )}
${range(
  NEAR_R,
  nearR,
  TONES.near,
  36,
  moon[0],
  '',
  floor(
    [
      [600, 650],
      [800, 636],
      [1000, 614],
      [W + 20, 596],
    ],
    38,
  ) + tufts(NEAR_R, 700, 1210, 15, 160),
)}${woods(
      NEAR_R,
      [
        [904, 60],
        [962, 84],
        [1024, 100],
        [1064, 62],
        [1122, 88],
        [1178, 70],
      ],
      120,
    )}
${grain(FAR, 0.06, 'screen')}`,
  )
}
