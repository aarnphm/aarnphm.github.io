// The default social card: dawn. The sun comes up behind the back range on the right and lights
// the summit pink, and the mascot climbs the peak's right shoulder toward the flag with a trekking
// pole, a pink pack and a sage bedroll on its back and a bucket hat on its head. Its eyes look left,
// so the slope rises to the left; the trail it walked comes up from the lower right.

import { bucketHat, mascot } from '../../util/mascot'
import {
  blob,
  densify,
  even,
  hand,
  INK,
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

// The right shoulder: a bench where the climber stands, steeper below, a saddle, then a lower top.
const PEAK: Pt[] = [
  ...FLANK,
  [470, 248],
  [520, 268],
  [590, 306],
  [710, 334],
  [790, 388],
  [860, 416],
  [920, 400],
  [990, 348],
  [1060, 372],
  [1140, 362],
  [W + 20, 390],
]

const STONE = mix(PAPER, SAGE, 0.42)
const BACK_FILL = mix(PAPER, SAGE, 0.68)
const DAY: Record<'far' | 'back' | 'peak' | 'near', Tone> = {
  far: {
    fill: mix(PAPER, SAGE, 0.32),
    shade: mix(PAPER, SAGE, 0.46),
    ink: mix(INK, PAPER, 0.66),
    pen: 1.2,
  },
  back: { fill: BACK_FILL, shade: mix(BACK_FILL, INK, 0.18), ink: mix(INK, PAPER, 0.44), pen: 2 },
  peak: { fill: STONE, shade: mix(STONE, INK, 0.3), ink: INK, pen: 3.2 },
  near: { fill: mix(SAGE, INK, 0.5), shade: mix(SAGE, INK, 0.6), ink: INK, pen: 4.4 },
}

const thin = (seed: number, base = 2.6) => pressure(seed, base, 0.35)
const SUN: Pt = [900, 272]

// A crescent just inside the body's outline between the angles a0..a1 (y down), `d` deep at its
// middle and tapering to nothing at the ends; `inset` keeps it inside the pen line.
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

export function card() {
  const far = ridge(FAR, 1.5, 31)
  const back = ridge(BACK, 2, 32)
  const peak = ridge(PEAK, 3, 33)
  const nearL = ridge(NEAR_L, 3, 35)
  const nearR = ridge(NEAR_R, 3, 36)
  const ground = (x: number) => ridgeAt(peak, x)
  const faces = shadeFaces(PEAK, peak, SUN[0])

  // Dawn: a faint haze low behind the far range, and the sun in the back range's saddle.
  const dawn = [
    [252, 0.04, 20],
    [272, 0.07, 21],
  ]
    .map(
      ([y, k, seed]) =>
        `<path d="${poly([
          ...hand(
            [
              [-20, y + 2],
              [400, y - 6],
              [800, y - 20],
              [W + 20, y - 28],
            ],
            10,
            3,
            seed,
          ),
          [W + 20, H],
          [-20, H],
        ])}" fill="${mix(PAPER, PINK, k)}"/>`,
    )
    .join('')
  const sun = loop(3, 0.05).map(([x, y]) => [SUN[0] + x * 52, SUN[1] + y * 52] as const)
  const sky = `${dawn}${shape(sun, PINK, thin(4, 1.8), mix(INK, PAPER, 0.4))}`

  // The rock's form under first light, then the meadows below it. Both go under the shade.
  const MEADOW: Pt[] = [
    [150, 436],
    [300, 426],
    [450, 442],
    [620, 418],
    [760, 432],
    [900, 424],
    [W + 20, 436],
  ]
  const form = facets(
    PEAK,
    peak,
    SUMMIT[0],
    870,
    MEADOW,
    {
      lit: mix(mix(STONE, PAPER, 0.5), PINK, 0.2),
      dark: mix(STONE, INK, 0.3),
      glow: mix(STONE, PINK, 0.6),
      seam: mix(STONE, INK, 0.5),
      shade: DAY.peak.shade,
    },
    81,
    'form',
  )
  const meadow =
    band(MEADOW, 28, mix(STONE, SAGE, 0.3)) +
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
      mix(STONE, SAGE, 0.55),
    )
  // Footprints up from the saddle, the last of them under the climber: drawn before it so they come
  // out from beneath its shadow.
  const trail = densify(
    [
      [884, ground(884) + 22],
      [770, ground(770) + 54],
      [812, ground(812) + 30],
      [712, ground(712) + 24],
      [652, ground(652) + 3],
    ],
    8,
  )
    .map(
      ([x, y]) =>
        `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.8" fill="${mix(STONE, INK, 0.5)}"/>`,
    )
    .join('')
  const peakOver = `${shading(faces, 'peak-hatch', mix(INK, STONE, 0.2), mix(DAY.peak.shade, INK, 0.16), 8, 23)}${snowCap(peak, faces, 356, 504, mix(PAPER, PINK, 0.12), mix(PAPER, INK, 0.16), mix(INK, PAPER, 0.3), 44)}${trail}`

  // The climber on the bench, leaning into the slope, a pole planted uphill in its front hand.
  const r = 48
  const tilt = (-18 * Math.PI) / 180
  const bx = 662
  const by = ground(bx) - r * 0.96
  const world = ([u, v]: Pt): Pt => [
    bx + r * (u * Math.cos(tilt) - v * Math.sin(tilt)),
    by + r * (u * Math.sin(tilt) + v * Math.cos(tilt)),
  ]
  const u = 1 / r
  const outline = blob(11, 0.25, 0.06, 0.03)
  const shade = `<path d="${poly(crescent(outline, 0.4 * Math.PI, 1.02 * Math.PI, 0.16))}" fill="${mix(PAPER, INK, 0.13)}" style="mix-blend-mode:multiply"/>`
  const rim = `<path d="${poly(crescent(outline, 1.3 * Math.PI, 1.9 * Math.PI, 0.07))}" fill="${mix(PAPER, PINK, 0.42)}"/>`
  const pack = shape(
    hand(rounded(1.14, 0.06, 0.66, 1.06, 0.24), 0.05, 0.02, 61),
    PINK,
    pressure(62, 2.8 * u, 0.35),
  )
  const packFold = `<path d="${poly(hand(rounded(1.14, 0.3, 0.6, 0.5, 0.18), 0.05, 0.02, 65))}" fill="${mix(PINK, INK, 0.12)}"/>`
  const roll = shape(
    hand(rounded(1.14, -0.58, 0.74, 0.32, 0.16), 0.05, 0.015, 63),
    SAGE,
    pressure(64, 2.6 * u, 0.35),
  )
  const [sx0, sx1] = [bx - r * 1.7, bx + r * 0.9]
  const contact = densify(
    [
      [sx0, ground(sx0)],
      [sx1, ground(sx1)],
    ],
    6,
  ).map(([x, y], i, a) => [x, y - 13 * Math.sin((Math.PI * i) / (a.length - 1)) ** 0.7] as const)
  const shadow = `<path d="${poly([...contact, [sx1, ground(sx1) + 2], [sx0, ground(sx0) + 2]])}" fill="${mix(STONE, INK, 0.24)}" style="mix-blend-mode:multiply"/>`
  const grip = world([-1.3, 0.16])
  const tipX = bx - 1.5 * r
  const pole = stroke(
    densify([world([-1.42, -0.98]), grip, [tipX, ground(tipX) + 1]], 4),
    thin(75, 2.6),
  )
  const basket = stroke(
    densify(
      [
        [tipX - 7, ground(tipX) - 7],
        [tipX + 7, ground(tipX) - 5],
      ],
      3,
    ),
    even(2.4),
  )
  const arm = stroke(
    hand([world([-0.86, 0.3]), world([-1.1, 0.24]), grip], 3, 0.6, 76),
    pressure(77, 3, 0.2),
  )
  const paw = shape(
    hand(rounded(grip[0], grip[1], 13, 11, 5), 2, 0.5, 78),
    mix(PAPER, PINK, 0.14),
    thin(79, 2.4),
  )

  const climber = `${shadow}${pole}${basket}<g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">${pack}${packFold}${roll}${mascot({ line: 3.2 * u, seed: 11 })}${shade}${rim}${bucketHat(u, PAPER)}</g>${arm}${paw}`

  const woods = (line: Pt[], at: (readonly [number, number])[], seed: number) =>
    pines(line, at, mix(SAGE, INK, 0.52), mix(INK, SAGE, 0.12), 2.2, seed)
  // The valley floor under each spur is darker, and scrub grows along the tread.
  const floor = (top: Pt[], seed: number) => band(top, seed, mix(SAGE, INK, 0.68))
  const tufts = (line: Pt[], xa: number, xb: number, n: number, seed: number) =>
    scrubField(line, xa, xb, [20, 58], n, mix(SAGE, INK, 0.72), seed)

  return svg(
    W,
    H,
    `<rect width="${W}" height="${H}" fill="${PAPER}"/>${sky}
${range(FAR, far, DAY.far, 31, SUN[0])}
${haze(far, FAR, 304, PAPER, 0.62, 25, 'mist-far')}
${range(BACK, back, DAY.back, 32, SUN[0])}
${haze(back, BACK, 374, PAPER, 0.55, 26, 'mist-back')}
${range(PEAK, peak, DAY.peak, 33, SUN[0], form + meadow, peakOver, false)}
${haze(peak, PEAK, 530, PAPER, 0.3, 29, 'mist-peak')}
${flag(INK, PINK, mix(PINK, INK, 0.14), thin)}
${climber}
${range(
  NEAR_L,
  nearL,
  DAY.near,
  35,
  SUN[0],
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
  DAY.near,
  36,
  SUN[0],
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
${grain(FAR, 0.08, 'multiply')}`,
  )
}
