// Rasterises the garden's mascot (quartz/util/mascot.ts) into the site icon and the default social
// card, where it climbs a mountain range, and the card's night twin, where it camps below the peak.
//
//   pnpm exec tsx quartz/scripts/brand-art.ts              writes the static assets
//   pnpm exec tsx quartz/scripts/brand-art.ts --preview D  writes previews (and favicon sizes) to D

import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { mulberry } from '../components/scripts/404-pixel'
import {
  baguette,
  beret,
  bucketHat,
  densify,
  hand,
  lawyer,
  artist,
  librarian,
  presenter,
  loop,
  mascot,
  PAPER,
  PINK,
  poly,
  quill,
  pressure,
  rounded,
  SAGE,
  stroke,
  sweatband,
  type Pt,
} from '../util/mascot'

const ROOT = path.resolve(import.meta.dirname, '../..')

const svg = (w: number, h: number, body: string) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${body}</svg>`,
  )

// The site icon: the mascot alone, with `behind` and `front` drawn either side of it in its units.
// Section icons carry props past the body, so they draw it smaller to keep them on the canvas.
const icon = (behind = '', front = '', scale = 410, x = 512) =>
  svg(
    1024,
    1024,
    `<g transform="translate(${x} 512) scale(${scale})">${behind}${mascot({ line: 0.055, seed: 7 })}${front}</g>`,
  )

const ICON_PEN = 0.042

// Triathlon: a pink sweatband round the brow, tied at the back with its tails flying.
const triathlonIcon = () => icon('', sweatband(ICON_PEN), 340)

// French: a black beret slumped to the back of the head and two baguettes, one held across the
// front and one behind.
const frenchIcon = () =>
  icon(
    baguette([0.3, 1.02], [1.46, -0.2], 223, ICON_PEN),
    `${baguette([-0.62, 1.2], [1.4, 0.3], 227, ICON_PEN)}${beret(ICON_PEN)}`,
    340,
  )

// Posts: the writer. A pink quill in a stubby hand on its right, nib down at the end of a line of
// loopy handwriting it is looking back along.
function writerIcon() {
  const { svg: pen, shaft } = quill([0.98, 1.34], [1.24, 0.4], [1.56, -0.92], ICON_PEN)
  const grip = shaft[8]
  const arm = stroke(
    hand([[0.84, 0.52], [(0.84 + grip[0]) / 2, grip[1] - 0.02], grip], 0.03, 0.01, 245),
    pressure(246, ICON_PEN * 1.5, 0.2),
  )
  const paw = shape(
    hand(rounded(grip[0] - 0.02, grip[1], 0.28, 0.23, 0.11), 0.03, 0.01, 247),
    PAPER,
    pressure(248, ICON_PEN, 0.2),
  )
  // Handwriting: a prolate cycloid, looping at the top like a run of cursive l's, broken into words.
  const words = [
    [-1.18, -0.6],
    [-0.42, 0.22],
    [0.4, 0.9],
  ]
    .map(([x0, x1], i) => {
      const n = 60
      const pts = Array.from({ length: n + 1 }, (_, k) => {
        const t = (k / n) * ((x1 - x0) / 0.13) * 2 * Math.PI
        return [
          x0 + (0.13 * t) / (2 * Math.PI) + 0.04 * Math.sin(t),
          1.38 - 0.11 * (1 - Math.cos(t)) - 0.02 * Math.sin(t / 3),
        ] as const
      })
      return stroke(hand(pts, 0.02, 0.006, 250 + i), () => ICON_PEN * 0.62)
    })
    .join('')
  return icon('', `${words}${pen}${arm}${paw}`, 340, 470)
}

// Terms and privacy: the lawyer, in a barrister's wig and a pink tie, a briefcase at its side.
function lawyerIcon() {
  const { wig, neck, briefcase } = lawyer(ICON_PEN)
  return icon('', `${neck}${wig}${briefcase}`, 340, 470)
}

// Quotes and images: the artist, palette in front, a pink-tipped brush over its shoulder.
function artistIcon() {
  const { palette, brush, smudge } = artist(ICON_PEN)
  return icon(brush, `${smudge}${palette}`, 340, 470)
}

// Bases: the librarian, round glasses and a stack of books.
function librarianIcon() {
  const { glasses, books } = librarian(ICON_PEN)
  return icon('', `${glasses}${books}`, 340, 470)
}

// Slides: the presenter, bow tie on, pointing at a bar chart on a small screen.
function presenterIcon() {
  const { bow, pointer, screen } = presenter(ICON_PEN)
  return icon(screen, `${bow}${pointer}`, 262, 620)
}

const SECTION_ICONS = {
  triathlon: triathlonIcon,
  fr: frenchIcon,
  posts: writerIcon,
  legal: lawyerIcon,
  art: artistIcon,
  library: librarianIcon,
  slides: presenterIcon,
} as const

const W = 1200
const H = 630

// The back range both cards share, before the scene's 80 drop for sky.
const FAR: Pt[] = [
  [-20, 330],
  [90, 262],
  [200, 318],
  [470, 300],
  [600, 262],
  [720, 190],
  [810, 246],
  [930, 150],
  [1060, 240],
  [W + 20, 214],
]

// A ridge's height at x, before the hand wobble, for setting feet on it.
function ridgeAt(pts: readonly Pt[], x: number) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i]
    const [bx, by] = pts[i + 1]
    if (x >= ax && x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
  }
  return H
}

// A closed shape in one colour inside a pen line.
const shape = (pts: readonly Pt[], fill: string, width: (t: number) => number, ink?: string) =>
  `<path d="${poly(pts)}" fill="${fill}"/>${stroke(pts, width, ink)}`

// The card: the mascot climbs the long right shoulder of a peak, a pink pack and a sage bedroll on
// its back and a bucket hat on its head, toward a flag on the summit. Its
// eyes already look left, so the slope rises to the left. A sage range stands behind with the pink
// sun coming up between its peaks. The whole scene sits 80 lower than drawn, for sky.
function card() {
  const near: Pt[] = [
    [-20, 520],
    [110, 430],
    [200, 455],
    [300, 250],
    [380, 140],
    [520, 250],
    [640, 318],
    [770, 392],
    [860, 418],
    [970, 318],
    [1060, 352],
    [1130, 330],
    [W + 20, 362],
  ]
  const ridge = (x: number) => ridgeAt(near, x)
  const range = (pts: Pt[], seed: number, fill: string, base: number) => {
    const line = hand(pts, 6, 3, seed)
    return `<path d="${poly([...line, [W + 20, H + 20], [-20, H + 20]])}" fill="${fill}"/>${stroke(line, pressure(seed + 1, base, 0.5))}`
  }
  const sun = loop(3, 0.07).map(([x, y]) => [845 + x * 62, 200 + y * 62] as const)
  const thin = (seed: number, base = 2.6) => pressure(seed, base, 0.35)

  const snow = hand(
    [
      [334, 192],
      [356, 214],
      [374, 196],
      [396, 224],
      [418, 198],
      [446, 214],
      [454, 196],
    ],
    4,
    1.2,
    43,
  )

  // The climber: body frame in mascot units, leaning into the hill.
  const r = 48
  const tilt = (-16 * Math.PI) / 180
  const bx = 655
  const by = ridge(655) - r - 40
  const world = ([u, v]: Pt): Pt => [
    bx + r * (u * Math.cos(tilt) - v * Math.sin(tilt)),
    by + r * (u * Math.sin(tilt) + v * Math.cos(tilt)),
  ]
  // Legs in world space so the feet land on the slope: the front one planted uphill, the back one
  // pushing off with its heel up.
  const leg = (hip: Pt, foot: number, lift: number, toe: number, seed: number) => {
    const h = world(hip)
    const f: Pt = [foot, ridge(foot) - lift]
    const knee: Pt = [(h[0] + f[0]) / 2 - 6, (h[1] + f[1]) / 2]
    return stroke(densify([h, knee, f, [f[0] + toe, f[1] + 1]], 3), thin(seed, 2.8))
  }
  const front = world([-0.3, 0.9])
  const back = world([0.32, 0.9])
  const legs =
    leg([-0.3, 0.9], front[0] - 10, 0, -7, 51) + leg([0.32, 0.9], back[0] + 12, 2, -6, 52)

  const u = 1 / r
  const pack = shape(
    hand(rounded(1.14, 0.06, 0.66, 1.06, 0.24), 0.05, 0.02, 61),
    PINK,
    pressure(62, 2.8 * u, 0.35),
  )
  const roll = shape(
    hand(rounded(1.14, -0.58, 0.74, 0.32, 0.16), 0.05, 0.015, 63),
    PAPER,
    pressure(64, 2.6 * u, 0.35),
  )
  const hat = bucketHat(u, PAPER)

  const flag = `${stroke(
    densify(
      [
        [380, 142],
        [381, 84],
      ],
      3,
    ),
    thin(72, 2.8),
  )}${shape(
    hand(
      [
        [381, 86],
        [428, 98],
        [381, 112],
        [381, 86],
      ],
      4,
      1,
      73,
    ),
    PINK,
    thin(74, 2.2),
  )}`
  const birds = [
    [640, 110, 1.1],
    [700, 86, 0.9],
  ]
    .map(([x, y, k], i) =>
      [-1, 1]
        .map(side =>
          stroke(
            densify(
              [
                [x, y],
                [x + side * 9 * k, y - 7 * k],
                [x + side * 18 * k, y - 3 * k],
              ],
              2,
            ),
            thin(80 + i * 2 + side, 2.2),
          ),
        )
        .join(''),
    )
    .join('')

  return svg(
    W,
    H,
    `<rect width="${W}" height="${H}" fill="${PAPER}"/>
  <g transform="translate(0 80)">
  ${shape(sun, PINK, pressure(4, 3))}
  ${range(FAR, 31, SAGE, 3)}
  ${range(near, 33, PAPER, 4.2)}
  ${stroke(snow, thin(44, 2.4))}
  ${flag}
  ${birds}
  ${legs}
  <g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">
    ${pack}${roll}
    ${mascot({ line: 3.2 * u, seed: 11 })}
    ${hat}
  </g>
  </g>`,
  )
}

// The site's dark palette (quartz.config.ts darkMode): the ground, the ink, a raised surface, and
// the tertiary that lights the fire. The press kit lays the night card on the encre ground, so the
// sky is that ground exactly.
const NIGHT = '#100f0f'
const MOONLIGHT = '#cecdc3'
const DUSK = '#282726'
const STONE = '#1c1b1a'
const EMBER = '#fcc192'

const mix = (a: string, b: string, t: number) => {
  const c = (h: string) => [1, 3, 5].map(i => Number.parseInt(h.slice(i, i + 2), 16))
  const [x, y] = [c(a), c(b)]
  return `#${x
    .map((v, i) =>
      Math.round(v + (y[i] - v) * t)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`
}

// The card at night: the same ranges under the moon, the flag still on the summit, and the mascot
// camped on a shelf below the peak it climbed. It sits facing a fire with a marshmallow on a stick,
// a sage tent behind it and the pink pack leant on the tent. The camp stands where the day card's
// climber is, so the press kit's crops keep it in view.
function night() {
  const near: Pt[] = [
    [-20, 520],
    [110, 430],
    [200, 455],
    [300, 250],
    [380, 140],
    [450, 214],
    [500, 290],
    [550, 324],
    [700, 330],
    [880, 334],
    [970, 318],
    [1060, 352],
    [1130, 330],
    [W + 20, 362],
  ]
  const ground = (x: number) => ridgeAt(near, x)
  const thin = (seed: number, base = 2.6) => pressure(seed, base, 0.35)
  const line = (pts: Pt[], seed: number, base = 2.6, ink = MOONLIGHT) =>
    stroke(densify(pts, 3), thin(seed, base), ink)
  const range = (pts: Pt[], seed: number, fill: string, base: number) => {
    const edge = hand(pts, 6, 3, seed)
    return `<path d="${poly([...edge, [W + 20, H + 20], [-20, H + 20]])}" fill="${fill}"/>${stroke(edge, pressure(seed + 1, base, 0.5), MOONLIGHT)}`
  }

  // The moon over the camp, low enough that the press kit's band crops keep all of it, with seas.
  const moon: Pt = [700, 108]
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
    const y = -66 + rand() * 300
    const k = rand()
    if (Math.hypot(x - moon[0], y - moon[1]) < mr + 30) continue
    if (y > Math.min(ridgeAt(FAR, x), ground(x)) - 26) continue
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
        [96, -30],
        [214, 14],
      ],
      3,
    ),
    t => 2.6 * t ** 1.4,
    MOONLIGHT,
  )

  const snow = hand(
    [
      [334, 192],
      [356, 214],
      [374, 196],
      [396, 224],
      [418, 198],
      [446, 214],
      [454, 196],
    ],
    4,
    1.2,
    43,
  )
  const flag = `${line(
    [
      [380, 142],
      [381, 84],
    ],
    72,
    2.8,
  )}${shape(
    hand(
      [
        [381, 86],
        [428, 98],
        [381, 112],
        [381, 86],
      ],
      4,
      1,
      73,
    ),
    PINK,
    thin(74, 2.2),
    MOONLIGHT,
  )}`

  // The fire: a patch of lit ground, two crossed logs, a flame inside a flame, and sparks rising.
  const fx = 584
  const fg = ground(fx)
  const glow = loop(9, 0.05)
    .map(([x, y]) => [fx + 6 + x * 84, fg + 3 + y * 10] as const)
    .filter(([, y]) => y >= fg - 1)
  const lit = `<path d="${poly(glow)}" fill="${mix(STONE, EMBER, 0.09)}"/>`
  const logs =
    line(
      [
        [fx - 38, fg + 2],
        [fx + 36, fg - 13],
      ],
      101,
      6,
    ) +
    line(
      [
        [fx - 36, fg - 13],
        [fx + 38, fg + 2],
      ],
      102,
      6,
    )
  const flame = (k: number, fill: string, seed: number) =>
    shape(
      hand(
        (
          [
            [-22, -4],
            [-18, -26],
            [-8, -20],
            [-2, -54],
            [8, -24],
            [16, -36],
            [22, -4],
            [-22, -4],
          ] as const
        ).map(([x, y]) => [fx + x * k, fg + y * k] as const),
        3,
        1.2,
        seed,
      ),
      fill,
      thin(seed + 1, 2.2 * k),
      MOONLIGHT,
    )
  const sparks = (
    [
      [fx + 12, fg - 100, 2.4],
      [fx - 11, fg - 120, 1.9],
      [fx + 18, fg - 142, 1.4],
    ] as const
  )
    .map(([x, y, r]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${EMBER}"/>`)
    .join('')

  // The camper, sat on the shelf facing the fire, legs out toward it.
  const r = 44
  const u = 1 / r
  const tilt = (-4 * Math.PI) / 180
  const bx = 704
  const by = ground(bx) - r * 0.96
  const world = ([a, b]: Pt): Pt => [
    bx + r * (a * Math.cos(tilt) - b * Math.sin(tilt)),
    by + r * (a * Math.sin(tilt) + b * Math.cos(tilt)),
  ]
  const legs = (
    [
      [[-0.5, 0.74], -1.62, 111],
      [[-0.04, 0.9], -1.26, 112],
    ] as const
  )
    .map(([hip, reach, seed]) => {
      const h = world(hip)
      const fxx = bx + reach * r
      const f: Pt = [fxx, ground(fxx) - 1]
      return line(
        [h, [(h[0] + f[0]) / 2, (h[1] + f[1]) / 2 + 2], f, [f[0] - 1, f[1] - 7]],
        seed,
        2.8,
      )
    })
    .join('')
  // The stick runs from a paw out over the flame; the marshmallow is browned on the fire side.
  const paw = world([-1.02, 0.2])
  const tip: Pt = [fx + 16, fg - 84]
  const arm = line([world([-0.82, 0.12]), paw], 113, 2.8)
  const stick = line([paw, tip], 114, 2.8)
  const mallow = shape(
    hand(rounded(tip[0] - 5, tip[1] - 1, 19, 14, 5), 2, 0.4, 115),
    '#f4e6c8',
    thin(116, 2),
    MOONLIGHT,
  )

  // The tent: a sage A-frame with its door open on the dark inside, a pole tip, and a guy line out
  // to a peg; the pink pack leans on its far side.
  const tg = (x: number) => ground(x) + 1
  const apex: Pt = [830, tg(830) - 116]
  const tent = shape(
    hand([[772, tg(772)], apex, [892, tg(892)], [772, tg(772)]], 4, 1.4, 121),
    SAGE,
    thin(122, 3),
    MOONLIGHT,
  )
  const door = shape(
    hand(
      [
        [apex[0] - 3, apex[1] + 22],
        [810, tg(810)],
        [840, tg(840)],
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
      [946, ground(946) - 1],
    ],
    126,
    1.4,
  )
  const peg = line(
    [
      [942, ground(946) - 9],
      [948, ground(946) + 2],
    ],
    127,
    2.6,
  )
  const pack = `<g transform="translate(906 ${tg(906) - 24}) rotate(-12)">${shape(
    hand(rounded(0, 0, 32, 46, 9), 3, 0.8, 128),
    PINK,
    thin(129, 2.6),
    MOONLIGHT,
  )}${line(
    [
      [-14, -8],
      [14, -8],
    ],
    130,
    2,
  )}</g>`

  return svg(
    W,
    H,
    `<rect width="${W}" height="${H}" fill="${NIGHT}"/>
  <g transform="translate(0 80)">
  ${stars.join('')}${comet}
  <path d="${poly(disc)}" fill="${MOONLIGHT}"/>${seas}${stroke(disc, pressure(6, 2.4), PAPER)}
  ${range(FAR, 31, mix(NIGHT, SAGE, 0.2), 3)}
  ${range(near, 35, STONE, 4.2)}
  ${stroke(snow, thin(44, 2.4), MOONLIGHT)}
  ${flag}
  ${lit}${tent}${door}${pole}${guy}${peg}${pack}
  ${logs}${flame(1.4, EMBER, 103)}${flame(0.74, PINK, 105)}${sparks}
  ${legs}${arm}
  <g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">
    ${mascot({ line: 3.2 * u, seed: 11, paper: DUSK, ink: MOONLIGHT })}
    ${bucketHat(u, DUSK, MOONLIGHT)}
  </g>
  ${stick}${mallow}
  </g>`,
  )
}

async function main() {
  const at = process.argv.indexOf('--preview')
  const preview = at > 0 ? path.resolve(process.argv[at + 1]) : null
  const ic = sharp(icon())
  const og = sharp(card())
  const dark = sharp(night())
  // Touch icons and the jpeg have no alpha, so they get the paper behind them.
  const flat = () => ic.clone().flatten({ background: PAPER })

  if (preview) {
    await fs.mkdir(preview, { recursive: true })
    await ic.clone().png().toFile(path.join(preview, 'icon.png'))
    await og.clone().png().toFile(path.join(preview, 'og-image.png'))
    await dark.clone().png().toFile(path.join(preview, 'og-image-dark.png'))
    for (const [name, draw] of Object.entries(SECTION_ICONS))
      await sharp(draw())
        .png()
        .toFile(path.join(preview, `icon-${name}.png`))
    // The Favicon emitter's own resize on a grey tab bar, blown back up to be looked at.
    for (const s of [16, 32, 48]) {
      const small = await ic.clone().resize(s, s).png().toBuffer()
      await sharp(small)
        .flatten({ background: '#cfcfcf' })
        .resize(s * 8, s * 8, { kernel: 'nearest' })
        .toFile(path.join(preview, `favicon-${s}.png`))
    }
    return
  }

  const stat = (f: string) => path.join(ROOT, 'quartz/static', f)
  await ic.clone().png({ compressionLevel: 9 }).toFile(stat('icon.png'))
  await ic.clone().webp({ quality: 92 }).toFile(stat('icon.webp'))
  await flat().jpeg({ quality: 92 }).toFile(stat('icon.jpeg'))
  await og.clone().png({ compressionLevel: 9 }).toFile(stat('og-image.png'))
  await og.clone().webp({ quality: 92 }).toFile(stat('og-image.webp'))
  await og.clone().jpeg({ quality: 92 }).toFile(stat('og-image.jpeg'))
  await dark.clone().png({ compressionLevel: 9 }).toFile(stat('og-image-dark.png'))
  await dark.clone().webp({ quality: 92 }).toFile(stat('og-image-dark.webp'))
  for (const [name, draw] of Object.entries(SECTION_ICONS))
    await sharp(draw())
      .png({ compressionLevel: 9 })
      .toFile(stat(`icon-${name}.png`))
  for (const [f, s] of [
    ['apple-touch-icon.png', 180],
    ['android-chrome-192x192.png', 192],
    ['android-chrome-512x512.png', 512],
  ] as const)
    await flat()
      .resize(s, s)
      .png()
      .toFile(path.join(ROOT, 'content', f))
}

await main()
