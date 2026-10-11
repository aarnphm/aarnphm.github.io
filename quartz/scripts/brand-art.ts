// Rasterises the garden's mascot (quartz/util/mascot.ts) into the site icon and the default social
// card, where it climbs a mountain range.
//
//   pnpm exec tsx quartz/scripts/brand-art.ts              writes the static assets
//   pnpm exec tsx quartz/scripts/brand-art.ts --preview D  writes previews (and favicon sizes) to D

import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import {
  densify,
  hand,
  loop,
  mascot,
  PAPER,
  PINK,
  poly,
  pressure,
  rounded,
  SAGE,
  stroke,
  type Pt,
} from '../util/mascot'

const ROOT = path.resolve(import.meta.dirname, '../..')

const svg = (w: number, h: number, body: string) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${body}</svg>`,
  )

const icon = () =>
  svg(
    1024,
    1024,
    `<g transform="translate(512 512) scale(410)">${mascot({ line: 0.055, seed: 7 })}</g>`,
  )

// A closed shape in one colour inside a pen line.
const shape = (pts: readonly Pt[], fill: string, width: (t: number) => number) =>
  `<path d="${poly(pts)}" fill="${fill}"/>${stroke(pts, width)}`

// The card: the mascot climbs the long right shoulder of a peak, a pink pack and a sage bedroll on
// its back and a bucket hat on its head, toward a flag on the summit. Its
// eyes already look left, so the slope rises to the left. A sage range stands behind with the pink
// sun coming up between its peaks. The whole scene sits 80 lower than drawn, for sky.
function card() {
  const W = 1200
  const H = 630
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
  const far: Pt[] = [
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
  // The near ridge's height at x, before the hand wobble, for setting feet on it.
  const ridge = (x: number) => {
    for (let i = 0; i < near.length - 1; i++) {
      const [ax, ay] = near[i]
      const [bx, by] = near[i + 1]
      if (x >= ax && x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
    }
    return H
  }
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
  const crown = shape(
    hand(
      [
        [-0.62, -0.78],
        [-0.52, -1.32],
        [-0.12, -1.52],
        [0.3, -1.48],
        [0.6, -1.24],
        [0.66, -0.78],
        [-0.62, -0.78],
      ],
      0.04,
      0.015,
      67,
    ),
    PAPER,
    pressure(68, 2.6 * u, 0.35),
  )
  const band = stroke(
    hand(
      [
        [-0.58, -0.9],
        [0.62, -0.9],
      ],
      0.04,
      0.01,
      69,
    ),
    () => 4 * u,
    PINK,
  )
  const brim = stroke(
    hand(
      [
        [-1.08, -0.7],
        [-0.7, -0.8],
        [0, -0.82],
        [0.7, -0.8],
        [1.08, -0.68],
      ],
      0.04,
      0.015,
      70,
    ),
    pressure(71, 3.4 * u, 0.4),
  )

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
  ${range(far, 31, SAGE, 3)}
  ${range(near, 33, PAPER, 4.2)}
  ${stroke(snow, thin(44, 2.4))}
  ${flag}
  ${birds}
  ${legs}
  <g transform="translate(${bx} ${by}) rotate(${(tilt * 180) / Math.PI}) scale(${r})">
    ${pack}${roll}
    ${mascot({ line: 3.2 * u, seed: 11 })}
    ${crown}${band}${brim}
  </g>
  </g>`,
  )
}

async function main() {
  const at = process.argv.indexOf('--preview')
  const preview = at > 0 ? path.resolve(process.argv[at + 1]) : null
  const ic = sharp(icon())
  const og = sharp(card())
  // Touch icons and the jpeg have no alpha, so they get the paper behind them.
  const flat = () => ic.clone().flatten({ background: PAPER })

  if (preview) {
    await fs.mkdir(preview, { recursive: true })
    await ic.clone().png().toFile(path.join(preview, 'icon.png'))
    await og.clone().png().toFile(path.join(preview, 'og-image.png'))
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
