// The social cards in the homepage trail's pixel grammar: the pen cards' five ranges, summit flag,
// climber and camp (og-card.ts, og-night.ts), sampled at 400 × 210 and blown up three times with no
// smoothing. The pet is the trail's pixel pet, the trees grow from its forest grammar, the fire is
// its campsite's.

import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import { trailCampsite } from '../../components/visitor-trail/campsite'
import { trailPetSprite } from '../../components/visitor-trail/trail-pet'
import { PINK, SAGE, type Pt } from '../../util/mascot/pen'
import {
  alpine,
  dawn,
  dots,
  grove,
  hills,
  meadow,
  orb,
  runs,
  sprite,
  stack,
  stars,
  style,
} from './og-pixel-paint'
import { day, night, type Palette } from './og-pixel-tones'
import { mix, ridgeAt } from './scene'

const W = 400
const H = 210

// scene.ts's ranges at a third.
const FAR: Pt[] = [
  [-7, 105],
  [27, 92],
  [57, 99],
  [87, 85],
  [110, 94],
  [150, 81],
  [187, 89],
  [213, 83],
  [240, 94],
  [267, 87],
  [293, 97],
  [320, 82],
  [353, 92],
  [380, 85],
  [407, 95],
]
const BACK: Pt[] = [
  [-7, 116],
  [20, 103],
  [50, 110],
  [77, 98],
  [107, 109],
  [133, 101],
  [157, 113],
  [187, 101],
  [217, 107],
  [247, 97],
  [277, 106],
  [300, 103],
  [327, 91],
  [353, 101],
  [380, 97],
  [407, 107],
]
const SUMMIT: Pt = [143, 76]
const FLANK: Pt[] = [[-7, 177], [23, 160], [60, 137], [83, 115], [103, 104], [123, 87], SUMMIT]
const PEAK: Pt[] = [
  ...FLANK,
  [157, 83],
  [173, 89],
  [197, 102],
  [237, 111],
  [263, 129],
  [287, 139],
  [307, 133],
  [330, 116],
  [353, 124],
  [380, 121],
  [407, 130],
]
const SHELF: Pt[] = [
  ...FLANK,
  [157, 83],
  [173, 92],
  [187, 110],
  [199, 128],
  [210, 135],
  [240, 136],
  [277, 136],
  [310, 137],
  [327, 131],
  [343, 119],
  [363, 125],
  [383, 121],
  [407, 129],
]
const NEAR_L: Pt[] = [
  [-7, 150],
  [20, 139],
  [47, 145],
  [77, 159],
  [110, 180],
  [143, 199],
  [200, 217],
]
const NEAR_R: Pt[] = [
  [187, 217],
  [233, 199],
  [260, 195],
  [290, 187],
  [320, 183],
  [350, 173],
  [380, 177],
  [407, 169],
]

const flag = (p: Palette) =>
  `<path fill="${p.pole}" d="M${SUMMIT[0]} ${SUMMIT[1] - 11}h1v12h-1Z"/><path fill="${p.flag}" d="M${SUMMIT[0] + 1} ${SUMMIT[1] - 11}h7v1h-7ZM${SUMMIT[0] + 1} ${SUMMIT[1] - 10}h5v1h-5ZM${SUMMIT[0] + 1} ${SUMMIT[1] - 9}h3v1h-3Z"/>`

// The pet, its base on the ridge, facing left as the sprite is drawn.
// The pet, its base on the ridge, facing left as the sprite is drawn, on a two-row contact shadow
// stepped at the ends as the trees' shadows are.
const pet = (x: number, y: number, shadow: string) =>
  `<path fill="${shadow}" opacity=".5" d="M${x - 12} ${y - 1}h24v1h2v1h-28v-1h2Z"/><g transform="translate(${x} ${y})">${trailPetSprite()}</g>`

// A few trees at the card's edges; the middle stays open for the peak, the pet and the trail.
const trees = {
  left: grove('near', NEAR_L, [
    [10, 'spruce', 40, 3],
    [30, 'pine', 28, 5],
    [52, 'fir', 22, 8],
  ]),
  right: grove('near', NEAR_R, [
    [350, 'fir', 24, 29],
    [372, 'spruce', 38, 31],
    [394, 'pine', 28, 33],
  ]),
}

function compose(p: Palette, ground: readonly Pt[], lightX: number, sky: string, figures: string) {
  const scene = runs(
    W,
    H,
    stack(
      meadow(NEAR_R, p.field, p.tuft, 41, H),
      meadow(NEAR_L, p.field, p.tuft, 42, H),
      alpine(ground, p.peak, SUMMIT, 14, lightX, 7),
      hills(BACK, p.back, lightX, 5),
      hills(FAR, p.far, lightX, 3),
    ),
  )
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" shape-rendering="crispEdges">${style(p)}<rect width="${W}" height="${H}" fill="${p.sky}"/>${sky}${scene}${flag(p)}${figures}${trees.left}${trees.right}</svg>`,
  )
}

export function pixelDay() {
  const p = day()
  const g = (x: number) => ridgeAt(PEAK, x)
  const sun = runs(W, H, stack(orb(300, 91, 17, p.orb), dawn(p.dawn, 24, 96)))
  const trail = dots(
    [
      [295, g(295) + 7],
      [257, g(257) + 18],
      [271, g(271) + 10],
      [237, g(237) + 8],
      [217, g(217) + 1],
    ],
    p.trail,
  )
  const pack = sprite(
    ['.PPP', 'PPPP', 'PLPP', 'PPPP', 'PPPP', '.PP.'],
    { P: PINK, L: mix(PINK, '#1c1b1a', 0.3) },
    228,
    84,
  )
  const roll = sprite(['SSSS', 'SSSS'], { S: SAGE }, 228, 80)
  // A pole planted uphill in its front hand.
  const by = Math.round(g(221))
  const pole = `<path fill="${p.pole}" d="${Array.from({ length: 6 }, (_, k) => `M${205 - 2 * k} ${by - 11 + 2 * k}h2v2h-2Z`).join('')}"/>`
  return compose(p, PEAK, 300, sun, `${trail}${roll}${pack}${pet(221, by, p.peak.fracture)}${pole}`)
}

export function pixelNight() {
  const p = night()
  const g = (x: number) => ridgeAt(SHELF, x)
  const [fx, fy] = [217, Math.round(g(217))]
  const sky = `${stars(9, x => ridgeAt(FAR, x), p.star)}${runs(
    W,
    H,
    orb(233, 63, 17, p.orb, [
      [228, 58],
      [238, 66],
      [230, 68],
    ]),
  )}`
  // The campsite's first flame drawing only; its side logs and kit are hidden by the stylesheet.
  const camp = `<g class="camp" transform="translate(${fx} ${fy})">${trailCampsite(false).replace(/<g class="trail-flame-frame" style="--phase:-(350|700)ms">.*?<\/g>/g, '')}</g>`
  const tent = sprite(
    Array.from({ length: 13 }, (_, j) => {
      const half = Math.floor(j * 0.7)
      return Array.from({ length: 19 }, (_, i) => {
        const d = i - 9
        if (Math.abs(d) > half) return '.'
        if (j > 6 && (d === 0 || d === 1)) return 'D'
        return Math.abs(d) === half || j === 0 ? 'L' : d < 0 ? 'W' : 'S'
      }).join('')
    }),
    {
      L: mix(SAGE, p.sky, 0.25),
      S: mix(p.sky, SAGE, 0.42),
      W: mix(mix(p.sky, SAGE, 0.42), '#fcc192', 0.3),
      D: mix(p.sky, '#282726', 0.5),
    },
    270,
    fy - 26,
  )
  const pack = sprite(
    ['.PP.', 'PPPP', 'PLPP', 'PPPP', 'PPPP'],
    { P: PINK, L: mix(PINK, p.sky, 0.3) },
    310,
    fy - 10,
  )
  const stick = dots(
    [
      [235, fy - 16],
      [fx + 7, fy - 28],
    ],
    '#cecdc3',
    1,
  )
  const mallow = `<path fill="#f4e6c8" d="M${fx + 5} ${fy - 31}h4v4h-4Z"/>`
  return compose(
    p,
    SHELF,
    233,
    sky,
    `${camp}${tent}${pack}${pet(249, fy, p.peak.fracture)}${stick}${mallow}`,
  )
}

// The cards as sharp pipelines at 1200 × 630, blown up three times with no smoothing.
export function pixelCards() {
  const up = (art: Buffer) =>
    sharp(art)
      .png()
      .toBuffer()
      .then(small => sharp(small).resize(1200, 630, { kernel: 'nearest' }))
  return Promise.all([up(pixelDay()), up(pixelNight())])
}

export async function writePixel(dir: string) {
  await fs.mkdir(dir, { recursive: true })
  const [og, dark] = await pixelCards()
  await fs.writeFile(path.join(dir, 'og-image.svg'), pixelDay())
  await fs.writeFile(path.join(dir, 'og-image-dark.svg'), pixelNight())
  await og.clone().png().toFile(path.join(dir, 'og-image.png'))
  await dark.clone().png().toFile(path.join(dir, 'og-image-dark.png'))
  // Slack and Twitter show the card at about 500 px wide; sharp's default lanczos stands in for them.
  await og.clone().resize(500).png().toFile(path.join(dir, 'og-image-500.png'))
  await dark.clone().resize(500).png().toFile(path.join(dir, 'og-image-dark-500.png'))
}
