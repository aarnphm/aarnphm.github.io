// Rasterises the garden's mascot (quartz/util/mascot.ts) into the site icon and the default social
// card, where its pixel twin climbs a mountain range, and the card's night twin, where it camps
// below the peak. The pen-drawn cards (brand-art/og-card.ts, og-night.ts) stay behind `--line`.
//
//   pnpm exec tsx quartz/scripts/brand-art.ts              writes the static assets
//   pnpm exec tsx quartz/scripts/brand-art.ts --preview D  writes previews (and favicon sizes) to D
//   pnpm exec tsx quartz/scripts/brand-art.ts --line D     writes the pen-drawn cards to D

import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import {
  baguette,
  beret,
  hand,
  lawyer,
  artist,
  librarian,
  presenter,
  reader,
  student,
  engineer,
  cinephile,
  mascot,
  PAPER,
  quill,
  pressure,
  rounded,
  stroke,
  sweatband,
} from '../util/mascot'
import { card } from './brand-art/og-card'
import { night } from './brand-art/og-night'
import { pixelCards, writePixel } from './brand-art/og-pixel'
import { shape, svg } from './brand-art/scene'

const ROOT = path.resolve(import.meta.dirname, '../..')

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

// Stream: the reader, glasses on, a dog-eared sheet held up in front with one line marked.
function readerIcon() {
  const { glasses, sheet } = reader(ICON_PEN)
  return icon('', `${glasses}${sheet}`, 340, 540)
}

// Courses: the student, mortarboard on, writing in a spiral notepad.
function studentIcon() {
  const { cap, pad, pencil } = student(ICON_PEN)
  return icon('', `${cap}${pad}${pencil}`, 330, 530)
}

// University notes: the engineer, hard hat on, an Erlenmeyer flask held up with bubbles rising.
function engineerIcon() {
  const { hardhat, flask, bubbles } = engineer(ICON_PEN)
  return icon('', `${hardhat}${flask}${bubbles}`, 340, 540)
}

// Cinema: the film nerd, thick square specs on, a striped bucket of popcorn held up.
function cinephileIcon() {
  const { specs, popcorn } = cinephile(ICON_PEN)
  return icon('', `${specs}${popcorn}`, 340, 540)
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
  stream: readerIcon,
  courses: studentIcon,
  engineering: engineerIcon,
  cinema: cinephileIcon,
} as const

async function main() {
  const flag = (name: string) => {
    const at = process.argv.indexOf(name)
    return at > 0 ? path.resolve(process.argv[at + 1]) : null
  }
  const line = flag('--line')
  if (line) {
    await fs.mkdir(line, { recursive: true })
    await sharp(card()).png().toFile(path.join(line, 'og-image.png'))
    await sharp(night()).png().toFile(path.join(line, 'og-image-dark.png'))
    return
  }
  const preview = flag('--preview')
  const ic = sharp(icon())
  const [og, dark] = await pixelCards()
  // Touch icons and the jpeg have no alpha, so they get the paper behind them.
  const flat = () => ic.clone().flatten({ background: PAPER })

  if (preview) {
    await fs.mkdir(preview, { recursive: true })
    await ic.clone().png().toFile(path.join(preview, 'icon.png'))
    await writePixel(preview)
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
  // Flat pixel runs: lossless webp is a fraction of lossy, and the jpeg keeps full chroma so the
  // pink and ink cells do not bleed.
  await og.clone().png({ compressionLevel: 9 }).toFile(stat('og-image.png'))
  await og.clone().webp({ lossless: true }).toFile(stat('og-image.webp'))
  await og.clone().jpeg({ quality: 95, chromaSubsampling: '4:4:4' }).toFile(stat('og-image.jpeg'))
  await dark.clone().png({ compressionLevel: 9 }).toFile(stat('og-image-dark.png'))
  await dark.clone().webp({ lossless: true }).toFile(stat('og-image-dark.webp'))
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
