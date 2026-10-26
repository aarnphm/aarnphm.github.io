// Draws the harness's page icon: the garden's blob holding a page whose text has become a grid of
// vector dots. Writes icon.svg next to this file.
//
//   pnpm exec tsx harness/embeddings/icon.ts

import fs from 'node:fs/promises'
import path from 'node:path'
import { mascot } from '../../quartz/util/mascot'
import {
  filled,
  hand,
  INK,
  PAPER,
  PINK,
  pressure,
  rounded,
  SAGE,
  stroke,
} from '../../quartz/util/mascot/pen'

const PEN = 0.05

// The page leans in front of the blob's lower right, turned a few degrees.
const page = filled(
  hand(rounded(0.66, 0.44, 0.84, 0.96, 0.07), 0.02, 0.005, 31),
  PAPER,
  pressure(32, PEN, 0.2),
  INK,
)
const dots = [0, 1, 2]
  .flatMap(r =>
    [0, 1, 2].map(c => {
      const fill = [PINK, SAGE, INK][(r + c) % 3]
      return `<circle cx="${(0.42 + c * 0.24).toFixed(2)}" cy="${(0.2 + r * 0.24).toFixed(2)}" r="0.085" style="fill:${fill}"/>`
    }),
  )
  .join('')
// A thumb over the page's left edge, so the blob holds it.
const thumb = stroke(
  hand(
    [
      [0.2, 0.5],
      [0.34, 0.46],
    ],
    0.02,
    0.004,
    33,
  ),
  () => PEN * 2.2,
  INK,
)

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.3 -1.3 2.76 2.76">${mascot({ line: PEN, seed: 7 })}<g transform="rotate(6 0.66 0.44)">${page}${dots}</g>${thumb}</svg>\n`
await fs.writeFile(path.join(import.meta.dirname, 'icon.svg'), svg)
