import { execFileSync } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import {
  mascot,
  hand,
  pressure,
  rounded,
  poly,
  stroke,
  type Pt,
} from '../../../../quartz/util/mascot'

const root = path.resolve(import.meta.dirname, '..')
const resources = path.join(root, 'Resources')
const iconset = path.join(root, '.build', 'AppIcon.iconset')
await fs.mkdir(resources, { recursive: true })
await fs.mkdir(iconset, { recursive: true })
const ink = '#1c1b1a'
const paper = '#fffcf0'
const sage = '#cdd597'
const salmon = '#fdb2a2'
const shape = (points: readonly Pt[], fill: string, seed: number) => {
  const outline = hand(points, 4, 0.6, seed)
  return `<path d="${poly(outline)}" fill="${fill}"/>${stroke(outline, pressure(seed + 1, 10, 0.12), ink)}`
}
const cable: Pt[] = Array.from({ length: 90 }, (_, index) => {
  const t = index / 89
  const a = Math.PI * (0.1 + t * 1.42)
  return [395 + 197 * Math.cos(a), 610 + 190 * Math.sin(a)]
})
const body = `
  ${shape(rounded(553, 511, 193, 616, 36), sage, 101)}
  ${stroke(hand(cable, 4, 1.1, 103), pressure(104, 23, 0.12), ink)}
  ${stroke(hand(cable, 4, 1.1, 103), pressure(104, 12, 0.12), salmon)}
  <g transform="translate(553 480) scale(214)">${mascot({ line: 0.066, seed: 7 })}</g>
  ${shape(rounded(704, 639, 54, 116, 12), salmon, 107)}
  ${shape(rounded(704, 574, 37, 40, 4), paper, 109)}
  ${stroke(
    hand(
      [
        [695, 563],
        [695, 579],
      ],
      3,
      0.3,
      110,
    ),
    () => 5,
    ink,
  )}
  ${stroke(
    hand(
      [
        [713, 563],
        [713, 579],
      ],
      3,
      0.3,
      111,
    ),
    () => 5,
    ink,
  )}
`
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
<rect x="54" y="54" width="916" height="916" rx="204" fill="${paper}"/>
${body}</svg>`
await fs.writeFile(path.join(resources, 'app-icon.svg'), svg)
await sharp(Buffer.from(svg)).png().toFile(path.join(resources, 'app-icon.png'))
for (const size of [16, 32, 128, 256, 512]) {
  for (const scale of [1, 2]) {
    const suffix = scale === 2 ? '@2x' : ''
    await sharp(Buffer.from(svg))
      .resize(size * scale, size * scale)
      .png()
      .toFile(path.join(iconset, `icon_${size}x${size}${suffix}.png`))
  }
}
execFileSync('/usr/bin/iconutil', [
  '-c',
  'icns',
  iconset,
  '-o',
  path.join(resources, 'AppIcon.icns'),
])
console.log(path.join(resources, 'AppIcon.icns'))
