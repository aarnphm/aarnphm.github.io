import { execFileSync } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import {
  hand,
  mascot,
  poly,
  pressure,
  rounded,
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
  return `<path d="${poly(outline)}" fill="${fill}"/>${stroke(outline, pressure(seed + 1, 15, 0.12), ink)}`
}
const headband: Pt[] = Array.from({ length: 100 }, (_, index) => {
  const angle = Math.PI + (index / 99) * Math.PI
  return [512 + 277 * Math.cos(angle), 490 + 271 * Math.sin(angle)]
})
const body = `
  <path d="M762 568 C842 716 718 842 570 825 L526 825" fill="none" stroke="${ink}" stroke-width="24" stroke-linecap="round"/>
  <path d="M762 568 C842 716 718 842 570 825 L526 825" fill="none" stroke="${salmon}" stroke-width="12" stroke-linecap="round"/>
  ${stroke(hand(headband, 4, 0.7, 211), pressure(212, 49, 0.08), ink)}
  ${stroke(hand(headband, 4, 0.7, 211), pressure(212, 23, 0.08), sage)}
  <g transform="translate(512 508) scale(238)">${mascot({ line: 0.078, seed: 7 })}</g>
  ${shape(rounded(259, 520, 91, 194, 37), sage, 213)}
  ${shape(rounded(765, 520, 91, 194, 37), sage, 215)}
  ${shape(rounded(499, 825, 77, 42, 9), salmon, 217)}
  <path d="M460 825 H429" fill="none" stroke="${ink}" stroke-width="17" stroke-linecap="round"/>
`
const document = (content: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">${content}</svg>`
const svg = document(
  `<rect x="54" y="54" width="916" height="916" rx="204" fill="${paper}"/>${body}`,
)
await fs.writeFile(path.join(resources, 'app-icon.svg'), svg)
await fs.writeFile(path.join(resources, 'logo.svg'), document(body))
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

for (const paused of [false, true]) {
  const name = paused ? 'MenuIconPaused' : 'MenuIcon'
  const mark = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18">
  <g fill="none" stroke="black" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2.75 10 V8.25 A6.25 6.25 0 0 1 15.25 8.25 V10"/>
    <rect x="1.5" y="8" width="3.5" height="6.5" rx="1.4"/>
    <rect x="13" y="8" width="3.5" height="6.5" rx="1.4"/>
    ${paused ? '<path d="M7.5 8.5 V12.5 M10.5 8.5 V12.5"/>' : '<path d="M7.5 9 V10.5 M10.5 9 V10.5 M14.75 14.5 Q14.75 16.5 11.5 16.5 H10"/>'}
  </g></svg>`
  await fs.writeFile(path.join(resources, `${name}.svg`), mark)
  for (const scale of [1, 2]) {
    await sharp(Buffer.from(mark))
      .resize(18 * scale, 18 * scale)
      .png()
      .toFile(path.join(resources, `${name}${scale === 2 ? '@2x' : ''}.png`))
  }
}
console.log(path.join(resources, 'AppIcon.icns'))
