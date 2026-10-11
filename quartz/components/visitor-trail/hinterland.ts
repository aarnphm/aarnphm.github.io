import { pixels } from './pixels'
import { trailRandom } from './random'

const ground = [
  '........GGGGGGGGG........',
  '.....GGGGGGGGGGGGGG......',
  '...GGGGGGGGGGGGGGGGGG....',
  '..GGGGGGGGGGGGGGGGGGGG...',
  'GGGGGGGGGGGGGGGGGGGGGGG..',
  '..EEEEGGGGGGGGGGGGGGGGGGG',
  '....EEEEEEEEGGGGGGGGGGG..',
  '.......EEEEEEEEEEEEE.....',
  '..........EEEEEE.........',
]
const grass = ['..L..', '.LLL.', 'L.D.L', 'L.D.L', '.DDD.']
const stone = ['.RR.', 'RRRR', '.SS.']

export function trailHinterland(width: number, camp: { x: number; y: number }, seed: number) {
  const random = trailRandom(seed ^ 0xa511e9b3)
  const size = width < 700 ? 5 : 7
  const groundWidth = ground[0].length * size
  const groundLeft = Math.max(6, Math.min(width - groundWidth - 6, camp.x - groundWidth / 2))
  const details = Array.from({ length: 11 }, (_, index) => {
    const x = groundLeft + 8 + random() * (groundWidth - 16)
    const y = camp.y + 26 + random() * 25
    return `<g transform="translate(${Math.round(x)} ${Math.round(y)})">${
      index % 3 === 0
        ? pixels(stone, { R: 'trail-upland-stone', S: 'trail-upland-edge' }, 2)
        : pixels(grass, { L: 'trail-tree-light', D: 'trail-tree-dark' }, 1)
    }</g>`
  }).join('')
  return `<g class="trail-upland">
    <g transform="translate(${groundLeft} ${camp.y + 15})">${pixels(ground, { G: 'trail-upland-ground', E: 'trail-upland-edge' }, size)}</g>
    ${details}
  </g>`
}
