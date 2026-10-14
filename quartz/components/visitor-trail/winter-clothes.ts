import type { Point } from './landscape'
import { pixels } from './pixels'

export function winterAccessories(head: Point, neck: Point, hands: Point[], size = 2) {
  return `<g class="trail-winter-clothes">
    <g transform="translate(${head.x} ${head.y - size})">${pixels(['.HH.', 'HHHH', 'FFFF'], { H: 'trail-winter-hat', F: 'trail-winter-cuff' }, size)}</g>
    <g transform="translate(${neck.x} ${neck.y})">${pixels(['RRRR', '..R.', '..R.'], { R: 'trail-winter-scarf' }, size)}</g>
    ${hands.map(p => `<path class="trail-winter-mitten" d="M${p.x} ${p.y}h${size * 2}v${size * 2}h-${size * 2}Z"/>`).join('')}
  </g>`
}
