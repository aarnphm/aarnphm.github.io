import { pixels } from './pixels'
import { trailRandom } from './random'

const body = [
  '...WWWWWWWW.....',
  '..WWPPWWWWWW....',
  '.WWPPPPWWWPPW...',
  '.WWWPPWWWWPPWW..',
  '.WWWWWWWWWWWWW..',
  '..L.WWNNWW.L....',
  '..L........L....',
  '..D........D....',
]
const head = ['..H..H.', '.WPWWP.', '..WEWW.', '..WWWW.', '...NNNN', '....NN.']
const tones = {
  W: 'trail-cow-coat',
  P: 'trail-cow-patch',
  L: 'trail-cow-leg',
  D: 'trail-cow-hoof',
  N: 'trail-cow-nose',
  E: 'trail-cow-eye',
  H: 'trail-cow-horn',
}

function cow(size: number, grazing: boolean) {
  return `<path class="trail-cow-shadow" d="M${size} ${size * 8}h${size * 14}v${size}h${-size * 14}Z"/>
    <path class="trail-cow-tail" d="M0 ${size * 2}h${size}v${size * 4}h${-size}ZM${-size} ${size * 6}h${size * 2}v${size}h${-size * 2}Z"/>
    ${pixels(body, tones, size)}
    <g transform="translate(${size * 12} ${grazing ? size * 2 : -size * 2})">${pixels(head, tones, size)}</g>`
}

export function trailCows(width: number, camp: { x: number; y: number }, seed: number) {
  const random = trailRandom(seed ^ 0x243f6a88)
  const left = Math.max(10, camp.x - (width < 700 ? 148 : 174))
  return [
    { x: left + 28, y: camp.y + 24, size: 2, grazing: true, russet: false },
    { x: left, y: camp.y + 52, size: 2, grazing: false, russet: true },
    { x: left + 56, y: camp.y + 55, size: 1, grazing: true, russet: false },
  ]
    .map(({ x, y, size, grazing, russet }, index) => {
      const offset = Math.round(random() * 5)
      return `<g class="trail-cow${russet ? ' trail-cow-russet' : ''}" transform="translate(${x + offset} ${y})" style="--trail-cow-delay:${index * -2700}ms">${cow(size, grazing)}</g>`
    })
    .join('')
}
