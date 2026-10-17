import { pixels } from './pixels'

const body = [
  '...WWWWWWWW.....',
  '..WWPPWWWWWW....',
  '.WWPPPPWWWPPW...',
  '.WWWPPWWWWPPWW..',
  '.WWWWWWWWWWWWW..',
  '....WWNNWW......',
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

export function cowPixels(size: number) {
  return `<path class="trail-cow-shadow" d="M${size} ${size * 8}h${size * 14}v${size}h${-size * 14}Z"/>
    <g transform="translate(0 ${size * 5})">
      <g class="trail-animal-leg-front">${pixels(['...........L....', '...........L....', '...........D....'], tones, size)}</g>
      <g class="trail-animal-leg-back">${pixels(['..L.............', '..L.............', '..D.............'], tones, size)}</g>
    </g>
    <g class="trail-cow-body">
      <path class="trail-cow-tail" d="M0 ${size * 2}h${size}v${size * 4}h${-size}ZM${-size} ${size * 6}h${size * 2}v${size}h${-size * 2}Z"/>
      ${pixels(body, tones, size)}
      <g transform="translate(${size * 12} ${-size * 2})"><g class="trail-cow-head">${pixels(head, tones, size)}</g></g>
    </g>`
}
