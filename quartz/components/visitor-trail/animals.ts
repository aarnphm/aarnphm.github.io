import { pixels } from './pixels'

export type TrailAnimalKind = 'cow' | 'dog' | 'cat' | 'goose' | 'bear'

const dog = [
  '.........DD....',
  '........DDDD...',
  '..CCCCCCDDCEE..',
  '.CCCCCCCDCCCNN.',
  '.CCCCCCCCCC....',
]
const cat = ['..C...C..', '..CCCCC..', '..CEEEC..', '...CCC...', '...CCCCC.', '...CCCCC.']
const goose = [
  '........HH..',
  '........HEBB',
  '........HN..',
  '........HN..',
  '..WWWW..HN..',
  '.WWDDWWWNN..',
  '.WWDDWWWW...',
  '..WWWWWW....',
]
const tones = {
  C: 'trail-animal-coat',
  D: 'trail-animal-patch',
  E: 'trail-animal-eye',
  N: 'trail-animal-nose',
  B: 'trail-animal-paw',
}

export function dogPixels(size: number) {
  return `<path class="trail-animal-shadow" d="M0 ${size * 8}h${size * 14}v2H0Z"/>
    <g class="trail-dog-tail">${pixels(['C...', 'C...', '.C..', '..C.'], tones, size)}</g>
    ${pixels(dog, tones, size)}
    <g transform="translate(0 ${size * 5})">
      <g class="trail-animal-leg-front">${pixels(['........C.C....', '........C.C....', '........B.B....'], tones, size)}</g>
      <g class="trail-animal-leg-back">${pixels(['..C..C.........', '..C..C.........', '..B..B.........'], tones, size)}</g>
    </g>`
}

export function catPixels(size: number) {
  return `<path class="trail-animal-shadow" d="M${size * 2} ${size * 8}h${size * 8}v2h-${size * 8}Z"/>
    <g class="trail-cat-tail" transform="translate(${size * 7} ${size * 2})">${pixels(['..CC', '...C', '...C', '..C.', '.C..'], tones, size)}</g>
    ${pixels(cat, tones, size)}
    <g transform="translate(0 ${size * 6})">
      <g class="trail-animal-leg-front">${pixels(['......C..', '.....BB..'], tones, size)}</g>
      <g class="trail-animal-leg-back">${pixels(['...C.....', '..BB.....'], tones, size)}</g>
    </g>`
}

export function goosePixels(size: number) {
  return `<path class="trail-goose-wake" d="M-4 ${size * 8}h${size * 14}v1H-4Z"/>
    ${pixels(
      goose,
      {
        W: 'trail-goose-feather',
        D: 'trail-goose-wing',
        H: 'trail-goose-neck',
        N: 'trail-goose-cheek',
        B: 'trail-goose-bill',
        E: 'trail-animal-eye',
      },
      size,
    )}`
}

export function bearPixels(size: number) {
  const tones = {
    C: 'trail-bear-coat',
    D: 'trail-bear-shade',
    E: 'trail-animal-eye',
    N: 'trail-bear-muzzle',
    B: 'trail-bear-paw',
  }
  return `<path class="trail-animal-shadow" d="M${size} ${size * 8}h${size * 16}v2H${size}Z"/>
    ${pixels(['...........CC..CC..', '...CCCCCC..CCCCCC..', '..CCCCCCCCCCCCCE...', '.CCCCCCCCCCCCCCNN..', '.CCCDDDCCCCCCC.....', '..CDDDDDDCCCC......', '..CDDDDDDCCC.......'], tones, size)}
    <g transform="translate(0 ${size * 7})"><g class="trail-animal-leg-front">${pixels(['............C.C....', '............B.B....'], tones, size)}</g><g class="trail-animal-leg-back">${pixels(['...D..D............', '...B..B............'], tones, size)}</g></g>`
}
