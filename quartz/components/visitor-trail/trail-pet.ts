import { pixels } from './pixels'

// The garden's pet (quartz/util/mascot.ts) as the trail's folk are drawn: 2-unit pixels in scene
// units, in its walking hat, a little taller than a hiker. The body is a disc 13 pixels across under
// the brim, the dash eyes set left of centre as the mascot's are, so it faces left as drawn. The feet
// stand on the origin.
//
// pet.inline.ts copies the site pet's data-* attributes onto the sprite's group, and
// shared-trail-frame.scss shows the parts below for them as pet.scss shows the site pet's: its
// expression for data-state, and what it wears or holds in the visitor's weather.
const tones = {
  K: 'trail-pet-ink',
  P: 'trail-pet-body',
  W: 'trail-pet-hat',
  R: 'trail-pet-band',
  L: 'trail-pet-leg',
  B: 'trail-pet-boot',
  H: 'trail-pet-hand',
  S: 'trail-pet-scarf',
  s: 'trail-pet-scarf-shade',
  T: 'trail-pet-stripe',
  U: 'trail-pet-cloth',
  G: 'trail-pet-panel',
  F: 'trail-pet-flame',
  Y: 'trail-pet-glass',
  N: 'trail-pet-brass',
  D: 'trail-pet-screen',
  E: 'trail-pet-air',
}

const CELL = 2
// Cells across the sprite, and down from the hat's crown to the soles.
const WIDE = 15
const TALL = 15

// Rows drawn from cell (x, y) of the sprite.
const at = (rows: readonly string[], x = 0, y = 0) =>
  `<g transform="translate(${x * CELL} ${y * CELL})">${pixels(rows, tones, CELL)}</g>`
const part = (name: string, body: string) => `<g data-part="${name}">${body}</g>`
const frames = (name: string, drawings: string[]) =>
  part(name, drawings.map(d => `<g class="trail-pet-frame">${d}</g>`).join(''))

const hat = ['.....KKKKK.....', '....KRRRRRK....', 'KKWWWWWWWWWWWKK']
const body = [
  '..KPPPPPPPPPK..',
  '.KPPPPPPPPPPPK.',
  '.KPPPPPPPPPPPK.',
  '.KPPPPPPPPPPPK.',
  '.KPPPPPPPPPPPK.',
  '.KPPPPPPPPPPPK.',
  '..KPPPPPPPPPK..',
  '..KPPPPPPPPPK..',
  '...KPPPPPPPK...',
  '....KKKKKKK....',
]
// Standing, then the two steps of a stride. The lower row is the boot.
const legs = [
  ['.....L...L.....', '....BB..BB.....'],
  ['....L.....L....', '...BB....BB....'],
  ['......L.L......', '.....BB.BB.....'],
]
const dash = ['K...K', 'K...K']
const eyes: Record<string, string> = {
  eyes: at(dash, 4, 5),
  'eyes-up': at(dash, 4, 4),
  'eyes-down': at(dash, 4, 6),
  'eyes-wide': at(['K...K', 'K...K', 'K...K'], 4, 4),
  'eyes-happy': at(['.K...K.', 'K.K.K.K'], 3, 5),
}
const mouth = at(['KK'], 5, 8)
const shades = at(['KKKKKKK', 'KKK.KKK'], 3, 5)
// Wrapped round twice, the tail hanging down its back.
const scarf = at(
  ['KSSTSSSTSSSTK', '.KsssssssssK.', '.........TS..', '.........ST..', '.........T.T.'],
  1,
  8,
)
const puffs = [at(['E.', '.E'], -3, 6), at(['E.', '.E'], -5, 4)]
// Held in the hand on its back, tipped forward over its head; in a storm the gust blows it inside
// out.
const shaft = at(
  ['K...', '.K..', '.K..', '.K..', '..K.', '..K.', '..K.', '...K', '...K', '..KH'],
  12,
  -1,
)
const canopy = at(
  [
    '.........K.........',
    '......KKKKKKK......',
    '....KKUUUGUUUKK....',
    '..KKUUUUUGUUUUUKK..',
    '.KUUUUUUUGUUUUUUUK.',
    'KUUUUUGUUUUUGUUUUUK',
    'KK.KKK.KKKKK.KKK.KK',
  ],
  3,
  -7,
)
const inside = at(
  [
    'K.................K',
    'KU...............UK',
    '.KUU...........UUK.',
    '..KUUUG.....GUUUK..',
    '....KKUUUGUUUKK....',
    '.......KKKKK.......',
    '.........K.........',
  ],
  3,
  -7,
)
// Out in front on a short reach, in a brass frame that shows on the dark ground.
const lamp = at(['.H.', '.N.', 'NNN', 'NYN', 'NFN', 'NNN'], -1, 8)
const flames = [at(['F', 'Y'], 0, 11), at(['Y', 'F'], 0, 11)]
// The hand on its back up on the brim, and the gusts streaming past in two drawings.
const hold = at(['H.', '.K', '.K', '.K', '.K', 'K.'], 13, 1)
const gusts = [
  at(['EEEE......', '..........', '..........', '......EEEE', '..........', '..EEEEE...'], -9, 2),
  at(['..EEEE....', '..........', '..........', '....EEEE..', '..........', 'EEEEE.....'], -9, 2),
]
// Waving with the hand in front, in two drawings.
const waves = [at(['H.', 'K.', 'K.', '.K'], -1, 4), at(['H..', '.K.', '..K', '..K'], -3, 4)]
// Both hands up to catch the snow.
const reach = at(['HH...............HH', '.K...............K.', '..K.............K..'], -2, 5)
const dots = at(['....KK', '....KK', '......', '..K...', '......', 'K.....'], 13, -6)
const laptop = at(['K.....', 'KK....', 'KDK...', 'KDDK..', 'KKKKKK'], -6, 10)

// Collision radius in scene units at ground scale 1; the trail's travellers take 12.
export const TRAIL_PET_RADIUS = 11

// Scene units from the soles up to the top of its hat, or of its umbrella in the rain: where its
// speech bubble sits.
export const trailPetTop = (rain: boolean) => (rain ? TALL + 7 : TALL) * CELL

export function trailPetSprite() {
  return `<g transform="translate(${(-WIDE * CELL) / 2} ${-TALL * CELL})">${[
    part('umbrella', part('brolly-open', canopy) + part('brolly-out', inside)),
    ...legs.map(
      (rows, index) => `<g class="trail-pet-legs" data-legs="${index}">${at(rows, 0, 13)}</g>`,
    ),
    part('laptop', laptop),
    at(body, 0, 3),
    part('scarf', scarf),
    ...Object.entries(eyes).map(([name, svg]) => part(name, svg)),
    part('mouth', mouth),
    part('shades', shades),
    `<g class="trail-pet-crown">${at(hat)}</g>`,
    part('shaft', shaft),
    part('lantern', lamp + frames('flame', flames)),
    part('hold', hold),
    frames('wave', waves),
    part('reach', reach),
    frames('breath', puffs),
    frames('gust', gusts),
    part('dots', dots),
  ].join('')}</g>`
}
