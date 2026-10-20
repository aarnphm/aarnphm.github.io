import { pixels } from './pixels'

// The garden's pet (quartz/util/mascot.ts) in 2-unit pixels in scene units, as the trail's folk are
// drawn, and printed in the register of the 404's pixel pet (scripts/404-pet.ts): a round blob 14
// pixels across in an ink keyline, lit up and to the right, with dash eyes set left of centre as the
// mascot's are, so it faces left as drawn. It wears its walking hat and stands a fifth taller than
// a hiker. The feet stand on the origin.
//
// pet.inline.ts copies the site pet's data-* attributes onto the sprite's group, and
// shared-trail-frame.scss shows the parts below for them as pet.scss shows the site pet's: its
// expression for data-state, and what it wears or holds in the visitor's weather.
const tones = {
  K: 'trail-pet-ink',
  C: 'trail-pet-body',
  P: 'trail-pet-plate',
  Q: 'trail-pet-shade',
  W: 'trail-pet-hat',
  w: 'trail-pet-hat-light',
  R: 'trail-pet-band',
  L: 'trail-pet-leg',
  B: 'trail-pet-boot',
  H: 'trail-pet-hand',
  V: 'trail-pet-shell-light',
  A: 'trail-pet-shell',
  Z: 'trail-pet-shell-shade',
  J: 'trail-pet-knit-light',
  j: 'trail-pet-knit',
  z: 'trail-pet-knit-shade',
  X: 'trail-pet-rib',
  U: 'trail-pet-cloth',
  G: 'trail-pet-panel',
  F: 'trail-pet-flame',
  Y: 'trail-pet-glass',
  N: 'trail-pet-brass',
  D: 'trail-pet-screen',
  E: 'trail-pet-air',
}

const CELL = 2
// Cells across the sprite, and down from the top row to the soles. The hat's crown starts a row
// down, so it stands 17 cells, a fifth taller than a hiker.
const WIDE = 17
const TALL = 18
// The body: a disc 7 cells in radius whose centre is 9.3 cells down, its edge cut a little uneven as
// the 404's noise roughs its blob.
const R = 7
const CX = WIDE / 2
const CY = 9.3

// Rows drawn from cell (x, y) of the sprite.
const at = (rows: readonly string[], x = 0, y = 0) =>
  `<g transform="translate(${x * CELL} ${y * CELL})">${pixels(rows, tones, CELL)}</g>`
const part = (name: string, body: string) => `<g data-part="${name}">${body}</g>`
const frames = (name: string, drawings: string[]) =>
  part(name, drawings.map(d => `<g class="trail-pet-frame">${d}</g>`).join(''))

// The body printed as the 404 prints its blob (404-pixel.ts paintBlob): an ink keyline round the
// edge, a two-cell rim of light on the edges facing the sun (up and to the right), a cell of shade on
// the edges away from it, and a flat plate between.
const inBody = (i: number, j: number) => {
  const x = i + 0.5 - CX
  const y = j + 0.5 - CY
  const a = Math.atan2(y, x)
  return Math.hypot(x, y) < R + 0.2 * Math.sin(5 * a + 1.3) + 0.15 * Math.sin(7 * a + 0.5)
}
const body = Array.from({ length: TALL }, (_, j) =>
  Array.from({ length: WIDE }, (_, i) => {
    const on = (di: number, dj: number) => inBody(i + di, j + dj)
    if (!on(0, 0)) return '.'
    if (!on(-1, 0) || !on(1, 0) || !on(0, -1) || !on(0, 1)) return 'K'
    if (!on(1, -1) || !on(2, 0) || !on(0, -2) || !on(2, -2)) return 'C'
    return on(-1, 1) ? 'P' : 'Q'
  }).join(''),
)
// Facing right the sprite is mirrored, so the light would fall on its left: that way it draws the
// body mirrored, which the flip turns back to the sun's side.
const bodyMirrored = body.map(row => [...row].reverse().join(''))

// Its walking hat, in the same keyline: a crown lit on the sun's side, the band, and the brim with
// an ink edge under it where it meets the body.
const hat = [
  '......KKKKK......',
  '.....KWWWwwK.....',
  '.KKKKKRRRRRKKKKK.',
  'KWWWWWWWWWWWWwwwK',
  '.KKKKKKKKKKKKKKK.',
]
// Standing, then the two steps of a stride. The lower row is the boot.
const legs = [
  ['......L...L......', '.....BB..BB......'],
  ['.....L.....L.....', '....BB....BB.....'],
  ['.......L.L.......', '......BB.BB......'],
]
// The dash eyes as the 404's: single ink strokes, set left of centre as the mascot's are.
const dash = ['K....K', 'K....K', 'K....K']
const eyes: Record<string, string> = {
  eyes: at(dash, 5, 7),
  'eyes-up': at(dash, 5, 6),
  'eyes-down': at(dash, 5, 8),
  'eyes-wide': at(['K....K', 'K....K', 'K....K', 'K....K'], 5, 6),
  'eyes-happy': at(['.K....K.', 'K.K..K.K'], 4, 7),
}
const mouth = at(['KK'], 7, 10)
const shades = at(['KKKKKKKK', 'KKK..KKK'], 4, 7)
// What it wears in the cold, cut from the body from its chin down: each cell of cloth takes the tone
// of the body cell under it, so the cloth is lit as the body is. `cut` gives false for bare body,
// true for cloth, or a letter for a seam drawn on it.
const clothes = (
  cut: (i: number, j: number) => boolean | string,
  light: string,
  plate: string,
  shade: string,
) => {
  const cloth = (rows: readonly string[]) =>
    rows.map((row, j) =>
      [...row]
        .map((t, i) => {
          const c = t === '.' || t === 'K' ? false : cut(i, j)
          if (typeof c === 'string') return c
          return !c ? '.' : t === 'C' ? light : t === 'Q' ? shade : plate
        })
        .join(''),
    )
  return `<g class="trail-pet-blob">${at(cloth(body))}</g><g class="trail-pet-blob-mirrored">${at(cloth(bodyMirrored))}</g>`
}
// The neck opening, round the mouth.
const neck = (i: number, j: number) => j === 10 && i >= 6 && i <= 8
// Cool: a windbreaker with a drawn-in hem, its zip a seam in the shade tone: an ink zip under the
// dash eyes reads as a third mark on the face.
const windbreaker = clothes(
  (i, j) => {
    if (j < 10 || neck(i, j)) return false
    return j === 14 || (i === 7 && j > 10) ? 'Z' : true
  },
  'V',
  'A',
  'Z',
)
// Cold: a knit jumper with a ribbed collar and hem and a row of cream stitches across it.
const jumper = clothes(
  (i, j) => {
    if (j < 10 || neck(i, j)) return false
    if ((j === 10 && (i === 5 || i === 9)) || (j === 11 && i >= 6 && i <= 8) || j === 14) return 'X'
    return j === 12 && i % 2 === 0 ? 'C' : true
  },
  'J',
  'j',
  'z',
)
const puffs = [at(['E.', '.E'], -2, 9), at(['E.', '.E'], -4, 7)]
// Held in the hand on its back, tipped forward over its head; in a storm the gust blows it inside
// out.
const shaft = at(
  ['K...', '.K..', '.K..', '.K..', '..K.', '..K.', '..K.', '..K.', '...K', '...K', '..KH'],
  13,
  0,
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
  4,
  -6,
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
  4,
  -6,
)
// Out in front on a short reach, in a brass frame that shows on the dark ground.
const lamp = at(['.H.', '.N.', 'NNN', 'NYN', 'NFN', 'NNN'], -1, 10)
const flames = [at(['F', 'Y'], 0, 13), at(['Y', 'F'], 0, 13)]
// The hand on its back up on the brim, and the gusts streaming past in two drawings.
const hold = at(['H..', '.K.', '..K', '..K', '..K'], 14, 3)
const gusts = [
  at(['EEEE......', '..........', '..........', '......EEEE', '..........', '..EEEEE...'], -9, 3),
  at(['..EEEE....', '..........', '..........', '....EEEE..', '..........', 'EEEEE.....'], -9, 3),
]
// Waving with the hand in front, in two drawings.
const waves = [at(['H.', 'K.', 'K.', '.K'], 0, 6), at(['H..', '.K.', '..K', '..K'], -2, 6)]
// Both hands up to catch the snow.
const reach = at(['HH.................HH', '.K.................K.', '..K...............K..'], -2, 6)
const dots = at(['....KK', '....KK', '......', '..K...', '......', 'K.....'], 14, -4)
const laptop = at(['K.....', 'KK....', 'KDK...', 'KDDK..', 'KKKKKK'], -6, 13)

// Collision radius in scene units at ground scale 1; the trail's travellers take 12.
export const TRAIL_PET_RADIUS = 12.5

// Scene units from the soles up to the top of its hat, or of its umbrella in the rain: where its
// speech bubble sits.
export const trailPetTop = (rain: boolean) => (rain ? TALL + 6 : TALL - 1) * CELL

export function trailPetSprite() {
  return `<g transform="translate(${(-WIDE * CELL) / 2} ${-TALL * CELL})">${[
    part('umbrella', part('brolly-open', canopy) + part('brolly-out', inside)),
    ...legs.map(
      (rows, index) => `<g class="trail-pet-legs" data-legs="${index}">${at(rows, 0, 16)}</g>`,
    ),
    part('laptop', laptop),
    `<g class="trail-pet-blob">${at(body)}</g><g class="trail-pet-blob-mirrored">${at(bodyMirrored)}</g>`,
    part('windbreaker', windbreaker),
    part('jumper', jumper),
    ...Object.entries(eyes).map(([name, svg]) => part(name, svg)),
    part('mouth', mouth),
    part('shades', shades),
    `<g class="trail-pet-crown">${at(hat, 0, 1)}</g>`,
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
