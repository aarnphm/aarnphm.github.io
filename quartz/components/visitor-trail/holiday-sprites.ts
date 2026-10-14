import { holidayDetails } from './holiday-details'
import { pixels } from './pixels'

// Opus pixel grids are palette indices, not executable SVG or external assets. Ground props draw
// at the scene's 2 px cell, like the campers and their gear; animals keep their 1.5 px cell.
const pumpkin = [
  '....bv....',
  '..iiiiii..',
  '.iosoosoi.',
  'iooloolooi',
  'iolloolloi',
  'iolooooloi',
  'ioolollooi',
  '.iosoosoi.',
  '..iiiiii..',
]
const pumpkinSmall = [
  '...bv...',
  '.iiiiii.',
  'iolooloi',
  'iooooooi',
  'iolllloi',
  '.iosooi.',
  '..iiii..',
]

// A pair of grids side by side, sharing the bottom row.
function beside(left: readonly string[], right: readonly string[]) {
  const height = Math.max(left.length, right.length)
  const lift = (rows: readonly string[]) => [
    ...Array<string>(height - rows.length).fill('.'.repeat(rows[0].length)),
    ...rows,
  ]
  const [a, b] = [lift(left), lift(right)]
  return a.map((row, index) => row + b[index])
}

export const holidayArt = {
  ...holidayDetails,
  turkey: [
    '...iiiiii.............',
    '.iobobobobi...........',
    'iobpbpbpbpoi....iii...',
    'iobpbpbpbpboi..ibbbi..',
    'iobpbpbpbpboi..ibibgg.',
    'iobpbpbpbpboi..ibbrr..',
    'iobpbpbiiibpoi..ibr...',
    'iobpbpibbbiiiiiibbi...',
    'iobpbibbbbbbbbbbbbi...',
    '.iobpibbbbsbbbbbbbi...',
    '..iobibbbsbsbbbbbi....',
    '...iiibbbbsbsbbbi.....',
    '.....ibbbbbbbbbi......',
    '......iibbbbbii.......',
    '........iiiii.........',
    '.........o..o.........',
    '.........o..o.........',
    '.........oo.oo........',
  ],
  rabbit: [
    '...........i..i...',
    '..........ipiipi..',
    '..........ipiipi..',
    '..........ipiipi..',
    '..........ipiipi..',
    '.........ippppppi.',
    '........ipppppippi',
    '....iiiippppppppri',
    '...ipppppppppppii.',
    '..ippppppppppppi..',
    '.ipppppppppppppi..',
    'ippppppppppppppi..',
    '.ippsspppppppppi..',
    '.ipspppspppppppi..',
    '..isppppsppppppi..',
    '...iiiiiiiiiiii...',
    '..ippi....ippi....',
    '..iiii....iiii....',
  ],
  // The toggle's icon. The tree wears the smaller star, on its own cell grid.
  star: [
    '.....i.....',
    '....igi....',
    '....igi....',
    'iiiigggiiii',
    'igggggggggi',
    '.igggggggi.',
    '..igggggi..',
    '.igggigggi.',
    '.iggi.iggi.',
    'iggi...iggi',
    'ii.......ii',
  ],
  treeStar: ['..g..', '.ggg.', 'ggggg', '.ggg.', '.g.g.'],
  // Carved faces are `l`: dark by day, lit from inside at night.
  pumpkin,
  lanterns: beside(pumpkin, pumpkinSmall),
  ghost: [
    '..iiii..',
    '.ippppi.',
    'ippppppi',
    'ipippipi',
    'ipippipi',
    'ippppppi',
    'ippiipsi',
    'ipppppsi',
    'ippppssi',
    'ipi.ipi.',
  ],
  spider: ['i.i.i', '.iii.', 'i.i.i'],
  bat: ['i..i.i..i', 'ii.iii.ii', '.iiiiiii.', '..i...i..'],
  batGlide: ['...i.i...', '...iii...', 'iiiiiiiii', 'i.i...i.i'],
  // Victorian mourning dress: black silk, a crimson band, bone where the face should be. The hat's
  // ink rim turns pale at night, so it still reads on a dark meadow.
  topHat: ['.iii.', '.ihi.', '.kkk.', 'iiiii'],
  skeleton: [
    '..hhh..',
    '..hhh..',
    '..kkk..',
    '.hhhhh.',
    '.ipppi.',
    '.hphpi.',
    '..ipi..',
    '.hpkph.',
    'hhhphhh',
    'hhhphhh',
    'p.hhh.p',
    '..h.h..',
    '..h.h..',
    '.pp.pp.',
  ],
  // A roadside gibbet: the cage still holds its tenant, and a raven keeps the arm.
  gibbet: [
    '.........hh.......',
    '........hhhk......',
    '.bb.....hhh.......',
    'ibbbbbbbbbbbbbbbbi',
    '.bbbbbbbbbbbbbbbb.',
    '.bbb.......i......',
    '.bb.b......i......',
    '.bb..b.....i......',
    '.bb...b....i......',
    '.bb........i......',
    '.bb......iiiii....',
    '.bb.....i..p..i...',
    '.bb.....i.php.i...',
    '.bb.....i.ppp.i...',
    '.bb.....i..s..i...',
    '.bb.....is.s.si...',
    '.bb.....i.sss.i...',
    '.bb.....is.s.si...',
    '.bb.....i.sss.i...',
    '.bb.....i..s..i...',
    '.bb.....i.s.s.i...',
    '.bb.....i.s.s.i...',
    '.bb.....iiikiii...',
    '.bb.......k.......',
    '.bb...............',
    '.bb.......k.......',
    '.bb...............',
    '.bb...............',
    'ibbi.....kkkk.....',
    'bbbb....kkkkkkk...',
  ],
  // Railed plot: a tablet, a cross, and a fresh grave that something is leaving.
  graveyard: [
    'n...n...n...n...n...n...n...n...n...n',
    'nn.nnn.nnn.nnn.nnn.nnn.nnn.nnn.nnn.nn',
    'n...nhh.n...n...n...n...n...n...n...n',
    'nnnnhhhknnnnnnnnnnnnnnnnnnnnnnnnnnnnn',
    'n..hhhh.n...niiin...n...n...n...n...n',
    'n...hhh.n...nisin...n...n...n...n...n',
    'n.iiiiiin.iiiisiiii.ni..n...n...n...n',
    'nissssssi.isssssssi.iii.n...n...n...n',
    'nisshsssi.iiiisiiii.ni..n...n...n...n',
    '.ishhhssi....isi.....i.......s.s.....',
    '.isshsssi....isi.....i.......sss.....',
    '.isshsssi....isi.....ss......ssk.....',
    '.issssssi...iisii...bssbb.bbbbsbbbb..',
    '.issssssi..isssssi..bbbbbbbhhhshhhhb.',
    '.issssssi..iiiiiii..bbbbbbbhhhkhhhhb.',
    '.iiiiiiii..bbbbbbb...bbbbb.bhhhhhhb..',
    '..bbbbbb....................bbbbbb...',
  ],
  gifts: [
    '.g.g.....',
    '..g......',
    'rrgrr....',
    'ggggg....',
    'rrgrr.r..',
    'rrgrrvrvv',
    'rrgrrrrrr',
    'srgrsvrvs',
  ],
  egg: ['.iii.', 'ipppi', 'ipppi', 'ivvvi', 'ipppi', '.iii.'],
  harvest: [
    '..bv.......',
    '.iiiii..v..',
    'ioosooi.yy.',
    'iooooooiyyy',
    'ioosoooiyys',
    '.iiiiii.ss.',
  ],
  trailMap: ['ppbppbpp', 'pvbpgbpp', 'pvvvbvpp', 'ppbpvvgp', 'ppbppbpp'],
}

export type HolidaySpriteName = keyof typeof holidayArt

const tones = {
  i: 'trail-holiday-ink',
  s: 'trail-holiday-shade',
  p: 'trail-holiday-paper',
  b: 'trail-holiday-brown',
  o: 'trail-holiday-orange',
  r: 'trail-holiday-red',
  g: 'trail-holiday-gold',
  v: 'trail-holiday-green',
  y: 'trail-holiday-yellow',
  l: 'trail-holiday-lantern',
  h: 'trail-holiday-hollow',
  k: 'trail-holiday-blood',
  n: 'trail-holiday-iron',
}

// Each tone also carries its colour as a presentation attribute, which any stylesheet rule
// overrides. Sprites in the page header paint before, or without, the trail stylesheet, and an
// unstyled path fills black.
const fallback: Record<string, string> = {
  'trail-holiday-ink': '#5e5a50',
  'trail-holiday-shade': '#a8a596',
  'trail-holiday-paper': '#ebe3cf',
  'trail-holiday-brown': '#7e725f',
  'trail-holiday-orange': '#be9770',
  'trail-holiday-red': '#b17464',
  'trail-holiday-gold': '#d0b98c',
  'trail-holiday-green': '#9ca58d',
  'trail-holiday-yellow': '#cbb978',
  'trail-holiday-lantern': '#5b4632',
  'trail-holiday-hollow': '#2b2622',
  'trail-holiday-blood': '#8c2f27',
  'trail-holiday-iron': '#3b3733',
}
const painted = (rows: readonly string[], size: number) =>
  pixels(rows, tones, size).replace(/class="(trail-holiday-[a-z]+)"/g, (match, tone: string) =>
    fallback[tone] ? `${match} fill="${fallback[tone]}"` : match,
  )

export function holidaySprite(name: HolidaySpriteName, size = 1) {
  return painted(holidayArt[name], size)
}

// (headX, headY) is where the figure's face starts; the brim sits there and the crown rises above.
export function halloweenCostume(headX: number, headY: number) {
  return `<g class="trail-halloween-costume" aria-hidden="true" transform="translate(${headX - 3} ${headY - 6})">${holidaySprite('topHat', 2)}</g>`
}

export function holidayAnimalSprite(name: 'turkey' | 'rabbit', size: number) {
  const rows = holidayArt[name]
  const width = rows[0].length
  const feet = rows.slice(-2)
  const leg = (front: boolean) =>
    feet.map(row => [...row].map((pixel, x) => (x >= width / 2 === front ? pixel : '.')).join(''))
  return (
    '<g class="trail-holiday-body">' +
    painted(rows.slice(0, -2), size) +
    '</g>' +
    '<g transform="translate(0 ' +
    (rows.length - 2) * size +
    ')"><g class="trail-animal-leg-front">' +
    painted(leg(true), size) +
    '</g><g class="trail-animal-leg-back">' +
    painted(leg(false), size) +
    '</g></g>'
  )
}
