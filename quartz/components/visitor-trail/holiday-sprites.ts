import { pixels } from './pixels'

// Opus pixel grids are palette indices, not executable SVG or external assets.
export const holidayArt = {
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
  gourd: [
    '......bbv.....',
    '......b.......',
    '...iiiiiiii...',
    '..ioobooboi...',
    '.iooboooobooi.',
    'iooobooooboooi',
    'iooobooooboooi',
    'iooobooooboosi',
    '.iooboooobosi.',
    '..ioobsoobsi..',
    '...iiiiiiii...',
  ],
  pumpkin: [
    '.......bv.......',
    '.......b........',
    '...iiiiiiiiii...',
    '..iooobooboooi..',
    '.iooooooooooooi.',
    'ioooiooooooioooi',
    'iooigiooooigiooi',
    'ioooooooiooooooi',
    'ioiooooooooooioi',
    'iooigggoggggiooi',
    '.ioooiiiiiioooi.',
    '..ioosoooosooi..',
    '...iiiiiiiiii...',
  ],
  ghost: [
    '....iiiiii....',
    '...ippppppi...',
    '..ippppppppi..',
    '.ippppppppppi.',
    '.ippiippiippi.',
    '.ippiippiippi.',
    '.ippppppppppi.',
    '.ipppppppppsi.',
    '.ippppppppppi.',
    '.ipppppppppsi.',
    '.ippppppppppi.',
    '.ipppppppppsi.',
    '.ipppppppppsi.',
    '.ippiippiippi.',
    '..ii..ii..ii..',
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
  egg: [
    '...iiii...',
    '..ippppi..',
    '.ippppppi.',
    '.ippppppi.',
    'ippppppppi',
    'ippppppppi',
    'ivvvvvvvvi',
    'ippppppppi',
    'ipppppppsi',
    '.ipppppsi.',
    '..ippssi..',
    '...iiii...',
  ],
  gift: [
    '...ii....ii...',
    '..irri..irri..',
    '..irrriirrri..',
    '.iiiiirriiiii.',
    '.ivvvvrrvvvvi.',
    '.ivvvvrrvvvvi.',
    '.irrrrrrrrrri.',
    '.ivvvvrrvvvvi.',
    '.ivvvvrrvvvsi.',
    '.ivvvvrrvvvsi.',
    '.iiiiiiiiiiii.',
  ],
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
  wreath: [
    '.....iiiiii.....',
    '...iivvvvvvii...',
    '..ivvrvvvvvvvi..',
    '.ivvvviiiivvvvi.',
    '.ivvvi....ivvvi.',
    'ivvvi......ivvri',
    'ivvi........ivvi',
    'ivri........ivvi',
    'ivvi........ivvi',
    'ivvvi......ivvvi',
    '.ivvvi....ivrvi.',
    '.ivvvviiiivvvvi.',
    '..ivvvvrrvvvvi..',
    '...iivrrrrvii...',
    '.....ir..ri.....',
  ],
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
}

export function holidaySprite(name: HolidaySpriteName, size = 1) {
  return pixels(holidayArt[name], tones, size)
}

export function holidayAnimalSprite(name: 'turkey' | 'rabbit', size: number) {
  const rows = holidayArt[name]
  const width = rows[0].length
  const feet = rows.slice(-2)
  const leg = (front: boolean) =>
    feet.map(row => [...row].map((pixel, x) => (x >= width / 2 === front ? pixel : '.')).join(''))
  return (
    '<g class="trail-holiday-body">' +
    pixels(rows.slice(0, -2), tones, size) +
    '</g>' +
    '<g transform="translate(0 ' +
    (rows.length - 2) * size +
    ')"><g class="trail-animal-leg-front">' +
    pixels(leg(true), tones, size) +
    '</g><g class="trail-animal-leg-back">' +
    pixels(leg(false), tones, size) +
    '</g></g>'
  )
}
