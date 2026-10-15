import { pixels } from './pixels'

// The garden's pet (quartz/util/mascot.ts) as the trail's folk are drawn: 2-unit pixels in scene
// units, in its walking hat, a head shorter than a hiker. The body is a disc 9 pixels across under
// the brim, the dash eyes set left of centre as the mascot's are, so it faces left as drawn. The feet
// stand on the origin.
const tones = { K: 'trail-pet-ink', P: 'trail-pet-body', W: 'trail-pet-hat', R: 'trail-pet-band' }

const body = [
  '...KKKKK...',
  '..KRRRRRK..',
  'KKWWWWWWWKK',
  '.KPPPPPPPK.',
  '.KPPPPPPPK.',
  '.KPPPPPPPK.',
  '.KPPPPPPPK.',
  '..KPPPPPK..',
  '...KKKKK...',
]
const eyes = ['...K..K....', '...K..K....']
// Standing, then the two steps of a stride.
const legs = [
  ['...K...K...', '..KK..KK...'],
  ['..K.....K..', '.KK....KK..'],
  ['....K.K....', '...KK.KK...'],
]

// Collision radius in scene units at ground scale 1; the trail's travellers take 12.
export const TRAIL_PET_RADIUS = 9

export function trailPetSprite() {
  return `<g transform="translate(-11 -22)">
    ${pixels(body, tones, 2)}
    <g class="trail-pet-eyes" transform="translate(0 8)">${pixels(eyes, tones, 2)}</g>
    ${legs.map((rows, index) => `<g class="trail-pet-legs" data-legs="${index}" transform="translate(0 18)">${pixels(rows, tones, 2)}</g>`).join('')}
  </g>`
}
