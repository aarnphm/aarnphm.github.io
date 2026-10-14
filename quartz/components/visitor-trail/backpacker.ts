import { halloweenCostume } from './holiday-sprites'
import { pixels } from './pixels'
import { winterAccessories } from './winter-clothes'

const tones = {
  H: 'trail-hiker-hat',
  S: 'trail-camper-skin',
  C: 'trail-hiker-coat',
  D: 'trail-camper-trousers',
  B: 'trail-hiker-boot',
}

export function trailBackpacker() {
  const legs = [
    ['.D.D...', '.D.D...', 'D...D..', 'BB..BB.'],
    ['.D.D...', '..DD...', '..DD...', '..BBB..'],
  ]
  return `
    <g class="trail-backpacker" data-phase="waiting" data-pack="shoulder" data-stride="0">
      <g class="trail-hiker-facing">
        <g class="trail-hiker-pack" transform="translate(-12 -22)">
          ${pixels(
            ['RRRRR', '.PPPP', '.PLPP', '.PPPP', '.PPPP', '.PLPP', '.PPPP'],
            { P: 'trail-hiker-pack-cloth', L: 'trail-hiker-pack-strap', R: 'trail-hiker-bedroll' },
            2,
          )}
        </g>
        <g data-hiker-pose="walk">
          <g transform="translate(-4 -24)">
            ${pixels(['.HHH...', '.HHHH..', '.HSS...', '.HSSS..', '.CCCC..', '.CCCCS.', '.CCC.S.', '.CCC...'], tones, 2)}
            ${winterAccessories({ x: 2, y: 0 }, { x: 2, y: 8 }, [{ x: 10, y: 12 }])}
            ${halloweenCostume(4, 4)}
          </g>
          ${legs.map((rows, index) => `<g class="trail-hiker-stride" data-stride="${index}" transform="translate(-4 -8)">${pixels(rows, tones, 2)}</g>`).join('')}
          <g class="trail-hiker-held-pole">
            <path class="trail-hiker-pole" d="M10-18h1V1h-1Z"/>
            <path class="trail-hiker-grip" d="M9-16h3v3H9ZM10 0h2v2h-2Z"/>
            <path class="trail-camper-skin" d="M6-12h4v2H6Z"/>
            <path class="trail-winter-clothes trail-winter-mitten" d="M6-13h4v4H6Z"/>
          </g>
        </g>
        <g data-hiker-pose="bend" transform="translate(-16 -20)">
          ${pixels(['..HHH.....', '..HHHH....', '..SSS.....', '...CCCC...', '..SCCCCC..', '..S.CCC...', '..S.DD....', '..S.D.D...', '....D..D..', '...BB..BB.'], tones, 2)}
          ${winterAccessories({ x: 4, y: 0 }, { x: 4, y: 6 }, [{ x: 4, y: 14 }])}
          ${halloweenCostume(4, 4)}
        </g>
        <g data-hiker-pose="rest" transform="translate(-6 -15)">
          ${pixels(['.HH.', '.HS.', 'CCC.', 'CCCS', 'DDBB'], tones)}
          ${winterAccessories({ x: 0, y: 0 }, { x: 0, y: 6 }, [{ x: 9, y: 9 }], 3)}
          ${halloweenCostume(3, 1)}
        </g>
        <path class="trail-hiker-parked-pole trail-hiker-pole" d="M-24-23h1v5h1v5h1v5h1v5h1v4h-1v-4h-1v-5h-1v-5h-1v-5h-1Z"/>
      </g>
    </g>`
}
