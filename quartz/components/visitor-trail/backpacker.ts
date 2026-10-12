import { pixels } from './pixels'

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
          </g>
          ${legs.map((rows, index) => `<g class="trail-hiker-stride" data-stride="${index}" transform="translate(-4 -8)">${pixels(rows, tones, 2)}</g>`).join('')}
          <g class="trail-hiker-held-pole">
            <path class="trail-hiker-pole" d="M10-18h1V1h-1Z"/>
            <path class="trail-hiker-grip" d="M9-16h3v3H9ZM10 0h2v2h-2Z"/>
            <path class="trail-camper-skin" d="M6-12h4v2H6Z"/>
          </g>
        </g>
        <g data-hiker-pose="bend" transform="translate(-16 -20)">
          ${pixels(['..HHH.....', '..HHHH....', '..SSS.....', '...CCCC...', '..SCCCCC..', '..S.CCC...', '..S.DD....', '..S.D.D...', '....D..D..', '...BB..BB.'], tones, 2)}
        </g>
        <g data-hiker-pose="rest" transform="translate(-6 -15)">
          ${pixels(['.HH.', '.HS.', 'CCC.', 'CCCS', 'DDBB'], tones)}
        </g>
        <path class="trail-hiker-parked-pole trail-hiker-pole" d="M-24-23h1v5h1v5h1v5h1v5h1v4h-1v-4h-1v-5h-1v-5h-1v-5h-1Z"/>
      </g>
    </g>`
}
