import { halloweenCostume } from './holiday-sprites'
import { pixels } from './pixels'

const tones = {
  H: 'trail-hiker-hat',
  S: 'trail-camper-skin',
  C: 'trail-camper-coat',
  D: 'trail-camper-trousers',
  B: 'trail-hiker-boot',
  P: 'trail-hiker-pack-cloth',
}

export function trailMapReader(index: number) {
  const strides = [
    ['.D.D...', '.D.D...', 'D...D..', 'BB..BB.'],
    ['.D.D...', '..DD...', '..DD...', '..BBB..'],
  ]
  return `<g class="trail-map-reader trail-reader-${index}" data-reader="${index}" data-phase="waiting" data-stride="0">
    <g class="trail-reader-facing">
      <g class="trail-reader-pack" transform="translate(-12 -23)">
        ${pixels(['.PPPP', 'PPPPP', 'PLPPP', 'PPPPP', 'PPPPP', 'PLPPP'], { P: 'trail-hiker-pack-cloth', L: 'trail-hiker-pack-strap' }, 2)}
      </g>
      <g data-reader-pose="walk">
        <g transform="translate(-4 -26)">
          <g class="trail-map-reader-head">${pixels(['.HHHH..', '.HHHHH.', '.HSS...', '.SSSS..'], tones, 2)}</g>
          <g transform="translate(0 8)">${pixels(['.CCCC..', '.CCCCS.', '.CCC.S.', '.CCC...'], tones, 2)}</g>
          ${halloweenCostume(4, 4)}
        </g>
        ${strides.map((rows, stride) => `<g class="trail-reader-stride" data-stride="${stride}" transform="translate(-4 -10)">${pixels(rows, tones, 2)}</g>`).join('')}
        <g class="trail-reader-pole">
          <path class="trail-hiker-pole" d="M10-19h1V1h-1Z"/>
          <path class="trail-camper-skin" d="M6-14h4v2H6Z"/>
          <path class="trail-hiker-grip" d="M9-17h3v3H9Z"/>
        </g>
        <path class="trail-reader-folded-map trail-map-paper" d="M0-13h4v6H0Z"/>
        <path class="trail-map-fold" d="M2-13h1v6H2Z"/>
      </g>
      <g data-reader-pose="read">
        <g transform="translate(-5 -28)">
          <g class="trail-map-reader-head">${pixels(['.HHHH.', '.HHHHH', '.HSS..', '..SSS.'], tones, 2)}</g>
          <g transform="translate(0 8)">${pixels(['.CCCC...', '.CCCC.S.', '.CCCSSS.', '.CCC....', '.DDD....', '.D.D....', '.D..D...', 'BB..BB..'], tones, 2)}</g>
          ${halloweenCostume(4, 4)}
        </g>
        <path class="trail-hiker-pole" d="M-10-22h1V0h-1Z"/>
        <g class="trail-reader-map" transform="translate(3 -19)">
          <path class="trail-map-edge" d="M0 0h5v1h6V0h5v12h-5v1H5v-1H0Z"/>
          <path class="trail-map-paper" d="M1 1h3v1h8V1h3v10h-4v1H5v-1H1Z"/>
          <path class="trail-map-fold" d="M5 2h1v9H5ZM10 2h1v9h-1Z"/>
          <path class="trail-map-route" d="M2 8h4V6h4V4h3v1h-2v2H7v2H2Z"/>
        </g>
      </g>
    </g>
  </g>`
}
