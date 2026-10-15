import { halloweenCostume } from './holiday-sprites'
import { pixels } from './pixels'
import { trailSnowBank } from './winter'

// The seated posture is shared with the 404 scene, with the second camper facing the fire.
const sitter = ['.HH.', '.HS.', 'CCC.', 'CCCS', 'DDDD']
const figure = {
  H: 'trail-camper-hair',
  S: 'trail-camper-skin',
  C: 'trail-camper-coat',
  D: 'trail-camper-trousers',
}
const winterFigure = {
  ...figure,
  H: 'trail-winter-hat',
  F: 'trail-winter-cuff',
  R: 'trail-winter-scarf',
  M: 'trail-winter-mitten',
  B: 'trail-winter-boot',
  W: 'trail-winter-blanket',
}
const flames = [
  ['...R...', '...O...', '..RO...', '..OY.R.', '.ROYYO.', '.OYCYO.', 'ROYCYOR', 'OOYYYOO'],
  ['....R..', '...RO..', '...OY..', '.R.OYO.', '.ORCYO.', 'ROYCYOR', '.OYCYO.', 'OOYYYOO'],
  ['..R....', '..O.R..', '..YO...', '.ROYOR.', '.OYCOO.', 'ROYCYO.', '.OYCCOR', 'OOYYYOO'],
]
const fire = {
  R: 'trail-flame-tip',
  O: 'trail-flame-orange',
  Y: 'trail-flame-gold',
  C: 'trail-flame-core',
}

export function campFigure(pose: 'walk' | 'build' | 'sit', packed = false) {
  const rows =
    pose === 'sit'
      ? sitter
      : pose === 'build'
        ? ['..HH...', '..HS...', '.CCC...', 'CCCSS..', '.DD..S.', '.DD....', '.D.D...']
        : ['.HH.', '.HS.', 'CCC.', 'CCCS', '.DD.', 'D.D.']
  const winterRows =
    pose === 'sit'
      ? ['.H..', 'HHH.', 'FFF.', '.HS.', 'CRRR', 'CCCM', 'WWWW', 'WBBB']
      : pose === 'build'
        ? [
            '..H....',
            '.HHH...',
            '.FFF...',
            '..HS...',
            '.CRRR..',
            'CCCMR..',
            '.DD..M.',
            '.DD....',
            '.B.BB..',
          ]
        : ['.H..', 'HHH.', 'FFF.', '.HS.', 'CRRR', 'CCCM', '.DD.', 'B.BB']
  return (
    `<g class="trail-fair-weather-clothes">${pixels(rows, figure, 2)}</g><g class="trail-winter-clothes" transform="translate(0 ${pose === 'sit' ? -6 : -4})">${pixels(winterRows, winterFigure, 2)}</g>` +
    halloweenCostume(pose === 'build' ? 4 : 2, 0) +
    (packed
      ? `<g class="trail-camp-pack" transform="translate(-5 1)">${pixels(['RRRR', 'PPPP', 'PLPP', 'PPPP', 'PLPP'], { R: 'trail-hiker-bedroll', P: 'trail-hiker-pack-cloth', L: 'trail-hiker-pack-strap' }, 2)}</g><path class="trail-hiker-pole" d="M11 5h1v10h-1Z"/>`
      : '')
  )
}

export function trailCampsite(seated = true) {
  const glow = `trail-firelight-${crypto.randomUUID()}`
  return `
    <defs><radialGradient id="${glow}" class="trail-firelight-gradient">
      <stop offset="0" stop-color="#f2bd72" stop-opacity="0.48"/>
      <stop offset="0.35" stop-color="#da702c" stop-opacity="0.22"/>
      <stop offset="1" stop-color="#da702c" stop-opacity="0"/>
    </radialGradient></defs>
    <g class="trail-camp-drift" transform="translate(-60 -8)">${trailSnowBank(0x29c763, 76, 25)}</g>
    <g class="trail-camp-drift" transform="translate(57 12)">${trailSnowBank(0x741a62, 60, 21)}</g>
    <path class="trail-fire-thaw" d="M-28-5h56v3h10v4h7v6h-7v4H28v3h-56v-3h-10V8h-7V2h7v-4h10Z"/>
    <g class="trail-firelight" aria-hidden="true">
      <ellipse class="trail-firelight-halo" cx="0" cy="-9" rx="68" ry="48" fill="url(#${glow})"/>
      <ellipse class="trail-firelight-pool" cx="0" cy="7" rx="84" ry="30" fill="url(#${glow})"/>
    </g>
    <path class="trail-hearth" d="M-26-1h52v3h6v3h-6v3h-52v-3h-6v-3h6Z"/>
    <path class="trail-camp-shadow" d="M-42 3h19v3h-19ZM23 3h19v3H23Z"/>
    <path class="trail-log" d="M-39 2h9v4h-9ZM30 2h9v4h-9Z"/>
    <g class="trail-winter-kit">
      <path class="trail-camp-snow" d="M-44 0h12v2h-14v-1h2ZM33 0h10v1h3v2H33Z"/>
      <g transform="translate(-57 7)">${pixels(['.SSSSSS.', 'SSSSSSSS', '.LLLLLL.', 'LLLLLLLL', '.LLLLLL.'], { S: 'trail-camp-snow', L: 'trail-log' }, 2)}</g>
      <g transform="translate(38 8)">${pixels(['.WWWWWW.', 'WWRRWWRW', 'WWRRWWRW', '.WWWWWW.'], { W: 'trail-winter-blanket', R: 'trail-winter-scarf' }, 2)}</g>
      <path class="trail-kettle" d="M15-9h5v1h-5ZM14-8h1v3h-1ZM20-8h1v3h-1ZM13-5h9v6h-9ZM22-4h3v2h-3ZM12-6h11v1H12Z"/>
      <path class="trail-winter-scarf" d="M28 6h4v9h-4Z"/>
      <path class="trail-winter-boot" d="M28 4h4v2h-4ZM28 9h4v1h-4Z"/>
    </g>
    ${
      seated
        ? `<g class="trail-camper trail-camper-left" transform="translate(-39 -12)">
      <g transform="scale(1.5)">${campFigure('sit')}</g>
    </g>
    <g class="trail-camper trail-camper-right" transform="translate(39 -12) scale(-1 1)">
      <g transform="scale(1.5)">${campFigure('sit')}</g>
    </g>`
        : ''
    }
    <g transform="translate(-13.5 0)">
      ${pixels(['BB....B..', '..BBBB...', '...DDDD..', '.BBB..BB.'], {
        B: 'trail-log',
        D: 'trail-log-lit',
      })}
    </g>
    <g class="trail-fire"><g transform="translate(-10.5 -24)">
      ${flames
        .map(
          (rows, frame) =>
            `<g class="trail-flame-frame" style="--phase:${frame * -350}ms">${pixels(rows, fire)}</g>`,
        )
        .join('')}
    </g>
    <path class="trail-ember" d="M2-27h2v2H2Z"/>
    <path class="trail-ember trail-ember-late" d="M-5-22h2v2h-2Z"/>
    <path class="trail-smoke" d="M-2-33h3v3h-3Z"/>
    <path class="trail-smoke trail-smoke-late" d="M4-40h3v3H4Z"/>
    </g>
    <g class="trail-smoulder" aria-hidden="true">
      ${[
        [-3, -8, 6],
        [0, -15, 4],
        [-3, -21, 6],
        [1, -28, 4],
        [-2, -35, 5],
        [2, -42, 3],
      ]
        .map(
          ([x, y, size], index) =>
            `<path class="trail-smoulder-puff" style="--smoulder-delay:${(index * -1).toFixed(1)}s" d="M${x} ${y}h${size}v${size}h${-size}Z"/>`,
        )
        .join('')}
    </g>
    <g class="trail-rocks">
      <path d="M-18 0h5v3h-5ZM13 0h5v3h-5ZM-10 9h5v3h-5ZM5 9h5v3H5Z"/>
    </g>
  `
}
