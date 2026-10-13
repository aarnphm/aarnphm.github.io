import { halloweenCostume } from './holiday-sprites'
import { pixels } from './pixels'

// The seated posture is shared with the 404 scene, with the second camper facing the fire.
const sitter = ['.HH.', '.HS.', 'CCC.', 'CCCS', 'DDDD']
const figure = {
  H: 'trail-camper-hair',
  S: 'trail-camper-skin',
  C: 'trail-camper-coat',
  D: 'trail-camper-trousers',
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
  return (
    pixels(rows, figure, 2) +
    halloweenCostume(pose === 'build' ? 4 : 2, 0) +
    (packed
      ? `<g class="trail-camp-pack" transform="translate(-5 1)">${pixels(['RRRR', 'PPPP', 'PLPP', 'PPPP', 'PLPP'], { R: 'trail-hiker-bedroll', P: 'trail-hiker-pack-cloth', L: 'trail-hiker-pack-strap' }, 2)}</g><path class="trail-hiker-pole" d="M11 5h1v10h-1Z"/>`
      : '')
  )
}

export function trailCampsite(seated = true) {
  const glow = `trail-firelight-${crypto.randomUUID()}`
  return `
    <defs><radialGradient id="${glow}">
      <stop offset="0" stop-color="#f2bd72" stop-opacity="0.48"/>
      <stop offset="0.35" stop-color="#da702c" stop-opacity="0.22"/>
      <stop offset="1" stop-color="#da702c" stop-opacity="0"/>
    </radialGradient></defs>
    <g class="trail-firelight" aria-hidden="true">
      <ellipse class="trail-firelight-halo" cx="0" cy="-9" rx="68" ry="48" fill="url(#${glow})"/>
      <ellipse class="trail-firelight-pool" cx="0" cy="7" rx="84" ry="30" fill="url(#${glow})"/>
    </g>
    <path class="trail-hearth" d="M-26-1h52v3h6v3h-6v3h-52v-3h-6v-3h6Z"/>
    <path class="trail-camp-shadow" d="M-42 3h19v3h-19ZM23 3h19v3H23Z"/>
    <path class="trail-log" d="M-39 2h9v4h-9ZM30 2h9v4h-9Z"/>
    ${
      seated
        ? `<g class="trail-camper trail-camper-left" transform="translate(-39 -12)">
      ${pixels(sitter, figure)}
    </g>
    <g class="trail-camper trail-camper-right" transform="translate(39 -12) scale(-1 1)">
      ${pixels(sitter, figure)}
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
    <g class="trail-rocks">
      <path d="M-18 0h5v3h-5ZM13 0h5v3h-5ZM-10 9h5v3h-5ZM5 9h5v3H5Z"/>
    </g>
  `
}
