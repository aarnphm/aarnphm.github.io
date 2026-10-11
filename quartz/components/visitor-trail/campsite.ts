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

export function trailCampsite() {
  return `
    <path class="trail-hearth" d="M-26-1h52v3h6v3h-6v3h-52v-3h-6v-3h6Z"/>
    <path class="trail-camp-shadow" d="M-42 3h19v3h-19ZM23 3h19v3H23Z"/>
    <path class="trail-log" d="M-39 2h9v4h-9ZM30 2h9v4h-9Z"/>
    <g class="trail-camper trail-camper-left" transform="translate(-39 -12)">
      ${pixels(sitter, figure)}
    </g>
    <g class="trail-camper trail-camper-right" transform="translate(39 -12) scale(-1 1)">
      ${pixels(sitter, figure)}
    </g>
    <g transform="translate(-13.5 0)">
      ${pixels(['BB....B..', '..BBBB...', '...DDDD..', '.BBB..BB.'], {
        B: 'trail-log',
        D: 'trail-log-lit',
      })}
    </g>
    <g transform="translate(-10.5 -24)">
      ${flames
        .map(
          (rows, frame) =>
            `<g class="trail-flame-frame" style="--phase:${frame * -350}ms">${pixels(rows, fire)}</g>`,
        )
        .join('')}
    </g>
    <g class="trail-toast">
      <path class="trail-toasting-stick" d="M-28-2-8-25"/>
      <path class="trail-marshmallow" d="M-11-28h5v5h-5Z"/>
      <path class="trail-marshmallow-edge" d="M-11-23h5v1h-5Z"/>
    </g>
    <g class="trail-rocks">
      <path d="M-18 0h5v3h-5ZM13 0h5v3h-5ZM-10 9h5v3h-5ZM5 9h5v3H5Z"/>
    </g>
    <path class="trail-ember" d="M2-27h2v2H2Z"/>
    <path class="trail-ember trail-ember-late" d="M-5-22h2v2h-2Z"/>
    <path class="trail-smoke" d="M-2-33h3v3h-3Z"/>
    <path class="trail-smoke trail-smoke-late" d="M4-40h3v3H4Z"/>
  `
}
