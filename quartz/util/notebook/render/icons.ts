export type NotebookIcon =
  | 'run'
  | 'stop'
  | 'reset'
  | 'debug'
  | 'edit'
  | 'save'
  | 'revert'
  | 'copy'
  | 'check'
  | 'expand'
  | 'vim'

export const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 6h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z"/></svg>',
  debug:
    '<svg class="notebook-debug-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 12a7 7 0 1 1-14 0 7 7 0 1 1 14 0"/><path class="notebook-debug-icon-core" d="M16 12a4 4 0 1 1-8 0 4 4 0 1 1 8 0"/></svg>',
  reset:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5.7 12a6.3 6.3 0 1 0 6.3-6.3 6.83 6.83 0 0 0-4.72 1.92L5.7 9.2"/><path d="M5.7 5.7v3.5h3.5"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 8h11v11H8z"/><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>',
  check:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m5 12 4 4L19 6"/></svg>',
  expand:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 5v14"/><path d="m6 13 6 6 6-6"/></svg>',
  vim: [
    '<svg class="notebook-vim-icon" viewBox="0 0 602 734" aria-hidden="true" focusable="false">',
    '<g transform="translate(2 3)">',
    '<path class="notebook-vim-icon-left" d="M0 155.5704 155-1l-.000003 728L0 572.237919z"/>',
    '<path class="notebook-vim-icon-right" d="M443.060403 156.982405 600-1l-3.181208 728L442 572.219941z" transform="translate(521 363.5) scale(-1 1) translate(-521 -363.5)"/>',
    '<path class="notebook-vim-icon-cross" d="M154.986294 0 558 615.189696 445.224605 728 42 114.172017z"/>',
    '</g>',
    '</svg>',
  ].join(''),
}

const notebookGoLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-go-icon" viewBox="0 0 207 78" aria-hidden="true" focusable="false">',
  '<g fill="#00acd7" fill-rule="evenodd">',
  '<path d="m16.2 24.1c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h35.7c.4 0 .5.3.3.6l-1.7 2.6c-.2.3-.7.6-1 .6z"/>',
  '<path d="m1.1 33.3c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h45.6c.4 0 .6.3.5.6l-.8 2.4c-.1.4-.5.6-.9.6z"/>',
  '<path d="m25.3 42.5c-.4 0-.5-.3-.3-.6l1.4-2.5c.2-.3.6-.6 1-.6h20c.4 0 .6.3.6.7l-.2 2.4c0 .4-.4.7-.7.7z"/>',
  '<g transform="translate(55)"><path d="m74.1 22.3c-6.3 1.6-10.6 2.8-16.8 4.4-1.5.4-1.6.5-2.9-1-1.5-1.7-2.6-2.8-4.7-3.8-6.3-3.1-12.4-2.2-18.1 1.5-6.8 4.4-10.3 10.9-10.2 19 .1 8 5.6 14.6 13.5 15.7 6.8.9 12.5-1.5 17-6.6.9-1.1 1.7-2.3 2.7-3.7-3.6 0-8.1 0-19.3 0-2.1 0-2.6-1.3-1.9-3 1.3-3.1 3.7-8.3 5.1-10.9.3-.6 1-1.6 2.5-1.6h36.4c-.2 2.7-.2 5.4-.6 8.1-1.1 7.2-3.8 13.8-8.2 19.6-7.2 9.5-16.6 15.4-28.5 17-9.8 1.3-18.9-.6-26.9-6.6-7.4-5.6-11.6-13-12.7-22.2-1.3-10.9 1.9-20.7 8.5-29.3 7.1-9.3 16.5-15.2 28-17.3 9.4-1.7 18.4-.6 26.5 4.9 5.3 3.5 9.1 8.3 11.6 14.1.6.9.2 1.4-1 1.7z"/>',
  '<path d="m107.2 77.6c-9.1-.2-17.4-2.8-24.4-8.8-5.9-5.1-9.6-11.6-10.8-19.3-1.8-11.3 1.3-21.3 8.1-30.2 7.3-9.6 16.1-14.6 28-16.7 10.2-1.8 19.8-.8 28.5 5.1 7.9 5.4 12.8 12.7 14.1 22.3 1.7 13.5-2.2 24.5-11.5 33.9-6.6 6.7-14.7 10.9-24 12.8-2.7.5-5.4.6-8 .9zm23.8-40.4c-.1-1.3-.1-2.3-.3-3.3-1.8-9.9-10.9-15.5-20.4-13.3-9.3 2.1-15.3 8-17.5 17.4-1.8 7.8 2 15.7 9.2 18.9 5.5 2.4 11 2.1 16.3-.6 7.9-4.1 12.2-10.5 12.7-19.1z" fill-rule="nonzero"/></g>',
  '</g>',
  '</svg>',
].join('')

const notebookRustLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-rust-icon" viewBox="0 0 106 106" aria-hidden="true" focusable="false" xmlns:xlink="http://www.w3.org/1999/xlink">',
  '<g transform="translate(53 53)">',
  '<path transform="translate(0.5 0.5)" stroke="black" stroke-width="1" stroke-linejoin="round" d="M -9,-15 H 4 C 12,-15 12,-7 4,-7 H -9 Z M -40,22 H 0 V 11 H -9 V 3 H 1 C 12,3 6,22 15,22 H 40 V 3 H 34 V 5 C 34,13 25,12 24,7 C 23,2 19,-2 18,-2 C 33,-10 24,-26 12,-26 H -35 V -15 H -25 V 11 H -40 Z"/>',
  '<g mask="url(#notebook-rust-holes)">',
  '<circle r="43" fill="none" stroke="black" stroke-width="9"/>',
  '<g><polygon id="notebook-rust-cog" stroke="black" stroke-width="3" stroke-linejoin="round" points="46,3 51,0 46,-3"/><use xlink:href="#notebook-rust-cog" transform="rotate(11.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(22.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(33.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(45.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(56.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(67.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(78.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(90.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(101.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(112.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(123.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(135.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(146.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(157.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(168.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(180.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(191.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(202.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(213.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(225.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(236.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(247.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(258.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(270.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(281.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(292.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(303.75)"/><use xlink:href="#notebook-rust-cog" transform="rotate(315.00)"/><use xlink:href="#notebook-rust-cog" transform="rotate(326.25)"/><use xlink:href="#notebook-rust-cog" transform="rotate(337.50)"/><use xlink:href="#notebook-rust-cog" transform="rotate(348.75)"/></g>',
  '<g><polygon id="notebook-rust-mount" stroke="black" stroke-width="6" stroke-linejoin="round" points="-7,-42 0,-35 7,-42"/><use xlink:href="#notebook-rust-mount" transform="rotate(72)"/><use xlink:href="#notebook-rust-mount" transform="rotate(144)"/><use xlink:href="#notebook-rust-mount" transform="rotate(216)"/><use xlink:href="#notebook-rust-mount" transform="rotate(288)"/></g>',
  '</g>',
  '<mask id="notebook-rust-holes"><rect x="-60" y="-60" width="120" height="120" fill="white"/><circle id="notebook-rust-hole" cy="-40" r="3"/><use xlink:href="#notebook-rust-hole" transform="rotate(72)"/><use xlink:href="#notebook-rust-hole" transform="rotate(144)"/><use xlink:href="#notebook-rust-hole" transform="rotate(216)"/><use xlink:href="#notebook-rust-hole" transform="rotate(288)"/></mask>',
  '</g>',
  '</svg>',
].join('')

const notebookOcamlLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-ocaml-icon" viewBox="0 0 165.552 144.277" aria-hidden="true" focusable="false">',
  '<g>',
  '<path fill="#FFFFFF" d="M86.085,127c-0.209-1.424,0.197-2.841-0.232-4.177c-0.367-1.166-1.209-1.273-1.762-2.221c-1.457-2.487-2.963-5.709-3.102-8.754c-0.127-2.735-1.133-5.206-1.27-7.917c-0.066-1.308,0.088-2.657,0.041-3.952c-0.025-0.63-0.061-1.176-0.186-1.86c-0.031-0.169-0.143-0.865-0.195-1.144l0.34-0.848c-0.15-0.291,2.902-0.194,3.812-0.188c1.545,0.019,2.998,0.099,4.539,0.173c3.148,0.156,6.016,0.117,9.082-0.356c6.832-1.055,9.973-3.845,11.58-5.005c6.273-4.523,9.146-11.918,9.146-11.918c1.035-2.31,1.031-6.431,3.25-8.276c2.615-2.179,7.006-2.022,10.008-3.359c1.756-0.777,3.023-1.205,4.818-0.833c1.332,0.278,3.73,1.821,4.281-0.345c-0.445-0.287-0.619-0.812-0.857-1.103c2.475-0.245,0.047-5.986-0.932-7.133c-1.512-1.77-4.035-2.581-6.719-3.293c-3.188-0.845-6.08-1.82-9.082-1.231c-5.242,1.026-4.85-1.974-7.939-1.974c-3.707,0-10.303,0.182-11.443,3.786c-0.531,1.683-1.078,1.753-1.998,3.044c-0.787,1.106,0.137,2.082-0.258,3.344c-0.408,1.297-1.007,5.865-1.632,7.459c-1.057,2.697-2.317,6.065-4.643,6.065c-3.261,0.39-5.824,0.515-8.469-0.445c-1.592-0.578-4.26-1.483-5.58-2.039c-6.088-2.563-7.088-5.367-7.088-5.367c-0.653-1.08-2.374-2.821-3.018-5.093c-0.708-2.502-1.903-4.589-2.387-5.891c-0.501-1.349-1.699-3.51-2.64-5.846c-1.205-2.991-2.9-5.223-4.141-6.331c-1.896-1.69-3.646-4.306-7.495-3.546c-0.688,0.136-3.188,0.249-5.104,1.856c-1.299,1.09-1.709,3.339-2.912,5.236c-0.695,1.096-1.917,4.24-3.038,6.863c-0.777,1.818-1.139,3.181-1.979,3.85c-0.657,0.524-1.471,1.201-2.456,0.832c-0.611-0.229-1.264-0.617-1.923-1.132c-0.89-0.695-2.913-4.138-4.156-6.681c-1.077-2.205-3.376-5.502-4.706-7.287c-1.914-2.568-3.036-3.219-5.864-3.219c-6.067,0-6.526,3.397-9.195,8.337c-1.172,2.17-1.599,5.614-3.952,8.313c-1.345,1.544-5.637,7.893-8.621,8.972v-0.031L0,66.366v45.257l0.008,0.063v-0.284c0.193-0.59,0.398-1.156,0.631-1.662c1.154-2.459,3.832-4.741,5.32-7.266c0.809-1.376,1.732-2.724,2.268-4.168c0.461-1.244,0.688-3.099,1.354-4.178c0.816-1.323,2.094-1.773,3.406-1.987c2.055-0.339,3.801,2.954,6.43,4.166c1.121,0.515,6.281,2.342,7.83,2.717c2.551,0.61,5.381,1.119,7.971,1.642c1.387,0.28,2.713,0.443,4.141,0.588c1.281,0.128,6.08,0.287,6.377,0.634c-2.439,1.244-3.869,4.736-4.785,7.207c-0.955,2.575-1.621,5.441-2.775,7.96c-1.279,2.783-3.961,3.941-3.641,7.184c0.123,1.294,0.359,2.651,0.143,4.075c-0.23,1.499-0.836,2.669-1.277,4.137c-0.566,1.915-1.24,8.1-2.113,9.918l5.337-0.669l0.009-0.003c0.583-1.386,1.12-7.237,1.309-7.794c0.998-2.934,2.322-5.348,4.359-7.617c1.986-2.211,1.883-5.061,3.043-7.637c1.256-2.8,2.945-5.039,4.539-7.671c2.881-4.759,4.781-10.767,10.906-11.989c0.654-0.135,4.404,2.569,6.068,4.177c1.906,1.832,3.988,3.954,5.24,6.48c2.424,4.896,4.48,11.988,5.258,15.899c0.447,2.246,0.803,2.38,2.322,4.159c0.699,0.815,2.094,3.362,2.553,4.34c0.482,1.044,1.215,3.42,1.799,4.633c0.344,0.722,1.236,2.94,1.885,4.856l4.987-0.156c0.018,0.042,0.109-0.012,0.13,0.027c0.002,0,0.005-0.001,0.007-0.002c-0.021-0.038-0.04-0.082-0.058-0.123C88.496,138.292,86.906,132.522,86.085,127z"/>',
  '<path fill="#484444" d="M82.919,97.901l0.023-0.061C82.908,97.686,82.896,97.651,82.919,97.901z"/>',
  '<linearGradient id="notebook-ocaml-gradient-1" gradientUnits="userSpaceOnUse" x1="-675.0754" y1="96.4384" x2="-675.0754" y2="96.6205" gradientTransform="matrix(1 0 0 1 758 1.28)"><stop offset="0" style="stop-color:#F29100"/><stop offset="1" style="stop-color:#EC670F"/></linearGradient>',
  '<path fill="url(#notebook-ocaml-gradient-1)" d="M82.919,97.901l0.023-0.061C82.908,97.686,82.896,97.651,82.919,97.901z"/>',
  '<linearGradient id="notebook-ocaml-gradient-2" gradientUnits="userSpaceOnUse" x1="-696.7245" y1="97.701" x2="-696.7245" y2="142.9972" gradientTransform="matrix(1 0 0 1 758 1.28)"><stop offset="0" style="stop-color:#F29100"/><stop offset="1" style="stop-color:#EC670F"/></linearGradient>',
  '<path fill="url(#notebook-ocaml-gradient-2)" d="M84.031,138.674c-0.584-1.213-1.316-3.589-1.799-4.633c-0.459-0.978-1.854-3.524-2.553-4.34c-1.52-1.779-1.875-1.913-2.322-4.159c-0.777-3.911-2.834-11.004-5.258-15.899c-1.252-2.526-3.334-4.648-5.24-6.48c-1.664-1.607-5.414-4.312-6.068-4.177c-6.125,1.223-8.025,7.23-10.906,11.989c-1.594,2.632-3.283,4.871-4.539,7.671c-1.16,2.575-1.057,5.426-3.043,7.637c-2.037,2.27-3.361,4.684-4.359,7.617c-0.189,0.557-0.726,6.408-1.309,7.794c0,0.001-0.001,0.002-0.001,0.003l9.104-0.641c8.482,0.578,6.033,3.829,19.273,3.121l20.906-0.647l0,0C85.267,141.614,84.374,139.396,84.031,138.674z"/>',
  '<linearGradient id="notebook-ocaml-gradient-3" gradientUnits="userSpaceOnUse" x1="-675.2191" y1="-1.2802" x2="-675.219" y2="142.9646" gradientTransform="matrix(1 0 0 1 758 1.28)"><stop offset="0" style="stop-color:#F29100"/><stop offset="1" style="stop-color:#EC670F"/></linearGradient>',
  '<path fill="url(#notebook-ocaml-gradient-3)" d="M144.695,0H20.865C9.347,0,0.01,9.339,0.01,20.857v45.476v0.031c2.984-1.079,7.276-7.428,8.621-8.972c2.353-2.7,2.78-6.144,3.952-8.313c2.669-4.94,3.128-8.337,9.195-8.337c2.828,0,3.951,0.652,5.864,3.219c1.331,1.785,3.63,5.083,4.706,7.287c1.242,2.544,3.266,5.986,4.156,6.681c0.659,0.516,1.312,0.903,1.923,1.132c0.984,0.369,1.798-0.308,2.456-0.832c0.84-0.669,1.202-2.032,1.979-3.85c1.122-2.623,2.343-5.766,3.038-6.863c1.203-1.896,1.613-4.146,2.912-5.236c1.916-1.607,4.416-1.72,5.104-1.856c3.849-0.76,5.599,1.856,7.495,3.546c1.241,1.108,2.937,3.34,4.141,6.331c0.941,2.336,2.139,4.497,2.64,5.846c0.484,1.302,1.679,3.389,2.387,5.891c0.643,2.272,2.364,4.013,3.018,5.093c0,0,1.001,2.804,7.088,5.367c1.32,0.556,3.988,1.46,5.58,2.039c2.645,0.961,5.207,0.836,8.469,0.445c2.326,0,3.586-3.368,4.643-6.065c0.625-1.594,1.224-6.162,1.632-7.459c0.395-1.262-0.529-2.238,0.258-3.344c0.92-1.291,1.467-1.361,1.998-3.044c1.141-3.604,7.736-3.786,11.443-3.786c3.09,0,2.697,3,7.939,1.974c3.002-0.589,5.895,0.387,9.082,1.231c2.684,0.712,5.207,1.523,6.719,3.293c0.979,1.146,3.406,6.888,0.932,7.133c0.238,0.291,0.412,0.816,0.857,1.103c-0.551,2.166-2.949,0.623-4.281,0.345c-1.795-0.372-3.062,0.056-4.818,0.833c-3.002,1.337-7.393,1.181-10.008,3.359c-2.219,1.846-2.215,5.967-3.25,8.276c0,0-2.873,7.394-9.146,11.918c-1.607,1.16-4.748,3.95-11.58,5.005c-3.066,0.474-5.934,0.513-9.082,0.356c-1.541-0.074-2.994-0.153-4.539-0.173c-0.91-0.007-3.963-0.104-3.812,0.188l-0.34,0.848c0.053,0.279,0.164,0.976,0.195,1.144c0.125,0.685,0.16,1.231,0.186,1.86c0.047,1.295-0.107,2.645-0.041,3.952c0.137,2.711,1.143,5.182,1.27,7.917c0.139,3.045,1.645,6.267,3.102,8.754c0.553,0.947,1.395,1.055,1.762,2.221c0.43,1.336,0.023,2.753,0.232,4.177c0.82,5.521,2.41,11.292,4.896,16.275c0.017,0.041,0.037,0.086,0.058,0.123c0,0,0,0.001,0.001,0.002c3.07-0.516,6.146-1.62,10.135-2.21c7.314-1.085,17.486-0.526,24.02-1.138c16.533-1.554,25.506,6.781,40.355,3.365V20.858C165.55,9.339,156.216,0,144.695,0zM82.919,97.901c-0.023-0.25-0.012-0.215,0.023-0.061L82.919,97.901z"/>',
  '<linearGradient id="notebook-ocaml-gradient-4" gradientUnits="userSpaceOnUse" x1="-735.129" y1="90.8344" x2="-735.129" y2="141.9687" gradientTransform="matrix(1 0 0 1 758 1.28)"><stop offset="0" style="stop-color:#F29100"/><stop offset="1" style="stop-color:#EC670F"/></linearGradient>',
  '<path fill="url(#notebook-ocaml-gradient-4)" d="M38.175,117.053c1.154-2.518,1.82-5.385,2.775-7.96c0.916-2.471,2.346-5.963,4.785-7.207c-0.297-0.347-5.096-0.506-6.377-0.634c-1.428-0.145-2.754-0.308-4.141-0.588c-2.59-0.523-5.42-1.031-7.971-1.642c-1.549-0.375-6.709-2.202-7.83-2.717c-2.629-1.212-4.375-4.505-6.43-4.166c-1.312,0.214-2.59,0.664-3.406,1.987c-0.666,1.079-0.893,2.933-1.354,4.178c-0.535,1.444-1.459,2.792-2.268,4.168c-1.488,2.524-4.166,4.807-5.32,7.266c-0.232,0.506-0.438,1.072-0.631,1.662v0.284v9.15v16.321v2.358c1.346,0.23,2.754,0.513,4.33,0.934c11.631,3.104,14.469,3.366,25.877,2.062l1.07-0.142v-0.001c0.873-1.818,1.547-8.003,2.113-9.918c0.441-1.468,1.047-2.638,1.277-4.137c0.217-1.424-0.02-2.781-0.143-4.075C34.214,120.994,36.896,119.836,38.175,117.053z"/>',
  '</g>',
  '</svg>',
].join('')

const notebookCLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-c-icon" viewBox="0 0 38.000089 42.000031" aria-hidden="true" focusable="false">',
  '<path fill="#004482" fill-rule="evenodd" clip-rule="evenodd" d="m17.903.28628166c.679-.381 1.515-.381 2.193 0 3.355 1.88300004 13.451 7.55100004 16.807 9.43400004C37.582 10.100282 38 10.804282 38 11.566282v18.867c0 .762-.418 1.466-1.097 1.847-3.355 1.883-13.451 7.551-16.807 9.434-.679.381-1.515.381-2.193 0-3.355-1.883-13.451-7.551-16.807-9.434C.418 31.899282 0 31.196282 0 30.434282v-18.867c0-.762.418-1.466 1.097-1.8470003C4.451 7.8372817 14.549 2.1692817 17.903.28628166z"/>',
  '<path fill="#659ad2" fill-rule="evenodd" clip-rule="evenodd" d="M.304 31.404282C.038 31.048282 0 30.710282 0 30.255282v-18.759c0-.758.417-1.458 1.094-1.8360003 3.343-1.872 13.405-7.507 16.748-9.38000004.677-.379 1.594-.371 2.271.008 3.343 1.87200004 13.371 7.45900004 16.714 9.33100004.27.152.476.335.66.5760003z"/>',
  '<path fill="#fff" fill-rule="evenodd" clip-rule="evenodd" d="M19 7.0002817c7.727 0 14 6.2730003 14 14.0000003 0 7.727-6.273 14-14 14s-14-6.273-14-14c0-7.727 6.273-14.0000003 14-14.0000003zm0 7.0000003c3.863 0 7 3.136 7 7 0 3.863-3.137 7-7 7s-7-3.137-7-7c0-3.864 3.136-7 7-7z"/>',
  '<path fill="#00599c" fill-rule="evenodd" clip-rule="evenodd" d="M37.485 10.205282c.516.483.506 1.211.506 1.784 0 3.795-.032 14.589.009 18.384.004.396-.127.813-.323 1.127l-19.084-10.5z"/>',
  '</svg>',
].join('')

const notebookCppLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-cpp-icon" viewBox="0 0 306 344.35" aria-hidden="true" focusable="false">',
  '<path fill="#00599c" d="M302.107 258.262c2.401-4.159 3.893-8.845 3.893-13.053V99.14c0-4.208-1.49-8.893-3.892-13.052L153 172.175z"/>',
  '<path fill="#004482" d="m166.25 341.193 126.5-73.034c3.644-2.104 6.956-5.737 9.357-9.897L153 172.175 3.893 258.263c2.401 4.159 5.714 7.793 9.357 9.896l126.5 73.034c7.287 4.208 19.213 4.208 26.5 0z"/>',
  '<path fill="#659ad2" d="M302.108 86.087c-2.402-4.16-5.715-7.793-9.358-9.897L166.25 3.156c-7.287-4.208-19.213-4.208-26.5 0L13.25 76.19C5.962 80.397 0 90.725 0 99.14v146.069c0 4.208 1.491 8.894 3.893 13.053L153 172.175z"/>',
  '<path fill="#fff" d="M153 274.175c-56.243 0-102-45.757-102-102s45.757-102 102-102c36.292 0 70.139 19.53 88.331 50.968l-44.143 25.544c-9.105-15.736-26.038-25.512-44.188-25.512-28.122 0-51 22.878-51 51 0 28.121 22.878 51 51 51 18.152 0 35.085-9.776 44.191-25.515l44.143 25.543c-18.192 31.441-52.04 50.972-88.334 50.972z"/>',
  '<polygon fill="#fff" points="255 166.508 243.666 166.508 243.666 155.175 232.334 155.175 232.334 166.508 221 166.508 221 177.841 232.334 177.841 232.334 189.175 243.666 189.175 243.666 177.841 255 177.841"/>',
  '<polygon fill="#fff" points="297.5 166.508 286.166 166.508 286.166 155.175 274.834 155.175 274.834 166.508 263.5 166.508 263.5 177.841 274.834 177.841 274.834 189.175 286.166 189.175 286.166 177.841 297.5 177.841"/>',
  '</svg>',
].join('')

const notebookWasmLanguageLogoSvg = [
  '<svg class="notebook-language-svg notebook-wasm-icon" viewBox="0 0 612 612" aria-hidden="true" focusable="false">',
  '<path fill="#654ff0" d="M376 0v3.3c0 38.76-31.42 70.17-70.17 70.17-38.76 0-70.17-31.42-70.17-70.17V0H0v612h612V0z"/>',
  '<path fill="#fff" d="M142.16 329.81h40.56l27.69 147.47h.5l33.28-147.47h37.94l30.06 149.28h.59l31.56-149.28h39.78L332.43 546.5h-40.25l-29.81-147.47h-.78L229.68 546.5h-41zm287.69 0h63.94l63.5 216.69h-41.84l-13.81-48.22H428.8l-10.66 48.22h-40.75zm24.34 53.41-17.69 79.5h55.06l-20.31-79.5z"/>',
  '</svg>',
].join('')

export const notebookLanguageIconSvg: Readonly<Record<string, string>> = {
  bash: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6.5h16v11H4zM7 10l2 2-2 2m5 0h4"/></svg>',
  c: notebookCLanguageLogoSvg,
  cpp: notebookCppLanguageLogoSvg,
  css: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 3h14l-1.3 15.1L12 21l-5.7-2.9z"/><path fill="var(--light)" d="M9 7h7l-.2 2H11l.1 1.5h4.5l-.5 5.2-3.1.9-3.1-.9-.2-2.4h2l.1 1 1.2.3 1.2-.3.2-1.8H8.6z"/></svg>',
  go: notebookGoLanguageLogoSvg,
  html: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 3h14l-1.3 15.1L12 21l-5.7-2.9z"/><path fill="var(--light)" d="M8.6 7h6.8l-.2 2h-4.5l.1 1.5H15l-.5 5.2-2.5.8-2.5-.8-.2-2.4h2l.1.9.6.2.6-.2.2-1.7H9z"/></svg>',
  java: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.9" d="M9 17h6a3 3 0 0 0 3-3H6a3 3 0 0 0 3 3m0 2h6m-4-8c-1.5-1.2 2.2-2.3.7-3.6M14 11c-1.3-1 1.8-2 .6-3"/></svg>',
  javascript:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect width="18" height="18" x="3" y="3" fill="currentColor" rx="2"/><text x="12" y="16.5" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="7.5" font-weight="900" fill="var(--light)">JS</text></svg>',
  json: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7.5A2.5 2.5 0 0 0 5 7.5V9a2 2 0 0 1-2 2 2 2 0 0 1 2 2v1.5A2.5 2.5 0 0 0 7.5 17H9m6-12h1.5A2.5 2.5 0 0 1 19 7.5V9a2 2 0 0 0 2 2 2 2 0 0 0-2 2v1.5a2.5 2.5 0 0 1-2.5 2.5H15"/></svg>',
  markdown:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" d="M3 6h18v12H3z"/><path fill="currentColor" d="M6 15V9h2l2 2.4L12 9h2v6h-2v-3l-2 2.3L8 12v3zm10-6h2v3h1.5L17 15.5 14.5 12H16z"/></svg>',
  ocaml: notebookOcamlLanguageLogoSvg,
  python:
    '<svg class="notebook-language-svg notebook-python-icon" viewBox="0 0 111 112" aria-hidden="true" focusable="false"><path fill="#3776ab" d="M54.918785.00091927421C50.335132.02221727 45.957846.41313697 42.106285 1.0946693 30.760069 3.0991731 28.700036 7.2947714 28.700035 15.032169v10.21875h26.8125v3.40625h-36.875c-7.792459 0-14.6157588 4.683717-16.7499998 13.59375-2.46181998 10.212966-2.57101508 16.586023 0 27.25 1.9059283 7.937852 6.4575432 13.593748 14.2499998 13.59375h9.21875v-12.25c0-8.849902 7.657144-16.656248 16.75-16.65625h26.78125c7.454951 0 13.406253-6.138164 13.40625-13.625v-25.53125c0-7.2663386-6.12998-12.7247771-13.40625-13.9374997C64.281548.32794397 59.502438-.02037903 54.918785.00091927421zM40.418785 8.2196694c2.769547 0 5.03125 2.2986456 5.03125 5.1249996-.000002 2.816336-2.261703 5.09375-5.03125 5.09375-2.779476-.000001-5.03125-2.277415-5.03125-5.09375-.000001-2.826353 2.251774-5.1249996 5.03125-5.1249996z"/><path fill="#ffd43b" d="M85.637535 28.657169v11.90625c0 9.230755-7.825895 16.999999-16.75 17h-26.78125c-7.335833 0-13.406249 6.278483-13.40625 13.625v25.531247c0 7.266344 6.318588 11.540324 13.40625 13.625004 8.487331 2.49561 16.626237 2.94663 26.78125 0 6.750155-1.95439 13.406253-5.88761 13.40625-13.625004V86.500919h-26.78125v-3.40625h40.187504c7.792461 0 10.696251-5.435408 13.406241-13.59375 2.79933-8.398886 2.68022-16.475776 0-27.25-1.92578-7.757441-5.60387-13.59375-13.406241-13.59375zm-15.0625 64.65625c2.779478.000003 5.03125 2.277417 5.03125 5.093747-.000002 2.826354-2.251775 5.125004-5.03125 5.125004-2.76955 0-5.03125-2.29865-5.03125-5.125004.000002-2.81633 2.261697-5.093747 5.03125-5.093747z"/></svg>',
  rust: notebookRustLanguageLogoSvg,
  text: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M6 7h12M6 12h12M6 17h8"/></svg>',
  typescript:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect width="18" height="18" x="3" y="3" fill="currentColor" rx="2"/><text x="12" y="16.5" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="7.5" font-weight="900" fill="var(--light)">TS</text></svg>',
  wasm: notebookWasmLanguageLogoSvg,
  zig: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 5h14v3L9.5 19H5v-3L14.5 5H19v3L9.5 19H5z"/></svg>',
}
