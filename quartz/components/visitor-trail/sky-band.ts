import type { Landscape } from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'

export type SkyBandState = {
  now: number
  // 0 overcast to 1 clear, and the forecast cloud fraction.
  clear: number
  cloud: number
  rain: number
  snow: number
  wind: number
  // Downwind direction across the screen.
  dx: number
  sunAltitude: number
}

const clouds = [
  ['...LL.....', '..LLLL.LL.', 'LLLLLLLLLL', '.SSSSSSSS.'],
  ['....LLL.....', '..LLLLLL.L..', '.LLLLLLLLLL.', 'LLLLLLLLLLLL', '..SSSSSSSS..'],
  ['..LL.LL.', '.LLLLLLL', 'LLLLLLLL', '.SSSSSS.'],
]
const wings = { up: ['B...B', '.B.B.', '..B..'], down: ['..B..', '.B.B.', 'B...B'] }

// Seconds into a loop of `period` at `now`, as a negative delay: the sky is drawn again every
// minute, and each loop picks up where it was rather than starting over.
const since = (now: number, period: number, offset: number) =>
  `${(-(((now / 1000 + offset) % period) + period) % period).toFixed(2)}s`

// The open sky above the ranges, around the reading board: a starfield on clear nights, shooting
// stars whenever the lighting is moonlit, clouds that drift downwind as thick as the forecast
// cover, and a flock that crosses on fair mornings. Forecast rain and snow fall through it from the
// homepage layer.
export function skyBand(
  landscape: Landscape,
  seed: number,
  state: SkyBandState,
  // The ridge line under a column, in scene space; the sky reaches down to it.
  crestAt: (x: number) => number = () => landscape.alpine.top,
) {
  const { width, alpine } = landscape
  const { now, clear, cloud, rain, snow, wind, dx, sunAltitude } = state
  // A new pattern each day, the same one through the day.
  const random = trailRandom(seed ^ Math.floor(now / 86_400_000) ^ 0x1d872b41)
  const top = 18
  const floor = alpine.top - 26
  const marks: string[] = []

  // The band follows the visitor's sky, whatever the site theme: stars once the sun is well down.
  // They fill the sky down to the ridges, so no empty strip parts the sky from the ranges; the
  // count keeps the density of the open band above them.
  const night = sunAltitude < -6
  const stars: string[] = []
  const horizon = (x: number) => Math.min(alpine.base, crestAt(x)) - 2
  const columns = 32
  const reach =
    Array.from(
      { length: columns },
      (_, column) => horizon(((column + 0.5) / columns) * width) - top,
    ).reduce((sum, span) => sum + span, 0) / columns
  const count = night ? Math.round((width / 26) * clear * Math.max(1, reach / (floor - top))) : 0
  for (let index = 0; index < count; index++) {
    const x = Math.round((8 + random() * (width - 16)) / 2) * 2
    const y = Math.round((top + random() * Math.max(1, horizon(x) - top)) / 2) * 2
    const size = random() < 0.12 ? 4 : 2
    stars.push(
      `<path class="trail-sky-star" data-star="${index % 3}" d="M${x} ${y}h${size}v${size}h${-size}Z"/>`,
    )
  }
  // Meteors are drawn for any sky that is not overcast; the stylesheet shows the group under
  // moonlit lighting, which is a real night or the dark theme. One streaks every few seconds,
  // anywhere from the top of the sky to just over the ranges.
  const meteors: string[] = []
  if (clear > 0.4)
    for (let index = 0; index < 5; index++) {
      const period = 11 + random() * 17
      const x = Math.round(width * (0.1 + random() * 0.8))
      const y = Math.round(top + random() * Math.max(1, alpine.top - 48 - top))
      const side = random() < 0.5 ? -1 : 1
      // A head and a tail fading back along its line, two cells a step.
      const tail = Array.from(
        { length: 7 },
        (_, step) =>
          `<path class="trail-sky-meteor-${step < 1 ? 'head' : step < 4 ? 'tail' : 'trace'}" d="M${-side * step * 4} ${-step * 2}h2v2h-2Z"/>`,
      ).join('')
      meteors.push(
        `<g transform="translate(${x} ${y})"><g class="trail-sky-meteor" style="--trail-meteor-x:${side * 72}px;--trail-meteor-y:36px;--trail-meteor-period:${period.toFixed(1)}s;--trail-meteor-delay:${since(now, period, index * 7.3)}">${tail}</g></g>`,
      )
    }
  marks.push(
    `<g class="trail-sky-night">${stars.join('')}</g><g class="trail-sky-meteors">${meteors.join('')}</g>`,
  )

  // Cloud enough to see: a few puffs in fair weather, a full deck under rain or snow.
  const cover = rain || snow ? Math.max(0.8, cloud) : cloud
  const puffs = Math.round(cover * (width / 220))
  const drift = dx < 0 ? -1 : 1
  const cloudMarks: string[] = []
  for (let index = 0; index < puffs; index++) {
    const shape = clouds[Math.floor(random() * clouds.length)]
    const cell = random() < 0.4 ? 6 : 4
    const span = shape[0].length * cell
    const y = Math.round((top + random() * Math.max(1, floor - top - shape.length * cell)) / 2) * 2
    const period = Math.round((420 - wind * 260) * (cell === 6 ? 1.25 : 1))
    const from = drift > 0 ? -span - 8 : width + 8
    const to = drift > 0 ? width + 8 : -span - 8
    cloudMarks.push(
      `<g transform="translate(0 ${y})"><g class="trail-sky-cloud" style="--trail-cloud-from:${from}px;--trail-cloud-to:${to}px;--trail-cloud-period:${period}s;--trail-cloud-delay:${since(now, period, (index / Math.max(1, puffs)) * period + random() * 30)};--trail-cloud-x:${Math.round(from + (to - from) * random())}px">${pixels(shape, { L: 'trail-sky-cloud-lit', S: 'trail-sky-cloud-shade' }, cell)}</g></g>`,
    )
  }
  marks.push(
    `<g class="trail-sky-clouds" data-overcast="${rain || snow ? 'true' : 'false'}">${cloudMarks.join('')}</g>`,
  )

  // Birds fly the fair mornings, from first light until the sun is well up.
  const hour = new Date(now).getHours()
  if (!rain && !snow && hour >= 5 && hour < 11 && sunAltitude > -3) {
    const period = 46
    const y = Math.round(top + 20 + random() * Math.max(1, floor - top - 60))
    const heading = random() < 0.5 ? -1 : 1
    const birds = Array.from({ length: 3 + Math.floor(random() * 3) }, (_, index) => {
      // A loose V: each bird behind and to one side of the one ahead.
      const rank = Math.ceil(index / 2)
      const side = index % 2 ? 1 : -1
      const bx = -heading * rank * 12
      const by = side * rank * 6 + Math.round(random() * 2) * 2
      return `<g class="trail-sky-bird" transform="translate(${bx} ${by})" style="--trail-bird-delay:${(-random()).toFixed(2)}s"><g class="trail-sky-bird-up">${pixels(wings.up, { B: 'trail-sky-bird-body' }, 2)}</g><g class="trail-sky-bird-down">${pixels(wings.down, { B: 'trail-sky-bird-body' }, 2)}</g></g>`
    }).join('')
    marks.push(
      `<g class="trail-sky-day"><g transform="translate(0 ${y})"><g class="trail-sky-flock" style="--trail-flock-from:${heading > 0 ? -60 : width + 60}px;--trail-flock-to:${heading > 0 ? width + 60 : -60}px;--trail-flock-period:${period}s;--trail-flock-delay:${since(now, period, 11)}">${birds}</g></g></g>`,
    )
  }
  return `<g class="trail-sky-band">${marks.join('')}</g>`
}
