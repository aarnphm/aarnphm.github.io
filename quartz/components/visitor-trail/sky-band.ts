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
// The birds that cross the band, each with its wings up and down, how fast it beats them, how
// long one crossing takes, where in the band it keeps (0 the top, 1 just over the ranges), how
// many fly together and how far apart they hold in the V. A hawk soars on set wings, alone.
type BirdKind = 'gull' | 'songbird' | 'goose' | 'hawk'
const birds: Record<
  BirdKind,
  {
    up: string[]
    down: string[] | null
    flap: number
    period: [number, number]
    band: [number, number]
    count: [number, number]
    gap: [number, number]
  }
> = {
  gull: {
    up: ['B...B', '.B.B.', '..B..'],
    down: ['..B..', '.B.B.', 'B...B'],
    flap: 0.5,
    period: [44, 62],
    band: [0.3, 0.75],
    count: [4, 6],
    gap: [12, 6],
  },
  songbird: {
    up: ['B.B', '.B.'],
    down: ['.B.', 'B.B'],
    flap: 0.24,
    period: [26, 36],
    band: [0.6, 1],
    count: [5, 7],
    gap: [7, 4],
  },
  goose: {
    up: ['B.....B', '.B...B.', '..BBB..'],
    down: ['..BBB..', '.B...B.', 'B.....B'],
    flap: 0.8,
    period: [70, 90],
    band: [0, 0.35],
    count: [6, 8],
    gap: [16, 7],
  },
  hawk: {
    up: ['B.......B', '.BB...BB.', '...BBB...', '....B....'],
    down: null,
    flap: 0,
    period: [96, 120],
    band: [0.05, 0.4],
    count: [1, 1],
    gap: [0, 0],
  },
}

// Seconds into a loop of `period` at `now`, as a negative delay: the sky is drawn again every
// minute, and each loop picks up where it was rather than starting over.
const since = (now: number, period: number, offset: number) =>
  `${(-(((now / 1000 + offset) % period) + period) % period).toFixed(2)}s`

// The open sky above the ranges, around the reading board: a starfield on clear nights, shooting
// stars whenever the lighting is moonlit, clouds that drift downwind as thick as the forecast
// cover, and flocks that cross while the sun is up. Forecast rain and snow fall through it from
// the homepage layer.
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

  // Birds fly while the sun is up and the weather holds: one flock under cloud, two on a clear
  // day, each of its own kind, at its own height, pace and heading. Geese need the room of a
  // wide band; a wide screen on a clear day also gets a third flock and a hawk soaring over the
  // ranges. Every bird is two frames and two animations, so the counts stay small.
  if (!rain && !snow && sunAltitude > -3) {
    const wide = width >= 700
    const bag: BirdKind[] = wide ? ['gull', 'songbird', 'goose'] : ['gull', 'songbird']
    const plan: BirdKind[] = []
    const flocks = (clear > 0.5 ? 2 : 1) + (clear > 0.5 && width >= 1280 ? 1 : 0)
    for (let flock = 0; flock < flocks; flock++) {
      const left = bag.filter(kind => !plan.includes(kind))
      plan.push(left[Math.floor(random() * left.length)] ?? 'gull')
    }
    if (clear > 0.5 && width >= 1280) plan.push('hawk')
    const span = Math.max(1, floor - top - 60)
    const dayMarks = plan.map((kind, flock) => {
      const bird = birds[kind]
      const period = Math.round(bird.period[0] + random() * (bird.period[1] - bird.period[0]))
      const y = Math.round(
        top + 20 + (bird.band[0] + random() * (bird.band[1] - bird.band[0])) * span,
      )
      const heading = random() < 0.5 ? -1 : 1
      const count = bird.count[0] + Math.floor(random() * (bird.count[1] - bird.count[0] + 1))
      const members = Array.from({ length: count }, (_, index) => {
        // A loose V: each bird behind and to one side of the one ahead.
        const rank = Math.ceil(index / 2)
        const side = index % 2 ? 1 : -1
        const bx = -heading * rank * bird.gap[0]
        const by = side * rank * bird.gap[1] + Math.round(random() * 2) * 2
        const up = pixels(bird.up, { B: 'trail-sky-bird-body' }, 2)
        if (!bird.down)
          return `<g class="trail-sky-bird" data-bird="${kind}" transform="translate(${bx} ${by})">${up}</g>`
        return `<g class="trail-sky-bird" data-bird="${kind}" transform="translate(${bx} ${by})" style="--trail-bird-delay:${(-random() * bird.flap).toFixed(2)}s"><g class="trail-sky-bird-up">${up}</g><g class="trail-sky-bird-down">${pixels(bird.down, { B: 'trail-sky-bird-body' }, 2)}</g></g>`
      }).join('')
      return `<g transform="translate(0 ${y})"><g class="trail-sky-flock" data-bird="${kind}" style="--trail-flock-from:${heading > 0 ? -60 : width + 60}px;--trail-flock-to:${heading > 0 ? width + 60 : -60}px;--trail-flock-period:${period}s;--trail-flock-delay:${since(now, period, 11 + flock * 29)};--trail-bird-flap:${bird.flap || 0.5}s">${members}</g></g>`
    })
    marks.push(`<g class="trail-sky-day">${dayMarks.join('')}</g>`)
  }
  return `<g class="trail-sky-band">${marks.join('')}</g>`
}
