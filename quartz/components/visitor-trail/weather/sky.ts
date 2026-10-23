import type { TrailWeather } from '../../../util/visitor-trail-weather'
import type { Landscape } from '../landscape'
import { bayerThreshold } from '../../../util/bayer'
import { almanac, guessSite, type Body } from '../../scripts/404-almanac'
import { mountainCrestAt, type MountainRange } from '../mountain-profile'
import { trailRandom } from '../random'
import { skyBand } from '../sky-band'
import { decoration } from './conditions'

const sunRadius = 6
const moonRadius = 10

// Where a body hangs over the trail: across the width by its bearing from the equator-facing view
// (±120° reaches the edges), and up from the crest under it by its altitude. Below the horizon it
// sinks behind that crest, a radius for every 1.5°.
function hang(body: Body, radius: number, landscape: Landscape, ranges: readonly MountainRange[]) {
  const { width, bottom, alpine } = landscape
  const x = width / 2 + (body.bearing / 120) * width * 0.5
  const crest = mountainCrestAt(x, ranges) ?? alpine.base
  const ceiling = bottom + 14 + radius
  const y =
    body.alt >= 0
      ? crest - (crest - ceiling) * Math.sin((Math.min(90, body.alt) * Math.PI) / 180) ** 0.6
      : crest - (body.alt / 1.5) * radius
  const visible = y - radius < crest && x > -radius * 2 && x < width + radius * 2
  return { x: Math.round(x), y: Math.round(y), visible }
}

// A storm's bolt drops from the cloud deck to a crest behind the ranges in 2 px steps, and shows
// only at the flash (see trail-weather.scss), on the same clock as the page's flash.
function boltMarks(landscape: Landscape, ranges: readonly MountainRange[], seed: number) {
  const random = trailRandom(seed ^ 0x5b0c71a3)
  const { width, bottom, alpine } = landscape
  // The left half of the ranges, clear of the far gate on the right.
  let x = Math.round((width * (0.15 + random() * 0.3)) / 2) * 2
  const crest = mountainCrestAt(x, ranges) ?? alpine.base
  let d = ''
  for (let y = bottom + 10; y < crest; y += 6) {
    d += `M${x} ${y}h2v6h-2Z`
    x += (random() < 0.5 ? -2 : 2) * (random() < 0.3 ? 2 : 1)
  }
  return `<path class="trail-sky-bolt" d="${d}"/>`
}

// Twilight lights the sky behind the ranges, strongest low over the sun's bearing. A 4 × 4 Bayer
// screen turns the glow into whole cells, two tones, merged into runs per row.
function duskGlow(landscape: Landscape, sunX: number, strength: number) {
  if (strength < 0.04) return ''
  const { width, alpine } = landscape
  const cell = width < 700 ? 2 : 4
  const top = alpine.top - 48
  const depth = alpine.height * 0.75 + 48
  const span = width * 0.45
  const runs: Record<'inner' | 'outer', string[]> = { inner: [], outer: [] }
  for (let row = 0; row * cell < depth; row++) {
    const y = top + row * cell
    const low = (row * cell) / depth
    let tone: 'inner' | 'outer' | null = null
    let start = 0
    for (let column = 0; column * cell <= width; column++) {
      const x = column * cell
      const near = Math.max(0, 1 - Math.abs(x - sunX) / span) ** 1.5
      const level = strength * near * (0.25 + low * 0.75)
      const threshold = bayerThreshold(column, row)
      const next = level > threshold ? (level > 0.55 ? 'inner' : 'outer') : null
      if (next === tone) continue
      if (tone) runs[tone].push(`M${start} ${y}h${x - start}v${cell}h${start - x}Z`)
      tone = next
      start = x
    }
    if (tone) runs[tone].push(`M${start} ${y}H${width + cell}v${cell}H${start}Z`)
  }
  return `<path class="trail-sky-glow trail-sky-glow-outer" d="${runs.outer.join('')}"/><path class="trail-sky-glow trail-sky-glow-inner" d="${runs.inner.join('')}"/>`
}

export function skyMarks(
  landscape: Landscape,
  ranges: readonly MountainRange[],
  weather: TrailWeather | null,
  now: number,
  seed: number,
) {
  const site =
    weather?.lat !== null && weather?.lat !== undefined && weather.lon !== null
      ? { lat: weather.lat, lon: weather.lon, place: null, weather: null }
      : guessSite()
  const sky = almanac(new Date(now), site)
  const { rain, snow, storm, wind, dx } = decoration(weather)
  const clear = rain || snow ? 0 : 1 - (weather?.cloud ?? 0.3)
  const sun = hang(sky.sun, sunRadius, landscape, ranges)
  const moon = hang(sky.moon, moonRadius, landscape, ranges)
  const marks = [
    skyBand(
      landscape,
      seed,
      { now, clear, cloud: weather?.cloud ?? 0.3, rain, snow, wind, dx, sunAltitude: sky.sun.alt },
      x => mountainCrestAt(x, ranges) ?? landscape.alpine.base,
    ),
    duskGlow(landscape, sun.x, sky.dusk * (0.35 + clear * 0.65)),
  ]
  if (storm) marks.push(boltMarks(landscape, ranges, seed))
  if (sun.visible && clear > 0.15) {
    const rays = sky.sun.alt > 4 ? 'M-1-12h2v3h-2ZM-1 9h2v3h-2ZM-12-1h3v2h-3ZM9-1h3v2H9Z' : ''
    marks.push(
      `<g class="trail-sky-sun" data-sun-altitude="${sky.sun.alt.toFixed(1)}" style="--trail-sun-low:${Math.max(0, Math.min(1, 1 - sky.sun.alt / 12)).toFixed(2)};--trail-sun-clear:${clear.toFixed(2)}" transform="translate(${sun.x} ${sun.y})"><path d="M-4-6h8v2h2v8h-2v2h-8V4h-2V-4h2Z${rays}"/></g>`,
    )
  }
  if (moon.visible)
    marks.push(
      `<g class="trail-sky-moon" data-moon-altitude="${sky.moon.alt.toFixed(1)}" transform="translate(${moon.x} ${moon.y})${sky.south ? ' scale(-1 1)' : ''}">
      <g class="trail-moon-motion">
      <path class="trail-moon-shadow" d="M-4-10h8v2h4v4h2v8H8v4H4v2h-8V8h-4V4h-2v-8h2v-4h4Z"/>
      <path class="trail-moon-lit" data-lunar-disc/>
      <path class="trail-moon-crater" d="M3-5h2v2H3ZM4 3h3v2H4Z"/>
      </g>
    </g>`,
    )
  return marks.join('')
}
