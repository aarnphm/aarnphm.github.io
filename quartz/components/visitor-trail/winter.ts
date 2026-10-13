import type { TrailWeather } from '../../util/visitor-trail-weather'
import { visitorPrecipitation } from '../../util/visitor-precipitation'
import { groundOffset, type Landscape } from './landscape'
import { trailRandom } from './random'

export function trailWinter(weather: TrailWeather | null, festiveSnow = false) {
  const snowing = visitorPrecipitation(weather?.code).kind === 'snow'
  const temperature = weather?.temperatureC ?? null
  const forecastCover = Math.max(
    snowing ? (weather?.code === 75 || weather?.code === 86 ? 0.95 : 0.65) : 0,
    temperature !== null && temperature <= 2 ? Math.min(1, (weather?.snowCm ?? 0) / 2.5) : 0,
  )
  // Seasonal ground cover is illustrative; the forecast only supplies recent snowfall.
  const seasonalCover =
    weather?.season === 'winter' && (temperature === null || temperature <= 4) ? 0.78 : 0
  const cover = Math.max(forecastCover, seasonalCover, festiveSnow ? 0.9 : 0)
  const frozen =
    cover >= 0.35 && (temperature === null ? weather?.season === 'winter' : temperature <= 0)
  const source = festiveSnow ? 'holiday' : seasonalCover > forecastCover ? 'season' : 'forecast'
  return { cover, forecastCover, frozen, source }
}

export function trailGroundSnow(
  landscape: Landscape,
  seed: number,
  region = { left: 0, top: 0, right: landscape.width, bottom: landscape.height },
) {
  const top = Math.max(region.top, landscape.meadowTop)
  if (top >= region.bottom) return ''
  const random = trailRandom(seed ^ 0x23680dab)
  let blanket = ''
  for (let x = region.left; x <= region.right; x += 8) {
    const y = Math.max(
      region.top,
      Math.round((landscape.meadowTop + groundOffset(x, landscape, landscape.meadowTop)) / 2) * 2,
    )
    blanket += `${blanket ? 'H' : 'M'}${x}${blanket ? 'V' : ' '}${y}`
  }
  blanket += `H${region.right}V${region.bottom}H${region.left}Z`
  const drifts: string[] = []
  for (let y = top + 8; y < region.bottom - 6; y += 24)
    for (let x = region.left + 4; x < region.right - 20; x += 40) {
      if (random() > 0.55) continue
      const left = Math.round((x + random() * 14) / 2) * 2
      const row = Math.round((y + random() * 8) / 2) * 2
      const span = 8 + Math.floor(random() * 7) * 2
      drifts.push(`M${left} ${row}h${span}v2h4v2h-${span + 8}v-2h4Z`)
    }
  return `<g class="trail-ground-snow" aria-hidden="true"><path class="trail-snow-blanket" d="${blanket}"/><path class="trail-snow-drift" d="${drifts.join('')}"/></g>`
}
