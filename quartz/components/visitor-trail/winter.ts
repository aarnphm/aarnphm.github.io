import type { TrailWeather } from '../../util/visitor-trail-weather'
import { visitorPrecipitation } from '../../util/visitor-precipitation'
import { type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

export function trailWinter(weather: TrailWeather | null, festiveSnow = false) {
  const snowing = visitorPrecipitation(weather?.code).fall === 'snow'
  const temperature = weather?.temperatureC ?? null
  const forecastCover = Math.max(
    snowing ? (weather?.code === 75 || weather?.code === 86 ? 0.95 : 0.65) : 0,
    temperature !== null && temperature <= 2 ? Math.min(1, (weather?.snowCm ?? 0) / 2.5) : 0,
  )
  // Seasonal ground cover is illustrative; the forecast only supplies recent snowfall.
  const seasonalCover =
    weather?.season === 'winter' && (temperature === null || temperature <= 4) ? 0.78 : 0
  const weatherCover = Math.max(forecastCover, seasonalCover)
  const cover = Math.max(weatherCover, festiveSnow ? 0.9 : 0)
  const forecastFrozen =
    weatherCover >= 0.35 && (temperature === null ? weather?.season === 'winter' : temperature <= 0)
  const frozen = festiveSnow || seasonalCover > 0 || forecastFrozen
  const source = festiveSnow ? 'holiday' : seasonalCover > forecastCover ? 'season' : 'forecast'
  return { cover, forecastCover, frozen, forecastFrozen, source }
}

export function trailSnowBase(
  landscape: Landscape,
  seed: number,
  region = { left: 0, top: 0, right: landscape.width, bottom: landscape.height },
) {
  const edge = landscape.alpine.base
  if (Math.max(region.top, edge) >= region.bottom) return ''
  let blanket = ''
  for (let x = region.left; x <= region.right; x += 4) {
    const column = Math.floor(x / 12)
    const random = trailRandom(seed ^ Math.imul(column + 1, 0x45d9f3b))
    const y = Math.max(region.top, edge + Math.floor(random() * 5) * 2)
    blanket += `${blanket ? 'H' : 'M'}${x}${blanket ? 'V' : ' '}${y}`
  }
  blanket += `H${region.right}V${region.bottom}H${region.left}Z`
  return `<g class="trail-ground-snow" aria-hidden="true"><path class="trail-snow-blanket" d="${blanket}"/>${snowDunes(landscape, seed, region)}</g>`
}

type SnowRegion = { left: number; top: number; right: number; bottom: number }

function clippedSnow(points: Point[], region: SnowRegion) {
  // Clip each face in world coordinates so adjacent patches share one opaque edge.
  for (const [axis, edge, direction] of [
    ['x', region.left, 1],
    ['x', region.right, -1],
    ['y', region.top, 1],
    ['y', region.bottom, -1],
  ] satisfies ['x' | 'y', number, number][]) {
    const clipped: Point[] = []
    for (let i = 0; i < points.length; i++) {
      const a = points[i]
      const b = points[(i + 1) % points.length]
      const insideA = (a[axis] - edge) * direction >= 0
      const insideB = (b[axis] - edge) * direction >= 0
      if (insideA) clipped.push(a)
      if (insideA !== insideB) {
        const t = (edge - a[axis]) / (b[axis] - a[axis])
        clipped.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
      }
    }
    points = clipped
  }
  return points.length < 3
    ? ''
    : `${points.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join('')}Z`
}

type SnowFaces = { body: string; shade: string; cap: string }

function snowMound(
  seed: number,
  center: Point,
  width: number,
  height: number,
  region: SnowRegion,
): SnowFaces {
  const random = trailRandom(seed)
  const phase = random() * Math.PI * 2
  const tilt = (random() - 0.5) * 0.07
  const snap = (v: number) => Math.round(v / 2) * 2
  const outline = (scale: number, dx = 0, dy = 0) =>
    Array.from({ length: 64 }, (_, i) => {
      const angle = (i / 64) * Math.PI * 2
      const lobe = 1 + 0.12 * Math.sin(angle * 3 + phase) + 0.06 * Math.cos(angle * 5 - phase)
      const x = Math.cos(angle) * width * 0.5 * lobe * scale
      return {
        x: snap(center.x + dx + x),
        y: snap(center.y + dy + Math.sin(angle) * height * 0.5 * lobe * scale + x * tilt),
      }
    })
  const outer = outline(1)
  const foot = outer.slice(0, 33)
  const inner = foot.map((point, i) => ({
    x: point.x,
    y: point.y - Math.sin((i / 32) * Math.PI) * height * 0.16,
  }))
  return {
    body: clippedSnow(outer, region),
    shade: clippedSnow([...foot, ...inner.toReversed()], region),
    cap: clippedSnow(outline(0.64, -width * 0.08, -height * 0.12), region),
  }
}

function snowFaces(faces: SnowFaces[]) {
  return `<path class="trail-snow-dune" d="${faces.map(f => f.body).join('')}"/><path class="trail-snow-lee" d="${faces.map(f => f.shade).join('')}"/><path class="trail-snow-crest" d="${faces.map(f => f.cap).join('')}"/>`
}

export function trailSnowBank(seed: number, width: number, height: number) {
  const region = { left: -width, right: width, top: -height, bottom: height }
  return `<g class="trail-ground-snow trail-snow-bank" aria-hidden="true">${snowFaces([
    snowMound(seed, { x: width * 0.1, y: 0 }, width, height, region),
    snowMound(
      seed ^ 0x632be5ab,
      { x: -width * 0.22, y: height * 0.12 },
      width * 0.55,
      height * 0.7,
      region,
    ),
  ])}</g>`
}

function snowDunes(landscape: Landscape, seed: number, region: SnowRegion) {
  const faces: SnowFaces[] = []
  const origin = landscape.meadowTop + 80
  // Large seed tiles keep expansion stable; cluster centres range over the whole tile.
  for (
    let row = Math.max(0, Math.floor((region.top - origin - 160) / 460));
    row <= Math.ceil((region.bottom - origin + 160) / 460);
    row++
  ) {
    for (
      let column = Math.floor((region.left - 300) / 620);
      column <= Math.ceil((region.right + 300) / 620);
      column++
    ) {
      const tileSeed = seed ^ Math.imul(row + 1, 0x45d9f3b) ^ Math.imul(column + 1, 0x27d4eb2d)
      const random = trailRandom(tileSeed)
      const clusters = Math.floor(random() * 3)
      for (let cluster = 0; cluster < clusters; cluster++) {
        const cx = column * 620 + random() * 620
        const cy = origin + row * 460 + random() * 460
        const count = 1 + Math.floor(random() * 3)
        for (let i = 0; i < count; i++) {
          const x = cx + (random() - 0.5) * 230
          const y = Math.max(origin, cy + (random() - 0.5) * 110)
          const width = 64 + random() ** 2 * 180
          const height = 12 + random() * 20
          faces.push(
            snowMound(
              tileSeed ^ Math.imul(cluster * 5 + i + 1, 16807),
              { x, y },
              width,
              height,
              region,
            ),
          )
        }
      }
    }
  }
  return snowFaces(faces)
}
