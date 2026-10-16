import type { Almanac, Sky } from '../components/scripts/404-almanac'
import type { TrailSeason } from './visitor-trail-weather'
import { smooth } from '../components/scripts/404-pixel'

// The year where the visitor is, from days since midwinter: how far the leaves have turned or
// fallen, how thick the flowers stand in the meadow, how far down the ranges the snow lies (as a
// height in scene units), and how much of it lies on the island.
export type Season = {
  autumn: number
  bare: number
  bloom: number
  // The trees in flower.
  blossom: number
  snowline: number
  ground: number
  // Hoarfrost on the island, and how far the sun has taken it back: at 0 it covers everything, at
  // 1 it holds only in the deepest shade.
  frost: number
  thaw: number
}

export type SeasonPhase = 'early' | 'mid' | 'late'

const OVERRIDE_WINTER_DAY: Record<TrailSeason, number> = {
  winter: 40,
  spring: 130,
  summer: 222,
  autumn: 300,
}

export function season(al: Almanac, sky: Sky, override?: TrailSeason | null): Season {
  const winterDay = override ? OVERRIDE_WINTER_DAY[override] : al.winterDay
  const near = (peak: number, width: number) => {
    const off = ((((winterDay - peak + 182.6) % 365.24) + 365.24) % 365.24) - 182.6
    return Math.max(0, 1 - Math.abs(off) / width)
  }
  // Lowest six weeks after midwinter, highest in late summer, and down to the water while it snows.
  let snowline = 435 + 165 * Math.cos((2 * Math.PI * (winterDay - 40)) / 365.24)
  const cold = sky.temp !== null && sky.temp < 1
  if (sky.fall === 'snow') snowline = Math.max(snowline, 760)
  else if (cold) snowline = Math.max(snowline, 560)
  const bare = near(40, 75)
  // Under a clear, still sky the grass radiates its warmth away and runs three to five degrees
  // colder than the air, so it rimes over with the air still above freezing. Cloud sends the warmth
  // back, and wind stirs the warmer air down onto it. The frost gathers in the shade first after
  // sunset; the morning sun takes it back from the slopes it strikes first, and by noon it holds
  // only where the sun never reaches.
  const still = (1 - smooth(0.2, 0.55, sky.cover)) * (1 - smooth(0.2, 0.45, sky.wind))
  const chill =
    sky.temp === null || sky.fall !== 'none' || sky.fog ? 0 : 1 - smooth(0, 4.5, sky.temp)
  const thaw =
    al.hour < 12
      ? Math.max(smooth(0, 20, al.sun.alt), smooth(9, 12, al.hour))
      : 1 - smooth(0, -18, al.sun.alt)
  const baseGround = sky.fall === 'snow' ? 0.5 + 0.5 * sky.heavy : cold && bare > 0.3 ? 0.35 : 0
  return {
    autumn: near(300, 55),
    bare,
    bloom: Math.max(0.15, 0.3 + 1.2 * near(130, 50) + 0.6 * near(200, 60) - 0.3 * bare),
    blossom: near(105, 35),
    snowline,
    ground: override === 'winter' ? Math.max(0.5, baseGround) : baseGround,
    frost: chill * still,
    thaw,
  }
}

export function seasonBucket(winterDay: number): { season: TrailSeason; phase: SeasonPhase } {
  const wrapped = ((winterDay % 365.24) + 365.24) % 365.24
  const d = wrapped >= 355 ? wrapped - 365.24 : wrapped
  const [currentSeason, start, span]: [TrailSeason, number, number] =
    d < 80
      ? ['winter', -10, 90]
      : d < 172
        ? ['spring', 80, 92]
        : d < 265
          ? ['summer', 172, 93]
          : ['autumn', 265, 90]
  const t = (d - start) / span
  const phase: SeasonPhase = t < 1 / 3 ? 'early' : t < 2 / 3 ? 'mid' : 'late'
  return { season: currentSeason, phase }
}
