import type { StravaActivityDetail } from '../plugins/stores/strava'

export type BestEffortSport = 'run' | 'bike'

export type BestEffortGroup = 'distance' | 'longest' | 'elevation' | 'power'

/** What `BestEffortEntry.value` measures: elapsed seconds over a fixed distance, ride km, ride elevation gain in m, or mean watts over a fixed duration. */
export type BestEffortUnit = 'seconds' | 'km' | 'm' | 'watts'

export interface BestEffortEntry {
  /** Strava activity id. */
  id: number
  /** Local activity date, YYYY-MM-DD. */
  date: string
  name: string
  value: number
  /** Elapsed seconds: the effort window, the power window, or the whole ride for longest and elevation. */
  timeS: number | null
  /** Target distance for distance efforts, the whole ride for longest and elevation, null for power windows. */
  distanceKm: number | null
  heartRate: number | null
  wattsPerKg: number | null
  /** Virtual ride, trainer or treadmill session; only power windows keep these. */
  indoor: boolean
  /** Rank among every activity in the category, 1 = best; ties go to the earlier date. */
  rank: number
  /** Rank within the entry's calendar year, same tie rule. */
  yearRank: number
  /** Beat every earlier entry when it was set: the best-to-date staircase. */
  pr: boolean
}

export interface BestEffortCategory {
  /** Stable key: `run:5K`, `bike:longest`, `bike:elevation`, `bike:distance:40K`, `bike:power:300`. */
  key: string
  sport: BestEffortSport
  group: BestEffortGroup
  /** Store label for distance efforts ('5K', 'Half marathon'), null for the other groups. */
  label: string | null
  unit: BestEffortUnit
  better: 'lower' | 'higher'
  distanceM: number | null
  durationS: number | null
  /** One entry per activity, ascending by date then id. */
  efforts: BestEffortEntry[]
}

export interface BestEffortsBlock {
  /** Calendar years holding at least one effort, descending. */
  years: number[]
  /** Run categories by distance, then bike: longest, elevation, distance ascending, power ascending. Empty categories are dropped. */
  categories: BestEffortCategory[]
}

/** Power windows ranked as ride best efforts; a subset of the store's per-activity windows. */
export const BEST_EFFORT_POWER_DURATIONS = [
  5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 5400, 7200,
] as const

export const emptyBestEfforts = (): BestEffortsBlock => ({ years: [], categories: [] })

/** Strava's own activity fields; the detail's distance is rounded and its elevation sums unthresholded altitude steps. */
export interface BestEffortNative {
  trainer: boolean
  distanceM: number
  elapsedTimeS: number
  elevationGainM: number
}

type CategoryHead = Omit<BestEffortCategory, 'efforts'>
type Effort = Omit<BestEffortEntry, 'rank' | 'yearRank' | 'pr'>
type Better = BestEffortCategory['better']

const POWER_DURATIONS = new Set<number>(BEST_EFFORT_POWER_DURATIONS)
const GROUP_ORDER: Record<BestEffortGroup, number> = {
  longest: 0,
  elevation: 1,
  distance: 2,
  power: 3,
}

const LONGEST: CategoryHead = {
  key: 'bike:longest',
  sport: 'bike',
  group: 'longest',
  label: null,
  unit: 'km',
  better: 'higher',
  distanceM: null,
  durationS: null,
}

const ELEVATION: CategoryHead = { ...LONGEST, key: 'bike:elevation', group: 'elevation', unit: 'm' }

const distanceHead = (sport: BestEffortSport, label: string, distanceM: number): CategoryHead => ({
  key: sport === 'run' ? `run:${label}` : `bike:distance:${label}`,
  sport,
  group: 'distance',
  label,
  unit: 'seconds',
  better: 'lower',
  distanceM,
  durationS: null,
})

const powerHead = (durationS: number): CategoryHead => ({
  key: `bike:power:${durationS}`,
  sport: 'bike',
  group: 'power',
  label: null,
  unit: 'watts',
  better: 'higher',
  distanceM: null,
  durationS,
})

const categoryOrder = (a: CategoryHead, b: CategoryHead): number =>
  Number(a.sport === 'bike') - Number(b.sport === 'bike') ||
  GROUP_ORDER[a.group] - GROUP_ORDER[b.group] ||
  (a.distanceM ?? 0) - (b.distanceM ?? 0) ||
  (a.durationS ?? 0) - (b.durationS ?? 0)

/** Positive when `a` holds the better value. */
const margin = (better: Better, a: Effort, b: Effort): number =>
  better === 'lower' ? b.value - a.value : a.value - b.value

const chronological = (a: Effort, b: Effort): number => a.date.localeCompare(b.date) || a.id - b.id

const rankEfforts = (better: Better, efforts: Effort[]): BestEffortEntry[] => {
  const ranks = new Map<Effort, { rank: number; yearRank: number }>()
  const yearCounts = new Map<string, number>()
  const byRank = [...efforts].sort((a, b) => margin(better, b, a) || chronological(a, b))
  for (const [index, effort] of byRank.entries()) {
    const year = effort.date.slice(0, 4)
    const yearRank = (yearCounts.get(year) ?? 0) + 1
    yearCounts.set(year, yearRank)
    ranks.set(effort, { rank: index + 1, yearRank })
  }
  let best: Effort | null = null
  return efforts.sort(chronological).map(effort => {
    const pr = best === null || margin(better, effort, best) > 0
    if (pr) best = effort
    return { ...effort, ...ranks.get(effort)!, pr }
  })
}

export const buildBestEffortsBlock = (
  details: readonly StravaActivityDetail[],
  natives: ReadonlyMap<number, BestEffortNative> = new Map(),
): BestEffortsBlock => {
  const categories = new Map<string, { head: CategoryHead; efforts: Effort[] }>()
  const add = (head: CategoryHead, effort: Effort): void => {
    if (!Number.isFinite(effort.value) || effort.value <= 0) return
    const category = categories.get(head.key) ?? { head, efforts: [] }
    category.efforts.push(effort)
    categories.set(head.key, category)
  }

  for (const detail of details) {
    const sport = detail.sport
    if (sport !== 'run' && sport !== 'bike') continue
    const native = natives.get(detail.id)
    const activity = {
      id: detail.id,
      date: detail.date,
      name: detail.name,
      indoor: detail.virtual === true || native?.trainer === true,
    }
    if (!activity.indoor)
      for (const effort of detail.bestEfforts?.distance ?? [])
        add(distanceHead(sport, effort.label, effort.targetDistanceM), {
          ...activity,
          value: effort.elapsedTimeS,
          timeS: effort.elapsedTimeS,
          distanceKm: effort.targetDistanceM / 1000,
          heartRate: effort.averageHeartRate,
          wattsPerKg: null,
        })
    if (sport !== 'bike') continue

    // Virtual-world and trainer distance is modeled, so only power ranks indoor rides.
    if (!activity.indoor) {
      const distanceKm = native ? native.distanceM / 1000 : detail.distanceKm
      const ride = {
        ...activity,
        timeS: native?.elapsedTimeS ?? detail.elapsedTimeS,
        distanceKm,
        heartRate: detail.avgHr,
        wattsPerKg: null,
      }
      add(LONGEST, { ...ride, value: distanceKm })
      add(ELEVATION, { ...ride, value: native?.elevationGainM ?? detail.elevationM })
    }
    for (const effort of detail.bestEfforts?.power ?? [])
      if (POWER_DURATIONS.has(effort.durationS))
        add(powerHead(effort.durationS), {
          ...activity,
          value: effort.averageWatts,
          timeS: effort.durationS,
          distanceKm: null,
          heartRate: effort.averageHeartRate,
          wattsPerKg: effort.wattsPerKg,
        })
  }

  const ranked = [...categories.values()]
    .sort((a, b) => categoryOrder(a.head, b.head))
    .map(({ head, efforts }) => ({ ...head, efforts: rankEfforts(head.better, efforts) }))
  const years = new Set(ranked.flatMap(c => c.efforts.map(e => Number(e.date.slice(0, 4)))))
  return { years: [...years].sort((a, b) => b - a), categories: ranked }
}
