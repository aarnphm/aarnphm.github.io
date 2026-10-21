import type { CoreBodyTemperatureCache } from '../core-body-temperature'
import type { GarminCache } from '../garmin'
import type { WeatherCache } from '../weather'
import type {
  HeatAcclimatisationState,
  HeatBlock,
  HeatConfidence,
  HeatDay,
  HeatMethod,
  HeatSauna,
  HeatTemperatureSource,
} from './schema'
import {
  normalizeKind,
  round,
  type RawStravaActivity,
  type StravaActivityDetail,
  type StravaRawCache,
} from '../strava'
import {
  coreHeatObservation,
  HEAT_STRAIN_THRESHOLD,
  type CoreTemperatureOrigin,
} from './distributions'
import { clamp, DAY_MS, dayMs } from './primitives'

const HEAT_THRESHOLD_C = 22

const HEAT_TARGET_MINUTES = 60
const HEAT_TARGET_DAYS = 14
const HEAT_DECAY_GRACE_DAYS = 3
const HEAT_DECAY_PER_DAY = 0.025
const HEAT_COVERAGE_DAYS = 42
const HEAT_HTL_PER_EXPOSURE = 10
const HEAT_PASSIVE_HTL_CAP = 8

interface HeatActivity {
  id: number
  date: string
  startedAt: string
  sport: 'bike' | 'run' | 'sauna'
  name: string
  durationS: number
  temperatureC: number | null
  heatStrainIndex: number | null
  sauna: HeatSauna | null
  hotS: number
  source: HeatTemperatureSource
  coreOrigin: CoreTemperatureOrigin | null
}

interface HeatDayBucket {
  observedS: number
  hotS: number
  saunaS: number
  saunaHtl: number | null
  coreObservedS: number
  coreTemperatureSeconds: number
  heatStrainSeconds: number
  ambientObservedS: number
  ambientTemperatureSeconds: number
  sources: Set<HeatTemperatureSource>
}

const heatMethod = (): HeatMethod => ({
  eligibleSports: ['bike', 'run', 'sauna'],
  hotThresholdC: HEAT_THRESHOLD_C,
  heatStrainThreshold: HEAT_STRAIN_THRESHOLD,
  targetMinutesPerDay: HEAT_TARGET_MINUTES,
  targetDays: HEAT_TARGET_DAYS,
  decayGraceDays: HEAT_DECAY_GRACE_DAYS,
  decayPerDay: HEAT_DECAY_PER_DAY,
  coverageDays: HEAT_COVERAGE_DAYS,
  htlPerExposure: HEAT_HTL_PER_EXPOSURE,
  passiveHtlCapPerDay: HEAT_PASSIVE_HTL_CAP,
  note: 'Runs prefer CORE recorded in Garmin FIT, with CORE app onboard data filling missing thermal metrics. Rides prefer CORE app onboard data, followed by Garmin FIT. WeatherKit ambient temperature fills activities without complete CORE heat data, then Strava device temperature fills remaining gaps. Manual sauna HTL contributes HTL / 10 to the Garden proxy, capped at 8 passive HTL per day and one combined daily exposure. Missing HTL earns no inferred credit. Sauna room conditions remain separate from measured HSI and workout temperature. This proxy is not the CORE Heat Adaptation Score.',
})

const emptyHeatSourceCounts = (): Record<HeatTemperatureSource, number> => ({
  core: 0,
  weatherkit: 0,
  strava: 0,
  'manual-sauna': 0,
})

const emptyCoreSourceCounts = (): Record<CoreTemperatureOrigin, number> => ({ app: 0, fit: 0 })

export function emptyHeat(): HeatBlock {
  return {
    currentPct: null,
    state: 'unexposed',
    confidence: 'none',
    coveragePct: 0,
    eligibleMinutes: 0,
    observedMinutes: 0,
    latestTemperatureC: null,
    lastObservedDate: null,
    lastHeatDate: null,
    heatMinutes14d: 0,
    heatDays14d: 0,
    saunaMinutes14d: 0,
    saunaHtl14d: null,
    saunaSessions14d: 0,
    sourceCounts: emptyHeatSourceCounts(),
    coreSourceCounts: emptyCoreSourceCounts(),
    activities: [],
    series: [],
    method: heatMethod(),
  }
}

export function buildHeat(
  cache: StravaRawCache,
  sourceActivities: RawStravaActivity[],
  weather: WeatherCache | null | undefined,
  coreCache: CoreBodyTemperatureCache | null | undefined,
  garmin: GarminCache | null | undefined,
  saunaDetails: readonly StravaActivityDetail[],
  windowFrom: number,
  windowTo: number,
): HeatBlock {
  const eligible: { date: string; durationS: number }[] = []
  const observed: HeatActivity[] = []
  const sourceCounts = emptyHeatSourceCounts()
  const coreSourceCounts = emptyCoreSourceCounts()

  for (const activity of sourceActivities) {
    const sport = normalizeKind(activity.sportType)
    if (sport !== 'bike' && sport !== 'run') continue
    const core = coreHeatObservation(activity, sport, coreCache, garmin)
    const route = cache.streams?.[String(activity.id)]?.latlng
    // Indoor CORE recordings do not need GPS; ambient fallbacks still require a route.
    if (!core && (!route || route.length < 2)) continue
    eligible.push({
      date: activity.startDateLocal.slice(0, 10),
      durationS: activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime,
    })

    if (
      core &&
      core.durationS > 0 &&
      core.coreTemperatureC != null &&
      core.heatStrainIndex != null
    ) {
      sourceCounts.core += 1
      coreSourceCounts[core.origin] += 1
      observed.push({
        id: activity.id,
        date: activity.startDateLocal.slice(0, 10),
        startedAt: activity.startDate,
        sport,
        name: activity.name,
        durationS: core.durationS,
        temperatureC: core.coreTemperatureC,
        heatStrainIndex: core.heatStrainIndex,
        sauna: null,
        hotS: core.hotS,
        source: 'core',
        coreOrigin: core.origin,
      })
      continue
    }

    const ambient = weather?.activities[String(activity.id)]
    const temperatureC = ambient?.temperatureC ?? activity.averageTemp ?? null
    if (temperatureC == null || !Number.isFinite(temperatureC)) continue
    const durationS = Math.max(
      1,
      ambient?.durationS ?? (activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime),
    )
    const source: HeatTemperatureSource = ambient?.temperatureC != null ? 'weatherkit' : 'strava'
    sourceCounts[source] += 1
    observed.push({
      id: activity.id,
      date: activity.startDateLocal.slice(0, 10),
      startedAt: activity.startDate,
      sport,
      name: activity.name,
      durationS,
      temperatureC,
      heatStrainIndex: null,
      sauna: null,
      hotS: temperatureC > HEAT_THRESHOLD_C ? durationS : 0,
      source,
      coreOrigin: null,
    })
  }

  for (const detail of saunaDetails) {
    if (dayMs(detail.date) < windowFrom || dayMs(detail.date) > windowTo) continue
    const durationS = detail.elapsedTimeS > 0 ? detail.elapsedTimeS : detail.movingTimeS
    if (!detail.sauna || !Number.isFinite(durationS) || durationS <= 0) continue
    const rawHtl = detail.sauna.heatTrainingLoad
    const heatTrainingLoad =
      rawHtl != null && Number.isFinite(rawHtl) && rawHtl >= 0 ? rawHtl : null
    eligible.push({ date: detail.date, durationS })
    sourceCounts['manual-sauna'] += 1
    observed.push({
      id: detail.id,
      date: detail.date,
      startedAt: detail.start,
      sport: 'sauna',
      name: detail.name,
      durationS: heatTrainingLoad == null ? 0 : durationS,
      temperatureC: null,
      heatStrainIndex: null,
      sauna: { ...detail.sauna, heatTrainingLoad, durationMinutes: durationS / 60 },
      hotS: 0,
      source: 'manual-sauna',
      coreOrigin: null,
    })
  }

  const byDay = new Map<string, HeatDayBucket>()
  for (const activity of observed) {
    const bucket = byDay.get(activity.date) ?? {
      observedS: 0,
      hotS: 0,
      saunaS: 0,
      saunaHtl: null,
      coreObservedS: 0,
      coreTemperatureSeconds: 0,
      heatStrainSeconds: 0,
      ambientObservedS: 0,
      ambientTemperatureSeconds: 0,
      sources: new Set<HeatTemperatureSource>(),
    }
    bucket.observedS += activity.durationS
    bucket.hotS += activity.hotS
    if (activity.sauna) {
      bucket.saunaS += activity.sauna.durationMinutes * 60
      if (activity.sauna.heatTrainingLoad != null)
        bucket.saunaHtl = (bucket.saunaHtl ?? 0) + activity.sauna.heatTrainingLoad
    } else if (activity.source === 'core' && activity.temperatureC != null) {
      bucket.coreObservedS += activity.durationS
      bucket.coreTemperatureSeconds += activity.temperatureC * activity.durationS
      bucket.heatStrainSeconds += (activity.heatStrainIndex ?? 0) * activity.durationS
    } else if (activity.temperatureC != null) {
      bucket.ambientObservedS += activity.durationS
      bucket.ambientTemperatureSeconds += activity.temperatureC * activity.durationS
    }
    bucket.sources.add(activity.source)
    byDay.set(activity.date, bucket)
  }

  let credits = 0
  let lastHeatMs: number | null = null
  let lastHeatDate: string | null = null
  const series: HeatDay[] = []
  for (let ms = windowFrom; ms <= windowTo; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const bucket = byDay.get(date)
    const observedMinutes = (bucket?.observedS ?? 0) / 60
    const hotMinutes = (bucket?.hotS ?? 0) / 60
    const passiveDose =
      clamp(bucket?.saunaHtl ?? 0, 0, HEAT_PASSIVE_HTL_CAP) / HEAT_HTL_PER_EXPOSURE
    const dose = clamp(hotMinutes / HEAT_TARGET_MINUTES + passiveDose, 0, 1)
    if (dose > 0) {
      credits = Math.min(HEAT_TARGET_DAYS, credits + dose)
      lastHeatMs = ms
      lastHeatDate = date
    } else if (
      lastHeatMs != null &&
      Math.round((ms - lastHeatMs) / DAY_MS) > HEAT_DECAY_GRACE_DAYS
    ) {
      credits *= 1 - HEAT_DECAY_PER_DAY
    }
    series.push({
      date,
      temperatureC:
        bucket && bucket.coreObservedS > 0
          ? round(bucket.coreTemperatureSeconds / bucket.coreObservedS, 2)
          : bucket && bucket.ambientObservedS > 0
            ? round(bucket.ambientTemperatureSeconds / bucket.ambientObservedS, 1)
            : null,
      heatStrainIndex:
        bucket && bucket.coreObservedS > 0
          ? round(bucket.heatStrainSeconds / bucket.coreObservedS, 1)
          : null,
      source:
        bucket == null || bucket.sources.size === 0
          ? null
          : bucket.sources.size > 1
            ? 'mixed'
            : (bucket.sources.values().next().value ?? null),
      observedMinutes: round(observedMinutes, 0),
      hotMinutes: round(hotMinutes, 0),
      saunaMinutes: round((bucket?.saunaS ?? 0) / 60, 0),
      saunaHtl: bucket?.saunaHtl == null ? null : round(bucket.saunaHtl, 1),
      dose: round(dose, 3),
      acclimatisationPct: round((credits / HEAT_TARGET_DAYS) * 100, 1),
    })
  }

  const latestObserved =
    [...series].reverse().find(day => (byDay.get(day.date)?.observedS ?? 0) > 0) ?? null
  const currentPct = observed.length ? (series[series.length - 1]?.acclimatisationPct ?? 0) : null
  const coverageFrom = windowTo - (HEAT_COVERAGE_DAYS - 1) * DAY_MS
  const eligibleRecent = eligible.filter(activity => dayMs(activity.date) >= coverageFrom)
  const observedRecent = observed.filter(
    activity => dayMs(activity.date) >= coverageFrom && activity.durationS > 0,
  )
  const eligibleSeconds = eligibleRecent.reduce((total, activity) => total + activity.durationS, 0)
  const observedSeconds = observedRecent.reduce((total, activity) => total + activity.durationS, 0)
  const coveragePct =
    eligibleSeconds > 0 ? round(clamp(observedSeconds / eligibleSeconds, 0, 1) * 100, 0) : 0
  const latestEligibleDate =
    eligibleRecent
      .map(activity => activity.date)
      .sort()
      .at(-1) ?? null
  const confidence: HeatConfidence =
    observedRecent.length === 0
      ? 'none'
      : coveragePct < 50 || observedRecent.length < 2 || latestObserved?.date !== latestEligibleDate
        ? 'low'
        : 'moderate'
  const recent14 = series.slice(-14)
  const heatMinutes14d = round(
    recent14.reduce((total, day) => total + day.hotMinutes, 0),
    0,
  )
  const heatDays14d = recent14.filter(day => day.dose > 0).length
  const daysSinceHeat = lastHeatDate ? Math.round((windowTo - dayMs(lastHeatDate)) / DAY_MS) : null
  const state: HeatAcclimatisationState =
    currentPct == null || currentPct < 10
      ? 'unexposed'
      : daysSinceHeat != null && daysSinceHeat > HEAT_DECAY_GRACE_DAYS
        ? 'decaying'
        : currentPct >= 80
          ? 'maintaining'
          : 'acclimating'

  return {
    currentPct,
    state,
    confidence,
    coveragePct,
    eligibleMinutes: round(eligibleSeconds / 60, 0),
    observedMinutes: round(observedSeconds / 60, 0),
    latestTemperatureC: latestObserved?.temperatureC ?? null,
    lastObservedDate: latestObserved?.date ?? null,
    lastHeatDate,
    heatMinutes14d,
    heatDays14d,
    saunaMinutes14d: round(
      recent14.reduce((total, day) => total + day.saunaMinutes, 0),
      0,
    ),
    saunaHtl14d: recent14.some(day => day.saunaHtl != null)
      ? round(
          recent14.reduce((total, day) => total + (day.saunaHtl ?? 0), 0),
          1,
        )
      : null,
    saunaSessions14d: observed.filter(
      activity => activity.sauna != null && dayMs(activity.date) >= windowTo - 13 * DAY_MS,
    ).length,
    sourceCounts,
    coreSourceCounts,
    activities: [...observed]
      .sort((a, b) => a.startedAt.localeCompare(b.startedAt))
      .map(activity => {
        const observedMinutes = activity.durationS / 60
        const hotMinutes = activity.hotS / 60
        return {
          id: activity.id,
          date: activity.date,
          startedAt: activity.startedAt,
          sport: activity.sport,
          name: activity.name,
          temperatureC:
            activity.temperatureC == null
              ? null
              : round(activity.temperatureC, activity.source === 'core' ? 2 : 1),
          heatStrainIndex:
            activity.heatStrainIndex == null ? null : round(activity.heatStrainIndex, 1),
          source: activity.source,
          coreOrigin: activity.coreOrigin,
          sauna: activity.sauna,
          observedMinutes: round(observedMinutes, 0),
          hotMinutes: round(hotMinutes, 0),
          dose: round(
            activity.sauna
              ? clamp(activity.sauna.heatTrainingLoad ?? 0, 0, HEAT_PASSIVE_HTL_CAP) /
                  HEAT_HTL_PER_EXPOSURE
              : clamp(hotMinutes / HEAT_TARGET_MINUTES, 0, 1),
            3,
          ),
        }
      }),
    series,
    method: heatMethod(),
  }
}
