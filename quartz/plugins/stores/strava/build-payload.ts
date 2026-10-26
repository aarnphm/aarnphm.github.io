import type { OuraCache } from '../oura'
import type { ActivityTrackingEntry } from '../tracking'
import type { WeatherCache } from '../weather'
import { estimateWahooCyclingStamina } from '../../../util/cycling-stamina'
import {
  cyclingMechanics,
  cyclingTorqueSamples,
  type CyclingTorqueHistory,
  type CyclingTorqueSample,
} from '../../../util/cycling-torque'
import { localIsoDay } from '../../../util/local-date'
import { latestProviderSync } from '../../../util/provider-sync'
import {
  CRITICAL_POWER_DURATIONS_S,
  fitCriticalPower,
  type CriticalPowerAnchor,
  type CriticalPowerEstimate,
} from '../critical-power'
import {
  matchGarminActivity,
  matchGarminHeartRateActivity,
  matchGarminTrainingEffectActivity,
  type GarminActivityMatch,
  type GarminCache,
  type GarminStreams,
} from '../garmin'
import { matchWahooActivity, type WahooActivityMatch, type WahooData } from '../wahoo'
import { projectActivityDetail, type ActivityProjectionScope } from './activity-projection'
import {
  emptyHealth,
  emptyPayload,
  emptyTotals,
  toHealth,
  type ActivityHealth,
  type StravaActivityDetail,
  type StravaDay,
  type StravaDayItem,
  type StravaPayload,
} from './activity-types'
import {
  bestObservedPowerWindows,
  effortTimeline,
  meanMaxCurve,
  mergeMaxCurves,
  type PowerCurvePoint,
  type PowerCurveSource,
} from './efforts-and-curves'
import { garminRespirationProjectionModel } from './provider-metadata'
import {
  DAY_MS,
  isTreatment,
  normalizeKind,
  WINDOW_DAYS,
  type ActivityHeartRate,
  type ActivityKind,
  type RawStravaActivity,
  type StravaRawCache,
  type StravaStreams,
} from './raw-and-kinds'
import { deriveHrBounds, derivePowerBounds, inferRouteHome, round } from './stream-primitives'
import {
  applyActivityTracking,
  mergeWahooTelemetry,
  resolveActivityHeartRate,
  selectEffortStreams,
  selectStreams,
} from './stream-selection'

export function buildPayload(
  cache: StravaRawCache | null,
  oura: OuraCache | null,
  garmin: GarminCache | null,
  since?: string,
  weather?: WeatherCache | null,
  inputFtp?: number | null,
  inputHrBounds?: number[] | null,
  timeZone?: string,
  wahoo?: WahooData | null,
  inputMaxHeartRate?: number | null,
  inputLactateThresholdHeartRate?: number | null,
  inputGeneratedAt?: number,
  activityTracking: readonly ActivityTrackingEntry[] = [],
): StravaPayload {
  if (!cache) return emptyPayload()
  const originalCache = cache
  cache = applyActivityTracking(cache, garmin, activityTracking, wahoo ?? null) ?? cache
  const trackingById = new Map(activityTracking.map(entry => [entry.activityId, entry]))
  const resolveGarmin = (activity: RawStravaActivity, sport: ActivityKind) =>
    matchGarminActivity(
      originalCache.activities[String(activity.id)] ?? activity,
      sport,
      garmin,
      trackingById.get(activity.id)?.garminActivityId,
    )

  const generatedAt = latestProviderSync(inputGeneratedAt, cache, oura, garmin, weather, wahoo)

  const sinceDay = since && /^\d{4}-\d{2}-\d{2}$/.test(since) ? since : null
  const allActivities = Object.values(cache.activities)
    .map(a => ({
      a,
      sport: isTreatment(a.sportType, a.name)
        ? ('treatment' as ActivityKind)
        : normalizeKind(a.sportType),
    }))
    .filter((x): x is { a: RawStravaActivity; sport: ActivityKind } => x.sport !== null)
  const activities = allActivities
    .filter(x => !sinceDay || x.a.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((p, q) => p.a.startDateLocal.localeCompare(q.a.startDateLocal))

  if (activities.length === 0) return { ...emptyPayload(cache.athleteId), generatedAt }

  const garminMatches = new Map<string, GarminActivityMatch | null>()
  const garminTrainingEffectMatches = new Map<string, GarminActivityMatch | null>()
  const garminHeartRateMatches = new Map<string, GarminActivityMatch | null>()
  const wahooMatches = new Map<string, WahooActivityMatch | null>()
  const selectedStreams = new Map<string, StravaStreams | GarminStreams | undefined>()
  const heartRates = new Map<string, ActivityHeartRate>()
  const respirationProjection = garminRespirationProjectionModel(garmin)
  for (const { a, sport } of activities) {
    const id = String(a.id)
    const original = originalCache.activities[id] ?? a
    const match = resolveGarmin(a, sport)
    const trainingEffectMatch =
      trackingById.get(a.id)?.garminActivityId != null
        ? match
        : matchGarminTrainingEffectActivity(a, sport, garmin, match)
    const hrMatch = matchGarminHeartRateActivity(a, sport, garmin)
    const wahooMatch = matchWahooActivity(
      original,
      sport,
      wahoo ?? null,
      trackingById.get(a.id)?.wahooFitPath,
    )
    const primary =
      wahooMatch && wahoo
        ? mergeWahooTelemetry(
            a,
            cache.streams?.[id],
            wahooMatch,
            wahoo,
            trackingById.get(a.id)?.power,
          )
        : cache.streams?.[id]
    const streams = selectStreams(primary, match, garmin, a)
    garminMatches.set(id, match)
    garminTrainingEffectMatches.set(id, trainingEffectMatch)
    garminHeartRateMatches.set(id, hrMatch)
    wahooMatches.set(id, wahooMatch)
    selectedStreams.set(id, streams)
    heartRates.set(
      id,
      resolveActivityHeartRate(
        a,
        sport,
        a.sportType.startsWith('Virtual')
          ? selectEffortStreams(cache.streams?.[id], streams)
          : streams,
        hrMatch,
        garmin,
      ),
    )
  }

  const totals = emptyTotals()
  const strengthTotal = { count: 0, movingTimeS: 0 }
  const byDate = new Map<string, StravaDayItem[]>()
  for (const { a, sport } of activities) {
    const t = totals.find(x => x.sport === sport)
    if (t) {
      t.count += 1
      t.distanceKm += a.distance / 1000
      t.movingTimeS += a.movingTime
      t.elevationM += a.totalElevationGain
    }
    if (sport === 'strength') {
      strengthTotal.count += 1
      strengthTotal.movingTimeS += a.movingTime
    }

    const date = a.startDateLocal.slice(0, 10)
    const items = byDate.get(date) ?? []
    items.push({
      id: a.id,
      sport,
      distanceKm: round(a.distance / 1000, 1),
      durationS: a.movingTime,
    })
    byDate.set(date, items)
  }

  const dayMs = (iso: string): number => Date.parse(`${iso}T00:00:00Z`)
  const firstMs = dayMs(activities[0].a.startDateLocal.slice(0, 10))
  const lastActMs = dayMs(activities[activities.length - 1].a.startDateLocal.slice(0, 10))
  const end = generatedAt ? dayMs(localIsoDay(generatedAt, timeZone)) : lastActMs
  const start = sinceDay ? dayMs(sinceDay) : Math.max(firstMs, end - (WINDOW_DAYS - 1) * DAY_MS)
  const days: StravaDay[] = []
  for (let ms = start; ms <= end; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const items = byDate.get(date) ?? []
    const dominant = items.reduce<StravaDayItem | null>(
      (best, item) => (item.distanceKm > (best?.distanceKm ?? -1) ? item : best),
      null,
    )
    days.push({
      date,
      durationS: items.reduce((s, item) => s + item.durationS, 0),
      items,
      dominant: dominant?.sport ?? null,
    })
  }

  const finalized = totals.map(t => ({
    ...t,
    distanceKm: round(t.distanceKm, 1),
    elevationM: Math.round(t.elevationM),
  }))

  let hrmax = 0
  for (const { a } of activities) {
    const maxHr = heartRates.get(String(a.id))?.maxHr
    if ((maxHr ?? 0) > hrmax) hrmax = maxHr ?? 0
  }
  if (hrmax < 100) hrmax = 190
  const recentCut = end - 41 * DAY_MS
  const powerCurveYear = new Date(end).getUTCFullYear()
  const yearCut = dayMs(`${powerCurveYear}-01-01`)
  const detailIds = new Set(activities.map(({ a }) => String(a.id)))
  let best20 = 0
  const powerCurves = new Map<string, PowerCurvePoint[]>()
  const recentCurves: PowerCurveSource[] = []
  const yearCurves: PowerCurveSource[] = []
  const recentRunCurves: PowerCurveSource[] = []
  const yearRunCurves: PowerCurveSource[] = []
  const torqueSamplesByActivity = new Map<string, CyclingTorqueSample[]>()
  const torqueHistory: CyclingTorqueHistory[] = []
  const criticalPowerAnchorsByActivity = new Map<string, CriticalPowerAnchor[]>()
  const activityCriticalPowers = new Map<string, CriticalPowerEstimate>()
  const recentCriticalPowerAnchors: CriticalPowerAnchor[] = []
  const yearCriticalPowerAnchors: CriticalPowerAnchor[] = []
  for (const { a, sport } of allActivities) {
    if ((sport !== 'bike' && sport !== 'run') || a.commute === true) continue
    const id = String(a.id)
    const activityDay = dayMs(a.startDateLocal.slice(0, 10))
    const inRecentWindow = activityDay >= recentCut && activityDay <= end
    const inYear = activityDay >= yearCut && activityDay <= end
    if (!detailIds.has(id) && !inRecentWindow && !inYear) continue
    const selected =
      selectedStreams.get(id) ??
      selectStreams(cache.streams?.[id], resolveGarmin(a, sport), garmin, a)
    if (!selectedStreams.has(id)) selectedStreams.set(id, selected)
    const streams = selectEffortStreams(cache.streams?.[id], selected)
    if (!streams?.watts?.some(v => v > 0)) continue
    const timeline = effortTimeline(streams, a.movingTime)
    const c = meanMaxCurve(timeline)
    powerCurves.set(id, c)
    if (sport === 'bike' && detailIds.has(id)) {
      const p20 = c.find(p => p.s === 1200)
      if (p20 && p20.w > best20) best20 = p20.w
    }
    const source = { activityId: a.id, activityDate: a.startDateLocal.slice(0, 10), curve: c }
    if (sport === 'run') {
      if (inRecentWindow) recentRunCurves.push(source)
      if (inYear) yearRunCurves.push(source)
      continue
    }
    if (inRecentWindow) recentCurves.push(source)
    if (inYear) yearCurves.push(source)
    const torqueSource =
      wahooMatches.get(id) ??
      matchWahooActivity(
        originalCache.activities[id] ?? a,
        sport,
        wahoo ?? null,
        trackingById.get(a.id)?.wahooFitPath,
      )
    const torqueSamples = torqueSource
      ? cyclingTorqueSamples(
          wahoo?.streams[torqueSource.activity.id],
          (Date.parse(torqueSource.activity.startDate) - Date.parse(a.startDate)) / 1_000,
          a.elapsedTime,
        )
      : []
    torqueSamplesByActivity.set(id, torqueSamples)
    const rideMechanics = cyclingMechanics(torqueSamples, 0, a.elapsedTime)
    if (rideMechanics)
      torqueHistory.push({ startMs: Date.parse(a.startDate), mechanics: rideMechanics })
    if (a.deviceWatts && timeline) {
      const activityDate = a.startDateLocal.slice(0, 10)
      const anchors = bestObservedPowerWindows(timeline, CRITICAL_POWER_DURATIONS_S).map(
        (window): CriticalPowerAnchor => ({
          durationS: window.durationS,
          meanPowerWatts: window.averageWatts,
          activityId: a.id,
          activityDate,
          startElapsedS: window.start,
          endElapsedS: window.end,
          mechanics: cyclingMechanics(torqueSamples, window.start, window.end),
        }),
      )
      criticalPowerAnchorsByActivity.set(id, anchors)
      const activityCriticalPower = fitCriticalPower(
        anchors,
        'activity',
        activityDate,
        activityDate,
      )
      if (activityCriticalPower) activityCriticalPowers.set(id, activityCriticalPower)
      if (inRecentWindow) recentCriticalPowerAnchors.push(...anchors)
      if (inYear) yearCriticalPowerAnchors.push(...anchors)
    }
  }
  const powerCurveRef = mergeMaxCurves(recentCurves)
  const powerCurveYearRef = mergeMaxCurves(yearCurves)
  const runPowerCurveRef = mergeMaxCurves(recentRunCurves)
  const runPowerCurveYearRef = mergeMaxCurves(yearRunCurves)
  const windowTo = new Date(end).toISOString().slice(0, 10)
  const criticalPower = fitCriticalPower(
    recentCriticalPowerAnchors,
    'six-weeks',
    new Date(recentCut).toISOString().slice(0, 10),
    windowTo,
  )
  const criticalPowerYear = fitCriticalPower(
    yearCriticalPowerAnchors,
    'calendar-year',
    `${powerCurveYear}-01-01`,
    windowTo,
  )
  const powerCurveActivityIds = new Set([
    ...[
      ...powerCurveRef,
      ...powerCurveYearRef,
      ...runPowerCurveRef,
      ...runPowerCurveYearRef,
    ].flatMap(point => (point.activityId == null ? [] : [String(point.activityId)])),
    ...[criticalPower, criticalPowerYear].flatMap(estimate =>
      estimate ? estimate.anchors.map(anchor => String(anchor.activityId)) : [],
    ),
  ])
  const projectedActivities = [
    ...activities,
    ...allActivities.filter(
      ({ a }) => !detailIds.has(String(a.id)) && powerCurveActivityIds.has(String(a.id)),
    ),
  ]
  const ftp = inputFtp ?? cache.zones?.ftp ?? (best20 > 0 ? Math.round(best20 * 0.95) : null)
  const hrBounds = inputHrBounds?.length
    ? inputHrBounds
    : cache.zones?.hr?.length
      ? cache.zones.hr
      : deriveHrBounds(hrmax)
  const powerBounds =
    inputFtp != null
      ? derivePowerBounds(inputFtp)
      : cache.zones?.power?.length
        ? cache.zones.power
        : ftp != null
          ? derivePowerBounds(ftp)
          : []
  const wahooStamina = estimateWahooCyclingStamina(wahoo ?? null, ftp, inputMaxHeartRate ?? hrmax)

  const starts: [number, number][] = []
  for (const { a } of activities) {
    if (a.sportType.startsWith('Virtual')) continue
    const ll = selectedStreams.get(String(a.id))?.latlng
    if (ll && ll.length >= 2) starts.push([ll[0][0], ll[0][1]])
  }
  const home = inferRouteHome(starts)

  const scope: ActivityProjectionScope = {
    cache,
    originalCache,
    oura,
    garmin,
    weather,
    wahoo,
    trackingById,
    resolveGarmin,
    garminMatches,
    garminTrainingEffectMatches,
    garminHeartRateMatches,
    wahooMatches,
    selectedStreams,
    heartRates,
    respirationProjection,
    wahooStamina,
    generatedAt,
    hrBounds,
    powerBounds,
    ftp,
    inputMaxHeartRate,
    inputLactateThresholdHeartRate,
    hrmax,
    criticalPower,
    criticalPowerYear,
    home,
    powerCurves,
    activityCriticalPowers,
    criticalPowerAnchorsByActivity,
    torqueSamplesByActivity,
    torqueHistory,
  }
  const details: Record<string, StravaActivityDetail> = {}
  for (const { a, sport } of projectedActivities)
    details[String(a.id)] = projectActivityDetail(scope, a, sport)

  const health: Record<string, ActivityHealth> = {}
  if (oura) for (const [date, o] of Object.entries(oura.days)) health[date] = toHealth(o)
  if (weather)
    for (const [date, w] of Object.entries(weather.days)) {
      const h = health[date] ?? emptyHealth()
      health[date] = {
        ...h,
        windKph: h.windKph ?? w.windKph,
        windDir: h.windDir ?? w.windDir,
        windDirDeg: h.windDirDeg ?? w.windDirDeg,
        windGustKph: h.windGustKph ?? w.windGustKph,
      }
    }

  return {
    generatedAt,
    athleteId: cache.athleteId,
    totalKm: round(
      finalized.reduce((s, t) => s + t.distanceKm, 0),
      1,
    ),
    totalTimeS: activities.reduce((s, { a }) => s + a.movingTime, 0),
    totalCount: activities.length,
    totals: finalized,
    strengthTotal,
    days,
    details,
    swimTrend: [],
    health,
    zones: { hr: hrBounds, power: powerBounds, ftp },
    powerCurveRef,
    powerCurveYearRef,
    runPowerCurveRef,
    runPowerCurveYearRef,
    powerCurveYear,
    criticalPower,
    criticalPowerYear,
  }
}
