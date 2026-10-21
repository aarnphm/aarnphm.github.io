import type { Analytics, AnalyticsInputs, RecentEstimates } from './analytics/schema'
import type { GarminVo2Point } from './analytics/vo2-and-ftp'
import { buildBestEffortsBlock, type BestEffortNative } from '../../util/best-efforts'
import { estimateFtpFromPowerCurve } from '../../util/cycling-ftp'
import { localIsoDay, shiftIsoDay } from '../../util/local-date'
import { latestProviderSync } from '../../util/provider-sync'
import { buildSwimPowerCurveBlock } from '../../util/swim-power'
import { buildActivitySummaries } from './analytics/activity-summaries'
import { parseDexa, parseVo2Lab } from './analytics/body-and-labs'
import { buildBody } from './analytics/body-phase'
import {
  buildCalibration,
  calibrationProjection,
  resolveLactateThresholdHeartRate,
} from './analytics/calibration-and-lactate'
import {
  activityLoad,
  buildDaily,
  buildWeekly,
  runningDynamicsByActivityId,
  swimMetricsByActivityId,
  trainingLoadRestingHeartRate,
} from './analytics/daily-and-weekly'
import { buildDistributions } from './analytics/distributions'
import { buildEngine, emptyAnalytics } from './analytics/engine'
import { buildHeat } from './analytics/heat'
import {
  buildPowerToWeightTrend,
  emptyPowerCurve,
  type PowerCurveBlock,
} from './analytics/power-to-weight'
import {
  ageOn,
  ANALYTICS_LOAD_NOTE,
  ATHLETE,
  DAY_MS,
  dayMs,
  gradeAdjSpeed,
  IF_CAP,
  K42,
  K7,
  type Act,
  type EffortBucket,
  type SportBest,
  type ThresholdEstimate,
} from './analytics/primitives'
import {
  buildActions,
  buildBest,
  buildReadiness,
  buildRisk,
  RACE_DISTANCES,
} from './analytics/readiness'
import { buildRecovery } from './analytics/recovery'
import { estimateThreshold } from './analytics/thresholds'
import { buildTrend, RECENT_RANGE_DAYS, trendSource, type SportTrend } from './analytics/trends'
import { matchGarminHeartRateActivity } from './garmin'
import { buildPowerRank } from './power-rank'
import {
  calculateHeartRateTss,
  isTreatment,
  normalizeKind,
  normalizeSport,
  resolveActivityHeartRate,
  round,
  SPORT_ORDER,
  type ActivityHeartRate,
  type RawStravaActivity,
  type Sport,
  type StravaRawCache,
} from './strava'
export { ATHLETE } from './analytics/primitives'
export type { Conf, DailyPoint, RecoveryStatus } from './analytics/primitives'
export { POWER_TO_WEIGHT_DURATIONS } from './analytics/power-to-weight'
export type {
  PowerCurveBlock,
  PowerToWeightDurationS,
  PowerToWeightEffort,
  PowerToWeightReference,
  PowerToWeightTrendPoint,
} from './analytics/power-to-weight'
export { hrZoneUppers, parseVo2Lab } from './analytics/body-and-labs'
export type {
  BodyBlock,
  BodyCompositionDay,
  DexaRecord,
  Vo2LabProfile,
  Vo2LabProfileSample,
  Vo2LabProfileStats,
  Vo2LabRecord,
  Vo2LabTargetStep,
} from './analytics/body-and-labs'
export type { ActivityDistributionPoint, CoreTemperatureOrigin } from './analytics/distributions'
export type { LactateThresholdHeartRate } from './analytics/thresholds'
export type { SportTrend, SportTrendForecastPoint } from './analytics/trends'
export type { RaceLeg } from './analytics/readiness'
export { buildFtpPedalingEvidence, computeFtpHypothesisFromVo2 } from './analytics/vo2-and-ftp'
export type { CardioKey, Vo2Method } from './analytics/vo2-and-ftp'
export type {
  ActivitySummary,
  Analytics,
  AnalyticsInputs,
  FeedActivityRow,
  FeedDayRow,
  FeedWeekRow,
  HeatDay,
  RadarAxis,
} from './analytics/schema'
export { buildDataFeed } from './analytics/data-feed'

export function buildAnalytics(
  cache: StravaRawCache | null,
  inputs: AnalyticsInputs = {},
): Analytics {
  const generatedAt = latestProviderSync(
    inputs.generatedAt,
    cache,
    inputs.oura,
    inputs.apple,
    inputs.core,
    inputs.garmin,
    inputs.weather,
  )
  const todayFromSync = generatedAt ? localIsoDay(generatedAt, inputs.timeZone) : null
  const dexaTests = parseDexa(inputs.dexa)
  const vo2Tests = parseVo2Lab(inputs.vo2labs)
  const latestDexa = dexaTests.length ? dexaTests[dexaTests.length - 1] : null
  const latestVo2Lab = vo2Tests.length ? vo2Tests[vo2Tests.length - 1] : null

  if (!cache) return emptyAnalytics(0, todayFromSync ?? '1970-01-01', inputs.garmin)

  const sinceDay = inputs.since && /^\d{4}-\d{2}-\d{2}$/.test(inputs.since) ? inputs.since : null
  const sourceActivities = Object.values(cache.activities)
    .filter(activity => !sinceDay || activity.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((left, right) => left.startDateLocal.localeCompare(right.startDateLocal))
  const saunaDetails = Object.values(inputs.activityDetails ?? {})
    .filter(detail => detail.sport === 'sauna' && detail.sauna != null)
    .filter(detail => !sinceDay || detail.date >= sinceDay)
  const saunaActivityIds = new Set(saunaDetails.map(detail => detail.id))
  const activityDays = [
    ...sourceActivities.map(activity => activity.startDateLocal.slice(0, 10)),
    ...saunaDetails.map(detail => detail.date),
  ].sort()

  if (activityDays.length === 0) {
    const fallback = todayFromSync ?? '1970-01-01'
    return emptyAnalytics(cache.athleteId, fallback, inputs.garmin)
  }

  const raw = sourceActivities
    .filter(activity => !saunaActivityIds.has(activity.id))
    .map(a => ({ a, sport: normalizeSport(a.sportType) }))
    .filter(
      (x): x is { a: RawStravaActivity; sport: Sport } =>
        x.sport !== null && !isTreatment(x.a.sportType, x.a.name),
    )

  const lastActDay = activityDays[activityDays.length - 1]
  const today = todayFromSync ?? lastActDay
  const todayMs = dayMs(today)
  const effortByDay = new Map<string, EffortBucket>()
  for (const activity of sourceActivities) {
    if (saunaActivityIds.has(activity.id) || activity.commute === true) continue
    const effort = activity.sufferScore
    if (effort == null || !Number.isFinite(effort)) continue
    const day = activity.startDateLocal.slice(0, 10)
    const bucket = effortByDay.get(day) ?? { effort: 0, sessions: 0 }
    bucket.effort += effort
    bucket.sessions += 1
    effortByDay.set(day, bucket)
  }
  const heartRateById = new Map<number, ActivityHeartRate>()
  for (const { a, sport } of raw) {
    const stream = cache.streams?.[String(a.id)]
    heartRateById.set(
      a.id,
      resolveActivityHeartRate(
        a,
        sport,
        stream,
        matchGarminHeartRateActivity(a, sport, inputs.garmin ?? null),
        inputs.garmin ?? null,
      ),
    )
  }

  const acts: Act[] = raw.map(({ a, sport }) => ({
    a,
    sport,
    day: a.startDateLocal.slice(0, 10),
    distanceKm: round(a.distance / 1000, sport === 'swim' ? 3 : 1),
    vGap: gradeAdjSpeed(a, sport, cache.streams?.[String(a.id)]),
  }))
  const trainingActs = acts.filter(act => act.a.commute !== true)
  const swimMetrics = swimMetricsByActivityId(acts, inputs.apple, inputs.activityDetails)
  const runningDynamics = runningDynamicsByActivityId(acts, inputs.apple)
  const distributions = buildDistributions(
    trainingActs,
    inputs.zones,
    inputs.activityDetails,
    cache.streams,
    swimMetrics,
    inputs.core,
    inputs.garmin,
  )

  const runAnchor = {
    streams: cache.streams,
    heartRateBpm: resolveLactateThresholdHeartRate(inputs.garmin, today)?.value ?? null,
  }
  const thresholdList = SPORT_ORDER.map(sport =>
    estimateThreshold(trainingActs, sport, todayMs, runAnchor),
  )
  const thresholds = new Map<Sport, ThresholdEstimate>(thresholdList.map(t => [t.sport, t]))

  const loadById = new Map<number, number>()
  const paceIntensityFactorById = new Map<number, number>()
  for (const act of acts) {
    if (act.a.commute === true) {
      loadById.set(act.a.id, 0)
      continue
    }
    const vThr = thresholds.get(act.sport)!.vThr
    loadById.set(
      act.a.id,
      activityLoad(act, vThr, inputs.ftp ?? null, inputs.activityDetails?.[String(act.a.id)]),
    )
    if (act.sport !== 'bike') paceIntensityFactorById.set(act.a.id, round(act.vGap / vThr, 3))
  }
  const supplementalLoadActivities = sourceActivities.flatMap(a => {
    if (saunaActivityIds.has(a.id)) return []
    const kind = normalizeKind(a.sportType)
    if ((kind !== 'strength' && kind !== 'yoga') || isTreatment(a.sportType, a.name)) return []
    const day = a.startDateLocal.slice(0, 10)
    const averageHeartRate =
      inputs.activityDetails?.[String(a.id)]?.avgHr ?? a.averageHeartrate ?? null
    const restingHeartRate = trainingLoadRestingHeartRate(inputs.oura, day)
    const maximumHeartRate = ATHLETE.hrMax
    const thresholdHeartRate = ATHLETE.lt
    const load =
      averageHeartRate != null &&
      restingHeartRate != null &&
      thresholdHeartRate != null &&
      maximumHeartRate != null
        ? calculateHeartRateTss(
            averageHeartRate,
            a.movingTime,
            restingHeartRate,
            thresholdHeartRate,
            maximumHeartRate,
            ATHLETE.sex,
          )
        : null
    return [{ a, day, load: load ?? 0 }]
  })
  const supplementalLoadById = new Map(
    supplementalLoadActivities.map(activity => [activity.a.id, activity.load]),
  )

  const firstDay = sinceDay ?? activityDays[0]
  const windowFrom = dayMs(firstDay)
  const windowTo = Math.max(todayMs, dayMs(lastActDay))
  const windowToDay = today > lastActDay ? today : lastActDay

  const daily = buildDaily(
    acts,
    supplementalLoadActivities,
    loadById,
    effortByDay,
    windowFrom,
    windowTo,
  )
  const heat = buildHeat(
    cache,
    sourceActivities.filter(activity => !saunaActivityIds.has(activity.id)),
    inputs.weather,
    inputs.core,
    inputs.garmin,
    saunaDetails,
    windowFrom,
    windowTo,
  )
  const { appleDays, latestWeight, powerToWeightMassByDate, body } = buildBody({
    inputs,
    daily,
    latestDexa,
    today,
  })
  const sourcePowerCurve = inputs.powerCurve ?? emptyPowerCurve(today)
  const powerCurve: PowerCurveBlock = {
    ...sourcePowerCurve,
    estimatedFtp: estimateFtpFromPowerCurve(sourcePowerCurve.sixWeeks),
    estimatedFtpYear: estimateFtpFromPowerCurve(sourcePowerCurve.year),
    ranking: buildPowerRank(sourcePowerCurve.sixWeeks, sourcePowerCurve.year, latestWeight, {
      sex: ATHLETE.sex,
      age: ageOn(today),
    }),
    powerToWeight: buildPowerToWeightTrend(
      inputs.activityDetails,
      daily,
      powerToWeightMassByDate,
      today,
    ),
  }
  const weekly = buildWeekly(
    acts,
    supplementalLoadActivities,
    loadById,
    effortByDay,
    windowFrom,
    windowTo,
  )
  const trendSources = SPORT_ORDER.map(sport =>
    trendSource(trainingActs, thresholds.get(sport)!, sport, cache.streams),
  )
  const trends = trendSources.map(source => buildTrend(source, todayMs))
  const trendMap = new Map<Sport, SportTrend>(trends.map(t => [t.sport, t]))
  const calibration = buildCalibration(acts, thresholds, trendMap, loadById, today, todayMs)
  const bestList = SPORT_ORDER.map(sport => buildBest(trainingActs, sport))
  const bests = new Map<Sport, SportBest>(bestList.map(b => [b.sport, b]))
  const risk = buildRisk(daily, weekly)
  const recovery = buildRecovery(daily, risk)

  const ctlNow = daily.length ? daily[daily.length - 1].ctl : 0
  const races = RACE_DISTANCES.map(distance =>
    buildReadiness(distance, thresholds, bests, ctlNow, trendMap),
  )

  // Thresholds and fitness stay current; the trend and the longest sessions come from the window.
  const recentCutoff = shiftIsoDay(today, -(RECENT_RANGE_DAYS - 1))
  const recentFrom = firstDay > recentCutoff ? firstDay : recentCutoff
  const recentFromMs = dayMs(recentFrom)
  const recentTrends = trendSources.map(source => buildTrend(source, todayMs, recentFromMs))
  const recentTrendMap = new Map<Sport, SportTrend>(recentTrends.map(t => [t.sport, t]))
  const recentActs = trainingActs.filter(act => dayMs(act.day) >= recentFromMs)
  const recentBests = new Map<Sport, SportBest>(
    SPORT_ORDER.map(sport => [sport, buildBest(recentActs, sport)]),
  )
  const recent: RecentEstimates = {
    from: recentFrom,
    trends: recentTrends,
    races: RACE_DISTANCES.map(distance =>
      buildReadiness(distance, thresholds, recentBests, ctlNow, recentTrendMap),
    ),
    projections: calibration.paces.map(pace =>
      calibrationProjection(pace.sport, pace.average, recentTrendMap.get(pace.sport)),
    ),
  }

  const loadCut = todayMs - 42 * DAY_MS
  const loadActs = acts.filter(act => dayMs(act.day) >= loadCut)
  const recentLoad = loadActs.reduce((s, act) => s + (loadById.get(act.a.id) ?? 0), 0)
  const loadShare: Record<Sport, number> = { swim: 0, bike: 0, run: 0 }
  if (recentLoad > 0) {
    for (const act of loadActs) loadShare[act.sport] += (loadById.get(act.a.id) ?? 0) / recentLoad
  }

  const appleVo2: { date: string; v: number }[] = []
  for (const d of Object.values(appleDays))
    if (d.vo2max != null) appleVo2.push({ date: d.date, v: d.vo2max })
  appleVo2.sort((p, q) => p.date.localeCompare(q.date))
  const garminVo2: GarminVo2Point[] = []
  for (const d of Object.values(inputs.garmin?.vo2max ?? {})) {
    const v = d.generic ?? d.cycling
    if (v != null) garminVo2.push({ date: d.date, v, generic: d.generic, cycling: d.cycling })
  }
  garminVo2.sort((p, q) => p.date.localeCompare(q.date))
  const engine = buildEngine(
    cache,
    trainingActs,
    daily,
    body,
    thresholds,
    today,
    garminVo2,
    appleVo2,
    heartRateById,
    swimMetrics,
    runningDynamics,
    latestVo2Lab,
    inputs.ftp ?? null,
    inputs.garmin,
    powerCurve,
  )

  const activities = buildActivitySummaries({
    cache,
    inputs,
    acts,
    sinceDay,
    saunaActivityIds,
    saunaDetails,
    supplementalLoadById,
    loadById,
    paceIntensityFactorById,
    swimMetrics,
  })

  const { weakest, actions } = buildActions(thresholds, bests, daily, loadShare)

  return {
    meta: {
      athleteId: cache.athleteId,
      today,
      windowFrom: firstDay,
      windowTo: windowToDay,
      activityCount: activities.length,
      method: {
        ctlTau: 42,
        atlTau: 7,
        k42: K42,
        k7: K7,
        ifCap: IF_CAP,
        thresholdWindowDays: 0,
        seededFrom: 'mean daily load over first 14 active days',
        note: ANALYTICS_LOAD_NOTE,
      },
    },
    calibration,
    thresholds: thresholdList,
    daily,
    weekly,
    trends,
    bests: bestList,
    risk,
    races,
    loadShare,
    body,
    recovery,
    powerCurve,
    swimPowerCurve: buildSwimPowerCurveBlock(
      Object.values(inputs.activityDetails ?? {}).filter(detail => detail.commute !== true),
      today,
    ),
    bestEfforts: buildBestEffortsBlock(
      Object.values(inputs.activityDetails ?? {}).filter(detail => detail.commute !== true),
      new Map<number, BestEffortNative>(
        Object.values(cache.activities).map(activity => [
          activity.id,
          {
            trainer: activity.trainer === true,
            distanceM: activity.distance,
            elapsedTimeS: activity.elapsedTime,
            elevationGainM: activity.totalElevationGain,
          },
        ]),
      ),
    ),
    heat,
    distributions,
    engine,
    events: inputs.events ?? [],
    activities,
    weakestSport: weakest,
    actions,
    tests: { dexa: dexaTests, vo2max: vo2Tests },
    recent,
  }
}
