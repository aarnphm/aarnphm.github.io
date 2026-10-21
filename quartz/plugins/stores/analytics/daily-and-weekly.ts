import type { AnalyticsInputs } from './schema'
import { matchAppleRun } from '../../../util/apple-run-match'
import { matchAppleSwims } from '../../../util/apple-swim-match'
import { swimPaceSeconds, swimStrokeRate } from '../../../util/swim-metrics'
import { weeklyTargetRanges, type WeeklyTargetRange } from '../../../util/weekly-target-range'
import { AppleCache } from '../apple'
import { OuraCache } from '../oura'
import { calculateExerciseLoad, round, type StravaActivityDetail } from '../strava'
import {
  DAY_MS,
  dayMs,
  HRV_BASE_SPAN,
  K42,
  K7,
  LOAD_SHARE_TARGET,
  mean,
  median,
  sd,
  type Act,
  type DailyPoint,
  type EffortBucket,
  type RunningDynamicsMetrics,
  type SupplementalLoadActivity,
  type SwimActivityMetrics,
  type WeeklyPoint,
} from './primitives'

export const swimMetricsByActivityId = (
  acts: readonly Act[],
  apple: AppleCache | null | undefined,
  details: AnalyticsInputs['activityDetails'],
): Map<number, SwimActivityMetrics> => {
  const swimActs = acts.filter(act => act.sport === 'swim')
  const matches = matchAppleSwims(
    Object.values(apple?.swims ?? {}),
    swimActs.map(act => ({
      id: act.a.id,
      date: act.day,
      start: act.a.startDate,
      distanceM: act.a.distance,
    })),
  )

  const out = new Map<number, SwimActivityMetrics>()
  for (const act of swimActs) {
    const detail = details?.[String(act.a.id)]
    const swim = matches.get(act.a.id)
    const applePace = swim ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0) : null
    out.set(act.a.id, {
      paceSPer100m:
        detail?.swimPaceSPer100m ?? applePace ?? swimPaceSeconds(act.a.distance, act.a.movingTime),
      strokeRateSpm:
        detail?.strokeRateSpm ??
        (swim ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0) : null),
      strokes:
        detail?.strokes ?? (swim && Object.keys(swim.strokes).length > 0 ? swim.strokes : null),
    })
  }
  return out
}

const runningDynamicsMean = (
  samples: readonly { value: number }[] | undefined,
  minimum: number,
  maximum: number,
): number | null => {
  const values = (samples ?? [])
    .map(sample => sample.value)
    .filter(value => Number.isFinite(value) && value >= minimum && value <= maximum)
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null
}

export const runningDynamicsByActivityId = (
  acts: readonly Act[],
  apple: AppleCache | null | undefined,
): Map<number, RunningDynamicsMetrics> => {
  const workouts = Object.values(apple?.workouts ?? {})
  const out = new Map<number, RunningDynamicsMetrics>()
  for (const act of acts) {
    if (act.sport !== 'run') continue
    const workout = matchAppleRun({ start: act.a.startDate, distanceM: act.a.distance }, workouts)
    if (!workout) continue
    const strideSampleCount = workout.strideLengthM?.length ?? 0
    const verticalOscillationSampleCount = workout.verticalOscillationCm?.length ?? 0
    out.set(act.a.id, {
      strideLengthM:
        strideSampleCount >= 2 ? runningDynamicsMean(workout.strideLengthM, 0.2, 3) : null,
      strideSampleCount,
      verticalOscillationCm:
        verticalOscillationSampleCount >= 2
          ? runningDynamicsMean(workout.verticalOscillationCm, 1, 30)
          : null,
    })
  }
  return out
}

// A metered ride takes power TSS (NP / FTP). Bike speed grows with about the cube root of power,
// so a speed ratio overstates intensity: speed IF ran about 2x power TSS on the same rides.
export function activityLoad(
  act: Act,
  vThr: number,
  ftp: number | null,
  detail: StravaActivityDetail | undefined,
): number {
  const metered = detail?.deviceWatts === true || (detail == null && act.a.deviceWatts === true)
  const np = detail?.npWatts ?? act.a.weightedAverageWatts ?? null
  const intensity =
    act.sport === 'bike' && metered && np != null && np > 0 && ftp != null && ftp > 0
      ? np / ftp
      : act.vGap / vThr
  return calculateExerciseLoad(intensity, act.a.movingTime) ?? 0
}

export function buildDaily(
  acts: Act[],
  supplementalActivities: readonly SupplementalLoadActivity[],
  loadById: Map<number, number>,
  effortByDay: ReadonlyMap<string, EffortBucket>,
  windowFrom: number,
  windowTo: number,
): DailyPoint[] {
  type DayBucket = { total: number; swim: number; bike: number; run: number }
  const emptyDay = (): DayBucket => ({ total: 0, swim: 0, bike: 0, run: 0 })
  const byDay = new Map<string, DayBucket>()
  for (const act of acts) {
    const load = loadById.get(act.a.id) ?? 0
    const bucket = byDay.get(act.day) ?? emptyDay()
    bucket.total += load
    bucket[act.sport] += load
    byDay.set(act.day, bucket)
  }
  for (const activity of supplementalActivities) {
    const bucket = byDay.get(activity.day) ?? emptyDay()
    bucket.total += activity.load
    byDay.set(activity.day, bucket)
  }

  const rows: ({ date: string } & DayBucket)[] = []
  for (let ms = windowFrom; ms <= windowTo; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const bucket = byDay.get(date) ?? emptyDay()
    rows.push({ date, ...bucket })
  }

  const active = rows.filter(r => r.total > 0)
  const seedRows = active.slice(0, 14)
  const seed = seedRows.length ? mean(seedRows.map(r => r.total)) : 0
  const warmupCut = windowFrom + 42 * DAY_MS

  let ctl = seed
  let atl = seed
  let swimCtl = seed * LOAD_SHARE_TARGET.swim
  let bikeCtl = seed * LOAD_SHARE_TARGET.bike
  let runCtl = seed * LOAD_SHARE_TARGET.run

  const out: DailyPoint[] = []
  for (const row of rows) {
    const tsb = ctl - atl
    out.push({
      date: row.date,
      load: round(row.total, 1),
      effort: round(effortByDay.get(row.date)?.effort ?? 0, 0),
      swimLoad: round(row.swim, 1),
      bikeLoad: round(row.bike, 1),
      runLoad: round(row.run, 1),
      ctl: round(ctl, 1),
      atl: round(atl, 1),
      tsb: round(tsb, 1),
      swimCtl: round(swimCtl, 1),
      bikeCtl: round(bikeCtl, 1),
      runCtl: round(runCtl, 1),
      readiness: null,
      hrv: null,
      rhr: null,
      sleepScore: null,
      sleepDurationS: null,
      sleepMetrics: null,
      tempDevC: null,
      weightKg: null,
      totalCalories: null,
      intakeKcal: null,
      warmup: dayMs(row.date) < warmupCut,
    })
    ctl += (row.total - ctl) * K42
    atl += (row.total - atl) * K7
    swimCtl += (row.swim - swimCtl) * K42
    bikeCtl += (row.bike - bikeCtl) * K42
    runCtl += (row.run - runCtl) * K42
  }
  return out
}

type WeeklyBucket = {
  sessions: number
  load: number
  km: number
  seconds: number
  effort: number
  effortSessions: number
  swimKm: number
  bikeKm: number
  runKm: number
  swimSeconds: number
  bikeSeconds: number
  runSeconds: number
}

const emptyWeeklyBucket = (): WeeklyBucket => ({
  sessions: 0,
  load: 0,
  km: 0,
  seconds: 0,
  effort: 0,
  effortSessions: 0,
  swimKm: 0,
  bikeKm: 0,
  runKm: 0,
  swimSeconds: 0,
  bikeSeconds: 0,
  runSeconds: 0,
})

function weekStartOf(day: string): string {
  const ms = dayMs(day)
  const offset = (new Date(ms).getUTCDay() + 6) % 7
  return new Date(ms - offset * DAY_MS).toISOString().slice(0, 10)
}

export function buildWeekly(
  acts: Act[],
  supplementalActivities: readonly SupplementalLoadActivity[],
  loadById: Map<number, number>,
  effortByDay: ReadonlyMap<string, EffortBucket>,
  windowFrom: number,
  windowTo: number,
): WeeklyPoint[] {
  const byWeek = new Map<string, WeeklyBucket>()
  for (const act of acts) {
    const weekStart = weekStartOf(act.day)
    const bucket = byWeek.get(weekStart) ?? emptyWeeklyBucket()
    bucket.sessions += 1
    bucket.load += loadById.get(act.a.id) ?? 0
    bucket.km += act.distanceKm
    bucket.seconds += act.a.movingTime
    if (act.sport === 'swim') {
      bucket.swimKm += act.distanceKm
      bucket.swimSeconds += act.a.movingTime
    } else if (act.sport === 'bike') {
      bucket.bikeKm += act.distanceKm
      bucket.bikeSeconds += act.a.movingTime
    } else {
      bucket.runKm += act.distanceKm
      bucket.runSeconds += act.a.movingTime
    }
    byWeek.set(weekStart, bucket)
  }
  for (const activity of supplementalActivities) {
    const weekStart = weekStartOf(activity.day)
    const bucket = byWeek.get(weekStart) ?? emptyWeeklyBucket()
    bucket.sessions += 1
    bucket.load += activity.load
    bucket.seconds += activity.a.movingTime
    byWeek.set(weekStart, bucket)
  }
  for (const [day, effort] of effortByDay) {
    const weekStart = weekStartOf(day)
    const bucket = byWeek.get(weekStart) ?? emptyWeeklyBucket()
    bucket.effort += effort.effort
    bucket.effortSessions += effort.sessions
    byWeek.set(weekStart, bucket)
  }
  const firstWeek = weekStartOf(new Date(windowFrom).toISOString().slice(0, 10))
  const lastWeek = weekStartOf(new Date(windowTo).toISOString().slice(0, 10))
  const weeks: string[] = []
  for (let ms = dayMs(firstWeek); ms <= dayMs(lastWeek); ms += 7 * DAY_MS)
    weeks.push(new Date(ms).toISOString().slice(0, 10))
  const firstDay = new Date(windowFrom).toISOString().slice(0, 10)
  const lastDay = new Date(windowTo).toISOString().slice(0, 10)
  const points: WeeklyPoint[] = weeks.map((weekStart, i) => {
    if (!byWeek.has(weekStart)) byWeek.set(weekStart, emptyWeeklyBucket())
    const cur = byWeek.get(weekStart)!
    const prev = i > 0 ? byWeek.get(weeks[i - 1])! : null
    const ramp = prev && prev.load > 0 ? round((cur.load - prev.load) / prev.load, 2) : null
    const sessionLoads: number[] = []
    for (const act of acts) {
      const ws = weekStartOf(act.day)
      if (ws === weekStart) sessionLoads.push(loadById.get(act.a.id) ?? 0)
    }
    for (const activity of supplementalActivities)
      if (weekStartOf(activity.day) === weekStart) sessionLoads.push(activity.load)
    const m = mean(sessionLoads)
    const s = sd(sessionLoads)
    const monotony = s > 0 ? round(m / s, 2) : null
    const strain = monotony != null ? round(cur.load * monotony, 0) : null
    const weekEnd = new Date(dayMs(weekStart) + 6 * DAY_MS).toISOString().slice(0, 10)
    return {
      weekStart,
      complete: weekStart >= firstDay && weekEnd <= lastDay,
      sessions: cur.sessions,
      load: round(cur.load, 1),
      km: round(cur.km, 1),
      hours: round(cur.seconds / 3600, 1),
      swimKm: round(cur.swimKm, 1),
      bikeKm: round(cur.bikeKm, 1),
      runKm: round(cur.runKm, 1),
      swimHours: round(cur.swimSeconds / 3600, 1),
      bikeHours: round(cur.bikeSeconds / 3600, 1),
      runHours: round(cur.runSeconds / 3600, 1),
      effort: round(cur.effort, 0),
      effortSessions: cur.effortSessions,
      loadRange: null,
      effortRange: null,
      ramp,
      monotony,
      strain,
    }
  })
  const loadRanges = weeklyTargetRanges(
    points.map(point => ({ value: point.load, complete: point.complete, observed: true })),
  )
  const effortRanges = weeklyTargetRanges(
    points.map(point => ({
      value: point.effort,
      complete: point.complete,
      observed: point.effortSessions > 0,
    })),
  )
  const roundedRange = (range: WeeklyTargetRange | null): WeeklyTargetRange | null =>
    range ? [round(range[0], 1), round(range[1], 1)] : null
  return points.map((point, index) => ({
    ...point,
    loadRange: roundedRange(loadRanges[index]),
    effortRange: roundedRange(effortRanges[index]),
  }))
}

export const trainingLoadRestingHeartRate = (
  oura: OuraCache | null | undefined,
  date: string,
): number | null => {
  const valid = (value: number | null | undefined): value is number =>
    value != null && Number.isFinite(value) && value >= 25 && value <= 120
  const sameDay = oura?.days[date]?.rhr
  if (valid(sameDay)) return sameDay
  const end = dayMs(date)
  const start = end - (HRV_BASE_SPAN - 1) * DAY_MS
  const values = Object.values(oura?.days ?? {}).flatMap(day => {
    const timestamp = dayMs(day.date)
    return timestamp >= start && timestamp <= end && valid(day.rhr) ? [day.rhr] : []
  })
  return values.length > 0 ? median(values) : null
}
