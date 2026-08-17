import type {
  Analytics,
  BodyCompositionDay,
  Conf,
  CoreTemperatureOrigin,
  HeatDay,
  RecoveryStatus,
  Vo2Method,
} from '../plugins/stores/analytics'
import type { OuraDayDetail, OuraSeries } from '../plugins/stores/oura'
import type { StravaActivityDetail } from '../plugins/stores/strava'
import { isRecord } from './type-guards'

export interface TriathlonDayRecovery {
  status: RecoveryStatus | null
  baselineDays: number | null
  readiness: number | null
  readinessBaseline: number | null
  hrv: number | null
  hrvBaseline: number | null
  hrvZ: number | null
  rhr: number | null
  rhrBaseline: number | null
  rhrZ: number | null
  temperatureDeviationC: number | null
  sleepDurationS: number | null
  sleepBaselineS: number | null
  sleepTargetS: number | null
  sleepDebtS: number | null
}

export interface TriathlonDaySleep {
  bedtimeStart: string | null
  bedtimeEnd: string | null
  phase5Min: string | null
  efficiency: number | null
  latencyS: number | null
  timeInBedS: number | null
  totalSleepS: number | null
  deepS: number | null
  lightS: number | null
  remS: number | null
  awakeS: number | null
  averageBreathsPerMinute: number | null
  averageHeartRate: number | null
  averageHrv: number | null
  lowestHeartRate: number | null
  restlessPeriods: number | null
  hrv: OuraSeries | null
  heartRate: OuraSeries | null
  readinessScore: number | null
  readinessContrib: Record<string, number | null> | null
  sleepScore: number | null
  sleepContrib: Record<string, number | null> | null
}

export interface TriathlonDayVo2max {
  value: number
  method: Vo2Method
  confidence: Conf
  asOfDate: string
}

export interface TriathlonDayTraining {
  activityCount: number
  load: number | null
  relativeEffort: number | null
  ctl: number | null
  atl: number | null
  tsb: number | null
  garminTss: number | null
  exerciseLoad: number | null
  exerciseLoadSource: 'garmin' | 'calculated' | 'mixed' | null
  vo2max: TriathlonDayVo2max | null
}

export interface TriathlonDayAnalytics {
  date: string
  body: BodyCompositionDay | null
  recovery: TriathlonDayRecovery | null
  sleep: TriathlonDaySleep | null
  training: TriathlonDayTraining | null
  heat: (HeatDay & { coreOrigin: CoreTemperatureOrigin | 'mixed' | null }) | null
}

export type TriathlonDailyAnalytics = Record<string, TriathlonDayAnalytics>

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const nullableFiniteNumber = (value: unknown): boolean => value === null || finiteNumber(value)
const nullableString = (value: unknown): boolean => value === null || typeof value === 'string'
const oneOf = (value: unknown, values: ReadonlySet<unknown>): boolean => values.has(value)
const recoveryStatuses: ReadonlySet<unknown> = new Set(['building', 'low', 'firm'])
const vo2Methods: ReadonlySet<unknown> = new Set([
  'garmin',
  'apple',
  'bike',
  'run',
  'hrratio',
  'lab',
  'none',
])
const confidences: ReadonlySet<unknown> = new Set(['firm', 'low', 'prior', 'stale'])
const exerciseLoadSources: ReadonlySet<unknown> = new Set(['garmin', 'calculated', 'mixed'])
const heatSources: ReadonlySet<unknown> = new Set(['core', 'weatherkit', 'strava', 'mixed'])
const coreOrigins: ReadonlySet<unknown> = new Set(['app', 'fit', 'mixed'])

const bodyIsValid = (value: unknown, date: string): boolean =>
  value === null ||
  (isRecord(value) &&
    value.date === date &&
    nullableFiniteNumber(value.kg) &&
    nullableFiniteNumber(value.bmi) &&
    nullableFiniteNumber(value.ffmi) &&
    nullableFiniteNumber(value.bodyFatPct) &&
    nullableFiniteNumber(value.bodyWaterPct) &&
    nullableFiniteNumber(value.muscleMassKg) &&
    nullableFiniteNumber(value.boneMassKg))

const recoveryIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    (value.status === null || oneOf(value.status, recoveryStatuses)) &&
    nullableFiniteNumber(value.baselineDays) &&
    nullableFiniteNumber(value.readiness) &&
    nullableFiniteNumber(value.readinessBaseline) &&
    nullableFiniteNumber(value.hrv) &&
    nullableFiniteNumber(value.hrvBaseline) &&
    nullableFiniteNumber(value.hrvZ) &&
    nullableFiniteNumber(value.rhr) &&
    nullableFiniteNumber(value.rhrBaseline) &&
    nullableFiniteNumber(value.rhrZ) &&
    nullableFiniteNumber(value.temperatureDeviationC) &&
    nullableFiniteNumber(value.sleepDurationS) &&
    nullableFiniteNumber(value.sleepBaselineS) &&
    nullableFiniteNumber(value.sleepTargetS) &&
    nullableFiniteNumber(value.sleepDebtS))

const ouraSeriesIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    typeof value.startTs === 'string' &&
    finiteNumber(value.intervalS) &&
    value.intervalS > 0 &&
    Array.isArray(value.items) &&
    value.items.every(nullableFiniteNumber))

const contributionIsValid = (value: unknown): boolean =>
  value === null || (isRecord(value) && Object.values(value).every(nullableFiniteNumber))

const sleepIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    nullableString(value.bedtimeStart) &&
    nullableString(value.bedtimeEnd) &&
    nullableString(value.phase5Min) &&
    nullableFiniteNumber(value.efficiency) &&
    nullableFiniteNumber(value.latencyS) &&
    nullableFiniteNumber(value.timeInBedS) &&
    nullableFiniteNumber(value.totalSleepS) &&
    nullableFiniteNumber(value.deepS) &&
    nullableFiniteNumber(value.lightS) &&
    nullableFiniteNumber(value.remS) &&
    nullableFiniteNumber(value.awakeS) &&
    nullableFiniteNumber(value.averageBreathsPerMinute) &&
    nullableFiniteNumber(value.averageHeartRate) &&
    nullableFiniteNumber(value.averageHrv) &&
    nullableFiniteNumber(value.lowestHeartRate) &&
    nullableFiniteNumber(value.restlessPeriods) &&
    ouraSeriesIsValid(value.hrv) &&
    ouraSeriesIsValid(value.heartRate) &&
    nullableFiniteNumber(value.readinessScore) &&
    contributionIsValid(value.readinessContrib) &&
    nullableFiniteNumber(value.sleepScore) &&
    contributionIsValid(value.sleepContrib))

const vo2maxIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    finiteNumber(value.value) &&
    oneOf(value.method, vo2Methods) &&
    oneOf(value.confidence, confidences) &&
    typeof value.asOfDate === 'string')

const trainingIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    finiteNumber(value.activityCount) &&
    nullableFiniteNumber(value.load) &&
    nullableFiniteNumber(value.relativeEffort) &&
    nullableFiniteNumber(value.ctl) &&
    nullableFiniteNumber(value.atl) &&
    nullableFiniteNumber(value.tsb) &&
    nullableFiniteNumber(value.garminTss) &&
    nullableFiniteNumber(value.exerciseLoad) &&
    (value.exerciseLoadSource === null || oneOf(value.exerciseLoadSource, exerciseLoadSources)) &&
    vo2maxIsValid(value.vo2max))

const heatIsValid = (value: unknown, date: string): boolean =>
  value === null ||
  (isRecord(value) &&
    value.date === date &&
    nullableFiniteNumber(value.temperatureC) &&
    nullableFiniteNumber(value.heatStrainIndex) &&
    (value.source === null || oneOf(value.source, heatSources)) &&
    finiteNumber(value.observedMinutes) &&
    finiteNumber(value.hotMinutes) &&
    finiteNumber(value.dose) &&
    finiteNumber(value.acclimatisationPct) &&
    (value.coreOrigin === null || oneOf(value.coreOrigin, coreOrigins)))

export const isTriathlonDailyAnalytics = (value: unknown): value is TriathlonDailyAnalytics =>
  isRecord(value) &&
  Object.entries(value).every(
    ([date, summary]) =>
      /^\d{4}-\d{2}-\d{2}$/.test(date) &&
      isRecord(summary) &&
      summary.date === date &&
      bodyIsValid(summary.body, date) &&
      recoveryIsValid(summary.recovery) &&
      sleepIsValid(summary.sleep) &&
      trainingIsValid(summary.training) &&
      heatIsValid(summary.heat, date),
  )

const addNullable = (values: readonly (number | null | undefined)[]): number | null => {
  const available: number[] = []
  for (const value of values)
    if (typeof value === 'number' && Number.isFinite(value)) available.push(value)
  return available.length > 0 ? available.reduce((sum, value) => sum + value, 0) : null
}

const exerciseLoadForDay = (
  activities: readonly StravaActivityDetail[],
): Pick<TriathlonDayTraining, 'exerciseLoad' | 'exerciseLoadSource'> => {
  let nativeCount = 0
  let calculatedCount = 0
  const values: number[] = []
  for (const activity of activities) {
    const native = activity.garmin?.exerciseLoad
    if (native != null) {
      values.push(native)
      nativeCount++
      continue
    }
    const calculated = activity.calculatedExerciseLoad?.value
    if (calculated != null) {
      values.push(calculated)
      calculatedCount++
    }
  }
  return {
    exerciseLoad: values.length > 0 ? values.reduce((sum, value) => sum + value, 0) : null,
    exerciseLoadSource:
      nativeCount > 0 && calculatedCount > 0
        ? 'mixed'
        : nativeCount > 0
          ? 'garmin'
          : calculatedCount > 0
            ? 'calculated'
            : null,
  }
}

const vo2maxAt = (analytics: Analytics, date: string): TriathlonDayVo2max | null => {
  let point: (typeof analytics.engine.vo2max.trend)[number] | null = null
  for (const candidate of analytics.engine.vo2max.trend)
    if (candidate.weekStart <= date) point = candidate
  if (!point) return null
  return {
    value: point.vo2max,
    method: point.method,
    asOfDate: point.weekStart,
    confidence:
      analytics.engine.vo2max.estimates.find(estimate => estimate.method === point.method)?.conf ??
      'prior',
  }
}

const sleepSummary = (detail: OuraDayDetail | undefined): TriathlonDaySleep | null =>
  detail
    ? {
        bedtimeStart: detail.bedtimeStart,
        bedtimeEnd: detail.bedtimeEnd,
        phase5Min: detail.phase5Min,
        efficiency: detail.efficiency,
        latencyS: detail.latencyS,
        timeInBedS: detail.timeInBedS,
        totalSleepS: detail.totalSleepS,
        deepS: detail.deepS,
        lightS: detail.lightS,
        remS: detail.remS,
        awakeS: detail.awakeS,
        averageBreathsPerMinute: detail.avgBreath,
        averageHeartRate: detail.avgHr,
        averageHrv: detail.avgHrv,
        lowestHeartRate: detail.lowestHr,
        restlessPeriods: detail.restlessPeriods,
        hrv: detail.hrv,
        heartRate: detail.hr,
        readinessScore: detail.readinessScore,
        readinessContrib: detail.readinessContrib,
        sleepScore: detail.sleepScore,
        sleepContrib: detail.sleepContrib,
      }
    : null

export function buildTriathlonDailyAnalytics(
  analytics: Analytics,
  ouraDetails: Readonly<Record<string, OuraDayDetail>> = {},
  activityDetails: Readonly<Record<string, StravaActivityDetail>> = {},
): TriathlonDailyAnalytics {
  const bodyByDate = new Map(analytics.body.composition.map(day => [day.date, day]))
  const weightByDate = new Map<string, number>()
  for (const measurement of analytics.body.series)
    weightByDate.set(measurement.date, measurement.kg)
  const recoveryByDate = new Map(analytics.recovery.series.map(day => [day.date, day]))
  const heatByDate = new Map(analytics.heat.series.map(day => [day.date, day]))
  const heatActivitiesByDate = new Map<string, Analytics['heat']['activities']>()
  for (const activity of analytics.heat.activities) {
    const entries = heatActivitiesByDate.get(activity.date) ?? []
    entries.push(activity)
    heatActivitiesByDate.set(activity.date, entries)
  }
  const activitySummariesByDate = new Map<string, Analytics['activities']>()
  for (const activity of analytics.activities) {
    const entries = activitySummariesByDate.get(activity.date) ?? []
    entries.push(activity)
    activitySummariesByDate.set(activity.date, entries)
  }
  const activityDetailsByDate = new Map<string, StravaActivityDetail[]>()
  for (const activity of Object.values(activityDetails)) {
    const entries = activityDetailsByDate.get(activity.date) ?? []
    entries.push(activity)
    activityDetailsByDate.set(activity.date, entries)
  }
  const dates = new Set([
    ...analytics.daily.map(day => day.date),
    ...bodyByDate.keys(),
    ...weightByDate.keys(),
    ...recoveryByDate.keys(),
    ...heatByDate.keys(),
    ...Object.keys(ouraDetails),
    ...activityDetailsByDate.keys(),
  ])
  const dailyByDate = new Map(analytics.daily.map(day => [day.date, day]))
  const result: TriathlonDailyAnalytics = {}

  for (const date of [...dates].sort()) {
    const daily = dailyByDate.get(date)
    const body = bodyByDate.get(date)
    const weight = weightByDate.get(date)
    const recovery = recoveryByDate.get(date)
    const activitySummaries = activitySummariesByDate.get(date) ?? []
    const activities = activityDetailsByDate.get(date) ?? []
    const effortAvailable = activitySummaries.some(activity => activity.effort != null)
    const exerciseLoad = exerciseLoadForDay(activities)
    const garminTss = addNullable(activities.map(activity => activity.garmin?.trainingStressScore))
    const vo2max = vo2maxAt(analytics, date)
    const heat = heatByDate.get(date)
    const coreOrigins = new Set(
      (heatActivitiesByDate.get(date) ?? []).flatMap(activity =>
        activity.coreOrigin == null ? [] : [activity.coreOrigin],
      ),
    )
    result[date] = {
      date,
      body:
        body ??
        (weight != null
          ? {
              date,
              kg: weight,
              bmi: null,
              ffmi: null,
              bodyFatPct: null,
              bodyWaterPct: null,
              muscleMassKg: null,
              boneMassKg: null,
            }
          : null),
      recovery: recovery
        ? {
            status: recovery.status,
            baselineDays: recovery.baselineDays,
            readiness: recovery.readiness,
            readinessBaseline: recovery.readinessBaseline,
            hrv: recovery.hrv,
            hrvBaseline: recovery.hrvBaseline,
            hrvZ: recovery.hrvZ,
            rhr: recovery.rhr,
            rhrBaseline: recovery.rhrBaseline,
            rhrZ: recovery.rhrZ,
            temperatureDeviationC: recovery.tempDevC,
            sleepDurationS: recovery.sleepS,
            sleepBaselineS: recovery.sleepBaselineS,
            sleepTargetS: recovery.sleepTargetS,
            sleepDebtS: recovery.sleepDebtS,
          }
        : null,
      sleep: sleepSummary(ouraDetails[date]),
      training:
        daily || activitySummaries.length > 0 || activities.length > 0 || vo2max
          ? {
              activityCount: activitySummaries.length,
              load: daily?.load ?? null,
              relativeEffort: effortAvailable ? (daily?.effort ?? null) : null,
              ctl: daily?.ctl ?? null,
              atl: daily?.atl ?? null,
              tsb: daily?.tsb ?? null,
              garminTss,
              ...exerciseLoad,
              vo2max,
            }
          : null,
      heat: heat
        ? {
            ...heat,
            coreOrigin:
              coreOrigins.size > 1
                ? 'mixed'
                : coreOrigins.has('app')
                  ? 'app'
                  : coreOrigins.has('fit')
                    ? 'fit'
                    : null,
          }
        : null,
    }
  }

  return result
}
