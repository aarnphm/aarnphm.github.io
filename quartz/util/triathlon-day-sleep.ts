import type { AppleSleepDay, AppleSleepSession } from '../plugins/stores/apple-sleep'
import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDaily, OuraDayDetail, OuraSeries, OuraSleepDetail } from '../plugins/stores/oura'
import { sleepStagesAreValid } from './sleep-metrics'
import { isRecord } from './type-guards'

export type SleepProvider = 'oura' | 'garmin' | 'apple'
export type OuraDailySleep = Pick<OuraDaily, 'sleepScore' | 'sleepDurationS'>
type SourcedDailySleep = OuraDailySleep & {
  sleepSources?: Partial<Record<keyof OuraDailySleep, SleepProvider>>
}
type GarminNight = NonNullable<GarminSleepSummary['sleep']>
type SleepStages = GarminNight['stages']
export type SleepSeries =
  | OuraSeries
  | { startTs: string; offsetsS: number[]; items: (number | null)[] }

interface SleepFields {
  bedtimeStart: string | null
  bedtimeEnd: string | null
  phase5Min: string | null
  phase30Sec?: string | null
  movement30Sec?: string | null
  lowBatteryAlert?: boolean
  sleepAlgorithmVersion?: string | null
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
  hrv: SleepSeries | null
  heartRate: SleepSeries | null
  stages?: SleepStages
  unmeasurableS?: number | null
  awakeCount?: number | null
  restingHeartRate?: number | null
  skinTemperatureDeviationC?: number | null
  readinessScore: number | null
  readinessContrib: Record<string, number | null> | null
  sleepScore: number | null
  sleepContrib: Record<string, number | null> | null
}

export interface TriathlonDaySleep extends SleepFields {
  nightSource?: SleepProvider | null
  fieldSources?: Partial<Record<keyof SleepFields, SleepProvider>>
}

export const sleepProviderLabel = (source: SleepProvider): string =>
  source === 'apple' ? 'Apple Health' : source === 'garmin' ? 'Garmin' : 'Oura'

export function ouraDailySleep(day: SourcedDailySleep | undefined): OuraDailySleep | undefined {
  if (!day) return undefined
  // Daily payloads created before source tracking only carried Oura sleep.
  return {
    sleepScore:
      day.sleepSources?.sleepScore && day.sleepSources.sleepScore !== 'oura'
        ? null
        : day.sleepScore,
    sleepDurationS:
      day.sleepSources?.sleepDurationS && day.sleepSources.sleepDurationS !== 'oura'
        ? null
        : day.sleepDurationS,
  }
}

export interface TriathlonDayNap extends TriathlonDaySleep {
  id: string
  type: 'sleep' | 'late_nap'
  providerFeedback?: string | null
  reportedDay: string | null
  sleepScoreDelta: number | null
  readinessScoreDelta: number | null
}

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const nullableFiniteNumber = (value: unknown): boolean => value === null || finiteNumber(value)
const nullableString = (value: unknown): boolean => value === null || typeof value === 'string'
const positive = (value: number | null | undefined): number | null =>
  finiteNumber(value) && value > 0 ? value : null

export function sleepTimestampWithOffset(timestamp: number, utcOffsetMinutes = 0): string {
  const absoluteOffset = Math.abs(utcOffsetMinutes)
  const hours = Math.floor(absoluteOffset / 60)
    .toString()
    .padStart(2, '0')
  const minutes = (absoluteOffset % 60).toString().padStart(2, '0')
  const local = new Date(timestamp + utcOffsetMinutes * 60_000).toISOString().slice(0, -1)
  return `${local}${utcOffsetMinutes < 0 ? '-' : '+'}${hours}:${minutes}`
}

export function sleepSeriesFromSamples<T extends { timestamp: number }>(
  samples: readonly T[] | undefined,
  value: (sample: T) => number | null,
  utcOffsetMinutes?: number,
): SleepSeries | null {
  if (!samples?.length) return null
  const start = samples[0].timestamp
  return {
    startTs: sleepTimestampWithOffset(start, utcOffsetMinutes),
    offsetsS: samples.map(sample => (sample.timestamp - start) / 1000),
    items: samples.map(value),
  }
}

const timedSeriesIsValid = (value: unknown, count: number): boolean =>
  Array.isArray(value) &&
  value.length === count &&
  value[0] === 0 &&
  value.every((offset, index) => finiteNumber(offset) && (index === 0 || offset > value[index - 1]))

const sleepSeriesIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    typeof value.startTs === 'string' &&
    Array.isArray(value.items) &&
    value.items.every(nullableFiniteNumber) &&
    (value.offsetsS === undefined
      ? finiteNumber(value.intervalS) && value.intervalS > 0
      : value.intervalS === undefined && timedSeriesIsValid(value.offsetsS, value.items.length)))

const contributionIsValid = (value: unknown): boolean =>
  value === null || (isRecord(value) && Object.values(value).every(nullableFiniteNumber))

export const sleepIsValid = (value: unknown): boolean =>
  value === null ||
  (isRecord(value) &&
    nullableString(value.bedtimeStart) &&
    nullableString(value.bedtimeEnd) &&
    nullableString(value.phase5Min) &&
    (value.phase30Sec === undefined || nullableString(value.phase30Sec)) &&
    (value.movement30Sec === undefined || nullableString(value.movement30Sec)) &&
    (value.lowBatteryAlert === undefined || typeof value.lowBatteryAlert === 'boolean') &&
    (value.sleepAlgorithmVersion === undefined || nullableString(value.sleepAlgorithmVersion)) &&
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
    [
      value.unmeasurableS,
      value.awakeCount,
      value.restingHeartRate,
      value.skinTemperatureDeviationC,
    ].every(item => item === undefined || nullableFiniteNumber(item)) &&
    sleepStagesAreValid(value.stages) &&
    (value.nightSource == null ||
      value.nightSource === 'oura' ||
      value.nightSource === 'garmin' ||
      value.nightSource === 'apple') &&
    (value.fieldSources === undefined ||
      (isRecord(value.fieldSources) &&
        Object.entries(value.fieldSources).every(
          ([key, source]) =>
            key !== 'fieldSources' &&
            key !== 'nightSource' &&
            Object.hasOwn(value, key) &&
            value[key] != null &&
            (source === 'oura' || source === 'garmin' || source === 'apple'),
        ))) &&
    sleepSeriesIsValid(value.hrv) &&
    sleepSeriesIsValid(value.heartRate) &&
    nullableFiniteNumber(value.readinessScore) &&
    contributionIsValid(value.readinessContrib) &&
    nullableFiniteNumber(value.sleepScore) &&
    contributionIsValid(value.sleepContrib))

const hasOuraNight = (detail: OuraSleepDetail | null | undefined): boolean =>
  !!detail &&
  (Date.parse(detail.bedtimeEnd ?? '') > Date.parse(detail.bedtimeStart ?? '') ||
    positive(detail.totalSleepS) != null ||
    positive(detail.timeInBedS) != null ||
    [detail.phase30Sec, detail.phase5Min, detail.movement30Sec].some(
      phase => phase != null && /[1-4]/.test(phase),
    ) ||
    [detail.deepS, detail.lightS, detail.remS, detail.awakeS].some(
      seconds => positive(seconds) != null,
    ))

const measuredSeries = (series: OuraSeries | null | undefined): OuraSeries | null =>
  series?.items.some(item => item != null) ? series : null

export function resolveSleepTotals(
  detail: OuraSleepDetail | null | undefined,
  garmin: GarminSleepSummary | null | undefined,
  daily?: OuraDailySleep | null,
  apple?: AppleSleepDay | null,
): OuraDailySleep & { sleepSources: NonNullable<SourcedDailySleep['sleepSources']> } {
  const ouraDuration = positive(detail?.totalSleepS) ?? positive(daily?.sleepDurationS)
  const garminDuration =
    ouraDuration == null && !hasOuraNight(detail) ? garmin?.sleep?.totalSleepS : null
  const appleDuration =
    ouraDuration == null && garminDuration == null && !hasOuraNight(detail)
      ? apple?.sleep?.totalSleepS
      : null
  const ouraScore = detail?.sleepScore ?? daily?.sleepScore
  const garminScore = garmin?.sleep?.sleepScore
  return {
    sleepDurationS: ouraDuration ?? garminDuration ?? appleDuration ?? null,
    sleepScore: ouraScore ?? garminScore ?? null,
    sleepSources: {
      sleepDurationS:
        ouraDuration != null
          ? 'oura'
          : garminDuration != null
            ? 'garmin'
            : appleDuration != null
              ? 'apple'
              : undefined,
      sleepScore: ouraScore != null ? 'oura' : garminScore != null ? 'garmin' : undefined,
    },
  }
}

export function resolveSleepSummary(
  detail: OuraSleepDetail | null | undefined,
  garmin: GarminSleepSummary | null | undefined,
  daily?: OuraDailySleep | null,
  apple?: AppleSleepDay | null,
): TriathlonDaySleep | null {
  if (!hasOuraNight(detail) && !positive(daily?.sleepDurationS) && !garmin?.sleep && apple?.sleep) {
    return appleSessionSummary(apple.sleep, resolveSleepSummary(detail, garmin, daily))
  }
  const ouraNight = hasOuraNight(detail)
  const garminNight = garmin?.sleep
  const useGarminTiming = !ouraNight && garminNight != null
  const ouraTiming = useGarminTiming ? null : detail
  const ouraHrv = measuredSeries(detail?.hrv)
  const ouraHeartRate = measuredSeries(detail?.hr)
  const totals = resolveSleepTotals(detail, garmin, daily)
  const fieldSources: Partial<Record<keyof SleepFields, SleepProvider>> = {}
  if (totals.sleepSources.sleepDurationS)
    fieldSources.totalSleepS = totals.sleepSources.sleepDurationS
  if (totals.sleepSources.sleepScore) fieldSources.sleepScore = totals.sleepSources.sleepScore
  const select = <K extends keyof SleepFields>(
    key: K,
    oura: NonNullable<SleepFields[K]> | null | undefined,
    fallback?: NonNullable<SleepFields[K]> | null,
  ): NonNullable<SleepFields[K]> | null => {
    if (oura != null) {
      fieldSources[key] = 'oura'
      return oura
    }
    if (fallback != null) {
      fieldSources[key] = 'garmin'
      return fallback
    }
    return null
  }
  const localTime = (iso: string | null | undefined): string | null =>
    iso != null && Number.isFinite(Date.parse(iso))
      ? sleepTimestampWithOffset(Date.parse(iso), garmin?.utcOffsetMinutes)
      : null
  const sleep: SleepFields = {
    bedtimeStart: select(
      'bedtimeStart',
      ouraTiming?.bedtimeStart,
      useGarminTiming ? localTime(garmin?.startTime) : null,
    ),
    bedtimeEnd: select(
      'bedtimeEnd',
      ouraTiming?.bedtimeEnd,
      useGarminTiming ? localTime(garmin?.endTime) : null,
    ),
    phase5Min: select('phase5Min', ouraTiming?.phase5Min),
    phase30Sec: select('phase30Sec', ouraTiming?.phase30Sec),
    movement30Sec: select('movement30Sec', ouraTiming?.movement30Sec),
    lowBatteryAlert: select('lowBatteryAlert', detail?.lowBatteryAlert || undefined) ?? undefined,
    sleepAlgorithmVersion: select('sleepAlgorithmVersion', detail?.sleepAlgorithmVersion),
    efficiency: select('efficiency', detail?.efficiency),
    latencyS: select('latencyS', detail?.latencyS),
    timeInBedS: select('timeInBedS', detail?.timeInBedS),
    totalSleepS: totals.sleepDurationS,
    deepS: select('deepS', ouraTiming?.deepS, useGarminTiming ? garminNight.deepS : null),
    lightS: select('lightS', ouraTiming?.lightS, useGarminTiming ? garminNight.lightS : null),
    remS: select('remS', ouraTiming?.remS, useGarminTiming ? garminNight.remS : null),
    awakeS: select('awakeS', ouraTiming?.awakeS, useGarminTiming ? garminNight.awakeS : null),
    stages: select('stages', undefined, useGarminTiming ? garminNight.stages : null) ?? undefined,
    unmeasurableS: select(
      'unmeasurableS',
      undefined,
      useGarminTiming ? garminNight.unmeasurableS : null,
    ),
    awakeCount: select('awakeCount', undefined, useGarminTiming ? garminNight.awakeCount : null),
    averageBreathsPerMinute: select(
      'averageBreathsPerMinute',
      positive(detail?.avgBreath),
      garminNight || ouraNight ? positive(garmin?.averageBreathsPerMinute) : null,
    ),
    averageHeartRate: select(
      'averageHeartRate',
      positive(detail?.avgHr),
      garminNight?.averageHeartRateBpm,
    ),
    averageHrv: select('averageHrv', positive(detail?.avgHrv), garminNight?.averageHrvMs),
    lowestHeartRate: select('lowestHeartRate', positive(detail?.lowestHr)),
    restingHeartRate: select(
      'restingHeartRate',
      undefined,
      useGarminTiming ? garminNight.restingHeartRateBpm : null,
    ),
    skinTemperatureDeviationC: select(
      'skinTemperatureDeviationC',
      undefined,
      useGarminTiming ? garminNight.skinTempDeviationC : null,
    ),
    restlessPeriods: select('restlessPeriods', detail?.restlessPeriods),
    hrv: select(
      'hrv',
      ouraHrv,
      ouraHrv
        ? null
        : sleepSeriesFromSamples(garminNight?.hrv, sample => sample.ms, garmin?.utcOffsetMinutes),
    ),
    heartRate: select(
      'heartRate',
      ouraHeartRate,
      ouraHeartRate
        ? null
        : sleepSeriesFromSamples(
            garminNight?.heartRate,
            sample => sample.bpm,
            garmin?.utcOffsetMinutes,
          ),
    ),
    readinessScore: select('readinessScore', detail?.readinessScore),
    readinessContrib: select('readinessContrib', detail?.readinessContrib),
    sleepScore: totals.sleepScore,
    sleepContrib: select('sleepContrib', detail?.sleepContrib),
  }
  if (Object.keys(fieldSources).length === 0) return null
  return {
    ...sleep,
    fieldSources,
    nightSource: ouraNight ? 'oura' : useGarminTiming ? 'garmin' : null,
  }
}

export const sleepSummary = (detail: OuraSleepDetail | undefined): TriathlonDaySleep | null =>
  resolveSleepSummary(detail, null)

const providerNaps = (
  detail: OuraDayDetail | null | undefined,
  garmin?: GarminSleepSummary | null,
): TriathlonDayNap[] | undefined => {
  const oura = detail?.naps?.flatMap(nap => {
    const sleep = sleepSummary(nap)
    return sleep
      ? [
          {
            ...sleep,
            id: nap.id,
            type: nap.type,
            reportedDay: nap.reportedDay,
            sleepScoreDelta: nap.sleepScoreDelta,
            readinessScoreDelta: nap.readinessScoreDelta,
          },
        ]
      : []
  })
  if (oura?.length || !garmin?.naps) return oura
  return garmin.naps.flatMap(nap => {
    const bedtimeStart = sleepTimestampWithOffset(
      nap.startTime,
      nap.startUtcOffsetMinutes ?? garmin.utcOffsetMinutes,
    )
    const bedtimeEnd = sleepTimestampWithOffset(
      nap.endTime,
      nap.endUtcOffsetMinutes ?? garmin.utcOffsetMinutes,
    )
    const sleep = resolveSleepSummary(null, null, {
      sleepDurationS: nap.durationS,
      sleepScore: null,
    })
    return sleep
      ? [
          {
            ...sleep,
            bedtimeStart,
            bedtimeEnd,
            fieldSources: { bedtimeStart: 'garmin', bedtimeEnd: 'garmin', totalSleepS: 'garmin' },
            nightSource: 'garmin',
            id: nap.id,
            type: 'sleep',
            providerFeedback: nap.feedback,
            reportedDay: null,
            sleepScoreDelta: null,
            readinessScoreDelta: null,
          },
        ]
      : []
  })
}

function appleSessionSummary(
  session: AppleSleepSession,
  existing?: TriathlonDaySleep | null,
): TriathlonDaySleep {
  const base =
    existing ??
    resolveSleepSummary(null, null, { sleepDurationS: session.totalSleepS, sleepScore: null })!
  const fields = {
    bedtimeStart: session.start,
    bedtimeEnd: session.end,
    totalSleepS: session.totalSleepS,
    deepS: session.deepS,
    lightS: session.lightS,
    remS: session.remS,
    awakeS: session.awakeS,
  }
  const fieldSources: TriathlonDaySleep['fieldSources'] = { ...base.fieldSources }
  for (const key of Object.keys(fields)) {
    if (
      key === 'bedtimeStart' ||
      key === 'bedtimeEnd' ||
      key === 'totalSleepS' ||
      key === 'deepS' ||
      key === 'lightS' ||
      key === 'remS' ||
      key === 'awakeS'
    ) {
      if (fields[key] != null) fieldSources[key] = 'apple'
    }
  }
  if (session.stages?.length) fieldSources.stages = 'apple'
  return {
    ...base,
    ...fields,
    ...(session.stages?.length ? { stages: session.stages } : {}),
    nightSource: 'apple',
    fieldSources,
  }
}

export function napSummaries(
  detail: OuraDayDetail | null | undefined,
  garmin?: GarminSleepSummary | null,
  apple?: AppleSleepDay | null,
): TriathlonDayNap[] | undefined {
  const existing = providerNaps(detail, garmin)
  if (existing?.length || !apple) return existing
  const naps: TriathlonDayNap[] = []
  for (const session of apple.naps) {
    const start = Date.parse(session.start),
      end = Date.parse(session.end)
    const overlaps = (a: string | null | undefined, b: string | null | undefined): boolean =>
      Date.parse(a ?? '') < end && Date.parse(b ?? '') > start
    if (
      overlaps(detail?.bedtimeStart, detail?.bedtimeEnd) ||
      overlaps(garmin?.startTime, garmin?.endTime)
    )
      continue
    naps.push({
      ...appleSessionSummary(session),
      id: session.id,
      type: 'sleep',
      reportedDay: null,
      sleepScoreDelta: null,
      readinessScoreDelta: null,
      providerFeedback: `Inferred daytime sleep session · ${session.sourceName || session.sourceBundleIdentifier}`,
    })
  }
  return naps.sort((a, b) => Date.parse(a.bedtimeStart ?? '') - Date.parse(b.bedtimeStart ?? ''))
}
