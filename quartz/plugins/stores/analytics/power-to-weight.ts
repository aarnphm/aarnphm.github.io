import type { PowerCurveFtpEstimate } from '../../../util/cycling-ftp'
import type { CyclingMechanics } from '../../../util/cycling-torque'
import { AppleCache } from '../apple'
import { fitCriticalPower, type CriticalPowerEstimate } from '../critical-power'
import { emptyPowerRank, type PowerRankBlock, type PowerRankMass } from '../power-rank'
import { round, type PowerCurvePoint, type StravaActivityDetail } from '../strava'
import { ageOn, ATHLETE, DAY_MS, dayMs, type DailyPoint } from './primitives'

export const POWER_TO_WEIGHT_DURATIONS: readonly [5, 60, 180, 300, 360, 720, 1200] = [
  5, 60, 180, 300, 360, 720, 1200,
]
const POWER_TO_WEIGHT_REFERENCE_DURATIONS: readonly [5, 60, 300, 1200] = [5, 60, 300, 1200]

export type PowerToWeightDurationS = (typeof POWER_TO_WEIGHT_DURATIONS)[number]
export type PowerToWeightAgeGroup = '20–29' | '30–39' | '40–49' | '50–59' | '60–69' | '70–79'

export interface PowerToWeightEffort {
  mechanics?: CyclingMechanics | null
  startElapsedS?: number
  endElapsedS?: number
  durationS: PowerToWeightDurationS
  watts: number
  wattsPerKg: number
  massKg: number
  massDate: string
  massSource: 'tracking' | 'garmin' | 'apple'
  activityId: number
  activityDate: string
}

export interface PowerToWeightTrendPoint {
  criticalPower?: CriticalPowerEstimate | null
  date: string
  efforts: Record<PowerToWeightDurationS, PowerToWeightEffort | null>
}

export interface PowerToWeightReference {
  durationS: (typeof POWER_TO_WEIGHT_REFERENCE_DURATIONS)[number]
  p10: number
  average: number
  p90: number
}

export interface PowerToWeightSource {
  url: 'https://www.youtube.com/watch?v=nwT8UtsaHds'
  label: 'GCN × Zwift data'
  population: 'Zwift riders'
  selectionBias: string
}

export interface PowerToWeightTrend {
  windowDays: 42
  sex: 'M' | 'F'
  age: number
  ageGroup: PowerToWeightAgeGroup | null
  references: PowerToWeightReference[]
  source: PowerToWeightSource
  points: PowerToWeightTrendPoint[]
}

interface PowerToWeightAgeBand {
  label: PowerToWeightAgeGroup
  min: number
  max: number
}

type PowerToWeightReferenceValues = Omit<PowerToWeightReference, 'durationS'>

const POWER_TO_WEIGHT_AGE_BANDS: readonly PowerToWeightAgeBand[] = [
  { label: '20–29', min: 20, max: 29 },
  { label: '30–39', min: 30, max: 39 },
  { label: '40–49', min: 40, max: 49 },
  { label: '50–59', min: 50, max: 59 },
  { label: '60–69', min: 60, max: 69 },
  { label: '70–79', min: 70, max: 79 },
]

const POWER_TO_WEIGHT_MALE_REFERENCES: Readonly<
  Record<
    PowerToWeightAgeGroup,
    Readonly<Record<PowerToWeightReference['durationS'], PowerToWeightReferenceValues>>
  >
> = {
  '20–29': {
    5: { p10: 4.16, average: 8.75, p90: 14.06 },
    60: { p10: 2.88, average: 4.75, p90: 6.82 },
    300: { p10: 2.27, average: 3.63, p90: 5.1 },
    1200: { p10: 1.95, average: 3.09, p90: 4.32 },
  },
  '30–39': {
    5: { p10: 3.94, average: 8.21, p90: 13 },
    60: { p10: 2.74, average: 4.52, p90: 6.47 },
    300: { p10: 2.15, average: 3.41, p90: 4.73 },
    1200: { p10: 1.86, average: 2.93, p90: 4.05 },
  },
  '40–49': {
    5: { p10: 3.91, average: 7.96, p90: 12.42 },
    60: { p10: 2.68, average: 4.37, p90: 6.21 },
    300: { p10: 2.1, average: 3.28, p90: 4.51 },
    1200: { p10: 1.83, average: 2.84, p90: 3.89 },
  },
  '50–59': {
    5: { p10: 3.74, average: 7.36, p90: 11.32 },
    60: { p10: 2.54, average: 4.05, p90: 5.7 },
    300: { p10: 1.99, average: 3.05, p90: 4.16 },
    1200: { p10: 1.73, average: 2.67, p90: 3.63 },
  },
  '60–69': {
    5: { p10: 3.44, average: 6.56, p90: 9.99 },
    60: { p10: 2.34, average: 3.67, p90: 5.09 },
    300: { p10: 1.82, average: 2.78, p90: 3.76 },
    1200: { p10: 1.61, average: 2.45, p90: 3.32 },
  },
  '70–79': {
    5: { p10: 2.85, average: 5.52, p90: 8.41 },
    60: { p10: 1.97, average: 3.18, p90: 4.44 },
    300: { p10: 1.54, average: 2.43, p90: 3.34 },
    1200: { p10: 1.36, average: 2.16, p90: 2.97 },
  },
}

const POWER_TO_WEIGHT_SOURCE: PowerToWeightSource = {
  url: 'https://www.youtube.com/watch?v=nwT8UtsaHds',
  label: 'GCN × Zwift data',
  population: 'Zwift riders',
  selectionBias:
    'Zwift riders are a self-selected cycling population, so these percentiles do not represent the general population.',
}

export interface PowerCurveBlock {
  sixWeeks: PowerCurvePoint[]
  year: PowerCurvePoint[]
  yearLabel: number | null
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
  estimatedFtp: PowerCurveFtpEstimate | null
  estimatedFtpYear: PowerCurveFtpEstimate | null
  ftp: number | null
  goalFtp: number | null
  ranking: PowerRankBlock
  powerToWeight: PowerToWeightTrend
}

export type PowerCurveInput = Omit<
  PowerCurveBlock,
  'ranking' | 'powerToWeight' | 'estimatedFtp' | 'estimatedFtpYear'
>

function powerToWeightAgeGroup(age: number): PowerToWeightAgeGroup | null {
  return POWER_TO_WEIGHT_AGE_BANDS.find(band => age >= band.min && age <= band.max)?.label ?? null
}

function powerToWeightReferences(
  sex: 'M' | 'F',
  ageGroup: PowerToWeightAgeGroup | null,
): PowerToWeightReference[] {
  if (sex !== 'M' || ageGroup == null) return []
  const reference = POWER_TO_WEIGHT_MALE_REFERENCES[ageGroup]
  return POWER_TO_WEIGHT_REFERENCE_DURATIONS.map(durationS => ({
    durationS,
    ...reference[durationS],
  }))
}

function emptyPowerToWeight(today: string): PowerToWeightTrend {
  const age = Math.max(0, ageOn(today))
  const ageGroup = powerToWeightAgeGroup(age)
  return {
    windowDays: 42,
    sex: ATHLETE.sex,
    age,
    ageGroup,
    references: powerToWeightReferences(ATHLETE.sex, ageGroup),
    source: POWER_TO_WEIGHT_SOURCE,
    points: [],
  }
}

export const emptyPowerCurve = (today = '1970-01-01'): PowerCurveBlock => ({
  sixWeeks: [],
  year: [],
  yearLabel: null,
  criticalPower: null,
  criticalPowerYear: null,
  estimatedFtp: null,
  estimatedFtpYear: null,
  ftp: null,
  goalFtp: null,
  ranking: emptyPowerRank(),
  powerToWeight: emptyPowerToWeight(today),
})

const emptyPowerToWeightEfforts = (): Record<
  PowerToWeightDurationS,
  PowerToWeightEffort | null
> => ({ 5: null, 60: null, 180: null, 300: null, 360: null, 720: null, 1200: null })

const finitePositive = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value) && value > 0

export function historicalPowerToWeightMass(
  daily: readonly DailyPoint[],
  primaryByDate: ReadonlyMap<string, PowerRankMass>,
  appleDays: Readonly<AppleCache['days']>,
): Map<string, PowerRankMass> {
  const byDate = new Map<string, PowerRankMass>()
  const primaryMeasurements = [...primaryByDate.values()].sort((left, right) =>
    left.date.localeCompare(right.date),
  )
  const appleMeasurements = Object.values(appleDays)
    .flatMap((day): PowerRankMass[] =>
      finitePositive(day.weightKg) ? [{ kg: day.weightKg, date: day.date, source: 'apple' }] : [],
    )
    .sort((left, right) => left.date.localeCompare(right.date))
  let primary: PowerRankMass | null = null
  let apple: PowerRankMass | null = null
  let primaryIndex = 0
  let appleIndex = 0
  for (const day of daily) {
    while (
      primaryIndex < primaryMeasurements.length &&
      primaryMeasurements[primaryIndex].date <= day.date
    ) {
      primary = primaryMeasurements[primaryIndex]
      primaryIndex++
    }
    while (
      appleIndex < appleMeasurements.length &&
      appleMeasurements[appleIndex].date <= day.date
    ) {
      apple = appleMeasurements[appleIndex]
      appleIndex++
    }
    const measurement = primary ?? apple
    if (measurement != null) byDate.set(day.date, measurement)
  }
  return byDate
}

function exactPowerToWeightWatts(
  detail: StravaActivityDetail,
  durationS: PowerToWeightDurationS,
): number | null {
  const bestEffort = detail.bestEfforts?.power.find(
    effort => effort.durationS === durationS && finitePositive(effort.averageWatts),
  )
  if (bestEffort != null) return bestEffort.averageWatts
  return (
    detail.powerCurve?.find(point => point.s === durationS && finitePositive(point.w))?.w ?? null
  )
}

function powerToWeightMass(
  detail: StravaActivityDetail,
  historicalMass: ReadonlyMap<string, PowerRankMass>,
): PowerRankMass | null {
  const serializedKg = detail.bestEfforts?.weightKg
  const serializedDate = detail.bestEfforts?.weightDate
  if (finitePositive(serializedKg) && serializedDate === detail.date)
    return { kg: serializedKg, date: serializedDate, source: 'garmin' }
  return historicalMass.get(detail.date) ?? null
}

function betterPowerToWeightEffort(
  candidate: PowerToWeightEffort,
  current: PowerToWeightEffort | null,
): boolean {
  if (current == null) return true
  if (candidate.wattsPerKg !== current.wattsPerKg) return candidate.wattsPerKg > current.wattsPerKg
  if (candidate.watts !== current.watts) return candidate.watts > current.watts
  return candidate.activityId < current.activityId
}

export function buildPowerToWeightTrend(
  activityDetails: Readonly<Record<string, StravaActivityDetail>> | undefined,
  daily: readonly DailyPoint[],
  historicalMass: ReadonlyMap<string, PowerRankMass>,
  today: string,
): PowerToWeightTrend {
  const trend = emptyPowerToWeight(today)
  const candidates: PowerToWeightEffort[] = []
  const details = Object.values(activityDetails ?? {}).sort(
    (left, right) => left.date.localeCompare(right.date) || left.id - right.id,
  )
  for (const detail of details) {
    if (detail.commute === true || detail.sport !== 'bike' || detail.deviceWatts !== true) continue
    const mass = powerToWeightMass(detail, historicalMass)
    if (mass == null || !Number.isFinite(dayMs(detail.date))) continue
    for (const durationS of POWER_TO_WEIGHT_DURATIONS) {
      const watts = exactPowerToWeightWatts(detail, durationS)
      if (watts == null) continue
      const effort = detail.bestEfforts?.power.find(effort => effort.durationS === durationS)
      candidates.push({
        durationS,
        watts,
        wattsPerKg: round(watts / mass.kg, 3),
        massKg: mass.kg,
        massDate: mass.date,
        massSource: mass.source,
        activityId: detail.id,
        activityDate: detail.date,
        mechanics: effort?.mechanics ?? null,
        startElapsedS: effort?.startElapsedS,
        endElapsedS: effort?.endElapsedS,
      })
    }
  }
  if (candidates.length === 0) return trend

  const firstDate = candidates.reduce(
    (earliest, candidate) =>
      candidate.activityDate < earliest ? candidate.activityDate : earliest,
    candidates[0].activityDate,
  )
  const points = daily.flatMap(day => {
    if (day.date < firstDate) return []
    const dayTimestamp = dayMs(day.date)
    const cutoffTimestamp = dayTimestamp - 41 * DAY_MS
    const efforts = emptyPowerToWeightEfforts()
    for (const candidate of candidates) {
      const candidateTimestamp = dayMs(candidate.activityDate)
      if (candidateTimestamp < cutoffTimestamp || candidateTimestamp > dayTimestamp) continue
      if (betterPowerToWeightEffort(candidate, efforts[candidate.durationS]))
        efforts[candidate.durationS] = candidate
    }
    const anchors = details.flatMap(detail => {
      const timestamp = dayMs(detail.date)
      return detail.sport === 'bike' &&
        detail.deviceWatts &&
        timestamp >= cutoffTimestamp &&
        timestamp <= dayTimestamp
        ? (detail.criticalPowerAnchors ?? detail.activityCriticalPower?.anchors ?? [])
        : []
    })
    const criticalPower = fitCriticalPower(
      anchors,
      'six-weeks',
      new Date(cutoffTimestamp).toISOString().slice(0, 10),
      day.date,
    )
    return [{ date: day.date, efforts, criticalPower }]
  })
  return { ...trend, points }
}
