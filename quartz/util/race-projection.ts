import type { Analytics, DailyPoint } from '../plugins/stores/analytics'
import type { PaceDayState, PaceLegSpec, PaceSport } from './pace-features'
import type { PaceForecaster } from './pace-forecast'
import type { TriathlonCalendarEvent } from './triathlon-calendar'
import { shiftIsoDay } from './local-date'
import { isPaceSport } from './pace-features'
import { Z80 } from './pace-forecast'
import { isRecord } from './type-guards'

const DAY_MS = 86_400_000
const dayNumber = (date: string): number => Date.parse(`${date}T00:00:00Z`) / DAY_MS
const clamp = (value: number, low: number, high: number): number =>
  Math.min(high, Math.max(low, value))

export interface RaceProjectionLeg extends PaceLegSpec {
  intensity: number
  referenceYear: number | null
  distanceSource: 'advertised' | 'route' | 'format'
}

export interface RaceProjectionSpec {
  id: string
  date: string | null
  kind: TriathlonCalendarEvent['kind']
  legs: RaceProjectionLeg[]
  extraMinutes: number | null
  runPenaltyPct: number
}

const formatLegs = (event: TriathlonCalendarEvent): [PaceSport, number, number][] => {
  if (event.kind === 'hyrox') return [['run', 8, 0.95]]
  const format = event.format.toLowerCase()
  if (event.kind === 'triathlon') {
    if (format === '70.3')
      return [
        ['swim', 1.9, 0.9],
        ['bike', 90, 0.8],
        ['run', 21.0975, 0.86],
      ]
    if (format === '140.6')
      return [
        ['swim', 3.8, 0.85],
        ['bike', 180, 0.7],
        ['run', 42.195, 0.78],
      ]
    if (format === 'olympic')
      return [
        ['swim', 1.5, 0.95],
        ['bike', 40, 0.9],
        ['run', 10, 0.95],
      ]
    if (format === 'sprint')
      return [
        ['swim', 0.75, 0.95],
        ['bike', 20, 0.9],
        ['run', 5, 1],
      ]
    if (event.series === 't100')
      return [
        ['swim', 2, 0.9],
        ['bike', 80, 0.8],
        ['run', 18, 0.86],
      ]
  }
  if (event.kind === 'running') {
    if (format.includes('marathon') && !format.includes('half')) return [['run', 42.195, 0.9]]
    if (format.includes('half')) return [['run', 21.0975, 0.95]]
  }
  const km = format.match(/(\d+(?:\.\d+)?)\s*km/)
  if (km && (event.kind === 'cycling' || event.kind === 'running'))
    return [
      [
        event.kind === 'cycling' ? 'bike' : 'run',
        Number(km[1]),
        event.kind === 'cycling' ? 0.75 : 0.95,
      ],
    ]
  return []
}

export const raceProjectionSpec = (event: TriathlonCalendarEvent): RaceProjectionSpec => {
  const course = event.details?.course ?? []
  const defaults = formatLegs(event)
  const distances = defaults.length
    ? defaults
    : course.map<[PaceSport, number, number]>(leg => [
        leg.leg,
        (leg.raceDistanceM ?? leg.distanceM) / 1000,
        0.85,
      ])
  return {
    id: event.id,
    date: event.date,
    kind: event.kind,
    legs: distances.map(([sport, distanceKm, intensity]) => {
      const route = course.find(leg => leg.leg === sport)
      return {
        sport,
        distanceKm: route?.raceDistanceM ? route.raceDistanceM / 1000 : distanceKm,
        elevationM: route?.elevationGainM ?? 0,
        tempC: null,
        windKph: null,
        intensity,
        referenceYear: route?.edition ?? null,
        distanceSource: route?.raceDistanceM ? 'advertised' : defaults.length ? 'format' : 'route',
      }
    }),
    extraMinutes: event.kind === 'hyrox' ? null : event.kind === 'triathlon' ? 5 : 0,
    runPenaltyPct: event.kind === 'hyrox' ? 10 : event.kind === 'triathlon' ? 5 : 0,
  }
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

export const parseRaceProjectionSpec = (text: string): RaceProjectionSpec | null => {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    return null
  }
  if (
    !isRecord(value) ||
    typeof value.id !== 'string' ||
    !Array.isArray(value.legs) ||
    (value.date !== null &&
      (typeof value.date !== 'string' ||
        !/^\d{4}-\d{2}-\d{2}$/.test(value.date) ||
        !Number.isFinite(dayNumber(value.date)))) ||
    (value.kind !== 'triathlon' &&
      value.kind !== 'running' &&
      value.kind !== 'cycling' &&
      value.kind !== 'hyrox') ||
    (value.extraMinutes !== null && (!finite(value.extraMinutes) || value.extraMinutes < 0)) ||
    !finite(value.runPenaltyPct) ||
    value.runPenaltyPct < 0
  )
    return null
  const legs: RaceProjectionLeg[] = []
  for (const leg of value.legs) {
    if (
      !isRecord(leg) ||
      !isPaceSport(leg.sport) ||
      !finite(leg.distanceKm) ||
      leg.distanceKm <= 0 ||
      !finite(leg.elevationM) ||
      !finite(leg.intensity) ||
      leg.intensity <= 0 ||
      leg.intensity > 1.15 ||
      (leg.referenceYear !== null && !finite(leg.referenceYear)) ||
      (leg.distanceSource !== 'advertised' &&
        leg.distanceSource !== 'route' &&
        leg.distanceSource !== 'format')
    )
      return null
    legs.push({
      sport: leg.sport,
      distanceKm: leg.distanceKm,
      elevationM: leg.elevationM,
      tempC: null,
      windKph: null,
      intensity: leg.intensity,
      referenceYear: leg.referenceYear,
      distanceSource: leg.distanceSource,
    })
  }
  return {
    id: value.id,
    date: value.date,
    kind: value.kind,
    legs,
    extraMinutes: value.extraMinutes,
    runPenaltyPct: value.runPenaltyPct,
  }
}

export interface RaceScenario {
  weeklyLoad: number
  taperDays: number
  paceGainPct: number
}

export interface RaceProjectionBaseline {
  day: PaceDayState
  daily: readonly DailyPoint[]
  weeklyLoad: number
  shares: Record<PaceSport, number>
  k42: number
  k7: number
  longest: Record<PaceSport, number>
  recentSessions: Record<PaceSport, number>
}

export const raceProjectionBaseline = (
  analytics: Analytics,
  day: PaceDayState,
): RaceProjectionBaseline | null => {
  const recent = analytics.daily.filter(
    row => row.date <= day.date && dayNumber(day.date) - dayNumber(row.date) < 42,
  )
  if (!recent.length || recent.at(-1)?.date !== day.date) return null
  const load = recent.reduce((total, row) => total + row.load, 0)
  const share = (key: 'swimLoad' | 'bikeLoad' | 'runLoad'): number =>
    load > 0 ? recent.reduce((total, row) => total + row[key], 0) / load : 0
  const longest: Record<PaceSport, number> = { swim: 0, bike: 0, run: 0 }
  const recentSessions: Record<PaceSport, number> = { swim: 0, bike: 0, run: 0 }
  for (const activity of analytics.activities) {
    if (
      !isPaceSport(activity.sport) ||
      activity.date > day.date ||
      dayNumber(day.date) - dayNumber(activity.date) >= 42
    )
      continue
    longest[activity.sport] = Math.max(longest[activity.sport], activity.distanceKm)
    recentSessions[activity.sport]++
  }
  return {
    day,
    daily: analytics.daily.filter(row => row.date <= day.date && !row.warmup),
    weeklyLoad: (load / recent.length) * 7,
    shares: { swim: share('swimLoad'), bike: share('bikeLoad'), run: share('runLoad') },
    k42: analytics.meta.method.k42,
    k7: analytics.meta.method.k7,
    longest,
    recentSessions,
  }
}

export const projectRaceDay = (
  baseline: RaceProjectionBaseline,
  date: string,
  scenario: RaceScenario,
): PaceDayState => {
  const days = Math.max(0, dayNumber(date) - dayNumber(baseline.day.date))
  let { ctl, atl, swimCtl, bikeCtl, runCtl } = baseline.day
  for (let i = 0; i < days; i++) {
    // Feed states are measured before that day's load, including the final observed day.
    const load =
      i === 0
        ? (baseline.daily.at(-1)?.load ?? 0)
        : (scenario.weeklyLoad / 7) * (days - i <= scenario.taperDays ? 0.5 : 1)
    const sportLoad = (sport: PaceSport): number =>
      i === 0 ? (baseline.daily.at(-1)?.[`${sport}Load`] ?? 0) : load * baseline.shares[sport]
    ctl += (load - ctl) * baseline.k42
    atl += (load - atl) * baseline.k7
    swimCtl += (sportLoad('swim') - swimCtl) * baseline.k42
    bikeCtl += (sportLoad('bike') - bikeCtl) * baseline.k42
    runCtl += (sportLoad('run') - runCtl) * baseline.k42
  }
  return {
    ...baseline.day,
    date,
    ctl,
    atl,
    tsb: ctl - atl,
    swimCtl,
    bikeCtl,
    runCtl,
    hrv: null,
    rhr: null,
    readiness: null,
    sleepDurationS: null,
    tempDeviationC: null,
  }
}

const FITNESS_KEYS = ['ctl', 'atl', 'tsb', 'swimCtl', 'bikeCtl', 'runCtl'] as const
export const boundedRaceDay = (
  baseline: RaceProjectionBaseline,
  day: PaceDayState,
): { day: PaceDayState; limited: boolean } => {
  const bounded = { ...day }
  let limited = false
  for (const key of FITNESS_KEYS) {
    const values = baseline.daily.map(row => row[key])
    if (!values.length) continue
    const value = clamp(day[key], Math.min(...values), Math.max(...values))
    limited ||= value !== day[key]
    bounded[key] = value
  }
  // Keep form consistent while restricting both fitness inputs to observed ranges.
  const form = baseline.daily.map(row => row.tsb)
  const fatigue = baseline.daily.map(row => row.atl)
  if (form.length) {
    const low = Math.max(Math.min(...fatigue), bounded.ctl - Math.max(...form))
    const high = Math.min(Math.max(...fatigue), bounded.ctl - Math.min(...form))
    const atl = clamp(bounded.atl, low, high)
    limited ||= atl !== bounded.atl
    bounded.atl = atl
  }
  bounded.tsb = bounded.ctl - bounded.atl
  return { day: bounded, limited }
}

export interface RaceTimeProjection {
  midSec: number
  fastSec: number
  slowSec: number
  tss: number | null
  complete: boolean
  splits: { sport: PaceSport; seconds: number; tss: number }[]
}

export const predictRaceTime = async (
  forecaster: PaceForecaster,
  day: PaceDayState,
  spec: RaceProjectionSpec,
  paceGainPct = 0,
  horizonDays = 0,
  enduranceExtrapolation = false,
): Promise<RaceTimeProjection | null> => {
  if (!spec.legs.length) return null
  const predictions = await Promise.all(spec.legs.map(leg => forecaster.forecastLegAt(day, leg)))
  let seconds = (spec.extraMinutes ?? 0) * 60
  let spread = 0
  const splits: RaceTimeProjection['splits'] = []
  for (const [i, leg] of spec.legs.entries()) {
    const prediction = predictions[i]
    if (
      !prediction ||
      !Number.isFinite(prediction.mu) ||
      prediction.mu <= 0 ||
      !Number.isFinite(prediction.sigma) ||
      prediction.sigma < 0
    )
      return null
    const penalty = leg.sport === 'run' ? 1 + spec.runPenaltyPct / 100 : 1
    const speedGain = 1 + paceGainPct / 100
    const time = (((leg.distanceKm * 1000) / prediction.mu) * penalty) / speedGain
    if (!Number.isFinite(time) || time <= 0) return null
    seconds += time
    const error = forecaster.validationErrors[leg.sport]
    // Session validation error supplies a planning margin, not a race confidence level.
    const validationSpread =
      error == null
        ? null
        : leg.sport === 'swim'
          ? (error * leg.distanceKm * 10 * penalty) / speedGain
          : leg.sport === 'run'
            ? (error * leg.distanceKm * penalty) / speedGain
            : (time * error) / (prediction.mu * 3.6)
    // Shared fitness and course errors correlate across legs, so add their margins.
    spread +=
      validationSpread == null
        ? (Z80 * time * prediction.sigma) / prediction.mu
        : 1.5 * validationSpread
    splits.push({
      sport: leg.sport,
      seconds: time,
      tss: ((100 * time) / 3600) * leg.intensity ** 2,
    })
  }
  const complete = spec.extraMinutes !== null
  const extraTss =
    spec.kind === 'hyrox' && complete
      ? ((100 * (spec.extraMinutes ?? 0)) / 60) * (spec.legs[0]?.intensity ?? 0.95) ** 2
      : 0
  // A planning range, widened for an unobserved horizon and hybrid station assumptions.
  spread =
    Math.max(spread, seconds * 0.1) +
    seconds * Math.min(0.15, (Math.max(0, horizonDays) / 180) * 0.15)
  if (enduranceExtrapolation) spread += seconds * 0.1
  if (spec.kind === 'hyrox') spread += (spec.extraMinutes ?? 0) * 60 * 0.25
  return {
    midSec: seconds,
    // Extra uncertainty expands the downside. Automatic upside stays within a 10% time gain;
    // larger ambitions belong in the explicit pace goal rather than an error band.
    fastSec: Math.max(seconds * 0.9, seconds - spread),
    slowSec: seconds + spread,
    tss: complete ? splits.reduce((total, split) => total + split.tss, extraTss) : null,
    complete,
    splits,
  }
}

export const raceTrendDates = (today: string): string[] =>
  [-42, -28, -14, 0].map(days => shiftIsoDay(today, days))
