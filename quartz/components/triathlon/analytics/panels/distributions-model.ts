import type { Sport } from '../../../../plugins/stores/strava'

export type DistributionSport = Sport

export type DistributionRange = '7' | '14' | '30' | '60' | 'custom'

export type TelemetryTrend = 'up' | 'down' | 'flat'

export interface DistributionModel {
  sport: DistributionSport
  range: DistributionRange
  startDate: string
}

export interface DistributionBounds {
  minimumDate: string
  maximumDate: string
  sports: readonly DistributionSport[]
}

export type DistributionMessage =
  | { type: 'restore'; model: DistributionModel }
  | { type: 'select-sport'; sport: DistributionSport }
  | { type: 'select-range'; range: DistributionRange }
  | { type: 'select-date'; date: string }
  | { type: 'clear-date' }

export const DISTRIBUTION_DAY_MS = 86_400_000

export const DISTRIBUTION_RANGES: { key: DistributionRange; label: string; days: number | null }[] =
  [
    { key: '7', label: '7d', days: 7 },
    { key: '14', label: '14d', days: 14 },
    { key: '30', label: '30d', days: 30 },
    { key: '60', label: '60d', days: 60 },
    { key: 'custom', label: 'custom', days: null },
  ]

export const distributionDateShift = (date: string, days: number): string =>
  new Date(Date.parse(`${date}T00:00:00Z`) + days * DISTRIBUTION_DAY_MS).toISOString().slice(0, 10)

export const telemetryTrend = (values: readonly (number | null)[]): TelemetryTrend | null => {
  const observed = values.filter(
    (value): value is number => value != null && Number.isFinite(value),
  )
  if (observed.length < 2) return null
  const previous = observed[observed.length - 2]
  const latest = observed[observed.length - 1]
  return latest > previous ? 'up' : latest < previous ? 'down' : 'flat'
}

export const telemetryWeightedAverage = (
  observations: readonly { value: number | null; observedSeconds: number }[],
): number | null => {
  let valueSeconds = 0
  let observedSeconds = 0
  for (const observation of observations) {
    if (
      observation.value == null ||
      !Number.isFinite(observation.value) ||
      !Number.isFinite(observation.observedSeconds) ||
      observation.observedSeconds <= 0
    )
      continue
    valueSeconds += observation.value * observation.observedSeconds
    observedSeconds += observation.observedSeconds
  }
  return observedSeconds > 0 ? valueSeconds / observedSeconds : null
}

const clampDate = (date: string, bounds: DistributionBounds): string =>
  date < bounds.minimumDate
    ? bounds.minimumDate
    : date > bounds.maximumDate
      ? bounds.maximumDate
      : date

const startForRange = (
  range: DistributionRange,
  current: string,
  bounds: DistributionBounds,
): string => {
  const days = DISTRIBUTION_RANGES.find(option => option.key === range)?.days
  return days == null
    ? clampDate(current, bounds)
    : clampDate(distributionDateShift(bounds.maximumDate, -(days - 1)), bounds)
}

export const initialDistributionModel = (bounds: DistributionBounds): DistributionModel => {
  const sport = bounds.sports.includes('bike') ? 'bike' : (bounds.sports[0] ?? 'run')
  return { sport, range: '30', startDate: startForRange('30', bounds.maximumDate, bounds) }
}

export const updateDistributions = (
  model: DistributionModel,
  message: DistributionMessage,
  bounds: DistributionBounds,
): DistributionModel => {
  if (message.type === 'select-sport') return { ...model, sport: message.sport }
  if (message.type === 'select-range')
    return {
      ...model,
      range: message.range,
      startDate: startForRange(message.range, model.startDate, bounds),
    }
  if (message.type === 'select-date')
    return { ...model, range: 'custom', startDate: clampDate(message.date, bounds) }
  if (message.type === 'clear-date')
    return { ...model, range: '30', startDate: startForRange('30', model.startDate, bounds) }
  return {
    sport: message.model.sport,
    range: message.model.range,
    startDate: startForRange(message.model.range, message.model.startDate, bounds),
  }
}
