import type { ActivityKind, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type {
  ActivityComparisonDomain,
  ActivityComparisonMetricPoint,
  ActivityComparisonSeries,
} from './comparison-plot'
import {
  gradeAt,
  interpolatePositiveMetricSeries,
  runGroundContactTimeMs,
  runStrideLengthValue,
  runVerticalOscillationCm,
} from './activity-metric-data'
import {
  activityCadenceScale,
  activityCadenceUnit,
  clock,
  elevationValue,
  excludesZeroPower,
  formatAltitude,
  formatGroundContactTime,
  formatRespirationRate,
  formatStrideLength,
  formatTemperature,
  formatThermalTemperature,
  formatVerticalOscillation,
  isImperial,
  positiveMetric,
  speedKph,
  swimTrendNumber,
  temperatureUnit,
  temperatureValue,
} from './metric-text'
import { traceResolution } from './sensor-traces'

export type ActivityComparisonMetric =
  | 'elevation'
  | 'speed'
  | 'hr'
  | 'power'
  | 'cadence'
  | 'respiration'
  | 'temperature'
  | 'skin-temperature'
  | 'stride-length'
  | 'ground-contact-time'
  | 'vertical-oscillation'
  | 'swim-pace'
  | 'stroke-rate'

export const activityComparisonFractionForKey = (
  key: string,
  fraction: number,
  step: number,
): number | null => {
  let next: number
  if (key === 'ArrowLeft' || key === 'ArrowDown') next = fraction - step
  else if (key === 'ArrowRight' || key === 'ArrowUp') next = fraction + step
  else if (key === 'Home') next = 0
  else if (key === 'End') next = 1
  else return null
  return Math.min(1, Math.max(0, next))
}

export type ActivityComparisonMetricSpec = {
  metric: ActivityComparisonMetric
  title: string
  display: (value: number) => number
  tick: (value: number, sport: ActivityKind) => string
  includeZero: boolean
  domain?: (values: number[]) => ActivityComparisonDomain
}

export const activityComparisonMetricSpecs = (
  presentation: TriathlonPresentation,
): Record<ActivityComparisonMetric, ActivityComparisonMetricSpec> => ({
  elevation: {
    metric: 'elevation',
    title: 'elevation',
    display: value => elevationValue(presentation, value),
    tick: value =>
      `${Math.round(value).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft' : 'm'}`,
    includeZero: false,
  },
  speed: {
    metric: 'speed',
    title: 'speed',
    display: value => value,
    tick: value => speedKph(presentation, value),
    includeZero: true,
  },
  hr: {
    metric: 'hr',
    title: 'heart rate',
    display: value => value,
    tick: value => `${Math.round(value)} bpm`,
    includeZero: false,
  },
  power: {
    metric: 'power',
    title: 'power',
    display: value => value,
    tick: value => `${Math.round(value)} W`,
    includeZero: true,
  },
  cadence: {
    metric: 'cadence',
    title: 'cadence',
    display: value => value,
    tick: (value, sport) => `${Math.round(value)} ${activityCadenceUnit(sport)}`,
    includeZero: false,
  },
  respiration: {
    metric: 'respiration',
    title: 'respiration',
    display: value => value,
    tick: value => `${Math.round(value)} brpm`,
    includeZero: false,
  },
  temperature: {
    metric: 'temperature',
    title: 'temperature',
    display: value => temperatureValue(presentation, value),
    tick: value => `${Math.round(value)}${temperatureUnit(presentation)}`,
    includeZero: false,
  },
  'skin-temperature': {
    metric: 'skin-temperature',
    title: 'skin temperature',
    display: value => temperatureValue(presentation, value),
    tick: value => `${value.toFixed(2)}${temperatureUnit(presentation)}`,
    includeZero: false,
    domain: values => {
      if (values.length === 0) return { min: 0, max: 1 }
      const resolution = traceResolution(values, isImperial(presentation) ? 0.09 : 0.05)
      return { min: Math.min(...values) - resolution, max: Math.max(...values) + resolution }
    },
  },
  'stride-length': {
    metric: 'stride-length',
    title: 'stride length',
    display: value => value,
    tick: value => formatStrideLength(presentation, value),
    includeZero: false,
  },
  'ground-contact-time': {
    metric: 'ground-contact-time',
    title: 'ground contact time',
    display: value => value,
    tick: formatGroundContactTime,
    includeZero: false,
  },
  'vertical-oscillation': {
    metric: 'vertical-oscillation',
    title: 'vertical oscillation',
    display: value => value,
    tick: value => formatVerticalOscillation(presentation, value),
    includeZero: false,
  },
  'swim-pace': {
    metric: 'swim-pace',
    title: 'pace /100m',
    display: value => value,
    tick: value => clock(value),
    includeZero: false,
  },
  'stroke-rate': {
    metric: 'stroke-rate',
    title: 'stroke rate',
    display: value => value,
    tick: value => `${swimTrendNumber(value)} str/min`,
    includeZero: false,
  },
})

const BIKE_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = [
  'elevation',
  'speed',
  'hr',
  'power',
  'cadence',
  'respiration',
  'temperature',
  'skin-temperature',
]

const RUN_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = [
  'elevation',
  'speed',
  'hr',
  'power',
  'cadence',
  'stride-length',
  'ground-contact-time',
  'vertical-oscillation',
  'temperature',
]

const SWIM_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = ['swim-pace', 'stroke-rate']

export const activityComparisonMetricsForSport = (
  sport: ActivityKind,
): readonly ActivityComparisonMetric[] => {
  if (sport === 'run') return RUN_COMPARISON_METRICS
  if (sport === 'swim') return SWIM_COMPARISON_METRICS
  return BIKE_COMPARISON_METRICS
}

const comparisonPowerCapable = (activity: StravaActivityDetail): boolean =>
  activity.deviceWatts || activity.route.some(point => Number.isFinite(point.w) && point.w > 0)

const comparisonMetricPointValue = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
  metric: ActivityComparisonMetric,
  powerCapable: boolean,
): number | null => {
  let value: number | null
  switch (metric) {
    case 'elevation':
      value = point.alt
      break
    case 'speed':
      value = point.speedKph > 0 ? point.speedKph : null
      break
    case 'hr':
      value = point.hr > 0 ? point.hr : null
      break
    case 'power':
      value =
        point.w > 0 ||
        (point.w === 0 &&
          powerCapable &&
          !(excludesZeroPower(presentation) && activity.sport === 'bike'))
          ? point.w
          : null
      break
    case 'cadence':
      value = point.cad > 0 ? point.cad * activityCadenceScale(activity.sport) : null
      break
    case 'respiration':
      value = point.resp != null && point.resp > 0 ? point.resp : null
      break
    case 'temperature':
      value = point.tempC
      break
    case 'skin-temperature':
      value = point.skinTemperatureC
      break
    case 'stride-length':
      value = activity.sport === 'run' ? runStrideLengthValue(activity, point) : null
      break
    case 'ground-contact-time':
      value = activity.sport === 'run' ? runGroundContactTimeMs(point) : null
      break
    case 'vertical-oscillation':
      value = activity.sport === 'run' ? runVerticalOscillationCm(point) : null
      break
    case 'swim-pace':
    case 'stroke-rate':
      value = null
      break
  }
  return value != null && Number.isFinite(value) ? value : null
}

export const comparisonSwimMetricSegments = (
  activity: StravaActivityDetail,
  metric: 'swim-pace' | 'stroke-rate',
): ActivityComparisonMetricPoint[][] => {
  if (activity.sport !== 'swim') return []
  const segments: ActivityComparisonMetricPoint[][] = []
  let segment: ActivityComparisonMetricPoint[] = []
  const flush = (): void => {
    if (segment.length > 0) segments.push(segment)
    segment = []
  }
  for (const interval of activity.swimIntervals) {
    const distanceKm = interval.cumulativeDistanceM / 1_000
    const startDistanceKm = (interval.cumulativeDistanceM - interval.distanceM) / 1_000
    const value = metric === 'swim-pace' ? interval.paceSPer100m : interval.strokeRateSpm
    if (
      !Number.isFinite(startDistanceKm) ||
      !Number.isFinite(distanceKm) ||
      startDistanceKm < 0 ||
      distanceKm <= startDistanceKm ||
      !positiveMetric(value)
    ) {
      flush()
      continue
    }
    const previous = segment[segment.length - 1]
    if (previous && distanceKm < previous.distanceKm) flush()
    if (segment.length === 0) segment.push({ distanceKm: startDistanceKm, value })
    const last = segment[segment.length - 1]
    if (last?.distanceKm === distanceKm) {
      last.value = value
      continue
    }
    segment.push({ distanceKm, value })
  }
  flush()
  return segments
}

const comparisonMetricSegments = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
): ActivityComparisonMetricPoint[][] => {
  if (metric === 'swim-pace' || metric === 'stroke-rate')
    return comparisonSwimMetricSegments(activity, metric)
  const powerCapable = comparisonPowerCapable(activity)
  const normalizedValues =
    excludesZeroPower(presentation) &&
    activity.sport === 'bike' &&
    (metric === 'power' || metric === 'cadence')
      ? interpolatePositiveMetricSeries(activity.route, point =>
          comparisonMetricPointValue(presentation, activity, point, metric, powerCapable),
        )
      : null
  const segments: ActivityComparisonMetricPoint[][] = []
  let segment: ActivityComparisonMetricPoint[] = []
  const flush = (): void => {
    if (segment.length > 0) segments.push(segment)
    segment = []
  }
  for (const [index, point] of activity.route.entries()) {
    if (!Number.isFinite(point.d) || point.d < 0) {
      flush()
      continue
    }
    const value =
      normalizedValues?.[index] ??
      comparisonMetricPointValue(presentation, activity, point, metric, powerCapable)
    if (value == null) {
      flush()
      continue
    }
    const previous = segment[segment.length - 1]
    if (previous && point.d < previous.distanceKm) flush()
    const last = segment[segment.length - 1]
    if (last?.distanceKm === point.d) {
      last.value = value
      continue
    }
    segment.push({ distanceKm: point.d, value })
  }
  flush()
  return segments
}

export const activityComparisonMetricAtDistance = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
  distanceKm: number,
): number | null => {
  if (!Number.isFinite(distanceKm)) return null
  for (const segment of comparisonMetricSegments(presentation, activity, metric)) {
    if (distanceKm < segment[0].distanceKm || distanceKm > segment[segment.length - 1].distanceKm)
      continue
    for (let index = 0; index < segment.length; index++) {
      const point = segment[index]
      if (distanceKm === point.distanceKm) return point.value
      if (index === 0 || distanceKm > point.distanceKm) continue
      const previous = segment[index - 1]
      const span = point.distanceKm - previous.distanceKm
      if (span <= 0) return point.value
      const fraction = (distanceKm - previous.distanceKm) / span
      return previous.value + (point.value - previous.value) * fraction
    }
  }
  return null
}

export const activityComparisonDisplayValueAtDistance = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
  distanceKm: number,
): string => {
  const value = activityComparisonMetricAtDistance(presentation, activity, metric, distanceKm)
  if (value == null) return '—'
  if (metric === 'elevation') {
    const altitude = formatAltitude(presentation, value)
    const route = activity.route
    let index = route.findIndex(point => point.d >= distanceKm)
    if (index < 0) return altitude
    if (index > 0 && distanceKm - route[index - 1].d < route[index].d - distanceKm) index--
    const window = route.slice(Math.max(0, index - 2), index + 3)
    if (
      window.length < 2 ||
      window.some(
        (point, i) =>
          !Number.isFinite(point.alt) ||
          !Number.isFinite(point.d) ||
          (i > 0 && point.d < window[i - 1].d),
      ) ||
      window[window.length - 1].d <= window[0].d
    )
      return altitude
    const grade = Math.round(gradeAt(route, index) * 10) / 10
    return `${altitude} · ${grade >= 0 ? '+' : ''}${grade.toFixed(1)}%`
  }
  if (metric === 'speed') return speedKph(presentation, value)
  if (metric === 'hr') return `${Math.round(value)} bpm`
  if (metric === 'power') return `${Math.round(value)} W`
  if (metric === 'cadence') return `${Math.round(value)} ${activityCadenceUnit(activity.sport)}`
  if (metric === 'respiration') return formatRespirationRate(value)
  if (metric === 'stride-length') return formatStrideLength(presentation, value)
  if (metric === 'ground-contact-time') return formatGroundContactTime(value)
  if (metric === 'vertical-oscillation') return formatVerticalOscillation(presentation, value)
  if (metric === 'swim-pace') return `${clock(value)} /100m`
  if (metric === 'stroke-rate') return `${swimTrendNumber(value)} str/min`
  if (metric === 'skin-temperature') return formatThermalTemperature(presentation, value)
  return formatTemperature(presentation, value)
}

export const comparisonMetricSeries = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  index: number,
  spec: ActivityComparisonMetricSpec,
): ActivityComparisonSeries => {
  const segments: ActivityComparisonMetricPoint[][] = []
  const values: number[] = []
  for (const rawSegment of comparisonMetricSegments(presentation, activity, spec.metric)) {
    const segment = rawSegment
      .map(point => ({ distanceKm: point.distanceKm, value: spec.display(point.value) }))
      .filter(point => Number.isFinite(point.value))
    if (segment.length < 2) continue
    segments.push(segment)
    values.push(...segment.map(point => point.value))
  }
  return { activity, index, segments, values }
}
