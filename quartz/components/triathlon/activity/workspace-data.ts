import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { MyWindsockGraphs } from '../../../util/mywindsock-graphs'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { preferredActivityWind } from '../../../util/activity-wind'
import { myWindsockTimelineSeries } from '../../../util/mywindsock-graphs'
import { swimLengthMetrics } from '../../../util/swim-metrics'
import {
  activityCadenceScale,
  activityCadenceUnit,
  activityHeartRateTracePoints,
  activityPhysiologyTracePoints,
  activityThermalTracePoints,
  activityTraceUsesElapsedAxis,
  clock,
  formatAltitude,
  formatImpactLoadFactor,
  formatStepSpeedLoss,
  formatTemperature,
  formatThermalTemperature,
  KM_TO_MI,
  routeStreamFlags,
  runGroundContactTimeMs,
  runImpactLoadFactor,
  runStepSpeedLossMps,
  runStrideLengthValue,
  runVerticalOscillationCm,
} from '../../../util/triathlon-card'

export type WorkspaceAxis = 'time' | 'distance'
export interface WorkspaceSample {
  elapsedS: number
  distanceKm: number
  value: number | null
}
export interface WorkspaceTrace {
  id: string
  label: string
  color: string
  samples: WorkspaceSample[]
  format(value: number): string
  estimated?: boolean
  source?: string
  domain?: [number, number]
  axisGroup?: string
}

const swimIntervals = (d: StravaActivityDetail) =>
  d.sport === 'swim'
    ? d.swimIntervals.filter(
        interval =>
          Number.isFinite(interval.startElapsedS) &&
          Number.isFinite(interval.endElapsedS) &&
          interval.startElapsedS >= 0 &&
          interval.endElapsedS > interval.startElapsedS &&
          interval.distanceM > 0 &&
          interval.cumulativeDistanceM >= interval.distanceM,
      )
    : []

const swimIntervalStart = (startElapsedS: number, previousEndElapsedS: number): number =>
  // FIT starts are rounded to whole seconds; subsecond gaps are not recorded rests.
  startElapsedS <= previousEndElapsedS + 1 ? previousEndElapsedS : startElapsedS

export const workspaceTimeline = (
  d: StravaActivityDetail,
): Pick<WorkspaceSample, 'elapsedS' | 'distanceKm'>[] => {
  if (d.route.length >= 2 && d.swimLocation !== 'pool')
    return d.route.map(p => ({ elapsedS: p.elapsedS, distanceKm: p.d }))
  const intervals = swimIntervals(d)
  if (intervals.length) {
    const points: Pick<WorkspaceSample, 'elapsedS' | 'distanceKm'>[] = []
    for (const interval of intervals) {
      const previous = points.at(-1)
      const start = previous
        ? swimIntervalStart(interval.startElapsedS, previous.elapsedS)
        : interval.startElapsedS
      if (start >= interval.endElapsedS) continue
      points.push(
        { elapsedS: start, distanceKm: (interval.cumulativeDistanceM - interval.distanceM) / 1000 },
        { elapsedS: interval.endElapsedS, distanceKm: interval.cumulativeDistanceM / 1000 },
      )
    }
    return points
  }
  if (d.route.length >= 2) return d.route.map(p => ({ elapsedS: p.elapsedS, distanceKm: p.d }))
  return d.heartRateTrace.map(p => ({ elapsedS: p.elapsedS, distanceKm: p.distanceKm }))
}

export const workspaceLocationAt = (
  samples: readonly Pick<WorkspaceSample, 'elapsedS' | 'distanceKm'>[],
  axis: WorkspaceAxis,
  position: number,
): Pick<WorkspaceSample, 'elapsedS' | 'distanceKm'> => {
  const pick = (p: Pick<WorkspaceSample, 'elapsedS' | 'distanceKm'>) =>
    axis === 'time' ? p.elapsedS : p.distanceKm
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (pick(samples[middle]) < position) low = middle + 1
    else high = middle
  }
  const right = samples[Math.min(low, samples.length - 1)]
  const left = samples[Math.max(0, low - 1)]
  const span = right && left ? pick(right) - pick(left) : 0
  const fraction = span > 0 ? Math.max(0, Math.min(1, (position - pick(left)) / span)) : 0
  return {
    elapsedS:
      axis === 'time'
        ? position
        : left
          ? left.elapsedS + fraction * (right.elapsedS - left.elapsedS)
          : 0,
    distanceKm:
      axis === 'distance'
        ? position
        : left
          ? left.distanceKm + fraction * (right.distanceKm - left.distanceKm)
          : 0,
  }
}

export const workspaceTraces = (
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): WorkspaceTrace[] => {
  const traces: WorkspaceTrace[] = []
  const route = d.route
  const flags = routeStreamFlags(d)
  const add = (
    id: string,
    label: string,
    color: string,
    samples: WorkspaceSample[],
    format: WorkspaceTrace['format'],
    estimated = false,
    source?: string,
  ): void => {
    if (samples.filter(sample => sample.value != null && Number.isFinite(sample.value)).length < 2)
      return
    traces.push({
      id,
      label,
      color,
      samples,
      format,
      estimated,
      ...(source ? { source } : {}),
      ...(id === 'elevation' ? { axisGroup: 'elevation' } : {}),
    })
  }
  const routeSamples = (pick: (point: StravaActivityDetail['route'][number]) => number | null) =>
    route.map(point => ({ elapsedS: point.elapsedS, distanceKm: point.d, value: pick(point) }))
  const numeric =
    (unit: string, digits = 0) =>
    (value: number) =>
      `${value.toFixed(digits)} ${unit}`
  const traceDistance = (point: { d: number }): number =>
    activityTraceUsesElapsedAxis(d) ? 0 : point.d
  const terrain =
    route.length >= 2 &&
    !(d.sport === 'swim' && d.swimLocation === 'pool') &&
    (d.mapRoute.some(segment => segment.length >= 2) || route.some(point => point.alt !== 0))
  if (terrain)
    add(
      'elevation',
      'elevation',
      'var(--darkgray)',
      routeSamples(p => p.alt),
      v => formatAltitude(presentation, v),
    )
  if (flags.power)
    add(
      'power',
      'power',
      '#8b6fd6',
      routeSamples(p => p.w),
      numeric('W'),
    )
  if (d.sport === 'walk' && !d.deviceWatts && d.walkPower)
    add(
      'walk-power',
      'walking power',
      '#8b6fd6',
      d.walkPower.points.map(point => ({
        elapsedS: point.elapsedS,
        distanceKm: point.distanceKm,
        value: point.watts,
      })),
      numeric('W'),
      true,
    )
  const cycling = d.cyclingPowerTrace
  if (cycling) {
    const points = cycling.points
    const samples = (pick: (p: (typeof points)[number]) => number | null) =>
      cycling.points.map(p => ({ elapsedS: p.elapsedS, distanceKm: p.distanceKm, value: pick(p) }))
    add(
      'power-30s',
      '30 s average power',
      '#205ea6',
      samples(p => p.power30sWatts),
      numeric('W'),
    )
    add(
      'power-5m',
      '5 min average power',
      '#da702c',
      samples(p => p.power5mWatts),
      numeric('W'),
    )
  }
  const intervals = swimIntervals(d)
  const swimSamples = (
    pick: (interval: StravaActivityDetail['swimIntervals'][number]) => number | null,
  ): WorkspaceSample[] => {
    const samples: WorkspaceSample[] = []
    for (const interval of intervals) {
      const previous = samples.at(-1)
      const start = previous
        ? swimIntervalStart(interval.startElapsedS, previous.elapsedS)
        : interval.startElapsedS
      if (start >= interval.endElapsedS) continue
      if (previous && start > previous.elapsedS) samples.push({ ...previous, value: null })
      const value = pick(interval)
      samples.push(
        {
          elapsedS: start,
          distanceKm: (interval.cumulativeDistanceM - interval.distanceM) / 1000,
          value,
        },
        { elapsedS: interval.endElapsedS, distanceKm: interval.cumulativeDistanceM / 1000, value },
      )
    }
    return samples
  }
  add(
    'swim-pace',
    'pace',
    '#287dd1',
    swimSamples(p => p.paceSPer100m),
    v => `${clock(v)} /100m`,
  )
  add(
    'stroke-rate',
    'stroke rate',
    '#3aa99f',
    swimSamples(p => p.strokeRateSpm),
    numeric('spm'),
  )
  if (d.swimLocation === 'pool') {
    add(
      'swim-cadence',
      'cadence',
      '#a47c1b',
      swimSamples(p => swimLengthMetrics(p)?.strokesPerLength ?? null),
      numeric('str/length', 1),
    )
    add(
      'swolf',
      'SWOLF',
      '#8b6fd6',
      swimSamples(p => swimLengthMetrics(p)?.swolf ?? null),
      v => v.toFixed(0),
    )
  }
  add(
    'hr',
    'heart rate',
    '#d14d41',
    activityHeartRateTracePoints(d).map(p => ({
      elapsedS: p.elapsedS,
      distanceKm: traceDistance(p),
      value: p.heartRate,
    })),
    numeric('bpm'),
  )
  if (flags.cad)
    add(
      'cadence',
      'cadence',
      '#3aa99f',
      routeSamples(p => (p.cad > 0 ? p.cad * activityCadenceScale(d.sport) : null)),
      numeric(activityCadenceUnit(d.sport)),
    )
  if (route.some(p => p.speedKph > 0))
    add(
      'speed',
      'speed',
      '#287dd1',
      routeSamples(p => p.speedKph),
      v =>
        numeric(
          presentation.distance === 'imperial' ? 'mph' : 'km/h',
          1,
        )(v * (presentation.distance === 'imperial' ? KM_TO_MI : 1)),
    )
  add(
    'respiration',
    'respiration',
    '#a47c1b',
    routeSamples(p => p.resp),
    numeric('brpm', 1),
  )
  add(
    'temperature',
    'device temp',
    '#af3029',
    routeSamples(p => p.tempC),
    v => formatTemperature(presentation, v),
  )
  const thermal = activityThermalTracePoints(d)
  const thermalSamples = (pick: (p: (typeof thermal)[number]) => number | null) =>
    thermal.map(p => ({ elapsedS: p.elapsedS, distanceKm: traceDistance(p), value: pick(p) }))
  add(
    'core-temperature',
    'CORE temperature',
    '#df9f22',
    thermalSamples(p => p.coreTemperatureC),
    v => formatThermalTemperature(presentation, v),
  )
  add(
    'skin-temperature',
    'skin temperature',
    '#a02f6f',
    thermalSamples(p => p.skinTemperatureC),
    v => formatThermalTemperature(presentation, v),
  )
  add(
    'heat-strain',
    'heat strain index',
    '#f27440',
    thermalSamples(p => p.heatStrainIndex),
    v => v.toFixed(1),
  )
  const physiology = (metric: 'stamina' | 'performanceCondition') =>
    activityPhysiologyTracePoints(d, metric).map(p => ({
      elapsedS: p.elapsedS,
      distanceKm: traceDistance(p),
      value: p[metric] ?? null,
    }))
  add(
    'stamina',
    'stamina',
    '#66800b',
    physiology('stamina'),
    numeric('%'),
    d.staminaTrace?.source === 'garden-estimate' ||
      (route.filter(point => point.stamina != null).length < 2 &&
        (d.swimPhysiology != null || d.heartRatePhysiology != null)),
  )
  add(
    'condition',
    'performance condition',
    '#6c6bbd',
    physiology('performanceCondition'),
    v => `${v > 0 ? '+' : ''}${v.toFixed(0)}`,
    d.performanceConditionTrace?.source === 'garden-estimate' ||
      (route.filter(point => point.performanceCondition != null).length < 2 &&
        (d.swimPhysiology != null || d.heartRatePhysiology != null)),
  )
  if (d.sport === 'run') {
    add(
      'stride',
      'stride length',
      '#3aa99f',
      routeSamples(p => runStrideLengthValue(d, p)),
      numeric('m', 2),
    )
    add(
      'ground-contact',
      'ground contact time',
      '#a47c1b',
      routeSamples(runGroundContactTimeMs),
      numeric('ms'),
    )
    add(
      'vertical',
      'vertical oscillation',
      '#8b6fd6',
      routeSamples(runVerticalOscillationCm),
      numeric('cm', 1),
    )
    add(
      'step-speed-loss',
      'step speed loss',
      '#a02f6f',
      routeSamples(runStepSpeedLossMps),
      formatStepSpeedLoss,
    )
    add(
      'impact-load-factor',
      'impact load factor',
      '#f27440',
      routeSamples(runImpactLoadFactor),
      formatImpactLoadFactor,
    )
  }
  const environment = d.analyses.derived.environment
  if (environment) {
    const points = environment.samples
    const samples = (pick: (p: (typeof points)[number]) => number | null) =>
      environment.samples.map(p => ({
        elapsedS: p.elapsedS,
        distanceKm: p.distanceKm,
        value: pick(p),
      }))
    add(
      'ambient',
      'ambient temp',
      '#da702c',
      samples(p => p.ambientTemperatureC),
      v => formatTemperature(presentation, v),
      true,
    )
    add(
      'uv',
      'UV index',
      '#a47c1b',
      samples(p => p.uvIndex),
      v => v.toFixed(1),
      true,
    )
  }
  const wind = preferredActivityWind(d)
  add(
    'wind',
    'headwind',
    '#205ea6',
    wind.samples.map(point => ({
      elapsedS: point.elapsedS,
      distanceKm: point.distanceKm,
      value: point.headwindKph,
    })),
    value =>
      numeric(
        presentation.distance === 'imperial' ? 'mph' : 'km/h',
        1,
      )(value * (presentation.distance === 'imperial' ? KM_TO_MI : 1)),
    wind.provider === 'garden',
    wind.provider === 'mywindsock'
      ? `myWindsock · captured ${wind.capturedAt}`
      : 'Garden wind estimate',
  )
  const provider = d.analyses.native.myWindsockRoute
  if (provider) {
    type ProviderSample = (typeof provider.samples)[number]
    for (const [id, label, pick, unit] of [
      [
        'provider-crosswind',
        'myWindsock crosswind',
        (point: (typeof provider.samples)[number]) => point.providerCrosswindKph,
        'km/h',
      ],
      [
        'provider-weather-cost',
        'myWindsock weather cost',
        (point: (typeof provider.samples)[number]) => point.weatherCostW,
        'W',
      ],
      [
        'provider-air-penalty',
        'myWindsock moving air penalty',
        (point: (typeof provider.samples)[number]) => point.movingAirPenaltyKm,
        'km',
      ],
    ] satisfies [string, string, (point: ProviderSample) => number | null, string][]) {
      add(
        id,
        label,
        '#205ea6',
        provider.samples.map(point => ({
          elapsedS: point.elapsedS,
          distanceKm: point.distanceKm,
          value: pick(point),
        })),
        value =>
          unit === 'km/h'
            ? numeric(
                presentation.distance === 'imperial' ? 'mph' : 'km/h',
                1,
              )(value * (presentation.distance === 'imperial' ? KM_TO_MI : 1))
            : unit === 'km'
              ? numeric(
                  presentation.distance === 'imperial' ? 'mi' : 'km',
                  2,
                )(value * (presentation.distance === 'imperial' ? KM_TO_MI : 1))
              : numeric('W', 1)(value),
      )
    }
  }
  return traces
}

export const myWindsockWorkspaceTraces = (
  archive: MyWindsockGraphs,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): WorkspaceTrace[] => {
  const timeline = workspaceTimeline(d)
  const colors = ['#205ea6', '#da702c', '#3aa99f', '#8b6fd6', '#d14d41', '#a47c1b']
  return myWindsockTimelineSeries(archive).flatMap((series, index) => {
    if (series.points.some(point => point.elapsedS > d.elapsedTimeS + 60)) return []
    const format = (value: number): string => {
      if (series.unit === 'm') return formatAltitude(presentation, value)
      if (series.unit === 'km/h')
        return `${(value * (presentation.distance === 'imperial' ? KM_TO_MI : 1)).toFixed(1)} ${presentation.distance === 'imperial' ? 'mph' : 'km/h'}`
      if (series.unit === 'km')
        return `${(value * (presentation.distance === 'imperial' ? KM_TO_MI : 1)).toFixed(2)} ${presentation.distance === 'imperial' ? 'mi' : 'km'}`
      return `${value.toFixed(series.unit === 'm2' ? 3 : 1)} ${series.unit === 'deg' ? '°' : series.unit === 'm2' ? 'm²' : series.unit}`
    }
    return [
      {
        id: series.id,
        label: `myWindsock ${series.label}`,
        color: colors[index % colors.length],
        samples: series.points.map(point => ({
          elapsedS: point.elapsedS,
          distanceKm: workspaceLocationAt(timeline, 'time', point.elapsedS).distanceKm,
          value: point.value,
        })),
        format,
        source: `myWindsock native chart · ${archive.capturedAt}`,
        ...(series.domain ? { domain: series.domain, axisGroup: 'mywindsock-resistance' } : {}),
        ...(series.unit === 'm2' ? { axisGroup: 'mywindsock-cda' } : {}),
        ...(series.unit === 'm' ? { axisGroup: 'elevation' } : {}),
      },
    ]
  })
}

export const workspacePosition = (sample: WorkspaceSample, axis: WorkspaceAxis): number =>
  axis === 'time' ? sample.elapsedS : sample.distanceKm

export const workspaceTraceDomain = (trace: WorkspaceTrace): [number, number] => {
  if (trace.domain) return trace.domain
  let low = Infinity
  let high = -Infinity
  for (const sample of trace.samples) {
    if (sample.value == null || !Number.isFinite(sample.value)) continue
    low = Math.min(low, sample.value)
    high = Math.max(high, sample.value)
  }
  return [low, high]
}

export const workspaceTraceY = (value: number, [low, high]: readonly [number, number]): number =>
  high === low ? 50 : ((high - value) / (high - low)) * 100

export const workspaceTracePaths = (
  trace: WorkspaceTrace,
  axis: WorkspaceAxis,
  start: number,
  end: number,
): { line: string; area: string } => {
  const domain = workspaceTraceDomain(trace)
  let line = ''
  let area = ''
  let segment = ''
  let firstX = 0
  let lastX = 0
  let previous: (WorkspaceSample & { value: number }) | null = null
  const close = (): void => {
    if (!segment) return
    line += `${segment} `
    area += `${segment} L ${lastX} 100 L ${firstX} 100 Z `
    segment = ''
  }
  const append = (position: number, value: number): void => {
    const x = Number((((position - start) / Math.max(end - start, 0.001)) * 100).toFixed(3))
    const y = workspaceTraceY(value, domain).toFixed(3)
    if (!segment) firstX = x
    segment += `${segment ? ' L' : 'M'} ${x} ${y}`
    lastX = x
  }
  for (const sample of trace.samples) {
    const position = workspacePosition(sample, axis)
    if (sample.value == null || !Number.isFinite(sample.value)) {
      close()
      previous = null
      continue
    }
    const value = sample.value
    if (previous) {
      const leftSample = previous
      const previousPosition = workspacePosition(leftSample, axis)
      if (position >= start && previousPosition <= end) {
        const valueAt = (boundary: number): number => {
          if (boundary === position) return value
          const fraction = (boundary - previousPosition) / (position - previousPosition)
          return leftSample.value + fraction * (value - leftSample.value)
        }
        if (!segment) {
          const left = Math.max(start, previousPosition)
          append(left, left === previousPosition ? previous.value : valueAt(left))
        }
        const right = Math.min(end, position)
        append(right, valueAt(right))
      }
    } else if (position >= start && position <= end) {
      append(position, sample.value)
    }
    if (position > end) close()
    previous = { ...sample, value: sample.value }
  }
  close()
  return { line, area }
}

export const workspaceValueAt = (trace: WorkspaceTrace, elapsedS: number): number | null => {
  const samples = trace.samples
  if (elapsedS < samples[0].elapsedS || elapsedS > samples[samples.length - 1].elapsedS) return null
  let low = 0
  let high = samples.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (samples[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const right = samples[low]
  const left = samples[Math.max(0, low - 1)]
  if (right.elapsedS === elapsedS) return right.value
  if (left.value == null || right.value == null) return null
  return elapsedS - left.elapsedS <= right.elapsedS - elapsedS ? left.value : right.value
}
