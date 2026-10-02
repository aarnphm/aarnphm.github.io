import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import {
  activityCadenceScale,
  activityCadenceUnit,
  activityHeartRateTracePoints,
  activityPhysiologyTracePoints,
  activityThermalTracePoints,
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
  ): void => {
    if (samples.filter(sample => sample.value != null && Number.isFinite(sample.value)).length < 2)
      return
    traces.push({ id, label, color, samples, format, estimated })
  }
  const routeSamples = (pick: (point: StravaActivityDetail['route'][number]) => number | null) =>
    route.map(point => ({ elapsedS: point.elapsedS, distanceKm: point.d, value: pick(point) }))
  const numeric =
    (unit: string, digits = 0) =>
    (value: number) =>
      `${value.toFixed(digits)} ${unit}`
  const hasDistance = d.distanceKm > 0 && route.length >= 2
  // Route-less physiology helpers use seconds in their d field, so distance must stay absent.
  const traceDistance = (point: { d: number }): number => (hasDistance ? point.d : 0)
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
      'wind',
      'headwind',
      '#205ea6',
      samples(p => p.headwindKph),
      v =>
        numeric(
          presentation.distance === 'imperial' ? 'mph' : 'km/h',
          1,
        )(v * (presentation.distance === 'imperial' ? KM_TO_MI : 1)),
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
  return traces
}

export const workspacePosition = (sample: WorkspaceSample, axis: WorkspaceAxis): number =>
  axis === 'time' ? sample.elapsedS : sample.distanceKm

export const workspaceTraceDomain = (trace: WorkspaceTrace): [number, number] => {
  let low = Infinity
  let high = -Infinity
  for (const sample of trace.samples) {
    if (sample.value == null || !Number.isFinite(sample.value)) continue
    low = Math.min(low, sample.value)
    high = Math.max(high, sample.value)
  }
  return [low, high]
}

export const workspaceTracePaths = (
  trace: WorkspaceTrace,
  axis: WorkspaceAxis,
  start: number,
  end: number,
): { line: string; area: string } => {
  const [low, high] = workspaceTraceDomain(trace)
  let line = ''
  let area = ''
  let segment = ''
  let firstX = 0
  let lastX = 0
  const close = (): void => {
    if (!segment) return
    line += `${segment} `
    area += `${segment} L ${lastX} 100 L ${firstX} 100 Z `
    segment = ''
  }
  for (const sample of trace.samples) {
    const position = workspacePosition(sample, axis)
    if (
      sample.value == null ||
      !Number.isFinite(sample.value) ||
      position < start ||
      position > end
    ) {
      close()
      continue
    }
    const x = Number((((position - start) / Math.max(end - start, 0.001)) * 100).toFixed(3))
    const y = (high === low ? 50 : ((high - sample.value) / (high - low)) * 100).toFixed(3)
    if (!segment) firstX = x
    segment += `${segment ? ' L' : 'M'} ${x} ${y}`
    lastX = x
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
