import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { DetailCtx } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { ActivityAnalysisRange } from './analysis'
import type { ActivityAnalysisController } from './analysis'
import type { ActivityRangeChange } from './analysis'
import type { ScrubSurface } from './analysis'
import type { DetailPayload } from './data'
import { activityStatRows } from '../../../util/triathlon-card'
import { activityHeartRateTracePoints } from '../../../util/triathlon-card'
import { activityThermalTracePoints } from '../../../util/triathlon-card'
import { activityTraceUsesElapsedAxis } from '../../../util/triathlon-card'
import { activityTrainingEffectLabel } from '../../../util/triathlon-card'
import { buildActivity as buildActivityNode } from '../../../util/triathlon-card'
import { buildCoreTemperatureTrace as buildCoreTemperatureTraceNode } from '../../../util/triathlon-card'
import { buildCyclingBestEfforts as buildCyclingBestEffortsNode } from '../../../util/triathlon-card'
import { buildHeatStrainTrace as buildHeatStrainTraceNode } from '../../../util/triathlon-card'
import { buildMuscleOxygenTrace as buildMuscleOxygenTraceNode } from '../../../util/triathlon-card'
import { buildPedalSmoothnessChart as buildPedalSmoothnessChartNode } from '../../../util/triathlon-card'
import { buildPowerBalanceChart as buildPowerBalanceChartNode } from '../../../util/triathlon-card'
import { buildPowerPhaseChart as buildPowerPhaseChartNode } from '../../../util/triathlon-card'
import { buildRespirationTrace as buildRespirationTraceNode } from '../../../util/triathlon-card'
import { buildRiderPositionChart as buildRiderPositionChartNode } from '../../../util/triathlon-card'
import { buildRunGroundContactTrace as buildRunGroundContactTraceNode } from '../../../util/triathlon-card'
import { buildRunLapSplits as buildRunLapSplitsNode } from '../../../util/triathlon-card'
import { buildRunStrideTrace as buildRunStrideTraceNode } from '../../../util/triathlon-card'
import { buildRunVerticalOscillationTrace as buildRunVerticalOscillationTraceNode } from '../../../util/triathlon-card'
import { buildShiftingChart as buildShiftingChartNode } from '../../../util/triathlon-card'
import { buildSkinTemperatureTrace as buildSkinTemperatureTraceNode } from '../../../util/triathlon-card'
import { buildStaminaChart as buildStaminaChartNode } from '../../../util/triathlon-card'
import { buildTemperatureTrace as buildTemperatureTraceNode } from '../../../util/triathlon-card'
import { buildTorqueEffectivenessChart as buildTorqueEffectivenessChartNode } from '../../../util/triathlon-card'
import { buildTrainingEffectDetails as buildTrainingEffectDetailsNode } from '../../../util/triathlon-card'
import { clock } from '../../../util/triathlon-card'
import { cyclingDynamicsIndexAtDistance } from '../../../util/triathlon-card'
import { dominantTrainingEffectGroup } from '../../../util/triathlon-card'
import { formatAltitude } from '../../../util/triathlon-card'
import { formatGroundContactTime } from '../../../util/triathlon-card'
import { formatMuscleOxygen } from '../../../util/triathlon-card'
import { formatRespirationRate } from '../../../util/triathlon-card'
import { formatStrideLength } from '../../../util/triathlon-card'
import { formatTemperature } from '../../../util/triathlon-card'
import { formatThermalTemperature } from '../../../util/triathlon-card'
import { formatVerticalOscillation } from '../../../util/triathlon-card'
import { gearShiftAtFraction } from '../../../util/triathlon-card'
import { gradeAt } from '../../../util/triathlon-card'
import { hasHeartRateTrace } from '../../../util/triathlon-card'
import { interpolatePositiveMetricSeries } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { positiveMetricDomain } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { powerBalanceText } from '../../../util/triathlon-card'
import { runGroundContactTimeMs } from '../../../util/triathlon-card'
import { routeStreamFlags } from '../../../util/triathlon-card'
import { runStrideLengthLabel } from '../../../util/triathlon-card'
import { runStrideLengthValue } from '../../../util/triathlon-card'
import { runVerticalOscillationCm } from '../../../util/triathlon-card'
import { riderPositionAtDistance } from '../../../util/triathlon-card'
import { scrubDist } from '../../../util/triathlon-card'
import { zoneClock } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import {
  triathlonTraceEnabled,
  type TriathlonTraceSettings,
} from '../../../util/triathlon-trace-settings'
import { buildMatchedRideGroup } from '../analytics/panels/matched'
import { buildMatchedRunGroup } from '../analytics/panels/matched'
import {
  CAD_RAMP,
  ELEV_RAMP,
  HEAT_RAMP,
  HR_RAMP,
  rampGradient,
  RESP_RAMP,
  SPD_RAMP,
  STRIDE_RAMP,
} from '../maps/palette'
import { createDomFactory } from '../runtime/dom'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import { nextMapMetricShortcutIndex } from '../shell/command-palette'
import { analysisRate } from './analysis'
import { linkScrub } from './analysis'
import { detailContextFromPayload } from './data'
import {
  buildElevation,
  buildHeartRateTrace,
  buildHrZones,
  buildIcon,
  buildPool,
  buildPowerCurve,
  buildPowerHist,
  buildPowerZones,
  buildTrace,
  statRow,
  zoneDuo,
} from './primitives'
import { cyclingChartMode, setCyclingChartMode } from './scrub'

export const buildHeatRoute = (
  route: StravaActivityDetail['route'],
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  colors?: string[],
  zeroGap = false,
): SVGElement => {
  const ramp = colors?.length ?? 7
  const pad = 6
  const span = 100 - pad * 2
  const vals = route.map((p, i) => pick(p, i))
  const pool = zeroGap ? vals.filter(v => v > 0) : vals
  let lo = Infinity
  let hi = -Infinity
  for (const v of pool.length ? pool : vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const range = hi > lo ? hi - lo : 1
  const sx = (p: StravaActivityDetail['route'][number]): number => pad + p.x * span
  const sy = (p: StravaActivityDetail['route'][number]): number => pad + (1 - p.y) * span
  const s = svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  const g = svg('g', { class: 'tri-heat' })
  for (let i = 0; i < route.length - 1; i++) {
    if (zeroGap && (vals[i] <= 0 || vals[i + 1] <= 0)) continue
    const mid = (vals[i] + vals[i + 1]) / 2
    const t = Math.min(1, Math.max(0, (mid - lo) / range))
    const bucket = Math.min(ramp, Math.max(1, Math.ceil(t * ramp) || 1))
    const attrs: Record<string, string | number> = {
      d: `M ${sx(route[i]).toFixed(2)} ${sy(route[i]).toFixed(2)} L ${sx(route[i + 1]).toFixed(2)} ${sy(route[i + 1]).toFixed(2)}`,
      class: colors ? 'tri-heat-seg' : `tri-heat-seg tri-heat--${bucket}`,
    }
    if (colors) attrs.style = `stroke: ${colors[bucket - 1]}`
    g.appendChild(svg('path', attrs))
  }
  s.appendChild(g)
  s.appendChild(svg('path', { class: 'tri-route-selected', d: '' }))
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return s
}

export const buildHeatLegend = (
  lo: number,
  hi: number,
  fmt: (v: number) => string,
  colors?: string[],
): HTMLElement => {
  const wrap = el('div', 'tri-map-legend')
  const bar = el('span', 'tri-map-legend-bar')
  if (colors) bar.style.background = rampGradient(colors)
  wrap.append(
    el('span', 'tri-map-legend-lo', fmt(lo)),
    bar,
    el('span', 'tri-map-legend-hi', fmt(hi)),
  )
  return wrap
}

const cyclingDynamicsPairAtDistance = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
  left: readonly (number | null)[],
  right: readonly (number | null)[],
  suffix: string,
): string => {
  const dynamics = d.cyclingDynamics
  if (!dynamics) return '—'
  const index = cyclingDynamicsIndexAtDistance(dynamics, point.d)
  if (index < 0) return '—'
  const format = (value: number | null | undefined): string =>
    value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(1)}${suffix}`
  return `L ${format(left[index])} · R ${format(right[index])}`
}

const powerPhaseAtDistance = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): string => {
  const dynamics = d.cyclingDynamics
  if (!dynamics) return '—'
  const index = cyclingDynamicsIndexAtDistance(dynamics, point.d)
  if (index < 0) return '—'
  const phase = (start: number | null | undefined, end: number | null | undefined): string =>
    start == null || end == null ? '—' : `${Math.round(start)}°→${Math.round(end)}°`
  return `L ${phase(dynamics.leftPowerPhaseStart[index], dynamics.leftPowerPhaseEnd[index])} · R ${phase(dynamics.rightPowerPhaseStart[index], dynamics.rightPowerPhaseEnd[index])}`
}

const riderPositionAtPoint = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): string => {
  const dynamics = d.cyclingDynamics
  if (!dynamics) return '—'
  const position = riderPositionAtDistance(dynamics, point.d)
  return position == null ? '—' : triText(presentation.locale, position)
}

const gearAtPoint = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): ReturnType<typeof gearShiftAtFraction> => {
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  return gearShiftAtFraction(d.gearShifts, maxDistanceKm, point.d / maxDistanceKm)
}

const staminaAtPoint = (
  presentation: TriathlonPresentation,
  point: StravaActivityDetail['route'][number],
): string =>
  `${triText(presentation.locale, 'current')} ${point.stamina == null ? '—' : `${Math.round(point.stamina)}%`} · ${triText(presentation.locale, 'potential')} ${point.potentialStamina == null ? '—' : `${Math.round(point.potentialStamina)}%`}`

const requiredMapProfile = (profile: Element | null, label: string): HTMLElement => {
  if (!(profile instanceof HTMLElement)) throw new Error(`Missing ${label} map profile`)
  return profile
}

type ThermalMetric = 'temperature' | 'heat-strain' | 'core-temperature' | 'skin-temperature'

type MapTraceSurface = {
  wrap: HTMLElement
  fmt: (point: StravaActivityDetail['route'][number], index: number) => string
}

export interface MapMetric {
  label: string
  shortLabel: string
  ramp: string[]
  zeroGap?: boolean
  pick: (p: StravaActivityDetail['route'][number], i: number) => number
  fmt: (v: number) => string
  profile: (graphDomain?: ActivityAnalysisRange | null) => HTMLElement
  readout: (p: StravaActivityDetail['route'][number], i: number) => string
  traces?: (graphDomain?: ActivityAnalysisRange | null) => MapTraceSurface[]
  extra?: () => (HTMLElement | null)[]
}

export const metricSpecs = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  detailContext: DetailCtx,
): MapMetric[] => {
  const domF = createDomFactory(presentation)
  const route = d.route
  const imperial = presentation.distance === 'imperial'
  const filterPowerZeros = d.sport === 'bike' && presentation.powerSamples === 'exclude-zero'
  const powerValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.w)
    : route.map(point => point.w)
  const powerDomain = filterPowerZeros ? positiveMetricDomain(powerValues) : undefined
  const pace = route.map(point => point.speedKph)
  const hasPower = d.deviceWatts && route.some(p => p.w > 0)
  const hasHr = hasHeartRateTrace(d)
  const hasCad = route.some(p => p.cad > 0)
  const strideLabel = runStrideLengthLabel(d)
  const hasStride =
    d.sport === 'run' && route.filter(p => runStrideLengthValue(d, p) != null).length >= 2
  const hasGroundContact =
    d.sport === 'run' && route.filter(p => runGroundContactTimeMs(p) != null).length >= 2
  const hasVerticalOscillation =
    d.sport === 'run' && route.filter(p => runVerticalOscillationCm(p) != null).length >= 2
  const hasResp = route.some(p => p.resp != null && p.resp > 0)
  const hasElev = d.maxAlt > d.minAlt
  const flags = routeStreamFlags(d)
  const hasMuscleOxygen = flags.muscleOxygen
  const maxDistanceKm = Math.max(route.at(-1)?.d ?? d.distanceKm, 0.001)
  const hasRiderPosition =
    d.sport === 'bike' &&
    d.cyclingDynamics?.positionChanges.some(
      change => change.distanceKm >= 0 && change.distanceKm <= maxDistanceKm,
    ) === true
  const hasShifting = d.sport === 'bike' && d.gearShifts.length > 0
  const thermalMetrics: ThermalMetric[] = []
  if (flags.temp) thermalMetrics.push('temperature')
  if (flags.heatStrain) thermalMetrics.push('heat-strain')
  if (flags.coreTemperature) thermalMetrics.push('core-temperature')
  if (flags.skinTemperature) thermalMetrics.push('skin-temperature')
  const hasThermal = thermalMetrics.length > 0
  const primaryThermal = thermalMetrics[0] ?? 'temperature'
  const thermalValue = (
    point: StravaActivityDetail['route'][number],
    metric: ThermalMetric,
  ): number | null => {
    if (metric === 'temperature') return point.tempC
    if (metric === 'heat-strain') return point.heatStrainIndex
    if (metric === 'core-temperature') return point.coreTemperatureC
    return point.skinTemperatureC
  }
  const primaryThermalValues = route
    .map(point => thermalValue(point, primaryThermal))
    .filter((value): value is number => value != null)
  const primaryThermalAverage =
    primaryThermalValues.reduce((total, value) => total + value, 0) /
    Math.max(1, primaryThermalValues.length)
  const cadUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const paceFmt = (kmh: number): string => {
    return analysisRate(presentation, d.sport, kmh)
  }
  const paceTick = (kmh: number): string => {
    if (kmh <= 0) return '0'
    if (d.sport === 'bike')
      return imperial ? `${(kmh * KM_TO_MI).toFixed(0)}mph` : `${kmh.toFixed(0)}km/h`
    if (d.sport === 'swim') return clock(3600 / (kmh * 10))
    return clock(3600 / (kmh * (imperial ? KM_TO_MI : 1)))
  }
  const paceSpec: MapMetric = {
    label: d.sport === 'bike' ? 'speed' : 'pace',
    shortLabel: d.sport === 'bike' ? 'S' : 'P',
    ramp: SPD_RAMP,
    pick: (_p, i) => pace[i],
    fmt: paceFmt,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (_p, i) => pace[i],
        d.sport === 'bike' ? 'speed' : 'pace',
        () => '',
        paceTick,
        graphDomain,
      ),
    readout: (p, i) => `${scrubDist(presentation, p.d, d.sport)} · ${paceFmt(pace[i])}`,
  }
  const powerSpec: MapMetric = {
    label: 'power',
    shortLabel: 'W',
    ramp: HEAT_RAMP,
    pick: (p, i) => powerValues[i] ?? p.w,
    fmt: v => `${Math.round(v)} W`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => powerValues[i] ?? p.w,
        'power',
        m => `${m} W peak`,
        v => `${Math.round(v)}w`,
        graphDomain,
        powerDomain,
        d.sport === 'bike' && detailContext.criticalPower
          ? {
              value: detailContext.criticalPower.criticalPowerWatts,
              label: `eCP ${detailContext.criticalPower.criticalPowerWatts} W`,
            }
          : null,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(powerValues[i] ?? p.w)} W`,
    traces: graphDomain => {
      const traces: MapTraceSurface[] = []
      const add = (wrap: Element | null, fmt: MapTraceSurface['fmt']): void => {
        if (wrap instanceof HTMLElement) traces.push({ wrap, fmt })
      }
      add(
        buildPowerBalanceChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.rightPowerPct == null || p.w <= 0 ? '—' : powerBalanceText(p.rightPowerPct)}`,
      )
      add(
        buildTorqueEffectivenessChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftTorqueEffectiveness ?? [], d.cyclingDynamics?.rightTorqueEffectiveness ?? [], '%')}`,
      )
      add(
        buildPedalSmoothnessChartNode(domF, d, null, false, graphDomain),
        p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftPedalSmoothness ?? [], d.cyclingDynamics?.rightPedalSmoothness ?? [], '%')}`,
      )
      add(
        buildPowerPhaseChartNode(domF, d, null, graphDomain),
        p => `${scrubDist(presentation, p.d, d.sport)} · ${powerPhaseAtDistance(d, p)}`,
      )
      return traces
    },
    extra: () => [
      zoneDuo(
        presentation,
        buildPowerCurve(presentation, d, detailContext),
        buildPowerHist(presentation, d),
      ),
      buildPowerZones(presentation, d, detailContext),
    ],
  }
  const hrSpec: MapMetric = {
    label: 'heart rate',
    shortLabel: 'HR',
    ramp: HR_RAMP,
    pick: p => p.hr,
    fmt: v => `${Math.round(v)} bpm`,
    profile: graphDomain => buildHeartRateTrace(presentation, d, graphDomain),
    readout: p => `${scrubDist(presentation, p.d, d.sport)} · ${p.hr} bpm`,
    extra: () => [
      buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null,
      buildHrZones(presentation, d, detailContext),
    ],
  }
  const cadScale = d.sport === 'run' ? 2 : 1
  const cadenceValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.cad * cadScale)
    : route.map(point => point.cad * cadScale)
  const cadenceDomain = filterPowerZeros ? positiveMetricDomain(cadenceValues) : undefined
  const cadSpec: MapMetric = {
    label: 'cadence',
    shortLabel: 'C',
    ramp: CAD_RAMP,
    zeroGap: !filterPowerZeros,
    pick: (p, i) => cadenceValues[i] ?? p.cad * cadScale,
    fmt: v => `${Math.round(v)} ${cadUnit}`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => cadenceValues[i] ?? p.cad * cadScale,
        'cadence',
        m => `${m} ${cadUnit} peak`,
        v => `${Math.round(v)}${cadUnit}`,
        graphDomain,
        cadenceDomain,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(cadenceValues[i] ?? p.cad * cadScale)} ${cadUnit}`,
  }
  const strideSpec: MapMetric = {
    label: strideLabel,
    shortLabel: 'SL',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runStrideLengthValue(d, p) ?? 0,
    fmt: value => formatStrideLength(presentation, value),
    profile: graphDomain => buildRunStrideTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const stride = runStrideLengthValue(d, p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
    },
  }
  const groundContactSpec: MapMetric = {
    label: 'ground contact time',
    shortLabel: 'GCT',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runGroundContactTimeMs(p) ?? 0,
    fmt: formatGroundContactTime,
    profile: graphDomain =>
      buildRunGroundContactTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const groundContact = runGroundContactTimeMs(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${groundContact == null ? '—' : formatGroundContactTime(groundContact)}`
    },
  }
  const verticalOscillationSpec: MapMetric = {
    label: 'vertical oscillation',
    shortLabel: 'VO',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runVerticalOscillationCm(p) ?? 0,
    fmt: value => formatVerticalOscillation(presentation, value),
    profile: graphDomain =>
      buildRunVerticalOscillationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const verticalOscillation = runVerticalOscillationCm(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${verticalOscillation == null ? '—' : formatVerticalOscillation(presentation, verticalOscillation)}`
    },
  }
  const respirationSpec: MapMetric = {
    label: 'respiration',
    shortLabel: 'R',
    ramp: RESP_RAMP,
    pick: p => p.resp ?? 0,
    fmt: formatRespirationRate,
    profile: graphDomain => buildRespirationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${p.resp == null ? '—' : formatRespirationRate(p.resp)}`,
  }
  const muscleOxygenSpec: MapMetric = {
    label: 'muscle oxygen',
    shortLabel: 'SmO₂',
    ramp: RESP_RAMP,
    pick: p => p.muscleOxygenPct ?? 0,
    fmt: formatMuscleOxygen,
    profile: graphDomain =>
      requiredMapProfile(buildMuscleOxygenTraceNode(domF, d, null, graphDomain), 'muscle oxygen'),
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${p.muscleOxygenPct == null ? '—' : formatMuscleOxygen(p.muscleOxygenPct)}`,
  }
  const riderPositionSpec: MapMetric = {
    label: 'rider position',
    shortLabel: 'RP',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => {
      const dynamics = d.cyclingDynamics
      if (!dynamics) return 0
      const position = riderPositionAtDistance(dynamics, p.d)
      return position === 'standing' ? 2 : position === 'seated' ? 1 : 0
    },
    fmt: value => triText(presentation.locale, value >= 1.5 ? 'standing' : 'seated'),
    profile: graphDomain =>
      requiredMapProfile(buildRiderPositionChartNode(domF, d, null, graphDomain), 'rider position'),
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${riderPositionAtPoint(presentation, d, p)}`,
  }
  const staminaSpec: MapMetric = {
    label: 'stamina',
    shortLabel: 'STA',
    ramp: HEAT_RAMP,
    zeroGap: true,
    pick: p => p.stamina ?? 0,
    fmt: value => `${Math.round(value)}%`,
    profile: graphDomain =>
      requiredMapProfile(buildStaminaChartNode(domF, d, null, graphDomain), 'stamina'),
    readout: p => `${scrubDist(presentation, p.d, d.sport)} · ${staminaAtPoint(presentation, p)}`,
  }
  const shiftingSpec: MapMetric = {
    label: 'electronic shifting',
    shortLabel: 'ES',
    ramp: CAD_RAMP,
    pick: p => {
      const shift = gearAtPoint(d, p)
      return shift ? shift.frontTeeth / shift.rearTeeth : 0
    },
    fmt: value => `${value.toFixed(2)}×`,
    profile: graphDomain =>
      requiredMapProfile(buildShiftingChartNode(domF, d, null, graphDomain), 'electronic shifting'),
    readout: p => {
      const shift = gearAtPoint(d, p)
      return `${zoneClock(p.elapsedS)} · ${scrubDist(presentation, p.d, d.sport)}${shift ? ` · ${shift.frontTeeth}×${shift.rearTeeth}` : ''}`
    },
  }
  const thermalProfile = (
    metric: ThermalMetric,
    graphDomain?: ActivityAnalysisRange | null,
  ): Element | null => {
    if (metric === 'temperature') return buildTemperatureTraceNode(domF, d, null, graphDomain)
    if (metric === 'heat-strain') return buildHeatStrainTraceNode(domF, d, null, graphDomain)
    if (metric === 'core-temperature')
      return buildCoreTemperatureTraceNode(domF, d, null, graphDomain)
    return buildSkinTemperatureTraceNode(domF, d, null, graphDomain)
  }
  const thermalReadout = (
    metric: ThermalMetric,
    point: StravaActivityDetail['route'][number],
  ): string => {
    const value = thermalValue(point, metric)
    if (value == null) return '—'
    if (metric === 'temperature') return formatTemperature(presentation, value)
    if (metric === 'heat-strain') return value.toFixed(1)
    return formatThermalTemperature(presentation, value)
  }
  const temperatureSpec: MapMetric = {
    label: 'temperature',
    shortLabel: 'T',
    ramp: HR_RAMP,
    pick: p => thermalValue(p, primaryThermal) ?? primaryThermalAverage,
    fmt: value => {
      if (primaryThermal === 'temperature') return formatTemperature(presentation, value)
      if (primaryThermal === 'heat-strain') return value.toFixed(1)
      return formatThermalTemperature(presentation, value)
    },
    profile: graphDomain =>
      requiredMapProfile(thermalProfile(primaryThermal, graphDomain), 'temperature'),
    readout: p => {
      const values = thermalMetrics.flatMap(metric => {
        const value = thermalValue(p, metric)
        if (value == null) return []
        const label =
          metric === 'temperature'
            ? 'temperature'
            : metric === 'heat-strain'
              ? 'heat strain index'
              : metric === 'core-temperature'
                ? 'CORE temperature'
                : 'skin temperature'
        return [`${triText(presentation.locale, label)} ${thermalReadout(metric, p)}`]
      })
      return `${scrubDist(presentation, p.d, d.sport)}${values.length > 0 ? ` · ${values.join(' · ')}` : ' · —'}`
    },
    traces: graphDomain => {
      const traces: MapTraceSurface[] = []
      for (const metric of thermalMetrics) {
        if (metric === primaryThermal) continue
        const wrap = thermalProfile(metric, graphDomain)
        if (wrap instanceof HTMLElement)
          traces.push({
            wrap,
            fmt: p => `${scrubDist(presentation, p.d, d.sport)} · ${thermalReadout(metric, p)}`,
          })
      }
      return traces
    },
  }
  const elevSpec: MapMetric = {
    label: 'elevation',
    shortLabel: 'E',
    ramp: ELEV_RAMP,
    pick: p => p.alt,
    fmt: value => formatAltitude(presentation, value),
    profile: graphDomain => buildElevation(presentation, d, graphDomain),
    readout: (p, i) => {
      const g = Math.round(gradeAt(route, i) * 10) / 10
      return `${scrubDist(presentation, p.d, d.sport)} · ${formatAltitude(presentation, p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%`
    },
  }
  const specs: MapMetric[] = []
  if (d.sport === 'bike') {
    if (hasPower) specs.push(powerSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasMuscleOxygen) specs.push(muscleOxygenSpec)
    specs.push(paceSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasRiderPosition) specs.push(riderPositionSpec)
    if (flags.stamina) specs.push(staminaSpec)
    if (hasShifting) specs.push(shiftingSpec)
    if (hasThermal) specs.push(temperatureSpec)
  } else if (d.sport === 'run') {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (hasStride) specs.push(strideSpec)
    if (hasGroundContact) specs.push(groundContactSpec)
    if (hasVerticalOscillation) specs.push(verticalOscillationSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasPower) specs.push(powerSpec)
    if (hasThermal) specs.push(temperatureSpec)
  } else {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasThermal) specs.push(temperatureSpec)
  }
  return specs
}

export interface MapDetailOpts {
  mapMode?: boolean
  initialMetric?: number
  onMetric?: (i: number) => void
  onHover?: (p: StravaActivityDetail['route'][number], i: number) => void
  analysis?: HTMLElement | null
  onRange?: ActivityRangeChange
  detailContext?: DetailCtx
}

export interface ActivityView {
  element: HTMLElement
  mount: () => () => void
}

export const renderMapDetail = (
  presentation: TriathlonPresentation,
  source: StravaActivityDetail,
  opts?: MapDetailOpts,
): ActivityView => {
  const domF = createDomFactory(presentation)
  const d = powerViewActivity(presentation, source)
  const wrap = el('section', 'tri-act tri-act--expanded')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildIcon(presentation, d.sport))
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const sbody = document.createElement('tbody')
  const summaryTrainingEffectGroup = dominantTrainingEffectGroup(activityTrainingEffectLabel(d))
  for (const [label, value] of activityStatRows(presentation, d))
    sbody.appendChild(
      statRow(
        presentation,
        label,
        value,
        label === 'training effect'
          ? { 'data-training-effect-group': summaryTrainingEffectGroup }
          : undefined,
      ),
    )
  stats.appendChild(sbody)
  wrap.appendChild(stats)

  const specs =
    d.route.length >= 2
      ? metricSpecs(presentation, d, opts?.detailContext ?? detailContextFromPayload())
      : []
  if (specs.length === 0) {
    const figs = el('div', 'tri-act-figs')
    if (d.sport === 'swim') figs.appendChild(buildPool(presentation, d))
    if (figs.childElementCount > 0) wrap.appendChild(figs)
    const more = el('div', 'tri-act-more')
    const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
    const heartRate = hasHeartRateTrace(d) ? buildHeartRateTrace(presentation, d) : null
    const trainingEffect = buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null
    for (const z of [
      heartRate,
      trainingEffect,
      zoneDuo(
        presentation,
        buildHrZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
        buildPowerZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
      ),
      zoneDuo(
        presentation,
        buildPowerCurve(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
        buildPowerHist(presentation, d),
      ),
    ])
      if (z) more.appendChild(z)
    if (bestEfforts) more.appendChild(bestEfforts)
    if (more.childElementCount > 0) wrap.appendChild(more)
    return { element: wrap, mount: () => () => {} }
  }

  const tablist = el('div', 'tri-map-tablist')
  tablist.setAttribute('role', 'tablist')
  const figs = el('div', 'tri-act-figs tri-map-figs')
  const profileBox = el('div', 'tri-map-profile')
  const runSplits = buildRunLapSplitsNode(domF, d) as HTMLElement | null
  const zoneBox = el('div', 'tri-act-more')
  const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
  wrap.append(tablist, figs)
  if (runSplits) wrap.appendChild(runSplits)
  wrap.appendChild(profileBox)
  wrap.appendChild(zoneBox)

  let active = Math.min(specs.length - 1, Math.max(0, opts?.initialMetric ?? 0))
  const routeSamples = d.route
  let graphDomain: ActivityAnalysisRange | null = null
  let routeMarker: SVGElement | null = null
  let analysisController: ActivityAnalysisController | null = null
  let linkedSurfaces: ScrubSurface[] = []
  let mounted = false
  const sameGraphDomain = (
    left: ActivityAnalysisRange | null,
    right: ActivityAnalysisRange | null,
  ): boolean =>
    left === right ||
    (left != null &&
      right != null &&
      left.startDistanceKm === right.startDistanceKm &&
      left.endDistanceKm === right.endDistanceKm)
  const renderProfile = (): void => {
    analysisController?.dispose()
    analysisController = null
    const existingCyclingChartMode = cyclingChartMode(wrap)
    const spec = specs[active]
    const profile = spec.profile(graphDomain)
    const traces = spec.traces?.(graphDomain) ?? []
    profileBox.replaceChildren(profile)
    zoneBox.replaceChildren(...traces.map(trace => trace.wrap))
    if (spec.extra) for (const node of spec.extra()) if (node) zoneBox.appendChild(node)
    if (bestEfforts) zoneBox.appendChild(bestEfforts)
    const cyclingChart = zoneBox.querySelector<HTMLElement>('.tri-cycling-mode-chart')
    if (cyclingChart) setCyclingChartMode(cyclingChart, existingCyclingChartMode)
    linkedSurfaces = [{ wrap: profile, fmt: spec.readout }, ...traces].map(surface => ({
      wrap: surface.wrap,
      samples: routeSamples,
      fmt: index => {
        const point = d.route[index]
        opts?.onHover?.(point, index)
        return surface.fmt(point, index)
      },
    }))
    if (mounted) mountProfile()
  }
  const mountProfile = (): void => {
    analysisController = linkScrub(
      presentation,
      wrap,
      routeMarker,
      linkedSurfaces,
      d.route,
      d,
      opts?.analysis,
      onRange,
    )
  }
  const onRange: ActivityRangeChange = (range, committed) => {
    if (committed && !sameGraphDomain(graphDomain, range)) {
      graphDomain = range
      renderProfile()
    }
    opts?.onRange?.(range, committed)
  }
  const draw = (animateTabs = true, notify = true) => {
    const spec = specs[active]
    const vals = d.route.map((p, i) => spec.pick(p, i))
    const pool = spec.zeroGap ? vals.filter(v => v > 0) : vals
    let lo = Infinity
    let hi = -Infinity
    for (const v of pool.length ? pool : vals) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    routeMarker = null
    if (opts?.mapMode) {
      figs.replaceChildren(buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
    } else {
      const heat = buildHeatRoute(d.route, spec.pick, spec.ramp, spec.zeroGap)
      figs.replaceChildren(heat, buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
      routeMarker = heat.querySelector<SVGElement>('.tri-route-cursor')
    }
    renderProfile()
    const tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab'))
    if (mounted && !animateTabs) tablist.dataset.motion = 'instant'
    tabs.forEach((tab, i) => {
      const on = i === active
      tab.setAttribute('aria-selected', on ? 'true' : 'false')
      tab.tabIndex = on ? 0 : -1
      tab.style.background = on ? spec.ramp[6] : ''
      tab.style.borderColor = on ? spec.ramp[6] : ''
    })
    if (mounted && !animateTabs) {
      tablist.getBoundingClientRect()
      delete tablist.dataset.motion
    }
    if (notify) opts?.onMetric?.(active)
  }
  specs.forEach((spec, i) => {
    const tab = el('button', 'tri-map-tab')
    tab.setAttribute('type', 'button')
    tab.setAttribute('role', 'tab')
    tab.setAttribute('aria-label', spec.label)
    tab.dataset.shortcut = spec.shortLabel[0].toLowerCase()
    tab.dataset.index = String(i)
    tab.style.setProperty('--tri-map-tab-shortcut-width', `${spec.shortLabel.length}ch`)
    tab.style.setProperty('--tri-map-tab-label-width', `${spec.label.length}ch`)
    const shortcut = el('span', 'tri-map-tab-shortcut', spec.shortLabel)
    const label = el('span', 'tri-map-tab-label', spec.label)
    shortcut.setAttribute('aria-hidden', 'true')
    label.setAttribute('aria-hidden', 'true')
    tab.append(shortcut, label)
    tablist.appendChild(tab)
  })
  const onTabClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const tab = target.closest<HTMLButtonElement>('.tri-map-tab[data-index]')
    if (!tab || !tablist.contains(tab)) return
    const index = Number(tab.dataset.index)
    if (!Number.isInteger(index) || index < 0 || index >= specs.length) return
    active = index
    draw(event.detail > 0)
  }
  const onTabKeydown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab'))
    let next = -1
    if (event.key === 'ArrowLeft') next = (active - 1 + tabs.length) % tabs.length
    else if (event.key === 'ArrowRight') next = (active + 1) % tabs.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = tabs.length - 1
    else
      next = nextMapMetricShortcutIndex(
        tabs.map(tab => tab.dataset.shortcut),
        active,
        event.key,
      )
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    if (next !== active) {
      active = next
      draw(false)
    }
    tabs[next]?.focus()
  }
  draw(false, false)
  return {
    element: wrap,
    mount: () => {
      mounted = true
      mountProfile()
      tablist.addEventListener('click', onTabClick)
      tablist.addEventListener('keydown', onTabKeydown)
      return () => {
        mounted = false
        tablist.removeEventListener('click', onTabClick)
        tablist.removeEventListener('keydown', onTabKeydown)
        analysisController?.dispose()
        analysisController = null
      }
    },
  }
}

export const setupStrengthExerciseOverflow = (root: ParentNode): (() => void) => {
  const cleanups = Array.from(root.querySelectorAll<HTMLElement>('.tri-act-strength'), strength => {
    const exercises = strength.querySelector<HTMLElement>('.tri-strength-exercises')
    if (!exercises) return () => {}
    let frame = 0
    const update = () => {
      const maxScroll = Math.max(0, exercises.scrollHeight - exercises.clientHeight)
      const scrollable = maxScroll > 1
      strength.dataset.scrollable = String(scrollable)
      strength.dataset.scrollEnd = String(!scrollable || exercises.scrollTop >= maxScroll - 1)
    }
    const schedule = () => {
      if (frame !== 0) return
      frame = window.requestAnimationFrame(() => {
        frame = 0
        update()
      })
    }
    const resize = new ResizeObserver(schedule)
    exercises.addEventListener('scroll', schedule, { passive: true })
    resize.observe(exercises)
    for (const exercise of exercises.children) resize.observe(exercise)
    schedule()
    return () => {
      exercises.removeEventListener('scroll', schedule)
      resize.disconnect()
      if (frame !== 0) window.cancelAnimationFrame(frame)
      strength.removeAttribute('data-scrollable')
      strength.removeAttribute('data-scroll-end')
    }
  })
  return () => {
    for (const cleanup of cleanups) cleanup()
  }
}

export const renderDetail = (
  presentation: TriathlonPresentation,
  source: StravaActivityDetail,
  payload?: DetailPayload | null,
  fillMissingRunPower = false,
  embedded = false,
  dayRouteHref?: string,
  traceSettings?: TriathlonTraceSettings,
  reserveFueling = false,
): ActivityView => {
  const domF = createDomFactory(presentation)
  const d = powerViewActivity(presentation, source)
  const normalizeBikeMetrics = d.sport === 'bike' && presentation.powerSamples === 'exclude-zero'
  const powerValues = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.w)
    : null
  const detailContext = detailContextFromPayload(payload)
  const wrap = buildActivityNode(
    domF,
    d,
    false,
    detailContext,
    fillMissingRunPower,
    embedded,
    traceSettings,
    reserveFueling,
  ) as HTMLElement
  if (d.sport === 'run') {
    const matchedGroup = payload?.matchedRuns?.groups.find(group =>
      group.efforts.some(effort => effort.id === d.id),
    )
    const more = wrap.querySelector<HTMLElement>(':scope > .tri-act-more')
    if (matchedGroup && more && triathlonTraceEnabled(traceSettings, 'matched-runs'))
      more.appendChild(buildMatchedRunGroup(presentation, matchedGroup, d.id, dayRouteHref))
  }
  if (d.sport === 'bike') {
    const matchedGroup = payload?.matchedRides?.groups.find(group =>
      group.efforts.some(effort => effort.id === d.id),
    )
    const more = wrap.querySelector<HTMLElement>(':scope > .tri-act-more')
    if (matchedGroup && more && triathlonTraceEnabled(traceSettings, 'matched-rides'))
      more.insertBefore(
        buildMatchedRideGroup(presentation, matchedGroup, d.id, dayRouteHref),
        more.querySelector(':scope > .tri-efforts'),
      )
  }
  const surfaces: ScrubSurface[] = []
  const routeSamples = d.route
  const heartRatePoints = activityHeartRateTracePoints(d)
  const thermalPoints = activityThermalTracePoints(d)
  const tracePosition = (point: { d: number; elapsedS: number }): string =>
    activityTraceUsesElapsedAxis(d)
      ? zoneClock(point.elapsedS)
      : scrubDist(presentation, point.d, d.sport)
  for (const trace of wrap.querySelectorAll<HTMLElement>('[data-tri-trace]')) {
    const name = trace.dataset.triTrace
    if (!name || !triathlonTraceEnabled(traceSettings, name)) trace.remove()
  }
  const elev = wrap.querySelector<HTMLElement>(
    '.tri-act-figs .tri-elev-wrap:not(.tri-elev-wrap--unavailable)',
  )
  if (elev) {
    surfaces.push({
      wrap: elev,
      samples: routeSamples,
      fmt: i => {
        const p = d.route[i]
        const g = Math.round(gradeAt(d.route, i) * 10) / 10
        return (
          `${scrubDist(presentation, p.d, d.sport)} · ${formatAltitude(presentation, p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%` +
          (p.hr > 0 ? ` · ${p.hr} bpm` : '')
        )
      },
    })
  }
  const cadenceScale = d.sport === 'run' ? 2 : 1
  const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const cadenceValues = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.cad * cadenceScale)
    : null
  for (const trace of wrap.querySelectorAll<HTMLElement>('[data-tri-trace]')) {
    if (trace.dataset.triTrace === 'hr')
      surfaces.push({
        wrap: trace,
        samples: heartRatePoints,
        fmt: i => {
          const point = heartRatePoints[i]
          return `${tracePosition(point)} · ${point.heartRate == null ? '—' : `${Math.round(point.heartRate)} bpm`}`
        },
      })
    else if (trace.dataset.triTrace === 'power')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(powerValues?.[i] ?? p.w)} W`
        },
      })
    else if (trace.dataset.triTrace === 'power-balance')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${p.rightPowerPct == null || p.w <= 0 ? '—' : powerBalanceText(p.rightPowerPct)}`
        },
      })
    else if (trace.dataset.triTrace === 'torque-effectiveness' && d.cyclingDynamics)
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftTorqueEffectiveness ?? [], d.cyclingDynamics?.rightTorqueEffectiveness ?? [], '%')}`
        },
      })
    else if (trace.dataset.triTrace === 'pedal-smoothness' && d.cyclingDynamics)
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${cyclingDynamicsPairAtDistance(d, p, d.cyclingDynamics?.leftPedalSmoothness ?? [], d.cyclingDynamics?.rightPedalSmoothness ?? [], '%')}`
        },
      })
    else if (trace.dataset.triTrace === 'power-phase' && d.cyclingDynamics)
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${powerPhaseAtDistance(d, p)}`
        },
      })
    else if (trace.dataset.triTrace === 'rider-position' && d.cyclingDynamics)
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${riderPositionAtPoint(presentation, d, p)}`
        },
      })
    else if (trace.dataset.triTrace === 'electronic-shifting')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const shift = gearAtPoint(d, p)
          return `${zoneClock(p.elapsedS)} · ${scrubDist(presentation, p.d, d.sport)}${shift ? ` · ${shift.frontTeeth}×${shift.rearTeeth}` : ''}`
        },
      })
    else if (trace.dataset.triTrace === 'stamina')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${staminaAtPoint(presentation, p)}`
        },
      })
    else if (trace.dataset.triTrace === 'cadence')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(cadenceValues?.[i] ?? p.cad * cadenceScale)} ${cadenceUnit}`
        },
      })
    else if (
      trace.dataset.triTrace === 'stride-length' ||
      trace.dataset.triTrace === 'estimated-stride-length'
    )
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const stride = runStrideLengthValue(d, p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
        },
      })
    else if (trace.dataset.triTrace === 'ground-contact-time')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const groundContact = runGroundContactTimeMs(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${groundContact == null ? '—' : formatGroundContactTime(groundContact)}`
        },
      })
    else if (trace.dataset.triTrace === 'vertical-oscillation')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const verticalOscillation = runVerticalOscillationCm(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${verticalOscillation == null ? '—' : formatVerticalOscillation(presentation, verticalOscillation)}`
        },
      })
    else if (trace.dataset.triTrace === 'respiration')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${p.resp == null ? '—' : formatRespirationRate(p.resp)}`
        },
      })
    else if (trace.dataset.triTrace === 'muscle-oxygen')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${p.muscleOxygenPct == null ? '—' : formatMuscleOxygen(p.muscleOxygenPct)}`
        },
      })
    else if (trace.dataset.triTrace === 'temperature')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          return `${scrubDist(presentation, p.d, d.sport)} · ${p.tempC == null ? '—' : formatTemperature(presentation, p.tempC)}`
        },
      })
    else if (trace.dataset.triTrace === 'heat-strain-index')
      surfaces.push({
        wrap: trace,
        samples: thermalPoints,
        fmt: i => {
          const p = thermalPoints[i]
          return `${tracePosition(p)} · ${p.heatStrainIndex == null ? '—' : p.heatStrainIndex.toFixed(1)}`
        },
      })
    else if (trace.dataset.triTrace === 'core-temperature')
      surfaces.push({
        wrap: trace,
        samples: thermalPoints,
        fmt: i => {
          const p = thermalPoints[i]
          return `${tracePosition(p)} · ${p.coreTemperatureC == null ? '—' : formatThermalTemperature(presentation, p.coreTemperatureC)}`
        },
      })
    else if (trace.dataset.triTrace === 'skin-temperature')
      surfaces.push({
        wrap: trace,
        samples: thermalPoints,
        fmt: i => {
          const p = thermalPoints[i]
          return `${tracePosition(p)} · ${p.skinTemperatureC == null ? '—' : formatThermalTemperature(presentation, p.skinTemperatureC)}`
        },
      })
  }
  const interactive =
    surfaces.some(surface => surface.samples.length >= 2) ||
    (wrap.querySelector('[data-tri-analysis]') != null && d.route.length >= 2)
  return {
    element: wrap,
    mount: () => {
      const cleanupExerciseOverflow = setupStrengthExerciseOverflow(wrap)
      if (!interactive) return cleanupExerciseOverflow
      const routeMarker = wrap.querySelector<SVGElement>('.tri-route-cursor')
      const controller = linkScrub(presentation, wrap, routeMarker, surfaces, d.route, d)
      return () => {
        cleanupExerciseOverflow()
        controller?.dispose()
      }
    },
  }
}
