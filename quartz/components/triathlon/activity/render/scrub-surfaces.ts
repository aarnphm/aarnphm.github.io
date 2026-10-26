import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { GardenEnvironmentSample } from '../../../../util/activity-environment'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import { preferredActivityWind } from '../../../../util/activity-wind'
import {
  activityCyclingPowerPoints,
  activityCyclingTorquePoints,
  activityHeartRateTracePoints,
  activityPhysiologyTracePoints,
  activityRunHeartCostPoints,
  activityTraceUsesElapsedAxis,
  activityWalkPowerPoints,
  cyclingDynamicsIndexAtDistance,
  formatImpactLoadFactor,
  formatPerformanceCondition,
  formatStrideLength,
  gearShiftAtFraction,
  riderPositionAtDistance,
  runStrideLengthValue,
  scrubDist,
  zoneClock,
  type DetailCtx,
} from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import { rampGradient } from '../../maps/palette'
import { el, svg } from '../../runtime/dom'
import {
  activityScrubElapsedIndexAt,
  type ActivityAnalysisRange,
  type ActivityRangeChange,
  type ScrubSurface,
} from '../analysis'
import { cyclingPowerReadout, cyclingPowerWindow } from '../cycling-power'
import { runHeartCostReadout, runHeartCostWindow } from '../run-heart-cost'

export const buildHeatRoute = (
  route: StravaActivityDetail['route'],
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  colors?: string[],
  zeroGap = false,
  valid?: (p: StravaActivityDetail['route'][number], i: number) => boolean,
): SVGElement => {
  const ramp = colors?.length ?? 7
  const pad = 6
  const span = 100 - pad * 2
  const vals = route.map((p, i) => pick(p, i))
  const pool = vals.filter((v, index) =>
    valid && !valid(route[index], index) ? false : !zeroGap || v > 0,
  )
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
    if (valid && (!valid(route[i], i) || !valid(route[i + 1], i + 1))) continue
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

export const cyclingDynamicsPairAtDistance = (
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

export const powerPhaseAtDistance = (
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

export const riderPositionAtPoint = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): string => {
  const dynamics = d.cyclingDynamics
  if (!dynamics) return '—'
  const position = riderPositionAtDistance(dynamics, point.d)
  return position == null ? '—' : triText(presentation.locale, position)
}

export const gearAtPoint = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): ReturnType<typeof gearShiftAtFraction> => {
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  return gearShiftAtFraction(d.gearShifts, maxDistanceKm, point.d / maxDistanceKm)
}

export const staminaAtPoint = (
  presentation: TriathlonPresentation,
  point: { stamina: number | null; potentialStamina: number | null },
): string =>
  `${triText(presentation.locale, 'current')} ${point.stamina == null ? '—' : `${Math.round(point.stamina)}%`} · ${triText(presentation.locale, 'potential')} ${point.potentialStamina == null ? '—' : `${Math.round(point.potentialStamina)}%`}`

export const torqueScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): ScrubSurface => {
  const points = activityCyclingTorquePoints(d)
  return {
    wrap,
    samples: points,
    fmt: i => {
      const point = points[i]
      const gear = d.gearShifts.findLast(shift => shift.elapsedS <= point.elapsedS)
      const position = activityTraceUsesElapsedAxis(d)
        ? zoneClock(point.elapsedS)
        : scrubDist(presentation, point.d, d.sport)
      return `${position} · ${point.torqueNm?.toFixed(1) ?? '—'} N·m · ${point.cadenceRpm?.toFixed(0) ?? '—'} rpm · ${point.watts?.toFixed(0) ?? '—'} W${gear ? ` · ${gear.frontTeeth ?? '—'}×${gear.rearTeeth ?? '—'}` : ''}`
    },
  }
}

export const cyclingImpactLoadScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): ScrubSurface => {
  const points = activityCyclingTorquePoints(d)
  const reference = d.cyclingTorque?.reference?.torqueNm
  return {
    wrap,
    samples: points,
    fmt: i => {
      const point = points[i]
      const position = activityTraceUsesElapsedAxis(d)
        ? zoneClock(point.elapsedS)
        : scrubDist(presentation, point.d, d.sport)
      return `${position} · ${point.torqueNm == null || !reference ? '—' : formatImpactLoadFactor(point.torqueNm / reference)} · ${point.torqueNm?.toFixed(1) ?? '—'} N·m`
    },
  }
}

export const cyclingPowerScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
  onHover?: MapDetailOpts['onHover'],
): ScrubSurface => {
  const points = activityCyclingPowerPoints(d)
  const distanceAvailable = d.distanceKm > 0 || points.some(point => point.distanceKm > 0)
  const wind = preferredActivityWind(d).samples
  return {
    wrap,
    samples: points,
    onHover: index => {
      const routeIndex = activityScrubElapsedIndexAt(d.route, points[index].elapsedS)
      if (routeIndex >= 0) onHover?.(d.route[routeIndex], routeIndex)
    },
    fmt: index =>
      cyclingPowerReadout(
        presentation,
        points[index],
        cyclingPowerWindow(wrap),
        wind,
        distanceAvailable,
      ),
  }
}

export const runHeartCostScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
  onHover?: MapDetailOpts['onHover'],
): ScrubSurface => {
  const points = activityRunHeartCostPoints(d)
  const wind = preferredActivityWind(d).samples
  return {
    wrap,
    samples: points,
    onHover: index => {
      const routeIndex = activityScrubElapsedIndexAt(d.route, points[index].elapsedS)
      if (routeIndex >= 0) onHover?.(d.route[routeIndex], routeIndex)
    },
    fmt: index =>
      runHeartCostReadout(presentation, d, points[index], runHeartCostWindow(wrap), wind),
  }
}

export const RUN_STRIDE_TRACE_SELECTOR =
  '[data-tri-trace="stride-length"], [data-tri-trace="estimated-stride-length"]'

export const strideScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): ScrubSurface => ({
  wrap,
  samples: d.route,
  fmt: i => {
    const p = d.route[i]
    const stride = runStrideLengthValue(d, p)
    return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
  },
})

export const walkPowerScrubSurface = (
  wrap: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): ScrubSurface => {
  const points = activityWalkPowerPoints(d)
  return {
    wrap,
    samples: points,
    fmt: index => {
      const point = points[index]
      const position = activityTraceUsesElapsedAxis(d)
        ? zoneClock(point.elapsedS)
        : scrubDist(presentation, point.d, d.sport)
      return `${position} · ${point.watts == null ? '—' : `${Math.round(point.watts)} W`} · ${triText(presentation.locale, 'estimated')}`
    },
  }
}

export const physiologyScrubSurfaces = (
  root: HTMLElement,
  d: StravaActivityDetail,
  presentation: TriathlonPresentation,
): ScrubSurface[] => {
  const position = (point: { d: number; elapsedS: number }) =>
    activityTraceUsesElapsedAxis(d)
      ? zoneClock(point.elapsedS)
      : scrubDist(presentation, point.d, d.sport)
  const heartRate = activityHeartRateTracePoints(d)
  const stamina = activityPhysiologyTracePoints(d, 'stamina')
  const condition = activityPhysiologyTracePoints(d, 'performanceCondition')
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      '[data-tri-trace="hr"], [data-tri-trace="stamina"], [data-tri-trace="performance-condition"]',
    ),
  ).map(wrap => {
    if (wrap.dataset.triTrace === 'hr')
      return {
        wrap,
        samples: heartRate,
        fmt: (index: number) =>
          `${position(heartRate[index])} · ${heartRate[index].heartRate ?? '—'} bpm`,
      }
    if (wrap.dataset.triTrace === 'stamina')
      return {
        wrap,
        samples: stamina,
        fmt: (index: number) =>
          `${position(stamina[index])} · ${staminaAtPoint(presentation, stamina[index])}`,
      }
    return {
      wrap,
      samples: condition,
      fmt: (index: number) =>
        `${position(condition[index])} · ${condition[index].performanceCondition == null ? '—' : formatPerformanceCondition(condition[index].performanceCondition)}`,
    }
  })
}

export const requiredMapProfile = (profile: Element | null, label: string): HTMLElement => {
  if (!(profile instanceof HTMLElement)) throw new Error(`Missing ${label} map profile`)
  return profile
}

export const environmentValueAtElapsed = (
  samples: readonly GardenEnvironmentSample[],
  elapsedS: number,
  pick: (sample: GardenEnvironmentSample) => number | null,
  stepped = false,
): number | null => {
  const length = samples.length
  if (length === 0 || elapsedS < samples[0].elapsedS) return null
  let low = 0
  let high = length
  while (low < high) {
    const mid = (low + high) >>> 1
    if (samples[mid].elapsedS < elapsedS) low = mid + 1
    else high = mid
  }
  if (low < length && samples[low].elapsedS === elapsedS) return pick(samples[low])
  const left = samples[low - 1]
  const right = samples[low]
  if (!left) return null
  const leftValue = pick(left)
  if (!right) return leftValue
  const rightValue = pick(right)
  if (leftValue == null || rightValue == null) return null
  if (stepped || right.elapsedS <= left.elapsedS) return leftValue
  const fraction = (elapsedS - left.elapsedS) / (right.elapsedS - left.elapsedS)
  return leftValue + (rightValue - leftValue) * fraction
}

export type MapTraceSurface = {
  wrap: HTMLElement
  fmt: (point: StravaActivityDetail['route'][number], index: number) => string
}

export interface MapMetric {
  label: string
  shortLabel: string
  ramp: string[]
  zeroGap?: boolean
  valid?: (p: StravaActivityDetail['route'][number], i: number) => boolean
  pick: (p: StravaActivityDetail['route'][number], i: number) => number
  fmt: (v: number) => string
  profile: (graphDomain?: ActivityAnalysisRange | null) => HTMLElement
  readout: (p: StravaActivityDetail['route'][number], i: number) => string
  traces?: (graphDomain?: ActivityAnalysisRange | null) => MapTraceSurface[]
  extra?: () => (HTMLElement | null)[]
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
