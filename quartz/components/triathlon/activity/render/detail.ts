import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import {
  activityCadenceScale,
  activityCadenceUnit,
  activityCyclingIntensityPoints,
  activityHeartRateTracePoints,
  activityPhysiologyTracePoints,
  activityThermalTracePoints,
  activityTraceUsesElapsedAxis,
  buildActivity as buildActivityNode,
  formatAltitude,
  formatGroundContactBalance,
  formatGroundContactTime,
  formatImpactLoadFactor,
  formatMuscleOxygen,
  formatPerformanceCondition,
  formatRespirationRate,
  formatStepSpeedLoss,
  formatStepSpeedLossPercent,
  formatTemperature,
  formatThermalTemperature,
  formatVerticalOscillation,
  formatVerticalRatio,
  gradeAt,
  interpolatePositiveMetricSeries,
  performanceConditionValue,
  powerBalanceText,
  powerViewActivity,
  runGroundContactBalanceLeftPct,
  runGroundContactTimeMs,
  runImpactLoadFactor,
  runStepSpeedLossMps,
  runStepSpeedLossPct,
  runVerticalOscillationCm,
  runVerticalRatioPct,
  scrubDist,
  speedKph,
  zoneClock,
} from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import {
  triathlonTraceEnabled,
  type TriathlonTraceSettings,
} from '../../../../util/triathlon-trace-settings'
import { buildMatchedRideGroup, buildMatchedRunGroup } from '../../analytics/panels/matched'
import { createDomFactory } from '../../runtime/dom'
import { linkScrub, type ScrubSurface } from '../analysis'
import { setupCyclingPowerCharts } from '../cycling-power'
import { detailContextFromPayload, type DetailPayload } from '../data'
import { setupRunHeartCostCharts } from '../run-heart-cost'
import { runMetricMode, setRunMetricMode } from '../scrub'
import {
  cyclingDynamicsPairAtDistance,
  cyclingPowerScrubSurface,
  gearAtPoint,
  powerPhaseAtDistance,
  riderPositionAtPoint,
  runHeartCostScrubSurface,
  staminaAtPoint,
  strideScrubSurface,
  torqueScrubSurface,
  walkPowerScrubSurface,
  type ActivityView,
} from './scrub-surfaces'

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
  const staminaPoints = activityPhysiologyTracePoints(d, 'stamina')
  const performanceConditionPoints = activityPhysiologyTracePoints(d, 'performanceCondition')
  const thermalPoints = activityThermalTracePoints(d)
  const tracePosition = (point: { d: number; elapsedS: number }): string =>
    activityTraceUsesElapsedAxis(d)
      ? zoneClock(point.elapsedS)
      : scrubDist(presentation, point.d, d.sport)
  for (const trace of wrap.querySelectorAll<HTMLElement>(
    '[data-tri-trace], [data-tri-unavailable]',
  )) {
    const name = trace.dataset.triTrace ?? trace.dataset.triUnavailable
    if (!name || !triathlonTraceEnabled(traceSettings, name)) trace.remove()
  }
  for (const chart of wrap.querySelectorAll<HTMLElement>(
    '.tri-run-metric-chart[data-run-metric-group]',
  )) {
    const panes = Array.from(chart.querySelectorAll<HTMLElement>('[data-run-metric-pane]'))
    if (panes.length === 0) {
      chart.remove()
      continue
    }
    if (panes.length === 1) {
      panes[0].hidden = false
      panes[0].setAttribute('aria-hidden', 'false')
      for (const controls of panes[0].querySelectorAll('.tri-run-metric-modes')) controls.remove()
      continue
    }
    const current = runMetricMode(chart)
    const available = current ? panes.some(pane => pane.dataset.runMetricPane === current) : false
    const fallback = runMetricMode(
      chart.querySelector<HTMLButtonElement>(
        `.tri-run-metric-mode[data-run-metric-mode="${panes[0].dataset.runMetricPane ?? ''}"]`,
      ),
    )
    if (available && current) setRunMetricMode(chart, current)
    else if (fallback) setRunMetricMode(chart, fallback)
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
  const cadenceScale = activityCadenceScale(d.sport)
  const cadenceUnit = activityCadenceUnit(d.sport)
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
    else if (trace.dataset.triTrace === 'speed')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = Number.isFinite(p.speedKph) && p.speedKph >= 0 ? p.speedKph : null
          return `${tracePosition(p)} · ${value == null ? '—' : speedKph(presentation, value)}`
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
    else if (trace.dataset.triTrace === 'walking-power')
      surfaces.push(walkPowerScrubSurface(trace, d, presentation))
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
          const gears = !shift
            ? ''
            : trace.dataset.shiftLabel === 'number'
              ? ` · ${shift.frontGearNum}×${shift.rearGearNum}`
              : ` · ${shift.frontTeeth}×${shift.rearTeeth}`
          return `${zoneClock(p.elapsedS)} · ${scrubDist(presentation, p.d, d.sport)}${gears}`
        },
      })
    else if (trace.dataset.triTrace === 'stamina')
      surfaces.push({
        wrap: trace,
        samples: staminaPoints,
        fmt: i => {
          const p = staminaPoints[i]
          return `${tracePosition(p)} · ${staminaAtPoint(presentation, p)}`
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
    else if (trace.dataset.triTrace === 'crank-torque')
      surfaces.push(torqueScrubSurface(trace, d, presentation))
    else if (trace.dataset.triTrace === 'cycling-power')
      surfaces.push(cyclingPowerScrubSurface(trace, d, presentation))
    else if (trace.dataset.triTrace === 'intensity-factor') {
      const points = activityCyclingIntensityPoints(d)
      surfaces.push({
        wrap: trace,
        samples: points,
        fmt: i => {
          const point = points[i]
          return `${tracePosition(point)} · ${point.intensityFactor?.toFixed(3) ?? '—'} · ${triText(presentation.locale, 'cumulative')}`
        },
      })
    } else if (trace.dataset.triTrace === 'performance-condition')
      surfaces.push({
        wrap: trace,
        samples: performanceConditionPoints,
        fmt: i => {
          const p = performanceConditionPoints[i]
          const value = performanceConditionValue(p)
          return `${tracePosition(p)} · ${value == null ? '—' : formatPerformanceCondition(value)}`
        },
      })
    else if (
      trace.dataset.triTrace === 'stride-length' ||
      trace.dataset.triTrace === 'estimated-stride-length'
    )
      surfaces.push(strideScrubSurface(trace, d, presentation))
    else if (trace.dataset.triTrace === 'run-heart-cost')
      surfaces.push(runHeartCostScrubSurface(trace, d, presentation))
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
    else if (trace.dataset.triTrace === 'ground-contact-balance')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = runGroundContactBalanceLeftPct(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatGroundContactBalance(value)}`
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
    else if (trace.dataset.triTrace === 'vertical-ratio')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = runVerticalRatioPct(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatVerticalRatio(value)}`
        },
      })
    else if (trace.dataset.triTrace === 'step-speed-loss')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = runStepSpeedLossMps(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatStepSpeedLoss(value)}`
        },
      })
    else if (trace.dataset.triTrace === 'step-speed-loss-percent')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = runStepSpeedLossPct(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatStepSpeedLossPercent(value)}`
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
    else if (trace.dataset.triTrace === 'impact-load-factor')
      surfaces.push({
        wrap: trace,
        samples: routeSamples,
        fmt: i => {
          const p = d.route[i]
          const value = runImpactLoadFactor(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${value == null ? '—' : formatImpactLoadFactor(value)}`
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
    wrap.querySelector('[data-analysis-range]') != null
  return {
    element: wrap,
    mount: () => {
      if (!interactive) return () => {}
      const cleanupCyclingPower = setupCyclingPowerCharts(wrap, presentation, d)
      const cleanupRunHeartCost = setupRunHeartCostCharts(wrap, presentation, d)
      const routeMarker = wrap.querySelector<SVGElement>('.tri-route-cursor')
      const controller = linkScrub(presentation, wrap, routeMarker, surfaces, d.route, d)
      return () => {
        cleanupCyclingPower()
        cleanupRunHeartCost()
        controller?.dispose()
      }
    },
  }
}
