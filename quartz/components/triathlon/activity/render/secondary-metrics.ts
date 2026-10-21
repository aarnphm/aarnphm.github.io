import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import type { ActivityAnalysisRange } from '../analysis'
import type { MetricSpecState, ThermalMetric } from './metric-state'
import {
  buildCoreTemperatureTrace as buildCoreTemperatureTraceNode,
  buildHeatStrainTrace as buildHeatStrainTraceNode,
  buildMuscleOxygenTrace as buildMuscleOxygenTraceNode,
  buildRiderPositionChart as buildRiderPositionChartNode,
  buildShiftingChart as buildShiftingChartNode,
  buildSkinTemperatureTrace as buildSkinTemperatureTraceNode,
  buildStaminaChart as buildStaminaChartNode,
  buildTemperatureTrace as buildTemperatureTraceNode,
  formatAltitude,
  formatMuscleOxygen,
  formatTemperature,
  formatThermalTemperature,
  gradeAt,
  riderPositionAtDistance,
  scrubDist,
  speedKph,
  zoneClock,
} from '../../../../util/triathlon-card'
import { triText } from '../../../../util/triathlon-i18n'
import {
  CAD_RAMP,
  ELEV_RAMP,
  HEAT_RAMP,
  HR_RAMP,
  RESP_RAMP,
  STRIDE_RAMP,
  UV_RAMP,
  WIND_RAMP,
} from '../../maps/palette'
import { buildElevation, buildTrace } from '../primitives'
import {
  gearAtPoint,
  requiredMapProfile,
  riderPositionAtPoint,
  staminaAtPoint,
  type MapMetric,
  type MapTraceSurface,
} from './scrub-surfaces'

export const secondaryMapMetrics = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  state: MetricSpecState,
) => {
  const {
    domF,
    route,
    staminaAt,
    uvValues,
    headwindValues,
    crosswindValues,
    apparentAirValues,
    yawValues,
    thermalMetrics,
    primaryThermal,
    thermalValue,
    primaryThermalAverage,
  } = state
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
    valid: p => staminaAt(p).stamina != null,
    pick: p => staminaAt(p).stamina ?? 0,
    fmt: value => `${Math.round(value)}%`,
    profile: graphDomain =>
      requiredMapProfile(buildStaminaChartNode(domF, d, null, graphDomain), 'stamina'),
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${staminaAtPoint(presentation, staminaAt(p))}`,
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
  const uvMaximum = Math.max(11, ...uvValues.flatMap(value => (value == null ? [] : [value])))
  const uvSpec: MapMetric = {
    label: 'UV',
    shortLabel: 'UV',
    ramp: UV_RAMP,
    valid: (_point, index) => uvValues[index] != null,
    pick: (_point, index) => uvValues[index] ?? 0,
    fmt: value => `UVI ${value.toFixed(1)}`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (_point, index) => uvValues[index],
        'UV index',
        maximum => `UVI ${maximum.toFixed(1)} peak`,
        value => value.toFixed(1),
        graphDomain,
        { min: 0, max: uvMaximum, intervals: 4 },
      ),
    readout: (point, index) =>
      `${scrubDist(presentation, point.d, d.sport)} · ${uvValues[index] == null ? '—' : `UVI ${uvValues[index]?.toFixed(1)}`}`,
  }
  const windMagnitude = Math.max(
    1,
    ...headwindValues.flatMap(value => (value == null ? [] : [Math.abs(value)])),
  )
  const windSpec: MapMetric = {
    label: 'wind',
    shortLabel: 'W',
    ramp: WIND_RAMP,
    valid: (_point, index) => headwindValues[index] != null,
    pick: (_point, index) => headwindValues[index] ?? 0,
    fmt: value => `${value > 0 ? '+' : ''}${speedKph(presentation, value)}`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (_point, index) => headwindValues[index],
        'headwind',
        () =>
          `${d.analyses.derived.apparentWind?.summary.coveragePct.toFixed(1) ?? '0.0'}% coverage`,
        value => `${value > 0 ? '+' : ''}${speedKph(presentation, value)}`,
        graphDomain,
        { min: -windMagnitude, max: windMagnitude, intervals: 4 },
      ),
    readout: (point, index) => {
      const headwind = headwindValues[index]
      const crosswind = crosswindValues[index]
      const apparent = apparentAirValues[index]
      const yaw = yawValues[index]
      const metrics = [
        headwind == null
          ? 'headwind —'
          : `headwind ${headwind > 0 ? '+' : ''}${speedKph(presentation, headwind)}`,
        crosswind == null
          ? 'crosswind —'
          : `crosswind ${crosswind > 0 ? '+' : ''}${speedKph(presentation, crosswind)}`,
        apparent == null ? 'apparent air —' : `apparent air ${speedKph(presentation, apparent)}`,
        yaw == null ? 'yaw —' : `yaw ${yaw > 0 ? '+' : ''}${yaw.toFixed(1)}°`,
      ]
      return `${scrubDist(presentation, point.d, d.sport)} · ${metrics.join(' · ')}`
    },
  }
  return {
    muscleOxygenSpec,
    riderPositionSpec,
    staminaSpec,
    shiftingSpec,
    temperatureSpec,
    elevSpec,
    uvSpec,
    windSpec,
  }
}
