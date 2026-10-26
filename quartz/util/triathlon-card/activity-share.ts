import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { DetailCtx, TriNodeFactory } from './construction-contracts'
import { EXTERNAL_LINK_ICON_PATH } from '../external-link-icon'
import {
  activityCardRoute,
  activityCardStatKey,
  type ActivityCardChart,
  type ActivityCardData,
  type ActivityCardStat,
  type PowerWindow,
} from '../press-kit-activity'
import { triathlonActivityHref } from '../triathlon-date-route'
import { triText } from '../triathlon-i18n'
import { activityGpsSegments } from '../triathlon-map-route'
import { activityAnalysisStatAttrs, relativeHumidityStatAttrs } from './activity-stat-details'
import { activityTableRows } from './activity-stat-rows'
import { analysisRangeMetrics } from './analysis-presentation'
import {
  cyclingDynamicsAverage,
  cyclingDynamicsValues,
  validCyclingDynamicsValue,
  type CyclingDynamicsSeriesKey,
} from './cycling-dynamics-data'
import { clock, dur, formatAltitude, isImperial, zoneClock, zoneRange } from './metric-text'
import { normalizePowerCurvePoints, powerCurvePathPoints } from './power-curve-data'
import {
  cyclingWorkoutLaps,
  runLapSplits,
  runPaceZoneRangeText,
  workoutLapTimeline,
} from './workout-lap-data'

// Zone bounds live on the payload, not on each activity, so callers pass them in.
export type ActivityShareZones = Pick<DetailCtx, 'zones' | 'vt1'>

// Garden computes time in zone from the HR stream against the payload bounds, so the card
// marks it as calculated. The chart needs one bound fewer than zones, strictly increasing.
const runHeartRateChart = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  zones: ActivityShareZones | null | undefined,
): ActivityCardChart | null => {
  const seconds = d.hrZones
  const bounds = zones?.zones?.hr
  if (
    d.sport !== 'run' ||
    !seconds ||
    !bounds?.length ||
    seconds.length !== bounds.length + 1 ||
    seconds.some(value => !Number.isFinite(value) || value < 0) ||
    !seconds.some(value => value > 0) ||
    bounds.some(
      (value, index) =>
        !Number.isFinite(value) || value <= 0 || (index > 0 && value <= bounds[index - 1]),
    )
  )
    return null
  const calculated = triText(presentation.locale, 'calculated')
  return {
    key: 'runHeartRate',
    kind: 'bars',
    title: triText(presentation.locale, 'heart rate distribution'),
    axis: 'min',
    note: zones?.vt1 != null ? `vt1 ${zones.vt1} bpm · ${calculated}` : calculated,
    points: seconds.map((value, index) => ({
      x: index,
      y: value / 60,
      label: `Z${index + 1}`,
      range: `${zoneRange(bounds, index)} bpm`,
      value: zoneClock(value),
    })),
  }
}

export const activityShareData = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  fillMissingRunPower = false,
  zones?: ActivityShareZones | null,
): ActivityCardData | null => {
  const url = triathlonActivityHref(d.date, d.id)
  if (!url) return null
  const stats: ActivityCardStat[] = []
  const rows = activityTableRows(presentation, d, fillMissingRunPower)
  rows.push(['elevation gain', formatAltitude(presentation, d.elevationM)])
  if (d.sport === 'bike') {
    const add = (key: string, label: string, value: string, provenance?: string): void => {
      stats.push({
        key: `metric:${key}`,
        label: `${triText(presentation.locale, label)}${provenance?.startsWith('Garden calculation') ? ` · ${triText(presentation.locale, 'calculated')}` : ''}`,
        value,
        provenance,
      })
    }
    const torque = d.cyclingTorque?.summary
    if (torque) {
      add(
        'crank torque',
        'crank torque',
        `${torque.averageTorqueNm.toFixed(1)} N·m`,
        'Garden calculation from Wahoo power and cadence',
      )
      add(
        'pedalling cadence',
        'cadence',
        `${torque.averageCadenceRpm.toFixed(0)} rpm`,
        'Wahoo · observed pedalling samples',
      )
    }
    const dynamics = d.cyclingDynamics
    if (dynamics) {
      for (const [key, side, metric, series] of [
        ['left pedal smoothness', 'L', 'pedal smoothness', 'leftPedalSmoothness'],
        ['right pedal smoothness', 'R', 'pedal smoothness', 'rightPedalSmoothness'],
        ['left torque effectiveness', 'L', 'torque effectiveness', 'leftTorqueEffectiveness'],
        ['right torque effectiveness', 'R', 'torque effectiveness', 'rightTorqueEffectiveness'],
      ] satisfies [string, string, string, CyclingDynamicsSeriesKey][]) {
        const average = cyclingDynamicsAverage(cyclingDynamicsValues(dynamics, series))
        if (average != null)
          add(
            key,
            `${triText(presentation.locale, metric)} · ${side}`,
            `${average.toFixed(1)}%`,
            'Recorded cycling dynamics · average of valid samples',
          )
      }
      for (const [side, boundary, series] of [
        ['L', 'start', 'leftPowerPhaseStart'],
        ['L', 'end', 'leftPowerPhaseEnd'],
        ['R', 'start', 'rightPowerPhaseStart'],
        ['R', 'end', 'rightPowerPhaseEnd'],
      ] satisfies [string, string, CyclingDynamicsSeriesKey][]) {
        const values = cyclingDynamicsValues(dynamics, series).filter(validCyclingDynamicsValue)
        if (!values.length) continue
        const range = values.reduce(
          (bounds, value) => ({
            min: Math.min(bounds.min, value),
            max: Math.max(bounds.max, value),
          }),
          { min: Infinity, max: -Infinity },
        )
        add(
          `${side.toLowerCase()} power phase ${boundary}`,
          `${triText(presentation.locale, 'power phase')} · ${side} · ${triText(presentation.locale, boundary)}`,
          `${range.min.toFixed(0)}–${range.max.toFixed(0)}°`,
          'Recorded cycling dynamics · observed angular range',
        )
      }
      if (dynamics.seatedTimeS != null) add('seated time', 'seated time', dur(dynamics.seatedTimeS))
      if (dynamics.standingTimeS != null)
        add('standing time', 'standing time', dur(dynamics.standingTimeS))
    }
  }
  for (const [label, value] of rows) {
    const key =
      d.sport === 'sauna' && label === 'time'
        ? 'metric:start-time'
        : (activityCardStatKey(label) ?? `metric:${label}`)
    if (!value.trim() || value === '—' || stats.some(stat => stat.key === key)) continue
    if (key === 'elevation' && !['bike', 'run', 'walk', 'hike'].includes(d.sport)) continue
    const provenance =
      activityAnalysisStatAttrs(d, label)?.title ??
      (label === 'humidity' ? relativeHumidityStatAttrs(d)?.title : undefined) ??
      (label === 'TSS' &&
      d.wahoo?.metrics.trainingStressScore == null &&
      d.garmin?.trainingStressScore == null
        ? 'Garden estimate'
        : undefined) ??
      (label === 'intensity factor' &&
      d.wahoo?.metrics.intensityFactor == null &&
      d.garmin?.intensityFactor == null &&
      d.calculatedIntensityFactor
        ? 'Garden estimate'
        : undefined) ??
      (label === 'exercise load' && d.garmin?.exerciseLoad == null && d.calculatedExerciseLoad
        ? 'Garden estimate'
        : undefined) ??
      (label === 'est power' || label === 'walking power' ? 'estimated' : undefined)
    const estimated = provenance && /Garden|modeled/i.test(provenance)
    stats.push({
      key,
      label: `${triText(presentation.locale, label)}${estimated ? ` · ${triText(presentation.locale, 'estimate')}` : ''}`,
      value,
      provenance,
    })
  }
  const charts: ActivityCardChart[] = []
  const power = d.cyclingPowerTrace
  if (d.sport === 'bike' && power && power.points.some(point => point.power30sWatts != null)) {
    const series = ([30, 300] satisfies PowerWindow[])
      .map(seconds => ({
        seconds,
        points: power.points.map(point => {
          const watts = seconds === 30 ? point.power30sWatts : point.power5mWatts
          return {
            x: point.elapsedS,
            y: watts,
            label: zoneClock(point.elapsedS),
            value: watts == null ? '—' : `${Math.round(watts)} W`,
          }
        }),
      }))
      .filter(series => series.points.some(point => point.y != null))
    charts.push({
      key: 'cyclingPower',
      kind: 'line',
      title: triText(presentation.locale, 'power'),
      axis: 'W',
      note: power.source,
      window: 30,
      points: series[0].points,
      series,
    })
  }
  const curve = normalizePowerCurvePoints(d.powerCurve)
  if (curve.length > 1)
    charts.push({
      key: 'powerCurve',
      kind: 'line',
      title: triText(presentation.locale, 'power curve'),
      axis: 'W',
      note: '',
      points: powerCurvePathPoints(curve).map(point => ({
        x: point.s,
        y: point.w,
        label: zoneClock(point.s),
        value: `${Math.round(point.w)} W`,
      })),
    })
  if (d.sport === 'bike') {
    const laps = cyclingWorkoutLaps(d)
    if (laps.some(lap => lap.powerWatts != null)) {
      const timeline = workoutLapTimeline(laps)
      charts.push({
        key: 'cyclingLaps',
        kind: 'intervals',
        title: triText(presentation.locale, 'lap splits'),
        axis: 'W',
        note: '',
        points: laps.map((lap, index) => ({
          x: timeline.spans[index].startS,
          end: timeline.spans[index].endS,
          y: lap.powerWatts,
          label: String(lap.index),
          value: lap.powerWatts == null ? '—' : `${Math.round(lap.powerWatts)} W`,
          detail: `${lap.range.label} · ${analysisRangeMetrics(presentation, d, lap.range).join(' · ')}`,
        })),
      })
    }
  }
  const pace = d.runPaceZones
  if (d.sport === 'run' && pace && pace.zoneSeconds.some(seconds => seconds > 0))
    charts.push({
      key: 'runPace',
      kind: 'bars',
      title: triText(presentation.locale, 'pace distribution'),
      axis: 'min',
      note: `10 km · ${clock(pace.tenKmRaceTimeS)}`,
      points: pace.zoneSeconds.map((seconds, index) => ({
        x: index,
        y: seconds / 60,
        label: `Z${index + 1}`,
        range: runPaceZoneRangeText(presentation, pace.boundsSPerKm, index),
        value: zoneClock(seconds),
      })),
    })
  const heartRate = runHeartRateChart(presentation, d, zones)
  if (heartRate) charts.push(heartRate)
  if (d.sport === 'run') {
    const laps = runLapSplits(presentation, d)
    const imperial = isImperial(presentation)
    const nativeSplits = imperial ? d.runSplitsStandard : d.runSplitsMetric
    if (laps.length)
      charts.push({
        key: 'runLaps',
        kind: 'bars',
        title: triText(presentation.locale, 'lap splits'),
        axis: isImperial(presentation) ? '/mi' : '/km',
        note: '',
        points: laps.map(lap => {
          const native = nativeSplits.find(
            split => lap.range.id === `split:${imperial ? 'standard' : 'metric'}:${split.split}`,
          )
          const heartRate =
            native?.averageHeartRate != null && native.averageHeartRate > 0
              ? native.averageHeartRate
              : null
          return {
            x: lap.index,
            y: lap.paceS,
            label: `lap ${lap.index}`,
            value: `${clock(lap.paceS)} ${imperial ? '/mi' : '/km'}`,
            split: {
              distanceKm: lap.range.distanceKm,
              elevationDifferenceM: native?.elevationDifferenceM ?? null,
              heartRate,
              heartRateSource: heartRate == null ? null : 'strava',
            },
          }
        }),
      })
  }
  return {
    url,
    title: d.name || d.sport,
    date: d.date,
    sport: d.sport,
    stats,
    route: activityCardRoute(activityGpsSegments(d)),
    charts,
  }
}

export const buildActivityShareButton = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  fillMissingRunPower = false,
  zones?: ActivityShareZones | null,
): N | null => {
  const data = activityShareData(f.presentation, d, fillMissingRunPower, zones)
  if (!data) return null
  const button = f.el('button', 'tri-activity-share', undefined, {
    type: 'button',
    'data-press-kit': 'note',
    'data-press-kit-activity': encodeURIComponent(JSON.stringify(data)),
    'data-site-cursor-action': '',
    'aria-label': `${triText(f.presentation.locale, 'share')} ${d.name || d.sport}`,
    'aria-haspopup': 'dialog',
    title: triText(f.presentation.locale, 'share'),
  })
  const icon = f.svg('svg', {
    viewBox: '0 0 15 15',
    fill: 'none',
    'aria-hidden': 'true',
    'data-site-cursor-icon': '',
  })
  f.add(
    icon,
    f.svg('path', {
      d: EXTERNAL_LINK_ICON_PATH,
      fill: 'currentColor',
      'fill-rule': 'evenodd',
      'clip-rule': 'evenodd',
    }),
  )
  f.add(button, icon)
  return button
}
