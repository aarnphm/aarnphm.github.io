import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import { STROKE_LABEL, SWIM_STROKES, type SwimStroke } from '../../plugins/stores/apple'
import { selectActivityAnalysisSummary } from '../activity-analysis-selection'
import { swimLengthAverages, swimPaceSeconds } from '../swim-metrics'
import { swimActivityHeaderValue, triText } from '../triathlon-i18n'
import { moreStatRows } from './activity-stat-details'
import { clock, dist, dur, formatTemperature, positiveMetric, rate, speedKph } from './metric-text'
import { strengthMass } from './strength-moves'
import { activityTrainingEffectLabel } from './training-effect-data'

const RUN_TREND_TARGETS = [
  { distanceKm: 5, label: '5k trend' },
  { distanceKm: 10, label: '10k trend' },
  { distanceKm: 21.0975, label: 'half trend' },
  { distanceKm: 42.195, label: 'marathon trend' },
]
const RUN_RIEGEL_EXPONENT = 1.06
const SWIM_PROJECTION_LENGTHS = [19, 38]

const runTrendRow = (distanceKm: number, movingTimeS: number): [string, string] | null => {
  if (
    !Number.isFinite(distanceKm) ||
    !Number.isFinite(movingTimeS) ||
    distanceKm <= 0 ||
    movingTimeS <= 0
  )
    return null
  const target = RUN_TREND_TARGETS.find(candidate => distanceKm < candidate.distanceKm)
  if (!target) return null
  const predictedTimeS = movingTimeS * Math.pow(target.distanceKm / distanceKm, RUN_RIEGEL_EXPONENT)
  return [target.label, dur(predictedTimeS)]
}

const swimProjection = (d: StravaActivityDetail): string => {
  const pace = positiveMetric(d.swimPaceSPer100m)
    ? d.swimPaceSPer100m
    : swimPaceSeconds(d.distanceKm * 1_000, d.movingTimeS)
  return positiveMetric(pace)
    ? SWIM_PROJECTION_LENGTHS.map(length => dur(pace * length)).join(' / ')
    : '—'
}

const primarySwimStroke = (d: StravaActivityDetail): string => {
  const entries: [SwimStroke, number][] = SWIM_STROKES.map((stroke): [SwimStroke, number] => [
    stroke,
    d.strokes?.[stroke] ?? 0,
  ])
    .filter(([, distanceM]) => distanceM > 0)
    .sort((left, right) => right[1] - left[1])
  return entries[0] ? STROKE_LABEL[entries[0][0]] : STROKE_LABEL.freestyle
}

// Elapsed is the native recorded clock. It earns a row only when it would print differently
// from moving time, as a row under it.
const timeRows = (d: StravaActivityDetail): [string, string][] => {
  const moving = dur(d.movingTimeS)
  const elapsed =
    Number.isFinite(d.elapsedTimeS) && d.elapsedTimeS > d.movingTimeS ? dur(d.elapsedTimeS) : null
  return elapsed && elapsed !== moving
    ? [
        ['time', moving],
        ['elapsed time', elapsed],
      ]
    : [['time', moving]]
}

const activityTrainingRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): [string, string][] => {
  const garmin = d.garmin
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const rows: [string, string][] = []
  const intensityFactor =
    d.wahoo?.metrics.intensityFactor ??
    garmin?.intensityFactor ??
    d.calculatedIntensityFactor?.value
  if (intensityFactor != null)
    rows.push([
      'intensity factor',
      intensityFactor.toLocaleString(locale, {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3,
      }),
    ])
  rows.push(['training effect', triText(presentation.locale, activityTrainingEffectLabel(d))])
  const exerciseLoad = garmin?.exerciseLoad ?? d.calculatedExerciseLoad?.value
  if (exerciseLoad != null)
    rows.push(['exercise load', Math.round(exerciseLoad).toLocaleString(locale)])
  const normalizedPower = d.wahoo?.metrics.normalizedPower
  const averagePower = d.wahoo?.metrics.avgPower
  if (
    d.sport === 'bike' &&
    normalizedPower != null &&
    Number.isFinite(normalizedPower) &&
    normalizedPower >= 0 &&
    positiveMetric(averagePower)
  )
    rows.push([
      'variability index',
      (normalizedPower / averagePower).toLocaleString(locale, {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3,
      }),
    ])
  return rows
}

const activityAnalysisStatRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): [string, string][] => {
  const selected = selectActivityAnalysisSummary(d.analyses)
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const rows: [string, string][] = []
  if (selected.weatherImpact)
    rows.push([
      'weather impact',
      `${selected.weatherImpact.valuePct.toLocaleString(locale, { maximumFractionDigits: 2 })}%`,
    ])
  const uv = selected.uvExposure
  if (uv?.kind === 'pelotan-score')
    rows.push([uv.label, `${uv.score} · ${uv.rawBand ?? uv.severity ?? '—'}`])
  else if (uv?.kind === 'garden-score')
    rows.push([
      uv.label,
      `${uv.score} · ${uv.severity[0].toUpperCase()}${uv.severity.slice(1)} · ${uv.sourceToken}`,
    ])
  else if (uv?.kind === 'garden-sed')
    rows.push([
      uv.label,
      `${uv.ambientSed.toLocaleString(locale, { maximumFractionDigits: 2 })} SED · ${uv.sourceToken}`,
    ])
  return rows
}

export const activityStatRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  includeAnalyses = true,
): [string, string][] => {
  if (d.sport === 'sauna' && d.sauna) {
    const rows: [string, string][] = [
      ['time', d.sauna.time],
      ['duration', dur(d.movingTimeS)],
      ['temperature', formatTemperature(presentation, d.sauna.temperatureC)],
      [
        'humidity',
        `${d.sauna.humidityPct.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
          maximumFractionDigits: 1,
        })}%`,
      ],
    ]
    if (d.avgHr)
      rows.push(['avg hr', `${d.avgHr} bpm${d.sauna.heartRateSource === 'oura' ? ' · Oura' : ''}`])
    if (d.sauna.heatTrainingLoad != null)
      rows.push([
        'HTL',
        d.sauna.heatTrainingLoad.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
          maximumFractionDigits: 1,
        }),
      ])
    rows.push(...activityTrainingRows(presentation, d))
    rows.push(['cooldown', d.sauna.cooldown])
    if (includeAnalyses) rows.push(...activityAnalysisStatRows(presentation, d))
    return rows
  }
  if (d.sport === 'strength') {
    const rows: [string, string][] = timeRows(d)
    if (d.strength?.volumeKg != null)
      rows.push(['volume', strengthMass(presentation, d.strength.volumeKg)])
    if (d.strength?.totalSets != null) rows.push(['sets', String(d.strength.totalSets)])
    if (d.strength?.totalReps != null) rows.push(['reps', String(d.strength.totalReps)])
    if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
    rows.push(...activityTrainingRows(presentation, d))
    if (includeAnalyses) rows.push(...activityAnalysisStatRows(presentation, d))
    return rows
  }
  if (d.sport === 'treatment' || d.sport === 'yoga') {
    const rows: [string, string][] = timeRows(d)
    if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
    rows.push(...activityTrainingRows(presentation, d))
    if (includeAnalyses) rows.push(...activityAnalysisStatRows(presentation, d))
    return rows
  }
  const activityRate =
    d.sport === 'swim' && positiveMetric(d.swimPaceSPer100m)
      ? `${clock(d.swimPaceSPer100m)} /100m`
      : rate(presentation, d.sport, d.distanceKm, d.movingTimeS)
  const rows: [string, string][] = [
    ['distance', dist(presentation, d.distanceKm, d.sport)],
    ...timeRows(d),
    [d.sport === 'bike' ? 'speed' : 'pace', activityRate],
  ]
  if (d.sport === 'bike' && d.maxSpeedKph != null)
    rows.push(['max speed', speedKph(presentation, d.maxSpeedKph)])
  if (d.sport === 'run') {
    const trend = runTrendRow(d.distanceKm, d.movingTimeS)
    if (trend) rows.push(trend)
  }
  if (d.sport === 'swim')
    rows.push([
      'stroke rate',
      positiveMetric(d.strokeRateSpm) ? `${Math.round(d.strokeRateSpm)} spm` : '—',
    ])
  if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
  rows.push(...activityTrainingRows(presentation, d))
  if (includeAnalyses) rows.push(...activityAnalysisStatRows(presentation, d))
  if (d.sport === 'swim') {
    const poolMetrics = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
    if (d.swimLocation !== 'pool') {
      rows.push([
        'water temp',
        d.waterTemperatureC == null ? '—' : formatTemperature(presentation, d.waterTemperatureC),
      ])
    }
    const swolf = poolMetrics
      ? swimActivityHeaderValue(presentation.locale, 'swolf', poolMetrics.swolf, '')
      : '—'
    rows.push(['SWOLF', swolf])
    rows.push(['1.9k / 3.8k', swimProjection(d)])
    rows.push(['stroke type', primarySwimStroke(d)])
    rows.push([
      'strokes',
      positiveMetric(d.strokeCount)
        ? `${Math.round(d.strokeCount).toLocaleString('en-US')} · ${(
            (d.distanceKm * 1_000) /
            d.strokeCount
          ).toFixed(2)} m/str`
        : '—',
    ])
  }
  return rows
}

export const activityTableRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  fillMissingRunPower = false,
): [string, string][] => [
  ...activityStatRows(presentation, d, false),
  ...moreStatRows(presentation, d, fillMissingRunPower),
  ...activityAnalysisStatRows(presentation, d),
]
