import type { TriathlonPresentation } from '../triathlon-presentation'
import {
  calculateExerciseLoad,
  type ActivityDevice,
  type StravaActivityDetail,
} from '../../plugins/stores/strava'
import { selectActivityAnalysisSummary } from '../activity-analysis-selection'
import { swimLengthAverages } from '../swim-metrics'
import {
  activityAverageCadence,
  activityCadenceUnit,
  formatTemperature,
  positiveMetric,
  swimTrendNumber,
} from './metric-text'

export const moreStatRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  fillMissingRunPower = false,
): [string, string][] => {
  const rows: [string, string][] = []
  const showRunPower = fillMissingRunPower && d.sport === 'run'
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  // Strava's run NP and average power use different time bases (NP can fall below average),
  // so run VI, NP and average power come from one native device pair.
  const runPower =
    d.sport !== 'run'
      ? null
      : [d.wahoo?.metrics, d.garmin]
          .map(metrics => ({ np: metrics?.normalizedPower, avg: metrics?.avgPower }))
          .find(
            (pair): pair is { np: number; avg: number } =>
              positiveMetric(pair.np) && positiveMetric(pair.avg),
          )
  if (runPower)
    rows.push([
      'variability index',
      (runPower.np / runPower.avg).toLocaleString(locale, {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3,
      }),
    ])
  else if (showRunPower) rows.push(['variability index', '—'])
  if (runPower) rows.push(['NP', `${Math.round(runPower.np)} W`])
  else if (d.deviceWatts && d.npWatts != null) rows.push(['NP', `${d.npWatts} W`])
  else if (showRunPower) rows.push(['NP', '—'])
  // Garmin reports no run TSS without a running FTP; rTSS uses the summary's pace-based IF.
  const runIntensityFactor =
    d.sport === 'run'
      ? (d.wahoo?.metrics.intensityFactor ??
        d.garmin?.intensityFactor ??
        d.calculatedIntensityFactor?.value)
      : null
  const trainingStressScore =
    d.wahoo?.metrics.trainingStressScore ??
    d.garmin?.trainingStressScore ??
    (runIntensityFactor == null ? null : calculateExerciseLoad(runIntensityFactor, d.movingTimeS))
  if (trainingStressScore != null)
    rows.push(['TSS', trainingStressScore.toLocaleString(locale, { maximumFractionDigits: 1 })])
  if (runPower) rows.push(['avg power', `${Math.round(runPower.avg)} W`])
  else if (d.avgWatts != null)
    rows.push([d.deviceWatts ? 'avg power' : 'est power', `${d.avgWatts} W`])
  else if (showRunPower) rows.push(['avg power', '—'])
  if (d.sport === 'walk' && !d.deviceWatts && d.walkPower)
    rows.push(['walking power', `${Math.round(d.walkPower.averageWatts)} W · estimated`])
  if (d.deviceWatts && d.maxWatts != null) rows.push(['max power', `${d.maxWatts} W`])
  else if (showRunPower) rows.push(['max power', '—'])
  if (d.kilojoules != null) rows.push(['energy', `${d.kilojoules} kJ`])
  else if (showRunPower) rows.push(['energy', '—'])
  if (d.calories != null) rows.push(['calories', `${d.calories.toLocaleString('en-US')} kcal`])
  if (d.sport === 'swim') {
    if (d.swimPower)
      rows.push(['drag power index', `${d.swimPower.averageIndex.toFixed(1)} · 2:30/100m = 100`])
    const poolMetrics = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
    rows.push([
      'cadence',
      poolMetrics ? `${swimTrendNumber(poolMetrics.strokesPerLength)} /length` : '—',
    ])
  } else {
    const cadence = activityAverageCadence(d)
    if (cadence != null)
      rows.push(['cadence', `${Math.round(cadence)} ${activityCadenceUnit(d.sport)}`])
    else if (d.sport === 'run') rows.push(['cadence', '—'])
  }
  if (d.maxHr != null) rows.push(['max hr', `${d.maxHr} bpm`])
  if (d.sufferScore != null) rows.push(['effort', `${d.sufferScore}`])
  if (d.deviceTemperatureC != null)
    rows.push(['device temp', formatTemperature(presentation, d.deviceTemperatureC)])
  if (d.ambientTemperatureC != null)
    rows.push(['ambient temp', formatTemperature(presentation, d.ambientTemperatureC)])
  if (d.windKph != null)
    rows.push([
      'wind',
      `${d.windKph} km/h${d.windDir ? ` ${d.windDir}` : ''}${d.windGustKph != null ? ` / gust ${d.windGustKph}` : ''}`,
    ])
  if (d.averageRelativeHumidityPct != null)
    rows.push(['humidity', `${d.averageRelativeHumidityPct}%`])
  if (d.equipment) rows.push(['equipment', d.equipment.name ?? d.equipment.id])
  const computer =
    d.computerOverride ??
    (d.computer ? (d.wahoo?.sourceDevice ?? COMPUTER_LABEL[d.computer]) : null)
  if (computer) rows.push(['computer', computer])
  if (d.device && (d.sport === 'run' || d.sport === 'walk' || d.sport === 'swim'))
    rows.push(['device', DEVICE_LABEL[d.device]])
  if (d.sources?.length || d.garmin || d.wahoo)
    rows.push([
      'source',
      d.swimProvenance
        ? 'Garmin + athlete reconstruction'
        : d.wahoo
          ? 'Wahoo'
          : d.distanceSource === 'strava'
            ? 'Strava'
            : d.garmin
              ? 'Garmin'
              : d.computer === 'wahoo'
                ? 'Wahoo'
                : 'Strava',
    ])
  return rows
}

export const activitySourceStatAttrs = (
  d: StravaActivityDetail,
): Record<string, string> | undefined => {
  if (!d.sources?.length) return undefined
  const labels = { strava: 'Strava', garmin: 'Garmin', wahoo: 'Wahoo' }
  const description = d.sources
    .map(source =>
      [`${labels[source.provider]} · ${source.activityId}`, source.name, source.fileName]
        .filter(Boolean)
        .join('\n'),
    )
    .join('\n\n')
  return { 'data-source-description': description }
}

export const activityEquipmentStatAttrs = (
  d: StravaActivityDetail,
): Record<string, string> | undefined =>
  d.equipment
    ? {
        'data-equipment-source': d.equipment.source,
        'data-equipment-id': d.equipment.id,
        title: `Strava equipment · ${d.equipment.id}`,
      }
    : undefined

const COMPUTER_LABEL = { garmin: 'Edge 1050', wahoo: 'ELEMNT BOLT 3' } as const
const DEVICE_LABEL: Record<ActivityDevice, string> = {
  'apple-watch-ultra-3': 'Apple Watch Ultra 3',
  'garmin-forerunner-970': 'Garmin Forerunner 970',
}

export const relativeHumidityStatAttrs = (
  d: StravaActivityDetail,
): Record<string, string> | undefined => {
  const provenance = d.relativeHumidityProvenance
  if (d.averageRelativeHumidityPct == null || provenance == null) return undefined
  return {
    'data-weather-source': provenance.source,
    'data-weather-source-kind': provenance.sourceKind,
    'data-weather-sampling-method': provenance.samplingMethod,
    'data-weather-input-timestamp': provenance.inputTimestamp,
    'data-weather-coverage-pct': `${provenance.coveragePct}`,
    title: `WeatherKit modeled weather · route-hour sampling · ${provenance.coveragePct}% coverage · ${provenance.inputTimestamp}`,
  }
}

export const activityAnalysisStatAttrs = (
  d: StravaActivityDetail,
  label: string,
): Record<string, string> | undefined => {
  const selected = selectActivityAnalysisSummary(d.analyses)
  if (label === 'weather impact' && selected.weatherImpact)
    return {
      'data-analysis-source': 'provider-native',
      'data-analysis-provider': 'mywindsock',
      title: 'Exact MyWindsock report attached to this Strava activity',
      'aria-label': `weather impact ${selected.weatherImpact.valuePct} percent, provider-native MyWindsock report`,
    }
  const uv = selected.uvExposure
  if (!uv || label !== uv.label) return undefined
  if (uv.kind === 'pelotan-score')
    return {
      'data-analysis-source': 'provider-native',
      'data-analysis-provider': 'pelotan',
      title: 'Exact Pelotan UV Load report attached to this Strava activity',
      'aria-label': `uv load ${uv.score}, ${uv.rawBand ?? uv.severity ?? 'unclassified'}, provider-native Pelotan report`,
    }
  return {
    'data-analysis-source': 'garden-estimate',
    title:
      uv.kind === 'garden-score'
        ? 'Garden score calibrated against held-out Pelotan reports'
        : 'Garden ambient erythemal dose from Apple WeatherKit',
    'aria-label':
      uv.kind === 'garden-score'
        ? `UV score ${uv.score}, ${uv.severity}, Garden estimate`
        : `UV dose ${uv.ambientSed} standard erythema doses, Garden estimate from Apple WeatherKit`,
  }
}
