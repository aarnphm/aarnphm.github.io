import type { EnvironmentChartSample } from '../activity-environment'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { EnvironmentChartView, EnvironmentScoreModel } from './environment-scale'
import { gardenUvScoreFromDose } from '../activity-uv-score'
import { triText } from '../triathlon-i18n'
import { formatTemperature, isImperial, KM_TO_MI, speedKph } from './metric-text'

export const environmentElapsedClock = (elapsedS: number): string => {
  const totalSeconds = Math.max(0, Math.round(elapsedS))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const tail = `${minutes.toString().padStart(hours > 0 ? 2 : 1, '0')}:${seconds.toString().padStart(2, '0')}`
  return hours > 0 ? `${hours}:${tail}` : tail
}

export const environmentChartReadout = (
  presentation: TriathlonPresentation,
  sample: EnvironmentChartSample,
  scoreModel: EnvironmentScoreModel | null,
  view?: EnvironmentChartView,
): string => {
  const values = [environmentElapsedClock(sample.elapsedS)]
  if (view === 'wind' && 'providerHeadwindKph' in sample) {
    values.push(
      `${triText(presentation.locale, 'myWindsock headwind')} ${sample.providerHeadwindKph == null ? '—' : formatSignedSpeed(presentation, sample.providerHeadwindKph)}`,
    )
    if (sample.providerCrosswindKph != null)
      values.push(
        `${triText(presentation.locale, 'crosswind')} ${formatSignedSpeed(presentation, sample.providerCrosswindKph)}`,
      )
    return values.join(' · ')
  }
  if (view === 'weather-cost') {
    values.push(
      `${triText(presentation.locale, 'weather cost')} ${sample.weatherCostW == null ? '—' : formatSigned(sample.weatherCostW, ' W')}`,
    )
    return values.join(' · ')
  }
  if (view === 'air-penalty') {
    values.push(
      `${triText(presentation.locale, 'moving air penalty')} ${sample.movingAirPenaltyKm == null ? '—' : isImperial(presentation) ? `${(sample.movingAirPenaltyKm * KM_TO_MI).toFixed(2)} mi` : `${sample.movingAirPenaltyKm.toFixed(2)} km`}`,
    )
    return values.join(' · ')
  }
  if (view === 'current-speed' || view === 'current-direction') {
    const value =
      view === 'current-speed' ? sample.surfaceCurrentSpeedMps : sample.surfaceCurrentDirectionDeg
    values.push(
      value == null
        ? '—'
        : view === 'current-speed'
          ? `${value.toFixed(2)} m/s`
          : `${value.toFixed(0)}° ${triText(presentation.locale, 'toward')}`,
    )
    return values.join(' · ')
  }
  if (sample.cumulativeSed != null) {
    const scoreDoseSed =
      scoreModel?.doseClock === 'moving-telemetry'
        ? sample.cumulativeMovingTelemetrySed
        : sample.cumulativeSed
    if (scoreModel != null && scoreDoseSed != null)
      values.push(
        `${gardenUvScoreFromDose(scoreDoseSed, scoreModel.coefficientSed)} ${triText(presentation.locale, 'score')}`,
      )
    values.push(`${sample.cumulativeSed.toFixed(2)} SED`)
  }
  if (sample.uvIndex != null) values.push(`UVI ${sample.uvIndex.toFixed(1)}`)
  if (sample.ambientTemperatureC != null)
    values.push(formatTemperature(presentation, sample.ambientTemperatureC))
  if (sample.cloudCoverPct != null)
    values.push(`${Math.round(sample.cloudCoverPct)}% ${triText(presentation.locale, 'cloud')}`)
  if (sample.windSpeedKph != null)
    values.push(
      `${triText(presentation.locale, 'ambient wind speed')} ${speedKph(presentation, sample.windSpeedKph)}`,
    )
  if (sample.headwindKph != null)
    values.push(
      `${triText(presentation.locale, 'headwind')} ${formatSignedSpeed(presentation, sample.headwindKph)}`,
    )
  if (sample.crosswindKph != null)
    values.push(
      `${triText(presentation.locale, 'crosswind')} ${formatSignedSpeed(presentation, sample.crosswindKph)}`,
    )
  if (sample.apparentAirSpeedKph != null)
    values.push(
      `${triText(presentation.locale, 'apparent air')} ${speedKph(presentation, sample.apparentAirSpeedKph)}`,
    )
  if (sample.yawDeg != null)
    values.push(
      `${triText(presentation.locale, 'yaw')} ${sample.yawDeg > 0 ? '+' : ''}${sample.yawDeg.toFixed(1)}°`,
    )
  return values.join(' · ')
}

// Sector shares of moving time: 0 is headwind and the sectors run clockwise toward the right.
export const relativeWindReadout = (
  presentation: TriathlonPresentation,
  shares: readonly number[],
): string => {
  const share = (indices: readonly number[]) =>
    Math.round(indices.reduce<number>((total, index) => total + (shares[index] ?? 0), 0))
  return [
    `${triText(presentation.locale, 'head side')} ${share([7, 0, 1])}%`,
    `${triText(presentation.locale, 'crosswind')} ${share([2, 6])}%`,
    `${triText(presentation.locale, 'tail side')} ${share([3, 4, 5])}%`,
  ].join(' · ')
}

export const formatSigned = (value: number, suffix: string): string =>
  `${value > 0 ? '+' : ''}${value.toFixed(1)}${suffix}`

const formatSignedSpeed = (presentation: TriathlonPresentation, value: number): string =>
  `${value > 0 ? '+' : ''}${speedKph(presentation, value)}`
