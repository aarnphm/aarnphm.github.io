import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { GardenEnvironmentSample } from '../../../util/activity-environment'
import type { CyclingPowerPoint } from '../../../util/cycling-power'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { formatAltitude, scrubDist, speedKph, zoneClock } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'

export type CyclingPowerWindow = '30' | '300'

export const cyclingPowerWindow = (chart: HTMLElement | null): CyclingPowerWindow =>
  chart?.dataset.cyclingPowerWindow === '300' ? '300' : '30'

export const setCyclingPowerWindow = (
  chart: HTMLElement,
  window: CyclingPowerWindow,
  presentation: TriathlonPresentation,
): void => {
  chart.dataset.cyclingPowerWindow = window
  for (const option of chart.querySelectorAll<HTMLButtonElement>('.tri-cycling-power-window'))
    option.setAttribute('aria-pressed', String(option.dataset.cyclingPowerWindow === window))
  for (const path of chart.querySelectorAll<SVGElement>('[data-cycling-power-series]'))
    path.toggleAttribute(
      'hidden',
      path.dataset.cyclingPowerSeries !== 'cumulative' &&
        path.dataset.cyclingPowerSeries !== window,
    )
  const label = chart.querySelector<HTMLElement>('[data-cycling-power-label]')
  if (label) {
    label.dataset.i18n = window === '30' ? '30 s average' : '5 min average'
    label.textContent = triText(presentation.locale, label.dataset.i18n)
  }
}

type WindMetric = 'headwindKph' | 'crosswindKph' | 'apparentAirSpeedKph' | 'yawDeg'

export const cyclingPowerWindAtElapsed = (
  samples: readonly GardenEnvironmentSample[],
  elapsedS: number,
  metric: WindMetric,
): number | null => {
  if (
    samples.length === 0 ||
    elapsedS < samples[0].elapsedS ||
    elapsedS > samples[samples.length - 1].elapsedS
  )
    return null
  const rightIndex = samples.findIndex(sample => sample.elapsedS >= elapsedS)
  const right = samples[rightIndex]
  if (!right) return null
  if (right.elapsedS === elapsedS) return right[metric]
  const left = samples[rightIndex - 1]
  if (!left || left[metric] == null || right[metric] == null) return null
  const fraction = (elapsedS - left.elapsedS) / (right.elapsedS - left.elapsedS)
  return left[metric] + (right[metric] - left[metric]) * fraction
}

export const cyclingPowerReadout = (
  presentation: TriathlonPresentation,
  point: CyclingPowerPoint,
  window: CyclingPowerWindow,
  weather: readonly GardenEnvironmentSample[],
  distanceAvailable = true,
): string => {
  const label = (key: string): string => triText(presentation.locale, key)
  const power = window === '30' ? point.power30sWatts : point.power5mWatts
  const watts = (value: number | null): string => (value == null ? '—' : `${Math.round(value)} W`)
  const values = [
    zoneClock(point.elapsedS),
    ...(distanceAvailable ? [scrubDist(presentation, point.distanceKm, 'bike')] : []),
    point.elevationM == null ? '—' : formatAltitude(presentation, point.elevationM),
    `${label(window === '30' ? '30 s average' : '5 min average')} ${watts(power)}`,
    `${label('ride average')} ${watts(point.cumulativePowerWatts)}`,
  ]
  const headwind = cyclingPowerWindAtElapsed(weather, point.elapsedS, 'headwindKph')
  const crosswind = cyclingPowerWindAtElapsed(weather, point.elapsedS, 'crosswindKph')
  const apparent = cyclingPowerWindAtElapsed(weather, point.elapsedS, 'apparentAirSpeedKph')
  const yaw = cyclingPowerWindAtElapsed(weather, point.elapsedS, 'yawDeg')
  const signedSpeed = (value: number): string =>
    `${value < 0 ? '−' : '+'}${speedKph(presentation, Math.abs(value))}`
  if (headwind != null) values.push(`${label('headwind')} ${signedSpeed(headwind)}`)
  if (crosswind != null) values.push(`${label('crosswind')} ${signedSpeed(crosswind)}`)
  if (apparent != null) values.push(`${label('apparent air')} ${speedKph(presentation, apparent)}`)
  if (yaw != null) values.push(`${label('yaw')} ${yaw > 0 ? '+' : ''}${yaw.toFixed(1)}°`)
  if (headwind == null && crosswind == null && apparent == null)
    values.push(label('wind unavailable'))
  return values.join(' · ')
}

export const setupCyclingPowerCharts = (
  scope: HTMLElement,
  presentation: TriathlonPresentation,
  detail: StravaActivityDetail,
): (() => void) => {
  const points = detail.cyclingPowerTrace?.points ?? []
  const distanceAvailable = detail.distanceKm > 0 || points.some(point => point.distanceKm > 0)
  const weather = detail.analyses.derived.environment?.samples ?? []
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('.tri-cycling-power-window')
    const chart = button?.closest<HTMLElement>('.tri-cycling-power')
    if (!button || !chart) return
    const window = button.dataset.cyclingPowerWindow === '30' ? '30' : '300'
    setCyclingPowerWindow(chart, window, presentation)
    const graph = chart.querySelector<SVGSVGElement>('.tri-cycling-power-plot')
    const elapsedS = Number(graph?.getAttribute('aria-valuenow'))
    const point = points.reduce<CyclingPowerPoint | null>(
      (nearest, candidate) =>
        nearest == null ||
        Math.abs(candidate.elapsedS - elapsedS) < Math.abs(nearest.elapsedS - elapsedS)
          ? candidate
          : nearest,
      null,
    )
    if (point) {
      const readout = cyclingPowerReadout(presentation, point, window, weather, distanceAvailable)
      graph?.setAttribute('aria-valuetext', readout)
      const output = chart.querySelector<HTMLElement>('.tri-fig-readout')
      if (output) output.textContent = readout
    }
  }
  scope.addEventListener('click', onClick)
  return () => scope.removeEventListener('click', onClick)
}
