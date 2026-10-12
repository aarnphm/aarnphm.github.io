import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { ActivityWindSample } from '../../../util/activity-wind'
import type { RunHeartCostPoint } from '../../../util/run-mechanics'
import type { RunHeartCostWindow } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { preferredActivityWind } from '../../../util/activity-wind'
import {
  activityThermalTracePoints,
  formatAltitude,
  formatThermalTemperature,
  runHeartCostValue,
  scrubDist,
  speedKph,
  zoneClock,
} from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { cyclingPowerWindAtElapsed } from './cycling-power'

export const runHeartCostWindow = (chart: HTMLElement | null): RunHeartCostWindow =>
  chart?.dataset.runCostWindow === '30' ? '30' : '300'

export const setRunHeartCostWindow = (
  chart: HTMLElement,
  window: RunHeartCostWindow,
  presentation: TriathlonPresentation,
): void => {
  chart.dataset.runCostWindow = window
  for (const option of chart.querySelectorAll<HTMLButtonElement>('.tri-run-cost-window'))
    option.setAttribute('aria-pressed', String(option.dataset.runCostWindow === window))
  for (const path of chart.querySelectorAll<SVGElement>('[data-run-cost-series]'))
    path.toggleAttribute(
      'hidden',
      path.dataset.runCostSeries !== 'cumulative' && path.dataset.runCostSeries !== window,
    )
  const label = chart.querySelector<HTMLElement>('[data-run-cost-label]')
  if (label) {
    label.dataset.i18n = window === '30' ? '30 s average' : '5 min average'
    label.textContent = triText(presentation.locale, label.dataset.i18n)
  }
}

const pace = (presentation: TriathlonPresentation, speedMps: number | null): string => {
  if (speedMps == null || !(speedMps > 0)) return '—'
  const imperial = presentation.distance === 'imperial'
  const seconds = Math.round((imperial ? 1_609.344 : 1_000) / speedMps)
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} ${imperial ? '/mi' : '/km'}`
}

const nearestCore = (
  core: readonly { elapsedS: number; coreTemperatureC: number | null }[],
  elapsedS: number,
): number | null => {
  let best: number | null = null
  let distance = Infinity
  for (const point of core) {
    if (point.coreTemperatureC == null) continue
    const gap = Math.abs(point.elapsedS - elapsedS)
    if (gap < distance) {
      distance = gap
      best = point.coreTemperatureC
    }
  }
  return distance <= 120 ? best : null
}

export const runHeartCostReadout = (
  presentation: TriathlonPresentation,
  detail: StravaActivityDetail,
  point: RunHeartCostPoint,
  window: RunHeartCostWindow,
  wind: readonly ActivityWindSample[],
): string => {
  const label = (key: string): string => triText(presentation.locale, key)
  const cost = runHeartCostValue(point, window)
  const speed = window === '30' ? point.gapSpeed30sMps : point.gapSpeed5mMps
  const heartRate = window === '30' ? point.heartRate30sBpm : point.heartRate5mBpm
  const perBeat = (value: number | null): string =>
    value == null ? '—' : `${value.toFixed(2)} m/b`
  const values = [
    zoneClock(point.elapsedS),
    scrubDist(presentation, point.distanceKm, 'run'),
    point.elevationM == null ? '—' : formatAltitude(presentation, point.elevationM),
  ]
  const core = nearestCore(activityThermalTracePoints(detail), point.elapsedS)
  if (core != null) values.push(formatThermalTemperature(presentation, core))
  values.push(
    `${label(window === '30' ? '30 s average' : '5 min average')} ${perBeat(cost)}`,
    `GAP ${pace(presentation, speed)}`,
    heartRate == null ? '— bpm' : `${Math.round(heartRate)} bpm`,
    `${label('run average')} ${perBeat(point.cumulativeMetersPerBeat)}`,
  )
  const headwind = cyclingPowerWindAtElapsed(wind, point.elapsedS, 'headwindKph')
  if (headwind != null)
    values.push(
      `${label('headwind')} ${headwind < 0 ? '−' : '+'}${speedKph(presentation, Math.abs(headwind))}`,
    )
  return values.join(' · ')
}

export const setupRunHeartCostCharts = (
  scope: HTMLElement,
  presentation: TriathlonPresentation,
  detail: StravaActivityDetail,
): (() => void) => {
  const points = detail.runHeartCost?.points ?? []
  const wind = preferredActivityWind(detail).samples
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('.tri-run-cost-window')
    const chart = button?.closest<HTMLElement>('.tri-run-heart-cost')
    if (!button || !chart) return
    const window = button.dataset.runCostWindow === '30' ? '30' : '300'
    setRunHeartCostWindow(chart, window, presentation)
    const graph = chart.querySelector<SVGSVGElement>('.tri-run-cost-plot')
    const elapsedS = Number(graph?.getAttribute('aria-valuenow'))
    let point: RunHeartCostPoint | null = null
    for (const candidate of points)
      if (
        point == null ||
        Math.abs(candidate.elapsedS - elapsedS) < Math.abs(point.elapsedS - elapsedS)
      )
        point = candidate
    if (!point) return
    const readout = runHeartCostReadout(presentation, detail, point, window, wind)
    graph?.setAttribute('aria-valuetext', readout)
    const output = chart.querySelector<HTMLElement>('.tri-fig-readout')
    if (output) output.textContent = readout
  }
  scope.addEventListener('click', onClick)
  return () => scope.removeEventListener('click', onClick)
}
