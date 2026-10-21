import type { ActivityCyclingDynamics, StravaActivityDetail } from '../../plugins/stores/strava'
import type { ActivityGraphDomain } from './chart-geometry'
import type { TriNodeFactory } from './construction-contracts'
import type { WattsHeatSample } from './watts-heatmap'
import { triText } from '../triathlon-i18n'

export type CyclingDynamicsPercentMetric = 'pedalSmoothness' | 'torqueEffectiveness'

export type CyclingDynamicsSeriesKey =
  | 'leftPedalSmoothness'
  | 'rightPedalSmoothness'
  | 'leftTorqueEffectiveness'
  | 'rightTorqueEffectiveness'
  | 'leftPowerPhaseStart'
  | 'leftPowerPhaseEnd'
  | 'rightPowerPhaseStart'
  | 'rightPowerPhaseEnd'

export const cyclingDynamicsSampleCount = (dynamics: ActivityCyclingDynamics): number => {
  const length = dynamics.elapsedS.length
  return length >= 2 && dynamics.distanceKm.length === length ? length : 0
}

export const cyclingDynamicsValues = (
  dynamics: ActivityCyclingDynamics,
  key: CyclingDynamicsSeriesKey,
): (number | null)[] => {
  const values = dynamics[key]
  return values.length === cyclingDynamicsSampleCount(dynamics) ? values : []
}

export const validCyclingDynamicsValue = (value: number | null): value is number =>
  value != null && Number.isFinite(value)

export const cyclingDynamicsIndexAtDistance = (
  dynamics: ActivityCyclingDynamics,
  distanceKm: number,
): number => {
  const length = cyclingDynamicsSampleCount(dynamics)
  if (length === 0) return -1
  let low = 0
  let high = length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (dynamics.distanceKm[middle] < distanceKm) low = middle + 1
    else high = middle
  }
  if (low === 0) return 0
  if (low >= length) return length - 1
  return distanceKm - dynamics.distanceKm[low - 1] <= dynamics.distanceKm[low] - distanceKm
    ? low - 1
    : low
}

export const riderPositionAtDistance = (
  dynamics: ActivityCyclingDynamics,
  distanceKm: number,
): 'seated' | 'standing' | null => {
  const changes = dynamics.positionChanges
  if (changes.length === 0 || distanceKm < changes[0].distanceKm) return null
  let low = 0
  let high = changes.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (changes[middle].distanceKm <= distanceKm) low = middle + 1
    else high = middle
  }
  return changes[Math.max(0, low - 1)]?.position ?? null
}

interface CyclingDynamicsPaths {
  measured: string
  bridges: string
}

export const cyclingDynamicsPaths = (
  dynamics: ActivityCyclingDynamics,
  values: readonly (number | null)[],
  maxDistanceKm: number,
  py: (value: number) => number,
  circular = false,
): CyclingDynamicsPaths => {
  const px = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * 100
  let measured = ''
  let bridges = ''
  let previousValid = -1
  let segmentOpen = false
  for (let index = 0; index < values.length; index++) {
    const value = values[index]
    if (!validCyclingDynamicsValue(value)) {
      segmentOpen = false
      continue
    }
    const previousValue = previousValid >= 0 ? values[previousValid] : null
    const wraps =
      circular && validCyclingDynamicsValue(previousValue) && Math.abs(previousValue - value) > 180
    const gap =
      previousValid >= 0 &&
      (index > previousValid + 1 ||
        dynamics.elapsedS[index] - dynamics.elapsedS[previousValid] > 60 ||
        wraps)
    if (gap && !wraps && validCyclingDynamicsValue(previousValue))
      bridges += `M ${px(dynamics.distanceKm[previousValid]).toFixed(2)} ${py(previousValue).toFixed(2)} L ${px(dynamics.distanceKm[index]).toFixed(2)} ${py(value).toFixed(2)} `
    measured += `${segmentOpen && !gap ? 'L' : 'M'} ${px(dynamics.distanceKm[index]).toFixed(2)} ${py(value).toFixed(2)} `
    previousValid = index
    segmentOpen = true
  }
  return { measured, bridges }
}

export const cyclingDynamicsDomain = (values: readonly number[]): [number, number] => {
  const sorted = [...values].sort((left, right) => left - right)
  const low = sorted[Math.round((sorted.length - 1) * 0.05)]
  const high = sorted[Math.round((sorted.length - 1) * 0.95)]
  const padding = Math.max(5, (high - low) * 0.15)
  let min = Math.max(0, Math.floor((low - padding) / 5) * 5)
  let max = Math.min(100, Math.ceil((high + padding) / 5) * 5)
  if (max - min < 20) {
    const middle = (min + max) / 2
    min = Math.max(0, Math.floor((middle - 10) / 5) * 5)
    max = Math.min(100, Math.ceil((middle + 10) / 5) * 5)
  }
  return max > min ? [min, max] : [0, 100]
}

export const cyclingDynamicsAverage = (values: readonly (number | null)[]): number | null => {
  let sum = 0
  let count = 0
  for (const value of values) {
    if (!validCyclingDynamicsValue(value)) continue
    sum += value
    count++
  }
  return count > 0 ? sum / count : null
}

const routeWattsAtElapsed = (
  route: StravaActivityDetail['route'],
  elapsedS: number,
): number | null => {
  if (route.length === 0) return null
  let low = 0
  let high = route.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (route[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const index =
    low === 0
      ? 0
      : low >= route.length
        ? route.length - 1
        : elapsedS - route[low - 1].elapsedS <= route[low].elapsedS - elapsedS
          ? low - 1
          : low
  return route[index].w > 0 ? route[index].w : null
}

export const cyclingDynamicsWattsSamples = (
  d: StravaActivityDetail,
  dynamics: ActivityCyclingDynamics,
  values: readonly (number | null)[],
  graphDomain?: ActivityGraphDomain | null,
): WattsHeatSample[] => {
  const samples: WattsHeatSample[] = []
  for (let index = 0; index < values.length; index++) {
    const value = values[index]
    const distanceKm = dynamics.distanceKm[index]
    if (
      !validCyclingDynamicsValue(value) ||
      (graphDomain &&
        (distanceKm < graphDomain.startDistanceKm || distanceKm > graphDomain.endDistanceKm))
    )
      continue
    const watts = routeWattsAtElapsed(d.route, dynamics.elapsedS[index])
    if (watts != null) samples.push({ watts, value })
  }
  return samples
}

export const buildCyclingDynamicsLegend = <N>(
  f: TriNodeFactory<N>,
  side: 'left' | 'right',
  suffix = '',
): N => {
  const item = f.el(
    'span',
    `tri-cycling-dynamics-legend-item tri-cycling-dynamics-legend-item--${side}${suffix ? ` tri-cycling-dynamics-legend-item--${suffix}` : ''}`,
  )
  f.add(
    item,
    f.el('span', 'tri-cycling-dynamics-legend-line', undefined, { 'aria-hidden': 'true' }),
    f.el(
      'span',
      'tri-cycling-dynamics-legend-label',
      `${triText(f.presentation.locale, side)}${suffix ? ` ${triText(f.presentation.locale, suffix)}` : ''}`,
    ),
  )
  return item
}

type CyclingDynamicsGlossTitle = 'torque effectiveness' | 'pedal smoothness' | 'power phase'

export const buildCyclingDynamicsTitle = <N>(
  f: TriNodeFactory<N>,
  title: CyclingDynamicsGlossTitle,
): N =>
  f.el('span', 'tri-elev-d', triText(f.presentation.locale, title), {
    'data-gloss': title,
    tabindex: '0',
  })
