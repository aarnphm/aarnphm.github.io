import {
  coreBodyTemperatureWindowReader,
  type CoreBodyTemperatureActivitySample,
  type CoreBodyTemperatureCache,
} from '../../plugins/stores/core-body-temperature'
import { prefersActivityDeviceThermal, type StravaPayload } from '../../plugins/stores/strava'

const CORE_SAMPLE_MAX_DISTANCE_S = 90

type CoreMetric = 'coreTemperatureC' | 'skinTemperatureC' | 'heatStrainIndex'

const validCoreMetric = (metric: CoreMetric, value: number): boolean => {
  if (metric === 'heatStrainIndex') return value >= 0 && value <= 20
  if (metric === 'coreTemperatureC') return value >= 25 && value <= 45
  return value >= 0 && value <= 50
}

interface CoreMetricSample {
  elapsedS: number
  value: number
}

function coreMetricSamples(
  samples: CoreBodyTemperatureActivitySample[],
  metric: CoreMetric,
): CoreMetricSample[] {
  return samples
    .map(sample => ({ elapsedS: sample.elapsedS, value: sample[metric] }))
    .filter(
      (sample): sample is CoreMetricSample =>
        sample.value != null &&
        Number.isFinite(sample.value) &&
        validCoreMetric(metric, sample.value),
    )
}

function coreMetricAt(values: CoreMetricSample[], elapsedS: number): number | null {
  if (values.length === 0) return null
  let low = 0
  let high = values.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (values[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const previous = values[low - 1]
  const next = values[low]
  if (!previous && !next) return null
  const nearest =
    previous && next
      ? elapsedS - previous.elapsedS <= next.elapsedS - elapsedS
        ? previous
        : next
      : (previous ?? next)
  if (Math.abs(nearest.elapsedS - elapsedS) > CORE_SAMPLE_MAX_DISTANCE_S) return null
  if (!previous || !next || next.elapsedS === previous.elapsedS) return nearest.value
  if (
    elapsedS < previous.elapsedS ||
    elapsedS > next.elapsedS ||
    next.elapsedS - previous.elapsedS > CORE_SAMPLE_MAX_DISTANCE_S * 2
  )
    return nearest.value
  const fraction = (elapsedS - previous.elapsedS) / (next.elapsedS - previous.elapsedS)
  return previous.value + (next.value - previous.value) * fraction
}

export function enrichCoreBodyTemperature(
  payload: StravaPayload,
  core: CoreBodyTemperatureCache | null,
): void {
  if (!core) return
  const samplesForWindow = coreBodyTemperatureWindowReader(core)
  for (const detail of Object.values(payload.details)) {
    if (!detail) continue
    const points = detail.route.length >= 2 ? detail.route : detail.heartRateTrace
    if (points.length < 2) continue
    const durationS = Math.max(detail.movingTimeS, points.at(-1)?.elapsedS ?? 0)
    const samples = samplesForWindow(detail.start, durationS).filter(
      sample => sample.quality == null || sample.quality >= 2,
    )
    if (samples.length === 0) continue
    const coreValues = coreMetricSamples(samples, 'coreTemperatureC')
    const skinValues = coreMetricSamples(samples, 'skinTemperatureC')
    const heatValues = coreMetricSamples(samples, 'heatStrainIndex')
    const nativeFirst = prefersActivityDeviceThermal(detail.sport)
    for (const point of points) {
      const coreTemperatureC = coreMetricAt(coreValues, point.elapsedS)
      const skinTemperatureC = coreMetricAt(skinValues, point.elapsedS)
      const heatStrainIndex = coreMetricAt(heatValues, point.elapsedS)
      if (coreTemperatureC != null && (!nativeFirst || point.coreTemperatureC == null)) {
        point.coreTemperatureC = Math.round(coreTemperatureC * 100) / 100
        point.coreTemperatureSource = 'core-app'
      }
      if (skinTemperatureC != null && (!nativeFirst || point.skinTemperatureC == null)) {
        point.skinTemperatureC = Math.round(skinTemperatureC * 100) / 100
        point.skinTemperatureSource = 'core-app'
      }
      if (heatStrainIndex != null && (!nativeFirst || point.heatStrainIndex == null)) {
        point.heatStrainIndex = Math.round(heatStrainIndex * 10) / 10
        point.heatStrainSource = 'core-app'
      }
    }
  }
}
