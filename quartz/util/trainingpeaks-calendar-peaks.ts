import type { PowerCurvePoint } from '../plugins/stores/strava'
import { TRAINING_PEAK_SECONDS, type TrainingPeaksCalendarPeak } from './trainingpeaks-calendar'

interface HeartRateStream {
  time?: readonly number[]
  heartrate?: readonly number[]
}

const heartRatePeaks = (stream: HeartRateStream | undefined): Map<number, number> => {
  const peaks = new Map<number, number>()
  const times = stream?.time
  const values = stream?.heartrate
  if (!times?.length || !values || values.length !== times.length) return peaks
  let previous = -1
  for (const time of times) {
    if (!Number.isFinite(time) || time < 0 || time < previous) return peaks
    previous = time
  }
  const first = Math.round(times[0])
  const length = Math.round(previous) - first + 1
  if (length < 5 || length > 7 * 24 * 60 * 60) return peaks
  const sums = new Float64Array(length)
  const counts = new Uint32Array(length)
  for (let index = 0; index < times.length; index++) {
    const value = values[index]
    if (!Number.isFinite(value) || value <= 0) continue
    const second = Math.round(times[index]) - first
    sums[second] += value
    counts[second]++
  }
  const sum = new Float64Array(length + 1)
  const observed = new Uint32Array(length + 1)
  for (let second = 0; second < length; second++) {
    sum[second + 1] = sum[second] + (counts[second] ? sums[second] / counts[second] : 0)
    observed[second + 1] = observed[second] + (counts[second] > 0 ? 1 : 0)
  }
  // A gap is unknown HR. Each elapsed second must be observed before it can form a peak.
  for (const duration of TRAINING_PEAK_SECONDS) {
    let best: number | null = null
    for (let end = duration; end <= length; end++) {
      const start = end - duration
      if (observed[end] - observed[start] !== duration) continue
      const average = (sum[end] - sum[start]) / duration
      if (best === null || average > best) best = average
    }
    if (best !== null) peaks.set(duration, best)
  }
  return peaks
}

export const trainingPeaksActivityPeaks = ({
  garmin,
  strava,
  powerCurve,
}: {
  garmin?: HeartRateStream
  strava?: HeartRateStream
  powerCurve?: readonly PowerCurvePoint[] | null
}): TrainingPeaksCalendarPeak[] | undefined => {
  const native = heartRatePeaks(garmin)
  const recorded = heartRatePeaks(strava)
  const peaks = TRAINING_PEAK_SECONDS.map((seconds): TrainingPeaksCalendarPeak => {
    const heartRateBpm = native.get(seconds) ?? recorded.get(seconds) ?? null
    const power = powerCurve?.find(point => point.s === seconds)?.w
    return {
      seconds,
      heartRateBpm,
      heartRateSource: heartRateBpm === null ? null : native.has(seconds) ? 'garmin' : 'strava',
      powerWatts: power != null && Number.isFinite(power) && power >= 0 ? power : null,
    }
  })
  return peaks.some(point => point.heartRateBpm !== null || point.powerWatts !== null)
    ? peaks
    : undefined
}
