import type { ActivityDistanceEffort, StravaStreams } from '../plugins/stores/strava'

const MILE_M = 1609.344
const RUN_DISTANCES: readonly (readonly [string, number])[] = [
  ['400m', 400],
  ['1/2 mile', MILE_M / 2],
  ['1K', 1_000],
  ['1 mile', MILE_M],
  ['2 mile', 2 * MILE_M],
  ['5K', 5_000],
  ['10K', 10_000],
  ['15K', 15_000],
  ['10 mile', 10 * MILE_M],
  ['20K', 20_000],
  ['Half marathon', 21_097.5],
  ['30K', 30_000],
  ['Marathon', 42_195],
  ['50K', 50_000],
]

interface Sample {
  time: number
  distance: number
  altitude: number | null
  heartRate: number | null
  heartRateSum: number
  heartRateSeconds: number
}

const interpolate = (left: Sample, right: Sample, fraction: number): Sample => {
  if (fraction === 0) return left
  if (fraction === 1) return right
  const duration = (right.time - left.time) * fraction
  const heartRate =
    left.heartRate != null && right.heartRate != null
      ? left.heartRate + (right.heartRate - left.heartRate) * fraction
      : null
  return {
    time: left.time + duration,
    distance: left.distance + (right.distance - left.distance) * fraction,
    altitude:
      left.altitude != null && right.altitude != null
        ? left.altitude + (right.altitude - left.altitude) * fraction
        : null,
    heartRate,
    heartRateSum:
      left.heartRateSum +
      (left.heartRate != null && heartRate != null
        ? ((left.heartRate + heartRate) / 2) * duration
        : 0),
    heartRateSeconds: left.heartRateSeconds + (heartRate != null ? duration : 0),
  }
}

export function runBestEfforts(
  streams: Pick<StravaStreams, 'time' | 'distance' | 'altitude' | 'heartrate'> | undefined,
): ActivityDistanceEffort[] {
  if (!streams?.time || streams.time.length < 2 || streams.time.length !== streams.distance.length)
    return []

  const samples: Sample[] = []
  for (let index = 0; index < streams.time.length; index++) {
    const time = streams.time[index]
    const distance = streams.distance[index]
    let previous = samples.at(-1)
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(distance) ||
      time < 0 ||
      distance < 0 ||
      (previous && (time < previous.time || distance < previous.distance))
    )
      return []
    // Multiple records can share a whole-second timestamp; retain its final sample.
    if (previous?.time === time) {
      samples.pop()
      previous = samples.at(-1)
    }
    const hr = streams.heartrate?.[index]
    const heartRate = hr != null && Number.isFinite(hr) && hr > 0 ? hr : null
    const previousHeartRate = previous?.heartRate
    const covered = previousHeartRate != null && heartRate != null
    const duration = previous ? time - previous.time : 0
    samples.push({
      time,
      distance,
      altitude: Number.isFinite(streams.altitude[index]) ? streams.altitude[index] : null,
      heartRate,
      heartRateSum:
        (previous?.heartRateSum ?? 0) +
        (covered ? ((previousHeartRate + heartRate) / 2) * duration : 0),
      heartRateSeconds: (previous?.heartRateSeconds ?? 0) + (covered ? duration : 0),
    })
  }
  if (samples.length < 2) return []

  const efforts: ActivityDistanceEffort[] = []
  for (const [label, targetDistanceM] of RUN_DISTANCES) {
    if (samples[samples.length - 1].distance - samples[0].distance < targetDistanceM) break
    let bestStart = samples[0]
    let bestEnd = samples[samples.length - 1]
    let bestDuration = Infinity
    const consider = (start: Sample, end: Sample): void => {
      const duration = end.time - start.time
      if (duration > 0 && duration < bestDuration) {
        bestDuration = duration
        bestStart = start
        bestEnd = end
      }
    }

    // On the interpolated distance trace, a fastest window has a recorded sample at one end.
    let endIndex = 1
    for (const start of samples) {
      const target = start.distance + targetDistanceM
      while (endIndex < samples.length && samples[endIndex].distance < target) endIndex++
      if (endIndex === samples.length) break
      const left = samples[endIndex - 1]
      const right = samples[endIndex]
      consider(
        start,
        interpolate(left, right, (target - left.distance) / (right.distance - left.distance)),
      )
    }
    let startIndex = 0
    for (const end of samples) {
      const target = end.distance - targetDistanceM
      if (target < samples[0].distance) continue
      while (startIndex + 1 < samples.length && samples[startIndex + 1].distance <= target)
        startIndex++
      const left = samples[startIndex]
      const right = samples[startIndex + 1]
      consider(
        interpolate(left, right, (target - left.distance) / (right.distance - left.distance)),
        end,
      )
    }

    const heartRateSeconds = bestEnd.heartRateSeconds - bestStart.heartRateSeconds
    efforts.push({
      label,
      targetDistanceM,
      elapsedTimeS: Math.ceil(bestDuration),
      averageSpeedKph: (targetDistanceM / bestDuration) * 3.6,
      averageHeartRate:
        heartRateSeconds > 0
          ? Math.round((bestEnd.heartRateSum - bestStart.heartRateSum) / heartRateSeconds)
          : null,
      elevationDeltaM:
        bestStart.altitude != null && bestEnd.altitude != null
          ? Math.round((bestEnd.altitude - bestStart.altitude) * 10) / 10
          : null,
    })
  }
  return efforts
}
