import type { WahooMetrics, WahooStreams } from '../plugins/stores/wahoo'

export interface CyclingIntensityPoint {
  elapsedS: number
  distanceKm: number
  intensityFactor: number | null
}

export interface CyclingIntensityTrace {
  source: 'wahoo'
  method: 'cumulative-power-30s-v1'
  ftpWatts: number
  ftpSource: 'wahoo-summary' | 'athlete'
  points: CyclingIntensityPoint[]
}

const positive = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value) && value > 0

export function buildCyclingIntensityTrace(input: {
  streams: Pick<WahooStreams, 'time' | 'watts'> | undefined
  metrics: Pick<WahooMetrics, 'normalizedPower' | 'intensityFactor'>
  startOffsetS: number
  elapsedTimeS: number
  route: readonly { elapsedS: number; d: number }[]
  athleteFtp: number | null
}): CyclingIntensityTrace | null {
  const { streams, metrics, startOffsetS, elapsedTimeS, route } = input
  if (
    !streams ||
    streams.time.length < 30 ||
    streams.time.length !== streams.watts.length ||
    !Number.isFinite(startOffsetS) ||
    !positive(elapsedTimeS) ||
    elapsedTimeS > 48 * 3_600 ||
    streams.time.some((time, i) => !Number.isFinite(time) || (i > 0 && time <= streams.time[i - 1]))
  )
    return null
  const nativeFtp =
    positive(metrics.normalizedPower) && positive(metrics.intensityFactor)
      ? metrics.normalizedPower / metrics.intensityFactor
      : null
  const ftpWatts = positive(nativeFtp)
    ? nativeFtp
    : positive(input.athleteFtp)
      ? input.athleteFtp
      : null
  if (ftpWatts == null) return null
  const ftpSource = positive(nativeFtp) ? 'wahoo-summary' : 'athlete'
  const points: CyclingIntensityPoint[] = []
  const stride = Math.max(1, Math.ceil(elapsedTimeS / 320))
  const window: number[] = []
  let windowSum = 0
  let fourthPowerSum = 0
  let normalizedSeconds = 0
  let streamIndex = -1
  let routeIndex = 0
  let bucketMissing = false
  let validPoints = 0
  const lastSourceTime = streams.time[streams.time.length - 1]
  for (let second = 0; second <= Math.floor(elapsedTimeS); second++) {
    const sourceTime = second - startOffsetS
    while (streamIndex + 1 < streams.time.length && streams.time[streamIndex + 1] <= sourceTime)
      streamIndex++
    const watts = streams.watts[streamIndex]
    // Bound sample holding so pauses and missing records never become synthetic power.
    const observed =
      streamIndex >= 0 &&
      sourceTime <= lastSourceTime &&
      sourceTime - streams.time[streamIndex] <= 2.5 &&
      watts != null &&
      Number.isFinite(watts) &&
      watts >= 0
    let intensityFactor: number | null = null
    if (observed) {
      window.push(watts)
      windowSum += watts
      if (window.length > 30) windowSum -= window.shift() ?? 0
      if (window.length === 30) {
        fourthPowerSum += (windowSum / 30) ** 4
        normalizedSeconds++
        const normalizedPower = (fourthPowerSum / normalizedSeconds) ** 0.25
        intensityFactor = normalizedPower / ftpWatts
      }
    } else {
      window.length = 0
      windowSum = 0
    }
    bucketMissing ||= intensityFactor == null
    if (second % stride !== 0 && second !== Math.floor(elapsedTimeS)) continue
    while (routeIndex + 1 < route.length && route[routeIndex + 1].elapsedS <= second) routeIndex++
    const left = route[routeIndex]
    const right = route[routeIndex + 1] ?? left
    const fraction =
      left && right.elapsedS > left.elapsedS
        ? Math.max(0, Math.min(1, (second - left.elapsedS) / (right.elapsedS - left.elapsedS)))
        : 0
    points.push({
      elapsedS: second,
      distanceKm: left ? left.d + (right.d - left.d) * fraction : 0,
      intensityFactor: bucketMissing ? null : intensityFactor,
    })
    if (!bucketMissing) validPoints++
    bucketMissing = false
  }
  return validPoints >= 2
    ? { source: 'wahoo', method: 'cumulative-power-30s-v1', ftpWatts, ftpSource, points }
    : null
}
