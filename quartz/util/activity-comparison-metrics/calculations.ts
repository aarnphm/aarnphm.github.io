import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type {
  ActivityComparisonStatKey,
  ActivityComparisonStatSource,
  ActivityComparisonStats,
} from '../activity-comparison-metrics'

export const DECOUPLING_MIN_S = 1_200
export const PEAK_DURATIONS: readonly [ActivityComparisonStatKey, number][] = [
  ['peak-5s', 5],
  ['peak-30s', 30],
  ['peak-1m', 60],
  ['peak-5m', 300],
  ['peak-20m', 1_200],
  ['peak-60m', 3_600],
]

export type TimedValue = { t: number; v: number | null }
export type RouteSample = {
  t: number
  d: number
  alt: number
  hr: number | null
  cad: number | null
}

export const finite = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value)

export const positive = (value: number | null | undefined): value is number =>
  finite(value) && value > 0

export const routeSamples = (activity: StravaActivityDetail): RouteSample[] =>
  activity.route
    .filter(point => finite(point.elapsedS) && finite(point.d) && point.d >= 0)
    .map(point => ({
      t: point.elapsedS,
      d: point.d,
      alt: point.alt,
      hr: positive(point.hr) ? point.hr : null,
      cad: positive(point.cad) ? point.cad : null,
    }))
    .sort((a, b) => a.t - b.t)

// Pool swims carry no GPS route; their distance timeline is the recorded lengths, with rests flat.
export const swimSamples = (activity: StravaActivityDetail): RouteSample[] =>
  activity.swimIntervals
    .filter(
      interval =>
        finite(interval.startElapsedS) &&
        finite(interval.endElapsedS) &&
        interval.endElapsedS > interval.startElapsedS &&
        positive(interval.distanceM) &&
        finite(interval.cumulativeDistanceM),
    )
    .flatMap(interval => [
      {
        t: interval.startElapsedS,
        d: (interval.cumulativeDistanceM - interval.distanceM) / 1_000,
        alt: 0,
        hr: null,
        cad: null,
      },
      {
        t: interval.endElapsedS,
        d: interval.cumulativeDistanceM / 1_000,
        alt: 0,
        hr: null,
        cad: null,
      },
    ])
    .sort((a, b) => a.t - b.t)

export const timedRoute = (activity: StravaActivityDetail): RouteSample[] => {
  const route = routeSamples(activity)
  return route.length >= 2 || activity.sport !== 'swim' ? route : swimSamples(activity)
}

/** Time of the first sample that reaches `km`, interpolated along the running distance maximum. */
export const timeAtDistance = (samples: readonly RouteSample[], km: number): number | null => {
  let reachedD = -Infinity
  let reachedT = 0
  for (const sample of samples) {
    if (sample.d <= reachedD) continue
    if (sample.d >= km) {
      if (!Number.isFinite(reachedD)) return sample.t
      return reachedT + ((sample.t - reachedT) * (km - reachedD)) / (sample.d - reachedD)
    }
    reachedD = sample.d
    reachedT = sample.t
  }
  return null
}

export const distanceAtTime = (samples: readonly RouteSample[], t: number): number | null => {
  if (samples.length === 0 || t < samples[0].t || t > samples[samples.length - 1].t) return null
  let reached = samples[0].d
  for (let index = 1; index < samples.length; index++) {
    const previous = samples[index - 1]
    const next = samples[index]
    if (t > next.t) {
      reached = Math.max(reached, next.d)
      continue
    }
    const span = next.t - previous.t
    const value = span > 0 ? previous.d + ((next.d - previous.d) * (t - previous.t)) / span : next.d
    return Math.max(reached, value)
  }
  return reached
}

/** Trapezoidal integral of linearly interpolated samples over [t0, t1], skipping gaps. */
export const integrate = (
  samples: readonly TimedValue[],
  t0: number,
  t1: number,
  transform: (value: number) => number = value => value,
): { total: number; coveredS: number } => {
  let total = 0
  let coveredS = 0
  for (let index = 1; index < samples.length; index++) {
    const left = samples[index - 1]
    const right = samples[index]
    if (left.v == null || right.v == null || right.t <= left.t) continue
    const lo = Math.max(left.t, t0)
    const hi = Math.min(right.t, t1)
    if (hi <= lo) continue
    const slope = (right.v - left.v) / (right.t - left.t)
    const startValue = left.v + slope * (lo - left.t)
    const endValue = left.v + slope * (hi - left.t)
    total += ((transform(startValue) + transform(endValue)) / 2) * (hi - lo)
    coveredS += hi - lo
  }
  return { total, coveredS }
}

export const meanOver = (
  samples: readonly TimedValue[],
  t0: number,
  t1: number,
  minCoverage: number,
): number | null => {
  const { total, coveredS } = integrate(samples, t0, t1)
  return coveredS > 0 && coveredS >= (t1 - t0) * minCoverage ? total / coveredS : null
}

export const normalizedPowerOver = (
  power30s: readonly TimedValue[],
  t0: number,
  t1: number,
): number | null => {
  const { total, coveredS } = integrate(power30s, t0, t1, value => Math.max(0, value) ** 4)
  return coveredS >= 30 && coveredS >= (t1 - t0) * 0.25 ? (total / coveredS) ** 0.25 : null
}

export type PowerTraceHoles = { riddenShare: number; watts: number }

/**
 * The 30 s trace is null for 30 s after every pause and downsampling widens those holes, so the
 * covered trace undercounts riding near stops. The activity's native moving time and work fix how
 * much hole time was ridden and at what power: a full-range selection then reproduces both totals,
 * while a stop-free stretch keeps its traced values.
 */
export const powerTraceHoles = (
  activity: StravaActivityDetail,
  power30s: readonly TimedValue[],
): PowerTraceHoles => {
  const start = power30s[0]?.t
  const end = power30s[power30s.length - 1]?.t
  if (!finite(start) || !finite(end) || end <= start || !positive(activity.movingTimeS))
    return { riddenShare: 0, watts: 0 }
  const whole = integrate(power30s, start, end)
  const holeS = end - start - whole.coveredS
  if (holeS <= 0) return { riddenShare: 0, watts: 0 }
  const riddenShare = Math.min(1, Math.max(0, (activity.movingTimeS - whole.coveredS) / holeS))
  const riddenS = riddenShare * holeS
  const watts =
    positive(activity.kilojoules) && riddenS > 0
      ? Math.max(0, (activity.kilojoules * 1_000 - whole.total) / riddenS)
      : 0
  return { riddenShare, watts }
}

export const power30sSamples = (activity: StravaActivityDetail): TimedValue[] =>
  activity.cyclingPowerTrace?.points.map(point => ({
    t: point.elapsedS,
    v: finite(point.power30sWatts) ? point.power30sWatts : null,
  })) ?? []

export const routePowerSamples = (
  activity: StravaActivityDetail,
  excludeZero: boolean,
): TimedValue[] =>
  activity.route
    .filter(point => finite(point.elapsedS))
    .map(point => ({
      t: point.elapsedS,
      v: finite(point.w) && point.w >= 0 && !(excludeZero && point.w === 0) ? point.w : null,
    }))
    .sort((a, b) => a.t - b.t)

export const routeHasPower = (activity: StravaActivityDetail): boolean =>
  activity.deviceWatts && activity.route.some(point => positive(point.w))

export const heartRateSamples = (
  activity: StravaActivityDetail,
  samples: readonly RouteSample[],
): TimedValue[] =>
  samples.some(sample => sample.hr != null)
    ? samples.map(sample => ({ t: sample.t, v: sample.hr }))
    : activity.heartRateTrace
        .filter(point => finite(point.elapsedS))
        .map(point => ({
          t: point.elapsedS,
          v: positive(point.heartRate) ? point.heartRate : null,
        }))
        .sort((a, b) => a.t - b.t)

/** Stroke rate over the lengths whose midpoint falls inside the distance window. */
export const swimStrokeRateOver = (
  activity: StravaActivityDetail,
  startKm: number,
  endKm: number,
): number | null => {
  let strokes = 0
  let strokeTimeS = 0
  for (const interval of activity.swimIntervals) {
    const midKm = (interval.cumulativeDistanceM - interval.distanceM / 2) / 1_000
    if (midKm < startKm || midKm > endKm) continue
    if (!positive(interval.strokeCount) || !positive(interval.strokeTimeS)) continue
    strokes += interval.strokeCount
    strokeTimeS += interval.strokeTimeS
  }
  return strokeTimeS > 0 ? (strokes / strokeTimeS) * 60 : null
}

/** Power-weighted left share from the sampled right-pedal balance. */
export const powerBalanceLeftPct = (
  activity: StravaActivityDetail,
  t0 = -Infinity,
  t1 = Infinity,
): number | null => {
  let weighted = 0
  let watts = 0
  let samples = 0
  for (const point of activity.route) {
    const right = point.rightPowerPct
    if (
      !finite(right) ||
      right < 0 ||
      right > 100 ||
      !positive(point.w) ||
      point.elapsedS < t0 ||
      point.elapsedS > t1
    )
      continue
    weighted += (100 - right) * point.w
    watts += point.w
    samples++
  }
  return samples >= 2 && watts > 0 ? weighted / watts : null
}

export const sideMean = (values: readonly (number | null)[]): number | null => {
  let total = 0
  let count = 0
  for (const value of values)
    if (finite(value) && value >= 0 && value <= 100) {
      total += value
      count++
    }
  return count > 0 ? total / count : null
}

export const elevationProfile = (activity: StravaActivityDetail): { d: number; alt: number }[] => {
  const trace = activity.cyclingPowerTrace?.points
    .filter(point => finite(point.elevationM) && finite(point.distanceKm))
    .map(point => ({ d: point.distanceKm, alt: point.elevationM as number }))
  if (trace && trace.length >= 2) return trace
  return activity.route
    .filter(point => finite(point.d) && finite(point.alt))
    .map(point => ({ d: point.d, alt: point.alt }))
    .sort((a, b) => a.d - b.d)
}

export const altitudeAt = (
  profile: readonly { d: number; alt: number }[],
  km: number,
): number | null => {
  if (profile.length === 0 || km < profile[0].d || km > profile[profile.length - 1].d) return null
  for (let index = 1; index < profile.length; index++) {
    const previous = profile[index - 1]
    const next = profile[index]
    if (km > next.d) continue
    const span = next.d - previous.d
    return span > 0
      ? previous.alt + ((next.alt - previous.alt) * (km - previous.d)) / span
      : next.alt
  }
  return profile[profile.length - 1].alt
}

export const elevationGainOver = (
  profile: readonly { d: number; alt: number }[],
  startKm: number,
  endKm: number,
): number | null => {
  const start = altitudeAt(profile, startKm)
  const end = altitudeAt(profile, endKm)
  if (start == null || end == null) return null
  let gain = 0
  let previous = start
  for (const point of profile) {
    if (point.d <= startKm || point.d >= endKm) continue
    gain += Math.max(0, point.alt - previous)
    previous = point.alt
  }
  return gain + Math.max(0, end - previous)
}

/** Peak of a trailing-window average whose window lies fully inside [t0, t1]. */
export const peakOver = (
  activity: StravaActivityDetail,
  field: 'power30sWatts' | 'power5mWatts',
  windowS: number,
  t0: number,
  t1: number,
): number | null => {
  let peak: number | null = null
  for (const point of activity.cyclingPowerTrace?.points ?? []) {
    const value = point[field]
    if (!finite(value) || point.elapsedS < t0 + windowS || point.elapsedS > t1) continue
    if (peak == null || value > peak) peak = value
  }
  return peak
}

export const nativeFtpWatts = (activity: StravaActivityDetail): number | null => {
  const metrics = activity.wahoo?.metrics
  return metrics && positive(metrics.normalizedPower) && positive(metrics.intensityFactor)
    ? metrics.normalizedPower / metrics.intensityFactor
    : null
}

export const outputOver = (
  activity: StravaActivityDetail,
  route: readonly RouteSample[],
  power30s: readonly TimedValue[],
  t0: number,
  t1: number,
): number | null => {
  if (activity.sport === 'bike') {
    if (power30s.length >= 2) return normalizedPowerOver(power30s, t0, t1)
    return routeHasPower(activity)
      ? meanOver(routePowerSamples(activity, false), t0, t1, 0.5)
      : null
  }
  const d0 = distanceAtTime(route, t0)
  const d1 = distanceAtTime(route, t1)
  return d0 != null && d1 != null && t1 > t0 && d1 > d0 ? ((d1 - d0) * 1_000) / (t1 - t0) : null
}

/** First-half versus second-half output per heartbeat over [t0, t1], in percent. */
export const decouplingOver = (
  activity: StravaActivityDetail,
  route: readonly RouteSample[],
  power30s: readonly TimedValue[],
  t0: number,
  t1: number,
): number | null => {
  if (activity.sport === 'swim' || t1 - t0 < DECOUPLING_MIN_S) return null
  const mid = (t0 + t1) / 2
  const heartRate = heartRateSamples(activity, route)
  const hr1 = meanOver(heartRate, t0, mid, 0.5)
  const hr2 = meanOver(heartRate, mid, t1, 0.5)
  const out1 = outputOver(activity, route, power30s, t0, mid)
  const out2 = outputOver(activity, route, power30s, mid, t1)
  if (!positive(hr1) || !positive(hr2) || !positive(out1) || !positive(out2)) return null
  const ef1 = out1 / hr1
  const ef2 = out2 / hr2
  return ((ef1 - ef2) / ef1) * 100
}

export const efficiencyFactor = (
  activity: StravaActivityDetail,
  normalizedPower: number | null | undefined,
  speedKph: number | null | undefined,
  heartRate: number | null | undefined,
): number | null => {
  if (!positive(heartRate)) return null
  if (activity.sport === 'bike')
    return positive(normalizedPower) ? normalizedPower / heartRate : null
  if (activity.sport === 'swim') return null
  return positive(speedKph) ? (speedKph * 1_000) / 60 / heartRate : null
}

// Body weight is the Garmin weigh-in nearest the ride, carried on the bike best-efforts block.
export const setPowerToWeight = (
  activity: StravaActivityDetail,
  stats: ActivityComparisonStats,
  set: StatWriter,
): void => {
  const weightKg = activity.bestEfforts?.weightKg
  if (!positive(weightKg)) return
  set('body-weight', weightKg, 'garmin')
  const normalizedPower = stats['normalized-power']
  if (normalizedPower)
    set('watts-per-kg', normalizedPower.value / weightKg, 'garden', normalizedPower.estimate)
}

export type StatWriter = (
  key: ActivityComparisonStatKey,
  value: number | null | undefined,
  source: ActivityComparisonStatSource,
  estimate?: boolean,
  right?: number | null,
) => void

export const statWriter = (stats: ActivityComparisonStats): StatWriter => {
  return (key, value, source, estimate = false, right) => {
    if (!finite(value)) return
    stats[key] = finite(right) ? { value, right, source, estimate } : { value, source, estimate }
  }
}
