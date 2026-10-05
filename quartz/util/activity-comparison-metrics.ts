import type { ActivityHealth, StravaActivityDetail } from '../plugins/stores/strava'
import { swimLengthAverages } from './swim-metrics'

export type ActivityComparisonStatKey =
  | 'tss'
  | 'intensity-factor'
  | 'aerobic-effect'
  | 'anaerobic-effect'
  | 'exercise-load'
  | 'relative-effort'
  | 'moving-time'
  | 'elapsed-time'
  | 'distance'
  | 'elevation-gain'
  | 'grade'
  | 'work'
  | 'calories'
  | 'average-power'
  | 'normalized-power'
  | 'watts-per-kg'
  | 'variability-index'
  | 'max-power'
  | 'critical-power'
  | 'w-prime'
  | 'power-balance'
  | 'peak-5s'
  | 'peak-30s'
  | 'peak-1m'
  | 'peak-5m'
  | 'peak-20m'
  | 'peak-60m'
  | 'average-hr'
  | 'max-hr'
  | 'efficiency-factor'
  | 'decoupling'
  | 'speed'
  | 'max-speed'
  | 'cadence'
  | 'stride-length'
  | 'ground-contact-time'
  | 'vertical-oscillation'
  | 'vertical-ratio'
  | 'stroke-rate'
  | 'strokes-per-length'
  | 'torque-effectiveness'
  | 'pedal-smoothness'
  | 'average-torque'
  | 'standing-time'
  | 'temperature'
  | 'humidity'
  | 'wind'
  | 'water-temperature'
  | 'readiness'
  | 'hrv'
  | 'resting-hr'
  | 'sleep-score'
  | 'sleep-duration'
  | 'body-weight'
  | 'carbs'
  | 'carbs-per-hour'
  | 'fluid'
  | 'sweat-loss'

/**
 * `garden` marks values this site derives. `estimate` separates modelled or sparsely sampled
 * values from recorded provider measurements and exact ratios of them.
 */
export type ActivityComparisonStatSource =
  | 'wahoo'
  | 'garmin'
  | 'strava'
  | 'oura'
  | 'apple'
  | 'weather'
  | 'manual'
  | 'garden'

export interface ActivityComparisonStat {
  value: number
  /** Right-leg value for paired left/right metrics; `value` holds the left leg. */
  right?: number
  source: ActivityComparisonStatSource
  estimate: boolean
}

export type ActivityComparisonStats = Partial<
  Record<ActivityComparisonStatKey, ActivityComparisonStat>
>

export interface ActivityComparisonWindow {
  startKm: number
  endKm: number
}

export interface ActivityComparisonStatsOptions {
  window?: ActivityComparisonWindow | null
  health?: ActivityHealth | null
  athleteFtpWatts?: number | null
  /** Display cadence for the whole activity, already scaled to rpm or spm. */
  averageCadence: number | null
  /** Multiplier from recorded route cadence to display cadence. */
  cadenceScale: number
  excludeZeroPower: boolean
}

const DECOUPLING_MIN_S = 1_200
const PEAK_DURATIONS: readonly [ActivityComparisonStatKey, number][] = [
  ['peak-5s', 5],
  ['peak-30s', 30],
  ['peak-1m', 60],
  ['peak-5m', 300],
  ['peak-20m', 1_200],
  ['peak-60m', 3_600],
]

type TimedValue = { t: number; v: number | null }
type RouteSample = { t: number; d: number; alt: number; hr: number | null; cad: number | null }

const finite = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value)

const positive = (value: number | null | undefined): value is number => finite(value) && value > 0

const routeSamples = (activity: StravaActivityDetail): RouteSample[] =>
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
const swimSamples = (activity: StravaActivityDetail): RouteSample[] =>
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

const timedRoute = (activity: StravaActivityDetail): RouteSample[] => {
  const route = routeSamples(activity)
  return route.length >= 2 || activity.sport !== 'swim' ? route : swimSamples(activity)
}

/** Time of the first sample that reaches `km`, interpolated along the running distance maximum. */
const timeAtDistance = (samples: readonly RouteSample[], km: number): number | null => {
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

const distanceAtTime = (samples: readonly RouteSample[], t: number): number | null => {
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
const integrate = (
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

const meanOver = (
  samples: readonly TimedValue[],
  t0: number,
  t1: number,
  minCoverage: number,
): number | null => {
  const { total, coveredS } = integrate(samples, t0, t1)
  return coveredS > 0 && coveredS >= (t1 - t0) * minCoverage ? total / coveredS : null
}

const normalizedPowerOver = (
  power30s: readonly TimedValue[],
  t0: number,
  t1: number,
): number | null => {
  const { total, coveredS } = integrate(power30s, t0, t1, value => Math.max(0, value) ** 4)
  return coveredS >= 30 && coveredS >= (t1 - t0) * 0.25 ? (total / coveredS) ** 0.25 : null
}

type PowerTraceHoles = { riddenShare: number; watts: number }

/**
 * The 30 s trace is null for 30 s after every pause and downsampling widens those holes, so the
 * covered trace undercounts riding near stops. The activity's native moving time and work fix how
 * much hole time was ridden and at what power: a full-range selection then reproduces both totals,
 * while a stop-free stretch keeps its traced values.
 */
const powerTraceHoles = (
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

const power30sSamples = (activity: StravaActivityDetail): TimedValue[] =>
  activity.cyclingPowerTrace?.points.map(point => ({
    t: point.elapsedS,
    v: finite(point.power30sWatts) ? point.power30sWatts : null,
  })) ?? []

const routePowerSamples = (activity: StravaActivityDetail, excludeZero: boolean): TimedValue[] =>
  activity.route
    .filter(point => finite(point.elapsedS))
    .map(point => ({
      t: point.elapsedS,
      v: finite(point.w) && point.w >= 0 && !(excludeZero && point.w === 0) ? point.w : null,
    }))
    .sort((a, b) => a.t - b.t)

const routeHasPower = (activity: StravaActivityDetail): boolean =>
  activity.deviceWatts && activity.route.some(point => positive(point.w))

const heartRateSamples = (
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
const swimStrokeRateOver = (
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
const powerBalanceLeftPct = (
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

const sideMean = (values: readonly (number | null)[]): number | null => {
  let total = 0
  let count = 0
  for (const value of values)
    if (finite(value) && value >= 0 && value <= 100) {
      total += value
      count++
    }
  return count > 0 ? total / count : null
}

const elevationProfile = (activity: StravaActivityDetail): { d: number; alt: number }[] => {
  const trace = activity.cyclingPowerTrace?.points
    .filter(point => finite(point.elevationM) && finite(point.distanceKm))
    .map(point => ({ d: point.distanceKm, alt: point.elevationM as number }))
  if (trace && trace.length >= 2) return trace
  return activity.route
    .filter(point => finite(point.d) && finite(point.alt))
    .map(point => ({ d: point.d, alt: point.alt }))
    .sort((a, b) => a.d - b.d)
}

const altitudeAt = (profile: readonly { d: number; alt: number }[], km: number): number | null => {
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

const elevationGainOver = (
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
const peakOver = (
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

const nativeFtpWatts = (activity: StravaActivityDetail): number | null => {
  const metrics = activity.wahoo?.metrics
  return metrics && positive(metrics.normalizedPower) && positive(metrics.intensityFactor)
    ? metrics.normalizedPower / metrics.intensityFactor
    : null
}

const outputOver = (
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
const decouplingOver = (
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

const efficiencyFactor = (
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
const setPowerToWeight = (
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

type StatWriter = (
  key: ActivityComparisonStatKey,
  value: number | null | undefined,
  source: ActivityComparisonStatSource,
  estimate?: boolean,
  right?: number | null,
) => void

const statWriter = (stats: ActivityComparisonStats): StatWriter => {
  return (key, value, source, estimate = false, right) => {
    if (!finite(value)) return
    stats[key] = finite(right) ? { value, right, source, estimate } : { value, source, estimate }
  }
}

const wholeActivityStats = (
  activity: StravaActivityDetail,
  options: ActivityComparisonStatsOptions,
): ActivityComparisonStats => {
  const stats: ActivityComparisonStats = {}
  const set = statWriter(stats)
  const wahoo = activity.wahoo?.metrics
  const garmin = activity.garmin
  const summary = (
    field: keyof NonNullable<StravaActivityDetail['wahoo']>['summarySources'],
  ): ActivityComparisonStatSource =>
    activity.wahoo?.summarySources[field] === 'wahoo' ? 'wahoo' : 'strava'
  const powered = activity.sport !== 'swim'
  const route = timedRoute(activity)
  const power30s = power30sSamples(activity)

  if (finite(wahoo?.trainingStressScore)) set('tss', wahoo.trainingStressScore, 'wahoo')
  else set('tss', garmin?.trainingStressScore, 'garmin')
  if (finite(wahoo?.intensityFactor)) set('intensity-factor', wahoo.intensityFactor, 'wahoo')
  else if (finite(garmin?.intensityFactor))
    set('intensity-factor', garmin.intensityFactor, 'garmin')
  else set('intensity-factor', activity.calculatedIntensityFactor?.value, 'garden', true)
  if (finite(garmin?.aerobicTrainingEffect))
    set('aerobic-effect', garmin.aerobicTrainingEffect, 'garmin')
  else set('aerobic-effect', activity.calculatedTrainingEffect?.aerobic, 'garden', true)
  if (finite(garmin?.anaerobicTrainingEffect))
    set('anaerobic-effect', garmin.anaerobicTrainingEffect, 'garmin')
  else set('anaerobic-effect', activity.calculatedTrainingEffect?.anaerobic, 'garden', true)
  if (finite(garmin?.exerciseLoad)) set('exercise-load', garmin.exerciseLoad, 'garmin')
  else set('exercise-load', activity.calculatedExerciseLoad?.value, 'garden', true)
  set('relative-effort', activity.sufferScore, 'strava')

  set('moving-time', positive(activity.movingTimeS) ? activity.movingTimeS : null, 'strava')
  set('elapsed-time', positive(activity.elapsedTimeS) ? activity.elapsedTimeS : null, 'strava')
  set(
    'distance',
    positive(activity.distanceKm) ? activity.distanceKm : null,
    activity.distanceSource ?? 'strava',
  )
  if (activity.sport !== 'swim') set('elevation-gain', activity.elevationM, 'strava')
  if (powered)
    set(
      'work',
      activity.kilojoules,
      summary('kilojoules'),
      !activity.deviceWatts && summary('kilojoules') !== 'wahoo',
    )
  set('calories', activity.calories, summary('calories'))

  if (powered) {
    const filtered =
      options.excludeZeroPower &&
      activity.sport === 'bike' &&
      activity.powerWithoutZeros?.avgWatts != null
    if (activity.avgWatts != null)
      set(
        'average-power',
        activity.avgWatts,
        filtered ? 'garden' : summary('avgWatts'),
        !activity.deviceWatts && summary('avgWatts') !== 'wahoo',
      )
    else if (activity.sport === 'walk')
      set('average-power', activity.walkPower?.averageWatts, 'garden', true)

    if (positive(wahoo?.normalizedPower)) set('normalized-power', wahoo.normalizedPower, 'wahoo')
    else if (positive(garmin?.normalizedPower))
      set('normalized-power', garmin.normalizedPower, 'garmin')
    else if (activity.deviceWatts) set('normalized-power', activity.npWatts, 'strava')

    setPowerToWeight(activity, stats, set)

    if (positive(wahoo?.normalizedPower) && positive(wahoo?.avgPower))
      set('variability-index', wahoo.normalizedPower / wahoo.avgPower, 'wahoo')
    else if (positive(garmin?.normalizedPower) && positive(garmin?.avgPower))
      set('variability-index', garmin.normalizedPower / garmin.avgPower, 'garmin')
    else if (activity.deviceWatts && !filtered && positive(activity.npWatts))
      set(
        'variability-index',
        positive(activity.avgWatts) ? activity.npWatts / activity.avgWatts : null,
        'strava',
      )

    if (positive(wahoo?.maxPower)) set('max-power', wahoo.maxPower, 'wahoo')
    else if (positive(garmin?.maxPower)) set('max-power', garmin.maxPower, 'garmin')
    else if (activity.deviceWatts) set('max-power', activity.maxWatts, 'strava')

    const criticalPower = activity.activityCriticalPower
    if (criticalPower) {
      set('critical-power', criticalPower.criticalPowerWatts, 'garden', true)
      set('w-prime', criticalPower.wPrimeJoules / 1_000, 'garden', true)
    }
    if (activity.sport === 'bike')
      set('power-balance', powerBalanceLeftPct(activity), activity.wahoo ? 'wahoo' : 'garmin', true)

    const curve = (activity.powerCurve ?? [])
      .filter(point => positive(point.s) && finite(point.w) && point.w >= 0)
      .sort((a, b) => a.s - b.s)
    const lastS = curve[curve.length - 1]?.s ?? 0
    for (const [key, seconds] of PEAK_DURATIONS) {
      if (seconds > lastS) continue
      let nearest = curve[0]
      for (const point of curve)
        if (Math.abs(point.s - seconds) < Math.abs(nearest.s - seconds)) nearest = point
      if (nearest && Math.abs(nearest.s - seconds) <= Math.max(1, seconds * 0.05))
        set(key, nearest.w, 'garden')
    }
  }

  const avgHrSource = summary('avgHr')
  set('average-hr', positive(activity.avgHr) ? activity.avgHr : null, avgHrSource)
  set('max-hr', positive(activity.maxHr) ? activity.maxHr : null, summary('maxHr'))

  const speedKph =
    activity.sport === 'swim' && positive(activity.swimPaceSPer100m)
      ? 360 / activity.swimPaceSPer100m
      : positive(activity.distanceKm) && positive(activity.movingTimeS)
        ? activity.distanceKm / (activity.movingTimeS / 3_600)
        : null
  set('speed', speedKph, 'strava')
  if (activity.sport === 'bike')
    set('max-speed', positive(activity.maxSpeedKph) ? activity.maxSpeedKph : null, 'strava')

  set(
    'efficiency-factor',
    efficiencyFactor(activity, stats['normalized-power']?.value, speedKph, activity.avgHr),
    'garden',
  )
  if (route.length >= 2)
    set(
      'decoupling',
      decouplingOver(activity, route, power30s, route[0].t, route[route.length - 1].t),
      'garden',
      true,
    )

  if (activity.sport === 'swim') {
    set('stroke-rate', positive(activity.strokeRateSpm) ? activity.strokeRateSpm : null, 'apple')
    if (activity.swimLocation === 'pool')
      set(
        'strokes-per-length',
        swimLengthAverages(activity.swimIntervals)?.strokesPerLength,
        'apple',
      )
  } else
    set(
      'cadence',
      positive(options.averageCadence) ? options.averageCadence : null,
      activity.sport === 'walk' && garmin?.avgCadence != null ? 'garmin' : summary('avgCadence'),
    )
  const dynamics = activity.sport === 'run' ? garmin?.runningDynamics : null
  if (dynamics) {
    set(
      'stride-length',
      positive(dynamics.averageStrideLengthCm) ? dynamics.averageStrideLengthCm / 100 : null,
      'garmin',
    )
    set('ground-contact-time', dynamics.averageGroundContactTimeMs, 'garmin')
    set('vertical-oscillation', dynamics.averageVerticalOscillationCm, 'garmin')
    set('vertical-ratio', dynamics.averageVerticalRatioPct, 'garmin')
  }

  const pedaling = activity.sport === 'bike' ? activity.cyclingDynamics : null
  if (pedaling) {
    const pedalingSource = activity.wahoo ? 'wahoo' : 'garmin'
    set(
      'torque-effectiveness',
      sideMean(pedaling.leftTorqueEffectiveness),
      pedalingSource,
      false,
      sideMean(pedaling.rightTorqueEffectiveness),
    )
    set(
      'pedal-smoothness',
      sideMean(pedaling.leftPedalSmoothness),
      pedalingSource,
      false,
      sideMean(pedaling.rightPedalSmoothness),
    )
    const seated = pedaling.seatedTimeS ?? 0
    const standing = pedaling.standingTimeS ?? 0
    if (seated + standing > 0)
      set('standing-time', (standing / (seated + standing)) * 100, pedalingSource)
  }
  if (activity.sport === 'bike')
    set('average-torque', activity.cyclingTorque?.summary.averageTorqueNm, 'wahoo')

  if (finite(activity.ambientTemperatureC))
    set('temperature', activity.ambientTemperatureC, 'weather')
  else set('temperature', activity.deviceTemperatureC, summary('deviceTemperatureC'))
  set('humidity', activity.averageRelativeHumidityPct, 'weather')
  set('wind', activity.windKph, 'weather')
  if (activity.sport === 'swim') set('water-temperature', activity.waterTemperatureC, 'apple')

  const health = options.health
  if (health) {
    set('readiness', health.readiness, 'oura')
    set('hrv', health.hrv, 'oura')
    set('resting-hr', health.rhr, 'oura')
    set('sleep-score', health.sleepScore, 'oura')
    set('sleep-duration', positive(health.sleepDurationS) ? health.sleepDurationS : null, 'oura')
  }

  const fueling = activity.fueling
  if (fueling) {
    const fuelingSource =
      fueling.source === 'manual' ? 'manual' : fueling.source === 'wahoo' ? 'wahoo' : 'garmin'
    set('carbs', fueling.carbsConsumedG, fuelingSource)
    if (finite(fueling.carbsConsumedG) && positive(activity.movingTimeS))
      set('carbs-per-hour', fueling.carbsConsumedG / (activity.movingTimeS / 3_600), fuelingSource)
    set('fluid', fueling.fluidMl, fuelingSource)
    set('sweat-loss', fueling.sweatLossMl, fuelingSource)
  }
  return stats
}

const windowStats = (
  activity: StravaActivityDetail,
  window: ActivityComparisonWindow,
  options: ActivityComparisonStatsOptions,
): ActivityComparisonStats => {
  const stats: ActivityComparisonStats = {}
  const set = statWriter(stats)
  const route = timedRoute(activity)
  if (route.length < 2) return stats
  const firstKm = route.reduce((min, sample) => Math.min(min, sample.d), Infinity)
  const lastKm = route.reduce((max, sample) => Math.max(max, sample.d), 0)
  const startKm = Math.max(window.startKm, firstKm)
  const endKm = Math.min(window.endKm, lastKm)
  if (!(endKm - startKm > 1e-4)) return stats
  const t0 = timeAtDistance(route, startKm)
  const t1 = timeAtDistance(route, endKm)
  if (t0 == null || t1 == null || t1 <= t0) return stats
  const durationS = t1 - t0
  const distanceKm = endKm - startKm
  const speedKph = distanceKm / (durationS / 3_600)

  set('elapsed-time', durationS, 'garden')
  set('distance', distanceKm, 'garden')
  set('speed', speedKph, 'garden')
  if (activity.sport !== 'swim') {
    const profile = elevationProfile(activity)
    set('elevation-gain', elevationGainOver(profile, startKm, endKm), 'garden', true)
    const startAlt = altitudeAt(profile, startKm)
    const endAlt = altitudeAt(profile, endKm)
    if (startAlt != null && endAlt != null)
      set('grade', ((endAlt - startAlt) / (distanceKm * 1_000)) * 100, 'garden', true)
  }

  const power30s = power30sSamples(activity)
  let normalizedPower: number | null = null
  if (activity.sport !== 'swim') {
    if (power30s.length >= 2) {
      const covered = integrate(power30s, t0, t1)
      const holes = powerTraceHoles(activity, power30s)
      const riddenHoleS = holes.riddenShare * Math.max(0, durationS - covered.coveredS)
      const riddenS = covered.coveredS + riddenHoleS
      const workJ = covered.total + holes.watts * riddenHoleS
      const averagePower =
        covered.coveredS >= 30 && covered.coveredS >= durationS * 0.25 ? workJ / riddenS : null
      normalizedPower = normalizedPowerOver(power30s, t0, t1)
      set('average-power', averagePower, 'garden', true)
      set('normalized-power', normalizedPower, 'garden', true)
      setPowerToWeight(activity, stats, set)
      if (positive(averagePower) && positive(normalizedPower))
        set('variability-index', normalizedPower / averagePower, 'garden', true)
      if (averagePower != null) set('work', workJ / 1_000, 'garden', true)
      set('peak-30s', peakOver(activity, 'power30sWatts', 30, t0, t1), 'garden', true)
      set('peak-5m', peakOver(activity, 'power5mWatts', 300, t0, t1), 'garden', true)
      const ftp = nativeFtpWatts(activity) ?? activity.cyclingIntensityTrace?.ftpWatts ?? null
      const ftpWatts = positive(ftp)
        ? ftp
        : positive(options.athleteFtpWatts)
          ? options.athleteFtpWatts
          : null
      if (positive(normalizedPower) && ftpWatts != null) {
        const intensity = normalizedPower / ftpWatts
        set('intensity-factor', intensity, 'garden', true)
        set('tss', (riddenS / 3_600) * intensity * intensity * 100, 'garden', true)
      }
    } else if (routeHasPower(activity)) {
      const excludeZero = options.excludeZeroPower && activity.sport === 'bike'
      set(
        'average-power',
        meanOver(routePowerSamples(activity, excludeZero), t0, t1, 0.5),
        'garden',
        true,
      )
    }
    if (activity.sport === 'bike')
      set(
        'power-balance',
        powerBalanceLeftPct(activity, t0, t1),
        activity.wahoo ? 'wahoo' : 'garmin',
        true,
      )
  }

  const heartRate = meanOver(heartRateSamples(activity, route), t0, t1, 0.5)
  set('average-hr', heartRate, 'garden', true)
  set(
    'efficiency-factor',
    efficiencyFactor(
      activity,
      normalizedPower ?? stats['average-power']?.value,
      speedKph,
      heartRate,
    ),
    'garden',
    true,
  )
  set('decoupling', decouplingOver(activity, route, power30s, t0, t1), 'garden', true)
  if (activity.sport === 'swim')
    set('stroke-rate', swimStrokeRateOver(activity, startKm, endKm), 'garden')
  else {
    const cadence = meanOver(
      route.map(sample => ({ t: sample.t, v: sample.cad })),
      t0,
      t1,
      0.3,
    )
    set('cadence', cadence == null ? null : cadence * options.cadenceScale, 'garden', true)
  }
  return stats
}

export const activityComparisonStats = (
  activity: StravaActivityDetail,
  options: ActivityComparisonStatsOptions,
): ActivityComparisonStats =>
  options.window
    ? windowStats(activity, options.window, options)
    : wholeActivityStats(activity, options)
