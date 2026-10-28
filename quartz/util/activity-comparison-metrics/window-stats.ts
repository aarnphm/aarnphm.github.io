import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type {
  ActivityComparisonStats,
  ActivityComparisonStatsOptions,
  ActivityComparisonWindow,
} from '../activity-comparison-metrics'
import {
  altitudeAt,
  decouplingOver,
  elevationGainOver,
  elevationProfile,
  efficiencyFactor,
  heartRateSamples,
  integrate,
  meanOver,
  nativeFtpWatts,
  normalizedPowerOver,
  peakOver,
  power30sSamples,
  powerBalanceLeftPct,
  powerTraceHoles,
  positive,
  routeHasPower,
  routePowerSamples,
  setPowerToWeight,
  statWriter,
  swimStrokeRateOver,
  timedRoute,
  timeAtDistance,
} from './calculations'

export const windowStats = (
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
