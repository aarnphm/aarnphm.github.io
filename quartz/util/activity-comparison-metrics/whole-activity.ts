import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type {
  ActivityComparisonStatSource,
  ActivityComparisonStats,
  ActivityComparisonStatsOptions,
} from '../activity-comparison-metrics'
import { swimLengthAverages } from '../swim-metrics'
import {
  PEAK_DURATIONS,
  finite,
  positive,
  decouplingOver,
  efficiencyFactor,
  powerBalanceLeftPct,
  setPowerToWeight,
  sideMean,
  statWriter,
  timedRoute,
  power30sSamples,
} from './calculations'

export const wholeActivityStats = (
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

  set(
    'moving-time',
    positive(activity.movingTimeS) ? activity.movingTimeS : null,
    'strava',
    !!activity.provenance,
  )
  set('elapsed-time', positive(activity.elapsedTimeS) ? activity.elapsedTimeS : null, 'strava')
  set(
    'distance',
    positive(activity.distanceKm) ? activity.distanceKm : null,
    activity.distanceSource ?? 'strava',
    !!activity.provenance,
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
    else if (activity.sport === 'walk' || activity.sport === 'hike')
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
  set(
    'speed',
    speedKph,
    activity.provenance && activity.sport === 'swim' ? 'garmin' : 'strava',
    !!activity.provenance,
  )
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
    set(
      'stroke-rate',
      positive(activity.strokeRateSpm) ? activity.strokeRateSpm : null,
      activity.provenance ? 'garmin' : 'apple',
      !!activity.provenance,
    )
    if (activity.swimLocation === 'pool')
      set(
        'strokes-per-length',
        swimLengthAverages(activity.swimIntervals)?.strokesPerLength,
        activity.provenance ? 'garmin' : 'apple',
        !!activity.provenance,
      )
  } else
    set(
      'cadence',
      positive(options.averageCadence) ? options.averageCadence : null,
      (activity.sport === 'walk' || activity.sport === 'hike') && garmin?.avgCadence != null
        ? 'garmin'
        : summary('avgCadence'),
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
