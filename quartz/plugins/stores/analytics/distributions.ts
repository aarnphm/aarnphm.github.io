import { runGradeFactors } from '../../../util/run-grade'
import { runPaceZoneReference } from '../../../util/run-pace-zones'
import {
  matchCoreBodyTemperatureActivity,
  type CoreBodyTemperatureCache,
} from '../core-body-temperature'
import { matchGarminActivity, type GarminCache, type GarminStreams } from '../garmin'
import {
  prefersActivityDeviceThermal,
  round,
  type RawStravaActivity,
  type Sport,
  type StravaActivityDetail,
  type StravaStreams,
  type StravaZones,
} from '../strava'
import {
  ATHLETE,
  clamp,
  oneHz,
  RUN_MOVING_MIN_MPS,
  SPRINT_CAP_MS,
  type Act,
  type SwimActivityMetrics,
} from './primitives'

export type DistributionPowerSource = 'device' | 'estimated'
export type DistributionCadenceUnit = 'rpm' | 'spm' | 'str/min'
export type DistributionThermalSource = 'core-app' | 'core-fit'

export interface ActivityDistributionPoint {
  id: number
  date: string
  startedAt: string
  sport: Sport
  name: string
  movingTimeS: number
  heartRateZoneSeconds: number[] | null
  powerZoneSeconds: number[] | null
  paceZoneSeconds: number[] | null
  averagePowerWatts: number | null
  powerSource: DistributionPowerSource | null
  cadence: number | null
  cadenceUnit: DistributionCadenceUnit
  skinTemperatureC: number | null
  heatStrainIndex: number | null
  skinThermalSource: DistributionThermalSource | null
  heatStrainThermalSource: DistributionThermalSource | null
  skinObservedSeconds: number
  heatStrainObservedSeconds: number
}

export interface DistributionsBlock {
  heartRateZoneBounds: number[]
  powerZoneBounds: number[]
  paceZoneBoundsSPerKm: number[]
  tenKmRaceTimeS: number | null
  activities: ActivityDistributionPoint[]
}

export const emptyDistributions = (): DistributionsBlock => ({
  heartRateZoneBounds: [],
  powerZoneBounds: [],
  ...runPaceZoneReference(ATHLETE.tenKmRaceTime),
  activities: [],
})

export const HEAT_STRAIN_THRESHOLD = 3

export type CoreTemperatureOrigin = 'app' | 'fit'

interface CoreHeatObservation {
  durationS: number
  hotS: number
  coreTemperatureC: number | null
  heatStrainIndex: number | null
  heatStrainObservedS: number
  skinTemperatureC: number | null
  skinObservedS: number
  origin: CoreTemperatureOrigin
}

function coreAppHeatObservation(
  activity: RawStravaActivity,
  core: CoreBodyTemperatureCache | null | undefined,
): CoreHeatObservation | null {
  const samples = matchCoreBodyTemperatureActivity(activity, core).filter(
    sample => sample.quality == null || sample.quality >= 2,
  )
  if (samples.length === 0) return null
  const activityDurationS = Math.max(
    1,
    activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime,
  )
  let observedS = 0
  let hotS = 0
  let heatStrainSeconds = 0
  let skinObservedS = 0
  let skinTemperatureSeconds = 0
  let coreObservedS = 0
  let coreTemperatureSeconds = 0
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index]
    const start = clamp(sample.elapsedS, 0, activityDurationS)
    const next = samples[index + 1]?.elapsedS ?? start + 60
    const end = clamp(Math.min(next, start + 90), start, activityDurationS)
    const durationS = end - start
    if (!(durationS > 0)) continue

    const heatStrainIndex = sample.heatStrainIndex
    if (heatStrainIndex != null && heatStrainIndex >= 0 && heatStrainIndex <= 20) {
      observedS += durationS
      heatStrainSeconds += heatStrainIndex * durationS
      if (heatStrainIndex >= HEAT_STRAIN_THRESHOLD) hotS += durationS
    }

    const coreTemperatureC = sample.coreTemperatureC
    if (coreTemperatureC != null && coreTemperatureC >= 25 && coreTemperatureC <= 45) {
      coreObservedS += durationS
      coreTemperatureSeconds += coreTemperatureC * durationS
    }

    const skinTemperatureC = sample.skinTemperatureC
    if (skinTemperatureC != null && skinTemperatureC >= 0 && skinTemperatureC <= 50) {
      skinObservedS += durationS
      skinTemperatureSeconds += skinTemperatureC * durationS
    }
  }
  if (!(observedS > 0) && !(skinObservedS > 0)) return null
  return {
    durationS: observedS,
    hotS,
    coreTemperatureC: coreObservedS > 0 ? coreTemperatureSeconds / coreObservedS : null,
    heatStrainIndex: observedS > 0 ? heatStrainSeconds / observedS : null,
    heatStrainObservedS: observedS,
    skinTemperatureC: skinObservedS > 0 ? skinTemperatureSeconds / skinObservedS : null,
    skinObservedS,
    origin: 'app',
  }
}

function coreFitHeatObservation(
  activity: RawStravaActivity,
  sport: Sport,
  garmin: GarminCache | null | undefined,
): CoreHeatObservation | null {
  const match = matchGarminActivity(activity, sport, garmin ?? null)
  const stream: GarminStreams | undefined = match ? garmin?.streams?.[match.activity.id] : undefined
  const time = stream?.time
  const heatStrain = stream?.heatStrainIndex
  const coreTemperature = stream?.coreTemperatureC
  const skinTemperature = stream?.skinTemperatureC
  if (!time || time.length < 2) return null
  const validHeatStrain = heatStrain?.length === time.length ? heatStrain : null
  const validCoreTemperature = coreTemperature?.length === time.length ? coreTemperature : null
  const validSkinTemperature = skinTemperature?.length === time.length ? skinTemperature : null
  if (!validHeatStrain && !validSkinTemperature) return null

  const activityDurationS = Math.max(
    1,
    activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime,
  )
  let observedS = 0
  let hotS = 0
  let heatStrainSeconds = 0
  let coreObservedS = 0
  let coreTemperatureSeconds = 0
  let skinObservedS = 0
  let skinTemperatureSeconds = 0
  for (let index = 0; index < time.length; index++) {
    const start = clamp(time[index], 0, activityDurationS)
    const end = clamp(time[index + 1] ?? activityDurationS, start, activityDurationS)
    const durationS = end - start
    if (!(durationS > 0)) continue

    const heatStrainIndex = validHeatStrain?.[index]
    if (
      heatStrainIndex != null &&
      Number.isFinite(heatStrainIndex) &&
      heatStrainIndex >= 0 &&
      heatStrainIndex <= 20
    ) {
      observedS += durationS
      heatStrainSeconds += heatStrainIndex * durationS
      if (heatStrainIndex >= HEAT_STRAIN_THRESHOLD) hotS += durationS
    }

    const temperatureC = validCoreTemperature?.[index]
    if (
      temperatureC != null &&
      Number.isFinite(temperatureC) &&
      temperatureC >= 25 &&
      temperatureC <= 45
    ) {
      coreObservedS += durationS
      coreTemperatureSeconds += temperatureC * durationS
    }

    const skinC = validSkinTemperature?.[index]
    if (skinC != null && Number.isFinite(skinC) && skinC >= 0 && skinC <= 50) {
      skinObservedS += durationS
      skinTemperatureSeconds += skinC * durationS
    }
  }
  if (!(observedS > 0) && !(skinObservedS > 0)) return null
  return {
    durationS: observedS,
    hotS,
    coreTemperatureC: coreObservedS > 0 ? coreTemperatureSeconds / coreObservedS : null,
    heatStrainIndex: observedS > 0 ? heatStrainSeconds / observedS : null,
    heatStrainObservedS: observedS,
    skinTemperatureC: skinObservedS > 0 ? skinTemperatureSeconds / skinObservedS : null,
    skinObservedS,
    origin: 'fit',
  }
}

export function coreHeatObservation(
  activity: RawStravaActivity,
  sport: Sport,
  core: CoreBodyTemperatureCache | null | undefined,
  garmin: GarminCache | null | undefined,
): CoreHeatObservation | null {
  const app = coreAppHeatObservation(activity, core)
  const fit = coreFitHeatObservation(activity, sport, garmin)
  const observations = prefersActivityDeviceThermal(sport) ? [fit, app] : [app, fit]
  return (
    observations.find(
      observation =>
        observation != null &&
        observation.durationS > 0 &&
        observation.coreTemperatureC != null &&
        observation.heatStrainIndex != null,
    ) ?? null
  )
}

function coreThermalObservation(
  activity: RawStravaActivity,
  sport: Sport,
  core: CoreBodyTemperatureCache | null | undefined,
  garmin: GarminCache | null | undefined,
): {
  skinTemperatureC: number | null
  skinObservedS: number
  skinSource: DistributionThermalSource | null
  heatStrainIndex: number | null
  heatStrainObservedS: number
  heatStrainSource: DistributionThermalSource | null
} | null {
  const app = coreAppHeatObservation(activity, core)
  const fit = coreFitHeatObservation(activity, sport, garmin)
  const observations = prefersActivityDeviceThermal(sport) ? [fit, app] : [app, fit]
  const skin = observations.find(observation => observation?.skinTemperatureC != null)
  const heatStrain = observations.find(observation => observation?.heatStrainIndex != null)
  if (skin?.skinTemperatureC == null && heatStrain?.heatStrainIndex == null) return null
  return {
    skinTemperatureC: skin?.skinTemperatureC ?? null,
    skinObservedS: skin?.skinObservedS ?? 0,
    skinSource:
      skin?.skinTemperatureC != null ? (skin.origin === 'app' ? 'core-app' : 'core-fit') : null,
    heatStrainIndex: heatStrain?.heatStrainIndex ?? null,
    heatStrainObservedS: heatStrain?.heatStrainObservedS ?? 0,
    heatStrainSource:
      heatStrain?.heatStrainIndex != null
        ? heatStrain.origin === 'app'
          ? 'core-app'
          : 'core-fit'
        : null,
  }
}

const RUN_PACE_ZONE_WINDOW_S = 3

function runPaceStreamTime(stream: StravaStreams, movingTimeS: number): number[] | null {
  if (stream.time?.length === stream.distance.length) {
    let previous = -Infinity
    for (const elapsedS of stream.time) {
      if (!Number.isFinite(elapsedS) || elapsedS < previous) return null
      previous = elapsedS
    }
    return stream.time
  }
  return oneHz(stream.distance.length, movingTimeS)
    ? stream.distance.map((_, index) => index)
    : null
}

function runPaceZoneSeconds(
  stream: StravaStreams | undefined,
  movingTimeS: number,
  paceZoneBoundsSPerKm: readonly number[],
): number[] | null {
  if (!stream || paceZoneBoundsSPerKm.length !== 5 || stream.distance.length < 2) return null
  const time = runPaceStreamTime(stream, movingTimeS)
  if (!time) return null
  const distance: number[] = []
  let previousDistance = 0
  for (const value of stream.distance) {
    if (!Number.isFinite(value)) return null
    previousDistance = Math.max(previousDistance, value)
    distance.push(previousDistance)
  }
  const gradeFactors = stream.altitude.length === distance.length ? runGradeFactors(stream) : null
  const seconds = Array.from({ length: paceZoneBoundsSPerKm.length + 1 }, () => 0)
  let windowStart = 0
  for (let index = 1; index < time.length; index++) {
    while (windowStart + 1 < index && time[index] - time[windowStart] > RUN_PACE_ZONE_WINDOW_S)
      windowStart++
    const intervalS = time[index] - time[index - 1]
    const elapsedS = time[index] - time[windowStart]
    const distanceM = distance[index] - distance[windowStart]
    if (!(intervalS > 0) || !(elapsedS > 0) || !(distanceM > 0)) continue
    const speedMps = distanceM / elapsedS
    const gapSpeedMps = speedMps * (gradeFactors?.[index] ?? 1)
    if (!(gapSpeedMps >= RUN_MOVING_MIN_MPS) || gapSpeedMps > SPRINT_CAP_MS.run) continue
    const paceSPerKm = 1000 / gapSpeedMps
    const zone = paceZoneBoundsSPerKm.findIndex(boundary => paceSPerKm > boundary)
    seconds[zone < 0 ? seconds.length - 1 : zone] += Math.min(intervalS, 1)
  }
  return seconds.some(value => value > 0) ? seconds.map(value => Math.round(value)) : null
}

export function buildDistributions(
  acts: readonly Act[],
  zones: StravaZones | null | undefined,
  activityDetails: Readonly<Record<string, StravaActivityDetail>> | undefined,
  streams: Readonly<Record<string, StravaStreams>> | undefined,
  swimMetrics: ReadonlyMap<number, SwimActivityMetrics>,
  core: CoreBodyTemperatureCache | null | undefined,
  garmin: GarminCache | null | undefined,
): DistributionsBlock {
  const heartRateZoneBounds = [...(zones?.hr ?? [])]
  const powerZoneBounds = [...(zones?.power ?? [])]
  const paceReference = runPaceZoneReference(ATHLETE.tenKmRaceTime)
  const activities = acts.map(({ a, sport, day }): ActivityDistributionPoint => {
    const detail = activityDetails?.[String(a.id)]
    const paceZoneSeconds =
      sport === 'run'
        ? runPaceZoneSeconds(
            streams?.[String(a.id)],
            a.movingTime,
            paceReference.paceZoneBoundsSPerKm,
          )
        : null
    const averagePower = detail?.avgWatts ?? a.averageWatts ?? null
    const averagePowerWatts =
      averagePower != null && Number.isFinite(averagePower) && averagePower > 0
        ? round(averagePower, 0)
        : null
    const rawCadence = detail?.avgCadence ?? a.averageCadence ?? null
    const cadence =
      sport === 'swim'
        ? (swimMetrics.get(a.id)?.strokeRateSpm ?? null)
        : rawCadence != null && Number.isFinite(rawCadence) && rawCadence > 0
          ? rawCadence * (sport === 'run' ? 2 : 1)
          : null
    const thermal = sport === 'swim' ? null : coreThermalObservation(a, sport, core, garmin)
    return {
      id: a.id,
      date: day,
      startedAt: a.startDate,
      sport,
      name: a.name,
      movingTimeS: a.movingTime,
      heartRateZoneSeconds: detail?.hrZones ? [...detail.hrZones] : null,
      powerZoneSeconds: sport === 'bike' && detail?.powerZones ? [...detail.powerZones] : null,
      paceZoneSeconds,
      averagePowerWatts,
      powerSource:
        averagePowerWatts == null
          ? null
          : detail?.deviceWatts === true || (detail == null && a.deviceWatts === true)
            ? 'device'
            : 'estimated',
      cadence: cadence == null ? null : round(cadence, sport === 'swim' ? 1 : 0),
      cadenceUnit: sport === 'bike' ? 'rpm' : sport === 'run' ? 'spm' : 'str/min',
      skinTemperatureC:
        thermal?.skinTemperatureC == null ? null : round(thermal.skinTemperatureC, 2),
      heatStrainIndex: thermal?.heatStrainIndex == null ? null : round(thermal.heatStrainIndex, 1),
      skinThermalSource: thermal?.skinSource ?? null,
      heatStrainThermalSource: thermal?.heatStrainSource ?? null,
      skinObservedSeconds: Math.round(thermal?.skinObservedS ?? 0),
      heatStrainObservedSeconds: Math.round(thermal?.heatStrainObservedS ?? 0),
    }
  })
  return { heartRateZoneBounds, powerZoneBounds, ...paceReference, activities }
}
