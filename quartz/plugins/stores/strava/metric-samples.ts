import type { CyclingStaminaEstimate } from '../../../util/cycling-stamina'
import type { GarminActivityMatch, GarminCache } from '../garmin'
import type { WahooActivityMatch, WahooData } from '../wahoo'
import type {
  ActivityHeartRateTracePoint,
  ActivityKind,
  ActivityStaminaTrace,
  RawStravaActivity,
  StravaRoutePoint,
} from './raw-and-kinds'
import { projectWahooRespiration, type RespirationProjectionModel } from './provider-metadata'
import {
  round,
  THERMAL_SAMPLE_MAX_DISTANCE_S,
  timedNullableMetricAt,
  type TimedMetricSample,
  type TimedNullableMetricSample,
} from './stream-primitives'

export interface ActivityMetricSamples {
  deviceTemperatureC: TimedNullableMetricSample[]
  rightBalance: TimedMetricSample[]
  stamina: TimedMetricSample[]
  potentialStamina: TimedMetricSample[]
  staminaTrace: ActivityStaminaTrace | null
  respiration: TimedMetricSample[]
  performanceCondition: TimedNullableMetricSample[]
  strideLengthCm: TimedNullableMetricSample[]
  verticalRatioPct: TimedNullableMetricSample[]
  verticalOscillationCm: TimedNullableMetricSample[]
  groundContactBalanceLeftPct: TimedNullableMetricSample[]
  groundContactTimeMs: TimedNullableMetricSample[]
  stepSpeedLossMps: TimedNullableMetricSample[]
  stepSpeedLossPct: TimedNullableMetricSample[]
  impactLoadFactor: TimedNullableMetricSample[]
  muscleOxygenPercent: TimedMetricSample[]
  heatStrainIndex: TimedNullableMetricSample[]
  coreTemperatureC: TimedNullableMetricSample[]
  skinTemperatureC: TimedNullableMetricSample[]
}

export function nativeThermalAt(
  samples: ActivityMetricSamples,
  elapsedS: number,
): Pick<
  ActivityHeartRateTracePoint,
  | 'heatStrainIndex'
  | 'heatStrainSource'
  | 'coreTemperatureC'
  | 'coreTemperatureSource'
  | 'skinTemperatureC'
  | 'skinTemperatureSource'
> {
  const heatStrainIndex = timedNullableMetricAt(
    samples.heatStrainIndex,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  const coreTemperatureC = timedNullableMetricAt(
    samples.coreTemperatureC,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  const skinTemperatureC = timedNullableMetricAt(
    samples.skinTemperatureC,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  return {
    heatStrainIndex: heatStrainIndex == null ? null : round(heatStrainIndex, 1),
    heatStrainSource: heatStrainIndex == null ? null : 'core-fit',
    coreTemperatureC: coreTemperatureC == null ? null : round(coreTemperatureC, 2),
    coreTemperatureSource: coreTemperatureC == null ? null : 'core-fit',
    skinTemperatureC: skinTemperatureC == null ? null : round(skinTemperatureC, 2),
    skinTemperatureSource: skinTemperatureC == null ? null : 'core-fit',
  }
}

const RUNNING_DYNAMICS_SAMPLE_MAX_DISTANCE_S = 15

export function nativeRunDynamicsAt(
  samples: ActivityMetricSamples,
  elapsedS: number,
): Pick<
  StravaRoutePoint,
  | 'strideLengthM'
  | 'verticalRatioPct'
  | 'verticalOscillationCm'
  | 'groundContactBalanceLeftPct'
  | 'groundContactTimeMs'
  | 'stepSpeedLossMps'
  | 'stepSpeedLossPct'
  | 'impactLoadFactor'
> {
  const at = (values: TimedNullableMetricSample[]): number | null =>
    timedNullableMetricAt(values, elapsedS, RUNNING_DYNAMICS_SAMPLE_MAX_DISTANCE_S)
  const strideLengthCm = at(samples.strideLengthCm)
  const verticalRatioPct = at(samples.verticalRatioPct)
  const verticalOscillationCm = at(samples.verticalOscillationCm)
  const groundContactBalanceLeftPct = at(samples.groundContactBalanceLeftPct)
  const groundContactTimeMs = at(samples.groundContactTimeMs)
  const stepSpeedLossMps = at(samples.stepSpeedLossMps)
  const stepSpeedLossPct = at(samples.stepSpeedLossPct)
  const impactLoadFactor = at(samples.impactLoadFactor)
  return {
    strideLengthM: strideLengthCm == null ? null : round(strideLengthCm / 100, 3),
    verticalRatioPct: verticalRatioPct == null ? null : round(verticalRatioPct, 1),
    verticalOscillationCm: verticalOscillationCm == null ? null : round(verticalOscillationCm, 1),
    groundContactBalanceLeftPct:
      groundContactBalanceLeftPct == null ? null : round(groundContactBalanceLeftPct, 1),
    groundContactTimeMs: groundContactTimeMs == null ? null : round(groundContactTimeMs, 1),
    stepSpeedLossMps: stepSpeedLossMps == null ? null : round(stepSpeedLossMps, 4),
    stepSpeedLossPct: stepSpeedLossPct == null ? null : round(stepSpeedLossPct, 2),
    impactLoadFactor: impactLoadFactor == null ? null : round(impactLoadFactor, 2),
  }
}

type GarminMetricStreamKey =
  | 'rightBalance'
  | 'stamina'
  | 'potentialStamina'
  | 'respiration'
  | 'muscleOxygenPercent'
  | 'heatStrainIndex'
  | 'coreTemperatureC'
  | 'skinTemperatureC'

type GarminNullableMetricStreamKey =
  | 'performanceCondition'
  | 'strideLengthCm'
  | 'verticalRatioPct'
  | 'verticalOscillationCm'
  | 'groundContactBalanceLeftPct'
  | 'groundContactTimeMs'
  | 'stepSpeedLossMps'
  | 'stepSpeedLossPct'
  | 'impactLoadFactor'

function activityGarminMetricSamples(
  activity: RawStravaActivity,
  match: GarminActivityMatch | null,
  garmin: GarminCache | null,
): ActivityMetricSamples {
  const empty = (): ActivityMetricSamples => ({
    deviceTemperatureC: [],
    rightBalance: [],
    stamina: [],
    potentialStamina: [],
    staminaTrace: null,
    respiration: [],
    performanceCondition: [],
    strideLengthCm: [],
    verticalRatioPct: [],
    verticalOscillationCm: [],
    groundContactBalanceLeftPct: [],
    groundContactTimeMs: [],
    stepSpeedLossMps: [],
    stepSpeedLossPct: [],
    impactLoadFactor: [],
    muscleOxygenPercent: [],
    heatStrainIndex: [],
    coreTemperatureC: [],
    skinTemperatureC: [],
  })
  if (!match) return empty()
  const stream = garmin?.streams?.[match.activity.id]
  const time = stream?.time
  if (!time) return empty()
  const startOffsetS =
    (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return empty()

  const collect = (
    key: GarminMetricStreamKey,
    valid: (value: number) => boolean,
  ): TimedMetricSample[] => {
    const values = stream?.[key]
    if (!values || time.length !== values.length) return []
    const samples: TimedMetricSample[] = []
    for (let index = 0; index < time.length; index++) {
      if (!Number.isFinite(time[index]) || !Number.isFinite(values[index])) continue
      if (!valid(values[index])) continue
      samples.push({ elapsedS: time[index] + startOffsetS, value: values[index] })
    }
    return samples.sort((left, right) => left.elapsedS - right.elapsedS)
  }

  const collectNullable = (
    key:
      | GarminNullableMetricStreamKey
      | 'heatStrainIndex'
      | 'coreTemperatureC'
      | 'skinTemperatureC',
    valid: (value: number) => boolean,
  ): TimedNullableMetricSample[] => {
    const values = stream?.[key]
    if (!values || time.length !== values.length) return []
    const samples: TimedNullableMetricSample[] = []
    for (let index = 0; index < time.length; index++) {
      if (!Number.isFinite(time[index])) continue
      const value = values[index]
      samples.push({
        elapsedS: time[index] + startOffsetS,
        value: typeof value === 'number' && Number.isFinite(value) && valid(value) ? value : null,
      })
    }
    return samples.sort((left, right) => left.elapsedS - right.elapsedS)
  }

  const stamina = collect('stamina', value => value >= 0 && value <= 100)
  const potentialStamina = collect('potentialStamina', value => value >= 0 && value <= 100)
  return {
    rightBalance: collect('rightBalance', value => value >= 0 && value <= 100),
    stamina,
    potentialStamina,
    staminaTrace:
      stamina.length >= 2 && potentialStamina.length >= 2
        ? { source: 'garmin', method: 'garmin-native', ftpWatts: null, maxHeartRateBpm: null }
        : null,
    respiration: collect('respiration', value => value > 0),
    performanceCondition: collectNullable(
      'performanceCondition',
      value => value >= -20 && value <= 20,
    ),
    strideLengthCm: collectNullable('strideLengthCm', value => value >= 20 && value <= 300),
    verticalRatioPct: collectNullable('verticalRatioPct', value => value >= 0 && value <= 50),
    verticalOscillationCm: collectNullable(
      'verticalOscillationCm',
      value => value >= 1 && value <= 30,
    ),
    groundContactBalanceLeftPct: collectNullable(
      'groundContactBalanceLeftPct',
      value => value >= 0 && value <= 100,
    ),
    groundContactTimeMs: collectNullable(
      'groundContactTimeMs',
      value => value >= 50 && value <= 1_000,
    ),
    stepSpeedLossMps: collectNullable('stepSpeedLossMps', value => value >= 0 && value <= 5),
    stepSpeedLossPct: collectNullable('stepSpeedLossPct', value => value >= 0 && value <= 100),
    impactLoadFactor: collectNullable('impactLoadFactor', value => value >= 0 && value <= 10),
    deviceTemperatureC: [],
    muscleOxygenPercent: collect('muscleOxygenPercent', value => value >= 0 && value <= 100),
    heatStrainIndex: collectNullable('heatStrainIndex', value => value >= 0 && value <= 20),
    coreTemperatureC: collectNullable('coreTemperatureC', value => value >= 25 && value <= 45),
    skinTemperatureC: collectNullable('skinTemperatureC', value => value >= 0 && value <= 50),
  }
}

export function activityMetricSamples(
  activity: RawStravaActivity,
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooData | null,
  wahooStamina: CyclingStaminaEstimate | null,
  respirationProjection: RespirationProjectionModel | null,
): ActivityMetricSamples {
  const samples = activityGarminMetricSamples(activity, garminMatch, garmin)
  if (!wahooMatch) return samples
  const stream = wahoo?.streams[wahooMatch.activity.id]
  if (!stream || stream.time.length === 0) return samples
  const startOffsetS =
    (Date.parse(wahooMatch.activity.startDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return samples
  const collect = (
    values: readonly (number | null)[],
    valid: (value: number) => boolean,
  ): TimedMetricSample[] => {
    if (values.length !== stream.time.length) return []
    const collected: TimedMetricSample[] = []
    for (let index = 0; index < stream.time.length; index++) {
      const elapsedS = stream.time[index]
      const value = values[index]
      if (!Number.isFinite(elapsedS) || typeof value !== 'number' || !Number.isFinite(value))
        continue
      if (!valid(value)) continue
      collected.push({ elapsedS: elapsedS + startOffsetS, value })
    }
    return collected.sort((left, right) => left.elapsedS - right.elapsedS)
  }
  const collectThermal = (
    values: readonly (number | null)[],
    valid: (value: number) => boolean,
  ): TimedNullableMetricSample[] => {
    if (values.length !== stream.time.length) return []
    const collected: TimedNullableMetricSample[] = []
    for (let index = 0; index < stream.time.length; index++) {
      const elapsedS = stream.time[index]
      const value = values[index]
      if (!Number.isFinite(elapsedS)) continue
      collected.push({
        elapsedS: elapsedS + startOffsetS,
        value: typeof value === 'number' && Number.isFinite(value) && valid(value) ? value : null,
      })
    }
    return collected.sort((left, right) => left.elapsedS - right.elapsedS)
  }
  const directWahooRespiration = collect(stream.respiration, value => value > 0)
  const fitStamina = collect(
    stream.garminStamina?.current ?? [],
    value => value >= 0 && value <= 100,
  )
  const fitPotentialStamina = collect(
    stream.garminStamina?.potential ?? [],
    value => value >= 0 && value <= 100,
  )
  if (fitStamina.length >= 2 && fitPotentialStamina.length >= 2) {
    samples.stamina = fitStamina
    samples.potentialStamina = fitPotentialStamina
    samples.staminaTrace = {
      source: 'garmin',
      method: 'garmin-native',
      ftpWatts: null,
      maxHeartRateBpm: null,
    }
  }
  const select = <T extends TimedNullableMetricSample>(
    garminSamples: T[],
    wahooSamples: T[],
  ): T[] => {
    if (!wahooSamples.some(sample => sample.value != null)) return garminSamples
    return wahooSamples.map(sample => ({
      ...sample,
      value:
        sample.value ??
        timedNullableMetricAt(garminSamples, sample.elapsedS, THERMAL_SAMPLE_MAX_DISTANCE_S),
    }))
  }
  const wahooRespiration =
    directWahooRespiration.length > 0 ||
    samples.respiration.length > 0 ||
    sport !== 'bike' ||
    !respirationProjection
      ? directWahooRespiration
      : projectWahooRespiration(stream, startOffsetS, respirationProjection)
  const estimatedStamina =
    samples.staminaTrace == null && wahooStamina
      ? {
          stamina: wahooStamina.samples.map(sample => ({
            elapsedS: sample.elapsedS + startOffsetS,
            value: sample.stamina,
          })),
          potentialStamina: wahooStamina.samples.map(sample => ({
            elapsedS: sample.elapsedS + startOffsetS,
            value: sample.potentialStamina,
          })),
          staminaTrace: {
            source: 'garden-estimate',
            method: wahooStamina.method,
            ftpWatts: wahooStamina.ftpWatts,
            maxHeartRateBpm: wahooStamina.maxHeartRateBpm,
          } satisfies ActivityStaminaTrace,
        }
      : null
  return {
    ...samples,
    deviceTemperatureC: collectThermal(stream.temperature, value => value >= -80 && value <= 80),
    stamina: estimatedStamina?.stamina ?? samples.stamina,
    potentialStamina: estimatedStamina?.potentialStamina ?? samples.potentialStamina,
    staminaTrace: estimatedStamina?.staminaTrace ?? samples.staminaTrace,
    rightBalance: select(
      samples.rightBalance,
      collect(stream.rightBalance, value => value >= 0 && value <= 100),
    ),
    respiration: select(samples.respiration, wahooRespiration),
    muscleOxygenPercent: select(
      samples.muscleOxygenPercent,
      collect(stream.muscleOxygenPercent, value => value >= 0 && value <= 100),
    ),
    heatStrainIndex: select(
      samples.heatStrainIndex,
      collectThermal(stream.heatStrainIndex, value => value >= 0 && value <= 20),
    ),
    coreTemperatureC: select(
      samples.coreTemperatureC,
      collectThermal(stream.coreTemperatureC, value => value >= 25 && value <= 45),
    ),
    skinTemperatureC: select(
      samples.skinTemperatureC,
      collectThermal(stream.skinTemperatureC, value => value >= 0 && value <= 50),
    ),
  }
}
