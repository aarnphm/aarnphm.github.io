import type { GarminRiderPosition } from '../garmin'
import type {
  WahooActivity,
  WahooCache,
  WahooCyclingDynamics,
  WahooFitFile,
  WahooGearShift,
  WahooMetrics,
  WahooStreams,
  WahooSummitSegment,
  WahooSummary,
  WahooSweatLoss,
} from '../wahoo'
import { isRecord } from '../../../util/type-guards'
import { isActivityKind, type ActivityKind } from '../strava'

function finiteNumber(value: unknown, label: string, nullable = false): number | null {
  if (value == null && nullable) return null
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${label} must be finite`)
  return value
}

function integer(value: unknown, label: string): number {
  const parsed = finiteNumber(value, label)
  if (parsed == null || !Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${label} must be a nonnegative integer`)
  return parsed
}

function stringValue(value: unknown, label: string, nullable = false): string | null {
  if (value == null && nullable) return null
  if (typeof value !== 'string') throw new Error(`${label} must be a string`)
  return value
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} must be a boolean`)
  return value
}

function dateValue(value: unknown, label: string): string {
  const parsed = stringValue(value, label)
  if (parsed == null || !Number.isFinite(Date.parse(parsed)))
    throw new Error(`${label} must be a date`)
  return parsed
}

function parseMetrics(value: unknown, label: string): WahooMetrics {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    totalCalories: finiteNumber(value.totalCalories, `${label}.totalCalories`, true),
    avgHeartRate: finiteNumber(value.avgHeartRate, `${label}.avgHeartRate`, true),
    maxHeartRate: finiteNumber(value.maxHeartRate, `${label}.maxHeartRate`, true),
    avgPower: finiteNumber(value.avgPower, `${label}.avgPower`, true),
    normalizedPower: finiteNumber(value.normalizedPower, `${label}.normalizedPower`, true),
    maxPower: finiteNumber(value.maxPower, `${label}.maxPower`, true),
    avgCadence: finiteNumber(value.avgCadence, `${label}.avgCadence`, true),
    totalAscentM: finiteNumber(value.totalAscentM, `${label}.totalAscentM`, true),
    totalDescentM: finiteNumber(value.totalDescentM, `${label}.totalDescentM`, true),
    totalWorkKJ: finiteNumber(value.totalWorkKJ, `${label}.totalWorkKJ`, true),
    trainingStressScore: finiteNumber(
      value.trainingStressScore,
      `${label}.trainingStressScore`,
      true,
    ),
    intensityFactor: finiteNumber(value.intensityFactor, `${label}.intensityFactor`, true),
    avgSpeedMps: finiteNumber(value.avgSpeedMps, `${label}.avgSpeedMps`, true),
    maxSpeedMps: finiteNumber(value.maxSpeedMps, `${label}.maxSpeedMps`, true),
    avgTemperatureC: finiteNumber(value.avgTemperatureC, `${label}.avgTemperatureC`, true),
  }
}

function parseSummary(value: unknown, label: string): WahooSummary {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    id: integer(value.id, `${label}.id`),
    name: stringValue(value.name, `${label}.name`, true),
    timeZone: stringValue(value.timeZone, `${label}.timeZone`, true),
    manual: booleanValue(value.manual, `${label}.manual`),
    edited: booleanValue(value.edited, `${label}.edited`),
    fitnessAppId: integer(value.fitnessAppId, `${label}.fitnessAppId`),
    durationPausedS: finiteNumber(value.durationPausedS, `${label}.durationPausedS`, true),
    createdAt: value.createdAt == null ? null : dateValue(value.createdAt, `${label}.createdAt`),
    updatedAt: value.updatedAt == null ? null : dateValue(value.updatedAt, `${label}.updatedAt`),
  }
}

function parseFitFile(value: unknown, label: string): WahooFitFile {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const url = stringValue(value.url, `${label}.url`)
  const sha256 = stringValue(value.sha256, `${label}.sha256`)
  const profileVersion = stringValue(value.profileVersion, `${label}.profileVersion`)
  if (url == null || sha256 == null || profileVersion == null)
    throw new Error(`${label} is incomplete`)
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error(`${label}.sha256 is invalid`)
  return {
    url,
    sha256,
    byteLength: integer(value.byteLength, `${label}.byteLength`),
    profileVersion,
  }
}

function parseSweatLoss(value: unknown, label: string): WahooSweatLoss {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    fluidMl: finiteNumber(value.fluidMl, `${label}.fluidMl`, true),
    sodiumMg: finiteNumber(value.sodiumMg, `${label}.sodiumMg`, true),
  }
}

function parseActivity(
  value: unknown,
  label: string,
  normalizeSport: (workoutTypeId: number) => ActivityKind | null,
): WahooActivity {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const id = stringValue(value.id, `${label}.id`)
  const workoutTypeId = integer(value.workoutTypeId, `${label}.workoutTypeId`)
  const sport = value.sport == null ? null : value.sport
  if (id == null) throw new Error(`${label}.id is missing`)
  if (sport != null && !isActivityKind(sport)) throw new Error(`${label}.sport is invalid`)
  const normalizedSport = normalizeSport(workoutTypeId)
  return {
    id,
    workoutId: integer(value.workoutId, `${label}.workoutId`),
    workoutTypeId,
    workoutUpdatedAt:
      value.workoutUpdatedAt == null
        ? null
        : dateValue(value.workoutUpdatedAt, `${label}.workoutUpdatedAt`),
    name: stringValue(value.name, `${label}.name`, true),
    sport:
      sport === 'walk' && normalizedSport === 'hike' ? normalizedSport : (sport ?? normalizedSport),
    startDate: dateValue(value.startDate, `${label}.startDate`),
    startDateLocal: dateValue(value.startDateLocal, `${label}.startDateLocal`),
    distanceM: finiteNumber(value.distanceM, `${label}.distanceM`, true),
    movingTimeS: finiteNumber(value.movingTimeS, `${label}.movingTimeS`, true),
    elapsedTimeS: finiteNumber(value.elapsedTimeS, `${label}.elapsedTimeS`, true),
    sourceDevice: stringValue(value.sourceDevice, `${label}.sourceDevice`, true),
    sourceFile: parseFitFile(value.sourceFile, `${label}.sourceFile`),
    sweatLoss: parseSweatLoss(value.sweatLoss, `${label}.sweatLoss`),
    metrics: parseMetrics(value.metrics, `${label}.metrics`),
    summary: parseSummary(value.summary, `${label}.summary`),
  }
}

function nullableNumberArray(value: unknown, label: string): (number | null)[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`, true))
}

function parseLatLngArray(value: unknown, label: string): ([number, number] | null)[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item, index) => {
    if (item == null) return null
    if (!Array.isArray(item) || item.length !== 2) throw new Error(`${label}[${index}] is invalid`)
    const lat = finiteNumber(item[0], `${label}[${index}][0]`)
    const lng = finiteNumber(item[1], `${label}[${index}][1]`)
    if (lat == null || lng == null || lat < -90 || lat > 90 || lng < -180 || lng > 180)
      throw new Error(`${label}[${index}] is outside coordinate bounds`)
    return [lat, lng]
  })
}

function parseStreams(value: unknown, label: string): WahooStreams {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  if (!Array.isArray(value.timestamps) || !value.timestamps.every(item => typeof item === 'string'))
    throw new Error(`${label}.timestamps must be a string array`)
  if (!Array.isArray(value.time) || !value.time.every(item => typeof item === 'number'))
    throw new Error(`${label}.time must be a number array`)
  const streams: WahooStreams = {
    timestamps: [...value.timestamps],
    time: [...value.time],
    latlng: parseLatLngArray(value.latlng, `${label}.latlng`),
    altitude: nullableNumberArray(value.altitude, `${label}.altitude`),
    distance: nullableNumberArray(value.distance, `${label}.distance`),
    watts: nullableNumberArray(value.watts, `${label}.watts`),
    rightBalance: nullableNumberArray(value.rightBalance, `${label}.rightBalance`),
    heartrate: nullableNumberArray(value.heartrate, `${label}.heartrate`),
    cadence: nullableNumberArray(value.cadence, `${label}.cadence`),
    speed: nullableNumberArray(value.speed, `${label}.speed`),
    temperature: nullableNumberArray(value.temperature, `${label}.temperature`),
    respiration: nullableNumberArray(value.respiration, `${label}.respiration`),
    muscleOxygenPercent: nullableNumberArray(
      value.muscleOxygenPercent,
      `${label}.muscleOxygenPercent`,
    ),
    totalHemoglobinConcentration: nullableNumberArray(
      value.totalHemoglobinConcentration,
      `${label}.totalHemoglobinConcentration`,
    ),
    heatStrainIndex: nullableNumberArray(value.heatStrainIndex, `${label}.heatStrainIndex`),
    coreTemperatureC: nullableNumberArray(value.coreTemperatureC, `${label}.coreTemperatureC`),
    skinTemperatureC: nullableNumberArray(value.skinTemperatureC, `${label}.skinTemperatureC`),
    minuteVentilation: nullableNumberArray(value.minuteVentilation, `${label}.minuteVentilation`),
    tidalVolume: nullableNumberArray(value.tidalVolume, `${label}.tidalVolume`),
    fluidLossMl: nullableNumberArray(value.fluidLossMl, `${label}.fluidLossMl`),
    sodiumLossMg: nullableNumberArray(value.sodiumLossMg, `${label}.sodiumLossMg`),
  }
  const lengths = Object.values(streams).map(stream => stream.length)
  if (lengths.some(length => length !== streams.time.length))
    throw new Error(`${label} arrays must have equal lengths`)
  return streams
}

function parseGearShift(value: unknown, label: string): WahooGearShift {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    timestamp: dateValue(value.timestamp, `${label}.timestamp`),
    frontGearNum: integer(value.frontGearNum, `${label}.frontGearNum`),
    frontTeeth: integer(value.frontTeeth, `${label}.frontTeeth`),
    rearGearNum: integer(value.rearGearNum, `${label}.rearGearNum`),
    rearTeeth: integer(value.rearTeeth, `${label}.rearTeeth`),
  }
}

function numberArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`)
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`) ?? 0)
}

function parseCyclingDynamics(value: unknown, label: string): WahooCyclingDynamics {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  if (!Array.isArray(value.positionChanges))
    throw new Error(`${label}.positionChanges must be an array`)
  const positionChanges = value.positionChanges.map((item, index) => {
    const itemLabel = `${label}.positionChanges[${index}]`
    if (!isRecord(item)) throw new Error(`${itemLabel} must be an object`)
    let position: GarminRiderPosition
    if (item.position === 'seated') position = 'seated'
    else if (item.position === 'standing') position = 'standing'
    else throw new Error(`${itemLabel}.position is invalid`)
    return {
      elapsedS: finiteNumber(item.elapsedS, `${itemLabel}.elapsedS`) ?? 0,
      distanceM: finiteNumber(item.distanceM, `${itemLabel}.distanceM`) ?? 0,
      position,
    }
  })
  const dynamics: WahooCyclingDynamics = {
    time: numberArray(value.time, `${label}.time`),
    distance: numberArray(value.distance, `${label}.distance`),
    leftPedalSmoothness: nullableNumberArray(
      value.leftPedalSmoothness,
      `${label}.leftPedalSmoothness`,
    ),
    rightPedalSmoothness: nullableNumberArray(
      value.rightPedalSmoothness,
      `${label}.rightPedalSmoothness`,
    ),
    leftTorqueEffectiveness: nullableNumberArray(
      value.leftTorqueEffectiveness,
      `${label}.leftTorqueEffectiveness`,
    ),
    rightTorqueEffectiveness: nullableNumberArray(
      value.rightTorqueEffectiveness,
      `${label}.rightTorqueEffectiveness`,
    ),
    leftPowerPhaseStart: nullableNumberArray(
      value.leftPowerPhaseStart,
      `${label}.leftPowerPhaseStart`,
    ),
    leftPowerPhaseEnd: nullableNumberArray(value.leftPowerPhaseEnd, `${label}.leftPowerPhaseEnd`),
    rightPowerPhaseStart: nullableNumberArray(
      value.rightPowerPhaseStart,
      `${label}.rightPowerPhaseStart`,
    ),
    rightPowerPhaseEnd: nullableNumberArray(
      value.rightPowerPhaseEnd,
      `${label}.rightPowerPhaseEnd`,
    ),
    positionChanges,
    seatedTimeS: finiteNumber(value.seatedTimeS, `${label}.seatedTimeS`, true),
    standingTimeS: finiteNumber(value.standingTimeS, `${label}.standingTimeS`, true),
  }
  const lengths = [
    dynamics.distance,
    dynamics.leftPedalSmoothness,
    dynamics.rightPedalSmoothness,
    dynamics.leftTorqueEffectiveness,
    dynamics.rightTorqueEffectiveness,
    dynamics.leftPowerPhaseStart,
    dynamics.leftPowerPhaseEnd,
    dynamics.rightPowerPhaseStart,
    dynamics.rightPowerPhaseEnd,
  ].map(stream => stream.length)
  if (lengths.some(length => length !== dynamics.time.length))
    throw new Error(`${label} metric arrays must have equal lengths`)
  return dynamics
}

function nonnegativeNumber(value: unknown, label: string, nullable = false): number | null {
  const parsed = finiteNumber(value, label, nullable)
  if (parsed != null && parsed < 0) throw new Error(`${label} must be nonnegative`)
  return parsed
}

function parseSummitSegment(value: unknown, label: string): WahooSummitSegment {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const feature = value.feature
  if (feature !== 'summit-segment' && feature !== 'summit-freeride')
    throw new Error(`${label}.feature is invalid`)
  const uuid = stringValue(value.uuid, `${label}.uuid`)
  if (
    uuid == null ||
    (feature === 'summit-segment' && !uuid.startsWith('WAHOO_ON_ROUTE_CLIMB-')) ||
    (feature === 'summit-freeride' && !uuid.startsWith('WAHOO_OFF_ROUTE_CLIMB-'))
  )
    throw new Error(`${label}.uuid is invalid`)
  const startDate = dateValue(value.startDate, `${label}.startDate`)
  const endDate = dateValue(value.endDate, `${label}.endDate`)
  if (Date.parse(endDate) <= Date.parse(startDate))
    throw new Error(`${label}.endDate must follow startDate`)
  const distanceM = nonnegativeNumber(value.distanceM, `${label}.distanceM`)
  const durationS = nonnegativeNumber(value.durationS, `${label}.durationS`)
  if (distanceM == null || distanceM === 0 || durationS == null || durationS === 0)
    throw new Error(`${label} must have positive distance and duration`)
  return {
    feature,
    uuid,
    name: stringValue(value.name, `${label}.name`, true),
    startDate,
    endDate,
    distanceM,
    durationS,
    elevationGainM: nonnegativeNumber(value.elevationGainM, `${label}.elevationGainM`, true),
    avgGradePct: finiteNumber(value.avgGradePct, `${label}.avgGradePct`, true),
    avgSpeedMps: nonnegativeNumber(value.avgSpeedMps, `${label}.avgSpeedMps`, true),
    avgHeartRate: nonnegativeNumber(value.avgHeartRate, `${label}.avgHeartRate`, true),
    avgPower: nonnegativeNumber(value.avgPower, `${label}.avgPower`, true),
    avgCadence: nonnegativeNumber(value.avgCadence, `${label}.avgCadence`, true),
  }
}

function recordValue<T>(
  value: unknown,
  label: string,
  parse: (item: unknown, itemLabel: string) => T,
): Record<string, T> {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const entries: [string, T][] = []
  for (const [key, item] of Object.entries(value))
    entries.push([key, parse(item, `${label}.${key}`)])
  return Object.fromEntries(entries)
}

export function parseWahooCacheValue(
  value: unknown,
  supportedVersion: WahooCache['version'],
  normalizeSport: (workoutTypeId: number) => ActivityKind | null,
): WahooCache {
  if (!isRecord(value)) throw new Error('Wahoo cache must be an object')
  const version = integer(value.version, 'Wahoo cache.version')
  if (version !== supportedVersion) throw new Error(`Wahoo cache version ${version} is unsupported`)
  const activities = recordValue(value.activities, 'Wahoo cache.activities', (item, label) =>
    parseActivity(item, label, normalizeSport),
  )
  const streams = recordValue(value.streams, 'Wahoo cache.streams', parseStreams)
  const gearShifts = recordValue(value.gearShifts, 'Wahoo cache.gearShifts', (item, label) => {
    if (!Array.isArray(item)) throw new Error(`${label} must be an array`)
    return item.map((shift, index) => parseGearShift(shift, `${label}[${index}]`))
  })
  const cyclingDynamics = recordValue(
    value.cyclingDynamics,
    'Wahoo cache.cyclingDynamics',
    parseCyclingDynamics,
  )
  const summitSegments = recordValue(
    value.summitSegments,
    'Wahoo cache.summitSegments',
    (item, label) => {
      if (!Array.isArray(item)) throw new Error(`${label} must be an array`)
      return item.map((segment, index) => parseSummitSegment(segment, `${label}[${index}]`))
    },
  )
  for (const [id, activity] of Object.entries(activities)) {
    if (activity.id !== id)
      throw new Error(`Wahoo cache activity key ${id} does not match activity id`)
    if (!streams[id]) throw new Error(`Wahoo cache activity ${id} is missing streams`)
    if (!summitSegments[id])
      throw new Error(`Wahoo cache activity ${id} is missing summit segments`)
  }
  return {
    version,
    lastSync: finiteNumber(value.lastSync, 'Wahoo cache.lastSync') ?? 0,
    activities,
    streams,
    gearShifts,
    cyclingDynamics,
    summitSegments,
  }
}
