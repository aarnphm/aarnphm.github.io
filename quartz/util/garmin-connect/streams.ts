import type { GarminStreams } from '../../plugins/stores/garmin'
import { isRecord, readNumber, readString, type UnknownRecord } from '../type-guards'
import { finite } from './scalars'

const METRIC_KEYS = {
  altitude: 'directElevation',
  cadence: 'directBikeCadence',
  distance: 'sumDistance',
  elapsedTime: 'sumElapsedDuration',
  groundContactBalanceLeftPct: 'directGroundContactBalanceLeft',
  groundContactTimeMs: 'directGroundContactTime',
  heartRate: 'directHeartRate',
  impactLoadFactor: 'directImpactLoadFactor',
  latitude: 'directLatitude',
  longitude: 'directLongitude',
  performanceCondition: 'directPerformanceCondition',
  potentialStamina: 'directPotentialStamina',
  power: 'directPower',
  rightBalance: 'directRightBalance',
  respiration: 'directRespirationRate',
  stepSpeedLossMps: 'directStepSpeedLoss',
  stepSpeedLossPct: 'directStepSpeedLossPercent',
  stamina: 'directAvailableStamina',
  strideLengthCm: 'directStrideLength',
  verticalOscillationCm: 'directVerticalOscillation',
  verticalRatioPct: 'directVerticalRatio',
}
const CORE_CONNECT_IQ_APP_ID = '6957fe68-83fe-4ed6-8613-413f70624bb5'
const CORE_DEVELOPER_FIELDS = { coreTemperatureC: 0, skinTemperatureC: 10, heatStrainIndex: 95 }
const MUSCLE_OXYGEN_METRIC_KEYS = new Set([
  'directSaturatedHemoglobinPercent',
  'directSmO2',
  'directMuscleOxygen',
])

function descriptorMetricIndex(
  detail: UnknownRecord,
  matches: (descriptor: UnknownRecord) => boolean,
): number | null {
  const descriptors = detail.metricDescriptors
  if (!Array.isArray(descriptors)) return null
  for (let i = 0; i < descriptors.length; i++) {
    const descriptor = descriptors[i]
    if (!isRecord(descriptor) || !matches(descriptor)) continue
    const metricsIndex = readNumber(descriptor, 'metricsIndex')
    return metricsIndex != null && Number.isInteger(metricsIndex) && metricsIndex >= 0
      ? metricsIndex
      : i
  }
  return null
}

const metricIndex = (detail: UnknownRecord, key: string): number | null =>
  descriptorMetricIndex(detail, descriptor => readString(descriptor, 'key') === key)

const metricIndexFrom = (detail: UnknownRecord, keys: ReadonlySet<string>): number | null =>
  descriptorMetricIndex(detail, descriptor => {
    const key = readString(descriptor, 'key')
    return key != null && keys.has(key)
  })

const coreMetricIndex = (
  detail: UnknownRecord,
  field: keyof typeof CORE_DEVELOPER_FIELDS,
): number | null =>
  descriptorMetricIndex(
    detail,
    descriptor =>
      readString(descriptor, 'appID')?.toLowerCase() === CORE_CONNECT_IQ_APP_ID &&
      readNumber(descriptor, 'developerFieldNumber') === CORE_DEVELOPER_FIELDS[field],
  )

function metricValue(row: UnknownRecord, index: number | null): number | null {
  if (index == null) return null
  const metrics = row.metrics
  if (!Array.isArray(metrics)) return null
  return finite(metrics[index])
}

function validLatLng(lat: number, lng: number): boolean {
  return lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180
}

function hasStreamData(streams: GarminStreams): boolean {
  return (
    streams.latlng.length >= 2 ||
    streams.altitude.length > 0 ||
    streams.distance.length > 0 ||
    (streams.watts?.some(value => value > 0) ?? false) ||
    (streams.rightBalance?.some(value => value >= 0 && value <= 100) ?? false) ||
    (streams.heartrate?.some(value => value > 0) ?? false) ||
    (streams.cadence?.some(value => value > 0) ?? false) ||
    (streams.stamina?.some(value => value >= 0) ?? false) ||
    (streams.potentialStamina?.some(value => value >= 0) ?? false) ||
    (streams.respiration?.some(value => value > 0) ?? false) ||
    (streams.performanceCondition?.some(value => value != null) ?? false) ||
    (streams.strideLengthCm?.some(value => value != null && value > 0) ?? false) ||
    (streams.verticalRatioPct?.some(value => value != null && value >= 0) ?? false) ||
    (streams.verticalOscillationCm?.some(value => value != null && value > 0) ?? false) ||
    (streams.groundContactBalanceLeftPct?.some(value => value != null && value >= 0) ?? false) ||
    (streams.groundContactTimeMs?.some(value => value != null && value > 0) ?? false) ||
    (streams.stepSpeedLossMps?.some(value => value != null && value >= 0) ?? false) ||
    (streams.stepSpeedLossPct?.some(value => value != null && value >= 0) ?? false) ||
    (streams.impactLoadFactor?.some(value => value != null && value >= 0) ?? false) ||
    (streams.muscleOxygenPercent?.some(value => value >= 0 && value <= 100) ?? false) ||
    (streams.heatStrainIndex?.some(value => value >= 0) ?? false) ||
    (streams.coreTemperatureC?.some(value => value > 0) ?? false) ||
    (streams.skinTemperatureC?.some(value => value > 0) ?? false)
  )
}

function polylineStreams(detail: UnknownRecord): GarminStreams | null {
  const geo = detail.geoPolylineDTO
  if (!isRecord(geo) || !Array.isArray(geo.polyline)) return null

  const streams: GarminStreams = { latlng: [], altitude: [], distance: [] }
  let distance = 0
  for (const item of geo.polyline) {
    if (!isRecord(item)) continue
    const lat = finite(item.lat)
    const lng = finite(item.lon)
    if (lat == null || lng == null || !validLatLng(lat, lng)) continue
    distance =
      finite(item.distanceInMeters) ?? distance + (finite(item.distanceFromPreviousPoint) ?? 0)
    streams.latlng.push([lat, lng])
    streams.altitude.push(finite(item.altitude) ?? 0)
    streams.distance.push(distance)
  }
  return hasStreamData(streams) ? streams : null
}

export function garminConnectStreams(detail: UnknownRecord | null): GarminStreams | null {
  if (!detail || !Array.isArray(detail.activityDetailMetrics))
    return detail ? polylineStreams(detail) : null

  const bikeCadence = metricIndex(detail, METRIC_KEYS.cadence)
  const doubleCadence = metricIndex(detail, 'directDoubleCadence')
  const cadenceScale = bikeCadence == null && doubleCadence != null ? 0.5 : 1
  const indices = {
    altitude: metricIndex(detail, METRIC_KEYS.altitude),
    cadence: bikeCadence ?? doubleCadence ?? metricIndex(detail, 'directRunCadence'),
    distance: metricIndex(detail, METRIC_KEYS.distance),
    elapsedTime: metricIndex(detail, METRIC_KEYS.elapsedTime),
    groundContactBalanceLeftPct: metricIndex(detail, METRIC_KEYS.groundContactBalanceLeftPct),
    groundContactTimeMs: metricIndex(detail, METRIC_KEYS.groundContactTimeMs),
    heartRate: metricIndex(detail, METRIC_KEYS.heartRate),
    impactLoadFactor: metricIndex(detail, METRIC_KEYS.impactLoadFactor),
    latitude: metricIndex(detail, METRIC_KEYS.latitude),
    longitude: metricIndex(detail, METRIC_KEYS.longitude),
    performanceCondition: metricIndex(detail, METRIC_KEYS.performanceCondition),
    potentialStamina: metricIndex(detail, METRIC_KEYS.potentialStamina),
    power: metricIndex(detail, METRIC_KEYS.power),
    rightBalance: metricIndex(detail, METRIC_KEYS.rightBalance),
    respiration: metricIndex(detail, METRIC_KEYS.respiration),
    stepSpeedLossMps: metricIndex(detail, METRIC_KEYS.stepSpeedLossMps),
    stepSpeedLossPct: metricIndex(detail, METRIC_KEYS.stepSpeedLossPct),
    muscleOxygenPercent: metricIndexFrom(detail, MUSCLE_OXYGEN_METRIC_KEYS),
    stamina: metricIndex(detail, METRIC_KEYS.stamina),
    strideLengthCm: metricIndex(detail, METRIC_KEYS.strideLengthCm),
    verticalOscillationCm: metricIndex(detail, METRIC_KEYS.verticalOscillationCm),
    verticalRatioPct: metricIndex(detail, METRIC_KEYS.verticalRatioPct),
    heatStrainIndex: coreMetricIndex(detail, 'heatStrainIndex'),
    coreTemperatureC: coreMetricIndex(detail, 'coreTemperatureC'),
    skinTemperatureC: coreMetricIndex(detail, 'skinTemperatureC'),
  }
  const streams: GarminStreams = {
    time: indices.elapsedTime == null ? undefined : [],
    latlng: [],
    altitude: [],
    distance: [],
    watts: [],
    rightBalance: indices.rightBalance == null ? undefined : [],
    heartrate: [],
    cadence: [],
    stamina: indices.stamina == null ? undefined : [],
    potentialStamina: indices.potentialStamina == null ? undefined : [],
    respiration: indices.respiration == null ? undefined : [],
    performanceCondition: indices.performanceCondition == null ? undefined : [],
    strideLengthCm: indices.strideLengthCm == null ? undefined : [],
    verticalRatioPct: indices.verticalRatioPct == null ? undefined : [],
    verticalOscillationCm: indices.verticalOscillationCm == null ? undefined : [],
    groundContactBalanceLeftPct: indices.groundContactBalanceLeftPct == null ? undefined : [],
    groundContactTimeMs: indices.groundContactTimeMs == null ? undefined : [],
    stepSpeedLossMps: indices.stepSpeedLossMps == null ? undefined : [],
    stepSpeedLossPct: indices.stepSpeedLossPct == null ? undefined : [],
    impactLoadFactor: indices.impactLoadFactor == null ? undefined : [],
    muscleOxygenPercent: indices.muscleOxygenPercent == null ? undefined : [],
    heatStrainIndex: indices.heatStrainIndex == null ? undefined : [],
    coreTemperatureC: indices.coreTemperatureC == null ? undefined : [],
    skinTemperatureC: indices.skinTemperatureC == null ? undefined : [],
  }

  const hasLocationMetrics = indices.latitude != null && indices.longitude != null
  let lastDistance = 0
  let lastElapsedTime = 0
  for (const item of detail.activityDetailMetrics) {
    if (!isRecord(item)) continue
    const lat = metricValue(item, indices.latitude)
    const lng = metricValue(item, indices.longitude)
    const hasLocation = lat != null && lng != null && validLatLng(lat, lng)
    if (hasLocationMetrics && !hasLocation) continue

    const distance = metricValue(item, indices.distance)
    if (distance != null) lastDistance = distance
    const elapsedTime = metricValue(item, indices.elapsedTime)
    if (elapsedTime != null) lastElapsedTime = Math.max(lastElapsedTime, elapsedTime)
    streams.time?.push(lastElapsedTime)
    if (lat != null && lng != null && validLatLng(lat, lng)) streams.latlng.push([lat, lng])
    streams.altitude.push(metricValue(item, indices.altitude) ?? 0)
    streams.distance.push(lastDistance)
    streams.watts?.push(metricValue(item, indices.power) ?? 0)
    streams.rightBalance?.push(metricValue(item, indices.rightBalance) ?? -1)
    streams.heartrate?.push(metricValue(item, indices.heartRate) ?? 0)
    streams.cadence?.push((metricValue(item, indices.cadence) ?? 0) * cadenceScale)
    streams.stamina?.push(metricValue(item, indices.stamina) ?? -1)
    streams.potentialStamina?.push(metricValue(item, indices.potentialStamina) ?? -1)
    streams.respiration?.push(metricValue(item, indices.respiration) ?? 0)
    streams.performanceCondition?.push(metricValue(item, indices.performanceCondition))
    streams.strideLengthCm?.push(metricValue(item, indices.strideLengthCm))
    streams.verticalRatioPct?.push(metricValue(item, indices.verticalRatioPct))
    streams.verticalOscillationCm?.push(metricValue(item, indices.verticalOscillationCm))
    streams.groundContactBalanceLeftPct?.push(
      metricValue(item, indices.groundContactBalanceLeftPct),
    )
    streams.groundContactTimeMs?.push(metricValue(item, indices.groundContactTimeMs))
    streams.stepSpeedLossMps?.push(metricValue(item, indices.stepSpeedLossMps))
    streams.stepSpeedLossPct?.push(metricValue(item, indices.stepSpeedLossPct))
    streams.impactLoadFactor?.push(metricValue(item, indices.impactLoadFactor))
    streams.muscleOxygenPercent?.push(metricValue(item, indices.muscleOxygenPercent) ?? -1)
    streams.heatStrainIndex?.push(metricValue(item, indices.heatStrainIndex) ?? -1)
    streams.coreTemperatureC?.push(metricValue(item, indices.coreTemperatureC) ?? -1)
    streams.skinTemperatureC?.push(metricValue(item, indices.skinTemperatureC) ?? -1)
  }

  if (streams.latlng.length < 2) {
    const polyline = polylineStreams(detail)
    if (polyline) {
      streams.latlng = polyline.latlng
      streams.altitude = polyline.altitude
      streams.distance = polyline.distance
    }
  }

  return hasStreamData(streams) ? streams : null
}
