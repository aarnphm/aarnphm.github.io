import type { CyclingStaminaEstimate } from '../../../util/cycling-stamina'
import type { STAMINA_LEDGER_OFFSET_METHOD } from '../../../util/stamina-ledger'
import type { MapRoutePoint } from '../../../util/triathlon-map-route'
import type { WahooMetrics } from '../wahoo'

export type Sport = 'swim' | 'bike' | 'run'

export type ActivityKind = Sport | 'strength' | 'walk' | 'yoga' | 'treatment' | 'sauna'

export const ACTIVITY_KINDS: readonly ActivityKind[] = [
  'swim',
  'bike',
  'run',
  'strength',
  'walk',
  'yoga',
  'treatment',
  'sauna',
]
export const SPORT_ORDER: readonly Sport[] = ['swim', 'bike', 'run']

export const DAY_MS = 86_400_000
export const WINDOW_DAYS = 364
export const ROUTE_POINTS = 140

export interface StravaAuth {
  refreshToken: string
  obtainedAt: number
}

export interface RawStravaActivity {
  id: number
  name: string
  sportType: string
  trainer?: boolean
  commute?: boolean
  distance: number
  movingTime: number
  elapsedTime: number
  totalElevationGain: number
  startDate: string
  startDateLocal: string
  averageSpeed: number
  maxSpeed?: number
  averageHeartrate?: number
  maxHeartrate?: number
  averageWatts?: number
  weightedAverageWatts?: number
  maxWatts?: number
  kilojoules?: number
  deviceWatts?: boolean
  averageCadence?: number
  sufferScore?: number
  averageTemp?: number
  calories?: number
  deviceName?: string
  gearId?: string | null
}

export interface StravaGear {
  id: string
  name: string | null
  brandName: string | null
  modelName: string | null
  description?: string | null
  distanceM: number | null
}

export interface ActivityEquipment {
  id: string
  name: string | null
  source: 'strava'
}

export interface RawStravaAnalysisRange {
  id: string
  name: string
  elapsedTime: number
  movingTime: number
  startDate: string | null
  distance: number
  startIndex: number | null
  endIndex: number | null
  totalElevationGain: number | null
  averageSpeed: number | null
  averageHeartrate: number | null
  averageWatts: number | null
  averageCadence: number | null
}

export interface RawStravaRunSplit {
  split: number
  distance: number
  elapsedTime: number
  movingTime: number
  averageSpeed: number
  averageHeartrate?: number | null
  elevationDifference: number | null
  paceZone: number | null
}

export interface RawStravaActivityDetail {
  description?: string | null
  fetchedAt?: number
  calories: number | null
  laps: RawStravaAnalysisRange[]
  segmentEfforts: RawStravaAnalysisRange[]
  splitsMetric: RawStravaRunSplit[]
  splitsStandard: RawStravaRunSplit[]
}

export interface StravaStreams {
  time?: number[]
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  watts?: number[]
  heartrate?: number[]
  cadence?: number[]
}

export interface StravaZones {
  hr: number[]
  power: number[]
  ftp: number | null
}

export interface StravaRawCache {
  version?: number
  athleteId: number
  auth: StravaAuth
  lastSync: number
  lastActivityStart: number
  activities: Record<string, RawStravaActivity>
  gear?: Record<string, StravaGear>
  activityDetails?: Record<string, RawStravaActivityDetail>
  streams?: Record<string, StravaStreams>
  geo?: Record<string, string>
  zones?: StravaZones
}

export type StravaMapPoint = MapRoutePoint

export type ActivityThermalSource = 'core-app' | 'core-fit'

export interface StravaRoutePoint {
  x: number
  y: number
  d: number
  alt: number
  w: number
  hr: number
  cad: number
  rightPowerPct?: number | null
  stamina: number | null
  potentialStamina: number | null
  resp: number | null
  muscleOxygenPct?: number | null
  tempC: number | null
  heatStrainIndex: number | null
  heatStrainSource: ActivityThermalSource | null
  coreTemperatureC: number | null
  coreTemperatureSource: ActivityThermalSource | null
  skinTemperatureC: number | null
  skinTemperatureSource: ActivityThermalSource | null
  lat: number
  lng: number
  elapsedS: number
  speedKph: number
  performanceCondition?: number | null
  strideLengthM?: number | null
  verticalRatioPct?: number | null
  groundContactTimeMs?: number | null
  groundContactBalanceLeftPct?: number | null
  verticalOscillationCm?: number | null
  stepSpeedLossMps?: number | null
  stepSpeedLossPct?: number | null
  impactLoadFactor?: number | null
}

export type ActivityComputer = 'garmin' | 'wahoo'

export type ActivityDevice = 'apple-watch-ultra-3' | 'garmin-forerunner-970'

export const ACTIVITY_DEVICES: readonly ActivityDevice[] = [
  'apple-watch-ultra-3',
  'garmin-forerunner-970',
]

export type ActivityStaminaTrace =
  | { source: 'garmin'; method: 'garmin-native'; ftpWatts: null; maxHeartRateBpm: null }
  | {
      source: 'garden-estimate'
      method: CyclingStaminaEstimate['method']
      ftpWatts: number
      maxHeartRateBpm: number
    }
  | {
      source: 'garden-estimate'
      method: typeof STAMINA_LEDGER_OFFSET_METHOD
      ftpWatts: null
      maxHeartRateBpm: null
      garminOffset: number
    }

export interface ActivityHeartRate {
  avgHr: number | null
  maxHr: number | null
  stream: number[]
}

export interface ActivityHeartRateTracePoint {
  distanceKm: number
  elapsedS: number
  heartRate: number | null
  heatStrainIndex: number | null
  heatStrainSource: ActivityThermalSource | null
  coreTemperatureC: number | null
  coreTemperatureSource: ActivityThermalSource | null
  skinTemperatureC: number | null
  skinTemperatureSource: ActivityThermalSource | null
}

export interface ActivitySource {
  provider: 'strava' | 'garmin' | 'wahoo'
  activityId: string
  name: string | null
  fileName: string | null
}

export interface WahooVerification {
  activityId: string
  fitPath: string | null
  sha256: string
  sourceDevice: string | null
  startOffsetS: number
  distanceM: number | null
  metrics: WahooMetrics
  summarySources: Partial<Record<WahooSummaryDetailField, 'wahoo'>>
  streamFallback: 'strava' | 'garmin' | null
}

export const POWER_SUMMARY_FIELDS = [
  ['averageWatts', 'avgPower', 'avgWatts'],
  ['weightedAverageWatts', 'normalizedPower', 'npWatts'],
  ['maxWatts', 'maxPower', 'maxWatts'],
  ['kilojoules', 'totalWorkKJ', 'kilojoules'],
] as const

export const WAHOO_SUMMARY_FIELDS = [
  ['averageHeartrate', 'avgHeartRate', 'avgHr'],
  ['maxHeartrate', 'maxHeartRate', 'maxHr'],
  ...POWER_SUMMARY_FIELDS,
  ['averageCadence', 'avgCadence', 'avgCadence'],
  ['calories', 'totalCalories', 'calories'],
  ['averageTemp', 'avgTemperatureC', 'deviceTemperatureC'],
] as const

type WahooSummaryDetailField = (typeof WAHOO_SUMMARY_FIELDS)[number][2]
export const WAHOO_TELEMETRY_MAX_DISTANCE_S = 2.5

export function normalizeSport(sportType: string): Sport | null {
  switch (sportType) {
    case 'Run':
    case 'TrailRun':
    case 'VirtualRun':
      return 'run'
    case 'Ride':
    case 'VirtualRide':
    case 'MountainBikeRide':
    case 'GravelRide':
    case 'EBikeRide':
      return 'bike'
    case 'Swim':
    case 'OpenWaterSwim':
      return 'swim'
    default:
      return null
  }
}

export function normalizeKind(sportType: string): ActivityKind | null {
  switch (sportType) {
    case 'WeightTraining':
    case 'Workout':
    case 'Crossfit':
      return 'strength'
    case 'Walk':
    case 'Hike':
      return 'walk'
    case 'Yoga':
    case 'Pilates':
      return 'yoga'
    case 'PhysicalTherapy':
    case 'Physiotherapy':
      return 'treatment'
    default:
      return normalizeSport(sportType)
  }
}

const TREATMENT_TYPES = new Set(['PhysicalTherapy', 'Physiotherapy'])
const TREATMENT_NAME_RE = /\b(physio|physiotherapy|physical[ -]?therapy|treatment|rehab|massage)\b/i

export function isTreatment(sportType: string, name: string | null | undefined): boolean {
  return TREATMENT_TYPES.has(sportType) || TREATMENT_NAME_RE.test(name ?? '')
}

export function normalizeActivityDevice(value: string | null | undefined): ActivityDevice | null {
  if (!value) return null
  const normalized = value
    .trim()
    .toLocaleLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '-')
  if (normalized.includes('apple-watch-ultra-3') || normalized.includes('appl-watch-ultra-3'))
    return 'apple-watch-ultra-3'
  if (normalized.includes('forerunner-970')) return 'garmin-forerunner-970'
  return null
}
