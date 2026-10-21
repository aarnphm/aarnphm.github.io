import type { ActivityKind, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'

export const KM_TO_MI = 0.621371
export const M_TO_FT = 3.28084
export const LB_TO_KG = 0.45359237

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export const isImperial = (presentation: TriathlonPresentation): boolean =>
  presentation.distance === 'imperial'

export const excludesZeroPower = (presentation: TriathlonPresentation): boolean =>
  presentation.powerSamples === 'exclude-zero'

export const dist = (
  presentation: TriathlonPresentation,
  km: number,
  sport: ActivityKind,
): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  return isImperial(presentation) ? `${(km * KM_TO_MI).toFixed(1)} mi` : `${km.toFixed(1)} km`
}

export const distCombined = (presentation: TriathlonPresentation, km: number): string =>
  isImperial(presentation)
    ? `${Math.round(km * KM_TO_MI).toLocaleString('en-US')} mi`
    : `${Math.round(km).toLocaleString('en-US')} km`

export const raceDistanceValue = (presentation: TriathlonPresentation, km: number): string => {
  if (!isImperial(presentation)) return String(km)
  const miles = km * KM_TO_MI
  return miles < 10 ? miles.toFixed(2) : miles.toFixed(1)
}

export const dur = (s: number): string => {
  const totalMinutes = Math.round(s / 60)
  const h = Math.floor(totalMinutes / 60)
  const m = totalMinutes % 60
  return h > 0 ? `${h}h${m.toString().padStart(2, '0')}'` : `${m}'`
}

export const clock = (s: number): string => {
  const seconds = Math.round(s)
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`
}

export const shortDate = (iso: string): string => {
  const [, m, d] = iso.split('-').map(Number)
  return `${MONTHS[(m || 1) - 1]} ${d || 1}`
}

export const prettyDate = (iso: string): string => {
  const [, m, dRaw] = iso.split('-').map(Number)
  const d = dRaw || 1
  const suffix =
    d % 100 >= 11 && d % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[d % 10] ?? 'th')
  return `${MONTHS[(m || 1) - 1]} ${d}${suffix}`
}

export const speedKph = (presentation: TriathlonPresentation, kmh: number): string =>
  isImperial(presentation) ? `${(kmh * KM_TO_MI).toFixed(1)} mph` : `${kmh.toFixed(1)} km/h`

export const rate = (
  presentation: TriathlonPresentation,
  sport: ActivityKind,
  km: number,
  s: number,
): string => {
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  if (sport === 'bike') return speedKph(presentation, km / (s / 3600))
  return isImperial(presentation) ? `${clock(s / (km * KM_TO_MI))} /mi` : `${clock(s / km)} /km`
}

export const scrubDist = (
  presentation: TriathlonPresentation,
  km: number,
  sport: ActivityKind,
): string =>
  sport === 'swim'
    ? `${Math.round(km * 1000).toLocaleString('en-US')} m`
    : isImperial(presentation)
      ? `${(km * KM_TO_MI).toFixed(2)} mi`
      : `${km.toFixed(2)} km`

export const elevationValue = (presentation: TriathlonPresentation, meters: number): number =>
  isImperial(presentation) ? meters * M_TO_FT : meters
export const temperatureValue = (presentation: TriathlonPresentation, celsius: number): number =>
  isImperial(presentation) ? (celsius * 9) / 5 + 32 : celsius
export const temperatureUnit = (presentation: TriathlonPresentation): string =>
  isImperial(presentation) ? '°F' : '°C'

export const formatTemperature = (presentation: TriathlonPresentation, celsius: number): string =>
  `${Math.round(temperatureValue(presentation, celsius))}${temperatureUnit(presentation)}`

export const formatThermalTemperature = (
  presentation: TriathlonPresentation,
  celsius: number,
): string => `${temperatureValue(presentation, celsius).toFixed(2)}${temperatureUnit(presentation)}`

export const formatRespirationRate = (breathsPerMinute: number): string =>
  `${breathsPerMinute.toFixed(1)} brpm`

export const formatMuscleOxygen = (percent: number): string => `${percent.toFixed(1)}% SmO₂`

export const formatMuscleOxygenMath = (percent: number): string =>
  `${percent.toFixed(1)}% $\\mathrm{SmO}_2$`

export const formatAltitude = (presentation: TriathlonPresentation, meters: number): string => {
  const rounded = Math.round(elevationValue(presentation, meters))
  return `${(rounded === 0 ? 0 : rounded).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft' : 'm'}`
}

export const formatElevationGain = (presentation: TriathlonPresentation, meters: number): string =>
  formatAltitude(presentation, meters)

export const formatVam = (presentation: TriathlonPresentation, metersPerHour: number): string =>
  `${Math.round(elevationValue(presentation, metersPerHour)).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft/h' : 'm/h'}`

export const formatMl = (value: number): string => {
  if (value < 1000) return `${Math.round(value)} ml`
  const liters = value / 1000
  return `${liters >= 10 ? liters.toFixed(0) : liters.toFixed(1)} L`
}

export const activityCadenceScale = (sport: ActivityKind): number =>
  sport === 'run' || sport === 'walk' ? 2 : 1

export const activityCadenceUnit = (sport: ActivityKind): 'rpm' | 'spm' =>
  sport === 'bike' ? 'rpm' : 'spm'

export const activityAverageCadence = (d: StravaActivityDetail): number | null => {
  if (d.sport === 'walk' && d.garmin?.avgCadence != null) return d.garmin.avgCadence
  return d.avgCadence == null ? null : d.avgCadence * activityCadenceScale(d.sport)
}

export const formatStrideLength = (presentation: TriathlonPresentation, meters: number): string =>
  isImperial(presentation) ? `${(meters * M_TO_FT).toFixed(2)} ft` : `${meters.toFixed(2)} m`

export const formatGroundContactTime = (milliseconds: number): string =>
  `${Math.round(milliseconds)} ms`

export const formatGroundContactTimeAverage = (milliseconds: number): string =>
  `${milliseconds.toFixed(1).replace(/\.0$/, '')} ms`

export const formatPerformanceCondition = (value: number): string =>
  `${value > 0 ? '+' : ''}${Math.round(value)}`

export const formatVerticalRatio = (value: number): string => `${value.toFixed(1)}%`

export const formatGroundContactBalance = (leftPct: number): string =>
  `${leftPct.toFixed(1)}% L / ${(100 - leftPct).toFixed(1)}% R`

export const formatStepSpeedLoss = (metersPerSecond: number): string =>
  `${(metersPerSecond * 100).toFixed(1)} cm/s`

export const formatStepSpeedLossPercent = (value: number): string => `${value.toFixed(2)}%`

export const formatImpactLoadFactor = (value: number): string => value.toFixed(2)

export const formatVerticalOscillation = (
  presentation: TriathlonPresentation,
  centimeters: number,
): string =>
  isImperial(presentation)
    ? `${(centimeters / 2.54).toFixed(1)} in`
    : `${centimeters.toFixed(1)} cm`

export const powerBalanceText = (rightPowerPct: number): string =>
  `L ${(100 - rightPowerPct).toFixed(1)}% / R ${rightPowerPct.toFixed(1)}%`

export const runPaceSeconds = (presentation: TriathlonPresentation, speedKph: number): number =>
  3600 / (speedKph * (isImperial(presentation) ? KM_TO_MI : 1))

export const positiveMetric = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

export const swimTrendNumber = (value: number): string =>
  value.toLocaleString('en-US', { maximumFractionDigits: 1 })

export const swimDistanceLabel = (distanceM: number): string =>
  `${Math.round(distanceM).toLocaleString('en-US')} m`

export const HR_ZONE_NAMES = ['recovery', 'endurance', 'tempo', 'threshold', 'anaerobic']
export const POWER_ZONE_NAMES = [
  'recovery',
  'endurance',
  'tempo',
  'threshold',
  'VO2max',
  'anaerobic',
  'neuromuscular',
]

export const zoneClock = (sec: number): string => {
  const s = Math.round(sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const x = s % 60
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${x.toString().padStart(2, '0')}`
  if (m > 0) return `${m}:${x.toString().padStart(2, '0')}`
  return `${x}s`
}

export const zoneRange = (bounds: number[], i: number): string => {
  if (i === 0) return `< ${bounds[0]}`
  if (i >= bounds.length) return `> ${bounds[bounds.length - 1]}`
  return `${bounds[i - 1] + 1}–${bounds[i]}`
}

export const dlabel = (sec: number): string => {
  const seconds = Math.max(0, Math.round(sec))
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainder = seconds % 60
    return remainder === 0 ? `${minutes}m` : `${minutes}m${remainder}s`
  }
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  if (minutes === 60) return `${hours + 1}h`
  return minutes === 0 ? `${hours}h` : `${hours}h${minutes}m`
}

export const cyclingSpeed = (presentation: TriathlonPresentation, kph: number): string =>
  isImperial(presentation) ? `${(kph * KM_TO_MI).toFixed(1)} mph` : `${kph.toFixed(1)} km/h`

export const heartRate = (bpm: number | null): string =>
  bpm == null ? '—' : `${Math.round(bpm)} bpm`

export const watts = (value: number | null): string =>
  value == null ? '—' : `${Math.round(value).toLocaleString('en-US')} W`

export const wattsPerKg = (value: number | null): string =>
  value == null ? '—' : `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} W/kg`
