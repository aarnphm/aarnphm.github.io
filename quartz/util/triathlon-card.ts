import type { TriathlonDailyAnalytics, TriathlonDayAnalytics } from './triathlon-day-analytics'
import type { Locale, TriathlonPresentation } from './triathlon-presentation'
import { STROKE_LABEL, SWIM_STROKES, type SwimStroke } from '../plugins/stores/apple'
import { criticalPowerCurve, type CriticalPowerEstimate } from '../plugins/stores/critical-power'
import {
  SPORT_ICON,
  type ActivityAnalysisKind,
  type ActivityAnalysisRange,
  type ActivityCyclingDynamics,
  type ActivityGearShift,
  type ActivityHealth,
  type ActivityKind,
  type PowerCurvePoint,
  type StravaActivityDetail,
  type StravaZones,
  type SwimActivityInterval,
  type SwimTrendPoint,
} from '../plugins/stores/strava'
import { RUN_PACE_ZONE_NAMES, runPaceZoneRange } from './run-pace-zones'
import {
  swimLengthAverages,
  swimLengthMetrics,
  swimPaceSeconds,
  swimStrokeRate,
  type SwimChartMetric,
} from './swim-metrics'
import { triathlonActivityAnchor } from './triathlon-date-route'
import {
  criticalPowerEvidenceText,
  criticalPowerSummaryText,
  swimActivityHeaderValue,
  triText,
} from './triathlon-i18n'
import { powerCurveActivityLinkAttributes } from './triathlon-power-activity'
import {
  triathlonTraceEnabled,
  triathlonTraceName,
  type TriathlonTraceSettings,
} from './triathlon-trace-settings'

export interface TriNodeFactory<N> {
  presentation: TriathlonPresentation
  el: (tag: string, cls?: string, text?: string, attrs?: Record<string, string>) => N
  svg: (tag: string, attrs: Record<string, string | number>) => N
  add: (parent: N, ...children: N[]) => void
}

export type DayCardExtras = {
  location?: string
  event?: string
  sport?: ActivityKind
  activityId?: string
  excludedActivityIds?: readonly string[]
  settings?: TriathlonTraceSettings
  analytics?: boolean
  expanded?: boolean
  embedded?: boolean
  dateHref?: string
  dayRouteHref?: string
}

export type DayCardPayload = {
  details: Record<string, StravaActivityDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  dailyAnalytics?: TriathlonDailyAnalytics
}

export const dayCardActivitiesExpanded = (extras: DayCardExtras): boolean =>
  extras.settings?.expanded ?? Boolean(extras.sport || extras.activityId || extras.expanded)

export type ActivityFueling = NonNullable<StravaActivityDetail['fueling']>
export type ActivityStrength = NonNullable<StravaActivityDetail['strength']>
export type ActivityStrengthExercise = ActivityStrength['exercises'][number]
export type ActivityStrengthSet = ActivityStrengthExercise['sets'][number]

export const KM_TO_MI = 0.621371
export const M_TO_FT = 3.28084
const LB_TO_KG = 0.45359237

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const isImperial = (presentation: TriathlonPresentation): boolean =>
  presentation.distance === 'imperial'

const excludesZeroPower = (presentation: TriathlonPresentation): boolean =>
  presentation.powerSamples === 'exclude-zero'

export const powerViewActivity = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
): StravaActivityDetail => {
  const filtered = activity.powerWithoutZeros
  if (!excludesZeroPower(presentation) || activity.sport !== 'bike' || !filtered) return activity
  return {
    ...activity,
    avgWatts: filtered.avgWatts ?? activity.avgWatts,
    powerZones: filtered.powerZones ?? activity.powerZones,
    powerHist: filtered.powerHist ?? activity.powerHist,
  }
}

export const interpolatePositiveMetricSeries = (
  route: StravaActivityDetail['route'],
  pick: (point: StravaActivityDetail['route'][number], index: number) => number | null | undefined,
): (number | null)[] => {
  const values = route.map((point, index) => {
    const value = pick(point, index)
    return value != null && Number.isFinite(value) && value > 0 ? value : null
  })
  const previous = Array.from({ length: values.length }, () => -1)
  const next = Array.from({ length: values.length }, () => -1)
  let measured = -1
  for (let index = 0; index < values.length; index++) {
    previous[index] = measured
    if (values[index] != null) measured = index
  }
  measured = -1
  for (let index = values.length - 1; index >= 0; index--) {
    next[index] = measured
    if (values[index] != null) measured = index
  }
  return values.map((value, index) => {
    if (value != null) return value
    const left = previous[index]
    const right = next[index]
    if (left < 0) return right < 0 ? null : values[right]
    if (right < 0) return values[left]
    const leftValue = values[left]
    const rightValue = values[right]
    if (leftValue == null || rightValue == null) return leftValue ?? rightValue
    const distanceSpan = route[right].d - route[left].d
    const fraction =
      distanceSpan > 0
        ? (route[index].d - route[left].d) / distanceSpan
        : (index - left) / (right - left)
    return leftValue + (rightValue - leftValue) * Math.max(0, Math.min(1, fraction))
  })
}

export const parseExcludedActivityIds = (value: string | undefined): string[] => {
  const filter = value?.startsWith('filter=') ? value.slice('filter='.length) : value
  if (!filter || !/^\d+(?:&\d+)*$/.test(filter)) return []
  return [...new Set(filter.split('&'))]
}

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

const elevationValue = (presentation: TriathlonPresentation, meters: number): number =>
  isImperial(presentation) ? meters * M_TO_FT : meters
const temperatureValue = (presentation: TriathlonPresentation, celsius: number): number =>
  isImperial(presentation) ? (celsius * 9) / 5 + 32 : celsius
const temperatureUnit = (presentation: TriathlonPresentation): string =>
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

export const formatAltitude = (presentation: TriathlonPresentation, meters: number): string => {
  const rounded = Math.round(elevationValue(presentation, meters))
  return `${(rounded === 0 ? 0 : rounded).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft' : 'm'}`
}

export const formatElevationGain = (presentation: TriathlonPresentation, meters: number): string =>
  formatAltitude(presentation, meters)

export const formatVam = (presentation: TriathlonPresentation, metersPerHour: number): string =>
  `${Math.round(elevationValue(presentation, metersPerHour)).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft/h' : 'm/h'}`

export const gradeAt = (route: StravaActivityDetail['route'], i: number): number => {
  const j0 = Math.max(0, i - 2)
  const j1 = Math.min(route.length - 1, i + 2)
  const dKm = route[j1].d - route[j0].d
  return dKm > 0 ? ((route[j1].alt - route[j0].alt) / (dKm * 1000)) * 100 : 0
}

export const formatMl = (value: number): string => {
  if (value < 1000) return `${Math.round(value)} ml`
  const liters = value / 1000
  return `${liters >= 10 ? liters.toFixed(0) : liters.toFixed(1)} L`
}

export const formatFuelingSource = (fueling: ActivityFueling): string => {
  if (fueling.source === 'manual') return 'manual'
  if (fueling.source === 'garmin+wahoo') return 'Garmin + Wahoo'
  const value = fueling.sourceDevice
  const clean = value?.trim()
  if (fueling.source === 'wahoo') {
    if (!clean) return 'Wahoo'
    return clean.toLowerCase().includes('wahoo') ? clean : `Wahoo ${clean}`
  }
  if (!clean) return 'Garmin'
  return clean.toLowerCase().includes('garmin') ? clean : `Garmin ${clean}`
}

export const recoveryRows = (h: ActivityHealth): [string, string][] => {
  const rows: [string, string][] = []
  if (h.readiness != null) rows.push(['readiness', `${h.readiness}`])
  if (h.sleepScore != null) rows.push(['sleep', `${h.sleepScore}`])
  if (h.sleepDurationS != null) rows.push(['slept', dur(h.sleepDurationS)])
  if (h.hrv != null) rows.push(['hrv', `${h.hrv} ms`])
  if (h.rhr != null) rows.push(['resting hr', `${h.rhr} bpm`])
  if (h.tempDeviationC != null)
    rows.push(['temp', `${h.tempDeviationC > 0 ? '+' : ''}${h.tempDeviationC.toFixed(1)}°C`])
  if (h.windKph != null)
    rows.push([
      'wind',
      `${h.windKph} km/h${h.windDir ? ` ${h.windDir}` : ''}${h.windGustKph != null ? ` / gust ${h.windGustKph}` : ''}`,
    ])
  if (h.totalCalories != null)
    rows.push(['total burn', `${Math.round(h.totalCalories).toLocaleString('en-US')} kcal`])
  if (h.activeCalories != null)
    rows.push(['active burn', `${Math.round(h.activeCalories).toLocaleString('en-US')} kcal`])
  return rows
}

export const fuelingRows = (f: ActivityFueling): [string, string][] => {
  const rows: [string, string][] = []
  const consumed: string[] = []
  if (f.caloriesConsumed != null) consumed.push(`${Math.round(f.caloriesConsumed)} kcal`)
  if (f.carbsConsumedG != null) consumed.push(`${Math.round(f.carbsConsumedG)} g carb`)
  if (consumed.length > 0) rows.push(['consumed', consumed.join(' / ')])
  if (f.fluidMl != null) rows.push(['fluid', formatMl(f.fluidMl)])

  const target: string[] = []
  if (f.carbsRecommendedG != null) target.push(`${Math.round(f.carbsRecommendedG)} g carb`)
  if (f.fluidRecommendedMl != null) target.push(formatMl(f.fluidRecommendedMl))
  if (target.length > 0) rows.push(['target', target.join(' / ')])

  if (f.sweatLossMl != null) rows.push(['sweat', formatMl(f.sweatLossMl)])
  if (f.sodiumLossMg != null)
    rows.push(['sodium loss', `${Math.round(f.sodiumLossMg).toLocaleString('en-US')} mg`])
  if (rows.length > 0) rows.push(['source', formatFuelingSource(f)])
  return rows
}

export const moreStatRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  fillMissingRunPower = false,
): [string, string][] => {
  const rows: [string, string][] = []
  const showRunPower = fillMissingRunPower && d.sport === 'run'
  if (d.deviceWatts && d.npWatts != null) rows.push(['NP', `${d.npWatts} W`])
  else if (showRunPower) rows.push(['NP', '—'])
  if (d.avgWatts != null) rows.push([d.deviceWatts ? 'avg power' : 'est power', `${d.avgWatts} W`])
  else if (showRunPower) rows.push(['avg power', '—'])
  if (d.deviceWatts && d.maxWatts != null) rows.push(['max power', `${d.maxWatts} W`])
  else if (showRunPower) rows.push(['max power', '—'])
  if (d.kilojoules != null) rows.push(['energy', `${d.kilojoules} kJ`])
  else if (showRunPower) rows.push(['energy', '—'])
  if (d.calories != null) rows.push(['calories', `${d.calories.toLocaleString('en-US')} kcal`])
  if (d.sport === 'swim') {
    const poolMetrics = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
    rows.push([
      'cadence',
      poolMetrics ? `${swimTrendNumber(poolMetrics.strokesPerLength)} /length` : '—',
    ])
  } else if (d.avgCadence != null)
    rows.push(['cadence', d.sport === 'run' ? `${d.avgCadence * 2} spm` : `${d.avgCadence} rpm`])
  else if (d.sport === 'run') rows.push(['cadence', '—'])
  if (d.maxHr != null) rows.push(['max hr', `${d.maxHr} bpm`])
  if (d.sufferScore != null) rows.push(['effort', `${d.sufferScore}`])
  if (d.avgTemp != null)
    rows.push([
      d.sport === 'swim' || d.sport === 'strength' ? 'air temp' : 'temp',
      formatTemperature(presentation, d.avgTemp),
    ])
  if (d.windKph != null)
    rows.push([
      'wind',
      `${d.windKph} km/h${d.windDir ? ` ${d.windDir}` : ''}${d.windGustKph != null ? ` / gust ${d.windGustKph}` : ''}`,
    ])
  return rows
}

export type ActivityThermalTracePoint = {
  d: number
  elapsedS: number
  heatStrainIndex: number | null
  coreTemperatureC: number | null
  skinTemperatureC: number | null
}

const routeLessTraceDistance = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['heartRateTrace'][number],
): number => (activityTraceUsesElapsedAxis(d) ? point.elapsedS : point.distanceKm)

export const activityTraceUsesElapsedAxis = (d: StravaActivityDetail): boolean =>
  d.route.length < 2 &&
  (d.sport !== 'swim' || !d.heartRateTrace.some(point => point.distanceKm > 0))

export const activityThermalTracePoints = (d: StravaActivityDetail): ActivityThermalTracePoint[] =>
  d.route.length >= 2
    ? d.route
    : d.heartRateTrace.map(point => ({
        d: routeLessTraceDistance(d, point),
        elapsedS: point.elapsedS,
        heatStrainIndex: point.heatStrainIndex,
        coreTemperatureC: point.coreTemperatureC,
        skinTemperatureC: point.skinTemperatureC,
      }))

export const routeStreamFlags = (
  d: StravaActivityDetail,
): {
  power: boolean
  powerBalance: boolean
  hr: boolean
  cad: boolean
  stride: boolean
  groundContact: boolean
  verticalOscillation: boolean
  stamina: boolean
  resp: boolean
  muscleOxygen: boolean
  temp: boolean
  heatStrain: boolean
  coreTemperature: boolean
  skinTemperature: boolean
} => {
  const thermalPoints = activityThermalTracePoints(d)
  return {
    power: d.deviceWatts && d.route.some(p => p.w > 0),
    powerBalance:
      d.sport === 'bike' &&
      d.route.filter(
        point =>
          point.rightPowerPct != null &&
          Number.isFinite(point.rightPowerPct) &&
          point.rightPowerPct >= 0 &&
          point.rightPowerPct <= 100 &&
          point.w > 0,
      ).length >= 2,
    hr: hasHeartRateTrace(d),
    cad: d.route.some(p => p.cad > 0),
    stride:
      d.sport === 'run' &&
      d.route.filter(point => runStrideLengthValue(d, point) != null).length >= 2,
    groundContact:
      d.sport === 'run' &&
      d.route.filter(point => runGroundContactTimeMs(point) != null).length >= 2,
    verticalOscillation:
      d.sport === 'run' &&
      d.route.filter(point => runVerticalOscillationCm(point) != null).length >= 2,
    stamina:
      d.sport === 'bike' &&
      d.route.filter(point => point.stamina != null && point.potentialStamina != null).length >= 2,
    resp: d.route.some(p => p.resp != null && p.resp > 0),
    muscleOxygen:
      d.route.filter(
        point =>
          point.muscleOxygenPct != null &&
          Number.isFinite(point.muscleOxygenPct) &&
          point.muscleOxygenPct >= 0 &&
          point.muscleOxygenPct <= 100,
      ).length >= 2,
    temp: d.route.some(p => p.tempC != null),
    heatStrain: thermalPoints.filter(point => point.heatStrainIndex != null).length >= 2,
    coreTemperature: thermalPoints.filter(point => point.coreTemperatureC != null).length >= 2,
    skinTemperature: thermalPoints.filter(point => point.skinTemperatureC != null).length >= 2,
  }
}

const nativeRunStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null => {
  const meters = point.strideLengthM
  return meters != null && Number.isFinite(meters) && meters >= 0.2 && meters <= 3 ? meters : null
}

const estimatedRunStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null => {
  if (!Number.isFinite(point.speedKph) || !Number.isFinite(point.cad) || point.cad <= 0) return null
  const meters = (point.speedKph * 1000) / 60 / (point.cad * 2)
  return Number.isFinite(meters) && meters >= 0.2 && meters <= 3 ? meters : null
}

export const runStrideLengthM = (point: StravaActivityDetail['route'][number]): number | null =>
  nativeRunStrideLengthM(point) ?? estimatedRunStrideLengthM(point)

export const runStrideLengthLabel = (d: StravaActivityDetail): string =>
  d.route.filter(point => nativeRunStrideLengthM(point) != null).length >= 2
    ? 'stride length'
    : 'estimated stride length'

export const runStrideLengthValue = (
  d: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
): number | null =>
  runStrideLengthLabel(d) === 'stride length'
    ? nativeRunStrideLengthM(point)
    : estimatedRunStrideLengthM(point)

export const runGroundContactTimeMs = (
  point: StravaActivityDetail['route'][number],
): number | null => {
  const milliseconds = point.groundContactTimeMs
  return milliseconds != null &&
    Number.isFinite(milliseconds) &&
    milliseconds >= 50 &&
    milliseconds <= 1_000
    ? milliseconds
    : null
}

export const runVerticalOscillationCm = (
  point: StravaActivityDetail['route'][number],
): number | null => {
  const centimeters = point.verticalOscillationCm
  return centimeters != null &&
    Number.isFinite(centimeters) &&
    centimeters >= 1 &&
    centimeters <= 30
    ? centimeters
    : null
}

export const formatStrideLength = (presentation: TriathlonPresentation, meters: number): string =>
  isImperial(presentation) ? `${(meters * M_TO_FT).toFixed(2)} ft` : `${meters.toFixed(2)} m`

export const formatGroundContactTime = (milliseconds: number): string =>
  `${Math.round(milliseconds)} ms`

export const formatVerticalOscillation = (
  presentation: TriathlonPresentation,
  centimeters: number,
): string =>
  isImperial(presentation)
    ? `${(centimeters / 2.54).toFixed(1)} in`
    : `${centimeters.toFixed(1)} cm`

export type ActivitySelectionSummary = {
  startElapsedS: number
  endElapsedS: number
  startDistanceKm: number
  endDistanceKm: number
  durationS: number
  distanceKm: number
  elevationGainM: number | null
  averageSpeedKph: number | null
  averageHeartRate: number | null
  averageWatts: number | null
  averageCadence: number | null
  averageRespirationRate: number | null
  averageTemperatureC: number | null
}

type WeightedRouteMetric = { total: number; durationS: number }

const addWeightedRouteMetric = (
  metric: WeightedRouteMetric,
  previous: number | null,
  next: number | null,
  durationS: number,
): void => {
  if (previous == null || next == null || durationS <= 0) return
  metric.total += ((previous + next) / 2) * durationS
  metric.durationS += durationS
}

const weightedRouteValue = (metric: WeightedRouteMetric): number | null =>
  metric.durationS > 0 ? metric.total / metric.durationS : null

export const activitySelectionSummary = (
  route: StravaActivityDetail['route'],
  anchorIndex: number,
  focusIndex: number,
): ActivitySelectionSummary | null => {
  if (route.length < 2) return null
  const first = Math.max(0, Math.min(route.length - 1, Math.round(anchorIndex)))
  const last = Math.max(0, Math.min(route.length - 1, Math.round(focusIndex)))
  const startIndex = Math.min(first, last)
  const endIndex = Math.max(first, last)
  if (startIndex === endIndex) return null
  const start = route[startIndex]
  const end = route[endIndex]
  const durationS = end.elapsedS - start.elapsedS
  const distanceKm = end.d - start.d
  if (durationS <= 0 || distanceKm <= 0) return null

  let elevationGainM = 0
  const heartRate: WeightedRouteMetric = { total: 0, durationS: 0 }
  const watts: WeightedRouteMetric = { total: 0, durationS: 0 }
  const cadence: WeightedRouteMetric = { total: 0, durationS: 0 }
  const respiration: WeightedRouteMetric = { total: 0, durationS: 0 }
  const temperature: WeightedRouteMetric = { total: 0, durationS: 0 }
  let hasPower = false
  let hasCadence = false
  for (let index = startIndex; index <= endIndex && (!hasPower || !hasCadence); index++) {
    hasPower ||= route[index].w > 0
    hasCadence ||= route[index].cad > 0
  }

  for (let index = startIndex + 1; index <= endIndex; index++) {
    const previous = route[index - 1]
    const next = route[index]
    const elapsedS = next.elapsedS - previous.elapsedS
    if (elapsedS <= 0) continue
    elevationGainM += Math.max(0, next.alt - previous.alt)
    addWeightedRouteMetric(
      heartRate,
      previous.hr > 0 ? previous.hr : null,
      next.hr > 0 ? next.hr : null,
      elapsedS,
    )
    addWeightedRouteMetric(watts, hasPower ? previous.w : null, hasPower ? next.w : null, elapsedS)
    addWeightedRouteMetric(
      cadence,
      hasCadence ? previous.cad : null,
      hasCadence ? next.cad : null,
      elapsedS,
    )
    addWeightedRouteMetric(respiration, previous.resp, next.resp, elapsedS)
    addWeightedRouteMetric(temperature, previous.tempC, next.tempC, elapsedS)
  }

  return {
    startElapsedS: start.elapsedS,
    endElapsedS: end.elapsedS,
    startDistanceKm: start.d,
    endDistanceKm: end.d,
    durationS,
    distanceKm,
    elevationGainM,
    averageSpeedKph: (distanceKm / durationS) * 3600,
    averageHeartRate: weightedRouteValue(heartRate),
    averageWatts: weightedRouteValue(watts),
    averageCadence: weightedRouteValue(cadence),
    averageRespirationRate: weightedRouteValue(respiration),
    averageTemperatureC: weightedRouteValue(temperature),
  }
}

const ANALYSIS_KIND_ORDER: ActivityAnalysisKind[] = ['lap', 'segment', 'climb']

const validAnalysisRanges = (d: StravaActivityDetail): ActivityAnalysisRange[] => {
  const seen = new Set<string>()
  return d.analysisRanges.filter(range => {
    const key = `${range.kind}:${range.id}`
    if (
      seen.has(key) ||
      range.id.trim().length === 0 ||
      range.label.trim().length === 0 ||
      !Number.isFinite(range.startElapsedS) ||
      !Number.isFinite(range.endElapsedS) ||
      range.startElapsedS < 0 ||
      range.endElapsedS <= range.startElapsedS ||
      !Number.isFinite(range.startDistanceKm) ||
      !Number.isFinite(range.endDistanceKm) ||
      range.startDistanceKm < 0 ||
      range.endDistanceKm <= range.startDistanceKm ||
      !Number.isFinite(range.durationS) ||
      range.durationS <= 0 ||
      !Number.isFinite(range.distanceKm) ||
      range.distanceKm <= 0
    )
      return false
    seen.add(key)
    return true
  })
}

const hasAnalysisWorkspace = (d: StravaActivityDetail): boolean =>
  d.route.length >= 2 &&
  d.route.every(
    point =>
      Number.isFinite(point.d) &&
      Number.isFinite(point.elapsedS) &&
      Number.isFinite(point.speedKph) &&
      Number.isFinite(point.lat) &&
      Number.isFinite(point.lng),
  )

const BATTERY = [
  'M23 10V14',
  'M1 16V8C1 6.89543 1.89543 6 3 6H18C19.1046 6 20 6.89543 20 8V16C20 17.1046 19.1046 18 18 18H3C1.89543 18 1 17.1046 1 16Z',
  'M10.1667 9L8.5 12H12.5L10.8333 15',
]

export const buildIcon = <N>(f: TriNodeFactory<N>, sport: ActivityKind): N => {
  const icon = f.svg('svg', {
    class: sport === 'treatment' || sport === 'yoga' ? 'tri-ico tri-ico--solid' : 'tri-ico',
    viewBox: '0 0 24 24',
    fill: 'none',
  })
  for (const d of SPORT_ICON[sport]) f.add(icon, f.svg('path', { d }))
  return icon
}

const COMPUTER_LABEL = { garmin: 'Edge 1050', wahoo: 'ELEMNT BOLT 3' } as const

export const buildBattery = <N>(f: TriNodeFactory<N>): N => {
  const icon = f.svg('svg', { class: 'tri-ico tri-battery', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of BATTERY) f.add(icon, f.svg('path', { d }))
  return icon
}

const buildRestStatus = <N>(f: TriNodeFactory<N>): N => {
  const rest = f.el('div', 'tri-pop-rest')
  f.add(
    rest,
    buildBattery(f),
    f.el('span', 'tri-pop-rest-label', triText(f.presentation.locale, 'rest'), {
      'data-i18n': 'rest',
    }),
  )
  return rest
}

export const LAYERS_ICON = [
  'M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z',
  'm22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65',
  'm22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65',
]

export const buildLayers = <N>(f: TriNodeFactory<N>): N => {
  const icon = f.svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of LAYERS_ICON) f.add(icon, f.svg('path', { d }))
  return icon
}

type RouteDrawPoint = { x: number; y: number }

const routePath = (route: RouteDrawPoint[]): string => {
  const pad = 6
  const span = 100 - pad * 2
  let d = ''
  route.forEach((p, i) => {
    d += `${i ? 'L' : 'M'} ${(pad + p.x * span).toFixed(2)} ${(pad + (1 - p.y) * span).toFixed(2)} `
  })
  return d
}

const routePointAtDistance = (
  route: StravaActivityDetail['route'],
  distanceKm: number,
): RouteDrawPoint => {
  if (distanceKm <= route[0].d) return route[0]
  for (let index = 1; index < route.length; index++) {
    const previous = route[index - 1]
    const next = route[index]
    if (distanceKm > next.d) continue
    const span = next.d - previous.d
    const fraction = span > 0 ? (distanceKm - previous.d) / span : 1
    return {
      x: previous.x + (next.x - previous.x) * fraction,
      y: previous.y + (next.y - previous.y) * fraction,
    }
  }
  return route[route.length - 1]
}

const selectedRoute = (
  route: StravaActivityDetail['route'],
  range: ActivityAnalysisRange | null,
): RouteDrawPoint[] => {
  if (!range) return []
  const start = Math.max(route[0].d, range.startDistanceKm)
  const end = Math.min(route[route.length - 1].d, range.endDistanceKm)
  if (end <= start) return []
  return [
    routePointAtDistance(route, start),
    ...route.filter(point => point.d > start && point.d < end),
    routePointAtDistance(route, end),
  ]
}

export const buildRoute = <N>(
  f: TriNodeFactory<N>,
  route: StravaActivityDetail['route'],
  range: ActivityAnalysisRange | null = null,
): N => {
  const fig = f.svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  f.add(fig, f.svg('path', { d: routePath(route), class: 'tri-route-path' }))
  const selection = selectedRoute(route, range)
  f.add(
    fig,
    f.svg('path', {
      d: selection.length >= 2 ? routePath(selection) : '',
      class: 'tri-route-selected',
    }),
  )
  f.add(fig, f.svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return fig
}

export const niceStep = (span: number, intervals: number): number => {
  if (!Number.isFinite(span) || span <= 0) return 1
  const raw = span / Math.max(1, intervals)
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const fraction = raw / magnitude
  const nice = fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10
  return nice * magnitude
}

const niceTicks = (min: number, max: number, intervals: number): number[] => {
  const step = niceStep(max - min, intervals)
  const first = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let value = first; value <= max + step * 1e-6; value += step)
    ticks.push(Math.round(value * 1e6) / 1e6)
  if (ticks.length >= 2) return ticks
  return [min, max]
}

export const axisNumber = (value: number, step: number): string => {
  const decimals = step >= 1 ? 0 : Math.min(2, Math.ceil(-Math.log10(step)))
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export type ActivityGraphDomain = { startDistanceKm: number; endDistanceKm: number }
export type ActivityTraceDomain = { min: number; max: number; intervals?: number }
export type ActivityTraceReference = { value: number; label: string }

export const positiveMetricDomain = (
  values: readonly (number | null)[],
  intervals = 3,
): ActivityTraceDomain | undefined => {
  const measured = values.filter(
    (value): value is number => value != null && Number.isFinite(value) && value > 0,
  )
  if (measured.length === 0) return undefined
  const min = Math.min(...measured)
  const max = Math.max(...measured)
  if (max > min) return { min, max, intervals }
  const pad = Math.max(1, max * 0.05)
  return { min: Math.max(Number.EPSILON, min - pad), max: max + pad, intervals }
}

const distanceXTicks = (
  presentation: TriathlonPresentation,
  startDistanceKm: number,
  endDistanceKm: number,
): AxisXTick[] => {
  const imperial = isImperial(presentation)
  const scale = imperial ? KM_TO_MI : 1
  const displayStart = startDistanceKm * scale
  const displayEnd = endDistanceKm * scale
  const displaySpan = displayEnd - displayStart
  const step = niceStep(displaySpan, 4)
  const first = Math.ceil(displayStart / step) * step
  const ticks: AxisXTick[] = []
  for (let value = first; value < displayEnd - step * 1e-6; value += step) {
    if (value <= displayStart + step * 1e-6) continue
    ticks.push({
      label: `${axisNumber(value, step)} ${imperial ? 'mi' : 'km'}`,
      pct: ((value - displayStart) / displaySpan) * 100,
    })
  }
  return ticks
}

const elapsedActivityXTicks = (totalElapsedS: number): AxisXTick[] => [
  { label: zoneClock(0), pct: 0, cls: 'tri-cax-xt--first' },
  { label: zoneClock(totalElapsedS / 2), pct: 50 },
  { label: zoneClock(totalElapsedS), pct: 100, cls: 'tri-cax-xt--last' },
]

const graphView = (
  d: StravaActivityDetail,
  domain?: ActivityGraphDomain | null,
): ActivityGraphDomain & { start: number; width: number } =>
  graphViewForDistance(d.route[d.route.length - 1].d || 1, domain)

const graphViewForDistance = (
  maxDistanceKm: number,
  domain?: ActivityGraphDomain | null,
): ActivityGraphDomain & { start: number; width: number } => {
  const resolved = domain
    ? {
        startDistanceKm: Math.max(0, Math.min(maxDistanceKm, domain.startDistanceKm)),
        endDistanceKm: Math.max(0, Math.min(maxDistanceKm, domain.endDistanceKm)),
      }
    : { startDistanceKm: 0, endDistanceKm: maxDistanceKm }
  const valid =
    resolved.endDistanceKm > resolved.startDistanceKm
      ? resolved
      : { startDistanceKm: 0, endDistanceKm: maxDistanceKm }
  return {
    ...valid,
    start: (valid.startDistanceKm / maxDistanceKm) * 100,
    width: ((valid.endDistanceKm - valid.startDistanceKm) / maxDistanceKm) * 100,
  }
}

const routeDistanceAtElapsed = (d: StravaActivityDetail, elapsedS: number): number => {
  const route = d.route
  if (elapsedS <= route[0].elapsedS) return route[0].d
  for (let index = 1; index < route.length; index++) {
    const previous = route[index - 1]
    const next = route[index]
    if (elapsedS > next.elapsedS) continue
    const elapsedSpan = next.elapsedS - previous.elapsedS
    const fraction = elapsedSpan > 0 ? (elapsedS - previous.elapsedS) / elapsedSpan : 1
    return previous.d + (next.d - previous.d) * Math.max(0, Math.min(1, fraction))
  }
  return route[route.length - 1].d
}

const analysisSelectionBounds = (
  d: StravaActivityDetail,
  range: ActivityAnalysisRange,
): { x: number; width: number } => {
  const maxD = d.route[d.route.length - 1].d || 1
  const start = Math.max(0, Math.min(maxD, routeDistanceAtElapsed(d, range.startElapsedS)))
  const end = Math.max(start, Math.min(maxD, routeDistanceAtElapsed(d, range.endElapsedS)))
  const x = (start / maxD) * 100
  return { x, width: Math.max(0, ((end - start) / maxD) * 100) }
}

const buildAnalysisSelection = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  height: number,
  range: ActivityAnalysisRange | null,
): N => {
  const bounds = range ? analysisSelectionBounds(d, range) : { x: 0, width: 0 }
  return f.svg('rect', {
    class: 'tri-analysis-selection',
    x: bounds.x.toFixed(2),
    y: 0,
    width: bounds.width.toFixed(2),
    height,
  })
}

export const buildElevation = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  domain?: ActivityGraphDomain | null,
): N => {
  const w = 100
  const h = 30
  const maxD = d.route[d.route.length - 1].d || 1
  const imperial = isImperial(f.presentation)
  const displayMinAlt = elevationValue(f.presentation, d.minAlt)
  const displayMaxAlt = elevationValue(f.presentation, d.maxAlt)
  const altPad = displayMinAlt === displayMaxAlt ? 1 : 0
  const minAlt = displayMinAlt - altPad
  const maxAlt = displayMaxAlt + altPad
  const altSpan = Math.max(1e-6, maxAlt - minAlt)
  const px = (km: number): number => (km / maxD) * w
  const py = (alt: number): number =>
    h - ((elevationValue(f.presentation, alt) - minAlt) / altSpan) * h
  const yValues = niceTicks(minAlt, maxAlt, 4)
  const yStep = niceStep(maxAlt - minAlt, 4)
  const yTicks = yValues.map(value => ({
    label: `${axisNumber(value, yStep)} ${imperial ? 'ft' : 'm'}`,
    vbY: h - ((value - minAlt) / altSpan) * h,
  }))
  const view = graphView(d, domain)
  const xTicks = distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    area += `L ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const fig = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${h}`,
    preserveAspectRatio: 'none',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(fig, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: w, y2: tick.vbY }))
  f.add(fig, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (selection !== undefined) f.add(fig, buildAnalysisSelection(f, d, h, selection))
  f.add(fig, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(fig, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap')
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', `+${formatElevationGain(f.presentation, d.elevationM)}`),
    f.el('span', 'tri-elev-d', `−${formatElevationGain(f.presentation, d.descentM)}`),
    f.el(
      'span',
      'tri-elev-range',
      `${formatAltitude(f.presentation, d.minAlt)}–${formatAltitude(f.presentation, d.maxAlt)}`,
    ),
  )
  const frame = axisFrame(f, fig, yTicks, h, xTicks, true, { top: 0, bottom: h })
  f.add(wrap, frame, cap)
  return wrap
}

const buildUnavailableElevation = <N>(f: TriNodeFactory<N>): N => {
  const wrap = f.el('div', 'tri-elev-wrap tri-elev-wrap--unavailable')
  const unavailable = f.el('div', 'tri-elev-unavailable', 'no data available', {
    'data-i18n': 'no data available',
  })
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--unavailable', undefined, {
    'aria-hidden': 'true',
  })
  f.add(wrap, axisFrame(f, unavailable, [], 30, [], false), cap)
  return wrap
}

const buildTraceSeries = <N, P extends { d: number; elapsedS: number }>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  points: readonly P[],
  pick: (point: P, index: number) => number | null,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
  domain?: ActivityTraceDomain,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  missing?: 'dotted',
  reference?: ActivityTraceReference | null,
): N => {
  const w = 100
  const h = 30
  const maxD = points.at(-1)?.d || d.distanceKm || 1
  let peak = 1
  const values = points.map(pick)
  values.forEach(value => {
    if (value != null && Number.isFinite(value) && value > peak) peak = value
  })
  const domainMin = Math.min(domain?.min ?? 0, reference?.value ?? Number.POSITIVE_INFINITY)
  const candidateMax = Math.max(domain?.max ?? peak, reference?.value ?? Number.NEGATIVE_INFINITY)
  const domainMax = candidateMax > domainMin ? candidateMax : domainMin + 1
  const px = (km: number): number => (km / maxD) * w
  const py = (v: number): number =>
    Math.min(h, Math.max(0, h - ((v - domainMin) / (domainMax - domainMin)) * (h - 1)))
  let area = ''
  let line = ''
  let missingLine = ''
  let segmentStart = -1
  const closeSegment = (start: number, end: number): void => {
    const first = values[start]
    if (first == null) return
    const startX = start === 0 ? 0 : px(points[start].d)
    const firstX = px(points[start].d)
    const firstY = py(first).toFixed(2)
    area += `M ${startX.toFixed(2).replace('.00', '')} ${h} L ${startX.toFixed(2).replace('.00', '')} ${firstY} `
    line += `M ${startX.toFixed(2).replace('.00', '')} ${firstY} `
    if (firstX !== startX) {
      area += `L ${firstX.toFixed(2)} ${firstY} `
      line += `L ${firstX.toFixed(2)} ${firstY} `
    }
    for (let i = start + 1; i <= end; i++) {
      const value = values[i]
      if (value == null) continue
      const x = px(points[i].d).toFixed(2)
      const y = py(value).toFixed(2)
      area += `L ${x} ${y} `
      line += `L ${x} ${y} `
    }
    area += `L ${px(points[end].d).toFixed(2)} ${h} Z `
  }
  values.forEach((value, index) => {
    const valid = value != null && Number.isFinite(value)
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === values.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  if (missing === 'dotted') {
    let previous = -1
    values.forEach((value, index) => {
      if (value == null || !Number.isFinite(value)) return
      const x = px(points[index].d).toFixed(2)
      const y = py(value).toFixed(2)
      if (previous < 0 && index > 0) missingLine += `M 0 ${y} L ${x} ${y} `
      else if (previous >= 0 && index > previous + 1) {
        const previousValue = values[previous]
        if (previousValue != null) {
          missingLine += `M ${px(points[previous].d).toFixed(2)} ${py(previousValue).toFixed(2)} L ${x} ${y} `
        }
      }
      previous = index
    })
    if (previous >= 0 && previous < values.length - 1) {
      const value = values[previous]
      if (value != null)
        missingLine += `M ${px(points[previous].d).toFixed(2)} ${py(value).toFixed(2)} L ${w} ${py(value).toFixed(2)} `
    }
  }
  const yTicks = niceTicks(domainMin, domainMax, domain?.intervals ?? 3).map(value => ({
    label: value === 0 ? '0' : tick(value),
    vbY: py(value),
  }))
  const view = graphViewForDistance(maxD, graphDomain)
  const usesElapsedAxis = activityTraceUsesElapsedAxis(d)
  const s = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${h}`,
    preserveAspectRatio: 'none',
    ...(usesElapsedAxis
      ? {
          'data-domain-start-elapsed-s': view.startDistanceKm,
          'data-domain-end-elapsed-s': view.endDistanceKm,
        }
      : {
          'data-domain-start-distance-km': view.startDistanceKm,
          'data-domain-end-distance-km': view.endDistanceKm,
        }),
  })
  for (const t of yTicks)
    f.add(s, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: t.vbY, x2: w, y2: t.vbY }))
  f.add(s, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (selection !== undefined && d.route.length >= 2)
    f.add(s, buildAnalysisSelection(f, d, h, selection))
  if (missingLine)
    f.add(s, f.svg('path', { d: missingLine, class: 'tri-elev-line tri-elev-line--missing' }))
  if (reference)
    f.add(
      s,
      f.svg('line', {
        class: 'tri-trace-reference',
        x1: 0,
        y1: py(reference.value).toFixed(2),
        x2: w,
        y2: py(reference.value).toFixed(2),
      }),
    )
  f.add(s, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(s, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap', undefined, {
    'data-tri-trace': triathlonTraceName(title),
  })
  const capEl = f.el('div', 'tri-elev-cap')
  f.add(
    capEl,
    f.el('span', 'tri-elev-d', triText(f.presentation.locale, title)),
    f.el('span', 'tri-elev-range', cap(peak)),
  )
  if (reference) f.add(capEl, f.el('span', 'tri-elev-range tri-trace-reference-k', reference.label))
  f.add(
    wrap,
    capEl,
    axisFrame(
      f,
      s,
      yTicks,
      h,
      d.sport === 'swim'
        ? swimActivityXTicks(maxD * 1_000)
        : usesElapsedAxis
          ? elapsedActivityXTicks(maxD)
          : distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: h },
    ),
  )
  return wrap
}

export const buildTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number | null,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
  domain?: ActivityTraceDomain,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
  missing?: 'dotted',
  reference?: ActivityTraceReference | null,
): N =>
  buildTraceSeries(
    f,
    d,
    d.route,
    pick,
    title,
    cap,
    tick,
    domain,
    d.route.length >= 2 ? selection : undefined,
    graphDomain,
    missing,
    reference,
  )

const HEART_RATE_TRACE_MIN_BPM = 80

export const activityHeartRateTracePoints = (
  d: StravaActivityDetail,
): { d: number; elapsedS: number; heartRate: number | null }[] => {
  const route = d.route.map(point => ({
    d: point.d,
    elapsedS: point.elapsedS,
    heartRate: point.hr > 0 ? point.hr : null,
  }))
  if (route.filter(point => point.heartRate != null).length >= 2) return route
  return d.heartRateTrace.map(point => ({
    d: routeLessTraceDistance(d, point),
    elapsedS: point.elapsedS,
    heartRate: point.heartRate,
  }))
}

export const hasHeartRateTrace = (d: StravaActivityDetail): boolean =>
  activityHeartRateTracePoints(d).filter(point => point.heartRate != null).length >= 2

export const buildHeartRateTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const points = activityHeartRateTracePoints(d)
  const heartRates = points.flatMap(point => (point.heartRate == null ? [] : [point.heartRate]))
  const minimum = Math.min(HEART_RATE_TRACE_MIN_BPM, Math.floor(Math.min(...heartRates) / 10) * 10)
  const maximum = Math.max(minimum + 20, ...heartRates)
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.heartRate,
    'hr',
    peak => `${peak} bpm peak`,
    value => `${Math.round(value)}bpm`,
    { min: minimum, max: maximum, intervals: 4 },
    selection,
    graphDomain,
  )
}

const routeRightPowerPct = (point: StravaActivityDetail['route'][number]): number | null => {
  const value = point.rightPowerPct
  return value != null && Number.isFinite(value) && value >= 0 && value <= 100 && point.w > 0
    ? value
    : null
}

type PowerBalanceSample = { watts: number; rightPowerPct: number; distanceKm: number }

type WattsHeatSample = { watts: number; value: number }
type WattsHeatCell = { count: number; x: number; y: number; width: number; height: number }
type CyclingModeTitle =
  | 'power balance'
  | 'torque effectiveness'
  | 'pedal smoothness'
  | 'electronic shifting'

const WATTS_HEAT_X_BINS = 64
const WATTS_HEAT_Y_BINS = 24

const powerBalanceSamples = (
  d: StravaActivityDetail,
  graphDomain?: ActivityGraphDomain | null,
): PowerBalanceSample[] => {
  const measured = d.route.flatMap(point => {
    const rightPowerPct = routeRightPowerPct(point)
    return rightPowerPct == null ? [] : [{ watts: point.w, rightPowerPct, distanceKm: point.d }]
  })
  if (!graphDomain) return measured
  const selected = measured.filter(
    sample =>
      sample.distanceKm >= graphDomain.startDistanceKm &&
      sample.distanceKm <= graphDomain.endDistanceKm,
  )
  return selected.length > 0 ? selected : measured
}

const wattsAxis = (
  d: StravaActivityDetail,
  samples: readonly { watts: number }[],
): { max: number; ticks: AxisXTick[] } => {
  const observedMax = Math.max(d.maxWatts ?? 0, ...samples.map(sample => sample.watts))
  const step = niceStep(observedMax, 5)
  const max = Math.max(step, Math.ceil(observedMax / step) * step)
  const ticks: AxisXTick[] = []
  for (let watts = 0; watts <= max + step * 1e-6; watts += step) {
    ticks.push({
      label: `${axisNumber(watts, step)} W`,
      pct: (watts / max) * 100,
      ...(watts === 0
        ? { cls: 'tri-cax-xt--first' }
        : watts >= max
          ? { cls: 'tri-cax-xt--last' }
          : {}),
    })
  }
  return { max, ticks }
}

const wattsHeatCells = (
  samples: readonly WattsHeatSample[],
  maxWatts: number,
  domainMin: number,
  domainMax: number,
  height: number,
  invertY: boolean,
): WattsHeatCell[] => {
  const counts = new Map<number, number>()
  for (const sample of samples) {
    const xBin = Math.min(
      WATTS_HEAT_X_BINS - 1,
      Math.max(0, Math.floor((sample.watts / maxWatts) * WATTS_HEAT_X_BINS)),
    )
    const normalized = (sample.value - domainMin) / Math.max(1e-6, domainMax - domainMin)
    const valueBin = Math.min(
      WATTS_HEAT_Y_BINS - 1,
      Math.max(0, Math.floor(normalized * WATTS_HEAT_Y_BINS)),
    )
    const yBin = invertY ? WATTS_HEAT_Y_BINS - 1 - valueBin : valueBin
    const key = yBin * WATTS_HEAT_X_BINS + xBin
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const xSize = 100 / WATTS_HEAT_X_BINS
  const ySize = height / WATTS_HEAT_Y_BINS
  return Array.from(counts, ([key, count]) => {
    const xBin = key % WATTS_HEAT_X_BINS
    const yBin = Math.floor(key / WATTS_HEAT_X_BINS)
    return { count, x: xBin * xSize, y: yBin * ySize, width: xSize, height: ySize }
  }).sort((left, right) => left.y - right.y || left.x - right.x)
}

const buildWattsHeatmap = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  title: CyclingModeTitle,
  className: string,
  dataName: string,
  series: readonly { side: 'left' | 'right' | 'single'; samples: readonly WattsHeatSample[] }[],
  domainMin: number,
  domainMax: number,
  height: number,
  invertY = true,
): { svg: N; wattsTicks: AxisXTick[] } => {
  const allSamples = series.flatMap(item => item.samples)
  const axis = wattsAxis(d, allSamples)
  const svg = f.svg('svg', {
    class: `tri-cycling-watts-heatmap ${className}`,
    viewBox: `0 0 100 ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, `${title} by watts`),
    'data-i18n-aria-label': `${title} by watts`,
    [`data-${dataName}-samples`]: allSamples.length,
    [`data-${dataName}-max-watts`]: axis.max,
  })
  const splitSides =
    series.some(item => item.side === 'left') && series.some(item => item.side === 'right')
  for (const item of series) {
    const cells = wattsHeatCells(item.samples, axis.max, domainMin, domainMax, height, invertY)
    const maxCellCount = Math.max(1, ...cells.map(cell => cell.count))
    for (const cell of cells) {
      const density = 0.16 + 0.84 * Math.sqrt(cell.count / maxCellCount)
      const width = splitSides ? cell.width / 2 : cell.width
      const x = cell.x + (splitSides && item.side === 'right' ? width : 0)
      f.add(
        svg,
        f.svg('rect', {
          class: `tri-cycling-watts-heat-cell tri-cycling-watts-heat-cell--${item.side}`,
          x: x.toFixed(3),
          y: cell.y.toFixed(3),
          width: width.toFixed(3),
          height: cell.height.toFixed(3),
          style: `--tri-cycling-watts-density:${density.toFixed(3)}`,
          'data-samples': cell.count,
        }),
      )
    }
  }
  return { svg, wattsTicks: axis.ticks }
}

const buildCyclingChartModes = <N>(f: TriNodeFactory<N>): N => {
  const modes = f.el('span', 'tri-cycling-chart-modes', undefined, {
    role: 'group',
    'aria-label': triText(f.presentation.locale, 'cycling charts view'),
    'data-i18n-aria-label': 'cycling charts view',
  })
  for (const [mode, label] of [
    ['distance', 'distance'],
    ['power', 'watts'],
  ] as const)
    f.add(
      modes,
      f.el('button', 'tri-cycling-chart-mode tri-curve-range', label, {
        type: 'button',
        'data-cycling-chart-mode': mode,
        'aria-pressed': String(mode === 'distance'),
        'data-i18n': label,
      }),
    )
  return modes
}

export const powerBalanceText = (rightPowerPct: number): string =>
  `L ${(100 - rightPowerPct).toFixed(1)}% / R ${rightPowerPct.toFixed(1)}%`

const activityRightPowerPct = (d: StravaActivityDetail): number | null => {
  let work = 0
  let rightWork = 0
  for (let index = 1; index < d.route.length; index++) {
    const previous = d.route[index - 1]
    const next = d.route[index]
    const previousRight = routeRightPowerPct(previous)
    const nextRight = routeRightPowerPct(next)
    const elapsedS = next.elapsedS - previous.elapsedS
    if (previousRight == null || nextRight == null || elapsedS <= 0) continue
    work += ((previous.w + next.w) / 2) * elapsedS
    rightWork += ((previous.w * previousRight + next.w * nextRight) / 2) * elapsedS
  }
  return work > 0 ? rightWork / work : null
}

interface PowerBalancePaths {
  measured: string
  bridges: string
}

const powerBalancePaths = (
  d: StravaActivityDetail,
  pick: (rightPowerPct: number) => number,
  domainMin: number,
  domainMax: number,
  width: number,
  height: number,
): PowerBalancePaths => {
  const maxDistanceKm = d.route.at(-1)?.d || 1
  const px = (distanceKm: number): number => (distanceKm / maxDistanceKm) * width
  const py = (value: number): number => {
    const bounded = Math.min(domainMax, Math.max(domainMin, value))
    return height - ((bounded - domainMin) / (domainMax - domainMin)) * (height - 1)
  }
  let measured = ''
  let bridges = ''
  let segmentStart = -1
  let previousValid = -1
  const closeSegment = (start: number, end: number): void => {
    for (let index = start; index <= end; index++) {
      const rightPowerPct = routeRightPowerPct(d.route[index])
      if (rightPowerPct == null) continue
      measured += `${index === start ? 'M' : 'L'} ${px(d.route[index].d).toFixed(2)} ${py(pick(rightPowerPct)).toFixed(2)} `
    }
  }
  d.route.forEach((point, index) => {
    const rightPowerPct = routeRightPowerPct(point)
    const valid = rightPowerPct != null
    if (valid && previousValid >= 0 && index > previousValid + 1) {
      const previousRightPowerPct = routeRightPowerPct(d.route[previousValid])
      if (previousRightPowerPct != null)
        bridges += `M ${px(d.route[previousValid].d).toFixed(2)} ${py(pick(previousRightPowerPct)).toFixed(2)} L ${px(point.d).toFixed(2)} ${py(pick(rightPowerPct)).toFixed(2)} `
    }
    if (valid) previousValid = index
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === d.route.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  return { measured, bridges }
}

export const buildPowerBalanceChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const samples = powerBalanceSamples(d, graphDomain)
  const rightValues = d.route.flatMap(point => {
    const rightPowerPct = routeRightPowerPct(point)
    return rightPowerPct == null ? [] : [rightPowerPct]
  })
  if (d.sport !== 'bike' || rightValues.length < 2) return null
  const width = 100
  const height = 30
  const deviations = rightValues.map(value => Math.abs(value - 50)).sort((a, b) => a - b)
  const observed = deviations[Math.round((deviations.length - 1) * 0.9)]
  const span = Math.min(25, Math.max(5, Math.ceil(observed / 2) * 2))
  const domainMin = 50 - span
  const domainMax = 50 + span
  const py = (value: number): number =>
    height - ((value - domainMin) / (domainMax - domainMin)) * (height - 1)
  const yTicks = [domainMin, 50, domainMax].map(value => ({ label: `${value}%`, vbY: py(value) }))
  const view = graphView(d, graphDomain)
  const leftPaths = powerBalancePaths(
    d,
    rightPowerPct => 100 - rightPowerPct,
    domainMin,
    domainMax,
    width,
    height,
  )
  const rightPaths = powerBalancePaths(
    d,
    rightPowerPct => rightPowerPct,
    domainMin,
    domainMax,
    width,
    height,
  )
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-power-balance-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'power balance'),
    'data-i18n-aria-label': 'power balance',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: width, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  if (leftPaths.bridges)
    f.add(
      svgEl,
      f.svg('path', {
        d: leftPaths.bridges,
        class:
          'tri-power-balance-line tri-power-balance-line--left tri-power-balance-line--missing',
      }),
    )
  if (rightPaths.bridges)
    f.add(
      svgEl,
      f.svg('path', {
        d: rightPaths.bridges,
        class:
          'tri-power-balance-line tri-power-balance-line--right tri-power-balance-line--missing',
      }),
    )
  f.add(
    svgEl,
    f.svg('path', {
      d: leftPaths.measured,
      class: 'tri-power-balance-line tri-power-balance-line--left',
    }),
    f.svg('path', {
      d: rightPaths.measured,
      class: 'tri-power-balance-line tri-power-balance-line--right',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const heatmap = buildWattsHeatmap(
    f,
    d,
    'power balance',
    'tri-power-balance-heatmap',
    'power-balance',
    [
      {
        side: 'single',
        samples: samples.map(sample => ({ watts: sample.watts, value: sample.rightPowerPct })),
      },
    ],
    0,
    100,
    height,
    false,
  )
  const heatSvg = heatmap.svg
  f.add(
    heatSvg,
    f.svg('line', {
      class: 'tri-power-balance-reference',
      x1: 0,
      y1: height / 2,
      x2: width,
      y2: height / 2,
    }),
  )
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-mode-chart tri-power-balance-chart',
    undefined,
    {
      'data-tri-trace': triathlonTraceName('power balance'),
      'data-cycling-chart-mode': 'distance',
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-power-balance-summary')
  const rightAverage = activityRightPowerPct(d)
  if (rightAverage != null)
    f.add(
      summary,
      f.el('span', 'tri-elev-range', `${powerBalanceText(rightAverage)}${embedded ? '' : ' avg'}`),
    )
  if (!embedded) {
    for (const side of ['left', 'right'] as const) {
      const item = f.el(
        'span',
        `tri-power-balance-legend-item tri-power-balance-legend-item--${side}`,
      )
      f.add(
        item,
        f.el('span', 'tri-power-balance-legend-line', undefined, { 'aria-hidden': 'true' }),
        f.el('span', 'tri-power-balance-legend-label', side, { 'data-i18n': side }),
      )
      f.add(summary, item)
    }
  }
  f.add(
    cap,
    f.el('span', 'tri-elev-d', 'power balance', { 'data-i18n': 'power balance' }),
    summary,
    buildCyclingChartModes(f),
  )
  const distancePane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-power-balance-pane--distance',
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(
    distancePane,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  const powerPane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--power tri-power-balance-pane--power',
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(
      f,
      heatSvg,
      [
        { label: '100% L', vbY: 0 },
        { label: '50/50', vbY: height / 2 },
        { label: '100% R', vbY: height },
      ],
      height,
      heatmap.wattsTicks,
      true,
      { top: 0, bottom: height },
    ),
  )
  f.add(wrap, cap, distancePane, powerPane)
  return wrap
}

type CyclingDynamicsPercentMetric = 'pedalSmoothness' | 'torqueEffectiveness'

type CyclingDynamicsSeriesKey =
  | 'leftPedalSmoothness'
  | 'rightPedalSmoothness'
  | 'leftTorqueEffectiveness'
  | 'rightTorqueEffectiveness'
  | 'leftPowerPhaseStart'
  | 'leftPowerPhaseEnd'
  | 'rightPowerPhaseStart'
  | 'rightPowerPhaseEnd'

const cyclingDynamicsSampleCount = (dynamics: ActivityCyclingDynamics): number => {
  const length = dynamics.elapsedS.length
  return length >= 2 && dynamics.distanceKm.length === length ? length : 0
}

const cyclingDynamicsValues = (
  dynamics: ActivityCyclingDynamics,
  key: CyclingDynamicsSeriesKey,
): (number | null)[] => {
  const values = dynamics[key]
  return values.length === cyclingDynamicsSampleCount(dynamics) ? values : []
}

const validCyclingDynamicsValue = (value: number | null): value is number =>
  value != null && Number.isFinite(value)

export const cyclingDynamicsIndexAtDistance = (
  dynamics: ActivityCyclingDynamics,
  distanceKm: number,
): number => {
  const length = cyclingDynamicsSampleCount(dynamics)
  if (length === 0) return -1
  let low = 0
  let high = length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (dynamics.distanceKm[middle] < distanceKm) low = middle + 1
    else high = middle
  }
  if (low === 0) return 0
  if (low >= length) return length - 1
  return distanceKm - dynamics.distanceKm[low - 1] <= dynamics.distanceKm[low] - distanceKm
    ? low - 1
    : low
}

export const riderPositionAtDistance = (
  dynamics: ActivityCyclingDynamics,
  distanceKm: number,
): 'seated' | 'standing' | null => {
  const changes = dynamics.positionChanges
  if (changes.length === 0 || distanceKm < changes[0].distanceKm) return null
  let low = 0
  let high = changes.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (changes[middle].distanceKm <= distanceKm) low = middle + 1
    else high = middle
  }
  return changes[Math.max(0, low - 1)]?.position ?? null
}

interface CyclingDynamicsPaths {
  measured: string
  bridges: string
}

const cyclingDynamicsPaths = (
  dynamics: ActivityCyclingDynamics,
  values: readonly (number | null)[],
  maxDistanceKm: number,
  py: (value: number) => number,
  circular = false,
): CyclingDynamicsPaths => {
  const px = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * 100
  let measured = ''
  let bridges = ''
  let previousValid = -1
  let segmentOpen = false
  for (let index = 0; index < values.length; index++) {
    const value = values[index]
    if (!validCyclingDynamicsValue(value)) {
      segmentOpen = false
      continue
    }
    const previousValue = previousValid >= 0 ? values[previousValid] : null
    const wraps =
      circular && validCyclingDynamicsValue(previousValue) && Math.abs(previousValue - value) > 180
    const gap =
      previousValid >= 0 &&
      (index > previousValid + 1 ||
        dynamics.elapsedS[index] - dynamics.elapsedS[previousValid] > 60 ||
        wraps)
    if (gap && !wraps && validCyclingDynamicsValue(previousValue))
      bridges += `M ${px(dynamics.distanceKm[previousValid]).toFixed(2)} ${py(previousValue).toFixed(2)} L ${px(dynamics.distanceKm[index]).toFixed(2)} ${py(value).toFixed(2)} `
    measured += `${segmentOpen && !gap ? 'L' : 'M'} ${px(dynamics.distanceKm[index]).toFixed(2)} ${py(value).toFixed(2)} `
    previousValid = index
    segmentOpen = true
  }
  return { measured, bridges }
}

const cyclingDynamicsDomain = (values: readonly number[]): [number, number] => {
  const sorted = [...values].sort((left, right) => left - right)
  const low = sorted[Math.round((sorted.length - 1) * 0.05)]
  const high = sorted[Math.round((sorted.length - 1) * 0.95)]
  const padding = Math.max(5, (high - low) * 0.15)
  let min = Math.max(0, Math.floor((low - padding) / 5) * 5)
  let max = Math.min(100, Math.ceil((high + padding) / 5) * 5)
  if (max - min < 20) {
    const middle = (min + max) / 2
    min = Math.max(0, Math.floor((middle - 10) / 5) * 5)
    max = Math.min(100, Math.ceil((middle + 10) / 5) * 5)
  }
  return max > min ? [min, max] : [0, 100]
}

const cyclingDynamicsAverage = (values: readonly (number | null)[]): number | null => {
  let sum = 0
  let count = 0
  for (const value of values) {
    if (!validCyclingDynamicsValue(value)) continue
    sum += value
    count++
  }
  return count > 0 ? sum / count : null
}

const routeWattsAtElapsed = (
  route: StravaActivityDetail['route'],
  elapsedS: number,
): number | null => {
  if (route.length === 0) return null
  let low = 0
  let high = route.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (route[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const index =
    low === 0
      ? 0
      : low >= route.length
        ? route.length - 1
        : elapsedS - route[low - 1].elapsedS <= route[low].elapsedS - elapsedS
          ? low - 1
          : low
  return route[index].w > 0 ? route[index].w : null
}

const cyclingDynamicsWattsSamples = (
  d: StravaActivityDetail,
  dynamics: ActivityCyclingDynamics,
  values: readonly (number | null)[],
  graphDomain?: ActivityGraphDomain | null,
): WattsHeatSample[] => {
  const samples: WattsHeatSample[] = []
  for (let index = 0; index < values.length; index++) {
    const value = values[index]
    const distanceKm = dynamics.distanceKm[index]
    if (
      !validCyclingDynamicsValue(value) ||
      (graphDomain &&
        (distanceKm < graphDomain.startDistanceKm || distanceKm > graphDomain.endDistanceKm))
    )
      continue
    const watts = routeWattsAtElapsed(d.route, dynamics.elapsedS[index])
    if (watts != null) samples.push({ watts, value })
  }
  return samples
}

const buildCyclingDynamicsLegend = <N>(
  f: TriNodeFactory<N>,
  side: 'left' | 'right',
  suffix = '',
): N => {
  const item = f.el(
    'span',
    `tri-cycling-dynamics-legend-item tri-cycling-dynamics-legend-item--${side}${suffix ? ` tri-cycling-dynamics-legend-item--${suffix}` : ''}`,
  )
  f.add(
    item,
    f.el('span', 'tri-cycling-dynamics-legend-line', undefined, { 'aria-hidden': 'true' }),
    f.el(
      'span',
      'tri-cycling-dynamics-legend-label',
      `${triText(f.presentation.locale, side)}${suffix ? ` ${triText(f.presentation.locale, suffix)}` : ''}`,
    ),
  )
  return item
}

type CyclingDynamicsGlossTitle = 'torque effectiveness' | 'pedal smoothness' | 'power phase'

const buildCyclingDynamicsTitle = <N>(f: TriNodeFactory<N>, title: CyclingDynamicsGlossTitle): N =>
  f.el('span', 'tri-elev-d', triText(f.presentation.locale, title), {
    'data-gloss': title,
    tabindex: '0',
  })

const buildCyclingDynamicsPercentChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  metric: CyclingDynamicsPercentMetric,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || cyclingDynamicsSampleCount(dynamics) < 2) return null
  const leftKey = metric === 'pedalSmoothness' ? 'leftPedalSmoothness' : 'leftTorqueEffectiveness'
  const rightKey =
    metric === 'pedalSmoothness' ? 'rightPedalSmoothness' : 'rightTorqueEffectiveness'
  const left = cyclingDynamicsValues(dynamics, leftKey)
  const right = cyclingDynamicsValues(dynamics, rightKey)
  const observed = [...left, ...right].filter(validCyclingDynamicsValue)
  if (observed.length < 2) return null
  const [domainMin, domainMax] = cyclingDynamicsDomain(observed)
  const height = 30
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const py = (value: number): number =>
    height -
    ((Math.min(domainMax, Math.max(domainMin, value)) - domainMin) / (domainMax - domainMin)) *
      (height - 1)
  const middle = Math.round((domainMin + domainMax) / 2)
  const yTicks = [domainMin, middle, domainMax].map(value => ({
    label: `${value}%`,
    vbY: py(value),
  }))
  const view = graphView(d, graphDomain)
  const title: CyclingModeTitle =
    metric === 'pedalSmoothness' ? 'pedal smoothness' : 'torque effectiveness'
  const className = metric === 'pedalSmoothness' ? 'pedal-smoothness' : 'torque-effectiveness'
  const leftHeat = cyclingDynamicsWattsSamples(d, dynamics, left, graphDomain)
  const rightHeat = cyclingDynamicsWattsSamples(d, dynamics, right, graphDomain)
  const heatAvailable = leftHeat.length + rightHeat.length > 0
  const heatmap = heatAvailable
    ? buildWattsHeatmap(
        f,
        d,
        title,
        `tri-${className}-heatmap`,
        className,
        [
          { side: 'left', samples: leftHeat },
          { side: 'right', samples: rightHeat },
        ],
        domainMin,
        domainMax,
        height,
      )
    : null
  const svgEl = f.svg('svg', {
    class: `tri-elev tri-cycling-dynamics-svg tri-${className}-svg`,
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, title),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  for (const [side, values] of [
    ['left', left],
    ['right', right],
  ] as const) {
    const paths = cyclingDynamicsPaths(dynamics, values, maxDistanceKm, py)
    if (paths.bridges)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.bridges,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-cycling-dynamics-line--missing`,
        }),
      )
    if (paths.measured)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.measured,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side}`,
        }),
      )
  }
  f.add(svgEl, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }))
  const wrap = f.el(
    'div',
    `tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-${className}-chart${heatmap ? ' tri-cycling-mode-chart' : ''}`,
    undefined,
    {
      'data-tri-trace': triathlonTraceName(title),
      ...(heatmap ? { 'data-cycling-chart-mode': 'distance' } : {}),
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-cycling-dynamics-summary')
  const leftAverage = cyclingDynamicsAverage(left)
  const rightAverage = cyclingDynamicsAverage(right)
  if (leftAverage != null && rightAverage != null)
    f.add(
      summary,
      f.el(
        'span',
        'tri-elev-range',
        `L ${leftAverage.toFixed(1)}% / R ${rightAverage.toFixed(1)}% avg`,
      ),
    )
  if (!embedded)
    f.add(summary, buildCyclingDynamicsLegend(f, 'left'), buildCyclingDynamicsLegend(f, 'right'))
  f.add(cap, buildCyclingDynamicsTitle(f, title), summary)
  const distanceFrame = axisFrame(
    f,
    svgEl,
    yTicks,
    height,
    distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
    true,
    { top: 0, bottom: height },
  )
  if (!heatmap) {
    f.add(wrap, cap, distanceFrame)
    return wrap
  }
  const distancePane = f.el(
    'div',
    `tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-${className}-pane--distance`,
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(distancePane, distanceFrame)
  const powerPane = f.el(
    'div',
    `tri-cycling-chart-pane tri-cycling-chart-pane--power tri-${className}-pane--power`,
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(f, heatmap.svg, yTicks, height, heatmap.wattsTicks, true, { top: 0, bottom: height }),
  )
  f.add(wrap, cap, distancePane, powerPane)
  return wrap
}

export const buildTorqueEffectivenessChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  buildCyclingDynamicsPercentChart(f, d, 'torqueEffectiveness', selection, embedded, graphDomain)

export const buildPedalSmoothnessChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  embedded = false,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  buildCyclingDynamicsPercentChart(f, d, 'pedalSmoothness', selection, embedded, graphDomain)

export const buildPowerPhaseChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || cyclingDynamicsSampleCount(dynamics) < 2) return null
  const series = [
    ['left', 'start', cyclingDynamicsValues(dynamics, 'leftPowerPhaseStart')],
    ['left', 'end', cyclingDynamicsValues(dynamics, 'leftPowerPhaseEnd')],
    ['right', 'start', cyclingDynamicsValues(dynamics, 'rightPowerPhaseStart')],
    ['right', 'end', cyclingDynamicsValues(dynamics, 'rightPowerPhaseEnd')],
  ] as const
  if (series.flatMap(([, , values]) => values).filter(validCyclingDynamicsValue).length < 2)
    return null
  const height = 36
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const py = (value: number): number => height - (Math.min(360, Math.max(0, value)) / 360) * height
  const yTicks = [0, 90, 180, 270, 360].map(value => ({ label: `${value}°`, vbY: py(value) }))
  const view = graphView(d, graphDomain)
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-cycling-dynamics-svg tri-power-phase-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'power phase'),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  for (const [side, boundary, values] of series) {
    const paths = cyclingDynamicsPaths(dynamics, values, maxDistanceKm, py, true)
    if (paths.bridges)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.bridges,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-power-phase-line--${boundary} tri-cycling-dynamics-line--missing`,
        }),
      )
    if (paths.measured)
      f.add(
        svgEl,
        f.svg('path', {
          d: paths.measured,
          class: `tri-cycling-dynamics-line tri-cycling-dynamics-line--${side} tri-power-phase-line--${boundary}`,
        }),
      )
  }
  f.add(svgEl, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }))
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-power-phase-chart',
    undefined,
    { 'data-tri-trace': triathlonTraceName('power phase') },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-cycling-dynamics-summary tri-power-phase-summary')
  for (const side of ['left', 'right'] as const)
    for (const boundary of ['start', 'end'] as const)
      f.add(summary, buildCyclingDynamicsLegend(f, side, boundary))
  f.add(cap, buildCyclingDynamicsTitle(f, 'power phase'), summary)
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}

export const buildRiderPositionChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const dynamics = d.cyclingDynamics
  if (!dynamics || dynamics.positionChanges.length === 0) return null
  const height = 24
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const px = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * 100
  const py = (position: 'seated' | 'standing'): number => (position === 'standing' ? 4 : 20)
  const yTicks = [
    { label: triText(f.presentation.locale, 'standing'), vbY: py('standing') },
    { label: triText(f.presentation.locale, 'seated'), vbY: py('seated') },
  ]
  const changes = dynamics.positionChanges
    .filter(change => change.distanceKm >= 0 && change.distanceKm <= maxDistanceKm)
    .sort((left, right) => left.distanceKm - right.distanceKm)
  if (changes.length === 0) return null
  const view = graphView(d, graphDomain)
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-cycling-dynamics-svg tri-rider-position-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': triText(f.presentation.locale, 'rider position'),
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: 100, y2: tick.vbY }),
    )
  for (let index = 0; index < changes.length; index++) {
    if (changes[index].position !== 'standing') continue
    const start = px(changes[index].distanceKm)
    const end = px(changes[index + 1]?.distanceKm ?? maxDistanceKm)
    if (end <= start) continue
    f.add(
      svgEl,
      f.svg('rect', {
        class: 'tri-rider-position-standing',
        x: start.toFixed(2),
        y: 0,
        width: (end - start).toFixed(2),
        height,
      }),
    )
  }
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  let path = `M ${px(changes[0].distanceKm).toFixed(2)} ${py(changes[0].position).toFixed(2)} `
  for (let index = 1; index < changes.length; index++) {
    const x = px(changes[index].distanceKm)
    path += `L ${x.toFixed(2)} ${py(changes[index - 1].position).toFixed(2)} L ${x.toFixed(2)} ${py(changes[index].position).toFixed(2)} `
  }
  path += `L 100 ${py(changes[changes.length - 1].position).toFixed(2)}`
  f.add(
    svgEl,
    f.svg('path', { d: path, class: 'tri-rider-position-line' }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el(
    'div',
    'tri-zone tri-elev-wrap tri-cycling-dynamics-chart tri-rider-position-chart',
    undefined,
    { 'data-tri-trace': triathlonTraceName('rider position') },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const standing = dynamics.standingTimeS
  const seated = dynamics.seatedTimeS
  const total = (standing ?? 0) + (seated ?? 0)
  const summary =
    standing == null
      ? null
      : `${triText(f.presentation.locale, 'standing')} ${zoneClock(standing)}${total > 0 ? ` · ${((standing / total) * 100).toFixed(1)}%` : ''}`
  f.add(
    cap,
    f.el('span', 'tri-elev-d', triText(f.presentation.locale, 'rider position')),
    ...(summary ? [f.el('span', 'tri-elev-range', summary)] : []),
  )
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}

export type GearShiftHover = ActivityGearShift & { index: number; xPct: number }

export function gearShiftAtFraction(
  shifts: readonly ActivityGearShift[],
  maxDistanceKm: number,
  fraction: number,
): GearShiftHover | null {
  if (shifts.length === 0 || !Number.isFinite(maxDistanceKm) || maxDistanceKm <= 0) return null
  const normalized = Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0
  const distanceKm = normalized * maxDistanceKm
  let low = 0
  let high = shifts.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (shifts[middle].distanceKm <= distanceKm) low = middle + 1
    else high = middle
  }
  const index = Math.min(shifts.length - 1, Math.max(0, low - 1))
  return { ...shifts[index], index, xPct: normalized * 100 }
}

const staminaSeriesPath = (
  d: StravaActivityDetail,
  pick: (point: StravaActivityDetail['route'][number]) => number | null,
  closeArea: boolean,
): string => {
  const width = 100
  const height = 30
  const maxDistanceKm = d.route.at(-1)?.d || 1
  const px = (distanceKm: number): number => (distanceKm / maxDistanceKm) * width
  const py = (value: number): number => height - (Math.min(100, Math.max(0, value)) / 100) * height
  let path = ''
  let segmentStart = -1
  const closeSegment = (start: number, end: number): void => {
    const first = pick(d.route[start])
    if (first == null) return
    const firstX = px(d.route[start].d)
    if (closeArea)
      path += `M ${firstX.toFixed(2)} ${height} L ${firstX.toFixed(2)} ${py(first).toFixed(2)} `
    else path += `M ${firstX.toFixed(2)} ${py(first).toFixed(2)} `
    for (let index = start + 1; index <= end; index++) {
      const value = pick(d.route[index])
      if (value == null) continue
      path += `L ${px(d.route[index].d).toFixed(2)} ${py(value).toFixed(2)} `
    }
    if (closeArea) path += `L ${px(d.route[end].d).toFixed(2)} ${height} Z `
  }
  d.route.forEach((point, index) => {
    const value = pick(point)
    const valid = value != null && Number.isFinite(value) && value >= 0 && value <= 100
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === d.route.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  return path
}

export const buildStaminaChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const points = d.route.filter(point => point.stamina != null && point.potentialStamina != null)
  if (d.sport !== 'bike' || points.length < 2) return null
  const width = 100
  const height = 30
  const view = graphView(d, graphDomain)
  const yTicks = [0, 25, 50, 75, 100].map(value => ({
    label: `${value}%`,
    vbY: height - (value / 100) * height,
  }))
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-stamina-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  for (const tick of yTicks)
    f.add(
      svgEl,
      f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: width, y2: tick.vbY }),
    )
  f.add(
    svgEl,
    f.svg('path', {
      d: staminaSeriesPath(d, point => point.stamina, true),
      class: 'tri-stamina-area',
    }),
  )
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  f.add(
    svgEl,
    f.svg('path', {
      d: staminaSeriesPath(d, point => point.stamina, false),
      class: 'tri-stamina-line tri-stamina-line--current',
    }),
    f.svg('path', {
      d: staminaSeriesPath(d, point => point.potentialStamina, false),
      class: 'tri-stamina-line tri-stamina-line--potential',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el('div', 'tri-zone tri-elev-wrap tri-stamina-chart', undefined, {
    'data-tri-trace': triathlonTraceName('stamina'),
  })
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  f.add(cap, f.el('span', 'tri-elev-d', triText(f.presentation.locale, 'stamina')))
  for (const kind of ['current', 'potential'] as const) {
    const item = f.el('span', `tri-stamina-legend-item tri-stamina-legend-item--${kind}`)
    f.add(
      item,
      f.el('span', 'tri-stamina-legend-line', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-stamina-legend-label', triText(f.presentation.locale, kind)),
    )
    f.add(cap, item)
  }
  f.add(
    wrap,
    cap,
    axisFrame(
      f,
      svgEl,
      yTicks,
      height,
      distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
      true,
      { top: 0, bottom: height },
    ),
  )
  return wrap
}

function sampledGearTicks(values: readonly number[], limit = 4): number[] {
  const unique = [...new Set(values)].sort((left, right) => left - right)
  if (unique.length <= limit) return unique
  return Array.from(
    new Set(
      Array.from(
        { length: limit },
        (_, index) => unique[Math.round((index / (limit - 1)) * (unique.length - 1))],
      ),
    ),
  )
}

function gearY(value: number, min: number, max: number, height: number): number {
  return max > min ? height - ((value - min) / (max - min)) * (height - 1) : height / 2
}

function gearStepPath(
  shifts: readonly ActivityGearShift[],
  read: (shift: ActivityGearShift) => number,
  y: (value: number) => number,
  maxDistanceKm: number,
  width: number,
): string {
  const x = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * width
  let path = `M 0 ${y(read(shifts[0])).toFixed(2)}`
  let previous = read(shifts[0])
  for (let index = 1; index < shifts.length; index++) {
    const current = read(shifts[index])
    const nextX = x(shifts[index].distanceKm).toFixed(2)
    path += ` L ${nextX} ${y(previous).toFixed(2)} L ${nextX} ${y(current).toFixed(2)}`
    previous = current
  }
  return `${path} L ${width} ${y(previous).toFixed(2)}`
}

interface GearPairingDuration {
  frontTeeth: number
  rearTeeth: number
  durationS: number
}

function gearPairingDurations(
  shifts: readonly ActivityGearShift[],
  workoutEndElapsedS: number,
): GearPairingDuration[] {
  if (shifts.length === 0 || !Number.isFinite(workoutEndElapsedS) || workoutEndElapsedS <= 0)
    return []
  const endElapsedS = Math.max(workoutEndElapsedS, shifts.at(-1)?.elapsedS ?? 0)
  const durations = new Map<string, GearPairingDuration>()
  for (let index = 0; index < shifts.length; index++) {
    const shift = shifts[index]
    const next = shifts[index + 1]
    const startElapsedS = index === 0 ? 0 : Math.min(endElapsedS, Math.max(0, shift.elapsedS))
    const nextElapsedS = next
      ? Math.min(endElapsedS, Math.max(startElapsedS, next.elapsedS))
      : endElapsedS
    const durationS = nextElapsedS - startElapsedS
    if (durationS <= 0) continue
    const key = `${shift.frontTeeth}:${shift.rearTeeth}`
    const existing = durations.get(key)
    if (existing) existing.durationS += durationS
    else durations.set(key, { frontTeeth: shift.frontTeeth, rearTeeth: shift.rearTeeth, durationS })
  }
  return [...durations.values()]
}

function dominantGearPairing(
  shifts: readonly ActivityGearShift[],
  workoutEndElapsedS: number,
): GearPairingDuration | null {
  let dominant: GearPairingDuration | null = null
  for (const pairing of gearPairingDurations(shifts, workoutEndElapsedS))
    if (!dominant || pairing.durationS > dominant.durationS) dominant = pairing
  return dominant
}

interface ShiftingWattsRow {
  key: string
  frontTeeth: number
  rearTeeth: number
  ratio: number
}

const shiftingWattsData = (
  d: StravaActivityDetail,
  shifts: readonly ActivityGearShift[],
  graphDomain?: ActivityGraphDomain | null,
): { rows: ShiftingWattsRow[]; samples: WattsHeatSample[] } => {
  const observed: { key: string; watts: number }[] = []
  const pairings = new Map<string, ShiftingWattsRow>()
  let shiftIndex = 0
  for (const point of d.route) {
    while (shiftIndex + 1 < shifts.length && shifts[shiftIndex + 1].elapsedS <= point.elapsedS)
      shiftIndex++
    if (
      point.w <= 0 ||
      (graphDomain &&
        (point.d < graphDomain.startDistanceKm || point.d > graphDomain.endDistanceKm))
    )
      continue
    const shift = shifts[shiftIndex]
    if (!shift || shift.frontTeeth <= 0 || shift.rearTeeth <= 0) continue
    const key = `${shift.frontTeeth}:${shift.rearTeeth}`
    pairings.set(key, {
      key,
      frontTeeth: shift.frontTeeth,
      rearTeeth: shift.rearTeeth,
      ratio: shift.frontTeeth / shift.rearTeeth,
    })
    observed.push({ key, watts: point.w })
  }
  const rows = [...pairings.values()].sort(
    (left, right) =>
      left.ratio - right.ratio ||
      left.frontTeeth - right.frontTeeth ||
      right.rearTeeth - left.rearTeeth,
  )
  const rowIndex = new Map(rows.map((row, index) => [row.key, index]))
  return {
    rows,
    samples: observed.flatMap(sample => {
      const index = rowIndex.get(sample.key)
      if (index == null) return []
      return [{ watts: sample.watts, value: rows.length === 1 ? 0.5 : index }]
    }),
  }
}

const shiftingWattsTicks = (
  rows: readonly ShiftingWattsRow[],
  height: number,
  limit = 4,
): { label: string; vbY: number }[] => {
  if (rows.length === 0) return []
  const indices =
    rows.length <= limit
      ? rows.map((_, index) => index)
      : Array.from(
          new Set(
            Array.from({ length: limit }, (_, index) =>
              Math.round((index / (limit - 1)) * (rows.length - 1)),
            ),
          ),
        )
  return indices.map(index => ({
    label: `${rows[index].frontTeeth}×${rows[index].rearTeeth}`,
    vbY: rows.length === 1 ? height / 2 : height - (index / (rows.length - 1)) * (height - 1),
  }))
}

export interface ActivityGearRatioDistributionPoint {
  ratio: number
  percentage: number
}

export const activityGearRatioDistribution = (
  activity: StravaActivityDetail,
): ActivityGearRatioDistributionPoint[] => {
  if (activity.sport !== 'bike') return []
  const workoutEndElapsedS = Math.max(activity.movingTimeS, activity.route.at(-1)?.elapsedS ?? 0)
  const durations = new Map<number, number>()
  for (const pairing of gearPairingDurations(activity.gearShifts, workoutEndElapsedS)) {
    if (
      !Number.isFinite(pairing.frontTeeth) ||
      !Number.isFinite(pairing.rearTeeth) ||
      pairing.frontTeeth <= 0 ||
      pairing.rearTeeth <= 0
    )
      continue
    const ratio = Number((pairing.frontTeeth / pairing.rearTeeth).toFixed(6))
    durations.set(ratio, (durations.get(ratio) ?? 0) + pairing.durationS)
  }
  const totalDurationS = [...durations.values()].reduce((total, durationS) => total + durationS, 0)
  if (totalDurationS <= 0) return []
  return [...durations.entries()]
    .sort(([left], [right]) => left - right)
    .map(([ratio, durationS]) => ({ ratio, percentage: (durationS / totalDurationS) * 100 }))
}

export const buildShiftingChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const shifts = d.gearShifts
  if (d.sport !== 'bike' || shifts.length === 0) return null
  const width = 100
  const height = 30
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const frontValues = shifts.map(shift => shift.frontTeeth)
  const rearValues = shifts.map(shift => shift.rearTeeth)
  const frontMin = Math.min(...frontValues)
  const frontMax = Math.max(...frontValues)
  const rearMin = Math.min(...rearValues)
  const rearMax = Math.max(...rearValues)
  const dominant = dominantGearPairing(
    shifts,
    Math.max(d.movingTimeS, d.route.at(-1)?.elapsedS ?? 0),
  )
  const view = graphViewForDistance(maxDistanceKm, graphDomain)
  const frontY = (value: number): number => gearY(value, frontMin, frontMax, height)
  const rearY = (value: number): number => gearY(value, rearMin, rearMax, height)
  const wattsData = shiftingWattsData(d, shifts, graphDomain)
  const heatAvailable = wattsData.samples.length > 0 && wattsData.rows.length > 0
  const heatmap = heatAvailable
    ? buildWattsHeatmap(
        f,
        d,
        'electronic shifting',
        'tri-shift-heatmap',
        'electronic-shifting',
        [{ side: 'single', samples: wattsData.samples }],
        0,
        wattsData.rows.length === 1 ? 1 : wattsData.rows.length - 1,
        height,
      )
    : null
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-shift-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  f.add(
    svgEl,
    f.svg('path', {
      d: gearStepPath(shifts, shift => shift.frontTeeth, frontY, maxDistanceKm, width),
      class: 'tri-shift-line tri-shift-line--front',
    }),
    f.svg('path', {
      d: gearStepPath(shifts, shift => shift.rearTeeth, rearY, maxDistanceKm, width),
      class: 'tri-shift-line tri-shift-line--rear',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el(
    'div',
    `tri-zone tri-elev-wrap tri-shift-chart${heatmap ? ' tri-cycling-mode-chart' : ''}`,
    undefined,
    {
      'data-tri-trace': triathlonTraceName('electronic shifting'),
      ...(heatmap ? { 'data-cycling-chart-mode': 'distance' } : {}),
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-shift-summary')
  if (dominant)
    f.add(
      summary,
      f.el(
        'span',
        'tri-elev-range',
        `${dominant.frontTeeth}×${dominant.rearTeeth} · ${zoneClock(dominant.durationS)}`,
      ),
    )
  for (const kind of ['front', 'rear'] as const) {
    const item = f.el('span', `tri-shift-legend-item tri-shift-legend-item--${kind}`)
    f.add(
      item,
      f.el('span', 'tri-shift-legend-line', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-shift-legend-label', kind, { 'data-i18n': kind }),
    )
    f.add(summary, item)
  }
  f.add(
    cap,
    f.el('span', 'tri-elev-d', 'electronic shifting', { 'data-i18n': 'electronic shifting' }),
    summary,
  )
  const distanceFrame = axisFrame(
    f,
    svgEl,
    sampledGearTicks(frontValues).map(value => ({ label: `${value}T`, vbY: frontY(value) })),
    height,
    distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
    true,
    { top: 0, bottom: height },
  )
  if (!heatmap) {
    f.add(wrap, cap, distanceFrame)
    return wrap
  }
  const distancePane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-shift-pane--distance',
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(distancePane, distanceFrame)
  const powerPane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--power tri-shift-pane--power',
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(
      f,
      heatmap.svg,
      shiftingWattsTicks(wattsData.rows, height),
      height,
      heatmap.wattsTicks,
      true,
      { top: 0, bottom: height },
    ),
  )
  f.add(wrap, cap, distancePane, powerPane)
  return wrap
}

export const buildRunStrideTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const imperial = isImperial(f.presentation)
  const label = runStrideLengthLabel(d)
  const valuesM = d.route
    .map(point => runStrideLengthValue(d, point))
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || valuesM.length < 2) return null
  const displayValue = (meters: number): number => (imperial ? meters * M_TO_FT : meters)
  const values = valuesM.map(displayValue)
  const averageM = valuesM.reduce((total, value) => total + value, 0) / valuesM.length
  const step = imperial ? 0.5 : 0.25
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  const unit = imperial ? 'ft' : 'm'
  return buildTrace(
    f,
    d,
    point => {
      const meters = runStrideLengthValue(d, point)
      return meters == null ? null : displayValue(meters)
    },
    label,
    () => `${formatStrideLength(f.presentation, averageM)} avg`,
    value => `${value.toFixed(1)}${unit}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildRunGroundContactTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const values = d.route
    .map(runGroundContactTimeMs)
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average = values.reduce((total, value) => total + value, 0) / values.length
  const step = 25
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    runGroundContactTimeMs,
    'ground contact time',
    () => `${formatGroundContactTime(average)} avg`,
    value => `${Math.round(value)}ms`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildRunVerticalOscillationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const values = d.route
    .map(runVerticalOscillationCm)
    .filter((value): value is number => value != null)
  if (d.sport !== 'run' || values.length < 2) return null
  const average = values.reduce((total, value) => total + value, 0) / values.length
  const step = 1
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    runVerticalOscillationCm,
    'vertical oscillation',
    () => `${formatVerticalOscillation(f.presentation, average)} avg`,
    value => `${value.toFixed(1)}cm`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const imperial = isImperial(f.presentation)
  const temperaturesC = d.route
    .map(point => point.tempC)
    .filter((value): value is number => value != null)
  const averageC =
    d.avgTemp ?? temperaturesC.reduce((total, value) => total + value, 0) / temperaturesC.length
  const values = temperaturesC.map(value => temperatureValue(f.presentation, value))
  const step = imperial ? 5 : 2
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    point => temperatureValue(f.presentation, point.tempC ?? averageC),
    'temperature',
    () => `${formatTemperature(f.presentation, averageC)} avg`,
    value => `${Math.round(value)}${temperatureUnit(f.presentation)}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

const traceResolution = (values: number[], fallback: number): number => {
  const unique = [...new Set(values)].sort((left, right) => left - right)
  let resolution = Infinity
  for (let index = 1; index < unique.length; index++) {
    const delta = unique[index] - unique[index - 1]
    if (delta > 1e-6 && delta < resolution) resolution = delta
  }
  return Number.isFinite(resolution) ? resolution : fallback
}

const thermalTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pickCelsius: (point: ActivityThermalTracePoint) => number | null,
  title: string,
  fallbackResolutionC: number,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const imperial = isImperial(f.presentation)
  const points = activityThermalTracePoints(d)
  const valuesC = points.map(pickCelsius).filter((value): value is number => value != null)
  if (valuesC.length < 2) return null
  const displayValues = valuesC.map(value => temperatureValue(f.presentation, value))
  const fallbackResolution = imperial ? (fallbackResolutionC * 9) / 5 : fallbackResolutionC
  const resolution = traceResolution(displayValues, fallbackResolution)
  const min = Math.min(...displayValues) - resolution
  const max = Math.max(...displayValues) + resolution
  const averageC = valuesC.reduce((total, value) => total + value, 0) / valuesC.length
  const digits = resolution < 0.1 ? 2 : 1
  return buildTraceSeries(
    f,
    d,
    points,
    point => {
      const celsius = pickCelsius(point)
      return celsius == null ? null : temperatureValue(f.presentation, celsius)
    },
    title,
    () => `${formatThermalTemperature(f.presentation, averageC)} avg`,
    value => `${value.toFixed(digits)}${temperatureUnit(f.presentation)}`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildHeatStrainTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const points = activityThermalTracePoints(d)
  const values = points
    .map(point => point.heatStrainIndex)
    .filter((value): value is number => value != null)
  if (values.length < 2) return null
  const resolution = traceResolution(values, 0.1)
  const min = Math.max(0, Math.min(...values) - resolution)
  const max = Math.max(...values) + resolution
  const average = values.reduce((total, value) => total + value, 0) / values.length
  return buildTraceSeries(
    f,
    d,
    points,
    point => point.heatStrainIndex,
    'heat strain index',
    () => `${average.toFixed(1)} avg`,
    value => value.toFixed(1),
    { min, max, intervals: 2 },
    selection,
    graphDomain,
    'dotted',
  )
}

export const buildCoreTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  thermalTemperatureTrace(
    f,
    d,
    point => point.coreTemperatureC,
    'CORE temperature',
    0.01,
    selection,
    graphDomain,
  )

export const buildSkinTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null =>
  thermalTemperatureTrace(
    f,
    d,
    point => point.skinTemperatureC,
    'skin temperature',
    0.05,
    selection,
    graphDomain,
  )

export const buildRespirationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const values = d.route
    .map(point => point.resp)
    .filter((value): value is number => value != null && value > 0)
  const average = values.reduce((total, value) => total + value, 0) / values.length
  const step = 5
  let min = Math.floor(Math.min(...values) / step) * step
  let max = Math.ceil(Math.max(...values) / step) * step
  if (max <= min) {
    min -= step
    max += step
  }
  return buildTrace(
    f,
    d,
    point => point.resp ?? average,
    'respiration',
    () => `${formatRespirationRate(average)} avg`,
    value => `${Math.round(value)}brpm`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

export const buildMuscleOxygenTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const values = d.route
    .map(point => point.muscleOxygenPct)
    .filter((value): value is number => value != null && value >= 0 && value <= 100)
  if (values.length < 2) return null
  const average = values.reduce((total, value) => total + value, 0) / values.length
  let min = Math.floor(Math.min(...values) / 5) * 5
  let max = Math.ceil(Math.max(...values) / 5) * 5
  if (max <= min) {
    min = Math.max(0, min - 5)
    max = Math.min(100, max + 5)
  }
  return buildTrace(
    f,
    d,
    point => point.muscleOxygenPct ?? average,
    'muscle oxygen',
    () => `${formatMuscleOxygen(average)} avg`,
    value => `${value.toFixed(1)}%`,
    { min, max, intervals: 2 },
    selection,
    graphDomain,
  )
}

const analysisRangeAttrs = (range: ActivityAnalysisRange): Record<string, string> => {
  const displayDurationS = range.movingTimeS ?? range.durationS
  const attrs: Record<string, string> = {
    type: 'button',
    'data-analysis-range': '',
    'data-range-kind': range.kind,
    'data-range-id': range.id,
    'data-range-label': range.label,
    'data-start-elapsed-s': `${range.startElapsedS}`,
    'data-end-elapsed-s': `${range.endElapsedS}`,
    'data-start-distance-km': `${range.startDistanceKm}`,
    'data-end-distance-km': `${range.endDistanceKm}`,
    'data-duration-s': `${displayDurationS}`,
    'data-distance-km': `${range.distanceKm}`,
  }
  if (range.elevationGainM != null) attrs['data-elevation-gain-m'] = `${range.elevationGainM}`
  if (range.averageSpeedKph != null) attrs['data-average-speed-kph'] = `${range.averageSpeedKph}`
  if (range.averageHeartRate != null) attrs['data-average-heart-rate'] = `${range.averageHeartRate}`
  if (range.averageWatts != null) attrs['data-average-watts'] = `${range.averageWatts}`
  if (range.averageCadence != null) attrs['data-average-cadence'] = `${range.averageCadence}`
  return attrs
}

const analysisRangeRate = (
  presentation: TriathlonPresentation,
  sport: ActivityKind,
  speedKphValue: number,
): string => {
  const imperial = isImperial(presentation)
  if (sport === 'bike') return speedKph(presentation, speedKphValue)
  if (sport === 'swim') return `${clock(360 / speedKphValue)} /100m`
  return `${clock(3600 / (speedKphValue * (imperial ? KM_TO_MI : 1)))} /${imperial ? 'mi' : 'km'}`
}

const analysisRangeMetrics = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  range: ActivityAnalysisRange,
): string[] => {
  const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const cadenceScale = d.sport === 'run' ? 2 : 1
  const values = [scrubDist(presentation, range.distanceKm, d.sport)]
  if (range.elevationGainM != null)
    values.push(`+${formatElevationGain(presentation, range.elevationGainM)}`)
  values.push(clock(range.movingTimeS ?? range.durationS))
  if (range.averageSpeedKph != null)
    values.push(analysisRangeRate(presentation, d.sport, range.averageSpeedKph))
  if (range.averageWatts != null) values.push(`${Math.round(range.averageWatts)} W`)
  if (range.averageHeartRate != null) values.push(`${Math.round(range.averageHeartRate)} bpm`)
  if (range.averageCadence != null)
    values.push(`${Math.round(range.averageCadence * cadenceScale)} ${cadenceUnit}`)
  return values
}

type RunLapSplit = {
  range: ActivityAnalysisRange
  index: number
  speedKph: number
  paceS: number
  deltaS: number | null
}

const runPaceSeconds = (presentation: TriathlonPresentation, speedKph: number): number =>
  3600 / (speedKph * (isImperial(presentation) ? KM_TO_MI : 1))

const projectedRunSplits = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): ActivityAnalysisRange[] => {
  const imperial = isImperial(presentation)
  const source = imperial ? (d.runSplitsStandard ?? []) : (d.runSplitsMetric ?? [])
  const ranges: ActivityAnalysisRange[] = []
  let startDistanceKm = 0
  let startElapsedS = 0
  for (const split of source) {
    if (
      !Number.isFinite(split.distanceKm) ||
      split.distanceKm <= 0 ||
      !Number.isFinite(split.elapsedTimeS) ||
      split.elapsedTimeS <= 0 ||
      !Number.isFinite(split.movingTimeS) ||
      split.movingTimeS <= 0 ||
      !Number.isFinite(split.averageSpeedKph) ||
      split.averageSpeedKph <= 0
    )
      continue
    const endDistanceKm = startDistanceKm + split.distanceKm
    const endElapsedS = startElapsedS + split.elapsedTimeS
    ranges.push({
      kind: 'lap',
      id: `split:${imperial ? 'standard' : 'metric'}:${split.split}`,
      label: `Split ${split.split}`,
      startElapsedS,
      endElapsedS,
      startDistanceKm,
      endDistanceKm,
      durationS: split.movingTimeS,
      distanceKm: split.distanceKm,
      elevationGainM: null,
      averageSpeedKph: split.averageSpeedKph,
      averageHeartRate: null,
      averageWatts: null,
      averageCadence: null,
    })
    startDistanceKm = endDistanceKm
    startElapsedS = endElapsedS
  }
  return ranges
}

const runLapSplits = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): RunLapSplit[] => {
  const nativeSplits = projectedRunSplits(presentation, d)
  const ranges =
    nativeSplits.length > 0
      ? nativeSplits
      : validAnalysisRanges(d).filter(candidate => candidate.kind === 'lap')
  const splits: RunLapSplit[] = []
  let previousPaceS: number | null = null
  for (const [index, range] of ranges.entries()) {
    const speedKph =
      range.averageSpeedKph != null && range.averageSpeedKph > 0
        ? range.averageSpeedKph
        : (range.distanceKm / (range.movingTimeS ?? range.durationS)) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    const paceS = runPaceSeconds(presentation, speedKph)
    splits.push({
      range,
      index: index + 1,
      speedKph,
      paceS,
      deltaS: previousPaceS == null ? null : previousPaceS - paceS,
    })
    previousPaceS = paceS
  }
  return splits
}

const paceDelta = (seconds: number | null): string => {
  if (seconds == null) return '—'
  const rounded = Math.round(seconds)
  if (rounded === 0) return '0:00'
  return `${rounded > 0 ? '+' : '−'}${clock(Math.abs(rounded))}`
}

const runWorkoutLaps = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): RunLapSplit[] => {
  const laps: RunLapSplit[] = []
  for (const [index, range] of validAnalysisRanges(d)
    .filter(candidate => candidate.kind === 'lap')
    .entries()) {
    const speedKph =
      range.averageSpeedKph != null && range.averageSpeedKph > 0
        ? range.averageSpeedKph
        : (range.distanceKm / (range.movingTimeS ?? range.durationS)) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    laps.push({
      range,
      index: index + 1,
      speedKph,
      paceS: runPaceSeconds(presentation, speedKph),
      deltaS: null,
    })
  }
  return laps
}

const runWorkoutPaceAxis = (paces: number[]): { min: number; max: number; ticks: number[] } => {
  const fastest = Math.min(...paces)
  const slowest = Math.max(...paces)
  const targetStep = Math.max(1, (slowest - fastest) / 5)
  const step = [30, 60, 120, 180, 300, 600, 900].find(candidate => candidate >= targetStep) ?? 900
  const min = Math.max(30, Math.floor(fastest / step) * step)
  let max = Math.ceil(slowest / step) * step
  if (max <= min) max = min + step
  const ticks: number[] = []
  for (let value = min; value <= max; value += step) ticks.push(value)
  return { min, max, ticks }
}

export const buildRunWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  showTitle = true,
): N | null => {
  if (d.sport !== 'run') return null
  const laps = runWorkoutLaps(f.presentation, d)
  if (laps.length === 0) return null
  const imperial = isImperial(f.presentation)
  const paceUnit = imperial ? '/mi' : '/km'
  const paceAxis = runWorkoutPaceAxis(laps.map(lap => lap.paceS))
  const paceSpan = paceAxis.max - paceAxis.min
  const speeds = laps.map(lap => lap.speedKph)
  const minSpeedKph = Math.min(...speeds)
  const maxSpeedKph = Math.max(...speeds)
  const speedSpan = maxSpeedKph - minSpeedKph
  const movingTimeS = laps.reduce(
    (total, lap) => total + (lap.range.distanceKm / lap.speedKph) * 3600,
    0,
  )
  const distanceKm = laps.reduce((total, lap) => total + lap.range.distanceKm, 0)
  const averageSpeedKph = (distanceKm / movingTimeS) * 3600
  const fastestPaceS = Math.min(...laps.map(lap => lap.paceS))
  const slowestPaceS = Math.max(...laps.map(lap => lap.paceS))

  const wrap = f.el('section', 'tri-run-workout', undefined, {
    'aria-label': 'Run workout analysis',
  })
  const head = f.el('div', 'tri-run-workout-head')
  const stats = f.el('div', 'tri-run-workout-stats')
  f.add(
    stats,
    f.el('span', undefined, `fastest ${clock(fastestPaceS)} ${paceUnit}`),
    f.el(
      'span',
      undefined,
      `avg ${clock(runPaceSeconds(f.presentation, averageSpeedKph))} ${paceUnit}`,
    ),
    f.el('span', undefined, `slowest ${clock(slowestPaceS)} ${paceUnit}`),
  )
  if (showTitle) f.add(head, f.el('span', 'tri-run-workout-title', 'workout analysis'))
  f.add(head, stats)

  const chart = f.el('div', 'tri-run-workout-chart')
  const yAxis = f.el('div', 'tri-run-workout-y-axis', undefined, { 'aria-hidden': 'true' })
  const viewport = f.el('div', 'tri-run-workout-viewport')
  const plot = f.el('div', 'tri-run-workout-plot', undefined, {
    style: `--tri-run-workout-laps:${laps.length}`,
  })
  const grid = f.el('div', 'tri-run-workout-grid', undefined, { 'aria-hidden': 'true' })
  const bars = f.el('div', 'tri-run-workout-bars')
  for (const tick of paceAxis.ticks) {
    const top = ((tick - paceAxis.min) / paceSpan) * 100
    f.add(
      yAxis,
      f.el('span', 'tri-run-workout-y-tick', clock(tick), { style: `top:${top.toFixed(3)}%` }),
    )
    f.add(
      grid,
      f.el('span', 'tri-run-workout-gridline', undefined, { style: `top:${top.toFixed(3)}%` }),
    )
  }
  for (const lap of laps) {
    const metrics = analysisRangeMetrics(f.presentation, d, lap.range)
    const attrs = analysisRangeAttrs(lap.range)
    const height = Math.max(3, ((paceAxis.max - lap.paceS) / paceSpan) * 100)
    const intensity = speedSpan > 0 ? 0.42 + ((lap.speedKph - minSpeedKph) / speedSpan) * 0.5 : 0.72
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] = `${lap.range.label}, ${metrics.join(', ')}`
    attrs.style = `--tri-run-workout-height:${height.toFixed(3)}%;--tri-run-workout-opacity:${intensity.toFixed(3)}`
    const button = f.el('button', 'tri-run-workout-lap', undefined, attrs)
    const column = f.el('span', 'tri-run-workout-column', undefined, { 'aria-hidden': 'true' })
    f.add(
      column,
      f.el('span', 'tri-run-workout-bar', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-run-workout-pace', `${clock(lap.paceS)} ${paceUnit}`, {
        'aria-hidden': 'true',
      }),
    )
    f.add(
      button,
      column,
      f.el('span', 'tri-run-workout-label', `${lap.index}`, { 'aria-hidden': 'true' }),
    )
    f.add(bars, button)
  }
  f.add(plot, grid, bars)
  f.add(viewport, plot)
  f.add(chart, yAxis, viewport)
  f.add(wrap, head, chart)
  return wrap
}

export const buildRunLapSplits = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  showTitle = true,
): N | null => {
  if (d.sport !== 'run') return null
  const imperial = isImperial(f.presentation)
  const splits = runLapSplits(f.presentation, d)
  const wrap = f.el('section', 'tri-run-splits', undefined, { 'aria-label': 'Run lap splits' })
  const head = f.el('div', 'tri-run-splits-head')
  if (showTitle) f.add(head, f.el('span', 'tri-run-splits-title', 'lap splits'))
  if (splits.length === 0) {
    const columns = f.el('div', 'tri-run-splits-columns', undefined, { 'aria-hidden': 'true' })
    const list = f.el('div', 'tri-run-splits-list')
    f.add(list, f.el('span', 'tri-run-splits-empty', 'no lap found'))
    f.add(wrap, head, columns, list)
    return wrap
  }
  const maxSpeedKph = Math.max(...splits.map(split => split.speedKph))
  const totalDistanceKm = splits.reduce((total, split) => total + split.range.distanceKm, 0)
  const totalDurationS = splits.reduce(
    (total, split) => total + (split.range.movingTimeS ?? split.range.durationS),
    0,
  )
  const averageSpeedKph = (totalDistanceKm / totalDurationS) * 3600
  const averagePct = Math.max(0, Math.min(100, (averageSpeedKph / maxSpeedKph) * 100))
  const paceUnit = imperial ? '/mi' : '/km'
  f.add(
    head,
    f.el(
      'span',
      'tri-run-splits-average',
      `avg ${clock(runPaceSeconds(f.presentation, averageSpeedKph))} ${paceUnit}`,
    ),
  )
  const columns = f.el('div', 'tri-run-splits-columns', undefined, { 'aria-hidden': 'true' })
  f.add(
    columns,
    f.el('span', undefined, 'split'),
    f.el('span', undefined, imperial ? 'mi' : 'km'),
    f.el('span', undefined, 'pace'),
    f.el('span', undefined, '+/−'),
  )
  const list = f.el('div', 'tri-run-splits-list')
  for (const split of splits) {
    const metrics = analysisRangeMetrics(f.presentation, d, split.range)
    const attrs = analysisRangeAttrs(split.range)
    const delta = paceDelta(split.deltaS)
    attrs['aria-pressed'] = 'false'
    attrs['aria-label'] =
      `${split.range.label}, ${metrics.join(', ')}, ${delta === '—' ? 'first lap' : `${delta} versus previous lap`}`
    attrs.style = `--tri-run-split-width:${Math.max(24, (split.speedKph / maxSpeedKph) * 100).toFixed(3)}%;--tri-run-split-average:${averagePct.toFixed(3)}%`
    const button = f.el('button', 'tri-run-split', undefined, attrs)
    const track = f.el('span', 'tri-run-split-track')
    f.add(
      track,
      f.el('span', 'tri-run-split-fill', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-run-split-average-marker', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-run-split-pace', `${clock(split.paceS)} ${paceUnit}`),
    )
    const deltaClass =
      split.deltaS == null || Math.round(split.deltaS) === 0
        ? 'tri-run-split-delta'
        : `tri-run-split-delta tri-run-split-delta--${split.deltaS > 0 ? 'faster' : 'slower'}`
    f.add(
      button,
      f.el('span', 'tri-run-split-lap', `${split.index}`),
      f.el(
        'span',
        'tri-run-split-distance',
        imperial
          ? (split.range.distanceKm * KM_TO_MI).toFixed(2)
          : split.range.distanceKm.toFixed(2),
      ),
      track,
      f.el('span', deltaClass, delta),
    )
    f.add(list, button)
  }
  f.add(wrap, head, columns, list)
  return wrap
}

const runPaceZoneRangeText = (
  presentation: TriathlonPresentation,
  boundsSPerKm: readonly number[],
  index: number,
): string => {
  const range = runPaceZoneRange(boundsSPerKm, index)
  if (!range) return '—'
  const imperial = isImperial(presentation)
  const scale = imperial ? KM_TO_MI : 1
  const unit = imperial ? '/mi' : '/km'
  if (range.fastestSPerKm == null)
    return range.slowestSPerKm == null ? '—' : `<${clock(range.slowestSPerKm / scale)}${unit}`
  if (range.slowestSPerKm == null) return `>${clock(range.fastestSPerKm / scale)}${unit}`
  return `${clock(range.fastestSPerKm / scale)}–${clock(range.slowestSPerKm / scale)}${unit}`
}

export const buildRunPaceDistribution = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  showTitle = true,
): N | null => {
  const distribution = d.runPaceZones
  if (
    d.sport !== 'run' ||
    !distribution ||
    distribution.boundsSPerKm.length !== 5 ||
    distribution.zoneSeconds.length !== 6 ||
    distribution.tenKmRaceTimeS <= 0 ||
    distribution.boundsSPerKm.some(value => !Number.isFinite(value) || value <= 0) ||
    distribution.zoneSeconds.some(value => !Number.isFinite(value) || value < 0)
  )
    return null
  const total = distribution.zoneSeconds.reduce((sum, seconds) => sum + seconds, 0)
  if (!(total > 0)) return null
  let majority = 0
  for (let index = 1; index < distribution.zoneSeconds.length; index++)
    if (distribution.zoneSeconds[index] > distribution.zoneSeconds[majority]) majority = index
  const maximum = Math.max(...distribution.zoneSeconds, 1)
  const wrap = f.el('section', 'tri-run-pace-distribution', undefined, {
    'aria-label': 'Run pace distribution',
  })
  const head = f.el('div', 'tri-run-pace-distribution-head')
  if (showTitle) f.add(head, f.el('span', 'tri-run-pace-distribution-title', 'pace distribution'))
  const summary = f.el('div', 'tri-training-zone-summary', undefined, { 'aria-live': 'polite' })
  f.add(
    summary,
    f.el(
      'strong',
      'tri-training-zone-summary-value',
      `${Math.round((distribution.zoneSeconds[majority] / total) * 100)}% in zone ${majority + 1}`,
    ),
    f.el('span', 'tri-training-zone-summary-time', zoneClock(total)),
  )
  f.add(head, summary)
  const grid = f.el('div', 'tri-training-zone-grid tri-training-zone-grid--pace', undefined, {
    role: 'list',
    'aria-label': 'pace zone distribution',
    style: `--tri-zone-count:${distribution.zoneSeconds.length}`,
  })
  for (let index = distribution.zoneSeconds.length - 1; index >= 0; index--) {
    const seconds = distribution.zoneSeconds[index]
    const percentage = (seconds / total) * 100
    const range = runPaceZoneRangeText(f.presentation, distribution.boundsSPerKm, index)
    const name = RUN_PACE_ZONE_NAMES[index] ?? `zone ${index + 1}`
    const row = f.el(
      'div',
      `tri-training-zone-row${index === majority ? ' tri-training-zone-row--majority' : ''}`,
      undefined,
      {
        role: 'listitem',
        tabindex: '0',
        'aria-label': `Z${index + 1} ${name}, ${zoneClock(seconds)}, ${percentage.toFixed(1)}%, ${range}`,
      },
    )
    const bar = f.el('span', 'tri-training-zone-bar', undefined, { 'aria-hidden': 'true' })
    f.add(
      bar,
      f.el(
        'span',
        `tri-training-zone-fill tri-training-zone-fill--pace tri-training-zone-fill--${index + 1}`,
        undefined,
        { style: `--tri-zone-share:${(seconds / maximum) * 100}%` },
      ),
    )
    const visual = f.el('span', 'tri-training-zone-visual')
    f.add(
      visual,
      bar,
      f.el('span', 'tri-training-zone-pct', `${percentage.toFixed(1)}%`, { 'aria-hidden': 'true' }),
    )
    f.add(
      row,
      f.el('span', 'tri-training-zone-name', `Z${index + 1}`, { 'aria-hidden': 'true' }),
      visual,
      f.el('span', 'tri-training-zone-range', range, { 'aria-hidden': 'true' }),
    )
    f.add(grid, row)
  }
  f.add(
    wrap,
    head,
    grid,
    f.el(
      'div',
      'tri-dist-cap tri-training-zone-source',
      `based on 10 km race time ${clock(distribution.tenKmRaceTimeS)}`,
    ),
  )
  return wrap
}

export const buildRunAnalysis = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const workout = buildRunWorkoutAnalysis(f, d, false)
  const splits = buildRunLapSplits(f, d, false)
  const pace = buildRunPaceDistribution(f, d, false)
  const views = [
    { key: 'workout', label: 'workout analysis', content: workout },
    { key: 'laps', label: 'lap splits', content: splits },
    { key: 'pace', label: 'pace distribution', content: pace },
  ].filter((view): view is { key: string; label: string; content: N } => view.content != null)
  if (views.length === 0) return null
  if (views.length === 1)
    return views[0].key === 'workout'
      ? buildRunWorkoutAnalysis(f, d)
      : views[0].key === 'laps'
        ? buildRunLapSplits(f, d)
        : buildRunPaceDistribution(f, d)

  const id = `tri-run-analysis-${d.id}`
  const selected = views[0].key
  const wrap = f.el('section', 'tri-run-analysis', undefined, {
    'aria-label': 'Run analysis',
    'data-run-analysis': '',
    'data-run-analysis-view': selected,
  })
  const tabs = f.el('div', 'tri-map-tablist tri-run-analysis-tabs', undefined, {
    role: 'tablist',
    'aria-label': 'Run analysis view',
  })
  const stage = f.el('div', 'tri-run-analysis-stage')
  for (const view of views) {
    const active = view.key === selected
    f.add(
      tabs,
      f.el('button', 'tri-map-tab tri-run-analysis-tab', view.label, {
        id: `${id}-${view.key}-tab`,
        type: 'button',
        role: 'tab',
        tabindex: active ? '0' : '-1',
        'aria-controls': `${id}-${view.key}-panel`,
        'aria-selected': String(active),
        'data-run-analysis-tab': view.key,
      }),
    )
    const panel = f.el('div', 'tri-run-analysis-panel', undefined, {
      id: `${id}-${view.key}-panel`,
      role: 'tabpanel',
      ...(active ? {} : { hidden: '' }),
      'aria-hidden': String(!active),
      'aria-labelledby': `${id}-${view.key}-tab`,
      'data-run-analysis-panel': view.key,
    })
    f.add(panel, view.content)
    f.add(stage, panel)
  }
  f.add(wrap, tabs, stage)
  return wrap
}

type PositionedAnalysisRange = { range: ActivityAnalysisRange; lane: number }

const positionAnalysisRanges = (
  ranges: ActivityAnalysisRange[],
  laneLimit: number,
): PositionedAnalysisRange[] => {
  const laneEnds = Array.from({ length: laneLimit }, () => Number.NEGATIVE_INFINITY)
  return ranges
    .slice()
    .sort((a, b) => a.startDistanceKm - b.startDistanceKm || a.endDistanceKm - b.endDistanceKm)
    .map(range => {
      let lane = laneEnds.findIndex(end => end <= range.startDistanceKm)
      if (lane < 0) {
        lane = 0
        for (let index = 1; index < laneEnds.length; index++)
          if (laneEnds[index] < laneEnds[lane]) lane = index
      }
      laneEnds[lane] = Math.max(laneEnds[lane], range.endDistanceKm)
      return { range, lane }
    })
}

export const buildAnalysisBar = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const ranges = validAnalysisRanges(d)
  if (!hasAnalysisWorkspace(d)) return null
  const wrapAttrs: Record<string, string> = {
    'data-tri-analysis': '',
    'data-activity-id': `${d.id}`,
  }
  if (ranges.length === 0) wrapAttrs['aria-hidden'] = 'true'
  else wrapAttrs['aria-label'] = 'Activity analysis'
  const wrap = f.el('section', 'tri-analysis', undefined, wrapAttrs)
  const readout = f.el('div', 'tri-analysis-readout', undefined, {
    'data-tri-analysis-readout': '',
    'data-visible': 'false',
    'aria-hidden': 'true',
    'aria-live': 'polite',
  })
  f.add(
    readout,
    f.el('span', 'tri-analysis-readout-label'),
    f.el('span', 'tri-analysis-readout-metrics'),
  )
  f.add(wrap, readout)

  const rangeBands = f.el('div', 'tri-analysis-ranges')
  const labels: Record<ActivityAnalysisKind, string> = {
    lap: 'Laps',
    climb: 'Climbs',
    segment: 'Segments',
  }
  for (const kind of ANALYSIS_KIND_ORDER) {
    const groupRanges = ranges.filter(range => range.kind === kind)
    const empty = groupRanges.length === 0
    const laneLimit = kind === 'lap' || kind === 'climb' ? 1 : 4
    const positioned = positionAnalysisRanges(groupRanges, laneLimit)
    const bandAttrs: Record<string, string> = { 'data-analysis-kind': kind }
    if (empty) bandAttrs['aria-hidden'] = 'true'
    else {
      bandAttrs.role = 'group'
      bandAttrs['aria-label'] = labels[kind]
    }
    const band = f.el('div', 'tri-analysis-band', undefined, bandAttrs)
    const items = f.el('div', 'tri-analysis-band-items', undefined, {
      style: `--tri-analysis-lanes:${laneLimit}`,
    })
    for (const { range, lane } of positioned) {
      const metrics = analysisRangeMetrics(f.presentation, d, range)
      const bounds = analysisSelectionBounds(d, range)
      const attrs = analysisRangeAttrs(range)
      attrs['aria-pressed'] = 'false'
      attrs['aria-label'] = `${range.label}, ${metrics.join(', ')}`
      attrs.style = `--tri-analysis-start:${bounds.x.toFixed(3)}%;--tri-analysis-width:${Math.max(0.18, bounds.width).toFixed(3)}%;--tri-analysis-lane:${lane}`
      f.add(items, f.el('button', 'tri-analysis-range', undefined, attrs))
    }
    f.add(band, f.el('span', 'tri-analysis-band-label', empty ? undefined : labels[kind]), items)
    f.add(rangeBands, band)
  }
  f.add(wrap, rangeBands)
  return wrap
}

export const buildSwimStrokes = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const strokes = d.strokes
  if (!strokes) return null
  const entries = SWIM_STROKES.map(s => [s, strokes[s] ?? 0] as const).filter(([, m]) => m > 0)
  const total = entries.reduce((sum, [, m]) => sum + m, 0)
  if (entries.length === 0 || total <= 0) return null
  const box = f.el('div', 'tri-pool-strokes')
  const bar = f.el('div', 'tri-stroke-bar')
  const legend = f.el('ul', 'tri-stroke-legend')
  for (const [s, m] of entries) {
    f.add(
      bar,
      f.el('span', `tri-stroke-seg tri-stroke-${s}`, undefined, {
        style: `width:${((m / total) * 100).toFixed(2)}%`,
      }),
    )
    const li = f.el('li', 'tri-stroke-leg')
    f.add(
      li,
      f.el('span', `tri-stroke-dot tri-stroke-${s}`),
      f.el('span', 'tri-stroke-name', STROKE_LABEL[s]),
      f.el('span', 'tri-stroke-val', `${Math.round(m)}m`),
    )
    f.add(legend, li)
  }
  f.add(box, bar, legend)
  return box
}

export const buildPool = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
  const lengths = Math.max(1, Math.round((d.distanceKm * 1000) / 25))
  const wrap = f.el('div', 'tri-pool-wrap')
  f.add(wrap, f.el('span', 'tri-pool-cap', `${lengths} × 25m`))
  const strokes = buildSwimStrokes(f, d)
  if (strokes) f.add(wrap, strokes)
  return wrap
}

const buildPoolOverview = <N>(f: TriNodeFactory<N>): N => {
  const fig = f.svg('svg', {
    class: 'tri-route tri-pool',
    viewBox: '0 0 100 56',
    preserveAspectRatio: 'xMidYMid meet',
  })
  f.add(
    fig,
    f.svg('rect', { x: 6, y: 12, width: 88, height: 32, rx: 16, ry: 16, class: 'tri-pool-lane' }),
  )
  f.add(fig, f.svg('line', { x1: 22, y1: 28, x2: 78, y2: 28, class: 'tri-pool-mid' }))
  return fig
}

type SwimActivityObservation = {
  interval: SwimActivityInterval
  index: number
  strokesPerLength: number | null
  swolf: number | null
}

export type SwimActivityBlock = SwimActivityInterval & {
  strokesPerLength: number | null
  swolf: number | null
}

type SwimActivityMetric = { observation: SwimActivityObservation; value: number }

export type SwimTrendMode = 'lengths' | '100m'

export type SwimTrendChartPoint = {
  elapsedS: number
  cumulativeDistanceM: number
  value: number
  xPct: number
  yPct: number
  windowStartDistanceM?: number
}

export type SwimTrendHover = SwimTrendChartPoint & { index: number }

export const swimTrendHoverAt = (
  points: SwimTrendChartPoint[],
  fraction: number,
): SwimTrendHover | null => {
  if (points.length === 0) return null
  const xPct = Math.max(0, Math.min(100, (Number.isFinite(fraction) ? fraction : 0) * 100))
  let index = 0
  let distance = Math.abs(points[0].xPct - xPct)
  for (let candidate = 1; candidate < points.length; candidate++) {
    const candidateDistance = Math.abs(points[candidate].xPct - xPct)
    if (candidateDistance < distance) {
      index = candidate
      distance = candidateDistance
    }
  }
  return { ...points[index], index }
}

const positiveMetric = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const swimRoundTenth = (value: number): number => Math.round(value * 10) / 10

export const swimActivityBlocks = (
  intervals: SwimActivityInterval[],
  blockDistanceM = 100,
): SwimActivityBlock[] => {
  if (!Number.isFinite(blockDistanceM) || blockDistanceM <= 0) return []
  const blocks: SwimActivityBlock[] = []
  let cumulativeDistanceM = 0
  let distanceM = 0
  let durationS = 0
  let strokeCount = 0
  let strokeTimeS = 0
  let strokesPerLength = 0
  let swolf = 0
  let measuredLengthWeight = 0
  let startElapsedS = 0
  let endElapsedS = 0
  let strokeComplete = true
  const flush = (): void => {
    if (distanceM <= 0) return
    const paceSPer100m = swimPaceSeconds(distanceM, durationS)
    const strokeRateSpm = strokeComplete ? swimStrokeRate(strokeCount, strokeTimeS) : null
    blocks.push({
      startElapsedS: swimRoundTenth(startElapsedS),
      endElapsedS: swimRoundTenth(endElapsedS),
      distanceM: swimRoundTenth(distanceM),
      durationS: swimRoundTenth(durationS),
      cumulativeDistanceM: swimRoundTenth(cumulativeDistanceM),
      paceSPer100m,
      strokeCount: strokeRateSpm == null ? null : swimRoundTenth(strokeCount),
      strokeTimeS: strokeRateSpm == null ? null : swimRoundTenth(strokeTimeS),
      strokeRateSpm,
      stroke: null,
      strokesPerLength:
        measuredLengthWeight > 0 ? swimRoundTenth(strokesPerLength / measuredLengthWeight) : null,
      swolf: measuredLengthWeight > 0 ? swimRoundTenth(swolf / measuredLengthWeight) : null,
    })
    distanceM = 0
    durationS = 0
    strokeCount = 0
    strokeTimeS = 0
    strokesPerLength = 0
    swolf = 0
    measuredLengthWeight = 0
    strokeComplete = true
  }
  for (const interval of intervals) {
    if (
      !positiveMetric(interval.distanceM) ||
      !positiveMetric(interval.durationS) ||
      interval.endElapsedS <= interval.startElapsedS
    )
      continue
    let consumedDistanceM = 0
    while (consumedDistanceM < interval.distanceM - 0.0001) {
      const remainingDistanceM = interval.distanceM - consumedDistanceM
      const availableDistanceM = blockDistanceM - distanceM
      const contributionDistanceM = Math.min(remainingDistanceM, availableDistanceM)
      const startFraction = consumedDistanceM / interval.distanceM
      const endFraction = (consumedDistanceM + contributionDistanceM) / interval.distanceM
      const contributionDurationS =
        interval.durationS * (contributionDistanceM / interval.distanceM)
      if (distanceM === 0)
        startElapsedS =
          startFraction === 0
            ? interval.startElapsedS
            : interval.startElapsedS + interval.durationS * startFraction
      endElapsedS =
        endFraction >= 1
          ? interval.endElapsedS
          : interval.startElapsedS + interval.durationS * endFraction
      distanceM += contributionDistanceM
      durationS += contributionDurationS
      cumulativeDistanceM += contributionDistanceM
      const fraction = contributionDistanceM / interval.distanceM
      const lengthMetrics = swimLengthMetrics(interval)
      if (lengthMetrics) {
        strokesPerLength += lengthMetrics.strokesPerLength * fraction
        swolf += lengthMetrics.swolf * fraction
        measuredLengthWeight += fraction
      }
      if (interval.stroke !== 'kickboard') {
        if (positiveMetric(interval.strokeCount) && positiveMetric(interval.strokeTimeS)) {
          strokeCount += interval.strokeCount * fraction
          strokeTimeS += interval.strokeTimeS * fraction
        } else {
          strokeComplete = false
        }
      }
      consumedDistanceM += contributionDistanceM
      if (distanceM >= blockDistanceM - 0.0001) flush()
    }
  }
  flush()
  return blocks
}

const swimTrendNumber = (value: number): string =>
  value.toLocaleString('en-US', { maximumFractionDigits: 1 })

const swimTrendDisplayValue = (kind: SwimChartMetric, value: number): string =>
  kind === 'pace'
    ? `${clock(value)} /100m`
    : kind === 'cadence'
      ? `${swimTrendNumber(value)} str/length`
      : kind === 'rate'
        ? `${swimTrendNumber(value)} spm`
        : `${Math.round(value)} SWOLF`

const swimTrendHeaderValue = (kind: SwimChartMetric, value: number): string =>
  kind === 'pace'
    ? clock(value)
    : kind === 'cadence' || kind === 'rate'
      ? swimTrendNumber(value)
      : Math.round(value).toLocaleString('en-US')

const swimTrendTitle = (kind: SwimChartMetric): string =>
  kind === 'pace'
    ? 'pace /100m'
    : kind === 'cadence'
      ? 'cadence str/length'
      : kind === 'rate'
        ? 'stroke rate spm'
        : 'SWOLF'

const SWIM_TREND_GLOSS: Partial<Record<SwimChartMetric, string>> = {
  rate: 'strokerate',
  cadence: 'swimcadence',
  swolf: 'swolf',
}

const swimTrendLabel = (kind: SwimChartMetric): string =>
  kind === 'pace'
    ? 'pace'
    : kind === 'cadence'
      ? 'cadence'
      : kind === 'rate'
        ? 'stroke rate'
        : 'SWOLF'

const swimDistanceLabel = (distanceM: number): string =>
  `${Math.round(distanceM).toLocaleString('en-US')} m`

export const swimActivityPointLabel = (
  point: Pick<SwimTrendChartPoint, 'elapsedS' | 'cumulativeDistanceM' | 'windowStartDistanceM'>,
): string => {
  const distance =
    point.windowStartDistanceM == null
      ? swimDistanceLabel(point.cumulativeDistanceM)
      : `${Math.round(point.windowStartDistanceM).toLocaleString('en-US')}–${Math.round(point.cumulativeDistanceM).toLocaleString('en-US')} m`
  return `${distance} · ${clock(point.elapsedS)} elapsed`
}

export const swimTrendAriaValue = (
  kind: SwimChartMetric,
  point: Pick<
    SwimTrendChartPoint,
    'elapsedS' | 'cumulativeDistanceM' | 'value' | 'windowStartDistanceM'
  >,
): string => {
  const position =
    point.windowStartDistanceM == null
      ? `${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
      : `${Math.round(point.cumulativeDistanceM - point.windowStartDistanceM)} metre block from ${Math.round(point.windowStartDistanceM)} to ${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
  if (kind === 'pace') return `${position}, swim pace ${clock(point.value)} per 100 metres`
  if (kind === 'cadence')
    return `${position}, swim cadence ${swimTrendNumber(point.value)} strokes per length`
  if (kind === 'rate')
    return `${position}, stroke rate ${swimTrendNumber(point.value)} strokes per minute`
  return `${position}, SWOLF score ${Math.round(point.value)}`
}

const swimTrendDomain = (values: number[], kind: SwimChartMetric): { min: number; max: number } => {
  const observedMin = Math.min(...values)
  const observedMax = Math.max(...values)
  if (kind === 'pace') {
    const step = Math.max(1, niceStep(observedMax, 3))
    return { min: 0, max: Math.max(step, Math.ceil(observedMax / step) * step) }
  }
  const span = observedMax - observedMin
  const step = niceStep(span > 0 ? span : Math.max(1, observedMax * 0.1), 3)
  const min = Math.max(0, Math.floor(observedMin / step) * step)
  const max = Math.ceil(observedMax / step) * step
  return { min, max: max > min ? max : min + step }
}

const swimActivityXTicks = (totalDistanceM: number): AxisXTick[] => {
  return [
    { label: '0 m', pct: 0, cls: 'tri-cax-xt--first' },
    { label: swimDistanceLabel(totalDistanceM / 2), pct: 50 },
    { label: swimDistanceLabel(totalDistanceM), pct: 100, cls: 'tri-cax-xt--last' },
  ]
}

const buildSwimModeToggle = <N>(f: TriNodeFactory<N>): N => {
  const toggle = f.el('div', 'tri-swim-mode-toggle', undefined, {
    role: 'group',
    'aria-label': 'swim chart aggregation',
    'data-i18n-aria-label': 'swim chart aggregation',
    'data-swim-mode': 'lengths',
  })
  f.add(
    toggle,
    f.el('button', 'tri-swim-mode', 'lengths', {
      type: 'button',
      'data-swim-mode': 'lengths',
      'aria-pressed': 'true',
      'data-i18n': 'lengths',
    }),
    f.el('button', 'tri-swim-mode', '100 m', {
      type: 'button',
      'data-swim-mode': '100m',
      'aria-pressed': 'false',
      'data-i18n': '100 m',
    }),
  )
  return toggle
}

const buildSwimTrendChart = <N>(
  f: TriNodeFactory<N>,
  observations: SwimActivityObservation[],
  hundredMetreObservations: SwimActivityObservation[],
  totalDistanceM: number,
  kind: SwimChartMetric,
  average: number | null,
  pick: (observation: SwimActivityObservation) => number | null,
  modeToggle?: N,
): N | null => {
  const metricSeries = (source: SwimActivityObservation[]): SwimActivityMetric[] => {
    const metrics: SwimActivityMetric[] = []
    for (const observation of source) {
      const value = pick(observation)
      if (positiveMetric(value)) metrics.push({ observation, value })
    }
    return metrics
  }
  const series = metricSeries(observations)
  const hundredMetreSeries = metricSeries(hundredMetreObservations)
  if (series.length < 2) return null
  const currentIndex = series.length - 1
  const activityAverage = positiveMetric(average)
    ? average
    : series.reduce((sum, metric) => sum + metric.value, 0) / series.length
  const title = swimTrendTitle(kind)
  const value = swimTrendHeaderValue(kind, activityAverage)
  const wrap = f.el('article', `tri-zone tri-swim-trend tri-swim-trend--${kind}`, undefined, {
    'data-tri-trace': triathlonTraceName(swimTrendLabel(kind)),
  })
  const head = f.el(
    'div',
    `tri-swim-trend-head${modeToggle === undefined ? '' : ' tri-swim-trend-head--with-mode'}`,
  )
  const glossKey = SWIM_TREND_GLOSS[kind]
  if (modeToggle === undefined)
    f.add(
      head,
      f.el('span', 'tri-swim-trend-title', title, {
        'data-i18n': title,
        ...(glossKey ? { 'data-gloss': glossKey, tabindex: '0' } : {}),
      }),
    )
  else f.add(head, modeToggle)
  f.add(
    head,
    f.el('strong', 'tri-swim-trend-value', value, {
      'data-swim-average-kind': kind,
      'data-swim-average-value': activityAverage.toString(),
    }),
  )
  const W = 100
  const H = 30
  const X = (observation: SwimActivityObservation): number =>
    (observation.interval.cumulativeDistanceM / totalDistanceM) * W
  const domain = swimTrendDomain(
    [...series, ...hundredMetreSeries].map(metric => metric.value),
    kind,
  )
  const domainSpan = domain.max - domain.min
  const Y =
    kind === 'pace'
      ? (metric: number): number => ((metric - domain.min) / domainSpan) * H
      : (metric: number): number => H - ((metric - domain.min) / domainSpan) * H
  const ticks = niceTicks(domain.min, domain.max, 3)
  const tickStep = niceStep(domainSpan, 3)
  const yTicks = ticks.map(tick => ({
    label: kind === 'pace' ? clock(tick) : axisNumber(tick, tickStep),
    vbY: Y(tick),
  }))
  const chartSeries = (
    metrics: SwimActivityMetric[],
    mode: SwimTrendMode,
  ): { points: SwimTrendChartPoint[]; linePath: string; areaPath: string } => {
    const startX = (metric: SwimActivityMetric): number =>
      ((metric.observation.interval.cumulativeDistanceM - metric.observation.interval.distanceM) /
        totalDistanceM) *
      W
    const points = metrics.map(metric => ({
      elapsedS: metric.observation.interval.endElapsedS,
      cumulativeDistanceM: metric.observation.interval.cumulativeDistanceM,
      value: metric.value,
      xPct: X(metric.observation),
      yPct: (Y(metric.value) / H) * 100,
      ...(mode === '100m'
        ? {
            windowStartDistanceM:
              metric.observation.interval.cumulativeDistanceM -
              metric.observation.interval.distanceM,
          }
        : {}),
    }))
    const runs: SwimActivityMetric[][] = []
    for (const metric of metrics) {
      const run = runs.at(-1)
      const prior = run?.at(-1)
      if (run && prior && metric.observation.index === prior.observation.index + 1) run.push(metric)
      else runs.push([metric])
    }
    const linePath = runs
      .map(run =>
        run
          .map(
            (metric, index) =>
              `${index === 0 ? 'M' : 'L'} ${startX(metric).toFixed(2)} ${Y(metric.value).toFixed(2)} L ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`,
          )
          .join(' '),
      )
      .join(' ')
    const areaPath = runs
      .map(run => {
        const first = run[0]
        const last = run[run.length - 1]
        const values = run
          .map(
            metric =>
              `L ${startX(metric).toFixed(2)} ${Y(metric.value).toFixed(2)} L ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`,
          )
          .join(' ')
        return `M ${startX(first).toFixed(2)} ${H} ${values} L ${X(last.observation).toFixed(2)} ${H} Z`
      })
      .join(' ')
    return { points, linePath, areaPath }
  }
  const lengthsChart = chartSeries(series, 'lengths')
  const hundredMetreChart = chartSeries(hundredMetreSeries, '100m')
  const currentChartPoint = lengthsChart.points[currentIndex]
  const svg = f.svg('svg', {
    class: `tri-swim-trend-svg tri-swim-trend-svg--${kind}`,
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': `Swim ${swimTrendLabel(kind)} by length`,
    'aria-orientation': 'horizontal',
    'aria-valuemin': 0,
    'aria-valuemax': Math.round(totalDistanceM),
    'aria-valuenow': Math.round(currentChartPoint.cumulativeDistanceM),
    'aria-valuetext': `${swimTrendAriaValue(kind, currentChartPoint)}. Activity average ${swimTrendDisplayValue(kind, activityAverage)}.`,
    'data-swim-series-lengths': JSON.stringify(lengthsChart.points),
    'data-swim-series-hundred': JSON.stringify(hundredMetreChart.points),
    'data-swim-mode': 'lengths',
    'data-swim-kind': kind,
    'data-swim-index': currentIndex,
  })
  for (const tick of yTicks)
    f.add(
      svg,
      f.svg('line', { class: 'tri-swim-trend-grid', x1: 0, y1: tick.vbY, x2: W, y2: tick.vbY }),
    )
  const addLayer = (
    mode: SwimTrendMode,
    chart: { linePath: string; areaPath: string },
    active: boolean,
  ): void => {
    const layer = f.svg('g', {
      class: `tri-swim-series tri-swim-series--${mode}${active ? ' tri-swim-series--active' : ''}`,
      'data-swim-mode': mode,
      'aria-hidden': String(!active),
    })
    f.add(
      layer,
      f.svg('path', {
        class: `tri-swim-trend-area tri-swim-trend-area--${mode}`,
        d: chart.areaPath,
      }),
      f.svg('path', {
        class: `tri-swim-trend-line tri-swim-trend-line--${mode}`,
        d: chart.linePath,
      }),
    )
    f.add(svg, layer)
  }
  addLayer('lengths', lengthsChart, true)
  if (hundredMetreChart.points.length >= 2) addLayer('100m', hundredMetreChart, false)
  f.add(
    svg,
    f.svg('line', {
      class: 'tri-chart-cursor',
      x1: currentChartPoint.xPct.toFixed(2),
      y1: 0,
      x2: currentChartPoint.xPct.toFixed(2),
      y2: H,
    }),
  )
  const hoverMarker = f.el('span', 'tri-swim-trend-hover', undefined, {
    'aria-hidden': 'true',
    hidden: '',
    style: `left:${currentChartPoint.xPct.toFixed(2)}%;top:${currentChartPoint.yPct.toFixed(2)}%`,
  })
  const readout = f.el('div', 'tri-chart-readout tri-swim-trend-readout', undefined, {
    'aria-hidden': 'true',
  })
  f.add(
    readout,
    f.el('span', 'tri-swim-trend-readout-position', swimActivityPointLabel(currentChartPoint)),
    f.el(
      'strong',
      'tri-swim-trend-readout-value',
      swimTrendDisplayValue(kind, currentChartPoint.value),
    ),
  )
  f.add(
    wrap,
    head,
    axisFrame(f, svg, yTicks, H, swimActivityXTicks(totalDistanceM), true, { top: 0, bottom: H }, [
      hoverMarker,
      readout,
    ]),
  )
  return wrap
}

export const buildSwimTrends = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  traceSettings?: TriathlonTraceSettings,
): N | null => {
  if (d.sport !== 'swim') return null
  const observations = d.swimIntervals
    .filter(
      interval => interval.endElapsedS > interval.startElapsedS && interval.cumulativeDistanceM > 0,
    )
    .map((interval, index) => {
      const length = d.swimLocation === 'pool' ? swimLengthMetrics(interval) : null
      return {
        interval,
        index,
        strokesPerLength: length?.strokesPerLength ?? null,
        swolf: length?.swolf ?? null,
      }
    })
  const totalDistanceM = observations.at(-1)?.interval.cumulativeDistanceM ?? 0
  if (observations.length < 2 || totalDistanceM <= 0) return null
  const hundredMetreObservations = swimActivityBlocks(
    observations.map(observation => observation.interval),
  ).map((interval, index) => ({
    interval,
    index,
    strokesPerLength: d.swimLocation === 'pool' ? interval.strokesPerLength : null,
    swolf: d.swimLocation === 'pool' ? interval.swolf : null,
  }))
  const hasSeries = (
    source: SwimActivityObservation[],
    pick: (observation: SwimActivityObservation) => number | null,
  ): boolean => source.filter(observation => positiveMetric(pick(observation))).length >= 2
  const paceVisible =
    triathlonTraceEnabled(traceSettings, 'pace') &&
    hasSeries(observations, observation => observation.interval.paceSPer100m)
  const rateVisible =
    triathlonTraceEnabled(traceSettings, 'stroke-rate') &&
    hasSeries(observations, observation => observation.interval.strokeRateSpm)
  const cadenceVisible =
    triathlonTraceEnabled(traceSettings, 'cadence') &&
    hasSeries(observations, observation => observation.strokesPerLength)
  const swolfVisible =
    triathlonTraceEnabled(traceSettings, 'swolf') &&
    hasSeries(observations, observation => observation.swolf)
  const canToggle =
    hundredMetreObservations.length >= 2 &&
    (!paceVisible ||
      hasSeries(hundredMetreObservations, observation => observation.interval.paceSPer100m)) &&
    (!rateVisible ||
      hasSeries(hundredMetreObservations, observation => observation.interval.strokeRateSpm)) &&
    (!cadenceVisible ||
      hasSeries(hundredMetreObservations, observation => observation.strokesPerLength)) &&
    (!swolfVisible || hasSeries(hundredMetreObservations, observation => observation.swolf))
  const normalizedObservations = canToggle ? hundredMetreObservations : []
  const modeToggle = canToggle ? buildSwimModeToggle(f) : undefined
  const paceAverage = positiveMetric(d.swimPaceSPer100m) ? d.swimPaceSPer100m : null
  const lengthAverages = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
  const pace = paceVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'pace',
        paceAverage,
        observation => observation.interval.paceSPer100m,
        modeToggle,
      )
    : null
  const rate = rateVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'rate',
        positiveMetric(d.strokeRateSpm) ? d.strokeRateSpm : null,
        observation => observation.interval.strokeRateSpm,
        paceVisible ? undefined : modeToggle,
      )
    : null
  const cadence = cadenceVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'cadence',
        lengthAverages?.strokesPerLength ?? null,
        observation => observation.strokesPerLength,
        paceVisible || rateVisible ? undefined : modeToggle,
      )
    : null
  const swolf = swolfVisible
    ? buildSwimTrendChart(
        f,
        observations,
        normalizedObservations,
        totalDistanceM,
        'swolf',
        lengthAverages?.swolf ?? null,
        observation => observation.swolf,
        paceVisible || rateVisible || cadenceVisible ? undefined : modeToggle,
      )
    : null
  const trends = f.el('div', 'tri-swim-chart-grid')
  let chartCount = 0
  for (const chart of [pace, rate, cadence, swolf])
    if (chart) {
      f.add(trends, chart)
      chartCount++
    }
  if (chartCount === 0) return null
  const wrap = f.el('section', 'tri-swim-trends', undefined, {
    'aria-label': 'Swim activity analysis',
    'data-i18n-aria-label': 'swim activity analysis',
  })
  f.add(wrap, trends)
  return wrap
}

export const statRow = <N>(
  f: TriNodeFactory<N>,
  label: string,
  value: string,
  attrs?: Record<string, string>,
): N => {
  const tr = f.el('tr', undefined, undefined, { ...attrs, 'data-stat-key': label })
  f.add(
    tr,
    f.el('th', 'tri-act-stat-k', label, { 'data-i18n': label }),
    f.el('td', 'tri-act-stat-v', value),
  )
  return tr
}

export const statsTable = <N>(
  f: TriNodeFactory<N>,
  rows: [string, string][],
  rowAttrs?: (label: string) => Record<string, string> | undefined,
): N => {
  const table = f.el('table', 'tri-act-stats')
  const tbody = f.el('tbody')
  for (const [k, v] of rows) f.add(tbody, statRow(f, k, v, rowAttrs?.(k)))
  f.add(table, tbody)
  return table
}

export const buildFueling = <N>(f: TriNodeFactory<N>, fueling: ActivityFueling): N | null => {
  const rows = fuelingRows(fueling)
  if (rows.length === 0) return null
  const wrap = f.el('div', 'tri-act-health tri-act-fueling')
  f.add(wrap, f.el('span', 'tri-act-health-h', 'fueling'), statsTable(f, rows))
  return wrap
}

const buildReservedFueling = <N>(f: TriNodeFactory<N>): N => {
  const wrap = f.el('div', 'tri-act-health tri-act-fueling tri-act-fueling--empty', undefined, {
    'aria-hidden': 'true',
  })
  f.add(wrap, f.el('span', 'tri-act-health-h', 'fueling'), statsTable(f, []))
  return wrap
}

export const buildRecovery = <N>(f: TriNodeFactory<N>, h: ActivityHealth): N | null => {
  const rows = recoveryRows(h)
  if (rows.length === 0) return null
  const wrap = f.el('div', 'tri-act-health')
  f.add(wrap, f.el('span', 'tri-act-health-h', 'recovery'), statsTable(f, rows))
  return wrap
}

export interface DetailCtx {
  zones: StravaZones | null
  curveRef: PowerCurvePoint[]
  curveYearRef: PowerCurvePoint[]
  curveYear: number | null
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
  ftp: number | null
  goalFtp: number | null
  vt1: number | null
}

export type AxisXTick = {
  label: string
  pct: number
  cls?: string
  tag?: 'span' | 'button'
  attrs?: Record<string, string>
}

const HR_ZONE_NAMES = ['recovery', 'endurance', 'tempo', 'threshold', 'anaerobic']
const POWER_ZONE_NAMES = [
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

const zoneRange = (bounds: number[], i: number): string => {
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

export type PowerCurveHover = {
  index: number
  durationS: number
  watts: number
  referenceWatts: number | null
  xPct: number
}

const POWER_CURVE_PATH_POINTS = 1_024

const encodePowerCurveActivities = (curve: readonly PowerCurvePoint[]): string => {
  if (
    curve.length === 0 ||
    curve.some(
      point =>
        point.activityId == null ||
        !Number.isInteger(point.activityId) ||
        point.activityId < 0 ||
        point.activityDate == null ||
        !/^\d{4}-\d{2}-\d{2}$/.test(point.activityDate),
    )
  )
    return ''
  const segments: string[] = []
  let start = 0
  for (let index = 1; index <= curve.length; index++) {
    const previous = curve[index - 1]
    const point = curve[index]
    if (
      point &&
      point.activityId === previous.activityId &&
      point.activityDate === previous.activityDate
    )
      continue
    segments.push(`${previous.activityId},${previous.activityDate},${index - start}`)
    start = index
  }
  return segments.join(';')
}

export const encodePowerCurve = (curve: PowerCurvePoint[]): string => {
  if (curve.length === 0) return ''
  const consecutive = curve.every((point, index) => point.s === curve[0].s + index)
  const encoded = consecutive
    ? `d|${curve[0].s}|${curve.map(point => point.w).join(',')}`
    : `s|${curve.map(point => `${point.s}:${point.w}`).join(',')}`
  const activities = encodePowerCurveActivities(curve)
  return activities ? `${encoded}|${activities}` : encoded
}

const decodePowerCurveActivities = (
  points: PowerCurvePoint[],
  encoded: string | undefined,
): boolean => {
  if (encoded == null) return true
  if (encoded.length === 0) return false
  let index = 0
  for (const segment of encoded.split(';')) {
    const fields = segment.split(',')
    if (fields.length !== 3) return false
    const activityId = Number(fields[0])
    const activityDate = fields[1]
    const count = Number(fields[2])
    if (
      !Number.isInteger(activityId) ||
      activityId < 0 ||
      !/^\d{4}-\d{2}-\d{2}$/.test(activityDate) ||
      !Number.isInteger(count) ||
      count <= 0 ||
      index + count > points.length
    )
      return false
    for (let offset = 0; offset < count; offset++) {
      points[index] = { ...points[index], activityId, activityDate }
      index += 1
    }
  }
  return index === points.length
}

export const decodePowerCurve = (encoded: string | undefined): PowerCurvePoint[] => {
  if (!encoded) return []
  const fields = encoded.split('|')
  const points: PowerCurvePoint[] = []
  if (fields[0] === 'd' && (fields.length === 3 || fields.length === 4)) {
    const start = Number(fields[1])
    if (!Number.isInteger(start) || start <= 0 || fields[2].length === 0) return []
    for (const [index, raw] of fields[2].split(',').entries()) {
      if (raw.length === 0) return []
      const watts = Number(raw)
      if (!Number.isFinite(watts)) return []
      points.push({ s: start + index, w: watts })
    }
    return decodePowerCurveActivities(points, fields[3]) ? points : []
  }
  if (fields[0] !== 's' || (fields.length !== 2 && fields.length !== 3) || fields[1].length === 0)
    return []
  let previousSeconds = 0
  for (const raw of fields[1].split(',')) {
    const separator = raw.indexOf(':')
    if (separator <= 0 || separator === raw.length - 1) return []
    const seconds = Number(raw.slice(0, separator))
    const watts = Number(raw.slice(separator + 1))
    if (!Number.isInteger(seconds) || seconds <= previousSeconds || !Number.isFinite(watts))
      return []
    points.push({ s: seconds, w: watts })
    previousSeconds = seconds
  }
  return decodePowerCurveActivities(points, fields[2]) ? points : []
}

export const powerCurveFraction = (
  seconds: number,
  minSeconds: number,
  maxSeconds: number,
): number => {
  if (minSeconds <= 0 || maxSeconds <= minSeconds) return 0
  const value = Math.min(maxSeconds, Math.max(minSeconds, seconds))
  return (Math.log(value) - Math.log(minSeconds)) / (Math.log(maxSeconds) - Math.log(minSeconds))
}

const POWER_CURVE_AXIS_MARKERS = [5, 10, 20, 30, 120]
const POWER_CURVE_ENDPOINT_GAP = 0.12
const EMBEDDED_POWER_CURVE_PRECISE_ENDPOINT_GAP = 0.14

export const powerCurveDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  durations: readonly number[],
): number[] =>
  [...new Set([...durations, ...POWER_CURVE_AXIS_MARKERS, maxSeconds])]
    .filter(seconds => seconds >= minSeconds && seconds <= maxSeconds)
    .sort((left, right) => left - right)
    .filter(
      seconds =>
        seconds === maxSeconds ||
        powerCurveFraction(seconds, minSeconds, maxSeconds) <= 1 - POWER_CURVE_ENDPOINT_GAP,
    )

const embeddedPowerCurveDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  durations: readonly number[],
): number[] => {
  const endpointGap = dlabel(maxSeconds).endsWith('s')
    ? EMBEDDED_POWER_CURVE_PRECISE_ENDPOINT_GAP
    : POWER_CURVE_ENDPOINT_GAP
  return powerCurveDurationTicks(minSeconds, maxSeconds, durations).filter(
    seconds =>
      seconds !== 10 &&
      seconds !== 20 &&
      (seconds === maxSeconds ||
        powerCurveFraction(seconds, minSeconds, maxSeconds) <= 1 - endpointGap),
  )
}

const nearestPowerCurveIndex = (curve: readonly PowerCurvePoint[], seconds: number): number => {
  let low = 0
  let high = curve.length - 1
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (curve[mid].s < seconds) low = mid + 1
    else high = mid
  }
  if (
    low > 0 &&
    Math.abs(Math.log(curve[low - 1].s) - Math.log(seconds)) <
      Math.abs(Math.log(curve[low].s) - Math.log(seconds))
  )
    return low - 1
  return low
}

export const powerCurvePathPoints = (curve: PowerCurvePoint[]): PowerCurvePoint[] => {
  if (curve.length <= POWER_CURVE_PATH_POINTS) return curve
  const minSeconds = curve[0].s
  const maxSeconds = curve[curve.length - 1].s
  const points: PowerCurvePoint[] = []
  let previousIndex = -1
  for (let sample = 0; sample < POWER_CURVE_PATH_POINTS; sample++) {
    const fraction = sample / (POWER_CURVE_PATH_POINTS - 1)
    const seconds = Math.exp(
      Math.log(minSeconds) + fraction * (Math.log(maxSeconds) - Math.log(minSeconds)),
    )
    const index = nearestPowerCurveIndex(curve, seconds)
    if (index === previousIndex) continue
    points.push(curve[index])
    previousIndex = index
  }
  if (previousIndex !== curve.length - 1) points.push(curve[curve.length - 1])
  return points
}

export const powerCurveHoverAt = (
  curve: PowerCurvePoint[],
  reference: PowerCurvePoint[],
  pointerFraction: number,
): PowerCurveHover | null => {
  if (curve.length < 2) return null
  const fraction = Math.min(1, Math.max(0, pointerFraction))
  const minSeconds = curve[0].s
  const maxSeconds = curve[curve.length - 1].s
  const targetSeconds = Math.exp(
    Math.log(minSeconds) + fraction * (Math.log(maxSeconds) - Math.log(minSeconds)),
  )
  const index = nearestPowerCurveIndex(curve, targetSeconds)
  const point = curve[index]
  let referenceWatts: number | null = null
  let low = 0
  let high = reference.length - 1
  while (low <= high) {
    const mid = Math.floor((low + high) / 2)
    if (reference[mid].s < point.s) low = mid + 1
    else if (reference[mid].s > point.s) high = mid - 1
    else {
      referenceWatts = reference[mid].w
      break
    }
  }
  return {
    index,
    durationS: point.s,
    watts: point.w,
    referenceWatts,
    xPct: powerCurveFraction(point.s, minSeconds, maxSeconds) * 100,
  }
}

const effortDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)} sec`
  if (seconds < 3600 && seconds % 60 === 0) {
    const minutes = seconds / 60
    return `${minutes} min`
  }
  if (seconds % 3600 === 0) {
    const hours = seconds / 3600
    return `${hours} hr`
  }
  return zoneClock(seconds)
}

const cyclingSpeed = (presentation: TriathlonPresentation, kph: number): string =>
  isImperial(presentation) ? `${(kph * KM_TO_MI).toFixed(1)} mph` : `${kph.toFixed(1)} km/h`

const heartRate = (bpm: number | null): string => (bpm == null ? '—' : `${Math.round(bpm)} bpm`)

const watts = (value: number | null): string =>
  value == null ? '—' : `${Math.round(value).toLocaleString('en-US')} W`

const wattsPerKg = (value: number | null): string =>
  value == null ? '—' : `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })} W/kg`

const effortTable = <N>(
  f: TriNodeFactory<N>,
  title: string,
  kind: string,
  headers: string[],
  rows: string[][],
): N => {
  const block = f.el('div', `tri-effort-block tri-effort-block--${kind}`)
  const table = f.el('table', `tri-effort-table tri-effort-table--${kind}`, undefined, {
    'aria-label': `${title} efforts`,
  })
  const thead = f.el('thead')
  const heading = f.el('tr')
  for (const label of headers) f.add(heading, f.el('th', undefined, label, { scope: 'col' }))
  f.add(thead, heading)
  const tbody = f.el('tbody')
  for (const cells of rows) {
    const row = f.el('tr')
    cells.forEach((value, index) =>
      f.add(
        row,
        f.el(index === 0 ? 'th' : 'td', undefined, value, index === 0 ? { scope: 'row' } : {}),
      ),
    )
    f.add(tbody, row)
  }
  f.add(table, thead, tbody)
  const scroll = f.el('div', 'tri-effort-scroll', undefined, {
    role: 'region',
    'aria-label': `${title} efforts`,
    tabindex: '0',
  })
  f.add(scroll, table)
  const viewport = f.el('div', 'tri-effort-viewport')
  f.add(viewport, scroll)
  f.add(block, f.el('div', 'tri-zone-title tri-effort-title', title), viewport)
  return block
}

export const buildCyclingBestEfforts = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const efforts = d.bestEfforts
  if (
    d.sport !== 'bike' ||
    !efforts ||
    (efforts.distance.length === 0 && efforts.power.length === 0 && efforts.climbs.length === 0)
  )
    return null

  const wrap = f.el('section', 'tri-efforts', undefined, { 'aria-label': 'Cycling best efforts' })
  if (efforts.distance.length > 0)
    f.add(
      wrap,
      effortTable(
        f,
        'Distance',
        'distance',
        ['Distance', 'Time', 'Speed', 'Heart rate', 'Elev'],
        efforts.distance.map(row => [
          row.label || scrubDist(f.presentation, row.targetDistanceM / 1000, 'bike'),
          zoneClock(row.elapsedTimeS),
          cyclingSpeed(f.presentation, row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          formatAltitude(f.presentation, row.elevationDeltaM),
        ]),
      ),
    )
  if (efforts.power.length > 0) {
    f.add(
      wrap,
      effortTable(
        f,
        'Power',
        'power',
        ['Time', 'Power', 'W/kg', 'Heart rate', 'Elev'],
        efforts.power.map(row => [
          effortDuration(row.durationS),
          watts(row.averageWatts),
          wattsPerKg(row.wattsPerKg),
          heartRate(row.averageHeartRate),
          formatAltitude(f.presentation, row.elevationDeltaM),
        ]),
      ),
    )
  }
  if (efforts.climbs.length > 0)
    f.add(
      wrap,
      effortTable(
        f,
        'Climbing',
        'climbing',
        [
          'Climb',
          'Time',
          'Distance',
          'Gain',
          'Grade',
          'Speed',
          'Heart rate',
          'Power',
          'W/kg',
          'VAM',
        ],
        efforts.climbs.map((row, index) => [
          row.name || `Climb ${index + 1}`,
          zoneClock(row.durationS),
          scrubDist(f.presentation, row.distanceM / 1000, 'bike'),
          formatElevationGain(f.presentation, row.elevationGainM),
          `${row.averageGradePct.toFixed(1)}%`,
          cyclingSpeed(f.presentation, row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          watts(row.averageWatts),
          wattsPerKg(row.wattsPerKg),
          formatVam(f.presentation, row.vamMPerHour),
        ]),
      ),
    )
  if (
    efforts.weightKg != null &&
    efforts.weightDate &&
    (efforts.power.length || efforts.climbs.length)
  )
    f.add(
      wrap,
      f.el(
        'p',
        'tri-effort-note',
        `W/kg from ${efforts.weightKg.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 2 })} kg Garmin weight · ${shortDate(efforts.weightDate)}`,
      ),
    )
  return wrap
}

export const axisFrame = <N>(
  f: TriNodeFactory<N>,
  svgEl: N,
  yTicks: { label: string; vbY: number }[],
  vbH: number,
  xTicks: AxisXTick[],
  axes = true,
  axisRange?: { top: number; bottom: number },
  stageOverlays: N[] = [],
  rightYTicks: { label: string; vbY: number }[] = [],
): N => {
  const frame = f.el('div', `tri-cax-frame${rightYTicks.length > 0 ? ' tri-cax-frame--dual' : ''}`)
  const yax = f.el('div', 'tri-cax-yax')
  for (const t of yTicks)
    f.add(
      yax,
      f.el('span', 'tri-cax-yt', t.label, { style: `top:${((t.vbY / vbH) * 100).toFixed(2)}%` }),
    )
  const stage = f.el('div', 'tri-cax-stage')
  if (axes && (yTicks.length >= 2 || axisRange)) {
    const pcts = yTicks.map(t => (t.vbY / vbH) * 100)
    const top = axisRange ? (axisRange.top / vbH) * 100 : Math.min(...pcts)
    const base = axisRange ? (axisRange.bottom / vbH) * 100 : Math.max(...pcts)
    f.add(
      stage,
      f.el('span', 'tri-cax-ax tri-cax-ax--y', undefined, {
        style: `top:${top.toFixed(2)}%;height:${(base - top).toFixed(2)}%`,
      }),
      f.el('span', 'tri-cax-ax tri-cax-ax--x', undefined, { style: `top:${base.toFixed(2)}%` }),
    )
  }
  f.add(stage, svgEl, ...stageOverlays)
  const xax = f.el('div', 'tri-cax-xax')
  for (const t of xTicks)
    f.add(
      xax,
      f.el(t.tag ?? 'span', `tri-cax-xt${t.cls ? ` ${t.cls}` : ''}`, t.label, {
        ...t.attrs,
        style: `left:${t.pct.toFixed(2)}%`,
      }),
    )
  if (rightYTicks.length > 0) {
    const rightYax = f.el('div', 'tri-cax-yax tri-cax-yax--right')
    for (const t of rightYTicks)
      f.add(
        rightYax,
        f.el('span', 'tri-cax-yt tri-cax-yt--right', t.label, {
          style: `top:${((t.vbY / vbH) * 100).toFixed(2)}%`,
        }),
      )
    f.add(frame, yax, stage, rightYax, xax)
  } else f.add(frame, yax, stage, xax)
  return frame
}

export const zoneDuo = <N>(f: TriNodeFactory<N>, a: N | null, b: N | null): N | null => {
  if (!a || !b) return a ?? b
  const duo = f.el('div', 'tri-zone-duo')
  f.add(duo, a, b)
  return duo
}

const zoneTable = <N>(
  f: TriNodeFactory<N>,
  title: string,
  times: number[],
  bounds: number[],
  names: string[],
  unit: string,
  caption: string,
): N => {
  const wrap = f.el('div', 'tri-zone', undefined, { 'data-tri-trace': triathlonTraceName(title) })
  f.add(wrap, f.el('div', 'tri-zone-title', title, { 'data-i18n': title }))
  const total = times.reduce((s, x) => s + x, 0) || 1
  let mx = 1
  for (const t of times) if (t > mx) mx = t
  const grid = f.el('div', 'tri-zone-grid', undefined, { role: 'list' })
  for (let i = times.length - 1; i >= 0; i--) {
    const name = names[i] ?? `Z${i + 1}`
    const range = `${zoneRange(bounds, i)}${unit}`
    const time = zoneClock(times[i])
    const percentage = `${((times[i] / total) * 100).toFixed(1)}%`
    const row = f.el('div', 'tri-zone-row', undefined, {
      role: 'listitem',
      'aria-label': `Z${i + 1}, ${name}, ${range}, ${time}, ${percentage}`,
    })
    const track = f.el('span', 'tri-zone-bar')
    f.add(
      track,
      f.el('span', `tri-zone-fill tri-zone-fill--${i + 1}`, undefined, {
        style: `width:${(times[i] / mx) * 100}%`,
      }),
    )
    f.add(
      row,
      f.el('span', 'tri-zone-z', `Z${i + 1}`),
      f.el('span', 'tri-zone-name', name),
      f.el('span', 'tri-zone-range', range),
      f.el('span', 'tri-zone-time', time),
      f.el('span', 'tri-zone-pct', percentage),
      track,
    )
    f.add(grid, row)
  }
  f.add(wrap, grid)
  if (caption) f.add(wrap, f.el('div', 'tri-zone-cap', caption))
  return wrap
}

export const buildHrZones = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
): N | null => {
  if (!d.hrZones || !ctx.zones?.hr.length) return null
  return zoneTable(
    f,
    'heart rate zones',
    d.hrZones,
    ctx.zones.hr,
    HR_ZONE_NAMES,
    '',
    ctx.vt1 != null ? `based on vt1 ${ctx.vt1} bpm` : '',
  )
}

export const buildPowerZones = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
): N | null => {
  if (!d.powerZones || !ctx.zones?.power.length) return null
  const ftp = ctx.zones.ftp
  return zoneTable(
    f,
    'power zones',
    d.powerZones,
    ctx.zones.power,
    POWER_ZONE_NAMES,
    'w',
    ftp != null ? `based on FTP ${ftp} W` : '',
  )
}

export const buildPowerHist = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const hist = d.powerHist
  if (!hist || hist.length < 2) return null
  const wrap = f.el('div', 'tri-zone', undefined, {
    'data-tri-trace': triathlonTraceName('25W power distribution'),
  })
  f.add(
    wrap,
    f.el('div', 'tri-zone-title', '25W power distribution', {
      'data-i18n': '25W power distribution',
    }),
  )
  const H = 34
  const n = hist.length
  const histMaxWatt = n * 25
  let mx = 1
  for (const t of hist) if (t > mx) mx = t
  const s = f.svg('svg', {
    class: 'tri-hist-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
    'data-hist': JSON.stringify(hist),
  })
  hist.forEach((t, i) => {
    if (t <= 0) return
    const h = (t / mx) * (H - 1)
    f.add(
      s,
      f.svg('rect', {
        x: i + 0.1,
        y: H - h,
        width: 0.8,
        height: h,
        class: 'tri-hist-bar',
        'data-bin': i,
      }),
    )
  })
  const np = d.npWatts ?? d.avgWatts
  if (np != null)
    f.add(
      s,
      f.svg('line', { x1: np / 25 + 0.5, y1: 0, x2: np / 25 + 0.5, y2: H, class: 'tri-hist-avg' }),
    )
  f.add(s, f.svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H }))
  const histStepW = histMaxWatt <= 300 ? 100 : histMaxWatt <= 700 ? 200 : 300
  const histXTicks: AxisXTick[] = []
  for (let w = 0; w < histMaxWatt; w += histStepW)
    histXTicks.push({
      label: `${w}w`,
      pct: (w / 25 / n) * 100,
      cls: w === 0 ? 'tri-cax-xt--first' : undefined,
    })
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      [
        { label: zoneClock(mx), vbY: H - (H - 1) },
        { label: zoneClock(Math.round(mx / 2)), vbY: H - (H - 1) / 2 },
        { label: '0', vbY: H },
      ],
      H,
      histXTicks,
    ),
  )
  f.add(wrap, f.el('div', 'tri-chart-readout'))
  const cap = f.el('div', 'tri-elev-cap')
  f.add(cap, f.el('span', 'tri-ana-k', `0–${(n - 1) * 25 + 24} W`))
  if (np != null) f.add(cap, f.el('span', 'tri-ana-k', `wtd avg ${np} W`))
  f.add(wrap, cap)
  return wrap
}

type PowerCurveRange = 'six-weeks' | 'year'

export const buildCriticalPowerAnchorLinks = <N>(
  f: TriNodeFactory<N>,
  estimate: CriticalPowerEstimate,
  range: PowerCurveRange,
  selected: boolean,
  excludeActivityId?: number | string,
): N => {
  const links = f.el('span', 'tri-critical-power-anchors', undefined, {
    'data-critical-power-range': range,
    ...(selected ? {} : { hidden: '' }),
  })
  for (const anchor of estimate.anchors) {
    const powerPoint = {
      s: anchor.durationS,
      w: anchor.meanPowerWatts,
      activityId: anchor.activityId,
      activityDate: anchor.activityDate,
    }
    const link = f.el(
      'a',
      'tri-critical-power-anchor',
      undefined,
      powerCurveActivityLinkAttributes(powerPoint, excludeActivityId),
    )
    f.add(
      link,
      f.el('span', 'tri-critical-power-anchor-duration', dlabel(anchor.durationS)),
      f.el('span', 'tri-critical-power-anchor-separator', '·', { 'aria-hidden': 'true' }),
      f.el('span', 'tri-critical-power-anchor-date', anchor.activityDate),
      f.el('span', 'tri-critical-power-anchor-separator', '·', { 'aria-hidden': 'true' }),
      f.el(
        'span',
        'tri-critical-power-anchor-power',
        `${anchor.meanPowerWatts.toLocaleString(f.presentation.locale === 'fr' ? 'fr-CA' : 'en-US', { maximumFractionDigits: 1 })}W`,
      ),
    )
    f.add(links, link)
  }
  return links
}

const addPowerCurveThresholdCaption = <N>(
  f: TriNodeFactory<N>,
  caption: N,
  estimates: ReadonlyArray<readonly [PowerCurveRange, CriticalPowerEstimate | null]>,
  selectedRange: PowerCurveRange,
  ftp: number | null,
  goalFtp: number | null,
  excludeActivityId?: number | string,
): void => {
  const thresholds = f.el('span', 'tri-curve-thresholds')
  const anchorRows: N[] = []
  let hasThreshold = false
  for (const [range, estimate] of estimates) {
    if (!estimate) continue
    f.add(
      thresholds,
      f.el(
        'span',
        'tri-ana-k tri-curve-cp-k',
        criticalPowerSummaryText(f.presentation.locale, estimate),
        {
          'data-critical-power-range': range,
          'data-gloss': '',
          'data-gloss-def': criticalPowerEvidenceText(f.presentation.locale, estimate),
          tabindex: '0',
          ...(selectedRange === range ? {} : { hidden: '' }),
        },
      ),
    )
    anchorRows.push(
      buildCriticalPowerAnchorLinks(f, estimate, range, selectedRange === range, excludeActivityId),
    )
    hasThreshold = true
  }
  if (ftp != null) {
    f.add(thresholds, f.el('span', 'tri-ana-k tri-curve-ftp-k', `FTP ${ftp}W`))
    hasThreshold = true
  }
  if (goalFtp != null) {
    f.add(thresholds, f.el('span', 'tri-ana-k tri-curve-goal-k', `goal ${goalFtp}W`))
    hasThreshold = true
  }
  if (hasThreshold) f.add(caption, thresholds, ...anchorRows)
}

const addActivityCriticalPowerCaption = <N>(
  f: TriNodeFactory<N>,
  caption: N,
  estimate: CriticalPowerEstimate,
): void => {
  const thresholds = f.el('span', 'tri-curve-thresholds')
  f.add(
    thresholds,
    f.el(
      'span',
      'tri-ana-k tri-curve-cp-k tri-curve-cp-k--ride',
      `${triText(f.presentation.locale, 'this ride')} · ${criticalPowerSummaryText(f.presentation.locale, estimate)}`,
      {
        'data-gloss': '',
        'data-gloss-def': criticalPowerEvidenceText(f.presentation.locale, estimate),
        tabindex: '0',
      },
    ),
  )
  f.add(caption, thresholds)
}

const buildPowerCurveRanges = <N>(
  f: TriNodeFactory<N>,
  selected: PowerCurveRange,
  year: number | null,
  sixWeeksAvailable: boolean,
  yearAvailable: boolean,
): N | null => {
  if (year == null || !yearAvailable) return null
  const ranges = f.el('div', 'tri-curve-ranges', undefined, {
    role: 'group',
    'aria-label': 'comparison range',
    'data-i18n-aria-label': 'comparison range',
  })
  const sixWeekAttrs: Record<string, string> = {
    type: 'button',
    'data-curve-range': 'six-weeks',
    'aria-pressed': String(selected === 'six-weeks'),
    'data-i18n': '6 weeks',
  }
  if (!sixWeeksAvailable) sixWeekAttrs.disabled = ''
  const yearButton = f.el('button', 'tri-curve-range', undefined, {
    type: 'button',
    'data-curve-range': 'year',
    'aria-pressed': String(selected === 'year'),
  })
  f.add(
    yearButton,
    f.el('span', undefined, 'all of', { 'data-i18n': 'all of' }),
    f.el('span', undefined, ` ${year}`),
  )
  f.add(ranges, f.el('button', 'tri-curve-range', '6 weeks', sixWeekAttrs), yearButton)
  return ranges
}

export const buildPowerCurve = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
  embedded = false,
): N | null => {
  const curve = d.powerCurve
  if (!curve || curve.length < 2) return null
  const isBike = d.sport === 'bike'
  const sixWeekRef = isBike ? ctx.curveRef : []
  const yearRef = isBike ? ctx.curveYearRef : []
  const activityCriticalPower = isBike ? d.activityCriticalPower : null
  const activityModel =
    activityCriticalPower != null
      ? criticalPowerCurve(activityCriticalPower, curve[0].s, curve[curve.length - 1].s)
      : []
  const ftpRef = isBike ? ctx.ftp : null
  const goalRef = isBike ? ctx.goalFtp : null
  const wrap = f.el('div', 'tri-zone tri-curve-chart', undefined, {
    'data-tri-trace': triathlonTraceName('power curve'),
  })
  const W = 100
  const H = 34
  const secs = curve.map(c => c.s)
  const visibleSixWeekRef = sixWeekRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const visibleYearRef = yearRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const defaultRange = visibleSixWeekRef.length > 0 ? 'six-weeks' : 'year'
  const visibleRef = defaultRange === 'six-weeks' ? visibleSixWeekRef : visibleYearRef
  const head = f.el('div', 'tri-curve-head')
  f.add(head, f.el('div', 'tri-zone-title', 'power curve', { 'data-i18n': 'power curve' }))
  const ranges = buildPowerCurveRanges(
    f,
    defaultRange,
    ctx.curveYear,
    visibleSixWeekRef.length > 0,
    visibleYearRef.length > 0,
  )
  if (ranges) f.add(head, ranges)
  f.add(wrap, head)
  const observedMaxW = Math.max(
    1,
    ...curve.map(c => c.w),
    ...visibleSixWeekRef.map(c => c.w),
    ...visibleYearRef.map(c => c.w),
    ...activityModel.map(c => c.w),
    activityCriticalPower?.criticalPowerWatts ?? 0,
    ftpRef ?? 0,
    goalRef ?? 0,
  )
  const curveStep = niceStep(observedMaxW, 4)
  const curveMax = Math.ceil(observedMaxW / curveStep) * curveStep
  const curveTicks = Array.from(
    { length: Math.round(curveMax / curveStep) + 1 },
    (_, index) => index * curveStep,
  )
  const X = (sec: number): number => powerCurveFraction(sec, secs[0], secs[secs.length - 1]) * W
  const Y = (w: number): number => H - (w / curveMax) * (H - 1)
  const toPath = (pts: PowerCurvePoint[]): string =>
    powerCurvePathPoints(pts)
      .map((c, i) => `${i ? 'L' : 'M'} ${X(c.s).toFixed(2)} ${Y(c.w).toFixed(2)}`)
      .join(' ')
  const initialValueText = `${zoneClock(curve[0].s)} · ${curve[0].w.toLocaleString('en-US')} W`
  const s = f.svg('svg', {
    class: 'tri-curve-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    'data-curve': encodePowerCurve(curve),
    'data-curve-ref-six-weeks': encodePowerCurve(visibleSixWeekRef),
    'data-curve-ref-year': encodePowerCurve(visibleYearRef),
    'data-curve-range': defaultRange,
    'data-curve-year': ctx.curveYear ?? '',
    'data-curve-domain-max': curveMax,
    'data-curve-selected-index': 0,
    'data-i18n-aria-label': 'power curve',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'power curve',
    'aria-orientation': 'horizontal',
    'aria-valuemin': curve[0].s,
    'aria-valuemax': curve[curve.length - 1].s,
    'aria-valuenow': curve[0].s,
    'aria-valuetext': initialValueText,
  })
  if (visibleSixWeekRef.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(visibleSixWeekRef),
        class: 'tri-curve-ref',
        'data-curve-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (visibleYearRef.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(visibleYearRef),
        class: 'tri-curve-ref',
        'data-curve-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  if (activityModel.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(activityModel),
        class: 'tri-curve-model tri-curve-model--ride',
        'aria-hidden': 'true',
      }),
    )
  if (activityCriticalPower)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(activityCriticalPower.criticalPowerWatts).toFixed(2),
        x2: W,
        y2: Y(activityCriticalPower.criticalPowerWatts).toFixed(2),
        class: 'tri-curve-cp tri-curve-cp--ride',
        'aria-hidden': 'true',
      }),
    )
  if (ftpRef != null)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(ftpRef).toFixed(2),
        x2: W,
        y2: Y(ftpRef).toFixed(2),
        class: 'tri-curve-ftp',
      }),
    )
  if (goalRef != null)
    f.add(
      s,
      f.svg('line', {
        x1: 0,
        y1: Y(goalRef).toFixed(2),
        x2: W,
        y2: Y(goalRef).toFixed(2),
        class: 'tri-curve-goal',
      }),
    )
  f.add(s, f.svg('path', { d: toPath(curve), class: 'tri-curve-line' }))
  f.add(s, f.svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H }))
  const durationMarkers = [1, 60, 300, 1200, 3600, 10_800]
  const curveDurTicks = embedded
    ? embeddedPowerCurveDurationTicks(secs[0], secs[secs.length - 1], durationMarkers)
    : powerCurveDurationTicks(secs[0], secs[secs.length - 1], durationMarkers)
  const pointMarkers: N[] = []
  if (visibleRef.length > 0) {
    const initialRef = visibleRef.find(point => point.s === curve[0].s)
    const attrs: Record<string, string> = {
      'aria-hidden': 'true',
      style: `left:${X(curve[0].s).toFixed(2)}%;top:${((Y(initialRef?.w ?? 0) / H) * 100).toFixed(2)}%`,
    }
    if (!initialRef) attrs.hidden = ''
    pointMarkers.push(f.el('span', 'tri-curve-point tri-curve-point--ref', undefined, attrs))
  }
  pointMarkers.push(
    f.el('span', 'tri-curve-point tri-curve-point--ride', undefined, {
      'aria-hidden': 'true',
      style: `left:${X(curve[0].s).toFixed(2)}%;top:${((Y(curve[0].w) / H) * 100).toFixed(2)}%`,
    }),
  )
  const readout = f.el('div', 'tri-chart-readout tri-curve-readout')
  f.add(readout, f.el('span', 'tri-curve-readout-duration'))
  const rideRow = f.el('span', 'tri-curve-readout-row')
  f.add(
    rideRow,
    f.el('span', 'tri-curve-readout-swatch tri-curve-readout-swatch--ride', undefined, {
      'aria-hidden': 'true',
    }),
    f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--ride'),
    f.el('span', 'tri-curve-readout-label', 'this ride', { 'data-i18n': 'this ride' }),
  )
  f.add(readout, rideRow)
  if (visibleRef.length > 0) {
    const referenceRow = f.el(
      'a',
      'tri-curve-readout-row tri-curve-readout-row--ref',
      undefined,
      powerCurveActivityLinkAttributes(nearestPowerCurvePoint(visibleRef, curve[0].s), d.id),
    )
    f.add(
      referenceRow,
      f.el('span', 'tri-curve-readout-swatch tri-curve-readout-swatch--ref', undefined, {
        'aria-hidden': 'true',
      }),
      f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--ref'),
      f.el(
        'span',
        'tri-curve-readout-label tri-curve-readout-label--ref',
        defaultRange === 'year' && ctx.curveYear != null ? `${ctx.curveYear} best` : '6-week best',
        defaultRange === 'six-weeks' ? { 'data-i18n': '6-week best' } : undefined,
      ),
    )
    f.add(readout, referenceRow)
  }
  const addCriticalPowerReadout = (estimate: CriticalPowerEstimate, label: string): void => {
    const durations = estimate.anchors.map(anchor => anchor.durationS)
    const minDuration = Math.min(...durations)
    const maxDuration = Math.max(...durations)
    const attrs: Record<string, string> = {
      'data-curve-critical-power': String(estimate.criticalPowerWatts),
      'data-curve-w-prime': String(estimate.wPrimeJoules),
      'data-curve-model-min-seconds': String(minDuration),
      'data-curve-model-max-seconds': String(maxDuration),
      hidden: '',
    }
    const row = f.el(
      'span',
      'tri-curve-readout-row tri-curve-readout-row--model tri-curve-readout-row--model-ride',
      undefined,
      attrs,
    )
    f.add(
      row,
      f.el(
        'span',
        'tri-curve-readout-swatch tri-curve-readout-swatch--model tri-curve-readout-swatch--model-ride',
        undefined,
        { 'aria-hidden': 'true' },
      ),
      f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--model'),
      f.el('span', 'tri-curve-readout-label tri-curve-readout-label--model', label),
    )
    f.add(readout, row)
  }
  if (activityCriticalPower)
    addCriticalPowerReadout(
      activityCriticalPower,
      triText(f.presentation.locale, 'this ride eCP model'),
    )
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      curveTicks.map(value => ({
        label: value === 0 ? '0' : `${axisNumber(value, curveStep)}w`,
        vbY: Y(value),
      })),
      H,
      curveDurTicks.map((sec, idx) => ({
        label: dlabel(sec),
        pct: X(sec),
        cls: `tri-curve-tick${idx === 0 ? ' tri-cax-xt--first' : sec === secs[secs.length - 1] ? ' tri-cax-xt--last' : ''}`,
        tag: 'button',
        attrs: {
          type: 'button',
          'data-curve-seconds': String(sec),
          'aria-pressed': String(sec === curve[0].s),
        },
      })),
      true,
      undefined,
      [...pointMarkers, readout],
    ),
  )
  const cap = f.el('div', 'tri-elev-cap')
  for (const sec of [5, 60, 300, 1200]) {
    const p = curve.find(c => c.s === sec)
    if (p) f.add(cap, f.el('span', 'tri-ana-k', `${dlabel(sec)} ${p.w}W`))
  }
  if (!embedded && ftpRef != null)
    f.add(cap, f.el('span', 'tri-ana-k tri-curve-ftp-k', `FTP ${ftpRef}W`))
  if (!embedded && goalRef != null)
    f.add(cap, f.el('span', 'tri-ana-k tri-curve-goal-k', `goal ${goalRef}W`))
  if (!embedded && activityCriticalPower)
    addActivityCriticalPowerCaption(f, cap, activityCriticalPower)
  f.add(wrap, cap)
  return wrap
}

const RUN_TREND_TARGETS = [
  { distanceKm: 5, label: '5k trend' },
  { distanceKm: 10, label: '10k trend' },
  { distanceKm: 21.0975, label: 'half trend' },
  { distanceKm: 42.195, label: 'marathon trend' },
]
const RUN_RIEGEL_EXPONENT = 1.06
const SWIM_PROJECTION_LENGTHS = [19, 38]

const runTrendRow = (distanceKm: number, movingTimeS: number): [string, string] | null => {
  if (
    !Number.isFinite(distanceKm) ||
    !Number.isFinite(movingTimeS) ||
    distanceKm <= 0 ||
    movingTimeS <= 0
  )
    return null
  const target = RUN_TREND_TARGETS.find(candidate => distanceKm < candidate.distanceKm)
  if (!target) return null
  const predictedTimeS = movingTimeS * Math.pow(target.distanceKm / distanceKm, RUN_RIEGEL_EXPONENT)
  return [target.label, dur(predictedTimeS)]
}

const swimProjection = (d: StravaActivityDetail): string => {
  const pace = positiveMetric(d.swimPaceSPer100m)
    ? d.swimPaceSPer100m
    : swimPaceSeconds(d.distanceKm * 1_000, d.movingTimeS)
  return positiveMetric(pace)
    ? SWIM_PROJECTION_LENGTHS.map(length => dur(pace * length)).join(' / ')
    : '—'
}

const primarySwimStroke = (d: StravaActivityDetail): string => {
  const entries: [SwimStroke, number][] = SWIM_STROKES.map((stroke): [SwimStroke, number] => [
    stroke,
    d.strokes?.[stroke] ?? 0,
  ])
    .filter(([, distanceM]) => distanceM > 0)
    .sort((left, right) => right[1] - left[1])
  return entries[0] ? STROKE_LABEL[entries[0][0]] : STROKE_LABEL.freestyle
}

const strengthMass = (presentation: TriathlonPresentation, kilograms: number): string => {
  const imperial = isImperial(presentation)
  const value = imperial ? kilograms / LB_TO_KG : kilograms
  const rounded = Math.round(value * 10) / 10
  return `${rounded.toLocaleString('en-US', { maximumFractionDigits: 1 })} ${imperial ? 'lb' : 'kg'}`
}

const strengthEffort = (presentation: TriathlonPresentation, set: ActivityStrengthSet): string => {
  const effort =
    set.repetitions != null
      ? `${set.repetitions} ${set.repetitions === 1 ? 'rep' : 'reps'}`
      : set.durationS != null
        ? dlabel(set.durationS)
        : 'set'
  return set.weightKg == null ? effort : `${effort} @ ${strengthMass(presentation, set.weightKg)}`
}

const activityTrainingRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): [string, string][] => {
  const garmin = d.garmin
  const locale = presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const rows: [string, string][] = []
  const intensityFactor = garmin?.intensityFactor ?? d.calculatedIntensityFactor?.value
  if (intensityFactor != null)
    rows.push([
      'intensity factor',
      intensityFactor.toLocaleString(locale, {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3,
      }),
    ])
  rows.push(['training effect', triText(presentation.locale, activityTrainingEffectLabel(d))])
  const exerciseLoad = garmin?.exerciseLoad ?? d.calculatedExerciseLoad?.value
  if (exerciseLoad != null)
    rows.push(['exercise load', Math.round(exerciseLoad).toLocaleString(locale)])
  return rows
}

const TRAINING_EFFECT_LABELS: Record<string, string> = {
  RECOVERY: 'recovery',
  AEROBIC_BASE: 'base',
  TEMPO: 'tempo',
  LACTATE_THRESHOLD: 'threshold',
  THRESHOLD: 'threshold',
  VO2_MAX: 'VO2max',
  VO2MAX: 'VO2max',
  ANAEROBIC_CAPACITY: 'anaerobic',
  ANAEROBIC: 'anaerobic',
  SPEED: 'speed',
  SPRINT: 'sprint',
}

export const formatTrainingEffectLabel = (value: string | null): string | null => {
  const key = value?.trim().toUpperCase()
  if (!key) return null
  return TRAINING_EFFECT_LABELS[key] ?? key.replaceAll('_', ' ').toLowerCase()
}

export const activityTrainingEffectLabel = (d: StravaActivityDetail): string =>
  formatTrainingEffectLabel(d.garmin?.trainingEffectLabel ?? null) ??
  (d.sport === 'strength' || d.sport === 'yoga' || d.sport === 'treatment' || d.sport === 'sauna'
    ? 'recovery'
    : 'base')

export const formatTrainingEffectNote = (value: string | null): string | null => {
  const key = value?.trim()
  if (!key) return null
  return key
    .replace(/_\d+$/, '')
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\bvo2 max\b/g, 'VO2max')
}

type TrainingEffectGroup = 'low-aerobic' | 'high-aerobic' | 'anaerobic'

export const dominantTrainingEffectGroup = (dominant: string | null): TrainingEffectGroup => {
  const key = dominant?.trim().toUpperCase() ?? ''
  if (/ANAEROBIC|SPEED|SPRINT/.test(key)) return 'anaerobic'
  return /TEMPO|THRESHOLD|VO2/.test(key) ? 'high-aerobic' : 'low-aerobic'
}

const trainingEffectGroup = (
  effect: 'aerobic' | 'anaerobic',
  message: string | null,
  dominant: string | null,
): TrainingEffectGroup => {
  if (effect === 'anaerobic') return 'anaerobic'
  const key = `${message ?? ''} ${dominant ?? ''}`.toUpperCase()
  return /TEMPO|THRESHOLD|VO2/.test(key) ? 'high-aerobic' : 'low-aerobic'
}

const trainingEffectScore = (score: number | null): number | null =>
  score == null || !Number.isFinite(score) ? null : Math.min(5, Math.max(0, score))

const fallbackTrainingEffectNote = (
  effect: 'aerobic' | 'anaerobic',
  score: number | null,
): string | null => {
  const value = trainingEffectScore(score)
  if (value == null) return null
  if (value < 1) return `no ${effect} benefit`
  if (value < 2) return `minor ${effect} benefit`
  if (value < 3) return `maintaining ${effect} fitness`
  if (value < 4) return `improving ${effect} fitness`
  if (value < 5) return `highly improving ${effect} fitness`
  return `overreaching ${effect} effect`
}

const trainingEffectNote = (
  effect: 'aerobic' | 'anaerobic',
  score: number | null,
  message: string | null,
): string | null => formatTrainingEffectNote(message) ?? fallbackTrainingEffectNote(effect, score)

export const buildTrainingEffectDetails = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const garmin = d.garmin
  const hasGarminDetails =
    garmin?.aerobicTrainingEffect != null ||
    garmin?.anaerobicTrainingEffect != null ||
    garmin?.aerobicTrainingEffectMessage != null ||
    garmin?.anaerobicTrainingEffectMessage != null
  const calculated = hasGarminDetails ? null : d.calculatedTrainingEffect
  if (!hasGarminDetails && !calculated) return null
  const locale = f.presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const effects = [
    {
      label: 'aerobic',
      score: garmin?.aerobicTrainingEffect ?? calculated?.aerobic ?? null,
      note: trainingEffectNote(
        'aerobic',
        garmin?.aerobicTrainingEffect ?? calculated?.aerobic ?? null,
        hasGarminDetails ? (garmin?.aerobicTrainingEffectMessage ?? null) : null,
      ),
      group: trainingEffectGroup(
        'aerobic',
        hasGarminDetails ? (garmin?.aerobicTrainingEffectMessage ?? null) : null,
        hasGarminDetails ? (garmin?.trainingEffectLabel ?? null) : null,
      ),
    },
    {
      label: 'anaerobic',
      score: garmin?.anaerobicTrainingEffect ?? calculated?.anaerobic ?? null,
      note: trainingEffectNote(
        'anaerobic',
        garmin?.anaerobicTrainingEffect ?? calculated?.anaerobic ?? null,
        hasGarminDetails ? (garmin?.anaerobicTrainingEffectMessage ?? null) : null,
      ),
      group: trainingEffectGroup(
        'anaerobic',
        hasGarminDetails ? (garmin?.anaerobicTrainingEffectMessage ?? null) : null,
        hasGarminDetails ? (garmin?.trainingEffectLabel ?? null) : null,
      ),
    },
  ].filter(effect => effect.score != null || effect.note != null)
  if (effects.length === 0) return null
  const titleKey = 'training effect'
  const wrap = f.el('section', 'tri-zone tri-training-effect', undefined, {
    'aria-label': triText(f.presentation.locale, titleKey),
    'data-i18n-aria-label': titleKey,
    'data-training-effect-source': calculated ? 'calculated' : 'garmin',
  })
  f.add(
    wrap,
    f.el('div', 'tri-zone-title', triText(f.presentation.locale, titleKey), {
      'data-i18n': titleKey,
    }),
  )
  const list = f.el('div', 'tri-training-effect-list')
  for (const effect of effects) {
    const score = trainingEffectScore(effect.score)
    const item = f.el('div', 'tri-training-effect-item', undefined, {
      'data-training-effect-group': effect.group,
    })
    const meter = f.el(
      'div',
      'tri-training-effect-meter',
      undefined,
      score == null
        ? { 'aria-hidden': 'true' }
        : {
            role: 'meter',
            'aria-label': effect.label,
            'data-i18n-aria-label': effect.label,
            'aria-valuemin': '0',
            'aria-valuemax': '5',
            'aria-valuenow': `${score}`,
          },
    )
    f.add(
      meter,
      f.el('span', 'tri-training-effect-meter-fill', undefined, {
        style: `--tri-training-effect-progress:${((score ?? 0) * 20).toFixed(1)}%`,
      }),
    )
    f.add(
      item,
      f.el('span', 'tri-training-effect-label', effect.label, { 'data-i18n': effect.label }),
      meter,
      f.el(
        'span',
        'tri-training-effect-score',
        effect.score?.toLocaleString(locale, {
          minimumFractionDigits: 1,
          maximumFractionDigits: 1,
        }) ?? '—',
      ),
    )
    if (effect.note) f.add(item, f.el('p', 'tri-training-effect-note', effect.note))
    f.add(list, item)
  }
  f.add(wrap, list)
  return wrap
}

export const strengthExerciseSummary = (
  presentation: TriathlonPresentation,
  exercise: ActivityStrengthExercise,
): string => {
  const sets = `${exercise.setCount} ${exercise.setCount === 1 ? 'set' : 'sets'}`
  if (exercise.sets.length === 0) {
    if (exercise.repetitions != null)
      return `${sets} · ${exercise.repetitions} ${exercise.repetitions === 1 ? 'rep' : 'reps'}`
    if (exercise.durationS != null) return `${sets} · ${dlabel(exercise.durationS)}`
    return sets
  }
  const efforts = exercise.sets.map(set => strengthEffort(presentation, set))
  if (efforts.every(effort => effort === efforts[0])) return `${sets} · ${efforts[0]} each`
  return `${sets} · ${efforts.join(', ')}`
}

export const buildStrengthExercises = <N>(
  f: TriNodeFactory<N>,
  strength: ActivityStrength,
): N | null => {
  if (strength.exercises.length === 0) return null
  const wrap = f.el('section', 'tri-act-strength')
  f.add(wrap, f.el('h3', 'tri-act-strength-h', 'exercises'))
  const list = f.el('ol', 'tri-strength-exercises')
  for (const exercise of strength.exercises) {
    const item = f.el('li', 'tri-strength-exercise')
    f.add(
      item,
      f.el('span', 'tri-strength-exercise-name', exercise.name),
      f.el(
        'span',
        'tri-strength-exercise-summary',
        strengthExerciseSummary(f.presentation, exercise),
      ),
    )
    f.add(list, item)
  }
  f.add(wrap, list)
  return wrap
}

export const activityStatRows = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): [string, string][] => {
  if (d.sport === 'sauna' && d.sauna) {
    const rows: [string, string][] = [
      ['time', d.sauna.time],
      ['duration', dur(d.movingTimeS)],
      ['temperature', formatTemperature(presentation, d.sauna.temperatureC)],
      [
        'humidity',
        `${d.sauna.humidityPct.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
          maximumFractionDigits: 1,
        })}%`,
      ],
      ['cooldown', d.sauna.cooldown],
    ]
    if (d.sauna.heatTrainingLoad != null)
      rows.push([
        'HTL',
        d.sauna.heatTrainingLoad.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
          maximumFractionDigits: 1,
        }),
      ])
    if (d.avgHr)
      rows.push(['avg hr', `${d.avgHr} bpm${d.sauna.heartRateSource === 'oura' ? ' · Oura' : ''}`])
    return rows
  }
  if (d.sport === 'strength') {
    const rows: [string, string][] = [['time', dur(d.movingTimeS)]]
    if (d.strength?.volumeKg != null)
      rows.push(['volume', strengthMass(presentation, d.strength.volumeKg)])
    if (d.strength?.totalSets != null) rows.push(['sets', String(d.strength.totalSets)])
    if (d.strength?.totalReps != null) rows.push(['reps', String(d.strength.totalReps)])
    if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
    rows.push(...activityTrainingRows(presentation, d))
    return rows
  }
  if (d.sport === 'treatment' || d.sport === 'yoga') {
    const rows: [string, string][] = [['time', dur(d.movingTimeS)]]
    if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
    rows.push(...activityTrainingRows(presentation, d))
    return rows
  }
  const activityRate =
    d.sport === 'swim' && positiveMetric(d.swimPaceSPer100m)
      ? `${clock(d.swimPaceSPer100m)} /100m`
      : rate(presentation, d.sport, d.distanceKm, d.movingTimeS)
  const rows: [string, string][] = [
    ['distance', dist(presentation, d.distanceKm, d.sport)],
    ['time', dur(d.movingTimeS)],
    [d.sport === 'bike' ? 'speed' : 'pace', activityRate],
  ]
  if (d.computer) rows.push(['computer', COMPUTER_LABEL[d.computer]])
  if (d.sport === 'bike' && d.maxSpeedKph != null)
    rows.push(['max speed', speedKph(presentation, d.maxSpeedKph)])
  if (d.sport === 'run') {
    const trend = runTrendRow(d.distanceKm, d.movingTimeS)
    if (trend) rows.push(trend)
  }
  if (d.sport === 'swim')
    rows.push([
      'stroke rate',
      positiveMetric(d.strokeRateSpm) ? `${Math.round(d.strokeRateSpm)} spm` : '—',
    ])
  if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
  rows.push(...activityTrainingRows(presentation, d))
  if (d.sport === 'swim') {
    const poolMetrics = d.swimLocation === 'pool' ? swimLengthAverages(d.swimIntervals) : null
    if (d.swimLocation !== 'pool') {
      rows.push([
        'water temp',
        d.waterTemperatureC == null ? '—' : formatTemperature(presentation, d.waterTemperatureC),
      ])
    }
    const swolf = poolMetrics
      ? swimActivityHeaderValue(presentation.locale, 'swolf', poolMetrics.swolf, '')
      : '—'
    rows.push(['SWOLF', swolf])
    rows.push(['1.9k / 3.8k', swimProjection(d)])
    rows.push(['stroke type', primarySwimStroke(d)])
    rows.push([
      'strokes',
      positiveMetric(d.strokeCount)
        ? `${Math.round(d.strokeCount).toLocaleString('en-US')} · ${(
            (d.distanceKm * 1_000) /
            d.strokeCount
          ).toFixed(2)} m/str`
        : '—',
    ])
  }
  return rows
}

export const buildActivity = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  expanded = false,
  ctx?: DetailCtx,
  fillMissingRunPower = false,
  embedded = false,
  traceSettings?: TriathlonTraceSettings,
  reserveFueling = false,
): N => {
  const normalizeBikeMetrics = excludesZeroPower(f.presentation) && d.sport === 'bike'
  const normalizedPower = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.w)
    : null
  const normalizedCadence = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.cad)
    : null
  const activityAnchor = triathlonActivityAnchor(d.id)
  const summaryTrainingEffectGroup = dominantTrainingEffectGroup(activityTrainingEffectLabel(d))
  const wrap = f.el('section', expanded ? 'tri-act tri-act--expanded' : 'tri-act', undefined, {
    ...(activityAnchor ? { id: activityAnchor } : {}),
    'data-activity-id': `${d.id}`,
    'data-activity-title': d.name || d.sport,
  })
  const head = f.el('div', 'tri-act-head')
  f.add(head, buildIcon(f, d.sport))
  f.add(wrap, head)
  f.add(
    wrap,
    statsTable(
      f,
      [
        ...activityStatRows(f.presentation, d),
        ...moreStatRows(f.presentation, d, fillMissingRunPower),
      ],
      label =>
        label === 'training effect'
          ? { 'data-training-effect-group': summaryTrainingEffectGroup }
          : undefined,
    ),
  )
  let hasSummaryVisual = false
  if (d.strength) {
    const strength = buildStrengthExercises(f, d.strength)
    if (strength) {
      f.add(wrap, strength)
      hasSummaryVisual = true
    }
  }
  let hasFueling = false
  if (d.fueling) {
    const fueling = buildFueling(f, d.fueling)
    if (fueling) {
      f.add(wrap, fueling)
      hasFueling = true
    }
  }
  if (embedded && reserveFueling && !hasFueling) f.add(wrap, buildReservedFueling(f))
  const analysis = buildAnalysisBar(f, d)
  const analysisSelection = null
  if (d.route.length >= 2) {
    const secondary =
      d.sport === 'swim'
        ? (buildSwimStrokes(f, d) ?? (embedded ? buildUnavailableElevation(f) : null))
        : buildElevation(f, d, analysisSelection)
    const figs = f.el(
      'div',
      `tri-act-figs tri-act-figs--route${secondary ? ' tri-act-figs--split' : ''}`,
    )
    f.add(figs, buildRoute(f, d.route))
    if (secondary) f.add(figs, secondary)
    if (analysis) f.add(figs, analysis)
    f.add(wrap, figs)
    hasSummaryVisual = true
  } else if (d.sport === 'swim') {
    const figs = f.el('div', 'tri-act-figs tri-act-figs--pool')
    const pool = buildPool(f, d)
    if (embedded) f.add(pool, buildPoolOverview(f))
    f.add(figs, pool)
    f.add(wrap, figs)
    hasSummaryVisual = true
  }
  if (embedded && !hasSummaryVisual)
    f.add(
      wrap,
      f.el('div', 'tri-act-figs tri-act-figs--empty', undefined, { 'aria-hidden': 'true' }),
    )
  const poolOverview =
    d.sport === 'swim' && d.route.length < 2 && !embedded ? buildPoolOverview(f) : null
  const swimTrends = buildSwimTrends(f, d, traceSettings)
  {
    const moreId = `tri-act-more-${d.id}`
    const more = f.el('div', 'tri-act-more', undefined, { id: moreId })
    const flags = routeStreamFlags(d)
    const runAnalysis = buildRunAnalysis(f, d)
    if (runAnalysis) f.add(more, runAnalysis)
    if (flags.hr) f.add(more, buildHeartRateTrace(f, d, analysisSelection))
    if (flags.power)
      f.add(
        more,
        buildTrace(
          f,
          d,
          (point, index) => normalizedPower?.[index] ?? point.w,
          'power',
          max => `${max} W peak`,
          value => `${Math.round(value)}w`,
          normalizedPower ? positiveMetricDomain(normalizedPower) : undefined,
          analysisSelection,
          undefined,
          undefined,
          d.sport === 'bike' && d.activityCriticalPower
            ? {
                value: d.activityCriticalPower.criticalPowerWatts,
                label: `eCP ${d.activityCriticalPower.criticalPowerWatts} W`,
              }
            : null,
        ),
      )
    const powerBalance = buildPowerBalanceChart(f, d, analysisSelection, embedded)
    if (powerBalance) f.add(more, powerBalance)
    const torqueEffectiveness = buildTorqueEffectivenessChart(f, d, analysisSelection, embedded)
    if (torqueEffectiveness) f.add(more, torqueEffectiveness)
    const pedalSmoothness = buildPedalSmoothnessChart(f, d, analysisSelection, embedded)
    if (pedalSmoothness) f.add(more, pedalSmoothness)
    const powerPhase = buildPowerPhaseChart(f, d, analysisSelection)
    if (powerPhase) f.add(more, powerPhase)
    const riderPosition = buildRiderPositionChart(f, d, analysisSelection)
    if (riderPosition) f.add(more, riderPosition)
    const stamina = buildStaminaChart(f, d, analysisSelection)
    if (stamina) f.add(more, stamina)
    const shifting = buildShiftingChart(f, d, analysisSelection)
    if (shifting) f.add(more, shifting)
    if (flags.cad) {
      const cadenceScale = d.sport === 'run' ? 2 : 1
      const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
      f.add(
        more,
        buildTrace(
          f,
          d,
          (point, index) => (normalizedCadence?.[index] ?? point.cad) * cadenceScale,
          'cadence',
          max => `${max} ${cadenceUnit} peak`,
          value => `${Math.round(value)}${cadenceUnit}`,
          normalizedCadence
            ? positiveMetricDomain(normalizedCadence.map(value => (value ?? 0) * cadenceScale))
            : undefined,
          analysisSelection,
        ),
      )
    }
    if (flags.stride) {
      const stride = buildRunStrideTrace(f, d, analysisSelection)
      if (stride) f.add(more, stride)
    }
    if (flags.groundContact) {
      const groundContact = buildRunGroundContactTrace(f, d, analysisSelection)
      if (groundContact) f.add(more, groundContact)
    }
    if (flags.verticalOscillation) {
      const verticalOscillation = buildRunVerticalOscillationTrace(f, d, analysisSelection)
      if (verticalOscillation) f.add(more, verticalOscillation)
    }
    if (flags.resp) f.add(more, buildRespirationTrace(f, d, analysisSelection))
    if (flags.muscleOxygen) {
      const muscleOxygen = buildMuscleOxygenTrace(f, d, analysisSelection)
      if (muscleOxygen) f.add(more, muscleOxygen)
    }
    if (flags.temp) f.add(more, buildTemperatureTrace(f, d, analysisSelection))
    if (flags.heatStrain) {
      const heatStrain = buildHeatStrainTrace(f, d, analysisSelection)
      if (heatStrain) f.add(more, heatStrain)
    }
    if (flags.coreTemperature) {
      const coreTemperature = buildCoreTemperatureTrace(f, d, analysisSelection)
      if (coreTemperature) f.add(more, coreTemperature)
    }
    if (flags.skinTemperature) {
      const skinTemperature = buildSkinTemperatureTrace(f, d, analysisSelection)
      if (skinTemperature) f.add(more, skinTemperature)
    }
    if (poolOverview) f.add(more, poolOverview)
    if (swimTrends) f.add(more, swimTrends)
    const trainingEffect = buildTrainingEffectDetails(f, d)
    if (trainingEffect) f.add(more, trainingEffect)
    if (ctx) {
      const zones = zoneDuo(
        f,
        triathlonTraceEnabled(traceSettings, 'heart-rate-zones') ? buildHrZones(f, d, ctx) : null,
        triathlonTraceEnabled(traceSettings, 'power-zones') ? buildPowerZones(f, d, ctx) : null,
      )
      if (zones) f.add(more, zones)
      const charts = zoneDuo(
        f,
        triathlonTraceEnabled(traceSettings, 'power-curve')
          ? buildPowerCurve(f, d, ctx, embedded)
          : null,
        triathlonTraceEnabled(traceSettings, '25w-power-distribution')
          ? buildPowerHist(f, d)
          : null,
      )
      if (charts) f.add(more, charts)
    }
    const bestEfforts = buildCyclingBestEfforts(f, d)
    if (bestEfforts) f.add(more, bestEfforts)
    f.add(
      wrap,
      f.el('button', 'tri-act-toggle', expanded ? '− see less' : '+ see more', {
        type: 'button',
        'aria-expanded': String(expanded),
        'aria-controls': moreId,
      }),
      more,
    )
  }
  return wrap
}

export type ActivityComparisonMetric =
  | 'elevation'
  | 'speed'
  | 'hr'
  | 'power'
  | 'cadence'
  | 'respiration'
  | 'temperature'
  | 'skin-temperature'
  | 'stride-length'
  | 'ground-contact-time'
  | 'vertical-oscillation'
  | 'swim-pace'
  | 'stroke-rate'

export const activityComparisonFractionForKey = (
  key: string,
  fraction: number,
  step: number,
): number | null => {
  let next: number
  if (key === 'ArrowLeft' || key === 'ArrowDown') next = fraction - step
  else if (key === 'ArrowRight' || key === 'ArrowUp') next = fraction + step
  else if (key === 'Home') next = 0
  else if (key === 'End') next = 1
  else return null
  return Math.min(1, Math.max(0, next))
}

type ActivityComparisonGeographicPoint = { lat: number; lng: number; d: number }

type ActivityComparisonMetricPoint = { distanceKm: number; value: number }

type ActivityComparisonSeries = {
  activity: StravaActivityDetail
  index: number
  segments: ActivityComparisonMetricPoint[][]
  values: number[]
}

type ActivityComparisonDomain = { min: number; max: number }

type ActivityComparisonMetricSpec = {
  metric: ActivityComparisonMetric
  title: string
  display: (value: number) => number
  tick: (value: number, sport: ActivityKind) => string
  includeZero: boolean
  domain?: (values: number[]) => ActivityComparisonDomain
}

const ACTIVITY_COMPARISON_WIDTH = 100
const ACTIVITY_COMPARISON_HEIGHT = 34

const activityComparisonMetricSpecs = (
  presentation: TriathlonPresentation,
): Record<ActivityComparisonMetric, ActivityComparisonMetricSpec> => ({
  elevation: {
    metric: 'elevation',
    title: 'elevation',
    display: value => elevationValue(presentation, value),
    tick: value =>
      `${Math.round(value).toLocaleString('en-US')} ${isImperial(presentation) ? 'ft' : 'm'}`,
    includeZero: false,
  },
  speed: {
    metric: 'speed',
    title: 'speed',
    display: value => value,
    tick: value => speedKph(presentation, value),
    includeZero: true,
  },
  hr: {
    metric: 'hr',
    title: 'heart rate',
    display: value => value,
    tick: value => `${Math.round(value)} bpm`,
    includeZero: false,
  },
  power: {
    metric: 'power',
    title: 'power',
    display: value => value,
    tick: value => `${Math.round(value)} W`,
    includeZero: true,
  },
  cadence: {
    metric: 'cadence',
    title: 'cadence',
    display: value => value,
    tick: (value, sport) => `${Math.round(value)} ${sport === 'run' ? 'spm' : 'rpm'}`,
    includeZero: false,
  },
  respiration: {
    metric: 'respiration',
    title: 'respiration',
    display: value => value,
    tick: value => `${Math.round(value)} brpm`,
    includeZero: false,
  },
  temperature: {
    metric: 'temperature',
    title: 'temperature',
    display: value => temperatureValue(presentation, value),
    tick: value => `${Math.round(value)}${temperatureUnit(presentation)}`,
    includeZero: false,
  },
  'skin-temperature': {
    metric: 'skin-temperature',
    title: 'skin temperature',
    display: value => temperatureValue(presentation, value),
    tick: value => `${value.toFixed(2)}${temperatureUnit(presentation)}`,
    includeZero: false,
    domain: values => {
      if (values.length === 0) return { min: 0, max: 1 }
      const resolution = traceResolution(values, isImperial(presentation) ? 0.09 : 0.05)
      return { min: Math.min(...values) - resolution, max: Math.max(...values) + resolution }
    },
  },
  'stride-length': {
    metric: 'stride-length',
    title: 'stride length',
    display: value => value,
    tick: value => formatStrideLength(presentation, value),
    includeZero: false,
  },
  'ground-contact-time': {
    metric: 'ground-contact-time',
    title: 'ground contact time',
    display: value => value,
    tick: formatGroundContactTime,
    includeZero: false,
  },
  'vertical-oscillation': {
    metric: 'vertical-oscillation',
    title: 'vertical oscillation',
    display: value => value,
    tick: value => formatVerticalOscillation(presentation, value),
    includeZero: false,
  },
  'swim-pace': {
    metric: 'swim-pace',
    title: 'pace /100m',
    display: value => value,
    tick: value => clock(value),
    includeZero: false,
  },
  'stroke-rate': {
    metric: 'stroke-rate',
    title: 'stroke rate',
    display: value => value,
    tick: value => `${swimTrendNumber(value)} str/min`,
    includeZero: false,
  },
})

const BIKE_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = [
  'elevation',
  'speed',
  'hr',
  'power',
  'cadence',
  'respiration',
  'temperature',
  'skin-temperature',
]

const RUN_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = [
  'elevation',
  'speed',
  'hr',
  'power',
  'cadence',
  'stride-length',
  'ground-contact-time',
  'vertical-oscillation',
  'temperature',
]

const SWIM_COMPARISON_METRICS: readonly ActivityComparisonMetric[] = ['swim-pace', 'stroke-rate']

export const activityComparisonMetricsForSport = (
  sport: ActivityKind,
): readonly ActivityComparisonMetric[] => {
  if (sport === 'run') return RUN_COMPARISON_METRICS
  if (sport === 'swim') return SWIM_COMPARISON_METRICS
  return BIKE_COMPARISON_METRICS
}

const COMPARE_LIGHTNESS = 0.62
const COMPARE_CHROMA = 0.18

const oklchLinearSrgb = (chroma: number, hueDegrees: number): [number, number, number] => {
  const hue = (hueDegrees * Math.PI) / 180
  const a = chroma * Math.cos(hue)
  const b = chroma * Math.sin(hue)
  const long = (COMPARE_LIGHTNESS + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const medium = (COMPARE_LIGHTNESS - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const short = (COMPARE_LIGHTNESS - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
    -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
    -0.0041960863 * long - 0.7034186147 * medium + 1.707614701 * short,
  ]
}

const withinSrgb = (channels: readonly number[]): boolean =>
  channels.every(channel => channel >= -1e-4 && channel <= 1 + 1e-4)

const srgbHex = (channels: readonly number[]): string =>
  `#${channels
    .map(channel => {
      const clamped = Math.min(1, Math.max(0, channel))
      const encoded = clamped <= 0.0031308 ? 12.92 * clamped : 1.055 * clamped ** (1 / 2.4) - 0.055
      return Math.round(encoded * 255)
        .toString(16)
        .padStart(2, '0')
    })
    .join('')}`

export const activityCompareColor = (index: number): string => {
  const position = Number.isFinite(index) ? Math.max(0, Math.trunc(index)) : 0
  const hue = (25 + position * 137.508) % 360
  if (withinSrgb(oklchLinearSrgb(COMPARE_CHROMA, hue)))
    return srgbHex(oklchLinearSrgb(COMPARE_CHROMA, hue))
  let low = 0
  let high = COMPARE_CHROMA
  for (let iteration = 0; iteration < 12; iteration++) {
    const chroma = (low + high) / 2
    if (withinSrgb(oklchLinearSrgb(chroma, hue))) low = chroma
    else high = chroma
  }
  return srgbHex(oklchLinearSrgb(low, hue))
}

const comparisonPowerCapable = (activity: StravaActivityDetail): boolean =>
  activity.deviceWatts || activity.route.some(point => Number.isFinite(point.w) && point.w > 0)

const comparisonMetricPointValue = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  point: StravaActivityDetail['route'][number],
  metric: ActivityComparisonMetric,
  powerCapable: boolean,
): number | null => {
  let value: number | null
  switch (metric) {
    case 'elevation':
      value = point.alt
      break
    case 'speed':
      value = point.speedKph > 0 ? point.speedKph : null
      break
    case 'hr':
      value = point.hr > 0 ? point.hr : null
      break
    case 'power':
      value =
        point.w > 0 ||
        (point.w === 0 &&
          powerCapable &&
          !(excludesZeroPower(presentation) && activity.sport === 'bike'))
          ? point.w
          : null
      break
    case 'cadence':
      value = point.cad > 0 ? point.cad * (activity.sport === 'run' ? 2 : 1) : null
      break
    case 'respiration':
      value = point.resp != null && point.resp > 0 ? point.resp : null
      break
    case 'temperature':
      value = point.tempC
      break
    case 'skin-temperature':
      value = point.skinTemperatureC
      break
    case 'stride-length':
      value = activity.sport === 'run' ? runStrideLengthValue(activity, point) : null
      break
    case 'ground-contact-time':
      value = activity.sport === 'run' ? runGroundContactTimeMs(point) : null
      break
    case 'vertical-oscillation':
      value = activity.sport === 'run' ? runVerticalOscillationCm(point) : null
      break
    case 'swim-pace':
    case 'stroke-rate':
      value = null
      break
  }
  return value != null && Number.isFinite(value) ? value : null
}

const comparisonSwimMetricSegments = (
  activity: StravaActivityDetail,
  metric: 'swim-pace' | 'stroke-rate',
): ActivityComparisonMetricPoint[][] => {
  if (activity.sport !== 'swim') return []
  const segments: ActivityComparisonMetricPoint[][] = []
  let segment: ActivityComparisonMetricPoint[] = []
  const flush = (): void => {
    if (segment.length > 0) segments.push(segment)
    segment = []
  }
  for (const interval of activity.swimIntervals) {
    const distanceKm = interval.cumulativeDistanceM / 1_000
    const startDistanceKm = (interval.cumulativeDistanceM - interval.distanceM) / 1_000
    const value = metric === 'swim-pace' ? interval.paceSPer100m : interval.strokeRateSpm
    if (
      !Number.isFinite(startDistanceKm) ||
      !Number.isFinite(distanceKm) ||
      startDistanceKm < 0 ||
      distanceKm <= startDistanceKm ||
      !positiveMetric(value)
    ) {
      flush()
      continue
    }
    const previous = segment[segment.length - 1]
    if (previous && distanceKm < previous.distanceKm) flush()
    if (segment.length === 0) segment.push({ distanceKm: startDistanceKm, value })
    const last = segment[segment.length - 1]
    if (last?.distanceKm === distanceKm) {
      last.value = value
      continue
    }
    segment.push({ distanceKm, value })
  }
  flush()
  return segments
}

const comparisonMetricSegments = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
): ActivityComparisonMetricPoint[][] => {
  if (metric === 'swim-pace' || metric === 'stroke-rate')
    return comparisonSwimMetricSegments(activity, metric)
  const powerCapable = comparisonPowerCapable(activity)
  const normalizedValues =
    excludesZeroPower(presentation) &&
    activity.sport === 'bike' &&
    (metric === 'power' || metric === 'cadence')
      ? interpolatePositiveMetricSeries(activity.route, point =>
          comparisonMetricPointValue(presentation, activity, point, metric, powerCapable),
        )
      : null
  const segments: ActivityComparisonMetricPoint[][] = []
  let segment: ActivityComparisonMetricPoint[] = []
  const flush = (): void => {
    if (segment.length > 0) segments.push(segment)
    segment = []
  }
  for (const [index, point] of activity.route.entries()) {
    if (!Number.isFinite(point.d) || point.d < 0) {
      flush()
      continue
    }
    const value =
      normalizedValues?.[index] ??
      comparisonMetricPointValue(presentation, activity, point, metric, powerCapable)
    if (value == null) {
      flush()
      continue
    }
    const previous = segment[segment.length - 1]
    if (previous && point.d < previous.distanceKm) flush()
    const last = segment[segment.length - 1]
    if (last?.distanceKm === point.d) {
      last.value = value
      continue
    }
    segment.push({ distanceKm: point.d, value })
  }
  flush()
  return segments
}

export const activityComparisonMetricAtDistance = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
  distanceKm: number,
): number | null => {
  if (!Number.isFinite(distanceKm)) return null
  for (const segment of comparisonMetricSegments(presentation, activity, metric)) {
    if (distanceKm < segment[0].distanceKm || distanceKm > segment[segment.length - 1].distanceKm)
      continue
    for (let index = 0; index < segment.length; index++) {
      const point = segment[index]
      if (distanceKm === point.distanceKm) return point.value
      if (index === 0 || distanceKm > point.distanceKm) continue
      const previous = segment[index - 1]
      const span = point.distanceKm - previous.distanceKm
      if (span <= 0) return point.value
      const fraction = (distanceKm - previous.distanceKm) / span
      return previous.value + (point.value - previous.value) * fraction
    }
  }
  return null
}

export const activityComparisonDisplayValueAtDistance = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  metric: ActivityComparisonMetric,
  distanceKm: number,
): string => {
  const value = activityComparisonMetricAtDistance(presentation, activity, metric, distanceKm)
  if (value == null) return '—'
  if (metric === 'elevation') return formatAltitude(presentation, value)
  if (metric === 'speed') return speedKph(presentation, value)
  if (metric === 'hr') return `${Math.round(value)} bpm`
  if (metric === 'power') return `${Math.round(value)} W`
  if (metric === 'cadence')
    return `${Math.round(value)} ${activity.sport === 'run' ? 'spm' : 'rpm'}`
  if (metric === 'respiration') return formatRespirationRate(value)
  if (metric === 'stride-length') return formatStrideLength(presentation, value)
  if (metric === 'ground-contact-time') return formatGroundContactTime(value)
  if (metric === 'vertical-oscillation') return formatVerticalOscillation(presentation, value)
  if (metric === 'swim-pace') return `${clock(value)} /100m`
  if (metric === 'stroke-rate') return `${swimTrendNumber(value)} str/min`
  if (metric === 'skin-temperature') return formatThermalTemperature(presentation, value)
  return formatTemperature(presentation, value)
}

const comparisonActivityRoutePoints = (
  activity: StravaActivityDetail,
): StravaActivityDetail['route'] =>
  activity.route.filter(point => Number.isFinite(point.d) && point.d >= 0).sort((a, b) => a.d - b.d)

const comparisonSwimDistanceKm = (activity: StravaActivityDetail): number =>
  activity.sport === 'swim'
    ? activity.swimIntervals.reduce(
        (max, interval) =>
          Number.isFinite(interval.cumulativeDistanceM)
            ? Math.max(max, interval.cumulativeDistanceM / 1_000)
            : max,
        Number.isFinite(activity.distanceKm) && activity.distanceKm > 0 ? activity.distanceKm : 0,
      )
    : 0

const comparisonMaxDistanceKm = (activities: readonly StravaActivityDetail[]): number =>
  activities.reduce((max, activity) => {
    const route = comparisonActivityRoutePoints(activity)
    return Math.max(max, route[route.length - 1]?.d ?? 0, comparisonSwimDistanceKm(activity))
  }, 0)

const comparisonRouteCapable = (activity: StravaActivityDetail): boolean => {
  const route = comparisonActivityRoutePoints(activity)
  return route.length >= 2 && route[route.length - 1].d > route[0].d
}

const splitComparisonGeographicPoints = (
  points: readonly ActivityComparisonGeographicPoint[],
): ActivityComparisonGeographicPoint[][] => {
  const segments: ActivityComparisonGeographicPoint[][] = []
  let segment: ActivityComparisonGeographicPoint[] = []
  const flush = (): void => {
    if (segment.length >= 2) segments.push(segment)
    segment = []
  }
  for (const point of points) {
    if (!Number.isFinite(point.lat) || !Number.isFinite(point.lng) || !Number.isFinite(point.d)) {
      flush()
      continue
    }
    if (segment.length > 0 && point.d < segment[segment.length - 1].d) flush()
    segment.push({ lat: point.lat, lng: point.lng, d: point.d })
  }
  flush()
  return segments
}

const comparisonGeographicSegments = (
  activity: StravaActivityDetail,
): ActivityComparisonGeographicPoint[][] => {
  const mapSegments = activity.mapRoute.flatMap(segment => splitComparisonGeographicPoints(segment))
  if (mapSegments.length > 0) return mapSegments
  return splitComparisonGeographicPoints(activity.route)
}

export const activityComparisonEligible = (activity: StravaActivityDetail): boolean => {
  const swimTelemetry =
    activity.sport === 'swim' &&
    [
      comparisonSwimMetricSegments(activity, 'swim-pace'),
      comparisonSwimMetricSegments(activity, 'stroke-rate'),
    ]
      .flat()
      .some(segment => segment.length >= 2)
  return (
    swimTelemetry ||
    (comparisonRouteCapable(activity) && comparisonGeographicSegments(activity).length > 0)
  )
}

export const normalizePowerCurvePoints = (
  curve: readonly PowerCurvePoint[] | null,
): PowerCurvePoint[] => {
  if (!curve) return []
  const sorted = curve
    .filter(
      point => Number.isFinite(point.s) && point.s > 0 && Number.isFinite(point.w) && point.w >= 0,
    )
    .map(point => ({
      s: point.s,
      w: point.w,
      ...(point.activityId == null ? {} : { activityId: point.activityId }),
      ...(point.activityDate == null ? {} : { activityDate: point.activityDate }),
    }))
    .sort((a, b) => a.s - b.s)
  const points: PowerCurvePoint[] = []
  for (const point of sorted) {
    if (points[points.length - 1]?.s === point.s) continue
    points.push(point)
  }
  return points
}

export const nearestPowerCurvePoint = (
  points: readonly PowerCurvePoint[],
  durationS: number,
): PowerCurvePoint | null => {
  if (
    points.length === 0 ||
    !Number.isFinite(durationS) ||
    durationS < points[0].s ||
    durationS > points[points.length - 1].s
  )
    return null
  return points[nearestPowerCurveIndex(points, durationS)]
}

export const nearestPowerCurveValue = (
  points: readonly PowerCurvePoint[],
  durationS: number,
): number | null => nearestPowerCurvePoint(points, durationS)?.w ?? null

export const activityZonePercentages = (values: readonly number[] | null | undefined): number[] => {
  if (!values) return []
  if (values.some(value => !Number.isFinite(value) || value < 0)) return []
  const total = values.reduce((sum, value) => sum + value, 0)
  return total > 0 ? values.map(value => (value / total) * 100) : values.map(() => 0)
}

export const activityPowerDistributionPercentages = (
  values: readonly number[] | null | undefined,
): number[] => (values && values.length >= 2 ? activityZonePercentages(values) : [])

const comparisonMetricSeries = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
  index: number,
  spec: ActivityComparisonMetricSpec,
): ActivityComparisonSeries => {
  const segments: ActivityComparisonMetricPoint[][] = []
  const values: number[] = []
  for (const rawSegment of comparisonMetricSegments(presentation, activity, spec.metric)) {
    const segment = rawSegment
      .map(point => ({ distanceKm: point.distanceKm, value: spec.display(point.value) }))
      .filter(point => Number.isFinite(point.value))
    if (segment.length < 2) continue
    segments.push(segment)
    values.push(...segment.map(point => point.value))
  }
  return { activity, index, segments, values }
}

const comparisonNumericDomain = (
  values: number[],
  includeZero: boolean,
): ActivityComparisonDomain => {
  if (values.length === 0) return { min: 0, max: 1 }
  let observedMin = values[0]
  let observedMax = values[0]
  for (let index = 1; index < values.length; index++) {
    observedMin = Math.min(observedMin, values[index])
    observedMax = Math.max(observedMax, values[index])
  }
  const lower = includeZero ? Math.min(0, observedMin) : observedMin
  const span = observedMax - lower
  const step = niceStep(span > 0 ? span : Math.max(1, Math.abs(observedMax)), 3)
  const min = Math.floor(lower / step) * step
  let max = Math.ceil(observedMax / step) * step
  if (max <= min) max = min + step
  return { min, max }
}

const comparisonSeriesPath = (
  segments: { distanceKm: number; value: number }[][],
  maxDistanceKm: number,
  domain: ActivityComparisonDomain,
): string => {
  const distanceSpan = Math.max(1e-6, maxDistanceKm)
  const valueSpan = Math.max(1e-6, domain.max - domain.min)
  return segments
    .map(segment =>
      segment
        .map((point, index) => {
          const x =
            Math.min(1, Math.max(0, point.distanceKm / distanceSpan)) * ACTIVITY_COMPARISON_WIDTH
          const y =
            ACTIVITY_COMPARISON_HEIGHT -
            ((point.value - domain.min) / valueSpan) * (ACTIVITY_COMPARISON_HEIGHT - 1)
          return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
        })
        .join(' '),
    )
    .filter(path => path.length > 0)
    .join(' ')
}

const comparisonMapReadout = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const readout = f.el('div', 'tri-compare-readout tri-compare-map-readout', undefined, {
    'data-compare-readout': '',
    'data-visible': 'false',
    'aria-hidden': 'true',
  })
  for (const [index, activity] of activities.entries()) {
    const row = f.el('div', 'tri-compare-readout-row', undefined, {
      'data-activity-id': `${activity.id}`,
      'data-activity-index': `${index}`,
      style: `--tri-compare-color:${activityCompareColor(index)}`,
    })
    f.add(
      row,
      f.el('span', 'tri-compare-readout-swatch', undefined, { 'aria-hidden': 'true' }),
      f.el('strong', 'tri-compare-readout-value', undefined, { 'data-compare-readout-value': '' }),
    )
    f.add(readout, row)
  }
  return readout
}

const comparisonChartHead = <N>(
  f: TriNodeFactory<N>,
  title: string,
  available: number,
  selected: number,
  controls?: N,
): N => {
  const head = f.el('div', 'tri-compare-chart-head')
  const coverage = f.el('span', 'tri-compare-coverage', undefined, {
    'data-available': `${available}`,
    'data-selected': `${selected}`,
  })
  f.add(
    coverage,
    f.el('span', 'tri-compare-coverage-count', `${available}/${selected} · `),
    f.el('span', 'tri-compare-coverage-label', 'sensor coverage', {
      'data-i18n': 'sensor coverage',
    }),
  )
  f.add(
    head,
    f.el('div', 'tri-compare-title', title, { 'data-i18n': title }),
    ...(controls === undefined ? [] : [controls]),
    coverage,
  )
  return head
}

const comparisonGraphAttrs = (
  chart: string,
  label: string,
  valueMin: number,
  valueMax: number,
  valueNow: number,
  valueText: string,
  available: number,
): Record<string, string | number> => {
  const attrs: Record<string, string | number> = {
    class: 'tri-compare-graph',
    viewBox: `0 0 ${ACTIVITY_COMPARISON_WIDTH} ${ACTIVITY_COMPARISON_HEIGHT}`,
    preserveAspectRatio: 'none',
    'data-compare-chart': chart,
    'aria-label': label,
    'data-i18n-aria-label': label,
  }
  if (available <= 0) {
    attrs.role = 'img'
    attrs['aria-disabled'] = 'true'
    return attrs
  }
  attrs.role = 'slider'
  attrs.tabindex = 0
  attrs['aria-orientation'] = 'horizontal'
  attrs['aria-valuemin'] = valueMin
  attrs['aria-valuemax'] = valueMax
  attrs['aria-valuenow'] = valueNow
  attrs['aria-valuetext'] = valueText
  return attrs
}

const addComparisonSelectionRegion = <N>(f: TriNodeFactory<N>, graph: N, clipId: string): void => {
  const clip = f.svg('clipPath', { id: clipId })
  f.add(
    clip,
    f.svg('rect', {
      class: 'tri-compare-selection-clip',
      x: 0,
      y: 0,
      width: 0,
      height: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const defs = f.svg('defs', {})
  f.add(defs, clip)
  f.add(
    graph,
    defs,
    f.svg('rect', {
      class: 'tri-compare-selection-region',
      x: 0,
      y: 0,
      width: 0,
      height: ACTIVITY_COMPARISON_HEIGHT,
      'aria-hidden': 'true',
    }),
  )
}

const buildComparisonMetricChart = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  spec: ActivityComparisonMetricSpec,
  maxDistanceKm: number,
): N => {
  const series = activities.map((activity, index) =>
    comparisonMetricSeries(f.presentation, activity, index, spec),
  )
  const available = series.filter(item => item.values.length > 0).length
  const values = series.flatMap(item => item.values)
  const domain = spec.domain
    ? spec.domain(values)
    : comparisonNumericDomain(
        values,
        spec.metric === 'power' && excludesZeroPower(f.presentation) ? false : spec.includeZero,
      )
  const selectionClipId = `tri-compare-${spec.metric}-selection-clip`
  const sport = activities[0]?.sport ?? 'bike'
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      spec.metric,
      spec.title,
      0,
      maxDistanceKm,
      0,
      scrubDist(f.presentation, 0, sport),
      available,
    ),
    class: 'tri-compare-graph tri-compare-distance-graph',
    'data-domain-x-min': 0,
    'data-domain-x-max': maxDistanceKm,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-available': available,
    'data-selected': activities.length,
  })
  addComparisonSelectionRegion(f, graph, selectionClipId)
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: value === 0 ? '0' : spec.tick(value, sport),
    vbY:
      ACTIVITY_COMPARISON_HEIGHT -
      ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  for (const item of series) {
    if (item.values.length === 0) continue
    const path = comparisonSeriesPath(item.segments, maxDistanceKm, domain)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
      }),
      f.svg('path', {
        class: 'tri-compare-selection-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
        'clip-path': `url(#${selectionClipId})`,
        'aria-hidden': 'true',
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': spec.metric,
    'data-available': `${available}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, spec.title, available, activities.length),
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      sport === 'swim'
        ? swimActivityXTicks(maxDistanceKm * 1_000)
        : distanceXTicks(f.presentation, 0, maxDistanceKm),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  return chart
}

const comparisonDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  selectedSeconds: number,
): AxisXTick[] => {
  const durations = powerCurveDurationTicks(minSeconds, maxSeconds, [
    minSeconds,
    1,
    60,
    300,
    1_200,
    3_600,
    10_800,
  ])
  return durations.map((seconds, index) => ({
    label: dlabel(seconds),
    pct: powerCurveFraction(seconds, minSeconds, maxSeconds) * 100,
    cls: `tri-curve-tick${index === 0 ? ' tri-cax-xt--first' : index === durations.length - 1 ? ' tri-cax-xt--last' : ''}`,
    tag: 'button',
    attrs: {
      type: 'button',
      'data-curve-seconds': String(seconds),
      'aria-pressed': String(seconds === selectedSeconds),
    },
  }))
}

const buildComparisonPowerCurve = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  ctx?: DetailCtx,
): N => {
  const isBike = activities.length > 0 && activities.every(activity => activity.sport === 'bike')
  const curves = activities.map((activity, index) => ({
    activity,
    index,
    points: normalizePowerCurvePoints(activity.powerCurve),
  }))
  const availableCurves = curves.filter(curve => curve.points.length >= 2)
  let minSeconds = 1
  let maxSeconds = 2
  if (availableCurves.length > 0) {
    minSeconds = availableCurves[0].points[0].s
    maxSeconds = availableCurves[0].points[availableCurves[0].points.length - 1].s
    for (let index = 1; index < availableCurves.length; index++) {
      const points = availableCurves[index].points
      minSeconds = Math.min(minSeconds, points[0].s)
      maxSeconds = Math.max(maxSeconds, points[points.length - 1].s)
    }
  }
  const sixWeekReference =
    isBike && availableCurves.length > 0
      ? normalizePowerCurvePoints(ctx?.curveRef ?? null).filter(
          point => point.s >= minSeconds && point.s <= maxSeconds,
        )
      : []
  const yearReference =
    isBike && availableCurves.length > 0
      ? normalizePowerCurvePoints(ctx?.curveYearRef ?? null).filter(
          point => point.s >= minSeconds && point.s <= maxSeconds,
        )
      : []
  const sixWeekModel =
    isBike && availableCurves.length > 0 && ctx?.criticalPower
      ? criticalPowerCurve(ctx.criticalPower, minSeconds, maxSeconds)
      : []
  const yearModel =
    isBike && availableCurves.length > 0 && ctx?.criticalPowerYear
      ? criticalPowerCurve(ctx.criticalPowerYear, minSeconds, maxSeconds)
      : []
  const defaultRange = sixWeekReference.length >= 2 ? 'six-weeks' : 'year'
  const reference = defaultRange === 'six-weeks' ? sixWeekReference : yearReference
  const ftp = isBike && availableCurves.length > 0 ? (ctx?.ftp ?? null) : null
  const goalFtp = isBike && availableCurves.length > 0 ? (ctx?.goalFtp ?? null) : null
  const domain = comparisonNumericDomain(
    [
      ...availableCurves.flatMap(curve => curve.points.map(point => point.w)),
      ...sixWeekReference.map(point => point.w),
      ...yearReference.map(point => point.w),
      ...sixWeekModel.map(point => point.w),
      ...yearModel.map(point => point.w),
      ...(!isBike || ctx?.criticalPower == null ? [] : [ctx.criticalPower.criticalPowerWatts]),
      ...(!isBike || ctx?.criticalPowerYear == null
        ? []
        : [ctx.criticalPowerYear.criticalPowerWatts]),
      ...(ftp == null ? [] : [ftp]),
      ...(goalFtp == null ? [] : [goalFtp]),
    ],
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'power-curve',
      'power curve',
      minSeconds,
      maxSeconds,
      minSeconds,
      dlabel(minSeconds),
      availableCurves.length,
    ),
    class: 'tri-compare-graph tri-compare-curve-graph',
    'data-compare-chart': 'power-curve',
    'data-domain-x-scale': 'log',
    'data-domain-x-min': minSeconds,
    'data-domain-x-max': maxSeconds,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-available': availableCurves.length,
    'data-selected': activities.length,
    'data-curve-ref-six-weeks': encodePowerCurve(sixWeekReference),
    'data-curve-ref-year': encodePowerCurve(yearReference),
    'data-curve-range': defaultRange,
    'data-curve-year': ctx?.curveYear ?? '',
  })
  const X = (seconds: number): number =>
    powerCurveFraction(seconds, minSeconds, maxSeconds) * ACTIVITY_COMPARISON_WIDTH
  const Y = (watts: number): number =>
    ACTIVITY_COMPARISON_HEIGHT -
    ((watts - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
  const toPath = (points: PowerCurvePoint[]): string =>
    powerCurvePathPoints(points)
      .map(
        (point, index) =>
          `${index === 0 ? 'M' : 'L'} ${X(point.s).toFixed(2)} ${Y(point.w).toFixed(2)}`,
      )
      .join(' ')
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: value === 0 ? '0' : `${Math.round(value)} W`,
    vbY: Y(value),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  if (sixWeekReference.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-ref',
        d: toPath(sixWeekReference),
        'data-curve-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (yearReference.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-ref',
        d: toPath(yearReference),
        'data-curve-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  if (sixWeekModel.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-model',
        d: toPath(sixWeekModel),
        'data-critical-power-range': 'six-weeks',
        ...(defaultRange === 'six-weeks' ? {} : { hidden: '' }),
      }),
    )
  if (yearModel.length >= 2)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-curve-model',
        d: toPath(yearModel),
        'data-critical-power-range': 'year',
        ...(defaultRange === 'year' ? {} : { hidden: '' }),
      }),
    )
  for (const [range, estimate] of [
    ['six-weeks', isBike ? (ctx?.criticalPower ?? null) : null],
    ['year', isBike ? (ctx?.criticalPowerYear ?? null) : null],
  ] as const) {
    if (!estimate) continue
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-cp',
        x1: 0,
        y1: Y(estimate.criticalPowerWatts).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(estimate.criticalPowerWatts).toFixed(2),
        'data-critical-power-range': range,
        ...(defaultRange === range ? {} : { hidden: '' }),
      }),
    )
  }
  if (ftp != null)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-ftp',
        x1: 0,
        y1: Y(ftp).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(ftp).toFixed(2),
      }),
    )
  if (goalFtp != null)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-curve-goal',
        x1: 0,
        y1: Y(goalFtp).toFixed(2),
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: Y(goalFtp).toFixed(2),
      }),
    )
  for (const curve of availableCurves) {
    const path = toPath(curve.points)
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': curve.activity.id,
        'data-activity-index': curve.index,
        style: `--tri-compare-color:${activityCompareColor(curve.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': 'power-curve',
    'data-available': `${availableCurves.length}`,
    'data-selected': `${activities.length}`,
  })
  const ranges = buildPowerCurveRanges(
    f,
    defaultRange,
    ctx?.curveYear ?? null,
    sixWeekReference.length >= 2,
    yearReference.length >= 2,
  )
  f.add(
    chart,
    comparisonChartHead(
      f,
      'power curve',
      availableCurves.length,
      activities.length,
      ranges ?? undefined,
    ),
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      comparisonDurationTicks(minSeconds, maxSeconds, minSeconds),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  if (
    sixWeekReference.length >= 2 ||
    yearReference.length >= 2 ||
    (isBike && ctx?.criticalPower != null) ||
    (isBike && ctx?.criticalPowerYear != null) ||
    ftp != null ||
    goalFtp != null
  ) {
    const cap = f.el('div', 'tri-elev-cap')
    if (reference.length >= 2)
      f.add(
        cap,
        f.el(
          'span',
          'tri-ana-k tri-compare-curve-reference-label',
          defaultRange === 'year' && ctx?.curveYear != null
            ? `${ctx.curveYear} best`
            : '6-week best',
          defaultRange === 'six-weeks' ? { 'data-i18n': '6-week best' } : undefined,
        ),
      )
    addPowerCurveThresholdCaption(
      f,
      cap,
      [
        ['six-weeks', isBike ? (ctx?.criticalPower ?? null) : null],
        ['year', isBike ? (ctx?.criticalPowerYear ?? null) : null],
      ],
      defaultRange,
      ftp,
      goalFtp,
    )
    f.add(chart, cap)
  }
  return chart
}

const comparisonPowerDistributionTicks = (binCount: number): AxisXTick[] => {
  const maxWatts = Math.max(25, (binCount - 1) * 25)
  const stepWatts = maxWatts <= 300 ? 100 : maxWatts <= 700 ? 200 : 300
  const ticks: AxisXTick[] = []
  for (let watts = 0; watts <= maxWatts; watts += stepWatts)
    ticks.push({
      label: `${watts} W`,
      pct: (watts / maxWatts) * 100,
      cls: watts === 0 ? 'tri-cax-xt--first' : watts === maxWatts ? 'tri-cax-xt--last' : undefined,
    })
  return ticks
}

const buildComparisonPowerDistribution = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const distributions = activities.map((activity, index) => ({
    activity,
    index,
    percentages: activityPowerDistributionPercentages(activity.powerHist),
  }))
  const available = distributions.filter(distribution =>
    distribution.percentages.some(value => value > 0),
  )
  const binCount = Math.max(
    2,
    ...distributions.map(distribution => distribution.percentages.length),
  )
  const maxWatts = (binCount - 1) * 25
  const domain = comparisonNumericDomain(
    available.flatMap(distribution => distribution.percentages),
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'power-distribution',
      '25W power distribution',
      0,
      maxWatts,
      0,
      '0–24 W',
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-distribution-graph',
    'data-compare-chart': 'power-distribution',
    'data-domain-x-min': 0,
    'data-domain-x-max': maxWatts,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-bin-count': binCount,
    'data-bin-width-watts': 25,
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: `${Math.round(value)}%`,
    vbY:
      ACTIVITY_COMPARISON_HEIGHT -
      ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  for (const distribution of available) {
    const path = Array.from(
      { length: binCount },
      (_, index) => distribution.percentages[index] ?? 0,
    )
      .map((value, index) => {
        const x = (index / (binCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT -
          ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': distribution.activity.id,
        'data-activity-index': distribution.index,
        style: `--tri-compare-color:${activityCompareColor(distribution.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': 'power-distribution',
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, '25W power distribution', available.length, activities.length),
    axisFrame(
      f,
      graph,
      yTicks,
      ACTIVITY_COMPARISON_HEIGHT,
      comparisonPowerDistributionTicks(binCount),
      true,
      { top: 0, bottom: ACTIVITY_COMPARISON_HEIGHT },
    ),
  )
  return chart
}

const buildComparisonGearRatioDistribution = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
): N => {
  const distributions = activities.map((activity, index) => ({
    activity,
    index,
    points: activityGearRatioDistribution(activity),
  }))
  const available = distributions.filter(distribution => distribution.points.length > 0)
  const ratios = [
    ...new Set(
      distributions.flatMap(distribution => distribution.points.map(point => point.ratio)),
    ),
  ].sort((left, right) => left - right)
  const ratioCount = ratios.length
  const domain = comparisonNumericDomain(
    available.flatMap(distribution => distribution.points.map(point => point.percentage)),
    true,
  )
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      'gear-ratio-distribution',
      'gear ratio distribution',
      0,
      Math.max(1, ratioCount - 1),
      0,
      ratios[0] == null ? 'no data' : `${ratios[0].toFixed(2)}×`,
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-distribution-graph',
    'data-compare-chart': 'gear-ratio-distribution',
    'data-domain-x-min': 0,
    'data-domain-x-max': Math.max(1, ratioCount - 1),
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-ratio-count': ratioCount,
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = niceTicks(domain.min, domain.max, 3).map(value => ({
    label: `${Math.round(value)}%`,
    vbY:
      ACTIVITY_COMPARISON_HEIGHT -
      ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  for (const distribution of available) {
    const percentages = new Map(
      distribution.points.map(point => [point.ratio, point.percentage] as const),
    )
    const path = ratios
      .map((ratio, index) => {
        const value = percentages.get(ratio) ?? 0
        const x =
          ratioCount <= 1
            ? ACTIVITY_COMPARISON_WIDTH / 2
            : (index / (ratioCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT -
          ((value - domain.min) / (domain.max - domain.min)) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': distribution.activity.id,
        'data-activity-index': distribution.index,
        style: `--tri-compare-color:${activityCompareColor(distribution.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: ratioCount <= 1 ? ACTIVITY_COMPARISON_WIDTH / 2 : 0,
      y1: 0,
      x2: ratioCount <= 1 ? ACTIVITY_COMPARISON_WIDTH / 2 : 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const sampledRatios = sampledGearTicks(ratios, 6)
  const xTicks = sampledRatios.map(ratio => {
    const index = ratios.indexOf(ratio)
    return {
      label: `${ratio.toFixed(2)}×`,
      pct: ratioCount <= 1 ? 50 : (index / (ratioCount - 1)) * 100,
      cls:
        index === 0
          ? 'tri-cax-xt--first'
          : index === ratioCount - 1
            ? 'tri-cax-xt--last'
            : undefined,
    }
  })
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': 'gear-ratio-distribution',
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(f, 'gear ratio distribution', available.length, activities.length),
    axisFrame(f, graph, yTicks, ACTIVITY_COMPARISON_HEIGHT, xTicks, true, {
      top: 0,
      bottom: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  return chart
}

const buildComparisonZones = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  kind: 'hr-zones' | 'power-zones',
): N => {
  const values = activities.map((activity, index) => ({
    activity,
    index,
    percentages: activityZonePercentages(
      kind === 'hr-zones' ? activity.hrZones : activity.powerZones,
    ),
  }))
  const available = values.filter(item => item.percentages.some(value => value > 0))
  const defaultZoneCount = kind === 'hr-zones' ? 5 : 7
  const observedZoneCount = values.reduce((max, item) => Math.max(max, item.percentages.length), 0)
  const zoneCount = observedZoneCount > 0 ? observedZoneCount : defaultZoneCount
  const domain = { min: 0, max: 100 }
  const graph = f.svg('svg', {
    ...comparisonGraphAttrs(
      kind,
      kind === 'hr-zones' ? 'heart rate zones' : 'power zones',
      1,
      zoneCount,
      1,
      'Z1',
      available.length,
    ),
    class: 'tri-compare-graph tri-compare-zone-graph',
    'data-domain-x-min': 1,
    'data-domain-x-max': zoneCount,
    'data-domain-y-min': domain.min,
    'data-domain-y-max': domain.max,
    'data-zone-unit': 'percent',
    'data-available': available.length,
    'data-selected': activities.length,
  })
  const yTicks = [0, 25, 50, 75, 100].map(value => ({
    label: `${value}%`,
    vbY: ACTIVITY_COMPARISON_HEIGHT - (value / domain.max) * (ACTIVITY_COMPARISON_HEIGHT - 1),
  }))
  for (const tick of yTicks)
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-compare-grid',
        x1: 0,
        y1: tick.vbY,
        x2: ACTIVITY_COMPARISON_WIDTH,
        y2: tick.vbY,
      }),
    )
  for (const item of available) {
    const path = item.percentages
      .map((value, index) => {
        const x =
          zoneCount === 1
            ? ACTIVITY_COMPARISON_WIDTH / 2
            : (index / (zoneCount - 1)) * ACTIVITY_COMPARISON_WIDTH
        const y =
          ACTIVITY_COMPARISON_HEIGHT - (value / domain.max) * (ACTIVITY_COMPARISON_HEIGHT - 1)
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
      })
      .join(' ')
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-compare-line',
        d: path,
        'data-activity-id': item.activity.id,
        'data-activity-index': item.index,
        style: `--tri-compare-color:${activityCompareColor(item.index)}`,
      }),
    )
  }
  f.add(
    graph,
    f.svg('line', {
      class: 'tri-compare-cursor',
      x1: 0,
      y1: 0,
      x2: 0,
      y2: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  const xTicks = Array.from({ length: zoneCount }, (_, index) => ({
    label: `Z${index + 1}`,
    pct: zoneCount === 1 ? 50 : (index / (zoneCount - 1)) * 100,
    cls:
      index === 0 ? 'tri-cax-xt--first' : index === zoneCount - 1 ? 'tri-cax-xt--last' : undefined,
  }))
  const chart = f.el('section', 'tri-compare-chart', undefined, {
    'data-compare-chart': kind,
    'data-available': `${available.length}`,
    'data-selected': `${activities.length}`,
  })
  f.add(
    chart,
    comparisonChartHead(
      f,
      kind === 'hr-zones' ? 'heart rate zones' : 'power zones',
      available.length,
      activities.length,
    ),
    axisFrame(f, graph, yTicks, ACTIVITY_COMPARISON_HEIGHT, xTicks, true, {
      top: 0,
      bottom: ACTIVITY_COMPARISON_HEIGHT,
    }),
  )
  return chart
}

const buildComparisonLegend = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  removable: boolean,
): N => {
  const legend = f.el(
    'div',
    `tri-compare-legend${removable ? '' : ' tri-compare-legend--static'}`,
    undefined,
    {
      role: 'list',
      'aria-label': 'selected activities',
      'data-i18n-aria-label': 'selected activities',
    },
  )
  for (const [index, activity] of activities.entries()) {
    const item = f.el('div', 'tri-compare-legend-item', undefined, {
      role: 'listitem',
      'data-activity-id': `${activity.id}`,
      'data-activity-index': `${index}`,
      style: `--tri-compare-color:${activityCompareColor(index)}`,
    })
    const toggle = f.el('button', 'tri-compare-legend-toggle', undefined, {
      type: 'button',
      'data-compare-activity-toggle': `${activity.id}`,
      'aria-pressed': 'true',
      'aria-label': 'toggle activity',
      'data-i18n-aria-label': 'toggle activity',
      title: activity.name,
    })
    f.add(
      toggle,
      f.el('span', 'tri-compare-legend-swatch', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-compare-legend-date', shortDate(activity.date)),
      f.el('span', 'tri-compare-legend-name', activity.name, { 'data-site-cursor-bracket': '' }),
    )
    f.add(item, toggle)
    if (removable) {
      const removeAttrs: Record<string, string> = {
        type: 'button',
        'data-compare-activity-remove': `${activity.id}`,
        'aria-label': 'remove activity',
        'data-i18n-aria-label': 'remove activity',
      }
      if (activities.length <= 2) removeAttrs.disabled = ''
      const remove = f.el('button', 'tri-compare-legend-remove', undefined, removeAttrs)
      const removeIcon = f.svg('svg', {
        class: 'tri-compare-legend-remove-icon',
        viewBox: '0 0 16 16',
        fill: 'none',
        'aria-hidden': 'true',
      })
      f.add(
        removeIcon,
        f.svg('path', {
          d: 'M4 4l8 8M12 4 4 12',
          stroke: 'currentColor',
          'stroke-width': 1.25,
          'stroke-linecap': 'round',
        }),
      )
      f.add(remove, removeIcon)
      f.add(item, remove)
    }
    f.add(legend, item)
  }
  return legend
}

const buildComparisonMap = <N>(
  f: TriNodeFactory<N>,
  activities: readonly StravaActivityDetail[],
  available: number,
): N => {
  const panel = f.el('section', 'tri-compare-map-panel', undefined, {
    'data-compare-chart': 'route',
    'data-available': `${available}`,
    'data-selected': `${activities.length}`,
    'aria-label': 'route overlay',
    'data-i18n-aria-label': 'route overlay',
  })
  const stage = f.el('div', 'tri-compare-map-stage')
  const canvas = f.el('div', 'tri-compare-map', undefined, {
    'data-compare-map': '',
    'data-available': `${available}`,
    'data-domain-x-max': `${comparisonMaxDistanceKm(activities)}`,
  })
  f.add(stage, canvas, comparisonMapReadout(f, activities))
  f.add(panel, comparisonChartHead(f, 'route overlay', available, activities.length), stage)
  return panel
}

const comparisonState = (
  activities: readonly StravaActivityDetail[],
): 'empty' | 'insufficient' | 'mixed-sport' | 'route-unavailable' | 'ready' => {
  if (activities.length === 0) return 'empty'
  if (activities.length < 2) return 'insufficient'
  const sport = activities[0].sport
  if (activities.some(activity => activity.sport !== sport)) return 'mixed-sport'
  if (activities.some(activity => !activityComparisonEligible(activity))) return 'route-unavailable'
  return 'ready'
}

const comparisonStateText = (
  state: 'empty' | 'insufficient' | 'mixed-sport' | 'route-unavailable',
): string => {
  if (state === 'mixed-sport') return 'Choose activities from one exact sport.'
  if (state === 'route-unavailable') return 'Route telemetry is unavailable for this comparison.'
  return 'Choose at least two activities.'
}

export const buildActivityComparison = <N>(
  f: TriNodeFactory<N>,
  activities: StravaActivityDetail[],
  ctx?: DetailCtx,
  options: { removable?: boolean } = {},
): N => {
  const state = comparisonState(activities)
  const rootAttrs: Record<string, string> = {
    'data-compare-state': state,
    'data-selected': `${activities.length}`,
  }
  if (activities.length > 0 && activities.every(activity => activity.sport === activities[0].sport))
    rootAttrs['data-sport'] = activities[0].sport
  const root = f.el('section', `tri-compare tri-compare--${state}`, undefined, rootAttrs)
  f.add(root, buildComparisonLegend(f, activities, options.removable !== false))
  if (state !== 'ready') {
    f.add(
      root,
      f.el('p', 'tri-compare-empty', comparisonStateText(state), { 'data-compare-empty': state }),
    )
    return root
  }
  const maxDistanceKm = Math.max(comparisonMaxDistanceKm(activities), 1e-6)
  const body = f.el('div', 'tri-compare-body')
  const routed = activities.filter(
    activity => comparisonGeographicSegments(activity).length > 0,
  ).length
  if (routed > 0) f.add(body, buildComparisonMap(f, activities, routed))
  const chartViewport = f.el('div', 'tri-compare-charts-viewport')
  const charts = f.el('div', 'tri-compare-charts', undefined, { 'data-keyboard-scroll': '' })
  const sport = activities[0].sport
  const metricSpecs = activityComparisonMetricSpecs(f.presentation)
  for (const metric of activityComparisonMetricsForSport(sport))
    f.add(charts, buildComparisonMetricChart(f, activities, metricSpecs[metric], maxDistanceKm))
  if (sport === 'bike') f.add(charts, buildComparisonGearRatioDistribution(f, activities))
  if (sport !== 'swim')
    f.add(
      charts,
      buildComparisonPowerDistribution(f, activities),
      buildComparisonPowerCurve(f, activities, ctx),
    )
  f.add(charts, buildComparisonZones(f, activities, 'hr-zones'))
  if (sport !== 'swim') f.add(charts, buildComparisonZones(f, activities, 'power-zones'))
  f.add(chartViewport, charts)
  f.add(body, chartViewport)
  f.add(root, body)
  return root
}

export const dayDetails = (payload: DayCardPayload, dateIso: string): StravaActivityDetail[] =>
  Object.values(payload.details)
    .filter(d => d.date === dateIso)
    .sort((a, b) => b.distanceKm - a.distanceKm)

export const recentLocation = (payload: DayCardPayload): string | undefined =>
  Object.values(payload.details)
    .sort((a, b) => b.date.localeCompare(a.date))
    .find(d => d.location)?.location ?? undefined

const timelineActivityValue = (
  presentation: TriathlonPresentation,
  activity: StravaActivityDetail,
): string =>
  activity.distanceKm > 0
    ? dist(presentation, activity.distanceKm, activity.sport)
    : dur(activity.movingTimeS)

export const buildTimelineDayCard = <N>(
  f: TriNodeFactory<N>,
  dateIso: string,
  payload: DayCardPayload | null,
): N => {
  const card = f.el('div', 'tri-timeline-card')
  f.add(card, f.el('span', 'tri-pop-date', prettyDate(dateIso)))
  if (!payload) {
    f.add(card, f.el('div', 'tri-pop-rest', '·'))
    return card
  }
  const day = dayDetails(payload, dateIso)
  if (day.length === 0) {
    const label = triText(f.presentation.locale, 'rest')
    const rest = f.el('span', 'tri-timeline-row tri-timeline-rest', undefined, {
      role: 'group',
      'aria-label': label,
    })
    f.add(rest, buildBattery(f), f.el('span', 'tri-timeline-value', label, { 'data-i18n': 'rest' }))
    f.add(card, rest)
    return card
  }

  const list = f.el('ul', 'tri-timeline-list')
  for (const activity of day) {
    const value = timelineActivityValue(f.presentation, activity)
    const item = f.el('li', 'tri-timeline-item')
    const entry = f.el('span', 'tri-timeline-row tri-timeline-activity', undefined, {
      role: 'group',
      'aria-label': `${triText(f.presentation.locale, activity.sport)} · ${activity.name} · ${value}`,
      title: activity.name,
    })
    f.add(entry, buildIcon(f, activity.sport), f.el('span', 'tri-timeline-value', value))
    f.add(item, entry)
    f.add(list, item)
  }
  f.add(card, list)
  return card
}

type DayAnalyticsMetric = { label: string; value: string; detail?: string }

const dayAnalyticsNumber = (
  presentation: TriathlonPresentation,
  value: number,
  digits = 0,
): string =>
  value.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })

const dayAnalyticsSigned = (
  presentation: TriathlonPresentation,
  value: number,
  digits = 1,
): string =>
  `${value > 0 ? '+' : ''}${dayAnalyticsNumber(presentation, value === 0 ? 0 : value, digits)}`

const dayAnalyticsDuration = (seconds: number): string => {
  const total = Math.max(0, Math.round(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const remaining = total % 60
  const parts: string[] = []
  if (hours > 0) parts.push(`${hours}h`)
  if (minutes > 0 || hours > 0) parts.push(`${minutes}m`)
  if (remaining > 0 || parts.length === 0) parts.push(`${remaining}s`)
  return parts.join(' ')
}

const dayAnalyticsClock = (presentation: TriathlonPresentation, value: string): string => {
  const match = /T(\d{2}):(\d{2})/.exec(value)
  if (!match) return value
  const hour = Number(match[1])
  const minute = match[2]
  if (presentation.locale === 'fr') return `${hour.toString().padStart(2, '0')}:${minute}`
  const suffix = hour < 12 ? 'am' : 'pm'
  const displayHour = hour % 12 || 12
  return `${displayHour}:${minute} ${suffix}`
}

const dayAnalyticsMass = (presentation: TriathlonPresentation, kilograms: number): string => {
  const imperial = isImperial(presentation)
  const value = imperial ? kilograms / LB_TO_KG : kilograms
  return `${dayAnalyticsNumber(presentation, value, 1)} ${imperial ? 'lb' : 'kg'}`
}

const dayAnalyticsTemperature = (
  presentation: TriathlonPresentation,
  celsius: number,
  signed = false,
): string => {
  const value = isImperial(presentation) ? (celsius * 9) / 5 + (signed ? 0 : 32) : celsius
  return `${signed && value > 0 ? '+' : ''}${dayAnalyticsNumber(presentation, value, 1)}${isImperial(presentation) ? '°F' : '°C'}`
}

const dayAnalyticsList = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: string,
  metrics: DayAnalyticsMetric[],
  className = 'tri-day-analytics-list',
): N => {
  const list = f.el('dl', className)
  for (const [index, metric] of metrics.entries()) {
    const detailId = `tri-day-${date}-${key}-detail-${index}`
    const rowAttrs: Record<string, string> = { 'data-metric': metric.label }
    if (metric.detail) {
      rowAttrs.tabindex = '0'
      rowAttrs['aria-describedby'] = detailId
    }
    const row = f.el('div', 'tri-day-analytics-metric', undefined, rowAttrs)
    const value = f.el('dd', 'tri-day-analytics-value', metric.value)
    if (metric.detail)
      f.add(
        value,
        f.el('span', 'tri-day-analytics-detail', metric.detail, { id: detailId, role: 'tooltip' }),
      )
    f.add(
      row,
      f.el('dt', 'tri-day-analytics-label', triText(f.presentation.locale, metric.label), {
        'data-i18n': metric.label,
      }),
      value,
    )
    f.add(list, row)
  }
  return list
}

const dayAnalyticsGroup = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: string,
  label: string,
  metrics: DayAnalyticsMetric[],
): N | null => {
  if (metrics.length === 0) return null
  const titleId = `tri-day-${date}-${key}`
  const group = f.el(
    'section',
    `tri-day-analytics-group tri-day-analytics-group--${key}`,
    undefined,
    { 'aria-labelledby': titleId },
  )
  f.add(
    group,
    f.el(
      'h3',
      'tri-ana-block-title tri-day-analytics-group-title',
      triText(f.presentation.locale, label),
      { id: titleId, 'data-i18n': label },
    ),
    dayAnalyticsList(f, date, key, metrics),
  )
  return group
}

const dayAnalyticsBodyMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const body = summary.body
  if (!body) return []
  const metrics: DayAnalyticsMetric[] = []
  if (body.kg != null)
    metrics.push({ label: 'body weight', value: dayAnalyticsMass(presentation, body.kg) })
  if (body.ffmi != null)
    metrics.push({ label: 'FFMI', value: dayAnalyticsNumber(presentation, body.ffmi, 2) })
  if (body.bmi != null)
    metrics.push({ label: 'bmi', value: dayAnalyticsNumber(presentation, body.bmi, 1) })
  if (body.bodyFatPct != null)
    metrics.push({
      label: 'body fat',
      value: `${dayAnalyticsNumber(presentation, body.bodyFatPct, 1)}%`,
    })
  if (body.bodyWaterPct != null)
    metrics.push({
      label: 'body water',
      value: `${dayAnalyticsNumber(presentation, body.bodyWaterPct, 1)}%`,
    })
  if (body.muscleMassKg != null)
    metrics.push({ label: 'muscle', value: dayAnalyticsMass(presentation, body.muscleMassKg) })
  if (body.boneMassKg != null)
    metrics.push({ label: 'bone', value: dayAnalyticsMass(presentation, body.boneMassKg) })
  return metrics
}

const dayAnalyticsRecoveryMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const recovery = summary.recovery
  if (!recovery) return []
  const metrics: DayAnalyticsMetric[] = []
  if (recovery.readiness != null)
    metrics.push({
      label: 'readiness',
      value: dayAnalyticsNumber(presentation, recovery.readiness),
      ...(recovery.readinessBaseline != null
        ? {
            detail: `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.readinessBaseline)}`,
          }
        : {}),
    })
  if (recovery.hrv != null)
    metrics.push({
      label: 'hrv',
      value: `${dayAnalyticsNumber(presentation, recovery.hrv)} ms`,
      detail: [
        recovery.hrvBaseline == null
          ? null
          : `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.hrvBaseline, 1)}`,
        recovery.hrvZ == null ? null : `z ${dayAnalyticsSigned(presentation, recovery.hrvZ, 2)}σ`,
      ]
        .filter(value => value != null)
        .join(' · '),
    })
  if (recovery.rhr != null)
    metrics.push({
      label: 'resting hr',
      value: `${dayAnalyticsNumber(presentation, recovery.rhr)} bpm`,
      detail: [
        recovery.rhrBaseline == null
          ? null
          : `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.rhrBaseline, 1)}`,
        recovery.rhrZ == null ? null : `z ${dayAnalyticsSigned(presentation, recovery.rhrZ, 2)}σ`,
      ]
        .filter(value => value != null)
        .join(' · '),
    })
  if (recovery.temperatureDeviationC != null)
    metrics.push({
      label: 'temperature deviation',
      value: dayAnalyticsTemperature(presentation, recovery.temperatureDeviationC, true),
    })
  return metrics
}

const dayAnalyticsSleepMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const sleep = summary.sleep
  const recovery = summary.recovery
  if (!sleep && !recovery) return []
  const metrics: DayAnalyticsMetric[] = []
  const score = sleep?.sleepScore ?? null
  if (score != null)
    metrics.push({ label: 'sleep score', value: dayAnalyticsNumber(presentation, score) })
  if (sleep?.readinessScore != null)
    metrics.push({
      label: 'readiness',
      value: dayAnalyticsNumber(presentation, sleep.readinessScore),
    })
  if (sleep?.bedtimeStart)
    metrics.push({ label: 'bedtime', value: dayAnalyticsClock(presentation, sleep.bedtimeStart) })
  if (sleep?.bedtimeEnd)
    metrics.push({ label: 'wake-up', value: dayAnalyticsClock(presentation, sleep.bedtimeEnd) })
  if (sleep?.timeInBedS != null)
    metrics.push({ label: 'time in bed', value: dayAnalyticsDuration(sleep.timeInBedS) })
  const totalSleep = sleep?.totalSleepS ?? recovery?.sleepDurationS
  if (totalSleep != null)
    metrics.push({ label: 'total sleep', value: dayAnalyticsDuration(totalSleep) })
  if (sleep?.efficiency != null)
    metrics.push({
      label: 'efficiency',
      value: `${dayAnalyticsNumber(presentation, sleep.efficiency)}%`,
    })
  if (sleep?.latencyS != null)
    metrics.push({ label: 'latency', value: dayAnalyticsDuration(sleep.latencyS) })
  if (sleep?.averageHeartRate != null)
    metrics.push({
      label: 'average hr',
      value: `${dayAnalyticsNumber(presentation, sleep.averageHeartRate, 1)} bpm`,
    })
  if (sleep?.lowestHeartRate != null)
    metrics.push({
      label: 'lowest hr',
      value: `${dayAnalyticsNumber(presentation, sleep.lowestHeartRate)} bpm`,
    })
  if (sleep?.averageHrv != null)
    metrics.push({
      label: 'average hrv',
      value: `${dayAnalyticsNumber(presentation, sleep.averageHrv)} ms`,
    })
  if (sleep?.averageBreathsPerMinute != null)
    metrics.push({
      label: 'breath',
      value: `${dayAnalyticsNumber(presentation, sleep.averageBreathsPerMinute, 1)} brpm`,
    })
  if (sleep?.restlessPeriods != null)
    metrics.push({
      label: 'restless periods',
      value: dayAnalyticsNumber(presentation, sleep.restlessPeriods),
    })
  if (recovery?.sleepDebtS != null)
    metrics.push({ label: 'sleep debt', value: dayAnalyticsDuration(recovery.sleepDebtS) })
  if (recovery?.sleepBaselineS != null)
    metrics.push({ label: 'sleep baseline', value: dayAnalyticsDuration(recovery.sleepBaselineS) })
  if (recovery?.sleepTargetS != null)
    metrics.push({ label: 'sleep target', value: dayAnalyticsDuration(recovery.sleepTargetS) })
  return metrics
}

const DAY_ANALYTICS_SLEEP_WIDTH = 100
const DAY_ANALYTICS_SLEEP_HEIGHT = 24

const dayAnalyticsWallMinute = (iso: string): number | null => {
  const match = /T(\d{2}):(\d{2})/.exec(iso)
  if (!match) return null
  return Number(match[1]) * 60 + Number(match[2])
}

const dayAnalyticsWallClock = (minutes: number): string => {
  const value = ((Math.round(minutes) % 1440) + 1440) % 1440
  return `${Math.floor(value / 60)
    .toString()
    .padStart(2, '0')}:${(value % 60).toString().padStart(2, '0')}`
}

const dayAnalyticsHourTicks = (
  startIso: string,
  intervalS: number,
  count: number,
  denominator: number,
): AxisXTick[] => {
  const startMinute = dayAnalyticsWallMinute(startIso)
  if (startMinute == null || count < 2 || denominator <= 0) return []
  const ticks: AxisXTick[] = []
  const startS = startMinute * 60
  let bucket = Math.floor(startS / 7200)
  for (let index = 1; index < count; index++) {
    const nextBucket = Math.floor((startS + index * intervalS) / 7200)
    if (nextBucket === bucket) continue
    bucket = nextBucket
    ticks.push({
      label: dayAnalyticsWallClock((nextBucket * 7200) / 60),
      pct: (index / denominator) * 100,
    })
  }
  return ticks
}

const dayAnalyticsContributionGroup = <N>(
  f: TriNodeFactory<N>,
  title: string,
  contributions: Record<string, number | null> | null,
): N | null => {
  if (!contributions) return null
  const rows = Object.entries(contributions).filter(
    (entry): entry is [string, number] => entry[1] != null,
  )
  if (rows.length === 0) return null
  const group = f.el('section', 'tri-sleep-contrib')
  f.add(
    group,
    f.el('h4', 'tri-ana-block-title', triText(f.presentation.locale, title), {
      'data-i18n': title,
    }),
  )
  for (const [key, value] of rows) {
    const row = f.el('div', 'tri-sleep-contrib-row')
    const bar = f.el('div', 'tri-sleep-contrib-bar', undefined, { 'aria-hidden': 'true' })
    f.add(
      bar,
      f.el(
        'div',
        value >= 70
          ? 'tri-sleep-contrib-fill'
          : 'tri-sleep-contrib-fill tri-sleep-contrib-fill--low',
        undefined,
        { style: `width:${Math.max(0, Math.min(100, value))}%` },
      ),
    )
    const label = key.replaceAll('_', ' ')
    f.add(
      row,
      f.el('span', 'tri-sleep-contrib-label', triText(f.presentation.locale, label), {
        'data-i18n': label,
      }),
      bar,
      f.el('span', 'tri-sleep-contrib-val', dayAnalyticsNumber(f.presentation, value)),
    )
    f.add(group, row)
  }
  return group
}

const DAY_ANALYTICS_SLEEP_LANES = ['awake', 'rem', 'light', 'deep'] as const

const DAY_ANALYTICS_SLEEP_LANE_BY_CODE: Record<string, number> = { '4': 0, '3': 1, '2': 2, '1': 3 }

const DAY_ANALYTICS_SLEEP_STAGE_INTERVAL_S = 300

export const daySleepStageLabel = (locale: Locale, lane: number | null): string =>
  lane == null || DAY_ANALYTICS_SLEEP_LANES[lane] == null
    ? '—'
    : triText(locale, DAY_ANALYTICS_SLEEP_LANES[lane])

const dayAnalyticsSleepStages = <N>(
  f: TriNodeFactory<N>,
  date: string,
  sleep: NonNullable<TriathlonDayAnalytics['sleep']>,
): N | null => {
  const durations = [
    ['deep', sleep.deepS],
    ['light', sleep.lightS],
    ['rem', sleep.remS],
    ['awake', sleep.awakeS],
  ] as const
  const phase = sleep.phase5Min
  if (!phase && durations.every(([, seconds]) => seconds == null)) return null
  const bedtimeStart = sleep.bedtimeStart
  const startMinute = bedtimeStart ? dayAnalyticsWallMinute(bedtimeStart) : null
  const hypnogram =
    phase && phase.length >= 2 && bedtimeStart && startMinute != null
      ? {
          bedtimeStart,
          startMinute,
          lanes: Array.from(phase, code => DAY_ANALYTICS_SLEEP_LANE_BY_CODE[code] ?? null),
        }
      : null
  const readoutId = `tri-day-${date}-sleep-stages-readout`
  const chart = f.el(
    'section',
    'tri-day-sleep-chart tri-day-sleep-stages',
    undefined,
    hypnogram
      ? {
          'data-day-sleep-series': 'stages',
          'data-day-sleep-values': hypnogram.lanes.map(lane => lane ?? '').join(','),
          'data-day-sleep-start': hypnogram.bedtimeStart,
          'data-day-sleep-interval': DAY_ANALYTICS_SLEEP_STAGE_INTERVAL_S.toString(),
          'data-day-sleep-width': hypnogram.lanes.length.toString(),
        }
      : undefined,
  )
  f.add(
    chart,
    f.el('h4', 'tri-ana-block-title', triText(f.presentation.locale, 'sleep stages'), {
      'data-i18n': 'sleep stages',
    }),
  )
  if (hypnogram) {
    const { lanes, startMinute: bedMinute } = hypnogram
    const height = 16
    const measured = lanes.findLastIndex(lane => lane != null)
    const cursorIndex = Math.max(0, measured)
    const readout = `${dayAnalyticsWallClock(bedMinute + (cursorIndex * DAY_ANALYTICS_SLEEP_STAGE_INTERVAL_S) / 60)} · ${daySleepStageLabel(f.presentation.locale, measured >= 0 ? lanes[measured] : null)}`
    const svg = f.svg('svg', {
      class: 'tri-ana-svg tri-day-sleep-stage-svg',
      viewBox: `0 0 ${lanes.length} ${height}`,
      preserveAspectRatio: 'none',
      role: 'slider',
      tabindex: 0,
      'aria-label': triText(f.presentation.locale, 'sleep stages'),
      'aria-valuemin': 0,
      'aria-valuemax': lanes.length - 1,
      'aria-valuenow': cursorIndex,
      'aria-valuetext': readout,
      'aria-describedby': readoutId,
    })
    let start = 0
    while (start < lanes.length) {
      const lane = lanes[start]
      let end = start + 1
      while (end < lanes.length && lanes[end] === lane) end++
      if (lane != null)
        f.add(
          svg,
          f.svg('rect', {
            x: start,
            y: lane * 4 + 0.3,
            width: end - start,
            height: 3.4,
            class: `tri-hyp--${DAY_ANALYTICS_SLEEP_LANES[lane]}`,
          }),
        )
      start = end
    }
    f.add(
      svg,
      f.svg('line', {
        x1: cursorIndex + 0.5,
        y1: 0,
        x2: cursorIndex + 0.5,
        y2: height,
        class: 'tri-ana-cursor',
      }),
    )
    f.add(
      chart,
      axisFrame(
        f,
        svg,
        DAY_ANALYTICS_SLEEP_LANES.map((lane, index) => ({
          label: triText(f.presentation.locale, lane),
          vbY: index * 4 + 2,
        })),
        height,
        dayAnalyticsHourTicks(
          hypnogram.bedtimeStart,
          DAY_ANALYTICS_SLEEP_STAGE_INTERVAL_S,
          lanes.length,
          lanes.length,
        ),
        false,
      ),
      f.el('div', 'tri-chart-readout', readout, { id: readoutId }),
    )
  }
  const summary = f.el('div', 'tri-day-sleep-stage-summary')
  for (const [label, seconds] of durations)
    if (seconds != null)
      f.add(
        summary,
        f.el(
          'span',
          'tri-ana-k',
          `${triText(f.presentation.locale, label)} ${dayAnalyticsDuration(seconds)}`,
        ),
      )
  f.add(chart, summary)
  return chart
}

const dayAnalyticsSeriesPaths = (
  items: readonly (number | null)[],
  x: (index: number) => number,
  y: (value: number) => number,
): string[] => {
  const paths: string[] = []
  let path = ''
  let points = 0
  const flush = (): void => {
    if (points >= 2) paths.push(path)
    path = ''
    points = 0
  }
  for (const [index, value] of items.entries()) {
    if (value == null) {
      flush()
      continue
    }
    const point = `${x(index).toFixed(3)} ${y(value).toFixed(3)}`
    path += `${points === 0 ? 'M' : 'L'}${point}`
    points++
  }
  flush()
  return paths
}

const dayAnalyticsSleepSeries = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: 'hrv' | 'heart-rate',
  title: string,
  series: NonNullable<TriathlonDayAnalytics['sleep']>['hrv'],
): N | null => {
  if (!series || series.items.length < 2) return null
  const values = series.items.filter((value): value is number => value != null)
  if (values.length < 2) return null
  const low = Math.min(...values)
  const high = Math.max(...values)
  const padding = Math.max((high - low) * 0.1, 1)
  const minimum = low - padding
  const maximum = high + padding
  const x = (index: number): number =>
    (index / Math.max(1, series.items.length - 1)) * DAY_ANALYTICS_SLEEP_WIDTH
  const y = (value: number): number =>
    DAY_ANALYTICS_SLEEP_HEIGHT -
    2 -
    ((value - minimum) / Math.max(1, maximum - minimum)) * (DAY_ANALYTICS_SLEEP_HEIGHT - 4)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  const unit = key === 'hrv' ? 'ms' : 'bpm'
  const readoutId = `tri-day-${date}-sleep-${key}-readout`
  const initialIndex = series.items.findLastIndex(value => value != null)
  const startMinute = dayAnalyticsWallMinute(series.startTs) ?? 0
  const initialTime = dayAnalyticsWallClock(
    startMinute + (Math.max(0, initialIndex) * series.intervalS) / 60,
  )
  const initialValue = initialIndex >= 0 ? series.items[initialIndex] : null
  const initialReadout = `${initialTime} · ${initialValue == null ? '—' : Math.round(initialValue)} ${unit}`
  const chart = f.el(
    'section',
    `tri-day-sleep-chart tri-day-sleep-series tri-day-sleep-series--${key}`,
    undefined,
    {
      'data-day-sleep-series': key,
      'data-day-sleep-values': series.items.map(value => value ?? '').join(','),
      'data-day-sleep-start': series.startTs,
      'data-day-sleep-interval': series.intervalS.toString(),
      'data-day-sleep-unit': unit,
      'data-day-sleep-width': DAY_ANALYTICS_SLEEP_WIDTH.toString(),
    },
  )
  f.add(
    chart,
    f.el('h4', 'tri-ana-block-title', triText(f.presentation.locale, title), {
      'data-i18n': title,
    }),
  )
  const svg = f.svg('svg', {
    class: 'tri-ana-svg tri-day-sleep-line-svg',
    viewBox: `0 0 ${DAY_ANALYTICS_SLEEP_WIDTH} ${DAY_ANALYTICS_SLEEP_HEIGHT}`,
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': triText(f.presentation.locale, title),
    'aria-valuemin': 0,
    'aria-valuemax': series.items.length - 1,
    'aria-valuenow': Math.max(0, initialIndex),
    'aria-valuetext': initialReadout,
    'aria-describedby': readoutId,
  })
  f.add(
    svg,
    f.svg('line', {
      x1: 0,
      y1: y(average),
      x2: DAY_ANALYTICS_SLEEP_WIDTH,
      y2: y(average),
      class: 'tri-rec-target',
    }),
  )
  for (const path of dayAnalyticsSeriesPaths(series.items, x, y))
    f.add(svg, f.svg('path', { d: path, class: `tri-day-sleep-line tri-day-sleep-line--${key}` }))
  f.add(
    svg,
    f.svg('line', {
      x1: x(Math.max(0, initialIndex)),
      y1: 0,
      x2: x(Math.max(0, initialIndex)),
      y2: DAY_ANALYTICS_SLEEP_HEIGHT,
      class: 'tri-ana-cursor',
    }),
  )
  f.add(
    chart,
    axisFrame(
      f,
      svg,
      [
        { label: dayAnalyticsNumber(f.presentation, high), vbY: y(high) },
        { label: dayAnalyticsNumber(f.presentation, low), vbY: y(low) },
      ],
      DAY_ANALYTICS_SLEEP_HEIGHT,
      dayAnalyticsHourTicks(
        series.startTs,
        series.intervalS,
        series.items.length,
        series.items.length - 1,
      ),
    ),
    f.el('div', 'tri-chart-readout', initialReadout, { id: readoutId }),
  )
  return chart
}

const buildDaySleepAnalytics = <N>(
  f: TriNodeFactory<N>,
  summary: TriathlonDayAnalytics,
): N | null => {
  const sleep = summary.sleep
  const metrics = dayAnalyticsSleepMetrics(f.presentation, summary)
  if (!sleep && metrics.length === 0) return null
  const titleId = `tri-day-${summary.date}-sleep`
  const group = f.el(
    'section',
    'tri-day-analytics-group tri-day-analytics-group--sleep',
    undefined,
    { 'aria-labelledby': titleId },
  )
  f.add(
    group,
    f.el(
      'h3',
      'tri-ana-block-title tri-day-analytics-group-title',
      triText(f.presentation.locale, 'sleep details'),
      { id: titleId, 'data-i18n': 'sleep details' },
    ),
  )
  if (metrics.length > 0)
    f.add(
      group,
      dayAnalyticsList(
        f,
        summary.date,
        'sleep',
        metrics,
        'tri-day-analytics-list tri-day-sleep-summary',
      ),
    )
  if (sleep) {
    const sleepContrib = dayAnalyticsContributionGroup(f, 'sleep score', sleep.sleepContrib)
    const readinessContrib = dayAnalyticsContributionGroup(f, 'readiness', sleep.readinessContrib)
    if (sleepContrib || readinessContrib) {
      const contributions = f.el('div', 'tri-day-sleep-contributions')
      if (sleepContrib) f.add(contributions, sleepContrib)
      if (readinessContrib) f.add(contributions, readinessContrib)
      f.add(group, contributions)
    }
    const stages = dayAnalyticsSleepStages(f, summary.date, sleep)
    const hrv = dayAnalyticsSleepSeries(f, summary.date, 'hrv', 'hrv', sleep.hrv)
    const heartRate = dayAnalyticsSleepSeries(
      f,
      summary.date,
      'heart-rate',
      'resting heart rate',
      sleep.heartRate,
    )
    if (stages) f.add(group, stages)
    if (hrv) f.add(group, hrv)
    if (heartRate) f.add(group, heartRate)
  }
  return group
}

const buildRestDaySleepAnalytics = <N>(
  f: TriNodeFactory<N>,
  summary: TriathlonDayAnalytics,
): N | null => {
  const sleep = buildDaySleepAnalytics(f, summary)
  if (!sleep) return null
  const section = f.el('section', 'tri-day-analytics tri-day-rest-analytics', undefined, {
    'aria-label': triText(f.presentation.locale, 'sleep details'),
    'data-analytics-date': summary.date,
    'data-i18n-aria-label': 'sleep details',
  })
  const groups = f.el('div', 'tri-day-analytics-grid')
  f.add(groups, sleep)
  f.add(section, groups)
  return section
}

const dayAnalyticsTrainingMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const training = summary.training
  if (!training) return []
  const metrics: DayAnalyticsMetric[] = []
  if (training.vo2max)
    metrics.push({
      label: 'VO2max',
      value: `${dayAnalyticsNumber(presentation, training.vo2max.value, 1)} ml/kg/min`,
      detail:
        training.vo2max.method === 'garmin'
          ? 'Garmin'
          : training.vo2max.method === 'apple'
            ? 'Apple'
            : triText(presentation.locale, training.vo2max.method),
    })
  if (training.ctl != null)
    metrics.push({
      label: 'fitness · CTL',
      value: dayAnalyticsNumber(presentation, training.ctl, 1),
    })
  if (training.atl != null)
    metrics.push({
      label: 'fatigue · ATL',
      value: dayAnalyticsNumber(presentation, training.atl, 1),
    })
  if (training.tsb != null)
    metrics.push({ label: 'form · TSB', value: dayAnalyticsSigned(presentation, training.tsb, 1) })
  if (training.load != null)
    metrics.push({
      label: 'today load · TSS',
      value: dayAnalyticsNumber(presentation, training.load, 1),
      detail: triText(presentation.locale, 'site'),
    })
  if (training.garminTss != null)
    metrics.push({
      label: 'Garmin TSS',
      value: dayAnalyticsNumber(presentation, training.garminTss, 1),
    })
  if (training.exerciseLoad != null)
    metrics.push({
      label: 'exercise load',
      value: dayAnalyticsNumber(presentation, training.exerciseLoad, 1),
      detail:
        training.exerciseLoadSource === 'garmin'
          ? 'Garmin'
          : triText(presentation.locale, training.exerciseLoadSource ?? 'calculated'),
    })
  if (training.relativeEffort != null)
    metrics.push({
      label: 'relative effort',
      value: dayAnalyticsNumber(presentation, training.relativeEffort),
      detail: 'Strava',
    })
  return metrics
}

const dayAnalyticsHeatMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const heat = summary.heat
  if (!heat) return []
  const metrics: DayAnalyticsMetric[] = []
  if (heat.heatStrainIndex != null)
    metrics.push({ label: 'HSI', value: dayAnalyticsNumber(presentation, heat.heatStrainIndex, 1) })
  if (heat.temperatureC != null)
    metrics.push({
      label: heat.source === 'core' ? 'CORE temperature' : 'ambient temperature',
      value: dayAnalyticsTemperature(presentation, heat.temperatureC),
    })
  metrics.push(
    { label: 'observed', value: `${dayAnalyticsNumber(presentation, heat.observedMinutes)} min` },
    { label: 'hot min', value: dayAnalyticsNumber(presentation, heat.hotMinutes) },
    {
      label: 'acclimatisation',
      value: `${dayAnalyticsNumber(presentation, heat.acclimatisationPct)}%`,
    },
  )
  if (heat.dose > 0)
    metrics.push({ label: 'heat dose', value: dayAnalyticsNumber(presentation, heat.dose, 1) })
  return metrics
}

export const buildDayAnalytics = <N>(f: TriNodeFactory<N>, summary: TriathlonDayAnalytics): N => {
  const section = f.el('section', 'tri-day-analytics', undefined, {
    'aria-label': triText(f.presentation.locale, 'daily analytics'),
    'data-analytics-date': summary.date,
    'data-i18n-aria-label': 'daily analytics',
  })
  const groups = f.el('div', 'tri-day-analytics-grid')
  for (const group of [
    dayAnalyticsGroup(f, summary.date, 'body-recovery', 'body · recovery', [
      ...dayAnalyticsBodyMetrics(f.presentation, summary),
      ...dayAnalyticsRecoveryMetrics(f.presentation, summary),
    ]),
    buildDaySleepAnalytics(f, summary),
    dayAnalyticsGroup(
      f,
      summary.date,
      'state-load',
      'state · load',
      dayAnalyticsTrainingMetrics(f.presentation, summary),
    ),
    dayAnalyticsGroup(
      f,
      summary.date,
      'thermal',
      'thermal',
      dayAnalyticsHeatMetrics(f.presentation, summary),
    ),
  ])
    if (group) f.add(groups, group)
  f.add(section, groups)
  return section
}

export const buildDayCard = <N>(
  f: TriNodeFactory<N>,
  dateIso: string,
  payload: DayCardPayload | null,
  extras: DayCardExtras = {},
  activity?: (d: StravaActivityDetail, reserveFueling: boolean) => N,
  ctx?: DetailCtx,
): N => {
  const allDay = payload ? dayDetails(payload, dateIso) : []
  const selectedDay = extras.activityId
    ? allDay.filter(d => `${d.id}` === extras.activityId)
    : allDay
  const excludedActivityIds = new Set(extras.excludedActivityIds)
  const visibleDay =
    excludedActivityIds.size > 0
      ? selectedDay.filter(d => !excludedActivityIds.has(`${d.id}`))
      : selectedDay
  const day = extras.sport ? visibleDay.filter(d => d.sport === extras.sport) : visibleDay
  const summaryRows =
    extras.embedded === true && day.length > 1
      ? Math.max(
          ...day.map(
            d =>
              activityStatRows(f.presentation, d).length +
              moreStatRows(f.presentation, d, extras.event != null).length,
          ),
        )
      : null
  const sharedFuelingRows =
    extras.embedded === true && day.length > 1
      ? Math.max(...day.map(d => (d.fueling ? fuelingRows(d.fueling).length : 0)))
      : 0
  const render =
    activity ??
    ((d: StravaActivityDetail) =>
      buildActivity(
        f,
        d,
        dayCardActivitiesExpanded(extras),
        ctx,
        extras.event != null,
        extras.embedded === true,
        extras.settings,
        sharedFuelingRows > 0,
      ))
  const cardStyles = [
    summaryRows == null ? null : `--tri-embedded-summary-rows:${summaryRows}`,
    sharedFuelingRows === 0 ? null : `--tri-embedded-fueling-rows:${sharedFuelingRows}`,
  ].filter(style => style != null)
  const card = f.el(
    'div',
    'tri-pop-card',
    undefined,
    cardStyles.length === 0 ? undefined : { style: cardStyles.join(';') },
  )
  const head = f.el('div', 'tri-pop-head')
  f.add(
    head,
    extras.dateHref
      ? f.el('a', 'tri-pop-date', prettyDate(dateIso), { href: extras.dateHref })
      : f.el('span', 'tri-pop-date', prettyDate(dateIso)),
  )
  if (day.length > 0) {
    f.add(
      head,
      f.el(
        'span',
        'tri-pop-loc',
        day[0].location ?? recentLocation(payload!) ?? extras.location ?? 'Toronto',
      ),
    )
  }
  if (extras.event) {
    const track = f.el('div', 'tri-pop-track')
    f.add(track, f.el('span', 'tri-pop-race', extras.event))
    f.add(head, track)
  }
  f.add(card, head)
  const dailyAnalytics = payload?.dailyAnalytics?.[dateIso]
  if (extras.analytics && dailyAnalytics) {
    f.add(card, buildDayAnalytics(f, dailyAnalytics))
    if (day.length > 0)
      f.add(
        card,
        f.el(
          'h2',
          'tri-ana-block-title tri-day-activities-title',
          triText(f.presentation.locale, 'activities'),
          { 'data-i18n': 'activities' },
        ),
      )
  }
  if (!payload) {
    f.add(card, f.el('div', 'tri-pop-rest', '·'))
  } else if (day.length === 0) {
    if (extras.sport) {
      const rest = f.el('div', 'tri-pop-rest')
      f.add(rest, f.el('span', 'tri-pop-rest-label', `no ${extras.sport}`))
      f.add(card, rest)
    } else {
      f.add(card, buildRestStatus(f))
    }
  } else {
    for (const d of day) f.add(card, render(d, sharedFuelingRows > 0))
  }
  if (
    !extras.analytics &&
    !extras.sport &&
    !extras.activityId &&
    allDay.length === 0 &&
    dailyAnalytics
  ) {
    const sleep = buildRestDaySleepAnalytics(f, dailyAnalytics)
    if (sleep) f.add(card, sleep)
  }
  if (!extras.sport && !extras.activityId && !extras.analytics) {
    const dh = payload?.health[dateIso]
    if (dh) {
      const rec = buildRecovery(f, dh)
      if (rec) f.add(card, rec)
    }
  }
  return card
}
