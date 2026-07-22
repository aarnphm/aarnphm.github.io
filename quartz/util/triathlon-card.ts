import { STROKE_LABEL, SWIM_STROKES } from '../plugins/stores/apple'
import {
  SPORT_ICON,
  type ActivityAnalysisKind,
  type ActivityAnalysisRange,
  type ActivityHealth,
  type ActivityKind,
  type PowerCurvePoint,
  type StravaActivityDetail,
  type StravaZones,
  type SwimActivityInterval,
  type SwimTrendPoint,
} from '../plugins/stores/strava'
import { swimPaceSeconds, swimStrokeRate } from './swim-metrics'

export interface TriNodeFactory<N> {
  el: (tag: string, cls?: string, text?: string, attrs?: Record<string, string>) => N
  svg: (tag: string, attrs: Record<string, string | number>) => N
  add: (parent: N, ...children: N[]) => void
}

export type DayCardExtras = {
  location?: string
  event?: string
  sport?: ActivityKind
  expanded?: boolean
  dateHref?: string
}

export type DayCardPayload = {
  details: Record<string, StravaActivityDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
}

export type ActivityFueling = NonNullable<StravaActivityDetail['fueling']>

export const KM_TO_MI = 0.621371
export const M_TO_FT = 3.28084

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

let imperial = false
export const setDistanceUnit = (v: boolean): void => {
  imperial = v
}
export const isImperialUnit = (): boolean => imperial

export const dist = (km: number, sport: ActivityKind): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  return imperial ? `${(km * KM_TO_MI).toFixed(1)} mi` : `${km.toFixed(1)} km`
}

export const distCombined = (km: number): string =>
  imperial
    ? `${Math.round(km * KM_TO_MI).toLocaleString('en-US')} mi`
    : `${Math.round(km).toLocaleString('en-US')} km`

export const dur = (s: number): string => {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
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

export const speedKph = (kmh: number): string =>
  imperial ? `${(kmh * KM_TO_MI).toFixed(1)} mph` : `${kmh.toFixed(1)} km/h`

export const rate = (sport: ActivityKind, km: number, s: number): string => {
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  if (sport === 'bike') return speedKph(km / (s / 3600))
  return imperial ? `${clock(s / (km * KM_TO_MI))} /mi` : `${clock(s / km)} /km`
}

export const scrubDist = (km: number, sport: ActivityKind): string =>
  sport === 'swim'
    ? `${Math.round(km * 1000).toLocaleString('en-US')} m`
    : imperial
      ? `${(km * KM_TO_MI).toFixed(2)} mi`
      : `${km.toFixed(2)} km`

const elevationValue = (meters: number): number => (imperial ? meters * M_TO_FT : meters)
const temperatureValue = (celsius: number): number => (imperial ? (celsius * 9) / 5 + 32 : celsius)
const temperatureUnit = (): string => (imperial ? '°F' : '°C')

export const formatTemperature = (celsius: number): string =>
  `${Math.round(temperatureValue(celsius))}${temperatureUnit()}`

export const formatRespirationRate = (breathsPerMinute: number): string =>
  `${breathsPerMinute.toFixed(1)} brpm`

export const formatAltitude = (meters: number): string => {
  const rounded = Math.round(elevationValue(meters))
  return `${(rounded === 0 ? 0 : rounded).toLocaleString('en-US')} ${imperial ? 'ft' : 'm'}`
}

export const formatElevationGain = (meters: number): string => formatAltitude(meters)

export const formatVam = (metersPerHour: number): string =>
  `${Math.round(elevationValue(metersPerHour)).toLocaleString('en-US')} ${imperial ? 'ft/h' : 'm/h'}`

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
  const value = fueling.sourceDevice
  const clean = value?.trim()
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
    rows.push(['day burn', `${Math.round(h.totalCalories).toLocaleString('en-US')} kcal`])
  if (h.activeCalories != null)
    rows.push(['day active', `${Math.round(h.activeCalories).toLocaleString('en-US')} kcal`])
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
  if (rows.length > 0) rows.push(['source', formatFuelingSource(f)])
  return rows
}

export const moreStatRows = (d: StravaActivityDetail): [string, string][] => {
  const rows: [string, string][] = []
  if (d.deviceWatts && d.npWatts != null) rows.push(['NP', `${d.npWatts} W`])
  if (d.avgWatts != null) rows.push([d.deviceWatts ? 'avg power' : 'est power', `${d.avgWatts} W`])
  if (d.deviceWatts && d.maxWatts != null) rows.push(['max power', `${d.maxWatts} W`])
  if (d.kilojoules != null) rows.push(['energy', `${d.kilojoules} kJ`])
  if (d.calories != null) rows.push(['calories', `${d.calories.toLocaleString('en-US')} kcal`])
  if (d.avgCadence != null)
    rows.push(['cadence', d.sport === 'run' ? `${d.avgCadence * 2} spm` : `${d.avgCadence} rpm`])
  if (d.maxHr != null) rows.push(['max hr', `${d.maxHr} bpm`])
  if (d.sufferScore != null) rows.push(['effort', `${d.sufferScore}`])
  if (d.avgTemp != null) rows.push(['temp', formatTemperature(d.avgTemp)])
  if (d.windKph != null)
    rows.push([
      'wind',
      `${d.windKph} km/h${d.windDir ? ` ${d.windDir}` : ''}${d.windGustKph != null ? ` / gust ${d.windGustKph}` : ''}`,
    ])
  return rows
}

export const routeStreamFlags = (
  d: StravaActivityDetail,
): {
  power: boolean
  hr: boolean
  cad: boolean
  stride: boolean
  groundContact: boolean
  verticalOscillation: boolean
  resp: boolean
  temp: boolean
} => ({
  power: d.deviceWatts && d.route.some(p => p.w > 0),
  hr: d.route.some(p => p.hr > 0),
  cad: d.route.some(p => p.cad > 0),
  stride:
    d.sport === 'run' &&
    d.route.filter(point => runStrideLengthValue(d, point) != null).length >= 2,
  groundContact:
    d.sport === 'run' && d.route.filter(point => runGroundContactTimeMs(point) != null).length >= 2,
  verticalOscillation:
    d.sport === 'run' &&
    d.route.filter(point => runVerticalOscillationCm(point) != null).length >= 2,
  resp: d.route.some(p => p.resp != null && p.resp > 0),
  temp: d.route.some(p => p.tempC != null),
})

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

export const formatStrideLength = (meters: number): string =>
  imperial ? `${(meters * M_TO_FT).toFixed(2)} ft` : `${meters.toFixed(2)} m`

export const formatGroundContactTime = (milliseconds: number): string =>
  `${Math.round(milliseconds)} ms`

export const formatVerticalOscillation = (centimeters: number): string =>
  imperial ? `${(centimeters / 2.54).toFixed(1)} in` : `${centimeters.toFixed(1)} cm`

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
  ) &&
  validAnalysisRanges(d).length > 0

export const hasMoreSection = (d: StravaActivityDetail): boolean => {
  const flags = routeStreamFlags(d)
  const efforts = d.bestEfforts
  return (
    flags.power ||
    flags.hr ||
    flags.cad ||
    flags.stride ||
    flags.groundContact ||
    flags.verticalOscillation ||
    flags.resp ||
    flags.temp ||
    (d.sport === 'run' && runLapSplits(d).length > 0) ||
    !!(efforts && (efforts.distance.length || efforts.power.length || efforts.climbs.length)) ||
    !!(d.hrZones || d.powerZones || d.powerHist || d.powerCurve)
  )
}

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

export const buildBattery = <N>(f: TriNodeFactory<N>): N => {
  const icon = f.svg('svg', { class: 'tri-ico tri-battery', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of BATTERY) f.add(icon, f.svg('path', { d }))
  return icon
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

const niceStep = (span: number, intervals: number): number => {
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

const axisNumber = (value: number, step: number): string => {
  const decimals = step >= 1 ? 0 : Math.min(2, Math.ceil(-Math.log10(step)))
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

const distanceXTicks = (maxDKm: number): AxisXTick[] => {
  const displayMaxD = maxDKm * (imperial ? KM_TO_MI : 1)
  const step = niceStep(displayMaxD, 4)
  const ticks: AxisXTick[] = []
  for (let value = step; value < displayMaxD - step * 1e-6; value += step)
    ticks.push({
      label: `${axisNumber(value, step)} ${imperial ? 'mi' : 'km'}`,
      pct: (value / displayMaxD) * 100,
    })
  return ticks
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
): N => {
  const w = 100
  const h = 30
  const maxD = d.route[d.route.length - 1].d || 1
  const displayMinAlt = elevationValue(d.minAlt)
  const displayMaxAlt = elevationValue(d.maxAlt)
  const altPad = displayMinAlt === displayMaxAlt ? 1 : 0
  const minAlt = displayMinAlt - altPad
  const maxAlt = displayMaxAlt + altPad
  const altSpan = Math.max(1e-6, maxAlt - minAlt)
  const px = (km: number): number => (km / maxD) * w
  const py = (alt: number): number => h - ((elevationValue(alt) - minAlt) / altSpan) * h
  const yValues = niceTicks(minAlt, maxAlt, 4)
  const yStep = niceStep(maxAlt - minAlt, 4)
  const yTicks = yValues.map(value => ({
    label: `${axisNumber(value, yStep)} ${imperial ? 'ft' : 'm'}`,
    vbY: h - ((value - minAlt) / altSpan) * h,
  }))
  const xTicks = distanceXTicks(maxD)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    area += `L ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(p.alt).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const fig = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `0 0 ${w} ${h}`,
    preserveAspectRatio: 'none',
  })
  for (const tick of yTicks)
    f.add(fig, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: tick.vbY, x2: w, y2: tick.vbY }))
  f.add(fig, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (selection !== undefined) f.add(fig, buildAnalysisSelection(f, d, h, selection))
  f.add(fig, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(fig, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap')
  const cap = f.el('div', 'tri-elev-cap')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', `+${formatElevationGain(d.elevationM)}`),
    f.el('span', 'tri-elev-d', `−${formatElevationGain(d.descentM)}`),
    f.el('span', 'tri-elev-range', `${formatAltitude(d.minAlt)}–${formatAltitude(d.maxAlt)}`),
  )
  const frame = axisFrame(f, fig, yTicks, h, xTicks, true, { top: 0, bottom: h })
  f.add(wrap, frame, cap)
  return wrap
}

export const buildTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number | null,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
  domain?: { min: number; max: number; intervals?: number },
  selection?: ActivityAnalysisRange | null,
): N => {
  const w = 100
  const h = 30
  const maxD = d.route[d.route.length - 1].d || 1
  let peak = 1
  const values = d.route.map(pick)
  values.forEach(value => {
    if (value != null && Number.isFinite(value) && value > peak) peak = value
  })
  const domainMin = domain?.min ?? 0
  const domainMax = Math.max(domain?.max ?? peak, domainMin + 1)
  const px = (km: number): number => (km / maxD) * w
  const py = (v: number): number => h - ((v - domainMin) / (domainMax - domainMin)) * (h - 1)
  let area = ''
  let line = ''
  let segmentStart = -1
  const closeSegment = (start: number, end: number): void => {
    const first = values[start]
    if (first == null) return
    const startX = start === 0 ? 0 : px(d.route[start].d)
    const firstX = px(d.route[start].d)
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
      const x = px(d.route[i].d).toFixed(2)
      const y = py(value).toFixed(2)
      area += `L ${x} ${y} `
      line += `L ${x} ${y} `
    }
    area += `L ${px(d.route[end].d).toFixed(2)} ${h} Z `
  }
  values.forEach((value, index) => {
    const valid = value != null && Number.isFinite(value)
    if (valid && segmentStart < 0) segmentStart = index
    if (segmentStart >= 0 && (!valid || index === values.length - 1)) {
      closeSegment(segmentStart, valid ? index : index - 1)
      segmentStart = -1
    }
  })
  const yTicks = niceTicks(domainMin, domainMax, domain?.intervals ?? 3).map(value => ({
    label: value === 0 ? '0' : tick(value),
    vbY: py(value),
  }))
  const s = f.svg('svg', {
    class: 'tri-elev',
    viewBox: `0 0 ${w} ${h}`,
    preserveAspectRatio: 'none',
  })
  for (const t of yTicks)
    f.add(s, f.svg('line', { class: 'tri-elev-grid', x1: 0, y1: t.vbY, x2: w, y2: t.vbY }))
  f.add(s, f.svg('path', { d: area, class: 'tri-elev-area' }))
  if (selection !== undefined) f.add(s, buildAnalysisSelection(f, d, h, selection))
  f.add(s, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(s, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap', undefined, { 'data-tri-trace': title })
  const capEl = f.el('div', 'tri-elev-cap')
  f.add(capEl, f.el('span', 'tri-elev-d', title), f.el('span', 'tri-elev-range', cap(peak)))
  f.add(wrap, capEl, axisFrame(f, s, yTicks, h, distanceXTicks(maxD), true, { top: 0, bottom: h }))
  return wrap
}

export const buildRunStrideTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N | null => {
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
    () => `${formatStrideLength(averageM)} avg`,
    value => `${value.toFixed(1)}${unit}`,
    { min, max, intervals: 2 },
    selection,
  )
}

export const buildRunGroundContactTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
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
  )
}

export const buildRunVerticalOscillationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
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
    () => `${formatVerticalOscillation(average)} avg`,
    value => `${value.toFixed(1)}cm`,
    { min, max, intervals: 2 },
    selection,
  )
}

const buildTemperatureTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
): N => {
  const temperaturesC = d.route
    .map(point => point.tempC)
    .filter((value): value is number => value != null)
  const averageC =
    d.avgTemp ?? temperaturesC.reduce((total, value) => total + value, 0) / temperaturesC.length
  const values = temperaturesC.map(temperatureValue)
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
    point => temperatureValue(point.tempC ?? averageC),
    'temperature',
    () => `${formatTemperature(averageC)} avg`,
    value => `${Math.round(value)}${temperatureUnit()}`,
    { min, max, intervals: 2 },
    selection,
  )
}

export const buildRespirationTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
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
  )
}

const analysisRangeAttrs = (range: ActivityAnalysisRange): Record<string, string> => {
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
    'data-duration-s': `${range.durationS}`,
    'data-distance-km': `${range.distanceKm}`,
  }
  if (range.elevationGainM != null) attrs['data-elevation-gain-m'] = `${range.elevationGainM}`
  if (range.averageSpeedKph != null) attrs['data-average-speed-kph'] = `${range.averageSpeedKph}`
  if (range.averageHeartRate != null) attrs['data-average-heart-rate'] = `${range.averageHeartRate}`
  if (range.averageWatts != null) attrs['data-average-watts'] = `${range.averageWatts}`
  if (range.averageCadence != null) attrs['data-average-cadence'] = `${range.averageCadence}`
  return attrs
}

const analysisRangeRate = (sport: ActivityKind, speedKphValue: number): string => {
  if (sport === 'bike') return speedKph(speedKphValue)
  if (sport === 'swim') return `${clock(360 / speedKphValue)} /100m`
  return `${clock(3600 / (speedKphValue * (imperial ? KM_TO_MI : 1)))} /${imperial ? 'mi' : 'km'}`
}

const analysisRangeMetrics = (d: StravaActivityDetail, range: ActivityAnalysisRange): string[] => {
  const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const cadenceScale = d.sport === 'run' ? 2 : 1
  const values = [scrubDist(range.distanceKm, d.sport)]
  if (range.elevationGainM != null) values.push(`+${formatElevationGain(range.elevationGainM)}`)
  values.push(clock(range.durationS))
  if (range.averageSpeedKph != null) values.push(analysisRangeRate(d.sport, range.averageSpeedKph))
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

const runPaceSeconds = (speedKph: number): number => 3600 / (speedKph * (imperial ? KM_TO_MI : 1))

const projectedRunSplits = (d: StravaActivityDetail): ActivityAnalysisRange[] => {
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

const runLapSplits = (d: StravaActivityDetail): RunLapSplit[] => {
  const nativeSplits = projectedRunSplits(d)
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
        : (range.distanceKm / range.durationS) * 3600
    if (!Number.isFinite(speedKph) || speedKph <= 0) continue
    const paceS = runPaceSeconds(speedKph)
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

export const buildRunLapSplits = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  if (d.sport !== 'run') return null
  const splits = runLapSplits(d)
  if (splits.length === 0) return null
  const maxSpeedKph = Math.max(...splits.map(split => split.speedKph))
  const totalDistanceKm = splits.reduce((total, split) => total + split.range.distanceKm, 0)
  const totalDurationS = splits.reduce((total, split) => total + split.range.durationS, 0)
  const averageSpeedKph = (totalDistanceKm / totalDurationS) * 3600
  const averagePct = Math.max(0, Math.min(100, (averageSpeedKph / maxSpeedKph) * 100))
  const paceUnit = imperial ? '/mi' : '/km'
  const wrap = f.el('section', 'tri-run-splits', undefined, { 'aria-label': 'Run lap splits' })
  const head = f.el('div', 'tri-run-splits-head')
  f.add(
    head,
    f.el('span', 'tri-run-splits-title', 'lap splits'),
    f.el(
      'span',
      'tri-run-splits-average',
      `avg ${clock(runPaceSeconds(averageSpeedKph))} ${paceUnit}`,
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
    const metrics = analysisRangeMetrics(d, split.range)
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
  const wrap = f.el('section', 'tri-analysis', undefined, {
    'data-tri-analysis': '',
    'data-activity-id': `${d.id}`,
    'aria-label': 'Activity analysis',
  })
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
    if (empty && kind !== 'climb') continue
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
      const metrics = analysisRangeMetrics(d, range)
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
  f.add(wrap, fig, f.el('span', 'tri-pool-cap', `${lengths} × 25m`))
  const strokes = buildSwimStrokes(f, d)
  if (strokes) f.add(wrap, strokes)
  return wrap
}

type SwimActivityObservation = { interval: SwimActivityInterval; index: number }

type SwimActivityMetric = { observation: SwimActivityObservation; value: number }

type SwimActivityComparison = { delta: number; priorCount: number }

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
): SwimActivityInterval[] => {
  if (!Number.isFinite(blockDistanceM) || blockDistanceM <= 0) return []
  const blocks: SwimActivityInterval[] = []
  let cumulativeDistanceM = 0
  let distanceM = 0
  let durationS = 0
  let strokeCount = 0
  let strokeTimeS = 0
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
    })
    distanceM = 0
    durationS = 0
    strokeCount = 0
    strokeTimeS = 0
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
      if (interval.stroke !== 'kickboard') {
        if (positiveMetric(interval.strokeCount) && positiveMetric(interval.strokeTimeS)) {
          const fraction = contributionDistanceM / interval.distanceM
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

const swimTrendDayTime = (date: string): number | null => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null
  const time = Date.parse(`${date}T00:00:00Z`)
  return Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === date ? time : null
}

const swimTrendTime = (start: string | null | undefined, date: string): number | null => {
  const dayTime = swimTrendDayTime(date)
  if (dayTime == null) return null
  if (!start || !/^\d{4}-\d{2}-\d{2}T/.test(start)) return dayTime
  const time = Date.parse(start)
  return Number.isFinite(time) ? time : dayTime
}

const swimTrendNumber = (value: number): string =>
  value.toLocaleString('en-US', { maximumFractionDigits: 1 })

const swimPaceDelta = (delta: number, priorCount: number): string => {
  const magnitude = swimTrendNumber(Math.abs(delta))
  if (Math.abs(delta) < 0.05) return `same as prior ${priorCount}`
  return `${magnitude}s ${delta < 0 ? 'faster' : 'slower'} vs prior ${priorCount}`
}

const swimStrokeDelta = (delta: number, priorCount: number): string => {
  if (Math.abs(delta) < 0.05) return `same as prior ${priorCount}`
  const sign = delta > 0 ? '+' : '−'
  return `${sign}${swimTrendNumber(Math.abs(delta))} str/min vs prior ${priorCount}`
}

const swimTrendDisplayValue = (kind: 'pace' | 'stroke', value: number): string =>
  kind === 'pace' ? `${clock(value)} /100m` : `${swimTrendNumber(value)} str/min`

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
  kind: 'pace' | 'stroke',
  point: Pick<
    SwimTrendChartPoint,
    'elapsedS' | 'cumulativeDistanceM' | 'value' | 'windowStartDistanceM'
  >,
): string => {
  const position =
    point.windowStartDistanceM == null
      ? `${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
      : `${Math.round(point.cumulativeDistanceM - point.windowStartDistanceM)} metre block from ${Math.round(point.windowStartDistanceM)} to ${Math.round(point.cumulativeDistanceM)} metres, ${clock(point.elapsedS)} elapsed`
  return kind === 'pace'
    ? `${position}, swim pace ${clock(point.value)} per 100 metres`
    : `${position}, stroke rate ${swimTrendNumber(point.value)} strokes per minute`
}

const swimTrendDomain = (values: number[], minimumStep = 0): { min: number; max: number } => {
  const observedMax = Math.max(...values)
  const step = Math.max(minimumStep, niceStep(observedMax, 3))
  return { min: 0, max: Math.max(step, Math.ceil(observedMax / step) * step) }
}

const swimActivityXTicks = (totalDistanceM: number): AxisXTick[] => {
  return [
    { label: '0 m', pct: 0, cls: 'tri-cax-xt--first' },
    { label: swimDistanceLabel(totalDistanceM / 2), pct: 50 },
    { label: swimDistanceLabel(totalDistanceM), pct: 100, cls: 'tri-cax-xt--last' },
  ]
}

const swimActivityComparison = (
  d: StravaActivityDetail,
  points: SwimTrendPoint[],
  kind: 'pace' | 'stroke',
  current: number,
): SwimActivityComparison | null => {
  const selectedTime = swimTrendTime(d.start, d.date)
  if (selectedTime == null) return null
  const candidates: { point: SwimTrendPoint; time: number; value: number }[] = []
  for (const point of points) {
    if (point.id === d.id) continue
    const time = swimTrendTime(point.start, point.date)
    const value = kind === 'pace' ? point.paceSPer100m : point.strokeRateSpm
    if (time == null || time > selectedTime || !positiveMetric(value)) continue
    candidates.push({ point, time, value })
  }
  const prior = candidates.sort((a, b) => a.time - b.time || a.point.id - b.point.id).slice(-4)
  if (prior.length === 0) return null
  const baseline = prior.reduce((sum, observation) => sum + observation.value, 0) / prior.length
  return { delta: current - baseline, priorCount: prior.length }
}

const buildSwimTrendChart = <N>(
  f: TriNodeFactory<N>,
  observations: SwimActivityObservation[],
  hundredMetreObservations: SwimActivityObservation[],
  totalDistanceM: number,
  kind: 'pace' | 'stroke',
  average: number | null,
  comparison: SwimActivityComparison | null,
  pick: (interval: SwimActivityInterval) => number | null,
): N | null => {
  const metricSeries = (source: SwimActivityObservation[]): SwimActivityMetric[] => {
    const metrics: SwimActivityMetric[] = []
    for (const observation of source) {
      const value = pick(observation.interval)
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
  const title = kind === 'pace' ? 'pace /100m' : 'stroke rate str/min'
  const value = kind === 'pace' ? clock(activityAverage) : swimTrendNumber(activityAverage)
  const deltaText = comparison
    ? kind === 'pace'
      ? swimPaceDelta(comparison.delta, comparison.priorCount)
      : swimStrokeDelta(comparison.delta, comparison.priorCount)
    : 'activity avg'
  const ariaDelta =
    comparison == null
      ? 'activity average'
      : Math.abs(comparison.delta) < 0.05
        ? `same as prior ${comparison.priorCount}`
        : kind === 'pace'
          ? `${swimTrendNumber(Math.abs(comparison.delta))} seconds ${comparison.delta < 0 ? 'faster' : 'slower'} than prior ${comparison.priorCount}`
          : `${swimTrendNumber(Math.abs(comparison.delta))} strokes per minute ${comparison.delta > 0 ? 'above' : 'below'} prior ${comparison.priorCount}`
  const wrap = f.el('article', `tri-zone tri-swim-trend tri-swim-trend--${kind}`)
  const head = f.el('div', 'tri-swim-trend-head')
  f.add(
    head,
    f.el('span', 'tri-swim-trend-title', title, { 'data-i18n': title }),
    f.el('strong', 'tri-swim-trend-value', value, {
      'data-swim-average-kind': kind,
      'data-swim-average-value': activityAverage.toString(),
    }),
    f.el('span', 'tri-swim-trend-delta', deltaText, {
      'data-swim-comparison-kind': kind,
      ...(comparison
        ? {
            'data-swim-comparison-delta': comparison.delta.toString(),
            'data-swim-comparison-prior': comparison.priorCount.toString(),
          }
        : {}),
    }),
  )
  const W = 100
  const H = 30
  const X = (observation: SwimActivityObservation): number =>
    (observation.interval.cumulativeDistanceM / totalDistanceM) * W
  const domain = swimTrendDomain(
    [...series, ...hundredMetreSeries].map(metric => metric.value),
    kind === 'pace' ? 1 : 0,
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
              `${index === 0 ? 'M' : 'L'} ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`,
          )
          .join(' '),
      )
      .join(' ')
    const areaPath = runs
      .filter(run => run.length > 1)
      .map(run => {
        const first = run[0]
        const last = run[run.length - 1]
        const values = run
          .map(metric => `L ${X(metric.observation).toFixed(2)} ${Y(metric.value).toFixed(2)}`)
          .join(' ')
        return `M ${X(first.observation).toFixed(2)} ${H} ${values} L ${X(last.observation).toFixed(2)} ${H} Z`
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
    'aria-label': `Swim ${kind === 'pace' ? 'pace' : 'stroke rate'} by length`,
    'aria-orientation': 'horizontal',
    'aria-valuemin': 0,
    'aria-valuemax': Math.round(totalDistanceM),
    'aria-valuenow': Math.round(currentChartPoint.cumulativeDistanceM),
    'aria-valuetext': `${swimTrendAriaValue(kind, currentChartPoint)}. Activity average ${swimTrendDisplayValue(kind, activityAverage)}. ${ariaDelta}.`,
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
  points: SwimTrendPoint[],
): N | null => {
  if (d.sport !== 'swim') return null
  const observations = d.swimIntervals
    .filter(
      interval => interval.endElapsedS > interval.startElapsedS && interval.cumulativeDistanceM > 0,
    )
    .map((interval, index) => ({ interval, index }))
  const totalDistanceM = observations.at(-1)?.interval.cumulativeDistanceM ?? 0
  if (observations.length < 2 || totalDistanceM <= 0) return null
  const hundredMetreObservations = swimActivityBlocks(
    observations.map(observation => observation.interval),
  ).map((interval, index) => ({ interval, index }))
  const hasSeries = (
    source: SwimActivityObservation[],
    pick: (interval: SwimActivityInterval) => number | null,
  ): boolean => source.filter(observation => positiveMetric(pick(observation.interval))).length >= 2
  const paceVisible = hasSeries(observations, interval => interval.paceSPer100m)
  const strokeVisible = hasSeries(observations, interval => interval.strokeRateSpm)
  const canToggle =
    hundredMetreObservations.length >= 2 &&
    (!paceVisible || hasSeries(hundredMetreObservations, interval => interval.paceSPer100m)) &&
    (!strokeVisible || hasSeries(hundredMetreObservations, interval => interval.strokeRateSpm))
  const normalizedObservations = canToggle ? hundredMetreObservations : []
  const paceAverage = positiveMetric(d.swimPaceSPer100m) ? d.swimPaceSPer100m : null
  const strokeAverage = positiveMetric(d.strokeRateSpm) ? d.strokeRateSpm : null
  const pace = buildSwimTrendChart(
    f,
    observations,
    normalizedObservations,
    totalDistanceM,
    'pace',
    paceAverage,
    paceAverage == null ? null : swimActivityComparison(d, points, 'pace', paceAverage),
    interval => interval.paceSPer100m,
  )
  const stroke = buildSwimTrendChart(
    f,
    observations,
    normalizedObservations,
    totalDistanceM,
    'stroke',
    strokeAverage,
    strokeAverage == null ? null : swimActivityComparison(d, points, 'stroke', strokeAverage),
    interval => interval.strokeRateSpm,
  )
  const trends = zoneDuo(f, pace, stroke)
  if (!trends) return null
  const wrap = f.el('section', 'tri-swim-trends', undefined, {
    'aria-label': 'Swim activity analysis',
    'data-i18n-aria-label': 'swim activity analysis',
  })
  if (canToggle) {
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
    f.add(wrap, toggle)
  }
  f.add(wrap, trends)
  return wrap
}

export const statRow = <N>(f: TriNodeFactory<N>, label: string, value: string): N => {
  const tr = f.el('tr')
  f.add(tr, f.el('th', 'tri-act-stat-k', label), f.el('td', 'tri-act-stat-v', value))
  return tr
}

export const statsTable = <N>(f: TriNodeFactory<N>, rows: [string, string][]): N => {
  const table = f.el('table', 'tri-act-stats')
  const tbody = f.el('tbody')
  for (const [k, v] of rows) f.add(tbody, statRow(f, k, v))
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
  ftp: number | null
  goalFtp: number | null
  vt1: number | null
}

export type AxisXTick = { label: string; pct: number; cls?: string }

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

export const dlabel = (sec: number): string =>
  sec < 60 ? `${sec}s` : sec < 3600 ? `${sec / 60}m` : `${sec / 3600}h`

export type PowerCurveHover = {
  index: number
  durationS: number
  watts: number
  referenceWatts: number | null
  xPct: number
}

const POWER_CURVE_PATH_POINTS = 1_024

export const encodePowerCurve = (curve: PowerCurvePoint[]): string => {
  if (curve.length === 0) return ''
  const consecutive = curve.every((point, index) => point.s === curve[0].s + index)
  return consecutive
    ? `d|${curve[0].s}|${curve.map(point => point.w).join(',')}`
    : `s|${curve.map(point => `${point.s}:${point.w}`).join(',')}`
}

export const decodePowerCurve = (encoded: string | undefined): PowerCurvePoint[] => {
  if (!encoded) return []
  const fields = encoded.split('|')
  const points: PowerCurvePoint[] = []
  if (fields[0] === 'd' && fields.length === 3) {
    const start = Number(fields[1])
    if (!Number.isInteger(start) || start <= 0 || fields[2].length === 0) return []
    for (const [index, raw] of fields[2].split(',').entries()) {
      if (raw.length === 0) return []
      const watts = Number(raw)
      if (!Number.isFinite(watts)) return []
      points.push({ s: start + index, w: watts })
    }
    return points
  }
  if (fields[0] !== 's' || fields.length !== 2 || fields[1].length === 0) return []
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
  return points
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

const nearestPowerCurveIndex = (curve: PowerCurvePoint[], seconds: number): number => {
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

const powerCurvePathPoints = (curve: PowerCurvePoint[]): PowerCurvePoint[] => {
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

const cyclingSpeed = (kph: number): string =>
  imperial ? `${(kph * KM_TO_MI).toFixed(1)} mph` : `${kph.toFixed(1)} km/h`

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
          row.label || scrubDist(row.targetDistanceM / 1000, 'bike'),
          zoneClock(row.elapsedTimeS),
          cyclingSpeed(row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          formatAltitude(row.elevationDeltaM),
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
          formatAltitude(row.elevationDeltaM),
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
          scrubDist(row.distanceM / 1000, 'bike'),
          formatElevationGain(row.elevationGainM),
          `${row.averageGradePct.toFixed(1)}%`,
          cyclingSpeed(row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          watts(row.averageWatts),
          wattsPerKg(row.wattsPerKg),
          formatVam(row.vamMPerHour),
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
): N => {
  const frame = f.el('div', 'tri-cax-frame')
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
      f.el('span', `tri-cax-xt${t.cls ? ` ${t.cls}` : ''}`, t.label, {
        style: `left:${t.pct.toFixed(2)}%`,
      }),
    )
  f.add(frame, yax, stage, xax)
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
  const wrap = f.el('div', 'tri-zone')
  f.add(wrap, f.el('div', 'tri-zone-title', title, { 'data-i18n': title }))
  const total = times.reduce((s, x) => s + x, 0) || 1
  let mx = 1
  for (const t of times) if (t > mx) mx = t
  const grid = f.el('div', 'tri-zone-grid')
  for (let i = times.length - 1; i >= 0; i--) {
    const row = f.el('div', 'tri-zone-row')
    const track = f.el('span', 'tri-zone-bar')
    f.add(
      track,
      f.el('span', `tri-zone-fill tri-zone-fill--${i + 1}`, undefined, {
        style: `width:${(times[i] / mx) * 100}%`,
      }),
    )
    f.add(
      row,
      f.el('span', 'tri-zone-z', `Z${i + 1}`, {
        'data-name': names[i] ?? `Z${i + 1}`,
        tabindex: '0',
      }),
      f.el('span', 'tri-zone-range', `${zoneRange(bounds, i)}${unit}`),
      f.el('span', 'tri-zone-time', zoneClock(times[i])),
      f.el('span', 'tri-zone-pct', `${((times[i] / total) * 100).toFixed(1)}%`),
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
  const wrap = f.el('div', 'tri-zone')
  f.add(
    wrap,
    f.el('div', 'tri-zone-title', '25W power distribution', {
      'data-i18n': '25W power distribution',
    }),
  )
  const H = 34
  const n = hist.length
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
  const histMaxWatt = n * 25
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

export const buildPowerCurve = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  ctx: DetailCtx,
): N | null => {
  const curve = d.powerCurve
  if (!curve || curve.length < 2) return null
  const sixWeekRef = ctx.curveRef
  const yearRef = ctx.curveYearRef
  const ftpRef = ctx.ftp
  const goalRef = ctx.goalFtp
  const wrap = f.el('div', 'tri-zone')
  const W = 100
  const H = 34
  const secs = curve.map(c => c.s)
  const visibleSixWeekRef = sixWeekRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const visibleYearRef = yearRef.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const defaultRange = visibleSixWeekRef.length > 0 ? 'six-weeks' : 'year'
  const visibleRef = defaultRange === 'six-weeks' ? visibleSixWeekRef : visibleYearRef
  const head = f.el('div', 'tri-curve-head')
  f.add(head, f.el('div', 'tri-zone-title', 'power curve', { 'data-i18n': 'power curve' }))
  if (ctx.curveYear != null && visibleYearRef.length > 0) {
    const ranges = f.el('div', 'tri-curve-ranges', undefined, {
      role: 'group',
      'aria-label': 'comparison range',
      'data-i18n-aria-label': 'comparison range',
    })
    const sixWeekAttrs: Record<string, string> = {
      type: 'button',
      'data-curve-range': 'six-weeks',
      'aria-pressed': String(defaultRange === 'six-weeks'),
      'data-i18n': '6 weeks',
    }
    if (visibleSixWeekRef.length === 0) sixWeekAttrs.disabled = ''
    const yearButton = f.el('button', 'tri-curve-range', undefined, {
      type: 'button',
      'data-curve-range': 'year',
      'aria-pressed': String(defaultRange === 'year'),
    })
    f.add(
      yearButton,
      f.el('span', undefined, 'all of', { 'data-i18n': 'all of' }),
      f.el('span', undefined, ` ${ctx.curveYear}`),
    )
    f.add(ranges, f.el('button', 'tri-curve-range', '6 weeks', sixWeekAttrs), yearButton)
    f.add(head, ranges)
  }
  f.add(wrap, head)
  const observedMaxW = Math.max(
    1,
    ...curve.map(c => c.w),
    ...visibleSixWeekRef.map(c => c.w),
    ...visibleYearRef.map(c => c.w),
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
  const curveDurTicks = [1, 5, 30, 60, 300, 1200, 3600, 10_800].filter(
    sec => sec >= secs[0] && sec <= secs[secs.length - 1],
  )
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
  const readout = f.el('div', 'tri-chart-readout tri-curve-readout', undefined, {
    'aria-hidden': 'true',
  })
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
    const referenceRow = f.el('span', 'tri-curve-readout-row tri-curve-readout-row--ref')
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
        cls:
          idx === 0
            ? 'tri-cax-xt--first'
            : sec === secs[secs.length - 1]
              ? 'tri-cax-xt--last'
              : undefined,
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
  if (ftpRef != null) f.add(cap, f.el('span', 'tri-ana-k tri-curve-ftp-k', `FTP ${ftpRef}W`))
  if (goalRef != null) f.add(cap, f.el('span', 'tri-ana-k tri-curve-goal-k', `goal ${goalRef}W`))
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

export const activityStatRows = (d: StravaActivityDetail): [string, string][] => {
  if (d.sport === 'strength' || d.sport === 'treatment' || d.sport === 'yoga')
    return [['time', dur(d.movingTimeS)]]
  const activityRate =
    d.sport === 'swim' && positiveMetric(d.swimPaceSPer100m)
      ? `${clock(d.swimPaceSPer100m)} /100m`
      : rate(d.sport, d.distanceKm, d.movingTimeS)
  const rows: [string, string][] = [
    ['distance', dist(d.distanceKm, d.sport)],
    ['time', dur(d.movingTimeS)],
    [d.sport === 'bike' ? 'speed' : 'pace', activityRate],
  ]
  if (d.sport === 'bike' && d.maxSpeedKph != null) rows.push(['max speed', speedKph(d.maxSpeedKph)])
  if (d.sport === 'run') {
    const trend = runTrendRow(d.distanceKm, d.movingTimeS)
    if (trend) rows.push(trend)
  }
  if (d.sport === 'swim' && positiveMetric(d.strokeRateSpm))
    rows.push(['stroke rate', `${swimTrendNumber(d.strokeRateSpm)} str/min`])
  if (d.sport === 'swim' && positiveMetric(d.strokeCount))
    rows.push(['strokes', Math.round(d.strokeCount).toLocaleString('en-US')])
  if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
  return rows
}

export const buildActivity = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  expanded = false,
  ctx?: DetailCtx,
  swimTrend: SwimTrendPoint[] = [],
): N => {
  const wrap = f.el('section', expanded ? 'tri-act tri-act--expanded' : 'tri-act', undefined, {
    'data-activity-id': `${d.id}`,
  })
  const head = f.el('div', 'tri-act-head')
  f.add(head, buildIcon(f, d.sport))
  f.add(wrap, head)
  f.add(wrap, statsTable(f, [...activityStatRows(d), ...moreStatRows(d)]))
  if (d.fueling) {
    const fueling = buildFueling(f, d.fueling)
    if (fueling) f.add(wrap, fueling)
  }
  const analysis = buildAnalysisBar(f, d)
  const analysisSelection = null
  if (d.route.length >= 2) {
    const secondary =
      d.sport === 'swim' ? buildSwimStrokes(f, d) : buildElevation(f, d, analysisSelection)
    const figs = f.el(
      'div',
      `tri-act-figs tri-act-figs--route${secondary ? ' tri-act-figs--split' : ''}`,
    )
    f.add(figs, buildRoute(f, d.route))
    if (secondary) f.add(figs, secondary)
    if (analysis) f.add(figs, analysis)
    f.add(wrap, figs)
  } else if (d.sport === 'swim') {
    const figs = f.el('div', 'tri-act-figs')
    f.add(figs, buildPool(f, d))
    f.add(wrap, figs)
  }
  const swimTrends = buildSwimTrends(f, d, swimTrend)
  if (hasMoreSection(d) || swimTrends) {
    const moreId = `tri-act-more-${d.id}`
    const more = f.el('div', 'tri-act-more', undefined, { id: moreId })
    const flags = routeStreamFlags(d)
    const runSplits = buildRunLapSplits(f, d)
    if (runSplits) f.add(more, runSplits)
    if (flags.hr)
      f.add(
        more,
        buildTrace(
          f,
          d,
          point => point.hr,
          'hr',
          max => `${max} bpm peak`,
          value => `${Math.round(value)}bpm`,
          undefined,
          analysisSelection,
        ),
      )
    if (flags.power)
      f.add(
        more,
        buildTrace(
          f,
          d,
          point => point.w,
          'power',
          max => `${max} W peak`,
          value => `${Math.round(value)}w`,
          undefined,
          analysisSelection,
        ),
      )
    if (flags.cad) {
      const cadenceScale = d.sport === 'run' ? 2 : 1
      const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
      f.add(
        more,
        buildTrace(
          f,
          d,
          point => point.cad * cadenceScale,
          'cadence',
          max => `${max} ${cadenceUnit} peak`,
          value => `${Math.round(value)}${cadenceUnit}`,
          undefined,
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
    if (flags.temp) f.add(more, buildTemperatureTrace(f, d, analysisSelection))
    if (ctx) {
      const zones = zoneDuo(f, buildHrZones(f, d, ctx), buildPowerZones(f, d, ctx))
      if (zones) f.add(more, zones)
      const charts = zoneDuo(f, buildPowerCurve(f, d, ctx), buildPowerHist(f, d))
      if (charts) f.add(more, charts)
    }
    const bestEfforts = buildCyclingBestEfforts(f, d)
    if (bestEfforts) f.add(more, bestEfforts)
    if (swimTrends) f.add(more, swimTrends)
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

export const dayDetails = (payload: DayCardPayload, dateIso: string): StravaActivityDetail[] =>
  Object.values(payload.details)
    .filter(d => d.date === dateIso)
    .sort((a, b) => b.distanceKm - a.distanceKm)

export const recentLocation = (payload: DayCardPayload): string | undefined =>
  Object.values(payload.details)
    .sort((a, b) => b.date.localeCompare(a.date))
    .find(d => d.location)?.location ?? undefined

export const buildDayCard = <N>(
  f: TriNodeFactory<N>,
  dateIso: string,
  payload: DayCardPayload | null,
  extras: DayCardExtras = {},
  activity?: (d: StravaActivityDetail) => N,
  ctx?: DetailCtx,
): N => {
  const render =
    activity ??
    ((d: StravaActivityDetail) =>
      buildActivity(f, d, !!extras.sport || !!extras.expanded, ctx, payload?.swimTrend ?? []))
  const card = f.el('div', 'tri-pop-card')
  const head = f.el('div', 'tri-pop-head')
  f.add(
    head,
    extras.dateHref
      ? f.el('a', 'tri-pop-date', prettyDate(dateIso), { href: extras.dateHref })
      : f.el('span', 'tri-pop-date', prettyDate(dateIso)),
  )
  const allDay = payload ? dayDetails(payload, dateIso) : []
  const day = extras.sport ? allDay.filter(d => d.sport === extras.sport) : allDay
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
  if (!payload) {
    f.add(card, f.el('div', 'tri-pop-rest', '·'))
  } else if (day.length === 0) {
    const rest = f.el('div', 'tri-pop-rest')
    if (extras.sport) {
      f.add(rest, f.el('span', 'tri-pop-rest-label', `no ${extras.sport}`))
    } else {
      f.add(rest, buildBattery(f), f.el('span', 'tri-pop-rest-label', 'rest'))
    }
    f.add(card, rest)
  } else {
    for (const d of day) f.add(card, render(d))
  }
  if (!extras.sport) {
    const dh = payload?.health[dateIso]
    if (dh) {
      const rec = buildRecovery(f, dh)
      if (rec) f.add(card, rec)
    }
  }
  return card
}
