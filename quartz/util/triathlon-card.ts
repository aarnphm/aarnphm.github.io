import { STROKE_LABEL, SWIM_STROKES } from '../plugins/stores/apple'
import {
  SPORT_ICON,
  type ActivityHealth,
  type ActivityKind,
  type PowerCurvePoint,
  type StravaActivityDetail,
  type StravaZones,
} from '../plugins/stores/strava'

export interface TriNodeFactory<N> {
  el: (tag: string, cls?: string, text?: string, attrs?: Record<string, string>) => N
  svg: (tag: string, attrs: Record<string, string | number>) => N
  add: (parent: N, ...children: N[]) => void
}

export type DayCardExtras = { location?: string; event?: string; sport?: ActivityKind }

export type DayCardPayload = {
  details: Record<string, StravaActivityDetail>
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

export const rate = (sport: ActivityKind, km: number, s: number): string => {
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  if (imperial) {
    const mi = km * KM_TO_MI
    return sport === 'bike' ? `${(mi / (s / 3600)).toFixed(1)} mph` : `${clock(s / mi)} /mi`
  }
  return sport === 'bike' ? `${(km / (s / 3600)).toFixed(1)} km/h` : `${clock(s / km)} /km`
}

export const scrubDist = (km: number, sport: ActivityKind): string =>
  sport === 'swim'
    ? `${Math.round(km * 1000).toLocaleString('en-US')} m`
    : imperial
      ? `${(km * KM_TO_MI).toFixed(2)} mi`
      : `${km.toFixed(2)} km`

const elevationValue = (meters: number): number => (imperial ? meters * M_TO_FT : meters)

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

export const formatGarminSource = (value: string | null): string => {
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
  if (rows.length > 0) rows.push(['source', formatGarminSource(f.sourceDevice)])
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
  if (d.avgTemp != null) rows.push(['temp', `${Math.round((d.avgTemp * 9) / 5 + 32)}°F`])
  if (d.windKph != null)
    rows.push([
      'wind',
      `${d.windKph} km/h${d.windDir ? ` ${d.windDir}` : ''}${d.windGustKph != null ? ` / gust ${d.windGustKph}` : ''}`,
    ])
  return rows
}

export const routeStreamFlags = (
  d: StravaActivityDetail,
): { power: boolean; hr: boolean; cad: boolean } => ({
  power: d.deviceWatts && d.route.some(p => p.w > 0),
  hr: d.route.some(p => p.hr > 0),
  cad: d.route.some(p => p.cad > 0),
})

export const hasMoreSection = (d: StravaActivityDetail): boolean => {
  const flags = routeStreamFlags(d)
  const efforts = d.bestEfforts
  return (
    moreStatRows(d).length > 0 ||
    flags.power ||
    flags.hr ||
    flags.cad ||
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

export const buildRoute = <N>(f: TriNodeFactory<N>, route: StravaActivityDetail['route']): N => {
  const pad = 6
  const span = 100 - pad * 2
  let d = ''
  route.forEach((p, i) => {
    d += `${i ? 'L' : 'M'} ${(pad + p.x * span).toFixed(2)} ${(pad + (1 - p.y) * span).toFixed(2)} `
  })
  const fig = f.svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  f.add(fig, f.svg('path', { d, class: 'tri-route-path' }))
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

export const buildElevation = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
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
  f.add(wrap, axisFrame(f, fig, yTicks, h, xTicks, true, { top: 0, bottom: h }), cap)
  return wrap
}

export const buildTrace = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  title: string,
  cap: (max: number) => string,
  tick: (value: number) => string,
): N => {
  const w = 100
  const h = 30
  const maxD = d.route[d.route.length - 1].d || 1
  let max = 1
  d.route.forEach((p, i) => {
    const v = pick(p, i)
    if (v > max) max = v
  })
  const px = (km: number): number => (km / maxD) * w
  const py = (v: number): number => h - (v / max) * (h - 1)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    const v = pick(p, i)
    area += `L ${px(p.d).toFixed(2)} ${py(v).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(v).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const yTicks = niceTicks(0, max, 3).map(value => ({
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
  f.add(s, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(s, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap')
  const capEl = f.el('div', 'tri-elev-cap')
  f.add(capEl, f.el('span', 'tri-elev-d', title), f.el('span', 'tri-elev-range', cap(max)))
  f.add(wrap, capEl, axisFrame(f, s, yTicks, h, distanceXTicks(maxD), true, { top: 0, bottom: h }))
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

export const powerCurveFraction = (
  seconds: number,
  minSeconds: number,
  maxSeconds: number,
): number => {
  if (minSeconds <= 0 || maxSeconds <= minSeconds) return 0
  const value = Math.min(maxSeconds, Math.max(minSeconds, seconds))
  return (Math.log(value) - Math.log(minSeconds)) / (Math.log(maxSeconds) - Math.log(minSeconds))
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
  let index = 0
  let distance = Infinity
  curve.forEach((point, candidate) => {
    const candidateDistance = Math.abs(
      powerCurveFraction(point.s, minSeconds, maxSeconds) - fraction,
    )
    if (candidateDistance >= distance) return
    index = candidate
    distance = candidateDistance
  })
  const point = curve[index]
  return {
    index,
    durationS: point.s,
    watts: point.w,
    referenceWatts: reference.find(candidate => candidate.s === point.s)?.w ?? null,
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
  f.add(stage, svgEl)
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
  const ref = ctx.curveRef
  const ftpRef = ctx.ftp
  const goalRef = ctx.goalFtp
  const wrap = f.el('div', 'tri-zone')
  f.add(wrap, f.el('div', 'tri-zone-title', 'power curve', { 'data-i18n': 'power curve' }))
  const W = 100
  const H = 34
  const secs = curve.map(c => c.s)
  const maxW = Math.max(1, ...curve.map(c => c.w), ...ref.map(c => c.w), ftpRef ?? 0, goalRef ?? 0)
  const X = (sec: number): number => powerCurveFraction(sec, secs[0], secs[secs.length - 1]) * W
  const Y = (w: number): number => H - (w / maxW) * (H - 1)
  const toPath = (pts: PowerCurvePoint[]): string =>
    pts.map((c, i) => `${i ? 'L' : 'M'} ${X(c.s).toFixed(2)} ${Y(c.w).toFixed(2)}`).join(' ')
  const visibleRef = ref.filter(c => c.s >= secs[0] && c.s <= secs[secs.length - 1])
  const initialValueText = `${dlabel(curve[0].s)} · ${curve[0].w.toLocaleString('en-US')} W`
  const s = f.svg('svg', {
    class: 'tri-curve-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    'data-curve': JSON.stringify(curve),
    'data-curve-ref': JSON.stringify(visibleRef),
    'data-i18n-aria-label': 'power curve',
    role: 'slider',
    tabindex: 0,
    'aria-label': 'power curve',
    'aria-orientation': 'horizontal',
    'aria-readonly': 'true',
    'aria-valuemin': curve[0].s,
    'aria-valuemax': curve[curve.length - 1].s,
    'aria-valuenow': curve[0].s,
    'aria-valuetext': initialValueText,
  })
  if (visibleRef.length >= 2)
    f.add(s, f.svg('path', { d: toPath(visibleRef), class: 'tri-curve-ref' }))
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
  const curveDurTicks = [1, 5, 30, 60, 300, 1200, 3600].filter(
    sec => sec >= secs[0] && sec <= secs[secs.length - 1],
  )
  const curveStep = niceStep(maxW, 4)
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      niceTicks(0, maxW, 4).map(value => ({
        label: value === 0 ? '0' : `${axisNumber(value, curveStep)}w`,
        vbY: Y(value),
      })),
      H,
      curveDurTicks.map((sec, idx) => ({
        label: dlabel(sec),
        pct: X(sec),
        cls: idx === 0 ? 'tri-cax-xt--first' : undefined,
      })),
    ),
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
  if (visibleRef.length >= 2) {
    const referenceRow = f.el('span', 'tri-curve-readout-row tri-curve-readout-row--ref')
    f.add(
      referenceRow,
      f.el('span', 'tri-curve-readout-swatch tri-curve-readout-swatch--ref', undefined, {
        'aria-hidden': 'true',
      }),
      f.el('strong', 'tri-curve-readout-value tri-curve-readout-value--ref'),
      f.el('span', 'tri-curve-readout-label', '6-week best', { 'data-i18n': '6-week best' }),
    )
    f.add(readout, referenceRow)
  }
  f.add(wrap, readout)
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

export const buildActivity = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  expanded = false,
  ctx?: DetailCtx,
): N => {
  const wrap = f.el('section', expanded ? 'tri-act tri-act--expanded' : 'tri-act')
  const head = f.el('div', 'tri-act-head')
  f.add(head, buildIcon(f, d.sport))
  f.add(wrap, head)
  const rows: [string, string][] =
    d.sport === 'strength' || d.sport === 'treatment' || d.sport === 'yoga'
      ? [['time', dur(d.movingTimeS)]]
      : [
          ['distance', dist(d.distanceKm, d.sport)],
          ['time', dur(d.movingTimeS)],
          [d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)],
        ]
  if (d.avgHr) rows.push(['avg hr', `${d.avgHr} bpm`])
  f.add(wrap, statsTable(f, rows))
  if (d.fueling) {
    const fueling = buildFueling(f, d.fueling)
    if (fueling) f.add(wrap, fueling)
  }
  if (d.route.length >= 2) {
    const secondary = d.sport === 'swim' ? buildSwimStrokes(f, d) : buildElevation(f, d)
    const figs = f.el(
      'div',
      `tri-act-figs tri-act-figs--route${secondary ? ' tri-act-figs--split' : ''}`,
    )
    f.add(figs, buildRoute(f, d.route))
    if (secondary) f.add(figs, secondary)
    f.add(wrap, figs)
  } else if (d.sport === 'swim') {
    const figs = f.el('div', 'tri-act-figs')
    f.add(figs, buildPool(f, d))
    f.add(wrap, figs)
  }
  if (hasMoreSection(d)) {
    const more = f.el('div', 'tri-act-more')
    const rows = moreStatRows(d)
    if (rows.length > 0) f.add(more, statsTable(f, rows))
    if (ctx) {
      const zones = zoneDuo(f, buildHrZones(f, d, ctx), buildPowerZones(f, d, ctx))
      if (zones) f.add(more, zones)
      const charts = zoneDuo(f, buildPowerCurve(f, d, ctx), buildPowerHist(f, d))
      if (charts) f.add(more, charts)
    }
    const bestEfforts = buildCyclingBestEfforts(f, d)
    if (bestEfforts) f.add(more, bestEfforts)
    f.add(wrap, f.el('button', 'tri-act-toggle', undefined, { type: 'button' }), more)
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
  const render = activity ?? ((d: StravaActivityDetail) => buildActivity(f, d, !!extras.sport, ctx))
  const card = f.el('div', 'tri-pop-card')
  const head = f.el('div', 'tri-pop-head')
  f.add(head, f.el('span', 'tri-pop-date', prettyDate(dateIso)))
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
