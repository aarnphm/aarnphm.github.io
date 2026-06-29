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

export const clock = (s: number): string =>
  `${Math.floor(s / 60)}:${Math.round(s % 60)
    .toString()
    .padStart(2, '0')}`

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
  return (
    moreStatRows(d).length > 0 ||
    flags.power ||
    flags.hr ||
    flags.cad ||
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

export const buildElevation = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
  const w = 100
  const h = 30
  const pad = 2
  const maxD = d.route[d.route.length - 1].d || 1
  const altSpan = Math.max(1, d.maxAlt - d.minAlt)
  const px = (km: number): number => (km / maxD) * w
  const py = (alt: number): number => h - pad - ((alt - d.minAlt) / altSpan) * (h - 2 * pad)
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
  f.add(fig, f.svg('path', { d: area, class: 'tri-elev-area' }))
  f.add(fig, f.svg('path', { d: line, class: 'tri-elev-line' }))
  f.add(fig, f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = f.el('div', 'tri-elev-wrap')
  const cap = f.el('div', 'tri-elev-cap')
  f.add(
    cap,
    f.el('span', 'tri-elev-d', `+${d.elevationM} m`),
    f.el('span', 'tri-elev-d', `−${d.descentM} m`),
    f.el('span', 'tri-elev-range', `${Math.round(d.minAlt)}–${Math.round(d.maxAlt)} m`),
  )
  f.add(wrap, fig, cap)
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

export const axisFrame = <N>(
  f: TriNodeFactory<N>,
  svgEl: N,
  yTicks: { label: string; vbY: number }[],
  vbH: number,
  xTicks: AxisXTick[],
  cssH: number,
): N => {
  const frame = f.el('div', 'tri-cax-frame')
  const yax = f.el('div', 'tri-cax-yax', undefined, { style: `height:${cssH}px` })
  for (const t of yTicks)
    f.add(
      yax,
      f.el('span', 'tri-cax-yt', t.label, { style: `top:${((t.vbY / vbH) * 100).toFixed(2)}%` }),
    )
  const plot = f.el('div', 'tri-cax-plot')
  f.add(plot, svgEl)
  const xax = f.el('div', 'tri-cax-xax')
  for (const t of xTicks)
    f.add(
      xax,
      f.el('span', `tri-cax-xt${t.cls ? ` ${t.cls}` : ''}`, t.label, {
        style: `left:${t.pct.toFixed(2)}%`,
      }),
    )
  f.add(plot, xax)
  f.add(frame, yax, plot)
  return frame
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
      60,
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
  const minLog = Math.log(secs[0])
  const maxLog = Math.log(secs[secs.length - 1])
  const span = Math.max(1e-6, maxLog - minLog)
  const X = (sec: number): number => ((Math.log(sec) - minLog) / span) * W
  const Y = (w: number): number => H - (w / maxW) * (H - 1)
  const toPath = (pts: PowerCurvePoint[]): string =>
    pts.map((c, i) => `${i ? 'L' : 'M'} ${X(c.s).toFixed(2)} ${Y(c.w).toFixed(2)}`).join(' ')
  const s = f.svg('svg', {
    class: 'tri-curve-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
    'data-curve': JSON.stringify(curve),
  })
  if (ref.length >= 2)
    f.add(
      s,
      f.svg('path', {
        d: toPath(ref.filter(c => c.s <= secs[secs.length - 1])),
        class: 'tri-curve-ref',
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
  const curveDurTicks = [1, 5, 30, 60, 300, 1200, 3600].filter(
    sec => sec >= secs[0] && sec <= secs[secs.length - 1],
  )
  f.add(
    wrap,
    axisFrame(
      f,
      s,
      [
        { label: `${Math.round(maxW)}w`, vbY: Y(maxW) },
        { label: `${Math.round(maxW / 2)}w`, vbY: Y(maxW / 2) },
        { label: '0', vbY: Y(0) },
      ],
      H,
      curveDurTicks.map((sec, idx) => ({
        label: dlabel(sec),
        pct: X(sec),
        cls: idx === 0 ? 'tri-cax-xt--first' : undefined,
      })),
      64,
    ),
  )
  f.add(wrap, f.el('div', 'tri-chart-readout'))
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
    const figs = f.el('div', 'tri-act-figs')
    f.add(figs, buildRoute(f, d.route))
    if (d.sport === 'swim') {
      const strokes = buildSwimStrokes(f, d)
      if (strokes) f.add(figs, strokes)
    } else {
      f.add(figs, buildElevation(f, d))
    }
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
      const hrz = buildHrZones(f, d, ctx)
      if (hrz) f.add(more, hrz)
      const pcurve = buildPowerCurve(f, d, ctx)
      if (pcurve) f.add(more, pcurve)
      const pz = buildPowerZones(f, d, ctx)
      if (pz) f.add(more, pz)
      const phist = buildPowerHist(f, d)
      if (phist) f.add(more, phist)
    }
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
