import {
  SPORT_ICON,
  type ActivityHealth,
  type ActivityKind,
  type StravaActivityDetail,
} from '../plugins/stores/strava'

export interface TriNodeFactory<N> {
  el: (tag: string, cls?: string, text?: string, attrs?: Record<string, string>) => N
  svg: (tag: string, attrs: Record<string, string | number>) => N
  add: (parent: N, ...children: N[]) => void
}

export type DayCardExtras = { location?: string; event?: string; weightLbs?: number }

export type DayCardPayload = {
  details: Record<string, StravaActivityDetail>
  health: Record<string, ActivityHealth>
}

export type ActivityFueling = NonNullable<StravaActivityDetail['fueling']>

export const KM_TO_MI = 0.621371
export const FT_PER_KM = 3280.84

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export const dist = (km: number, sport: ActivityKind): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  const mi = km * KM_TO_MI
  return mi < 1 ? `${Math.round(km * FT_PER_KM)} ft` : `${mi.toFixed(1)} mi`
}

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
  const mi = km * KM_TO_MI
  if (sport === 'bike') return `${(mi / (s / 3600)).toFixed(1)} mph`
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  return `${clock(s / mi)} /mi`
}

export const scrubDist = (km: number, sport: ActivityKind): string =>
  sport === 'swim'
    ? `${Math.round(km * 1000).toLocaleString('en-US')} m`
    : `${(km * KM_TO_MI).toFixed(2)} mi`

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
  if (d.avgTemp != null) rows.push(['temp', `${d.avgTemp}°C`])
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
  const icon = f.svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
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

export const buildActivity = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N => {
  const wrap = f.el('section', 'tri-act')
  const head = f.el('div', 'tri-act-head')
  f.add(head, buildIcon(f, d.sport))
  f.add(wrap, head)
  const rows: [string, string][] =
    d.sport === 'strength'
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
  if (d.sport === 'swim') {
    const figs = f.el('div', 'tri-act-figs')
    f.add(figs, buildPool(f, d))
    f.add(wrap, figs)
  } else if (d.route.length >= 2) {
    const figs = f.el('div', 'tri-act-figs')
    f.add(figs, buildRoute(f, d.route), buildElevation(f, d))
    f.add(wrap, figs)
  }
  if (hasMoreSection(d)) {
    const more = f.el('div', 'tri-act-more')
    const rows = moreStatRows(d)
    if (rows.length > 0) f.add(more, statsTable(f, rows))
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
): N => {
  const render = activity ?? ((d: StravaActivityDetail) => buildActivity(f, d))
  const card = f.el('div', 'tri-pop-card')
  const head = f.el('div', 'tri-pop-head')
  f.add(head, f.el('span', 'tri-pop-date', prettyDate(dateIso)))
  const day = payload ? dayDetails(payload, dateIso) : []
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
  f.add(card, head)
  if (extras.event || extras.weightLbs != null) {
    const track = f.el('div', 'tri-pop-track')
    if (extras.event) f.add(track, f.el('span', 'tri-pop-race', extras.event))
    if (extras.weightLbs != null) {
      f.add(track, f.el('span', 'tri-pop-weight', `${extras.weightLbs} lbs`))
    }
    f.add(card, track)
  }
  if (!payload) {
    f.add(card, f.el('div', 'tri-pop-rest', '·'))
  } else if (day.length === 0) {
    const rest = f.el('div', 'tri-pop-rest')
    f.add(rest, buildBattery(f), f.el('span', 'tri-pop-rest-label', 'rest'))
    f.add(card, rest)
  } else {
    for (const d of day) f.add(card, render(d))
  }
  const dh = payload?.health[dateIso]
  if (dh) {
    const rec = buildRecovery(f, dh)
    if (rec) f.add(card, rec)
  }
  return card
}
