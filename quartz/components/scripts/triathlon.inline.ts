import type { RoughAnnotation } from 'rough-notation/lib/model'
import katex from 'katex'
import { annotate } from 'rough-notation'
import type { Analytics, BodyBlock, DailyPoint } from '../../plugins/stores/analytics'
import {
  type ActivityHealth,
  type PowerCurvePoint,
  type Sport,
  SPORT_ICON,
  type StravaActivityDetail,
  type StravaZones,
} from '../../plugins/stores/strava'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

export {}

type DetailPayload = {
  details: Record<string, StravaActivityDetail>
  health: Record<string, ActivityHealth>
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
}

let DETAIL_ZONES: StravaZones | null = null
let DETAIL_CURVE_REF: PowerCurvePoint[] = []

const SVGNS = 'http://www.w3.org/2000/svg'
const KM_TO_MI = 0.621371
const FT_PER_KM = 3280.84

const el = (tag: string, cls?: string, text?: string): HTMLElement => {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

const svg = (tag: string, attrs: Record<string, string | number>): SVGElement => {
  const e = document.createElementNS(SVGNS, tag)
  for (const k in attrs) e.setAttribute(k, String(attrs[k]))
  return e
}

const dist = (km: number, sport: Sport): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  const mi = km * KM_TO_MI
  return mi < 1 ? `${Math.round(km * FT_PER_KM)} ft` : `${mi.toFixed(1)} mi`
}
const dur = (s: number): string => {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h > 0 ? `${h}h${m.toString().padStart(2, '0')}'` : `${m}'`
}
const clock = (s: number): string =>
  `${Math.floor(s / 60)}:${Math.round(s % 60)
    .toString()
    .padStart(2, '0')}`
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const shortDate = (iso: string): string => {
  const [, m, d] = iso.split('-').map(Number)
  return `${MONTHS[(m || 1) - 1]} ${d || 1}`
}
const rate = (sport: Sport, km: number, s: number): string => {
  const mi = km * KM_TO_MI
  if (sport === 'bike') return `${(mi / (s / 3600)).toFixed(1)} mph`
  if (sport === 'swim') return `${clock(s / (km * 10))} /100m`
  return `${clock(s / mi)} /mi`
}

const buildIcon = (sport: Sport): SVGElement => {
  const s = svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of SPORT_ICON[sport]) s.appendChild(svg('path', { d }))
  return s
}

const BATTERY = [
  'M23 10V14',
  'M1 16V8C1 6.89543 1.89543 6 3 6H18C19.1046 6 20 6.89543 20 8V16C20 17.1046 19.1046 18 18 18H3C1.89543 18 1 17.1046 1 16Z',
  'M10.1667 9L8.5 12H12.5L10.8333 15',
]

const buildBattery = (): SVGElement => {
  const s = svg('svg', { class: 'tri-ico tri-battery', viewBox: '0 0 24 24', fill: 'none' })
  for (const d of BATTERY) s.appendChild(svg('path', { d }))
  return s
}

const buildRoute = (route: StravaActivityDetail['route']): SVGElement => {
  const pad = 6
  const span = 100 - pad * 2
  let d = ''
  route.forEach((p, i) => {
    d += `${i ? 'L' : 'M'} ${(pad + p.x * span).toFixed(2)} ${(pad + (1 - p.y) * span).toFixed(2)} `
  })
  const s = svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(svg('path', { d, class: 'tri-route-path' }))
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return s
}

const buildElevation = (d: StravaActivityDetail): HTMLElement => {
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
  const s = svg('svg', { class: 'tri-elev', viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none' })
  s.appendChild(svg('path', { d: area, class: 'tri-elev-area' }))
  s.appendChild(svg('path', { d: line, class: 'tri-elev-line' }))
  s.appendChild(svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = el('div', 'tri-elev-wrap')
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-elev-d', `+${d.elevationM} m`),
    el('span', 'tri-elev-d', `−${d.descentM} m`),
    el('span', 'tri-elev-range', `${Math.round(d.minAlt)}–${Math.round(d.maxAlt)} m`),
  )
  wrap.append(s, cap)
  return wrap
}

const buildPool = (d: StravaActivityDetail): HTMLElement => {
  const lengths = Math.max(1, Math.round((d.distanceKm * 1000) / 25))
  const wrap = el('div', 'tri-pool-wrap')
  const s = svg('svg', {
    class: 'tri-route tri-pool',
    viewBox: '0 0 100 56',
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(
    svg('rect', { x: 6, y: 12, width: 88, height: 32, rx: 16, ry: 16, class: 'tri-pool-lane' }),
  )
  s.appendChild(svg('line', { x1: 22, y1: 28, x2: 78, y2: 28, class: 'tri-pool-mid' }))
  wrap.append(s, el('span', 'tri-pool-cap', `${lengths} × 25m`))
  return wrap
}

const buildTrace = (
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  title: string,
  cap: (max: number) => string,
): HTMLElement => {
  const w = 100
  const h = 30
  const pad = 2
  const maxD = d.route[d.route.length - 1].d || 1
  let max = 1
  d.route.forEach((p, i) => {
    const v = pick(p, i)
    if (v > max) max = v
  })
  const px = (km: number): number => (km / maxD) * w
  const py = (v: number): number => h - pad - (v / max) * (h - 2 * pad)
  let area = `M 0 ${h} `
  let line = ''
  d.route.forEach((p, i) => {
    const v = pick(p, i)
    area += `L ${px(p.d).toFixed(2)} ${py(v).toFixed(2)} `
    line += `${i ? 'L' : 'M'} ${px(p.d).toFixed(2)} ${py(v).toFixed(2)} `
  })
  area += `L ${w} ${h} Z`
  const s = svg('svg', { class: 'tri-elev', viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none' })
  s.appendChild(svg('path', { d: area, class: 'tri-elev-area' }))
  s.appendChild(svg('path', { d: line, class: 'tri-elev-line' }))
  s.appendChild(svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: h }))
  const wrap = el('div', 'tri-elev-wrap')
  const cap2 = el('div', 'tri-elev-cap')
  cap2.append(el('span', 'tri-elev-d', title), el('span', 'tri-elev-range', cap(max)))
  wrap.append(s, cap2)
  return wrap
}

const statRow = (label: string, value: string): HTMLElement => {
  const tr = document.createElement('tr')
  tr.append(el('th', 'tri-act-stat-k', label), el('td', 'tri-act-stat-v', value))
  return tr
}

const recoveryRows = (h: ActivityHealth): [string, string][] => {
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

const buildRecovery = (h: ActivityHealth): HTMLElement | null => {
  const rows = recoveryRows(h)
  if (rows.length === 0) return null
  const wrap = el('div', 'tri-act-health')
  wrap.appendChild(el('span', 'tri-act-health-h', 'recovery'))
  const table = el('table', 'tri-act-stats')
  const tbody = document.createElement('tbody')
  for (const [k, v] of rows) tbody.appendChild(statRow(k, v))
  table.appendChild(tbody)
  wrap.appendChild(table)
  return wrap
}

type ActivityFueling = NonNullable<StravaActivityDetail['fueling']>

const formatMl = (value: number): string => {
  if (value < 1000) return `${Math.round(value)} ml`
  const liters = value / 1000
  return `${liters >= 10 ? liters.toFixed(0) : liters.toFixed(1)} L`
}

const formatGarminSource = (value: string | null): string => {
  const clean = value?.trim()
  if (!clean) return 'Garmin'
  return clean.toLowerCase().includes('garmin') ? clean : `Garmin ${clean}`
}

const fuelingRows = (f: ActivityFueling): [string, string][] => {
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

const buildFueling = (f: ActivityFueling): HTMLElement | null => {
  const rows = fuelingRows(f)
  if (rows.length === 0) return null
  const wrap = el('div', 'tri-act-health tri-act-fueling')
  wrap.appendChild(el('span', 'tri-act-health-h', 'fueling'))
  const table = el('table', 'tri-act-stats')
  const tbody = document.createElement('tbody')
  for (const [k, v] of rows) tbody.appendChild(statRow(k, v))
  table.appendChild(tbody)
  wrap.appendChild(table)
  return wrap
}

const gradeAt = (route: StravaActivityDetail['route'], i: number): number => {
  const j0 = Math.max(0, i - 2)
  const j1 = Math.min(route.length - 1, i + 2)
  const dKm = route[j1].d - route[j0].d
  return dKm > 0 ? ((route[j1].alt - route[j0].alt) / (dKm * 1000)) * 100 : 0
}

const scrubDist = (km: number, sport: Sport): string =>
  sport === 'swim'
    ? `${Math.round(km * 1000).toLocaleString('en-US')} m`
    : `${(km * KM_TO_MI).toFixed(2)} mi`

type ScrubSurface = {
  wrap: HTMLElement
  fmt: (p: StravaActivityDetail['route'][number], i: number) => string
}

const linkScrub = (
  act: HTMLElement,
  marker: SVGElement | null,
  surfaces: ScrubSurface[],
  route: StravaActivityDetail['route'],
): void => {
  const maxD = route[route.length - 1].d || 1
  const pad = 6
  const span = 88
  const resolved: {
    wrap: HTMLElement
    svgEl: SVGElement
    cursor: SVGElement
    readout: HTMLElement
    fmt: ScrubSurface['fmt']
  }[] = []
  for (const s of surfaces) {
    const svgEl = s.wrap.querySelector<SVGElement>('.tri-elev')
    const cursor = svgEl?.querySelector<SVGElement>('.tri-elev-cursor')
    if (!svgEl || !cursor) continue
    const readout = el('div', 'tri-fig-readout')
    s.wrap.appendChild(readout)
    resolved.push({ wrap: s.wrap, svgEl, cursor, readout, fmt: s.fmt })
  }
  if (resolved.length === 0) return
  const indexAt = (clientX: number, svgEl: SVGElement): number => {
    const r = svgEl.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (clientX - r.left) / r.width))
    const targetD = frac * maxD
    let i = 0
    let best = Infinity
    for (let k = 0; k < route.length; k++) {
      const dd = Math.abs(route[k].d - targetD)
      if (dd < best) {
        best = dd
        i = k
      }
    }
    return i
  }
  for (const surf of resolved) {
    const onMove = (event: MouseEvent) => {
      const i = indexAt(event.clientX, surf.svgEl)
      const p = route[i]
      const x = ((p.d / maxD) * 100).toFixed(2)
      for (const r of resolved) {
        r.cursor.setAttribute('x1', x)
        r.cursor.setAttribute('x2', x)
      }
      if (marker) {
        marker.setAttribute('cx', (pad + p.x * span).toFixed(2))
        marker.setAttribute('cy', (pad + (1 - p.y) * span).toFixed(2))
      }
      surf.readout.textContent = surf.fmt(p, i)
      act.classList.add('tri-act--scrub')
      for (const r of resolved) r.wrap.classList.toggle('tri-elev-wrap--read', r === surf)
    }
    const onLeave = () => {
      act.classList.remove('tri-act--scrub')
      surf.wrap.classList.remove('tri-elev-wrap--read')
    }
    surf.svgEl.addEventListener('mousemove', onMove)
    surf.svgEl.addEventListener('mouseleave', onLeave)
  }
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

const zoneClock = (sec: number): string => {
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

const buildZoneTable = (
  title: string,
  times: number[],
  bounds: number[],
  names: string[],
  unit: string,
  caption: string,
): HTMLElement => {
  const wrap = el('div', 'tri-zone')
  wrap.appendChild(el('div', 'tri-zone-title', title))
  const total = times.reduce((s, x) => s + x, 0) || 1
  let mx = 1
  for (const t of times) if (t > mx) mx = t
  const grid = el('div', 'tri-zone-grid')
  for (let i = times.length - 1; i >= 0; i--) {
    const row = el('div', 'tri-zone-row')
    const z = el('span', 'tri-zone-z', `Z${i + 1}`)
    z.dataset.name = names[i] ?? `Z${i + 1}`
    z.tabIndex = 0
    row.append(
      z,
      el('span', 'tri-zone-range', `${zoneRange(bounds, i)}${unit}`),
      el('span', 'tri-zone-time', zoneClock(times[i])),
      el('span', 'tri-zone-pct', `${((times[i] / total) * 100).toFixed(1)}%`),
    )
    const track = el('span', 'tri-zone-bar')
    const fill = el('span', `tri-zone-fill tri-zone-fill--${i + 1}`)
    fill.style.width = `${(times[i] / mx) * 100}%`
    track.appendChild(fill)
    row.appendChild(track)
    grid.appendChild(row)
  }
  wrap.appendChild(grid)
  if (caption) wrap.appendChild(el('div', 'tri-zone-cap', caption))
  return wrap
}

const buildHrZones = (d: StravaActivityDetail): HTMLElement | null => {
  if (!d.hrZones || !DETAIL_ZONES?.hr.length) return null
  return buildZoneTable('heart rate zones', d.hrZones, DETAIL_ZONES.hr, HR_ZONE_NAMES, '', '')
}

const buildPowerZones = (d: StravaActivityDetail): HTMLElement | null => {
  if (!d.powerZones || !DETAIL_ZONES?.power.length) return null
  const ftp = DETAIL_ZONES.ftp
  return buildZoneTable(
    'power zones',
    d.powerZones,
    DETAIL_ZONES.power,
    POWER_ZONE_NAMES,
    'w',
    ftp != null ? `based on FTP ${ftp} W` : '',
  )
}

const buildPowerHist = (d: StravaActivityDetail): HTMLElement | null => {
  const hist = d.powerHist
  if (!hist || hist.length < 2) return null
  const wrap = el('div', 'tri-zone')
  wrap.appendChild(el('div', 'tri-zone-title', '25W power distribution'))
  const H = 34
  const n = hist.length
  let mx = 1
  for (const t of hist) if (t > mx) mx = t
  const total = hist.reduce((a, b) => a + b, 0) || 1
  const s = svg('svg', {
    class: 'tri-hist-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  const barByBin = new Map<number, SVGElement>()
  hist.forEach((t, i) => {
    if (t <= 0) return
    const h = (t / mx) * (H - 1)
    const r = svg('rect', { x: i + 0.1, y: H - h, width: 0.8, height: h, class: 'tri-hist-bar' })
    s.appendChild(r)
    barByBin.set(i, r)
  })
  const np = d.npWatts ?? d.avgWatts
  if (np != null)
    s.appendChild(
      svg('line', { x1: np / 25 + 0.5, y1: 0, x2: np / 25 + 0.5, y2: H, class: 'tri-hist-avg' }),
    )
  s.appendChild(svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H }))
  wrap.appendChild(s)
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k', `0–${(n - 1) * 25 + 24} W`))
  if (np != null) cap.appendChild(el('span', 'tri-ana-k', `wtd avg ${np} W`))
  wrap.appendChild(cap)
  const cursor = s.querySelector<SVGElement>('.tri-chart-cursor')!
  let lastBar: SVGElement | null = null
  const onMove = (event: MouseEvent) => {
    const r = s.getBoundingClientRect()
    const bin = Math.max(0, Math.min(n - 1, Math.floor(((event.clientX - r.left) / r.width) * n)))
    cursor.setAttribute('x1', `${bin + 0.5}`)
    cursor.setAttribute('x2', `${bin + 0.5}`)
    lastBar?.classList.remove('tri-hist-bar--on')
    lastBar = barByBin.get(bin) ?? null
    lastBar?.classList.add('tri-hist-bar--on')
    readout.textContent = `${bin * 25}–${bin * 25 + 24} W · ${zoneClock(hist[bin])} (${((hist[bin] / total) * 100).toFixed(1)}%)`
    wrap.classList.add('tri-chart--hover')
  }
  const onLeave = () => {
    wrap.classList.remove('tri-chart--hover')
    lastBar?.classList.remove('tri-hist-bar--on')
    lastBar = null
  }
  s.addEventListener('mousemove', onMove)
  s.addEventListener('mouseleave', onLeave)
  return wrap
}

const buildPowerCurve = (d: StravaActivityDetail): HTMLElement | null => {
  const curve = d.powerCurve
  if (!curve || curve.length < 2) return null
  const ref = DETAIL_CURVE_REF
  const wrap = el('div', 'tri-zone')
  wrap.appendChild(el('div', 'tri-zone-title', 'power curve'))
  const W = 100
  const H = 34
  const secs = curve.map(c => c.s)
  const maxW = Math.max(1, ...curve.map(c => c.w), ...ref.map(c => c.w))
  const minLog = Math.log(secs[0])
  const maxLog = Math.log(secs[secs.length - 1])
  const span = Math.max(1e-6, maxLog - minLog)
  const X = (sec: number): number => ((Math.log(sec) - minLog) / span) * W
  const Y = (w: number): number => H - (w / maxW) * (H - 1)
  const toPath = (pts: PowerCurvePoint[]): string =>
    pts.map((c, i) => `${i ? 'L' : 'M'} ${X(c.s).toFixed(2)} ${Y(c.w).toFixed(2)}`).join(' ')
  const s = svg('svg', {
    class: 'tri-curve-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
  })
  if (ref.length >= 2)
    s.appendChild(
      svg('path', {
        d: toPath(ref.filter(c => c.s <= secs[secs.length - 1])),
        class: 'tri-curve-ref',
      }),
    )
  s.appendChild(svg('path', { d: toPath(curve), class: 'tri-curve-line' }))
  const cursor = svg('line', { class: 'tri-chart-cursor', x1: 0, y1: 0, x2: 0, y2: H })
  s.appendChild(cursor)
  wrap.appendChild(s)
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  const cap = el('div', 'tri-elev-cap')
  const dlabel = (sec: number): string =>
    sec < 60 ? `${sec}s` : sec < 3600 ? `${sec / 60}m` : `${sec / 3600}h`
  for (const sec of [5, 60, 300, 1200]) {
    const p = curve.find(c => c.s === sec)
    if (p) cap.appendChild(el('span', 'tri-ana-k', `${dlabel(sec)} ${p.w}W`))
  }
  wrap.appendChild(cap)
  const onMove = (event: MouseEvent) => {
    const r = s.getBoundingClientRect()
    const fx = Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) * W
    let bi = 0
    let best = Infinity
    for (let i = 0; i < curve.length; i++) {
      const dd = Math.abs(X(curve[i].s) - fx)
      if (dd < best) {
        best = dd
        bi = i
      }
    }
    const c = curve[bi]
    cursor.setAttribute('x1', X(c.s).toFixed(2))
    cursor.setAttribute('x2', X(c.s).toFixed(2))
    readout.textContent = `${dlabel(c.s)} · ${c.w} W`
    wrap.classList.add('tri-chart--hover')
  }
  const onLeave = () => wrap.classList.remove('tri-chart--hover')
  s.addEventListener('mousemove', onMove)
  s.addEventListener('mouseleave', onLeave)
  return wrap
}

const derivePace = (route: StravaActivityDetail['route'], movingTimeS: number): number[] => {
  const n = route.length
  const out = Array.from({ length: n }, () => 0)
  if (n < 2) return out
  const dtPer = movingTimeS / Math.max(1, n - 1)
  for (let i = 0; i < n; i++) {
    const j0 = Math.max(0, i - 2)
    const j1 = Math.min(n - 1, i + 2)
    const dKm = route[j1].d - route[j0].d
    const dt = (j1 - j0) * dtPer
    out[i] = dt > 0 ? dKm / (dt / 3600) : 0
  }
  return out
}

const buildHeatRoute = (
  route: StravaActivityDetail['route'],
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
  ramp = 7,
): SVGElement => {
  const pad = 6
  const span = 100 - pad * 2
  const vals = route.map((p, i) => pick(p, i))
  let lo = Infinity
  let hi = -Infinity
  for (const v of vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const range = hi > lo ? hi - lo : 1
  const sx = (p: StravaActivityDetail['route'][number]): number => pad + p.x * span
  const sy = (p: StravaActivityDetail['route'][number]): number => pad + (1 - p.y) * span
  const s = svg('svg', {
    class: 'tri-route',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'xMidYMid meet',
  })
  const g = svg('g', { class: 'tri-heat' })
  for (let i = 0; i < route.length - 1; i++) {
    const mid = (vals[i] + vals[i + 1]) / 2
    const t = Math.min(1, Math.max(0, (mid - lo) / range))
    const bucket = Math.min(ramp, Math.max(1, Math.ceil(t * ramp) || 1))
    g.appendChild(
      svg('path', {
        d: `M ${sx(route[i]).toFixed(2)} ${sy(route[i]).toFixed(2)} L ${sx(route[i + 1]).toFixed(2)} ${sy(route[i + 1]).toFixed(2)}`,
        class: `tri-heat-seg tri-heat--${bucket}`,
      }),
    )
  }
  s.appendChild(g)
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return s
}

const buildHeatLegend = (lo: number, hi: number, fmt: (v: number) => string): HTMLElement => {
  const wrap = el('div', 'tri-map-legend')
  wrap.append(
    el('span', 'tri-map-legend-lo', fmt(lo)),
    el('span', 'tri-map-legend-bar'),
    el('span', 'tri-map-legend-hi', fmt(hi)),
  )
  return wrap
}

interface MapMetric {
  label: string
  pick: (p: StravaActivityDetail['route'][number], i: number) => number
  fmt: (v: number) => string
  profile: () => HTMLElement
  readout: (p: StravaActivityDetail['route'][number], i: number) => string
  extra?: () => (HTMLElement | null)[]
}

const metricSpecs = (d: StravaActivityDetail): MapMetric[] => {
  const route = d.route
  const pace = derivePace(route, d.movingTimeS)
  const hasPower = d.deviceWatts && route.some(p => p.w > 0)
  const hasHr = route.some(p => p.hr > 0)
  const hasCad = route.some(p => p.cad > 0)
  const hasElev = d.maxAlt > d.minAlt
  const cadUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const paceFmt = (kmh: number): string => {
    if (kmh <= 0) return '—'
    if (d.sport === 'bike') return `${(kmh * KM_TO_MI).toFixed(1)} mph`
    if (d.sport === 'swim') return `${clock(3600 / (kmh * 10))} /100m`
    return `${clock(3600 / (kmh * KM_TO_MI))} /mi`
  }
  const paceSpec: MapMetric = {
    label: d.sport === 'bike' ? 'speed' : 'pace',
    pick: (_p, i) => pace[i],
    fmt: paceFmt,
    profile: () =>
      buildTrace(
        d,
        (_p, i) => pace[i],
        d.sport === 'bike' ? 'speed' : 'pace',
        () => '',
      ),
    readout: (p, i) => `${scrubDist(p.d, d.sport)} · ${paceFmt(pace[i])}`,
  }
  const powerSpec: MapMetric = {
    label: 'power',
    pick: p => p.w,
    fmt: v => `${Math.round(v)} W`,
    profile: () =>
      buildTrace(
        d,
        p => p.w,
        'power',
        m => `${m} W peak`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.w} W`,
    extra: () => [buildPowerCurve(d), buildPowerZones(d), buildPowerHist(d)],
  }
  const hrSpec: MapMetric = {
    label: 'heart rate',
    pick: p => p.hr,
    fmt: v => `${Math.round(v)} bpm`,
    profile: () =>
      buildTrace(
        d,
        p => p.hr,
        'hr',
        m => `${m} bpm peak`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.hr} bpm`,
    extra: () => [buildHrZones(d)],
  }
  const cadSpec: MapMetric = {
    label: 'cadence',
    pick: p => p.cad,
    fmt: v => `${Math.round(v)} ${cadUnit}`,
    profile: () =>
      buildTrace(
        d,
        p => p.cad,
        'cadence',
        m => `${m} ${cadUnit} peak`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.cad} ${cadUnit}`,
  }
  const elevSpec: MapMetric = {
    label: 'elevation',
    pick: p => p.alt,
    fmt: v => `${Math.round(v)} m`,
    profile: () => buildElevation(d),
    readout: (p, i) => {
      const g = Math.round(gradeAt(route, i) * 10) / 10
      return `${scrubDist(p.d, d.sport)} · ${Math.round(p.alt)} m · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%`
    },
  }
  const specs: MapMetric[] = []
  if (d.sport === 'bike') {
    if (hasPower) specs.push(powerSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    specs.push(paceSpec)
    if (hasElev) specs.push(elevSpec)
  } else if (d.sport === 'run') {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasPower) specs.push(powerSpec)
  } else {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
  }
  return specs
}

interface MapDetailOpts {
  mapMode?: boolean
  onMetric?: (i: number) => void
  onHover?: (p: StravaActivityDetail['route'][number], i: number) => void
}

const renderMapDetail = (d: StravaActivityDetail, opts?: MapDetailOpts): HTMLElement => {
  const wrap = el('section', 'tri-act tri-act--expanded')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildIcon(d.sport))
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const sbody = document.createElement('tbody')
  sbody.append(
    statRow('distance', dist(d.distanceKm, d.sport)),
    statRow('time', dur(d.movingTimeS)),
    statRow(d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)),
  )
  if (d.avgHr) sbody.appendChild(statRow('avg hr', `${d.avgHr} bpm`))
  stats.appendChild(sbody)
  wrap.appendChild(stats)

  const specs = d.route.length >= 2 ? metricSpecs(d) : []
  if (specs.length === 0) {
    const figs = el('div', 'tri-act-figs')
    if (d.sport === 'swim') figs.appendChild(buildPool(d))
    if (figs.childElementCount > 0) wrap.appendChild(figs)
    const more = el('div', 'tri-act-more')
    for (const z of [buildPowerCurve(d), buildPowerZones(d), buildPowerHist(d), buildHrZones(d)])
      if (z) more.appendChild(z)
    if (more.childElementCount > 0) wrap.appendChild(more)
    return wrap
  }

  const tablist = el('div', 'tri-map-tablist')
  tablist.setAttribute('role', 'tablist')
  const figs = el('div', 'tri-act-figs tri-map-figs')
  const profileBox = el('div', 'tri-map-profile')
  const zoneBox = el('div', 'tri-act-more')
  wrap.append(tablist, figs, profileBox, zoneBox)

  let active = 0
  const draw = () => {
    const spec = specs[active]
    const vals = d.route.map((p, i) => spec.pick(p, i))
    let lo = Infinity
    let hi = -Infinity
    for (const v of vals) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    let marker: SVGElement | null = null
    if (opts?.mapMode) {
      figs.replaceChildren(buildHeatLegend(lo, hi, spec.fmt))
    } else {
      const heat = buildHeatRoute(d.route, spec.pick)
      figs.replaceChildren(heat, buildHeatLegend(lo, hi, spec.fmt))
      marker = heat.querySelector<SVGElement>('.tri-route-cursor')
    }
    const prof = spec.profile()
    profileBox.replaceChildren(prof)
    zoneBox.replaceChildren()
    if (spec.extra) for (const z of spec.extra()) if (z) zoneBox.appendChild(z)
    linkScrub(
      wrap,
      marker,
      [
        {
          wrap: prof,
          fmt: (p, i) => {
            opts?.onHover?.(p, i)
            return spec.readout(p, i)
          },
        },
      ],
      d.route,
    )
    Array.from(tablist.children).forEach((t, i) =>
      t.setAttribute('aria-selected', i === active ? 'true' : 'false'),
    )
    opts?.onMetric?.(active)
  }
  specs.forEach((spec, i) => {
    const tab = el('button', 'tri-map-tab', spec.label)
    tab.setAttribute('type', 'button')
    tab.setAttribute('role', 'tab')
    tab.addEventListener('click', () => {
      active = i
      draw()
    })
    tablist.appendChild(tab)
  })
  draw()
  return wrap
}

const renderDetail = (d: StravaActivityDetail): HTMLElement => {
  const wrap = el('section', 'tri-act')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildIcon(d.sport))
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const body = document.createElement('tbody')
  body.append(
    statRow('distance', dist(d.distanceKm, d.sport)),
    statRow('time', dur(d.movingTimeS)),
    statRow(d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)),
  )
  if (d.avgHr) body.appendChild(statRow('avg hr', `${d.avgHr} bpm`))
  stats.appendChild(body)
  wrap.appendChild(stats)
  if (d.fueling) {
    const fueling = buildFueling(d.fueling)
    if (fueling) wrap.appendChild(fueling)
  }

  let routeMarker: SVGElement | null = null
  const surfaces: ScrubSurface[] = []

  if (d.sport === 'swim') {
    const figs = el('div', 'tri-act-figs')
    figs.appendChild(buildPool(d))
    wrap.appendChild(figs)
  } else if (d.route.length >= 2) {
    const figs = el('div', 'tri-act-figs')
    const routeSvg = buildRoute(d.route)
    const elev = buildElevation(d)
    figs.append(routeSvg, elev)
    routeMarker = routeSvg.querySelector<SVGElement>('.tri-route-cursor')
    surfaces.push({
      wrap: elev,
      fmt: (p, i) => {
        const g = Math.round(gradeAt(d.route, i) * 10) / 10
        return (
          `${scrubDist(p.d, d.sport)} · ${Math.round(p.alt)} m · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%` +
          (p.hr > 0 ? ` · ${p.hr} bpm` : '')
        )
      },
    })
    wrap.appendChild(figs)
  }

  const moreRows: [string, string][] = []
  if (d.deviceWatts && d.npWatts != null) moreRows.push(['NP', `${d.npWatts} W`])
  if (d.avgWatts != null)
    moreRows.push([d.deviceWatts ? 'avg power' : 'est power', `${d.avgWatts} W`])
  if (d.deviceWatts && d.maxWatts != null) moreRows.push(['max power', `${d.maxWatts} W`])
  if (d.kilojoules != null) moreRows.push(['energy', `${d.kilojoules} kJ`])
  if (d.calories != null) moreRows.push(['calories', `${d.calories.toLocaleString('en-US')} kcal`])
  if (d.avgCadence != null)
    moreRows.push(['cadence', `${d.avgCadence} ${d.sport === 'run' ? 'spm' : 'rpm'}`])
  if (d.maxHr != null) moreRows.push(['max hr', `${d.maxHr} bpm`])
  if (d.sufferScore != null) moreRows.push(['effort', `${d.sufferScore}`])
  if (d.avgTemp != null) moreRows.push(['temp', `${d.avgTemp}°C`])

  const hasPowerStream = d.deviceWatts && d.route.some(p => p.w > 0)
  const hasHrStream = d.route.some(p => p.hr > 0)
  const hasCadStream = d.route.some(p => p.cad > 0)
  const hasZones = !!(d.hrZones || d.powerZones || d.powerHist || d.powerCurve)

  if (moreRows.length > 0 || hasPowerStream || hasHrStream || hasCadStream || hasZones) {
    const toggle = el('button', 'tri-act-toggle')
    toggle.setAttribute('type', 'button')
    const more = el('div', 'tri-act-more')
    if (moreRows.length > 0) {
      const mt = el('table', 'tri-act-stats')
      const mb = document.createElement('tbody')
      for (const [k, v] of moreRows) mb.appendChild(statRow(k, v))
      mt.appendChild(mb)
      more.appendChild(mt)
    }
    if (hasHrStream) {
      const t = buildTrace(
        d,
        p => p.hr,
        'hr',
        m => `${m} bpm peak`,
      )
      more.appendChild(t)
      surfaces.push({ wrap: t, fmt: p => `${scrubDist(p.d, d.sport)} · ${p.hr} bpm` })
    }
    if (hasPowerStream) {
      const t = buildTrace(
        d,
        p => p.w,
        'power',
        m => `${m} W peak`,
      )
      more.appendChild(t)
      surfaces.push({ wrap: t, fmt: p => `${scrubDist(p.d, d.sport)} · ${p.w} W` })
    }
    if (hasCadStream) {
      const t = buildTrace(
        d,
        p => p.cad,
        'cadence',
        m => `${m} ${d.sport === 'run' ? 'spm' : 'rpm'} peak`,
      )
      more.appendChild(t)
      surfaces.push({
        wrap: t,
        fmt: p => `${scrubDist(p.d, d.sport)} · ${p.cad} ${d.sport === 'run' ? 'spm' : 'rpm'}`,
      })
    }
    const hrz = buildHrZones(d)
    if (hrz) more.appendChild(hrz)
    const pcurve = buildPowerCurve(d)
    if (pcurve) more.appendChild(pcurve)
    const pz = buildPowerZones(d)
    if (pz) more.appendChild(pz)
    const phist = buildPowerHist(d)
    if (phist) more.appendChild(phist)
    wrap.append(toggle, more)
  }
  if (surfaces.length > 0 && d.route.length >= 2) linkScrub(wrap, routeMarker, surfaces, d.route)
  return wrap
}

const setup = (root: HTMLElement): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const location = root.dataset.location ?? 'Toronto'
  let active: HTMLElement | null = null
  let activeIdx = -1
  let details: Record<string, StravaActivityDetail> | null = null
  let healthByDate: Record<string, ActivityHealth> = {}
  let pinned = false
  let locked = false
  let hideTimer = 0

  const scroller = el('div', 'tri-pop-scroll')
  pop.appendChild(scroller)
  const updateOverflow = () => {
    pop.classList.toggle('tri-pop--top', scroller.scrollTop > 4)
    pop.classList.toggle(
      'tri-pop--more',
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 4,
    )
  }
  const setLocked = (on: boolean) => {
    locked = on
    barsEl.classList.toggle('tri-bars--locked', on)
  }

  let audio: AudioContext | null = null
  let lastDrop = 0
  try {
    audio = new AudioContext()
    if (navigator.userActivation?.hasBeenActive) void audio.resume()
  } catch {
    audio = null
  }
  const armAudio = () => {
    if (audio && audio.state === 'suspended') void audio.resume()
  }
  const raindrop = (idx: number) => {
    if (!audio || audio.state !== 'running') return
    const t = audio.currentTime
    if (t - lastDrop < 0.05) return
    lastDrop = t
    const base = 560 + (idx % 8) * 28
    const osc = audio.createOscillator()
    const gain = audio.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(base * 1.8, t)
    osc.frequency.exponentialRampToValueAtTime(base, t + 0.08)
    gain.gain.setValueAtTime(0.0001, t)
    gain.gain.exponentialRampToValueAtTime(0.05, t + 0.004)
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.13)
    osc.connect(gain)
    gain.connect(audio.destination)
    osc.start(t)
    osc.stop(t + 0.15)
  }
  window.addEventListener('pointerdown', armAudio)
  window.addEventListener('keydown', armAudio)

  const buildCard = (bar: HTMLElement): HTMLElement => {
    const card = el('div', 'tri-pop-card')
    const idsAttr = bar.dataset.ids
    const head = el('div', 'tri-pop-head')
    head.appendChild(el('span', 'tri-pop-date', bar.dataset.date ?? ''))
    if (idsAttr) {
      const first = details?.[idsAttr.split(',')[0]]
      head.appendChild(el('span', 'tri-pop-loc', first?.location ?? location))
    }
    card.appendChild(head)
    if (!idsAttr) {
      const rest = el('div', 'tri-pop-rest')
      rest.append(buildBattery(), el('span', 'tri-pop-rest-label', 'rest'))
      card.appendChild(rest)
    } else if (details) {
      const day = idsAttr
        .split(',')
        .map(id => details![id])
        .filter(Boolean)
        .sort((a, b) => b.distanceKm - a.distanceKm)
      for (const d of day) card.appendChild(renderDetail(d))
    } else {
      card.appendChild(el('div', 'tri-pop-rest', '·'))
    }
    const dh = healthByDate[bar.dataset.dateIso ?? '']
    if (dh) {
      const rec = buildRecovery(dh)
      if (rec) card.appendChild(rec)
    }
    return card
  }

  const place = (cx: number, cy: number) => {
    const r = pop.getBoundingClientRect()
    const gap = 18
    let left = cx + gap
    if (left + r.width > window.innerWidth - 8) left = cx - gap - r.width
    left = Math.max(8, left)
    let top = cy - r.height / 2
    top = Math.max(8, Math.min(top, window.innerHeight - r.height - 8))
    pop.style.left = `${left}px`
    pop.style.top = `${top}px`
  }

  const nearest = (clientX: number): number => {
    let best = Infinity
    let found = -1
    bars.forEach((bar, i) => {
      const r = bar.getBoundingClientRect()
      const d = Math.abs(r.left + r.width / 2 - clientX)
      if (d < best) {
        best = d
        found = i
      }
    })
    return found
  }

  const showFor = (idx: number, cx: number, cy: number) => {
    const bar = bars[idx]
    if (bar !== active) {
      const dir = activeIdx === -1 ? 0 : Math.sign(idx - activeIdx)
      if (active) active.classList.remove('tri-bar--active')
      active = bar
      activeIdx = idx
      bar.classList.add('tri-bar--active')
      raindrop(idx)
      const card = buildCard(bar)
      scroller.replaceChildren(card)
      scroller.scrollTop = 0
      updateOverflow()
      if (!reduce)
        card.animate(
          [
            { opacity: 0, transform: `translateX(${dir * 12}px)` },
            { opacity: 1, transform: 'none' },
          ],
          { duration: 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
        )
    }
    place(cx, cy)
    root.classList.add('tri-hovering')
  }

  const setExpanded = (on: boolean) => {
    for (const a of pop.querySelectorAll('.tri-act')) a.classList.toggle('tri-act--expanded', on)
    updateOverflow()
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    activeIdx = -1
    pinned = false
    setLocked(false)
    root.classList.remove('tri-hovering')
  }

  const onMove = (event: MouseEvent) => {
    if (pinned || locked) return
    window.clearTimeout(hideTimer)
    const idx = nearest(event.clientX)
    if (idx >= 0) showFor(idx, event.clientX, event.clientY)
  }
  const onBarsLeave = () => {
    if (!pinned && !locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onPopEnter = () => {
    window.clearTimeout(hideTimer)
    pinned = true
  }
  const onPopLeave = () => {
    pinned = false
    if (!locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onBarsClick = (event: MouseEvent) => {
    const idx = nearest(event.clientX)
    if (idx < 0) return
    if (locked && bars[idx] === active) {
      setLocked(false)
      setExpanded(false)
    } else {
      showFor(idx, event.clientX, event.clientY)
      setLocked(true)
      setExpanded(true)
    }
  }
  const onToggle = (event: MouseEvent) => {
    const btn = (event.target as HTMLElement | null)?.closest('.tri-act-toggle')
    btn?.closest('.tri-act')?.classList.toggle('tri-act--expanded')
  }
  const dismiss = () => {
    if (!locked) return
    setLocked(false)
    setExpanded(false)
    hide()
  }
  const onDocClick = (event: MouseEvent) => {
    const t = event.target as Node
    if (locked && !barsEl.contains(t) && !pop.contains(t)) dismiss()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') dismiss()
  }

  const path = root.dataset.detailPath
  if (path)
    fetch(path)
      .then(res => res.json())
      .then((data: DetailPayload) => {
        details = data.details
        healthByDate = data.health ?? {}
        DETAIL_ZONES = data.zones ?? null
        DETAIL_CURVE_REF = data.powerCurveRef ?? []
        if (active) {
          scroller.replaceChildren(buildCard(active))
          if (locked) setExpanded(true)
          updateOverflow()
        }
      })
      .catch(() => {})

  const onFocusDay = (event: Event) => {
    const date = (event as CustomEvent<{ date?: string }>).detail?.date
    if (!date) return
    const idx = bars.findIndex(b => b.dataset.dateIso === date)
    if (idx < 0) return
    bars[idx].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    const r = bars[idx].getBoundingClientRect()
    showFor(idx, r.left + r.width / 2, r.top + r.height / 2)
    setLocked(true)
    setExpanded(true)
  }

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onBarsLeave)
  barsEl.addEventListener('click', onBarsClick)
  pop.addEventListener('mouseenter', onPopEnter)
  pop.addEventListener('mouseleave', onPopLeave)
  pop.addEventListener('click', onToggle)
  scroller.addEventListener('scroll', updateOverflow, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
  window.addEventListener('tri:focus-day', onFocusDay)

  return () => {
    window.clearTimeout(hideTimer)
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    barsEl.removeEventListener('click', onBarsClick)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
    pop.removeEventListener('click', onToggle)
    scroller.removeEventListener('scroll', updateOverflow)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('pointerdown', armAudio)
    window.removeEventListener('keydown', armAudio)
    window.removeEventListener('tri:focus-day', onFocusDay)
    void audio?.close()
  }
}

const setupCalc = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-calc-btn')
  const calc = root.querySelector<HTMLElement>('.tri-calc')
  const closeBtn = root.querySelector<HTMLElement>('.tri-calc-close')
  if (!btn || !calc) return null

  const parseClock = (s: string): number => {
    const parts = s.split(':').map(Number)
    return parts.length === 2 ? (parts[0] || 0) * 60 + (parts[1] || 0) : Number(s) || 0
  }
  const fmt = (sec: number): string => {
    const t = Math.round(sec)
    const h = Math.floor(t / 3600)
    const m = Math.floor((t % 3600) / 60)
    const s = t % 60
    return h > 0
      ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
      : `${m}:${String(s).padStart(2, '0')}`
  }
  const inputVal = (k: string): string =>
    calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)?.value ?? ''
  const setResult = (leg: string, sec: number): void => {
    const e = calc.querySelector<HTMLElement>(`.tri-calc-r[data-leg="${leg}"]`)
    if (e) e.textContent = fmt(sec)
  }

  const compute = () => {
    const swimKm = Number(calc.dataset.swim) || 0
    const bikeKm = Number(calc.dataset.bike) || 0
    const runKm = Number(calc.dataset.run) || 0
    const swimSec = ((swimKm * 1000) / 100) * parseClock(inputVal('swim'))
    const t1 = parseClock(inputVal('t1'))
    const mph = Number(inputVal('bike')) || 0
    const bikeSec = mph > 0 ? ((bikeKm * 0.621371) / mph) * 3600 : 0
    const t2 = parseClock(inputVal('t2'))
    const runSec = runKm * 0.621371 * parseClock(inputVal('run'))
    setResult('swim', swimSec)
    setResult('t1', t1)
    setResult('bike', bikeSec)
    setResult('t2', t2)
    setResult('run', runSec)
    setResult('total', swimSec + t1 + bikeSec + t2 + runSec)
  }

  const open = () => {
    root.classList.add('tri-calc-open')
    calc.setAttribute('aria-hidden', 'false')
    compute()
  }
  const close = () => {
    root.classList.remove('tri-calc-open')
    calc.setAttribute('aria-hidden', 'true')
  }
  const onCalcClick = (event: MouseEvent) => {
    const p = (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-calc-preset')
    if (!p) return
    calc.dataset.swim = p.dataset.swim ?? ''
    calc.dataset.bike = p.dataset.bike ?? ''
    calc.dataset.run = p.dataset.run ?? ''
    for (const x of calc.querySelectorAll('.tri-calc-preset'))
      x.classList.toggle('tri-calc-preset--on', x === p)
    compute()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', open)
  closeBtn?.addEventListener('click', close)
  calc.addEventListener('click', onCalcClick)
  calc.addEventListener('input', compute)
  document.addEventListener('keydown', onKey)
  calc.querySelectorAll('.tri-calc-preset')[1]?.classList.add('tri-calc-preset--on')

  return () => {
    btn.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    calc.removeEventListener('click', onCalcClick)
    calc.removeEventListener('input', compute)
    document.removeEventListener('keydown', onKey)
  }
}

const setupDropdown = (
  root: HTMLElement,
  wrapSel: string,
  btnSel: string,
  panelSel: string,
  openClass: string,
): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>(btnSel)
  const wrap = root.querySelector<HTMLElement>(wrapSel)
  const panel = root.querySelector<HTMLElement>(panelSel)
  if (!btn || !wrap || !panel) return null

  const close = () => {
    wrap.classList.remove(openClass)
    panel.setAttribute('aria-hidden', 'true')
  }
  const onBtn = () => {
    const open = wrap.classList.toggle(openClass)
    panel.setAttribute('aria-hidden', open ? 'false' : 'true')
  }
  const onDocClick = (event: MouseEvent) => {
    if (!wrap.contains(event.target as Node)) close()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', onBtn)
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', onBtn)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
  }
}

const ANA_W = 100
const ANA_H = 30
const RACE_LABEL: Record<string, string> = {
  sprint: 'sprint',
  olympic: 'olympic',
  '70.3': '70.3',
  ironman: 'ironman',
}

const clampN = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))
const polyD = (pts: [number, number][]): string =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')
const signed = (n: number): string => (n > 0 ? `+${n}` : `${n}`)
const hms = (sec: number): string => {
  const t = Math.max(0, Math.round(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`
}
const GLOSS: Record<string, { term: string; def: string }> = {
  ctl: {
    term: 'fitness/CTL',
    def: 'Chronic Training Load is a 42-day weighted average of daily training stress. Climbs slowly with consistent work and is the best single proxy for fitness.',
  },
  atl: {
    term: 'fatigue/ATL',
    def: 'Acute Training Load is a 7-day weighted average of training stress. Spikes after hard days; the proxy for how tired you are right now.',
  },
  tsb: {
    term: 'form/TSB',
    def: 'Training Stress Balance $\\mathrm{TSB}=\\mathrm{CTL}-\\mathrm{ATL}$ (fitness − fatigue). Positive means fresh and tapered; negative means loaded and carrying fatigue.',
  },
  acwr: {
    term: 'ACWR',
    def: 'Acute:Chronic Workload Ratio $\\mathrm{ACWR}=\\text{7d}/\\text{28d}$ load. 0.8–1.3 is the safe zone; above 1.5 flags an injury-risk spike.',
  },
  ramp: {
    term: 'ramp',
    def: 'Week-over-week change in fitness (CTL). Positive is building; large jumps are the classic too-much-too-soon risk.',
  },
  monotony: {
    term: 'monotony',
    def: 'Daily-load sameness across a week, $\\text{monotony}=\\mu/\\sigma$ (mean over standard deviation). Above ~2 with high load is Foster’s strain red flag.',
  },
  strain: {
    term: 'strain',
    def: '$\\text{strain}=\\text{weekly load}\\times\\text{monotony}$. High, unvarying training scores high, often dictates overtraining.',
  },
  load: {
    term: 'load',
    def: 'Per-session training stress $\\mathrm{load}\\approx\\mathrm{IF}^2\\cdot t$, scaled so an hour at threshold ≈ 100. HR, power and cadence are captured per activity; this load stays pace-derived for now.',
  },
  score: {
    term: 'readiness',
    def: 'A 0–100 blend of your fitness against the race demand (45%) and how much of each leg’s distance you have actually covered in training (55%).',
  },
  binding: {
    term: 'binding leg',
    def: 'The discipline limiting your readiness most — the weakest distance-coverage × recency. Train this one first.',
  },
  predtime: {
    term: 'predicted time',
    def: 'Each leg is paced from your threshold pace through an endurance-decay model, so longer legs slow down, plus transitions and a run-off-the-bike penalty.',
  },
  conf: {
    term: 'confidence',
    def: 'How much data backs the estimate. firm = enough recent efforts; low = only a couple; stale = newest effort over 45 days old; prior = no data, a generic assumption.',
  },
  threshold: {
    term: 'threshold pace',
    def: 'Your grade-adjusted pace at roughly 1-hour effort, taken from the 90th percentile of your sessions. The anchor for every prediction.',
  },
  trend: {
    term: 'pace trend',
    def: 'Which way your threshold pace is moving, fit by EWMA or least-squares over recent sessions. The shaded band is the forecast’s uncertainty.',
  },
  weight: {
    term: 'body weight',
    def: 'Daily weigh-ins. Feeds recovery context and the bodyweight-dependent energy estimates.',
  },
  wtrend: {
    term: 'weight trend',
    def: 'Least-squares slope of your logged weight in kg per week. Negative means trending down.',
  },
  effort: {
    term: 'relative effort',
    def: 'Strava’s suffer score—duration weighted by how far above resting your heart rate sat. The bars sum each week’s sessions, so it tracks acute training stress across all three sports at once.',
  },
}

const markGloss = (e: HTMLElement, key: string): HTMLElement => {
  e.dataset.gloss = key
  e.tabIndex = 0
  return e
}

const anaTitle = (text: string, key?: string): HTMLElement => {
  const e = el('div', 'tri-ana-block-title', text)
  if (key) markGloss(e, key)
  return e
}
const bySport = <T extends { sport: Sport }>(arr: T[], sport: Sport): T | undefined =>
  arr.find(x => x.sport === sport)
const thLabel = (th: { paceLabel: string; unit: string }): string =>
  th.unit === 'km/h' ? `${th.paceLabel} km/h` : `${th.paceLabel}${th.unit.slice(1)}`
const buildIconLeg = (sport: Sport): HTMLElement => {
  const wrap = el('span', `tri-ana-ico tri-leg-${sport}`)
  wrap.appendChild(buildIcon(sport))
  return wrap
}
const trendDir = (invert: boolean, slope: number | null): number => {
  if (slope == null || slope === 0) return 0
  return (invert ? slope < 0 : slope > 0) ? 1 : -1
}

const buildHeadline = (data: Analytics): HTMLElement => {
  const wrap = el('div', 'tri-ana-head')
  wrap.appendChild(el('div', 'tri-ana-head-line', data.headline || 'training analytics'))
  const r = data.risk
  const stats = el('div', 'tri-ana-head-stats')
  const stat = (k: string, v: string, sub: string, cls: string, key: string): HTMLElement => {
    const c = el('div', 'tri-ana-stat')
    c.append(
      el('span', `tri-ana-stat-v ${cls}`, v),
      el('span', 'tri-ana-stat-k', k),
      el('span', 'tri-ana-stat-sub', sub),
    )
    return markGloss(c, key)
  }
  stats.append(
    stat('fitness', `${Math.round(r.ctl)}`, 'CTL', '', 'ctl'),
    stat('form', signed(Math.round(r.tsb)), r.tsbZone, `tri-zone-${r.tsbZone}`, 'tsb'),
    stat(
      'load',
      r.acwrState === 'building' ? 'base' : (r.acwr?.toFixed(2) ?? '—'),
      r.acwrState,
      `tri-acwr-${r.acwrState}`,
      'acwr',
    ),
  )
  wrap.appendChild(stats)
  return wrap
}

const buildGauge = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-gauge')
  block.appendChild(anaTitle('form · ramp'))
  const r = data.risk
  const W = 100
  const H = 12
  const lo = -40
  const hi = 25
  const xf = (v: number): number => ((clampN(v, lo, hi) - lo) / (hi - lo)) * W
  const s = svg('svg', {
    class: 'tri-gauge-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('rect', { x: 0, y: 4, width: xf(-10), height: 4, class: 'tri-gauge-zone tri-gauge-z-low' }),
  )
  s.appendChild(
    svg('rect', {
      x: xf(-10),
      y: 4,
      width: xf(5) - xf(-10),
      height: 4,
      class: 'tri-gauge-zone tri-gauge-z-mid',
    }),
  )
  s.appendChild(
    svg('rect', {
      x: xf(5),
      y: 4,
      width: W - xf(5),
      height: 4,
      class: 'tri-gauge-zone tri-gauge-z-high',
    }),
  )
  const k42 = data.meta.method.k42
  const k7 = data.meta.method.k7
  let c = r.ctl
  let a = r.atl
  for (let i = 0; i < 14; i++) {
    c = c * (1 - k42)
    a = a * (1 - k7)
  }
  const projTsb = c - a
  const track = el('div', 'tri-gauge-track')
  const proj = el('span', 'tri-gauge-proj')
  proj.style.left = `${clampN(xf(projTsb), 1.5, 98.5)}%`
  const needle = el('span', 'tri-gauge-needle')
  needle.style.left = `${clampN(xf(r.tsb), 1.5, 98.5)}%`
  track.append(s, proj, needle)
  block.appendChild(track)
  const chips = el('div', 'tri-gauge-chips')
  chips.append(
    markGloss(
      el(
        'span',
        `tri-ana-chip tri-acwr-${r.acwrState}`,
        r.acwrState === 'building'
          ? 'ACWR building base'
          : `ACWR ${r.acwr?.toFixed(2)} ${r.acwrState}`,
      ),
      'acwr',
    ),
    markGloss(
      el('span', 'tri-ana-chip', `ramp ${signed(Math.round((r.rampWeek || 0) * 100))}%`),
      'ramp',
    ),
    markGloss(
      el(
        'span',
        'tri-ana-chip',
        r.monotony != null ? `monotony ${r.monotony.toFixed(2)}` : 'monotony —',
      ),
      'monotony',
    ),
    markGloss(
      el('span', 'tri-ana-chip', r.strain != null ? `strain ${Math.round(r.strain)}` : 'strain —'),
      'strain',
    ),
  )
  block.appendChild(chips)
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(
    el(
      'span',
      'tri-ana-k',
      `form ${signed(Math.round(r.tsb))} → ${signed(Math.round(projTsb))} after 14d rest`,
    ),
  )
  block.appendChild(cap)
  return block
}

const buildPmc = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-pmc')
  block.appendChild(anaTitle('fitness · fatigue · form'))
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  let maxFit = 1
  for (const d of daily) {
    if (d.ctl > maxFit) maxFit = d.ctl
    if (d.atl > maxFit) maxFit = d.atl
  }
  let maxTsb = 8
  for (const d of daily) {
    const ab = Math.abs(d.tsb)
    if (ab > maxTsb) maxTsb = ab
  }
  const fitBot = 20
  const fitTop = 3
  const tsbBase = 27
  const tsbAmp = 5.5
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const yFit = (v: number): number => fitBot - (v / maxFit) * (fitBot - fitTop)
  const yTsb = (v: number): number => tsbBase - (v / maxTsb) * tsbAmp
  let warmEnd = 0
  while (warmEnd < n && daily[warmEnd].warmup) warmEnd++
  const ctlPts = daily.map((d, i) => [x(i), yFit(d.ctl)] as [number, number])
  const atlPts = daily.map((d, i) => [x(i), yFit(d.atl)] as [number, number])
  const tsbPts = daily.map((d, i) => [x(i), yTsb(d.tsb)] as [number, number])
  const s = svg('svg', {
    class: 'tri-ana-svg tri-pmc-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const areaInner = ctlPts.map(([px, py]) => `L ${px.toFixed(2)} ${py.toFixed(2)}`).join(' ')
  s.appendChild(
    svg('path', { d: `M 0 ${fitBot} ${areaInner} L ${ANA_W} ${fitBot} Z`, class: 'tri-elev-area' }),
  )
  s.appendChild(svg('line', { x1: 0, y1: tsbBase, x2: ANA_W, y2: tsbBase, class: 'tri-ana-zero' }))
  s.appendChild(svg('path', { d: polyD(tsbPts), class: 'tri-ana-tsb' }))
  if (warmEnd > 1)
    s.appendChild(
      svg('path', { d: polyD(ctlPts.slice(0, warmEnd)), class: 'tri-elev-line tri-pmc-warm' }),
    )
  s.appendChild(
    svg('path', {
      d: polyD(ctlPts.slice(Math.max(0, warmEnd - 1))),
      class: 'tri-elev-line tri-pmc-ctl',
    }),
  )
  s.appendChild(svg('path', { d: polyD(atlPts), class: 'tri-pmc-atl' }))
  s.appendChild(svg('line', { x1: ANA_W, y1: 0, x2: ANA_W, y2: ANA_H, class: 'tri-pmc-now' }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
  block.appendChild(s)
  const r = data.risk
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    markGloss(el('span', 'tri-ana-k', `CTL ${Math.round(r.ctl)}`), 'ctl'),
    markGloss(el('span', 'tri-ana-k', `ATL ${Math.round(r.atl)}`), 'atl'),
    markGloss(
      el('span', `tri-ana-k tri-zone-${r.tsbZone}`, `TSB ${signed(Math.round(r.tsb))}`),
      'tsb',
    ),
  )
  block.appendChild(cap)
  block.appendChild(el('div', 'tri-chart-readout'))
  return block
}

const buildCtlSport = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-ctlsport')
  block.appendChild(anaTitle('fitness by discipline', 'ctl'))
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  let mx = 1
  for (const d of daily) mx = Math.max(mx, d.swimCtl, d.bikeCtl, d.runCtl)
  const top = 3
  const bot = 28
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const y = (v: number): number => bot - (v / mx) * (bot - top)
  const s = svg('svg', {
    class: 'tri-ana-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const series: { sp: Sport; get: (d: DailyPoint) => number }[] = [
    { sp: 'swim', get: d => d.swimCtl },
    { sp: 'bike', get: d => d.bikeCtl },
    { sp: 'run', get: d => d.runCtl },
  ]
  for (const { sp, get } of series) {
    s.appendChild(
      svg('path', {
        d: polyD(daily.map((d, i) => [x(i), y(get(d))])),
        class: `tri-elev-line tri-line-${sp}`,
      }),
    )
  }
  block.appendChild(s)
  const cap = el('div', 'tri-elev-cap')
  for (const sp of ['swim', 'bike', 'run'] as Sport[]) {
    const days = bySport(data.thresholds, sp)?.staleDays ?? 0
    const leg = el('span', `tri-ana-leg tri-leg-${sp}`)
    leg.append(buildIcon(sp), el('span', 'tri-ana-k', days > 45 ? `${days}d stale` : `${days}d`))
    cap.appendChild(leg)
  }
  block.appendChild(cap)
  return block
}

const buildWeekly = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-weekly')
  block.appendChild(anaTitle('weekly load', 'load'))
  const wk = data.weekly
  if (!wk.length) {
    block.appendChild(el('div', 'tri-ana-empty', 'no weeks'))
    return block
  }
  let mx = 1
  for (const w of wk) if (w.load > mx) mx = w.load
  const n = wk.length
  const H = 32
  const bot = H - 0.5
  const s = svg('svg', {
    class: 'tri-ana-svg tri-ana-weekly-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  wk.forEach((w, i) => {
    if (w.load <= 0) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (w.load / mx) * (H - 2)
    const spike = w.ramp != null && w.ramp > 0.1
    s.appendChild(
      svg('rect', {
        x: i + 0.12,
        y: bot - h,
        width: 0.76,
        height: h,
        class: spike ? 'tri-seg--load tri-seg--spike' : 'tri-seg--load',
      }),
    )
  })
  block.appendChild(s)
  const active = wk.filter(w => w.load > 0).length
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', `${active} active wk`),
    el('span', 'tri-ana-k', `peak ${Math.round(mx)}/wk`),
  )
  block.appendChild(cap)
  return block
}

const buildReadiness = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-readiness')
  block.appendChild(anaTitle('race readiness', 'score'))
  if (!data.races.length) {
    block.appendChild(el('div', 'tri-ana-empty', '—'))
    return block
  }
  for (const r of data.races) {
    const row = el('div', 'tri-rdy-row')
    row.appendChild(el('span', 'tri-rdy-label', RACE_LABEL[r.distance] ?? r.distance))
    const track = el('div', 'tri-rdy-bar')
    const score = clampN(r.score, 0, 100)
    const fill = el('div', 'tri-rdy-fill')
    fill.style.width = `${Math.max(2, score)}%`
    const bw = Math.min(40, r.bandPct)
    const band = el('div', 'tri-rdy-band')
    band.style.left = `${clampN(score - bw / 2, 0, 100 - bw)}%`
    band.style.width = `${bw}%`
    track.append(fill, band)
    row.appendChild(track)
    const meta = el('span', 'tri-rdy-meta')
    meta.append(
      markGloss(el('span', `tri-rdy-bind tri-leg-${r.bindingLeg}`, r.bindingLeg), 'binding'),
      markGloss(el('span', 'tri-rdy-time', hms(r.predictedTotalS)), 'predtime'),
    )
    row.appendChild(meta)
    block.appendChild(row)
  }
  return block
}

const METHOD_WIKI: Record<string, string> = {
  ols: 'Ordinary least squares',
  ewma: 'Exponential smoothing',
}

const buildMethod = (method: string, n: number): HTMLElement => {
  const span = el('span', 'tri-ana-k')
  const title = METHOD_WIKI[method]
  if (!title) {
    span.textContent = `${method} · n=${n}`
    return span
  }
  const a = document.createElement('a')
  a.className = 'internal tri-ana-wiki'
  a.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, '_'))}`
  a.target = '_blank'
  a.rel = 'noopener noreferrer'
  a.dataset.wikipediaLang = 'en'
  a.dataset.wikipediaTitle = title
  a.textContent = method
  span.append(a, ` · n=${n}`)
  return span
}

const buildTrendPanel = (data: Analytics, sport: Sport): HTMLElement => {
  const tr = bySport(data.trends, sport)
  const th = bySport(data.thresholds, sport)
  const wrap = el('div', `tri-trend-panel${tr?.stale ? ' tri-trend-stale' : ''}`)
  const head = el('div', 'tri-trend-head')
  head.append(
    buildIconLeg(sport),
    markGloss(el('span', 'tri-trend-unit', th ? thLabel(th) : sport), 'threshold'),
  )
  if (th)
    head.appendChild(markGloss(el('span', `tri-ana-conf tri-conf-${th.conf}`, th.conf), 'conf'))
  wrap.appendChild(head)
  if (!tr || tr.method === 'none') {
    const msg =
      tr?.note ||
      (th && th.staleDays > 45 ? `stale · last ${th.staleDays}d ago` : 'not enough data')
    wrap.appendChild(el('div', 'tri-trend-note', msg))
    return wrap
  }
  const fc = tr.forecast
  if (fc.length >= 1 && tr.level != null) {
    const level = tr.level
    const m = fc.length
    const weeks = m / 7
    const slope = tr.slopePerWeek ?? 0
    const endVal = level + slope * weeks
    const change = Math.abs(endVal - level)
    const pad = Math.max(change * 0.8, Math.abs(level) * 0.05, 1e-6)
    const lo = Math.min(level, endVal) - pad
    const hi = Math.max(level, endVal) + pad
    const span = Math.max(1e-6, hi - lo)
    const top = 4
    const bot = 24
    const xOf = (p: number): number => 8 + p * (ANA_W - 8)
    const Y = (v: number): number => {
      const t = (v - lo) / span
      return tr.invert ? top + t * (bot - top) : bot - t * (bot - top)
    }
    const Yc = (v: number): number => clampN(Y(v), 0.5, ANA_H - 0.5)
    const valAt = (p: number): number => level + (endVal - level) * p
    const ciHalf = (fc[m - 1].hi - fc[m - 1].lo) / 2
    const coneEnd = clampN(ciHalf, change * 0.4 + 1e-6, span * 0.42)
    const s = svg('svg', {
      class: 'tri-ana-svg tri-trend-svg',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    const N = 12
    const hiPts: string[] = []
    const loPts: string[] = []
    for (let k = 0; k <= N; k++) {
      const p = k / N
      const hw = coneEnd * Math.sqrt(p)
      const c = valAt(p)
      hiPts.push(`${xOf(p).toFixed(2)} ${Yc(c + hw).toFixed(2)}`)
      loPts.push(`${xOf(p).toFixed(2)} ${Yc(c - hw).toFixed(2)}`)
    }
    s.appendChild(
      svg('path', {
        d: `M ${hiPts.join(' L ')} L ${loPts.reverse().join(' L ')} Z`,
        class: `tri-trend-band tri-fill-${sport}`,
      }),
    )
    s.appendChild(
      svg('path', {
        d: `M ${xOf(0).toFixed(2)} ${Y(level).toFixed(2)} L ${xOf(1).toFixed(2)} ${Y(endVal).toFixed(2)}`,
        class: `tri-trend-proj tri-line-${sport}`,
      }),
    )
    const track = el('div', 'tri-trend-track')
    const dot = el('span', `tri-trend-dot tri-bg-${sport}`)
    dot.style.left = `${clampN((xOf(0) / ANA_W) * 100, 2, 98)}%`
    dot.style.top = `${clampN((Y(level) / ANA_H) * 100, 4, 96)}%`
    track.append(s, dot)
    wrap.appendChild(track)
  }
  const dir = trendDir(tr.invert, tr.slopePerWeek)
  const note = el('div', 'tri-trend-note')
  note.append(
    markGloss(
      el(
        'span',
        `tri-trend-dir tri-dir-${dir > 0 ? 'up' : dir < 0 ? 'down' : 'flat'}`,
        dir > 0 ? 'faster' : dir < 0 ? 'slower' : 'flat',
      ),
      'trend',
    ),
    buildMethod(tr.method, tr.sampleSize),
  )
  wrap.appendChild(note)
  return wrap
}

const buildTrend = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-trend')
  block.appendChild(anaTitle('pace trend + forecast'))
  for (const sport of ['swim', 'bike', 'run'] as Sport[])
    block.appendChild(buildTrendPanel(data, sport))
  return block
}

const buildActions = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-actions')
  block.appendChild(anaTitle('things to improve'))
  const banner = el('div', 'tri-actions-head')
  banner.append(
    el('span', 'tri-actions-weak', 'weakest'),
    buildIconLeg(data.weakestSport),
    el('span', 'tri-ana-k', data.weakestSport),
  )
  block.appendChild(banner)
  if (data.actions.length) {
    const tbl = el('table', 'tri-act-stats')
    const body = document.createElement('tbody')
    data.actions.forEach((a, i) => {
      const tr = document.createElement('tr')
      tr.append(
        el('th', 'tri-act-stat-k', `${i + 1}. ${a.text}`),
        el('td', 'tri-act-stat-v', a.value),
      )
      body.appendChild(tr)
    })
    tbl.appendChild(body)
    block.appendChild(tbl)
  }
  const chips = el('div', 'tri-gauge-chips')
  for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
    const th = bySport(data.thresholds, sport)
    if (th)
      chips.appendChild(
        el('span', `tri-ana-chip tri-chip-${sport}`, `${sport} ${thLabel(th)} ${th.conf}`),
      )
  }
  block.appendChild(chips)
  return block
}

const buildBody = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-bodywt')
  block.appendChild(anaTitle('body weight', 'weight'))
  const b: BodyBlock = data.body
  if (b.latestKg == null) {
    block.appendChild(el('div', 'tri-ana-empty', 'no weight logged'))
    return block
  }
  const head = el('div', 'tri-bodywt-head')
  head.append(el('span', 'tri-bodywt-kg', `${b.latestKg.toFixed(1)} kg`))
  if (b.latestLbs != null)
    head.appendChild(el('span', 'tri-bodywt-lbs', `${b.latestLbs.toFixed(0)} lb`))
  block.appendChild(head)
  const pts = b.series
  if (pts.length >= 2) {
    let min = Infinity
    let max = -Infinity
    for (const p of pts) {
      if (p.kg < min) min = p.kg
      if (p.kg > max) max = p.kg
    }
    const range = Math.max(0.5, max - min)
    const lo = min - range * 0.18
    const hi = max + range * 0.18
    const n = pts.length
    const xPct = (i: number): number => (n === 1 ? 50 : (i / (n - 1)) * 100)
    const yPct = (kg: number): number => (1 - (kg - lo) / (hi - lo)) * 100
    const chart = el('div', 'tri-bodywt-chart')
    const yax = el('div', 'tri-bodywt-yax')
    yax.append(el('span', '', `${hi.toFixed(1)}`), el('span', '', `${lo.toFixed(1)}`))
    const plot = el('div', 'tri-bodywt-plot')
    const s = svg('svg', {
      class: 'tri-bodywt-svg',
      viewBox: '0 0 100 100',
      preserveAspectRatio: 'none',
    })
    for (const gy of [0, 50, 100])
      s.appendChild(svg('line', { x1: 0, y1: gy, x2: 100, y2: gy, class: 'tri-bodywt-grid' }))
    for (let i = 0; i < n; i++)
      s.appendChild(
        svg('line', {
          x1: xPct(i),
          y1: 0,
          x2: xPct(i),
          y2: 100,
          class: 'tri-bodywt-grid tri-bodywt-grid--v',
        }),
      )
    s.appendChild(
      svg('path', { d: polyD(pts.map((p, i) => [xPct(i), yPct(p.kg)])), class: 'tri-bodywt-line' }),
    )
    plot.appendChild(s)
    pts.forEach((p, i) => {
      const m = el('span', `tri-bodywt-pt${i === n - 1 ? ' tri-bodywt-pt--last' : ''}`)
      m.style.left = `${xPct(i).toFixed(2)}%`
      m.style.top = `${yPct(p.kg).toFixed(2)}%`
      m.title = `${p.date} · ${p.kg.toFixed(1)} kg`
      plot.appendChild(m)
    })
    chart.append(yax, plot)
    const xax = el('div', 'tri-bodywt-xax')
    xax.append(el('span', '', shortDate(pts[0].date)), el('span', '', shortDate(pts[n - 1].date)))
    block.append(chart, xax)
  }
  const cap = el('div', 'tri-elev-cap')
  if (b.trendKgPerWeek != null)
    cap.appendChild(
      markGloss(
        el(
          'span',
          'tri-ana-k',
          `${b.trendKgPerWeek > 0 ? '+' : ''}${b.trendKgPerWeek.toFixed(2)} kg/wk`,
        ),
        'wtrend',
      ),
    )
  const next = (data.events ?? [])
    .filter(e => e.date >= data.meta.today)
    .sort((a, b2) => a.date.localeCompare(b2.date))[0]
  if (next) cap.appendChild(el('span', 'tri-ana-k', `${next.event ?? 'race'} · ${next.date}`))
  block.appendChild(cap)
  return block
}

const buildEffort = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-effort')
  block.appendChild(anaTitle('relative effort', 'effort'))
  const all = data.weekly
  const active = all.filter(w => w.effort > 0)
  if (!active.length) {
    block.appendChild(el('div', 'tri-ana-empty', 'no effort logged'))
    return block
  }
  let mx = 1
  for (const w of all) if (w.effort > mx) mx = w.effort
  const n = all.length
  const H = 32
  const bot = H - 0.5
  const s = svg('svg', {
    class: 'tri-ana-svg tri-ana-weekly-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  all.forEach((w, i) => {
    if (w.effort <= 0) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (w.effort / mx) * (H - 2)
    s.appendChild(
      svg('rect', { x: i + 0.12, y: bot - h, width: 0.76, height: h, class: 'tri-seg--effort' }),
    )
  })
  block.appendChild(s)
  const last = all[all.length - 1]
  const prev = all.length >= 2 ? all[all.length - 2] : null
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', `this wk ${Math.round(last?.effort ?? 0)}`),
    el('span', 'tri-ana-k', `peak ${Math.round(mx)}`),
  )
  if (prev && prev.effort > 0) {
    const d = Math.round((last?.effort ?? 0) - prev.effort)
    cap.appendChild(el('span', 'tri-ana-k', `${d >= 0 ? '+' : ''}${d} vs last`))
  }
  block.appendChild(cap)
  return block
}

const ANALYTICS_BUILDERS: Record<string, (data: Analytics) => HTMLElement> = {
  body: buildBody,
  gauge: buildGauge,
  pmc: buildPmc,
  'ctl-sport': buildCtlSport,
  weekly: buildWeekly,
  effort: buildEffort,
  readiness: buildReadiness,
  trend: buildTrend,
  actions: buildActions,
}

const wireScrub = (panel: HTMLElement, daily: DailyPoint[]): (() => void) => {
  const block = panel.querySelector<HTMLElement>('.tri-ana-pmc')
  const svgEl = block?.querySelector<SVGElement>('.tri-pmc-svg')
  const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
  const readout = block?.querySelector<HTMLElement>('.tri-chart-readout')
  if (!block || !svgEl || !cursor || !readout || daily.length < 2) return () => {}
  const onMove = (event: MouseEvent) => {
    const r = svgEl.getBoundingClientRect()
    const frac = clampN((event.clientX - r.left) / r.width, 0, 1)
    const d = daily[Math.round(frac * (daily.length - 1))]
    cursor.setAttribute('x1', `${(frac * ANA_W).toFixed(2)}`)
    cursor.setAttribute('x2', `${(frac * ANA_W).toFixed(2)}`)
    readout.textContent = `${d.date} · CTL ${Math.round(d.ctl)} ATL ${Math.round(d.atl)} TSB ${signed(Math.round(d.tsb))}`
    block.classList.add('tri-chart--hover')
  }
  const onLeave = () => block.classList.remove('tri-chart--hover')
  svgEl.addEventListener('mousemove', onMove)
  svgEl.addEventListener('mouseleave', onLeave)
  return () => {
    svgEl.removeEventListener('mousemove', onMove)
    svgEl.removeEventListener('mouseleave', onLeave)
  }
}

const SEARCH_SECTIONS: { label: string; chart: string; hay: string }[] = [
  { label: 'body weight', chart: 'body', hay: 'body weight kg lbs mass cut' },
  { label: 'form · ramp', chart: 'gauge', hay: 'form ramp gauge taper peak projection' },
  { label: 'fitness · fatigue · form', chart: 'pmc', hay: 'pmc fitness fatigue form ctl atl tsb' },
  {
    label: 'fitness by discipline',
    chart: 'ctl-sport',
    hay: 'fitness discipline swim bike run ctl',
  },
  { label: 'weekly load', chart: 'weekly', hay: 'weekly load volume tss' },
  { label: 'relative effort', chart: 'effort', hay: 'relative effort suffer score weekly' },
  {
    label: 'race readiness',
    chart: 'readiness',
    hay: 'race readiness predicted time sprint olympic 70.3 ironman binding leg',
  },
  {
    label: 'pace trend + forecast',
    chart: 'trend',
    hay: 'pace trend forecast threshold faster slower ewma ols',
  },
  { label: 'things to improve', chart: 'actions', hay: 'actions things improve weakest' },
]

const GLOSS_CHART: Record<string, string> = {
  ctl: 'pmc',
  atl: 'pmc',
  tsb: 'gauge',
  acwr: 'gauge',
  ramp: 'gauge',
  monotony: 'gauge',
  strain: 'gauge',
  load: 'weekly',
  effort: 'effort',
  score: 'readiness',
  binding: 'readiness',
  predtime: 'readiness',
  conf: 'trend',
  threshold: 'trend',
  trend: 'trend',
  weight: 'body',
  wtrend: 'body',
}

const setupAnalytics = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-analytics-btn')
  const panel = root.querySelector<HTMLElement>('.tri-analytics')
  const scrim = root.querySelector<HTMLElement>('.tri-analytics-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-ana-close')
  const title = root.querySelector<HTMLElement>('.tri-ana-title')
  const headline = root.querySelector<HTMLElement>('.tri-ana-headline')
  const search = root.querySelector<HTMLInputElement>('.tri-ana-search')
  const results = root.querySelector<HTMLElement>('.tri-ana-results')
  if (!btn || !panel) return null

  const body = root.querySelector<HTMLElement>('.tri-ana-body')
  const detail = root.querySelector<HTMLElement>('.tri-ana-detail')
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let loaded = false
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailLoaded = false
  let scrubCleanup: (() => void) | null = null
  let selIndex = -1

  const render = (d: Analytics) => {
    data = d
    if (headline) headline.replaceChildren(buildHeadline(d))
    for (const block of Array.from(panel.querySelectorAll<HTMLElement>('.tri-ana-block'))) {
      const build = ANALYTICS_BUILDERS[block.dataset.chart ?? '']
      if (build) block.replaceChildren(build(d))
    }
    scrubCleanup = wireScrub(panel, d.daily)
    document.dispatchEvent(
      new CustomEvent('contentdecrypted', { detail: { article: panel, content: panel } }),
    )
  }
  const load = () => {
    if (loaded) return
    loaded = true
    const path = root.dataset.analyticsPath
    if (!path) return
    fetch(path)
      .then(res => res.json())
      .then((d: Analytics) => render(d))
      .catch(() => {})
  }
  const closeDetail = () => {
    panel.classList.remove('tri-analytics--detail')
    if (detail) detail.replaceChildren()
  }
  const toMain = () => {
    closeDetail()
    if (search) search.value = ''
    panel.classList.remove('tri-analytics--searching')
    if (results) results.replaceChildren()
    selIndex = -1
  }
  const close = () => {
    root.classList.remove('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'true')
    if (search) search.value = ''
    panel.classList.remove('tri-analytics--searching')
    closeDetail()
    if (results) results.replaceChildren()
    selIndex = -1
  }
  const loadDetails = (): Promise<void> => {
    if (detailLoaded) return Promise.resolve()
    detailLoaded = true
    const p = root.dataset.detailPath
    if (!p) return Promise.resolve()
    return fetch(p)
      .then(res => res.json())
      .then((d: DetailPayload) => {
        detailData = d
        DETAIL_ZONES = d.zones ?? null
        DETAIL_CURVE_REF = d.powerCurveRef ?? []
      })
      .catch(() => {})
  }
  const showActivity = (id: string) => {
    if (!detail) return
    void loadDetails().then(() => {
      const d = detailData?.details?.[id]
      if (!d) return
      const card = el('div', 'tri-pop-card')
      const head = el('div', 'tri-pop-head')
      const back = el('button', 'tri-ana-back')
      back.setAttribute('type', 'button')
      back.textContent = '← back'
      head.append(el('span', 'tri-pop-date', shortDate(d.date)), back)
      card.appendChild(head)
      const act = renderDetail(d)
      act.classList.add('tri-act--expanded')
      card.appendChild(act)
      const h = detailData?.health?.[d.date]
      if (h) {
        const rec = buildRecovery(h)
        if (rec) card.appendChild(rec)
      }
      detail.replaceChildren(card)
      panel.classList.add('tri-analytics--detail')
      back.addEventListener('click', closeDetail, { once: true })
      body?.scrollTo({ top: 0 })
    })
  }

  const scrollToChart = (chart: string) => {
    const block = panel.querySelector<HTMLElement>(`.tri-ana-block[data-chart="${chart}"]`)
    if (search) search.value = ''
    panel.classList.remove('tri-analytics--searching')
    block?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    block?.classList.add('tri-ana-block--flash')
    window.setTimeout(() => block?.classList.remove('tri-ana-block--flash'), 900)
  }
  const ritem = (title: HTMLElement | string, sub: string): HTMLElement => {
    const it = el('button', 'tri-ana-ritem')
    it.setAttribute('type', 'button')
    const t = el('span', 'tri-ana-ritem-t')
    if (typeof title === 'string') t.textContent = title
    else t.appendChild(title)
    it.append(t, el('span', 'tri-ana-ritem-s', sub))
    return it
  }
  const matchHay = (hay: string, tokens: string[]): boolean => tokens.every(t => hay.includes(t))
  const resultItems = (): HTMLElement[] =>
    results ? Array.from(results.querySelectorAll<HTMLElement>('.tri-ana-ritem')) : []
  const setSel = (i: number) => {
    const its = resultItems()
    if (its.length === 0) {
      selIndex = -1
      return
    }
    selIndex = ((i % its.length) + its.length) % its.length
    its.forEach((it, k) => it.classList.toggle('tri-ana-ritem--sel', k === selIndex))
    its[selIndex].scrollIntoView({ block: 'nearest' })
  }
  const activate = (it: HTMLElement | undefined) => {
    if (!it) return
    if (it.dataset.chart) scrollToChart(it.dataset.chart)
    else if (it.dataset.id) showActivity(it.dataset.id)
    else if (it.dataset.insert) {
      const tokens = search!.value.trim().split(/\s+/)
      tokens[tokens.length - 1] = it.dataset.insert
      search!.value = tokens.join(' ') + (it.dataset.insert.endsWith(':') ? '' : ' ')
      search!.focus()
      runSearch()
    }
  }
  const runSearch = () => {
    if (!search || !results) return
    const q = search.value.trim().toLowerCase()
    results.replaceChildren()
    if (!q) {
      panel.classList.remove('tri-analytics--searching')
      results.setAttribute('aria-hidden', 'true')
      return
    }
    panel.classList.add('tri-analytics--searching')
    results.setAttribute('aria-hidden', 'false')
    const rawTokens = q.split(/\s+/)
    let filterSport: string | null = null
    let sortKey: string | null = null
    const tokens: string[] = []

    for (const t of rawTokens) {
      if (t.startsWith('filter:')) {
        filterSport = t.slice(7)
      } else if (t.startsWith('sort:')) {
        sortKey = t.slice(5)
      } else {
        if (t) tokens.push(t)
      }
    }

    const metrics: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1]
    const hints: HTMLElement[] = []

    if (lastToken.startsWith('filter:') && !['bike', 'run', 'swim'].includes(lastToken.slice(7))) {
      const prefix = lastToken.slice(7)
      for (const f of ['bike', 'run', 'swim']) {
        if (f.startsWith(prefix)) {
          const it = ritem(`filter:${f}`, 'filter activities')
          it.dataset.insert = `filter:${f}`
          hints.push(it)
        }
      }
    } else if (
      lastToken.startsWith('sort:') &&
      !['distance', 'cadence', 'pace'].includes(lastToken.slice(5))
    ) {
      const prefix = lastToken.slice(5)
      for (const s of ['distance', 'cadence', 'pace']) {
        if (s.startsWith(prefix)) {
          const it = ritem(`sort:${s}`, 'sort activities')
          it.dataset.insert = `sort:${s}`
          hints.push(it)
        }
      }
    } else if (lastToken.length > 0 && 'filter:'.startsWith(lastToken) && lastToken !== 'filter:') {
      const it = ritem('filter:', 'filter by sport (bike, run, swim)')
      it.dataset.insert = 'filter:'
      hints.push(it)
    } else if (lastToken.length > 0 && 'sort:'.startsWith(lastToken) && lastToken !== 'sort:') {
      const it = ritem('sort:', 'sort by distance, cadence, pace')
      it.dataset.insert = 'sort:'
      hints.push(it)
    }

    if (!filterSport && !sortKey) {
      for (const s of SEARCH_SECTIONS)
        if (matchHay(`${s.label} ${s.hay}`.toLowerCase(), tokens)) {
          const it = ritem(s.label, 'section')
          it.dataset.chart = s.chart
          metrics.push(it)
        }
      for (const key of Object.keys(GLOSS)) {
        const g = GLOSS[key]
        if (matchHay(`${key} ${g.term} ${g.def}`.toLowerCase(), tokens)) {
          const it = ritem(g.term, g.def)
          it.dataset.chart = GLOSS_CHART[key] ?? 'pmc'
          metrics.push(it)
        }
      }
    }

    let acts = (data?.activities ?? []).filter(a => {
      if (filterSport && a.sport !== filterSport) return false
      return tokens.length === 0 || matchHay(`${a.name} ${a.sport} ${a.date}`.toLowerCase(), tokens)
    })

    if (sortKey) {
      acts.sort((a, b) => {
        if (sortKey === 'distance') return b.distanceKm - a.distanceKm
        if (sortKey === 'cadence') return (b.cadence ?? 0) - (a.cadence ?? 0)
        if (sortKey === 'pace') {
          const speedA = a.movingTimeS > 0 ? a.distanceKm / a.movingTimeS : 0
          const speedB = b.movingTimeS > 0 ? b.distanceKm / b.movingTimeS : 0
          return speedB - speedA
        }
        return 0
      })
    }

    if (hints.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
      for (const it of hints) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (metrics.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'metrics & terms'))
      for (const it of metrics.slice(0, 8)) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'activities'))
      for (const a of acts.slice(0, 50)) {
        const head = el('span', 'tri-ana-ritem-h')
        head.append(buildIcon(a.sport), el('span', '', a.name || a.sport))
        const sub =
          `${a.date} · ${dist(a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}` +
          (a.cadence ? ` · ${a.cadence} rpm/spm` : '')
        const it = ritem(head, sub)
        it.dataset.id = String(a.id)
        grp.appendChild(it)
      }
      results.appendChild(grp)
    }
    if (!metrics.length && !acts.length && !hints.length)
      results.appendChild(el('div', 'tri-ana-empty', 'no matches'))
    setSel(0)
  }
  const onResultsClick = (event: MouseEvent) => {
    activate(
      (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-ana-ritem') ?? undefined,
    )
  }
  const onSearchKey = (event: KeyboardEvent) => {
    if (!panel.classList.contains('tri-analytics--searching')) return
    if (event.key === 'ArrowDown' || (event.ctrlKey && (event.key === 'n' || event.key === 'N'))) {
      event.preventDefault()
      setSel(selIndex + 1)
    } else if (
      event.key === 'ArrowUp' ||
      (event.ctrlKey && (event.key === 'p' || event.key === 'P'))
    ) {
      event.preventDefault()
      setSel(selIndex - 1)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const its = resultItems()
      activate(its[selIndex] ?? its[0])
    }
  }

  const open = () => {
    root.classList.add('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    if (reduce) return
    const br = btn.getBoundingClientRect()
    const pr = panel.getBoundingClientRect()
    if (pr.width < 1 || pr.height < 1) return
    const dx = br.left + br.width / 2 - (pr.left + pr.width / 2)
    const dy = br.top + br.height / 2 - (pr.top + pr.height / 2)
    const sx = Math.max(0.05, br.width / pr.width)
    const sy = Math.max(0.05, br.height / pr.height)
    panel.animate(
      [
        {
          opacity: 0,
          transform: `translate(-50%, -50%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
        },
        { opacity: 1, transform: 'translate(-50%, -50%) scale(1, 1)' },
      ],
      { duration: 300, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
    )
  }
  const onDetailToggle = (event: MouseEvent) => {
    const t = (event.target as HTMLElement | null)?.closest('.tri-act-toggle')
    t?.closest('.tri-act')?.classList.toggle('tri-act--expanded')
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (panel.classList.contains('tri-analytics--detail')) {
      closeDetail()
      return
    }
    if (search && search.value) {
      search.value = ''
      runSearch()
      return
    }
    close()
  }

  btn.addEventListener('click', open)
  closeBtn?.addEventListener('click', close)
  title?.addEventListener('click', toMain)
  scrim?.addEventListener('click', close)
  search?.addEventListener('input', runSearch)
  search?.addEventListener('keydown', onSearchKey)
  results?.addEventListener('click', onResultsClick)
  detail?.addEventListener('click', onDetailToggle)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    results?.removeEventListener('click', onResultsClick)
    detail?.removeEventListener('click', onDetailToggle)
    document.removeEventListener('keydown', onKey)
    scrubCleanup?.()
  }
}

const HEAT_RAMP = ['#997c6d', '#a9745b', '#b96c4a', '#ca6538', '#da5d27', '#ea5515', '#fc4c02']

type GeoFC = { type: 'FeatureCollection'; features: unknown[] }
const emptyFC = (): GeoFC => ({ type: 'FeatureCollection', features: [] })

const lineFeature = (
  route: StravaActivityDetail['route'],
  props: Record<string, unknown> = {},
) => ({
  type: 'Feature',
  properties: props,
  geometry: { type: 'LineString', coordinates: route.map(p => [p.lng, p.lat]) },
})

const heatFC = (dp: DetailPayload | null): GeoFC => {
  const features: unknown[] = []
  const det = dp?.details ?? {}
  for (const k in det) {
    const d = det[k]
    if ((d.sport === 'run' || d.sport === 'bike') && d.route.length >= 2)
      features.push(lineFeature(d.route, { id: d.id }))
  }
  return { type: 'FeatureCollection', features }
}

const routeFC = (d: StravaActivityDetail): GeoFC => ({
  type: 'FeatureCollection',
  features: [lineFeature(d.route)],
})

const pointFC = (lng: number, lat: number): GeoFC => ({
  type: 'FeatureCollection',
  features: [
    { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [lng, lat] } },
  ],
})

const gradientExpr = (
  d: StravaActivityDetail,
  pick: (p: StravaActivityDetail['route'][number], i: number) => number,
): unknown[] => {
  const vals = d.route.map((p, i) => pick(p, i))
  let lo = Infinity
  let hi = -Infinity
  for (const v of vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const range = hi > lo ? hi - lo : 1
  const dN = d.route[d.route.length - 1].d || 1
  const pairs: [number, string][] = []
  let lastT = -1
  d.route.forEach((p, i) => {
    const t = Math.min(1, Math.max(0, p.d / dN))
    if (t <= lastT) return
    lastT = t
    const bucket = Math.min(7, Math.max(1, Math.ceil(((vals[i] - lo) / range) * 7) || 1))
    pairs.push([t, HEAT_RAMP[bucket - 1]])
  })
  if (pairs.length === 0) pairs.push([0, HEAT_RAMP[3]])
  if (pairs[0][0] > 0) pairs.unshift([0, pairs[0][1]])
  if (pairs[pairs.length - 1][0] < 1) pairs.push([1, pairs[pairs.length - 1][1]])
  const stops: unknown[] = ['interpolate', ['linear'], ['line-progress']]
  for (const [t, c] of pairs) stops.push(t, c)
  return stops
}

const fcBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  let minLng = Infinity
  let minLat = Infinity
  let maxLng = -Infinity
  let maxLat = -Infinity
  for (const f of fc.features) {
    const coords = (f as { geometry: { coordinates: [number, number][] } }).geometry.coordinates
    for (const [lng, lat] of coords) {
      if (lng < minLng) minLng = lng
      if (lng > maxLng) maxLng = lng
      if (lat < minLat) minLat = lat
      if (lat > maxLat) maxLat = lat
    }
  }
  return Number.isFinite(minLng)
    ? [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    : null
}

const setupMap = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-map-btn')
  const panel = root.querySelector<HTMLElement>('.tri-map')
  const scrim = root.querySelector<HTMLElement>('.tri-map-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-map-close')
  const title = root.querySelector<HTMLElement>('.tri-map-title')
  const search = root.querySelector<HTMLInputElement>('.tri-map-search')
  const results = root.querySelector<HTMLElement>('.tri-map-results')
  const detail = root.querySelector<HTMLElement>('.tri-map-detail')
  if (!btn || !panel) return null

  const body = root.querySelector<HTMLElement>('.tri-map-body')
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let loaded = false
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailLoaded = false
  let selIndex = -1
  const canvas = root.querySelector<HTMLElement>('.tri-map-canvas')

  const mapCtl = (() => {
    let map: any = null
    let started = false
    let okFlag = false
    const recolor = (d: StravaActivityDetail, i: number) => {
      if (!map) return
      const spec = metricSpecs(d)[i]
      if (spec) map.setPaintProperty('tri-sel', 'line-gradient', gradientExpr(d, spec.pick))
    }
    const init = async (): Promise<void> => {
      if (started) return
      started = true
      if (!canvas) return
      const mapboxgl = await loadMapbox()
      if (!mapboxgl) {
        started = false
        canvas.classList.add('tri-map-canvas--down')
        canvas.textContent = 'map unavailable'
        return
      }
      canvas.classList.remove('tri-map-canvas--down')
      canvas.textContent = ''
      map = new mapboxgl.Map({
        container: canvas,
        style: 'mapbox://styles/mapbox/light-v11',
        center: [-79.4, 43.7],
        zoom: 9,
        attributionControl: false,
      })
      ;(canvas as unknown as { _mapInstance: unknown })._mapInstance = map
      await new Promise<void>(resolve => map.once('load', () => resolve()))
      applyMonochromeMapPalette(map)
      map.addSource('tri-heat', { type: 'geojson', data: emptyFC() })
      map.addLayer({
        id: 'tri-heat',
        type: 'line',
        source: 'tri-heat',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#fc4c02',
          'line-opacity': 0.2,
          'line-blur': 0.6,
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 1.2, 14, 2, 16, 3],
        },
      })
      map.addSource('tri-sel', { type: 'geojson', lineMetrics: true, data: emptyFC() })
      map.addLayer({
        id: 'tri-sel-casing',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fff9f3', 'line-width': 5 },
      })
      map.addLayer({
        id: 'tri-sel',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-width': 3.4,
          'line-gradient': [
            'interpolate',
            ['linear'],
            ['line-progress'],
            0,
            '#fc4c02',
            1,
            '#fc4c02',
          ],
        },
      })
      map.addSource('tri-dot', { type: 'geojson', data: emptyFC() })
      map.addLayer({
        id: 'tri-dot',
        type: 'circle',
        source: 'tri-dot',
        paint: {
          'circle-radius': 4,
          'circle-color': '#fc4c02',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#fff9f3',
        },
      })
      okFlag = true
    }
    const drawHeatmap = () => {
      if (!map) return
      const fc = heatFC(detailData)
      map.getSource('tri-heat')?.setData(fc)
      const b = fcBounds(fc)
      if (b) map.fitBounds(b, { padding: 48, maxZoom: 13, duration: reduce ? 0 : 600 })
    }
    const select = (d: StravaActivityDetail, i: number) => {
      if (!map) return
      map.getSource('tri-sel')?.setData(routeFC(d))
      recolor(d, i)
      map.setPaintProperty('tri-heat', 'line-opacity', 0.08)
      const b = fcBounds(routeFC(d))
      if (b) map.fitBounds(b, { padding: 40, maxZoom: 15, duration: reduce ? 0 : 600 })
    }
    const moveDot = (lng: number, lat: number) =>
      map?.getSource('tri-dot')?.setData(pointFC(lng, lat))
    const clearSelection = () => {
      if (!map) return
      map.getSource('tri-sel')?.setData(emptyFC())
      map.getSource('tri-dot')?.setData(emptyFC())
      map.setPaintProperty('tri-heat', 'line-opacity', 0.2)
      drawHeatmap()
    }
    const resize = () => map?.resize()
    const dispose = () => {
      if (map?.remove) map.remove()
      map = null
      started = false
      okFlag = false
      if (canvas) (canvas as unknown as { _mapInstance: unknown })._mapInstance = null
    }
    return {
      init,
      ok: () => okFlag,
      drawHeatmap,
      select,
      recolor,
      moveDot,
      clearSelection,
      resize,
      dispose,
    }
  })()

  const load = () => {
    if (loaded) return
    loaded = true
    const path = root.dataset.analyticsPath
    if (!path) return
    fetch(path)
      .then(res => res.json())
      .then((d: Analytics) => {
        data = d
        if (search?.value) runSearch()
      })
      .catch(() => {})
  }
  const loadDetails = (): Promise<void> => {
    if (detailLoaded) return Promise.resolve()
    detailLoaded = true
    const p = root.dataset.detailPath
    if (!p) return Promise.resolve()
    return fetch(p)
      .then(res => res.json())
      .then((d: DetailPayload) => {
        detailData = d
        DETAIL_ZONES = d.zones ?? null
        DETAIL_CURVE_REF = d.powerCurveRef ?? []
        if (search?.value) runSearch()
      })
      .catch(() => {})
  }
  const closeDetail = () => {
    panel.classList.remove('tri-map--detail')
    if (detail) detail.replaceChildren()
    mapCtl.clearSelection()
    requestAnimationFrame(() => mapCtl.resize())
  }
  const toMain = () => {
    closeDetail()
    if (search) search.value = ''
    panel.classList.remove('tri-map--searching')
    if (results) results.replaceChildren()
    selIndex = -1
  }
  const close = () => {
    root.classList.remove('tri-map-open')
    panel.setAttribute('aria-hidden', 'true')
    toMain()
  }
  const showRoute = (id: string) => {
    if (!detail) return
    void loadDetails().then(() => {
      const d = detailData?.details?.[id]
      if (!d) return
      const card = el('div', 'tri-pop-card')
      const head = el('div', 'tri-pop-head')
      const back = el('button', 'tri-ana-back')
      back.setAttribute('type', 'button')
      back.textContent = '← back'
      head.append(el('span', 'tri-pop-date', `${shortDate(d.date)} · ${d.name || d.sport}`), back)
      card.appendChild(head)
      const mapMode = mapCtl.ok()
      card.appendChild(
        mapMode
          ? renderMapDetail(d, {
              mapMode: true,
              onMetric: i => mapCtl.recolor(d, i),
              onHover: p => mapCtl.moveDot(p.lng, p.lat),
            })
          : renderMapDetail(d),
      )
      detail.replaceChildren(card)
      panel.classList.add('tri-map--detail')
      back.addEventListener('click', closeDetail, { once: true })
      if (mapMode) {
        requestAnimationFrame(() => {
          mapCtl.resize()
          mapCtl.select(d, 0)
        })
      } else {
        body?.scrollTo({ top: 0 })
      }
    })
  }
  const ritem = (titleEl: HTMLElement | string, sub: string): HTMLElement => {
    const it = el('button', 'tri-ana-ritem')
    it.setAttribute('type', 'button')
    const t = el('span', 'tri-ana-ritem-t')
    if (typeof titleEl === 'string') t.textContent = titleEl
    else t.appendChild(titleEl)
    it.append(t, el('span', 'tri-ana-ritem-s', sub))
    return it
  }
  const matchHay = (hay: string, tokens: string[]): boolean => tokens.every(t => hay.includes(t))
  const resultItems = (): HTMLElement[] =>
    results ? Array.from(results.querySelectorAll<HTMLElement>('.tri-ana-ritem')) : []
  const setSel = (i: number) => {
    const its = resultItems()
    if (its.length === 0) {
      selIndex = -1
      return
    }
    selIndex = ((i % its.length) + its.length) % its.length
    its.forEach((it, k) => it.classList.toggle('tri-ana-ritem--sel', k === selIndex))
    its[selIndex].scrollIntoView({ block: 'nearest' })
  }
  const activate = (it: HTMLElement | undefined) => {
    if (!it) return
    if (it.dataset.id) showRoute(it.dataset.id)
    else if (it.dataset.insert) {
      const tokens = search!.value.trim().split(/\s+/)
      tokens[tokens.length - 1] = it.dataset.insert
      search!.value = tokens.join(' ') + (it.dataset.insert.endsWith(':') ? '' : ' ')
      search!.focus()
      runSearch()
    }
  }
  const drawableIds = (): Set<string> => {
    const ids = new Set<string>()
    const det = detailData?.details ?? {}
    for (const k in det) if ((det[k].route?.length ?? 0) >= 2) ids.add(k)
    return ids
  }
  const runSearch = () => {
    if (!search || !results) return
    const q = search.value.trim().toLowerCase()
    results.replaceChildren()
    if (!q) {
      panel.classList.remove('tri-map--searching')
      results.setAttribute('aria-hidden', 'true')
      return
    }
    panel.classList.add('tri-map--searching')
    results.setAttribute('aria-hidden', 'false')
    const rawTokens = q.split(/\s+/)
    let filterSport: string | null = null
    const tokens: string[] = []
    for (const t of rawTokens) {
      if (t.startsWith('filter:')) filterSport = t.slice(7)
      else if (t) tokens.push(t)
    }
    const hints: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1]
    if (lastToken.startsWith('filter:') && !['bike', 'run'].includes(lastToken.slice(7))) {
      const prefix = lastToken.slice(7)
      for (const f of ['bike', 'run'])
        if (f.startsWith(prefix)) {
          const it = ritem(`filter:${f}`, 'filter routes')
          it.dataset.insert = `filter:${f}`
          hints.push(it)
        }
    } else if (lastToken.length > 0 && 'filter:'.startsWith(lastToken) && lastToken !== 'filter:') {
      const it = ritem('filter:', 'filter by sport (bike, run)')
      it.dataset.insert = 'filter:'
      hints.push(it)
    }
    const ids = drawableIds()
    const acts = (data?.activities ?? []).filter(a => {
      if (!ids.has(String(a.id))) return false
      if (filterSport && a.sport !== filterSport) return false
      return tokens.length === 0 || matchHay(`${a.name} ${a.sport} ${a.date}`.toLowerCase(), tokens)
    })
    if (hints.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
      for (const it of hints) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'routes'))
      for (const a of acts.slice(0, 50)) {
        const head = el('span', 'tri-ana-ritem-h')
        head.append(buildIcon(a.sport), el('span', '', a.name || a.sport))
        const sub = `${a.date} · ${dist(a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}`
        const it = ritem(head, sub)
        it.dataset.id = String(a.id)
        grp.appendChild(it)
      }
      results.appendChild(grp)
    }
    if (!acts.length && !hints.length)
      results.appendChild(el('div', 'tri-ana-empty', detailLoaded ? 'no routes' : 'loading…'))
    setSel(0)
  }
  const onResultsClick = (event: MouseEvent) => {
    activate(
      (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-ana-ritem') ?? undefined,
    )
  }
  const onSearchKey = (event: KeyboardEvent) => {
    if (!panel.classList.contains('tri-map--searching')) return
    if (event.key === 'ArrowDown' || (event.ctrlKey && (event.key === 'n' || event.key === 'N'))) {
      event.preventDefault()
      setSel(selIndex + 1)
    } else if (
      event.key === 'ArrowUp' ||
      (event.ctrlKey && (event.key === 'p' || event.key === 'P'))
    ) {
      event.preventDefault()
      setSel(selIndex - 1)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const its = resultItems()
      activate(its[selIndex] ?? its[0])
    }
  }
  const startMap = () =>
    void mapCtl.init().then(() => {
      mapCtl.resize()
      mapCtl.drawHeatmap()
    })
  const open = () => {
    root.classList.add('tri-map-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    void loadDetails()
    if (reduce) {
      startMap()
      return
    }
    const br = btn.getBoundingClientRect()
    const pr = panel.getBoundingClientRect()
    if (pr.width < 1 || pr.height < 1) {
      startMap()
      return
    }
    const dx = br.left + br.width / 2 - (pr.left + pr.width / 2)
    const dy = br.top + br.height / 2 - (pr.top + pr.height / 2)
    const sx = Math.max(0.05, br.width / pr.width)
    const sy = Math.max(0.05, br.height / pr.height)
    const anim = panel.animate(
      [
        {
          opacity: 0,
          transform: `translate(-50%, -50%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
        },
        { opacity: 1, transform: 'translate(-50%, -50%) scale(1, 1)' },
      ],
      { duration: 300, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
    )
    anim.finished.then(startMap).catch(startMap)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (panel.classList.contains('tri-map--detail')) {
      closeDetail()
      return
    }
    if (search && search.value) {
      search.value = ''
      runSearch()
      return
    }
    close()
  }

  btn.addEventListener('click', open)
  closeBtn?.addEventListener('click', close)
  title?.addEventListener('click', toMain)
  scrim?.addEventListener('click', close)
  search?.addEventListener('input', runSearch)
  search?.addEventListener('keydown', onSearchKey)
  results?.addEventListener('click', onResultsClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    results?.removeEventListener('click', onResultsClick)
    document.removeEventListener('keydown', onKey)
    mapCtl.dispose()
  }
}

const setupCheat = (root: HTMLElement): (() => void) | null => {
  const unit = root.querySelector<HTMLButtonElement>('.tri-cheat-unit')
  const cells = root.querySelectorAll<HTMLElement>('.tri-cheat td[data-km]')
  if (!unit || cells.length === 0) return null

  const target = root.querySelector<HTMLElement>('.tri-cheat-target')
  let ann: RoughAnnotation | null = null
  let showTimer = 0
  if (target) {
    const color = getComputedStyle(root).getPropertyValue('--tri-accent').trim() || '#fc4c02'
    ann = annotate(target, {
      type: 'circle',
      color,
      strokeWidth: 1.6,
      padding: 5,
      animationDuration: 800,
      iterations: 2,
    })
    const a = ann
    showTimer = window.setTimeout(() => a.show(), 200)
  }

  let mi = false
  const onClick = () => {
    mi = !mi
    unit.textContent = mi ? 'mi' : 'km'
    for (const c of cells) {
      if (!mi) {
        c.textContent = c.dataset.km ?? ''
      } else {
        const v = Number(c.dataset.km) * KM_TO_MI
        c.textContent = v < 10 ? v.toFixed(2) : v.toFixed(1)
      }
    }
  }
  unit.addEventListener('click', onClick)
  return () => {
    window.clearTimeout(showTimer)
    unit.removeEventListener('click', onClick)
    ann?.remove()
  }
}

const renderGlossDef = (def: string): HTMLElement => {
  const span = el('span', 'tri-gloss-def')
  def.split(/\$([^$]+)\$/).forEach((part, i) => {
    if (i % 2 === 1) {
      const m = el('span', 'tri-gloss-math')
      m.innerHTML = katex.renderToString(part, {
        displayMode: false,
        output: 'html',
        throwOnError: false,
        strict: false,
      })
      span.appendChild(m)
    } else if (part) {
      span.appendChild(document.createTextNode(part))
    }
  })
  return span
}

const setupGloss = (root: HTMLElement): (() => void) | null => {
  const zones = ['.tri-analytics', '.tri-ana-headline']
    .map(s => root.querySelector<HTMLElement>(s))
    .filter((z): z is HTMLElement => z != null)
  if (zones.length === 0) return null
  const pop = el('div', 'tri-gloss')
  pop.setAttribute('role', 'tooltip')
  root.appendChild(pop)
  let current: HTMLElement | null = null
  const place = (term: HTMLElement) => {
    const r = term.getBoundingClientRect()
    const pr = pop.getBoundingClientRect()
    let left = r.left
    if (left + pr.width > window.innerWidth - 8) left = window.innerWidth - 8 - pr.width
    let top = r.bottom + 6
    if (top + pr.height > window.innerHeight - 8) top = r.top - 6 - pr.height
    pop.style.left = `${Math.max(8, left)}px`
    pop.style.top = `${Math.max(8, top)}px`
  }
  const show = (term: HTMLElement) => {
    const g = GLOSS[term.dataset.gloss ?? '']
    if (!g) return
    current = term
    pop.replaceChildren(el('span', 'tri-gloss-h', g.term), renderGlossDef(g.def))
    pop.classList.add('tri-gloss--on')
    place(term)
  }
  const hide = (term?: HTMLElement) => {
    if (term && term !== current) return
    current = null
    pop.classList.remove('tri-gloss--on')
  }
  const onOver = (event: Event) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (t) show(t)
  }
  const onOut = (event: Event) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (!t) return
    const to = (event as MouseEvent).relatedTarget as Node | null
    if (to && t.contains(to)) return
    hide(t)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') hide()
  }
  for (const z of zones) {
    z.addEventListener('mouseover', onOver)
    z.addEventListener('mouseout', onOut)
    z.addEventListener('focusin', onOver)
    z.addEventListener('focusout', onOut)
  }
  document.addEventListener('keydown', onKey)
  return () => {
    for (const z of zones) {
      z.removeEventListener('mouseover', onOver)
      z.removeEventListener('mouseout', onOut)
      z.removeEventListener('focusin', onOver)
      z.removeEventListener('focusout', onOut)
    }
    document.removeEventListener('keydown', onKey)
    pop.remove()
  }
}

const setupShortcuts = (root: HTMLElement): (() => void) => {
  let waitingForG = false
  let gTimeout: number | null = null

  const onKey = (e: KeyboardEvent) => {
    const el = e.target as HTMLElement | null
    if (el) {
      const tag = el.tagName.toLowerCase()
      if (
        tag === 'input' ||
        tag === 'textarea' ||
        el.isContentEditable ||
        el.closest('.search-container') !== null
      ) {
        return
      }
    }

    if (e.ctrlKey || e.metaKey || e.altKey) return

    if (waitingForG) {
      const key = e.key.toLowerCase()
      let handled = false
      if (key === 'a') {
        root.querySelector<HTMLElement>('.tri-analytics-btn')?.click()
        handled = true
      } else if (key === 'g') {
        root.querySelector<HTMLElement>('.tri-gear-btn')?.click()
        handled = true
      } else if (key === 'p') {
        root.querySelector<HTMLElement>('.tri-pace-btn')?.click()
        handled = true
      } else if (key === 'c') {
        root.querySelector<HTMLElement>('.tri-calc-btn')?.click()
        handled = true
      } else if (key === 's') {
        root.querySelector<HTMLElement>('.tri-total')?.click()
        handled = true
      }

      if (handled) {
        e.preventDefault()
        e.stopImmediatePropagation()
      }

      waitingForG = false
      if (gTimeout) {
        clearTimeout(gTimeout)
        gTimeout = null
      }
    } else if (e.key.toLowerCase() === 'g') {
      waitingForG = true
      gTimeout = window.setTimeout(() => {
        waitingForG = false
      }, 1000)
    }
  }

  document.addEventListener('keydown', onKey, true)
  return () => document.removeEventListener('keydown', onKey, true)
}

const setupPaceUnit = (root: HTMLElement): (() => void) | null => {
  const buttons = root.querySelectorAll<HTMLButtonElement>('.tri-pace-unit')
  const cells = root.querySelectorAll<HTMLElement>('.tri-pace [data-kph]')
  if (buttons.length === 0 || cells.length === 0) return null
  let mph = false
  const onClick = () => {
    mph = !mph
    for (const b of buttons) b.textContent = mph ? 'mph' : 'km/h'
    for (const c of cells) c.textContent = (mph ? c.dataset.mph : c.dataset.kph) ?? ''
  }
  for (const b of buttons) b.addEventListener('click', onClick)
  return () => {
    for (const b of buttons) b.removeEventListener('click', onClick)
  }
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (!root) return
  const cleanup = setup(root)
  if (cleanup) window.addCleanup?.(cleanup)
  const calcCleanup = setupCalc(root)
  if (calcCleanup) window.addCleanup?.(calcCleanup)
  const gearCleanup = setupDropdown(
    root,
    '.tri-gear-wrap',
    '.tri-gear-btn',
    '.tri-gear',
    'tri-gear-open',
  )
  if (gearCleanup) window.addCleanup?.(gearCleanup)
  const paceCleanup = setupDropdown(
    root,
    '.tri-pace-wrap',
    '.tri-pace-btn',
    '.tri-pace',
    'tri-pace-open',
  )
  if (paceCleanup) window.addCleanup?.(paceCleanup)
  const paceUnitCleanup = setupPaceUnit(root)
  if (paceUnitCleanup) window.addCleanup?.(paceUnitCleanup)
  const cheatCleanup = setupCheat(root)
  if (cheatCleanup) window.addCleanup?.(cheatCleanup)
  const anaCleanup = setupAnalytics(root)
  if (anaCleanup) window.addCleanup?.(anaCleanup)
  const mapCleanup = setupMap(root)
  if (mapCleanup) window.addCleanup?.(mapCleanup)
  const glossCleanup = setupGloss(root)
  if (glossCleanup) window.addCleanup?.(glossCleanup)
  const shortcutsCleanup = setupShortcuts(root)
  if (shortcutsCleanup) window.addCleanup?.(shortcutsCleanup)
})
