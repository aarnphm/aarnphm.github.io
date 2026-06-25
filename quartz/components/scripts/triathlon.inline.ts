import type { RoughAnnotation } from 'rough-notation/lib/model'
import katex from 'katex'
import { annotate } from 'rough-notation'
import type {
  ActivitySummary,
  Analytics,
  BodyBlock,
  DailyPoint,
} from '../../plugins/stores/analytics'
import {
  type ActivityHealth,
  type ActivityKind,
  type StravaMapPoint,
  type PowerCurvePoint,
  type Sport,
  type StravaActivityDetail,
  type StravaZones,
} from '../../plugins/stores/strava'
import {
  computeTriathlonCalcTimes,
  formatDurationClock,
  parseClockSeconds,
  solveTriathlonCalcLeg,
  solveTriathlonCalcTarget,
  type TriathlonCalcInput,
  type TriathlonCalcLeg,
} from '../../util/triathlon-calculator'
import {
  buildActivity as buildActivityNode,
  buildDayCard as buildDayCardNode,
  buildElevation as buildElevationNode,
  buildIcon as buildIconNode,
  buildPool as buildPoolNode,
  buildRecovery as buildRecoveryNode,
  clock,
  dist,
  distCombined,
  dur,
  gradeAt,
  KM_TO_MI,
  rate,
  routeStreamFlags,
  scrubDist,
  setDistanceUnit,
  shortDate,
  statRow as statRowNode,
  type DayCardExtras,
  type TriNodeFactory,
} from '../../util/triathlon-card'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

export {}

type DetailPayload = {
  details: Record<string, StravaActivityDetail>
  health: Record<string, ActivityHealth>
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
  ftp?: number | null
  goalFtp?: number | null
}

type TrainingPlan = {
  id: string
  meta: string
  distance: string
  date: string
  target: string
  author: string
  html: string
}
type TrainingPayload = { plans: TrainingPlan[] }

let DETAIL_ZONES: StravaZones | null = null
let DETAIL_CURVE_REF: PowerCurvePoint[] = []
let DETAIL_FTP: number | null = null
let DETAIL_GOAL_FTP: number | null = null
let DETAIL_PAYLOAD: Promise<DetailPayload | null> | null = null

const loadDetailPayload = (path: string): Promise<DetailPayload | null> => {
  DETAIL_PAYLOAD ??= fetch(path)
    .then(res => res.json())
    .then((data: DetailPayload) => {
      DETAIL_ZONES = data.zones ?? null
      DETAIL_CURVE_REF = data.powerCurveRef ?? []
      DETAIL_FTP = data.ftp ?? null
      DETAIL_GOAL_FTP = data.goalFtp ?? null
      return data
    })
    .catch(() => null)
  return DETAIL_PAYLOAD
}

const SVGNS = 'http://www.w3.org/2000/svg'

const el = (
  tag: string,
  cls?: string,
  text?: string,
  attrs?: Record<string, string>,
): HTMLElement => {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k])
  return e
}

const svg = (tag: string, attrs: Record<string, string | number>): SVGElement => {
  const e = document.createElementNS(SVGNS, tag)
  for (const k in attrs) e.setAttribute(k, String(attrs[k]))
  return e
}

const mathFrag = (text: string): Node[] => {
  const out: Node[] = []
  text.split(/\$([^$]+)\$/).forEach((part, i) => {
    if (i % 2 === 1) {
      const m = el('span', 'tri-math')
      m.innerHTML = katex.renderToString(part, {
        displayMode: false,
        output: 'html',
        throwOnError: false,
        strict: false,
      })
      out.push(m)
    } else if (part) {
      out.push(document.createTextNode(part))
    }
  })
  return out
}

const setMath = (host: HTMLElement, text: string): void => {
  host.replaceChildren(...mathFrag(text))
}

const mathK = (cls: string, text: string): HTMLElement => {
  const span = el('span', cls)
  setMath(span, text)
  return span
}

const domF: TriNodeFactory<HTMLElement | SVGElement> = {
  el,
  svg,
  add: (parent, ...children) => parent.append(...children),
}

const buildIcon = (sport: ActivityKind): SVGElement => buildIconNode(domF, sport) as SVGElement

const buildElevation = (d: StravaActivityDetail): HTMLElement =>
  buildElevationNode(domF, d) as HTMLElement

const buildPool = (d: StravaActivityDetail): HTMLElement => buildPoolNode(domF, d) as HTMLElement

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

const statRow = (label: string, value: string): HTMLElement =>
  statRowNode(domF, label, value) as HTMLElement

const buildRecovery = (h: ActivityHealth): HTMLElement | null =>
  buildRecoveryNode(domF, h) as HTMLElement | null

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
  const ftpRef = DETAIL_FTP
  const goalRef = DETAIL_GOAL_FTP
  const wrap = el('div', 'tri-zone')
  wrap.appendChild(el('div', 'tri-zone-title', 'power curve'))
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
  const hline = (w: number, cls: string): SVGElement =>
    svg('line', { x1: 0, y1: Y(w).toFixed(2), x2: W, y2: Y(w).toFixed(2), class: cls })
  if (ftpRef != null) s.appendChild(hline(ftpRef, 'tri-curve-ftp'))
  if (goalRef != null) s.appendChild(hline(goalRef, 'tri-curve-goal'))
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
  if (ftpRef != null) cap.appendChild(el('span', 'tri-ana-k tri-curve-ftp-k', `FTP ${ftpRef}W`))
  if (goalRef != null) cap.appendChild(el('span', 'tri-ana-k tri-curve-goal-k', `goal ${goalRef}W`))
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
  colors?: string[],
  zeroGap = false,
): SVGElement => {
  const ramp = colors?.length ?? 7
  const pad = 6
  const span = 100 - pad * 2
  const vals = route.map((p, i) => pick(p, i))
  const pool = zeroGap ? vals.filter(v => v > 0) : vals
  let lo = Infinity
  let hi = -Infinity
  for (const v of pool.length ? pool : vals) {
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
    const attrs: Record<string, string | number> = {
      d: `M ${sx(route[i]).toFixed(2)} ${sy(route[i]).toFixed(2)} L ${sx(route[i + 1]).toFixed(2)} ${sy(route[i + 1]).toFixed(2)}`,
      class: colors ? 'tri-heat-seg' : `tri-heat-seg tri-heat--${bucket}`,
    }
    if (colors) attrs.style = `stroke: ${colors[bucket - 1]}`
    g.appendChild(svg('path', attrs))
  }
  s.appendChild(g)
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return s
}

const rampGradient = (colors: string[]): string =>
  `linear-gradient(to right, ${colors[0]}, ${colors[3]}, ${colors[6]})`

const buildHeatLegend = (
  lo: number,
  hi: number,
  fmt: (v: number) => string,
  colors?: string[],
): HTMLElement => {
  const wrap = el('div', 'tri-map-legend')
  const bar = el('span', 'tri-map-legend-bar')
  if (colors) bar.style.background = rampGradient(colors)
  wrap.append(
    el('span', 'tri-map-legend-lo', fmt(lo)),
    bar,
    el('span', 'tri-map-legend-hi', fmt(hi)),
  )
  return wrap
}

const HEAT_RAMP = ['#997c6d', '#a9745b', '#b96c4a', '#ca6538', '#da5d27', '#ea5515', '#fc4c02']

const ramp7 = (from: string, to: string): string[] => {
  const rgb = (h: string): number[] => [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ]
  const a = rgb(from)
  const b = rgb(to)
  return Array.from({ length: 7 }, (_, i) => {
    const t = i / 6
    return `#${a
      .map((v, j) =>
        Math.round(v + (b[j] - v) * t)
          .toString(16)
          .padStart(2, '0'),
      )
      .join('')}`
  })
}

const HR_RAMP = ramp7('#9c7f7a', '#af3029')
const CAD_RAMP = ramp7('#8a8197', '#5e409d')
const SPD_RAMP = ramp7('#7d8a96', '#205ea6')
const ELEV_RAMP = ramp7('#868a72', '#66800b')

interface MapMetric {
  label: string
  ramp: string[]
  zeroGap?: boolean
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
    ramp: SPD_RAMP,
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
    ramp: HEAT_RAMP,
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
    ramp: HR_RAMP,
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
  const cadScale = d.sport === 'run' ? 2 : 1
  const cadSpec: MapMetric = {
    label: 'cadence',
    ramp: CAD_RAMP,
    zeroGap: true,
    pick: p => p.cad * cadScale,
    fmt: v => `${Math.round(v)} ${cadUnit}`,
    profile: () =>
      buildTrace(
        d,
        p => p.cad * cadScale,
        'cadence',
        m => `${m} ${cadUnit} peak`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.cad * cadScale} ${cadUnit}`,
  }
  const elevSpec: MapMetric = {
    label: 'elevation',
    ramp: ELEV_RAMP,
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
  if (d.sport === 'strength') {
    sbody.appendChild(statRow('time', dur(d.movingTimeS)))
  } else {
    sbody.append(
      statRow('distance', dist(d.distanceKm, d.sport)),
      statRow('time', dur(d.movingTimeS)),
      statRow(d.sport === 'bike' ? 'speed' : 'pace', rate(d.sport, d.distanceKm, d.movingTimeS)),
    )
  }
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
    const pool = spec.zeroGap ? vals.filter(v => v > 0) : vals
    let lo = Infinity
    let hi = -Infinity
    for (const v of pool.length ? pool : vals) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    let marker: SVGElement | null = null
    if (opts?.mapMode) {
      figs.replaceChildren(buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
    } else {
      const heat = buildHeatRoute(d.route, spec.pick, spec.ramp, spec.zeroGap)
      figs.replaceChildren(heat, buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
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
    Array.from(tablist.children).forEach((t, i) => {
      const on = i === active
      t.setAttribute('aria-selected', on ? 'true' : 'false')
      const tab = t as HTMLElement
      tab.style.background = on ? spec.ramp[6] : ''
      tab.style.borderColor = on ? spec.ramp[6] : ''
    })
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
  const wrap = buildActivityNode(domF, d) as HTMLElement
  const surfaces: ScrubSurface[] = []
  const elev = wrap.querySelector<HTMLElement>('.tri-act-figs .tri-elev-wrap')
  if (elev) {
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
  }
  const more = wrap.querySelector<HTMLElement>(':scope > .tri-act-more')
  if (more) {
    const flags = routeStreamFlags(d)
    if (flags.hr) {
      const t = buildTrace(
        d,
        p => p.hr,
        'hr',
        m => `${m} bpm peak`,
      )
      more.appendChild(t)
      surfaces.push({ wrap: t, fmt: p => `${scrubDist(p.d, d.sport)} · ${p.hr} bpm` })
    }
    if (flags.power) {
      const t = buildTrace(
        d,
        p => p.w,
        'power',
        m => `${m} W peak`,
      )
      more.appendChild(t)
      surfaces.push({ wrap: t, fmt: p => `${scrubDist(p.d, d.sport)} · ${p.w} W` })
    }
    if (flags.cad) {
      const cadScale = d.sport === 'run' ? 2 : 1
      const cadUnit = d.sport === 'run' ? 'spm' : 'rpm'
      const t = buildTrace(
        d,
        p => p.cad * cadScale,
        'cadence',
        m => `${m} ${cadUnit} peak`,
      )
      more.appendChild(t)
      surfaces.push({
        wrap: t,
        fmt: p => `${scrubDist(p.d, d.sport)} · ${p.cad * cadScale} ${cadUnit}`,
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
  }
  if (surfaces.length > 0 && d.route.length >= 2) {
    const routeMarker = wrap.querySelector<SVGElement>('.tri-route-cursor')
    linkScrub(wrap, routeMarker, surfaces, d.route)
  }
  return wrap
}

const onCardToggle = (event: Event) => {
  const btn = (event.target as HTMLElement | null)?.closest('.tri-act-toggle')
  btn?.closest('.tri-act')?.classList.toggle('tri-act--expanded')
}

const buildDayCard = (
  dateIso: string,
  payload: DetailPayload | null,
  extras: DayCardExtras = {},
): HTMLElement => {
  const card = buildDayCardNode(domF, dateIso, payload, extras, renderDetail) as HTMLElement
  card.addEventListener('click', onCardToggle)
  return card
}

const dayExtrasFromDataset = (data: DOMStringMap): DayCardExtras => ({
  location: data.triathlonLoc,
  event: data.triathlonEvent,
})

const setupDayEmbeds = (): (() => void) | null => {
  const embeds = Array.from(
    document.querySelectorAll<HTMLElement>('.tri-day-embed[data-triathlon-date]'),
  )
  if (embeds.length === 0) return null
  let live = true
  const teardowns: (() => void)[] = []
  for (const embed of embeds) {
    const date = embed.dataset.triathlonDate!
    const extras = dayExtrasFromDataset(embed.dataset)
    const detailPath = embed.dataset.detailPath ?? '/static/strava-detail.json'
    let upgraded = false
    const upgrade = () => {
      if (upgraded) return
      upgraded = true
      void loadDetailPayload(detailPath).then(data => {
        if (!live || !embed.isConnected || !data) return
        const fresh = buildDayCard(date, data, extras)
        const expanded = Array.from(embed.querySelectorAll('.tri-act'), a =>
          a.classList.contains('tri-act--expanded'),
        )
        fresh.querySelectorAll('.tri-act').forEach((a, i) => {
          if (expanded[i]) a.classList.add('tri-act--expanded')
        })
        embed.replaceChildren(fresh)
      })
    }
    const ssr = embed.querySelector<HTMLElement>(':scope > .tri-pop-card')
    if (ssr) {
      ssr.addEventListener('click', onCardToggle)
      const events = ['pointerenter', 'focusin', 'touchstart'] as const
      for (const ev of events) embed.addEventListener(ev, upgrade, { once: true, passive: true })
      teardowns.push(() => {
        for (const ev of events) embed.removeEventListener(ev, upgrade)
      })
    } else {
      embed.replaceChildren(buildDayCard(date, null, extras))
      upgrade()
    }
  }
  return () => {
    live = false
    for (const td of teardowns) td()
  }
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
  let payload: DetailPayload | null = null
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

  const buildCard = (bar: HTMLElement): HTMLElement =>
    buildDayCard(bar.dataset.dateIso ?? '', payload, { location, event: bar.dataset.event })

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
    void loadDetailPayload(path).then(data => {
      payload = data
      if (active) {
        scroller.replaceChildren(buildCard(active))
        if (locked) setExpanded(true)
        updateOverflow()
      }
    })

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

  const onUnit = () => {
    if (!active) return
    scroller.replaceChildren(buildCard(active))
    if (locked) setExpanded(true)
    updateOverflow()
  }

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onBarsLeave)
  barsEl.addEventListener('click', onBarsClick)
  pop.addEventListener('mouseenter', onPopEnter)
  pop.addEventListener('mouseleave', onPopLeave)
  scroller.addEventListener('scroll', updateOverflow, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
  window.addEventListener('tri:focus-day', onFocusDay)
  window.addEventListener('tri:unit', onUnit)

  return () => {
    window.clearTimeout(hideTimer)
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    barsEl.removeEventListener('click', onBarsClick)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
    scroller.removeEventListener('scroll', updateOverflow)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('pointerdown', armAudio)
    window.removeEventListener('keydown', armAudio)
    window.removeEventListener('tri:focus-day', onFocusDay)
    window.removeEventListener('tri:unit', onUnit)
    void audio?.close()
  }
}

const setupCalc = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-calc-btn')
  const calc = root.querySelector<HTMLElement>('.tri-calc')
  const closeBtn = root.querySelector<HTMLElement>('.tri-calc-close')
  const pageMode = root.dataset.triView === 'tools'
  if (!calc || (!btn && !pageMode)) return null

  const inputVal = (k: string): string =>
    calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)?.value ?? ''
  const setInputVal = (k: string, value: string): void => {
    const input = calc.querySelector<HTMLInputElement>(`.tri-calc-in[data-k="${k}"]`)
    if (input) input.value = value
  }
  const targetInput = (): HTMLInputElement | null =>
    calc.querySelector<HTMLInputElement>('.tri-calc-target')
  const setResult = (leg: string, sec: number, forceTarget = false): void => {
    if (leg === 'total') {
      const target = targetInput()
      if (target && (forceTarget || document.activeElement !== target)) {
        target.value = formatDurationClock(sec)
      }
      return
    }
    const legInput = calc.querySelector<HTMLInputElement>(
      `.tri-calc-legtime[data-legtime="${leg}"]`,
    )
    if (legInput) {
      if (document.activeElement !== legInput) legInput.value = formatDurationClock(sec)
      return
    }
    const e = calc.querySelector<HTMLElement>(`.tri-calc-r[data-leg="${leg}"]`)
    if (e) e.textContent = formatDurationClock(sec)
  }

  const readCalcInput = (): TriathlonCalcInput => ({
    swimKm: Number(calc.dataset.swim) || 0,
    bikeKm: Number(calc.dataset.bike) || 0,
    runKm: Number(calc.dataset.run) || 0,
    swimPaceSec: parseClockSeconds(inputVal('swim')),
    t1Sec: parseClockSeconds(inputVal('t1')),
    bikeMph: Number(inputVal('bike')) || 0,
    t2Sec: parseClockSeconds(inputVal('t2')),
    runPaceSec: parseClockSeconds(inputVal('run')),
  })

  const compute = (forceTarget = false): void => {
    const times = computeTriathlonCalcTimes(readCalcInput())
    setResult('swim', times.swimSec)
    setResult('t1', times.t1Sec)
    setResult('bike', times.bikeSec)
    setResult('t2', times.t2Sec)
    setResult('run', times.runSec)
    setResult('total', times.totalSec, forceTarget)
  }

  const commitTarget = (): void => {
    const input = targetInput()
    if (!input) return
    const paces = solveTriathlonCalcTarget(readCalcInput(), parseClockSeconds(input.value))
    if (!paces) {
      compute(true)
      return
    }
    setInputVal('swim', clock(paces.swimPaceSec))
    setInputVal('bike', paces.bikeMph.toFixed(1))
    setInputVal('run', clock(paces.runPaceSec))
    compute(true)
  }

  const commitLeg = (leg: TriathlonCalcLeg): void => {
    const input = calc.querySelector<HTMLInputElement>(`.tri-calc-legtime[data-legtime="${leg}"]`)
    if (!input) return
    const solved = solveTriathlonCalcLeg(readCalcInput(), leg, parseClockSeconds(input.value))
    if (!solved) {
      compute(true)
      return
    }
    if (solved.swimPaceSec != null) setInputVal('swim', clock(solved.swimPaceSec))
    if (solved.bikeMph != null) setInputVal('bike', solved.bikeMph.toFixed(1))
    if (solved.runPaceSec != null) setInputVal('run', clock(solved.runPaceSec))
    compute(true)
  }

  let analytics: Analytics | null = null
  let userEdited = false
  const source = calc.querySelector<HTMLElement>('.tri-calc-source')
  const paceHuman = (which: 'avg' | 'pred', sport: Sport): number | null => {
    if (!analytics) return null
    const th = bySport(analytics.thresholds, sport)
    if (!th || !(th.vThr > 0)) return null
    const avg = sport === 'swim' ? 100 / th.vThr : sport === 'bike' ? th.vThr * 3.6 : 1000 / th.vThr
    if (which === 'avg') return avg
    const tr = bySport(analytics.trends, sport)
    if (!tr || !tr.level) return avg
    const end = tr.forecast[tr.forecast.length - 1]?.value ?? tr.level
    const ratio = end / tr.level
    return Number.isFinite(ratio) && ratio > 0 ? avg * ratio : avg
  }
  const toCalcInput = (sport: Sport, v: number): string =>
    sport === 'bike' ? (v * KM_TO_MI).toFixed(1) : clock(sport === 'run' ? v / KM_TO_MI : v)
  const applySource = (which: 'avg' | 'pred'): void => {
    let any = false
    for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
      const v = paceHuman(which, sport)
      if (v == null || !Number.isFinite(v) || v <= 0) continue
      setInputVal(sport, toCalcInput(sport, v))
      any = true
    }
    if (!any) return
    for (const b of calc.querySelectorAll<HTMLElement>('.tri-calc-src')) {
      const on = b.dataset.src === which
      b.classList.toggle('tri-calc-src--on', on)
      b.setAttribute('aria-selected', String(on))
    }
    compute()
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
    const target = event.target
    const targetElement = target instanceof HTMLElement ? target : null
    const src = targetElement?.closest<HTMLElement>('.tri-calc-src')
    if (src?.dataset.src === 'avg' || src?.dataset.src === 'pred') {
      applySource(src.dataset.src)
      return
    }
    const p = targetElement?.closest<HTMLElement>('.tri-calc-preset')
    if (!p) return
    calc.dataset.swim = p.dataset.swim ?? ''
    calc.dataset.bike = p.dataset.bike ?? ''
    calc.dataset.run = p.dataset.run ?? ''
    for (const x of calc.querySelectorAll('.tri-calc-preset'))
      x.classList.toggle('tri-calc-preset--on', x === p)
    compute()
  }
  const onInput = (event: Event) => {
    userEdited = true
    const target = event.target
    if (
      target instanceof HTMLInputElement &&
      (target.classList.contains('tri-calc-target') ||
        target.classList.contains('tri-calc-legtime'))
    )
      return
    compute()
  }
  const onChange = (event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) return
    if (target.classList.contains('tri-calc-target')) {
      userEdited = true
      commitTarget()
    } else if (target.classList.contains('tri-calc-legtime')) {
      userEdited = true
      commitLeg(target.dataset.legtime as TriathlonCalcLeg)
    }
  }
  const onCalcKey = (event: KeyboardEvent) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement) || event.key !== 'Enter') return
    if (target.classList.contains('tri-calc-target')) {
      event.preventDefault()
      userEdited = true
      commitTarget()
      target.blur()
    } else if (target.classList.contains('tri-calc-legtime')) {
      event.preventDefault()
      userEdited = true
      commitLeg(target.dataset.legtime as TriathlonCalcLeg)
      target.blur()
    }
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  if (pageMode) {
    calc.classList.add('tri-calc--page')
    calc.setAttribute('aria-hidden', 'false')
    compute()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
  }
  calc.addEventListener('click', onCalcClick)
  calc.addEventListener('input', onInput)
  calc.addEventListener('change', onChange)
  calc.addEventListener('keydown', onCalcKey)
  document.addEventListener('keydown', onKey)
  calc.querySelectorAll('.tri-calc-preset')[1]?.classList.add('tri-calc-preset--on')

  const apath = root.dataset.analyticsPath
  if (apath)
    fetch(apath)
      .then(res => res.json())
      .then((d: Analytics) => {
        analytics = d
        const usable = (['swim', 'bike', 'run'] as Sport[]).some(
          s => paceHuman('avg', s) != null || paceHuman('pred', s) != null,
        )
        if (source && usable) source.hidden = false
        if (usable && !userEdited) applySource('avg')
      })
      .catch(() => {})

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    calc.removeEventListener('click', onCalcClick)
    calc.removeEventListener('input', onInput)
    calc.removeEventListener('change', onChange)
    calc.removeEventListener('keydown', onCalcKey)
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

  const scroller = panel.querySelector<HTMLElement>(`${panelSel}-scroll`)
  const base = panelSel.slice(1)
  const updateFade = () => {
    if (!scroller) return
    panel.classList.toggle(`${base}--top`, scroller.scrollTop > 4)
    panel.classList.toggle(
      `${base}--more`,
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 4,
    )
  }

  const close = () => {
    wrap.classList.remove(openClass)
    panel.setAttribute('aria-hidden', 'true')
  }
  const onBtn = () => {
    const open = wrap.classList.toggle(openClass)
    panel.setAttribute('aria-hidden', open ? 'false' : 'true')
    if (open) updateFade()
  }
  const onDocClick = (event: MouseEvent) => {
    if (!wrap.contains(event.target as Node)) close()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') close()
  }

  btn.addEventListener('click', onBtn)
  scroller?.addEventListener('scroll', updateFade, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', onBtn)
    scroller?.removeEventListener('scroll', updateFade)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
  }
}

const ANA_W = 100
const ANA_H = 30

type WeightUnit = 'kg' | 'lb'
const WEIGHT_UNIT_KEY = 'tri-weight-unit'
const KG_PER_LB = 0.45359237
const readWeightUnit = (): WeightUnit => {
  try {
    return localStorage.getItem(WEIGHT_UNIT_KEY) === 'kg' ? 'kg' : 'lb'
  } catch {
    return 'lb'
  }
}
let weightUnit: WeightUnit = readWeightUnit()
const wConv = (kg: number): number => (weightUnit === 'kg' ? kg : kg / KG_PER_LB)
const wNum = (kg: number, kgDp = 1, lbDp = 0): string =>
  wConv(kg).toFixed(weightUnit === 'kg' ? kgDp : lbDp)
const wFmt = (kg: number, kgDp = 1, lbDp = 0): string => `${wNum(kg, kgDp, lbDp)} ${weightUnit}`
const wSigned = (kg: number, dp: number): string => {
  const v = wConv(kg)
  return `${v > 0 ? '+' : ''}${v.toFixed(dp)}`
}
const setWeightUnit = (u: WeightUnit): void => {
  if (u === weightUnit) return
  weightUnit = u
  try {
    localStorage.setItem(WEIGHT_UNIT_KEY, u)
  } catch {}
  document.dispatchEvent(new CustomEvent('tri-weightunit', { detail: {} }))
}
const weightSwitch = (): HTMLElement => {
  const g = el('div', 'tri-unit-switch', undefined, { role: 'group', 'aria-label': 'weight unit' })
  for (const u of ['kg', 'lb'] as WeightUnit[]) {
    const on = u === weightUnit
    const opt = el('button', on ? 'tri-unit-opt tri-unit-opt--on' : 'tri-unit-opt', u, {
      type: 'button',
      'aria-pressed': String(on),
    })
    opt.addEventListener('click', () => setWeightUnit(u))
    g.appendChild(opt)
  }
  return g
}
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
    def: 'Training Stress Balance $\\mathrm{TSB}=\\mathrm{CTL}-\\mathrm{ATL}$. Positive means fresh and tapered; negative means loaded and carrying fatigue.',
  },
  acwr: {
    term: 'ACWR',
    def: 'Acute:Chronic Workload Ratio $\\mathrm{ACWR}=\\text{7d}/\\text{28d}$ load. $0.8\\text{–}1.3$ is the safe zone; above $1.5$ flags an injury-risk spike.',
  },
  ramp: {
    term: 'ramp',
    def: 'Week-over-week change in fitness (CTL). Positive is building; large jumps are the classic too-much-too-soon risk.',
  },
  monotony: {
    term: 'monotony',
    def: 'Daily-load sameness across a week, $\\text{monotony}=\\mu/\\sigma$ (mean over standard deviation). Above $\\approx 2$ with high load is Foster’s strain red flag.',
  },
  strain: {
    term: 'strain',
    def: '$\\text{strain}=\\text{weekly load}\\times\\text{monotony}$. High, unvarying training scores high, often dictates overtraining.',
  },
  load: {
    term: 'load',
    def: 'Per-session training stress $\\mathrm{load}\\approx\\mathrm{IF}^2\\cdot t$, scaled so an hour at threshold $\\approx 100$. HR, power and cadence are captured per activity; this load stays pace-derived for now.',
  },
  score: {
    term: 'readiness',
    def: 'A 0–100 blend of your fitness against the race demand ($45\\%$) and how much of each leg’s distance you have actually covered in training ($55\\%$).',
  },
  binding: {
    term: 'binding leg',
    def: 'The discipline limiting your readiness most — the weakest distance-coverage $\\times$ recency. Train this one first.',
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
  wgoal: {
    term: 'weight goal',
    def: 'Target weight from Garmin Connect. The delta is current minus goal; the ETA divides the gap by your current weekly trend, shown only while the trend actually converges.',
  },
  bodyfat: {
    term: 'body fat',
    def: 'Bio-impedance body-fat percentage from the Index scale. Trend matters more than any single reading — hydration swings a measurement by $\\pm 1\\%$.',
  },
  bmi: {
    term: 'BMI',
    def: 'Body mass index, $\\mathrm{kg}/\\mathrm{m}^2$. Blunt for muscular athletes — read it alongside body fat, not instead of it.',
  },
  bmr: {
    term: 'BMR (Katch-McArdle)',
    def: 'Resting daily burn from lean mass, $\\mathrm{BMR}=370+21.6\\,\\mathrm{LBM}$ with $\\mathrm{LBM}=\\text{weight}\\,(1-\\text{bodyfat})$ in kg. Driven by the Index scale’s body-fat reading, so it tracks composition rather than scale weight.',
  },
  effort: {
    term: 'relative effort',
    def: 'Strava’s suffer score—duration weighted by how far above resting your heart rate sat. The bars sum each week’s sessions, so it tracks acute training stress across all three sports at once.',
  },
  hrv: {
    term: 'HRV',
    def: 'Heart-rate variability (RMSSD, ms). Tracked as the 7-day mean of $\\ln(\\mathrm{RMSSD})$ against a 28-day personal baseline, $z=(\\overline{\\ln\\mathrm{RMSSD}}_{7}-\\mu_{28})/\\sigma_{28}$. Below $-1\\sigma$ flags parasympathetic suppression.',
  },
  rhr: {
    term: 'resting HR',
    def: 'Overnight low heart rate in bpm. A rise of $\\ge 5$ bpm or $+1\\sigma$ over the 28-day baseline is an early fatigue or illness signal.',
  },
  tempdev: {
    term: 'temp deviation',
    def: 'Skin temperature against your personal baseline ($^\\circ\\mathrm{C}$). $\\ge +0.5\\,^\\circ\\mathrm{C}$ reads as a possible immune response, often $24\\text{–}48$ h before symptoms.',
  },
  sleepdebt: {
    term: 'sleep debt',
    def: 'Rolling 14-night shortfall against an 8 h target, $D=\\sum_{i=1}^{14}\\max(0,\\,T-s_i)$. The 7 h floor is the adult minimum; athletes need $8\\text{–}10$ h.',
  },
  overreaching: {
    term: 'overreaching',
    def: 'Suppressed HRV ($z\\le-1$) while load spikes (ACWR caution/high or ramp $>10\\%$) — the combination that precedes non-functional overreaching.',
  },
  oreadiness: {
    term: 'readiness',
    def: 'Oura’s $0\\text{–}100$ daily readiness: $\\ge 85$ optimal, $70\\text{–}84$ good, $<70$ pay attention. Streaks under 70 flag accumulated strain.',
  },
  vo2max: {
    term: 'VO₂max',
    def: 'Maximal oxygen uptake in $\\mathrm{ml/kg/min}$. Bike path: $\\dot{V}O_2 = 10.8\\,\\mathrm{MAP}/m + 7$ with $\\mathrm{MAP}=\\mathrm{FTP}/0.75$ and FTP as $95\\%$ of best 20-min power.',
  },
  fitage: {
    term: 'fitness age',
    def: 'The age whose population-median VO₂max (FRIEND registry, male) equals yours, clamped $20\\text{–}80$. Lower than calendar age means the engine outruns the birthday.',
  },
  vam: {
    term: 'VAM',
    def: 'Velocità ascensionale media — vertical metres climbed per hour, $\\mathrm{gain}\\cdot 3600/t$. Recreational $\\approx 600\\text{–}1000$; pro climbers exceed $1600$.',
  },
  radar: {
    term: 'abilities',
    def: 'Six axes normalised $0\\text{–}100$: Coggan $\\mathrm{W/kg}$ anchors for sprint and threshold, CTL for endurance, VAM for climb, cadence against 90 rpm / 180 spm, mean readiness for recovery.',
  },
  ef: {
    term: 'efficiency factor',
    def: 'Aerobic output per heartbeat: $\\mathrm{NP}/\\overline{\\mathrm{HR}}$ on the bike, graded speed per beat on the run. Rising EF at equal effort means the engine is getting cheaper.',
  },
  decouple: {
    term: 'decoupling',
    def: 'Pw:Hr drift $(E_1-E_2)/E_1$ across session halves. $<5\\%$ means coupled — a durable aerobic base; $>10\\%$ the engine fades late.',
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
const buildIconLeg = (sport: ActivityKind): HTMLElement => {
  const wrap = el('span', `tri-ana-ico tri-leg-${sport}`)
  wrap.appendChild(buildIcon(sport))
  return wrap
}
const trendDir = (invert: boolean, slope: number | null): number => {
  if (slope == null || slope === 0) return 0
  return (invert ? slope < 0 : slope > 0) ? 1 : -1
}

const buildGauge = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-gauge')
  block.appendChild(anaTitle('training load · injury risk', 'acwr'))
  const r = data.risk
  const chips = el('div', 'tri-gauge-chips')
  chips.append(
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
  if (r.acwr == null) {
    block.appendChild(el('div', 'tri-ana-empty', 'building base — ACWR needs ~4 weeks'))
    block.appendChild(chips)
    return block
  }
  const W = 100
  const H = 12
  const lo = 0.5
  const hi = 1.8
  const xf = (v: number): number => ((clampN(v, lo, hi) - lo) / (hi - lo)) * W
  const s = svg('svg', {
    class: 'tri-gauge-svg',
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: 'none',
  })
  const zone = (a: number, b: number, cls: string): void => {
    s.appendChild(
      svg('rect', {
        x: xf(a),
        y: 4,
        width: xf(b) - xf(a),
        height: 4,
        class: `tri-gauge-zone ${cls}`,
      }),
    )
  }
  zone(lo, 0.8, 'tri-acwr-z-under')
  zone(0.8, 1.3, 'tri-acwr-z-sweet')
  zone(1.3, 1.5, 'tri-acwr-z-caution')
  zone(1.5, hi, 'tri-acwr-z-high')
  const track = el('div', 'tri-gauge-track')
  const needle = el('span', 'tri-gauge-needle')
  needle.style.left = `${clampN(xf(r.acwr), 1.5, 98.5)}%`
  track.append(s, needle)
  block.appendChild(track)
  const scale = el('div', 'tri-gauge-scale')
  for (const v of [0.8, 1.3, 1.5]) {
    const tick = el('span', 'tri-gauge-tick', v.toFixed(1))
    tick.style.left = `${xf(v).toFixed(2)}%`
    scale.appendChild(tick)
  }
  block.appendChild(scale)
  const val = el('div', 'tri-gauge-val')
  val.append(
    el('span', 'tri-gauge-num', r.acwr.toFixed(2)),
    el('span', `tri-gauge-state tri-acwr-${r.acwrState}`, r.acwrState),
  )
  block.appendChild(val)
  block.appendChild(chips)
  return block
}

const PMC_MONTHS = [
  'jan',
  'feb',
  'mar',
  'apr',
  'may',
  'jun',
  'jul',
  'aug',
  'sep',
  'oct',
  'nov',
  'dec',
]
const PMC_H = 82
const PMC_TOP = 4
const PMC_BOT = 52
const PMC_TSB_ZERO = 32
const PMC_TSB_HALF = 24
const PMC_BAR_TOP = 61
const PMC_BAR_BOT = 80

const niceUp = (v: number): number => {
  const step = v <= 20 ? 5 : v <= 60 ? 10 : v <= 200 ? 20 : 50
  return Math.max(step, Math.ceil(v / step) * step)
}

type AxisXTick = { label: string; pct: number; cls?: string }
const monthTicks = (dates: string[], xPct: (i: number) => number): AxisXTick[] => {
  const out: AxisXTick[] = []
  const seen = new Set<string>()
  for (let i = 0; i < dates.length; i++) {
    const mo = dates[i].slice(0, 7)
    if (seen.has(mo)) continue
    seen.add(mo)
    out.push({
      label: PMC_MONTHS[Number(dates[i].slice(5, 7)) - 1],
      pct: xPct(i),
      cls: i === 0 ? 'tri-cax-xt--first' : undefined,
    })
  }
  return out
}

const axisFrame = (
  svgEl: SVGElement,
  yTicks: { label: string; vbY: number }[],
  vbH: number,
  xTicks: AxisXTick[],
  cssH: number,
): HTMLElement => {
  const frame = el('div', 'tri-cax-frame')
  const yax = el('div', 'tri-cax-yax')
  yax.style.height = `${cssH}px`
  for (const t of yTicks) {
    const lab = el('span', 'tri-cax-yt', t.label)
    lab.style.top = `${((t.vbY / vbH) * 100).toFixed(2)}%`
    yax.appendChild(lab)
  }
  const plot = el('div', 'tri-cax-plot')
  plot.appendChild(svgEl)
  const xax = el('div', 'tri-cax-xax')
  for (const t of xTicks) {
    const lab = el('span', `tri-cax-xt${t.cls ? ` ${t.cls}` : ''}`, t.label)
    lab.style.left = `${t.pct.toFixed(2)}%`
    xax.appendChild(lab)
  }
  plot.appendChild(xax)
  frame.append(yax, plot)
  return frame
}

const PMC_PROJ_DAYS = 14
const K42 = 1 - Math.exp(-1 / 42)
const K7 = 1 - Math.exp(-1 / 7)

const buildPmc = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-pmc')
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  const activitiesByDate = new Map<string, ActivitySummary[]>()
  for (const activity of data.activities) {
    const current = activitiesByDate.get(activity.date)
    if (current) current.push(activity)
    else activitiesByDate.set(activity.date, [activity])
  }
  const r = data.risk
  const ago = Math.max(0, n - 8)
  const delta = (get: (d: DailyPoint) => number): number =>
    Math.round(get(daily[n - 1]) - get(daily[ago]))
  const stat = (
    cls: string,
    label: string,
    value: string,
    d: number,
    gloss: string,
    zone?: string,
  ): HTMLElement => {
    const wrap = el('div', `tri-pmc-stat ${cls}`)
    const head = el('div', 'tri-pmc-stat-head')
    head.append(el('span', 'tri-pmc-dot'), el('span', 'tri-pmc-stat-k', label))
    wrap.append(
      head,
      el('div', `tri-pmc-stat-v${zone ? ` tri-zone-${zone}` : ''}`, value),
      el('div', 'tri-pmc-stat-d', `${signed(d)} · 7d`),
    )
    return markGloss(wrap, gloss)
  }
  const readout = el('div', 'tri-pmc-now-row')
  readout.append(
    stat(
      'tri-pmc-fit',
      'fitness',
      String(Math.round(r.ctl)),
      delta(d => d.ctl),
      'ctl',
    ),
    stat(
      'tri-pmc-fat',
      'fatigue',
      String(Math.round(r.atl)),
      delta(d => d.atl),
      'atl',
    ),
    stat(
      'tri-pmc-form',
      'form',
      signed(Math.round(r.tsb)),
      delta(d => d.tsb),
      'tsb',
      r.tsbZone,
    ),
  )
  block.appendChild(readout)

  const H = PMC_PROJ_DAYS
  const N = n + H
  const lastMs = Date.parse(`${daily[n - 1].date}T00:00:00Z`)
  const projDate = (k: number): string => new Date(lastMs + k * 86400000).toISOString().slice(0, 10)
  type Proj = { ctl: number; atl: number; tsb: number }
  const project = (load: number): Proj[] => {
    let c = daily[n - 1].ctl
    let a = daily[n - 1].atl
    const out: Proj[] = []
    for (let k = 0; k < H; k++) {
      c += (load - c) * K42
      a += (load - a) * K7
      out.push({ ctl: c, atl: a, tsb: c - a })
    }
    return out
  }

  let maxFitRaw = 1
  let tsbAbsRaw = 10
  let maxLoad = 1
  let loadSum = 0
  for (const d of daily) {
    maxFitRaw = Math.max(maxFitRaw, d.ctl, d.atl)
    tsbAbsRaw = Math.max(tsbAbsRaw, Math.abs(d.tsb))
    maxLoad = Math.max(maxLoad, d.load)
  }
  for (const d of daily.slice(Math.max(0, n - 14))) loadSum += d.load
  const avgRecent = Math.round(loadSum / Math.min(14, n))
  const LOAD_MAX = niceUp(Math.max(120, Math.round(avgRecent * 1.4)))
  let futLoad = clampN(avgRecent, 0, LOAD_MAX)
  for (const p of project(LOAD_MAX)) maxFitRaw = Math.max(maxFitRaw, p.ctl, p.atl)
  for (const p of [...project(LOAD_MAX), ...project(0)])
    tsbAbsRaw = Math.max(tsbAbsRaw, Math.abs(p.tsb))
  const maxFit = niceUp(maxFitRaw)
  const tsbAbs = niceUp(tsbAbsRaw)

  const x = (i: number): number => (i / (N - 1)) * ANA_W
  const yFit = (v: number): number => PMC_BOT - (v / maxFit) * (PMC_BOT - PMC_TOP)
  const yTsb = (v: number): number => PMC_TSB_ZERO - (v / tsbAbs) * PMC_TSB_HALF
  const yBar = (v: number): number => PMC_BAR_BOT - (v / maxLoad) * (PMC_BAR_BOT - PMC_BAR_TOP)
  const nowX = x(n - 1)

  const ctlPts = daily.map((d, i) => [x(i), yFit(d.ctl)] as [number, number])
  const atlPts = daily.map((d, i) => [x(i), yFit(d.atl)] as [number, number])
  const tsbPts = daily.map((d, i) => [x(i), yTsb(d.tsb)] as [number, number])

  let projSeries = project(futLoad)
  const projPath = (
    anchor: number,
    get: (p: Proj) => number,
    yfn: (v: number) => number,
  ): string => {
    const pts: [number, number][] = [[nowX, yfn(anchor)]]
    projSeries.forEach((p, k) => pts.push([x(n + k), yfn(get(p))]))
    return polyD(pts)
  }

  const frame = el('div', 'tri-pmc-frame')
  const yax = el('div', 'tri-pmc-yax')
  for (const gv of [maxFit, maxFit / 2, 0]) {
    const lab = el('span', 'tri-pmc-yt', String(Math.round(gv)))
    lab.style.top = `${((yFit(gv) / PMC_H) * 100).toFixed(2)}%`
    yax.appendChild(lab)
  }
  frame.appendChild(yax)

  const plot = el('div', 'tri-pmc-plot')
  const s = svg('svg', {
    class: 'tri-ana-svg tri-pmc-svg',
    viewBox: `0 0 ${ANA_W} ${PMC_H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('line', { x1: 0, y1: yFit(maxFit), x2: ANA_W, y2: yFit(maxFit), class: 'tri-pmc-grid' }),
  )
  const areaInner = ctlPts.map(([px, py]) => `L ${px.toFixed(2)} ${py.toFixed(2)}`).join(' ')
  s.appendChild(
    svg('path', {
      d: `M 0 ${PMC_BOT} ${areaInner} L ${nowX.toFixed(2)} ${PMC_BOT} Z`,
      class: 'tri-pmc-area',
    }),
  )
  const bw = (ANA_W / N) * 0.62
  for (let i = 0; i < n; i++) {
    const load = daily[i].load
    if (load <= 0) continue
    const by = yBar(load)
    s.appendChild(
      svg('rect', {
        x: (x(i) - bw / 2).toFixed(2),
        y: by.toFixed(2),
        width: bw.toFixed(2),
        height: (PMC_BAR_BOT - by).toFixed(2),
        class: i === n - 1 ? 'tri-pmc-bar tri-pmc-bar--now' : 'tri-pmc-bar',
      }),
    )
  }
  s.appendChild(
    svg('line', { x1: 0, y1: PMC_BAR_BOT, x2: ANA_W, y2: PMC_BAR_BOT, class: 'tri-pmc-baseline' }),
  )
  s.appendChild(
    svg('line', { x1: 0, y1: PMC_TSB_ZERO, x2: ANA_W, y2: PMC_TSB_ZERO, class: 'tri-ana-zero' }),
  )
  s.appendChild(svg('line', { x1: nowX, y1: 0, x2: nowX, y2: PMC_BAR_BOT, class: 'tri-pmc-now' }))
  const hits = svg('g', { class: 'tri-pmc-hit-layer' })
  for (let i = 0; i < N; i++) {
    const left = i === 0 ? 0 : (x(i - 1) + x(i)) / 2
    const right = i === N - 1 ? ANA_W : (x(i) + x(i + 1)) / 2
    hits.appendChild(
      svg('rect', {
        x: left.toFixed(2),
        y: 0,
        width: Math.max(0.1, right - left).toFixed(2),
        height: PMC_H,
        class: 'tri-pmc-hit',
        'data-i': i,
      }),
    )
  }
  s.appendChild(svg('path', { d: polyD(tsbPts), class: 'tri-pmc-l-form' }))
  s.appendChild(svg('path', { d: polyD(atlPts), class: 'tri-pmc-l-fat' }))
  s.appendChild(svg('path', { d: polyD(ctlPts), class: 'tri-pmc-l-fit' }))
  const tsbProj = svg('path', {
    d: projPath(daily[n - 1].tsb, p => p.tsb, yTsb),
    class: 'tri-pmc-l-form tri-pmc-proj',
  })
  const atlProj = svg('path', {
    d: projPath(daily[n - 1].atl, p => p.atl, yFit),
    class: 'tri-pmc-l-fat tri-pmc-proj',
  })
  const ctlProj = svg('path', {
    d: projPath(daily[n - 1].ctl, p => p.ctl, yFit),
    class: 'tri-pmc-l-fit tri-pmc-proj',
  })
  s.append(tsbProj, atlProj, ctlProj)
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: PMC_H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  s.appendChild(hits)
  plot.appendChild(s)

  const xax = el('div', 'tri-pmc-xax')
  const seenMonth = new Set<string>()
  for (let i = 0; i < n; i++) {
    const mo = daily[i].date.slice(0, 7)
    if (seenMonth.has(mo)) continue
    seenMonth.add(mo)
    const lab = el(
      'span',
      `tri-pmc-xt${i === 0 ? ' tri-pmc-xt--first' : ''}`,
      PMC_MONTHS[Number(daily[i].date.slice(5, 7)) - 1],
    )
    lab.style.left = `${x(i).toFixed(2)}%`
    xax.appendChild(lab)
  }
  const today = el('span', 'tri-pmc-xt tri-pmc-xt--now', 'today')
  today.style.left = `${nowX.toFixed(2)}%`
  xax.appendChild(today)
  const endLab = el('span', 'tri-pmc-xt tri-pmc-xt--end', `+${H}d`)
  endLab.style.left = '100%'
  xax.appendChild(endLab)
  plot.appendChild(xax)
  const readoutEl = el('div', 'tri-chart-readout')
  plot.appendChild(readoutEl)
  frame.appendChild(plot)
  block.appendChild(frame)

  const ctrl = el('div', 'tri-pmc-ctrl')
  const slider = el('input', 'tri-pmc-load') as HTMLInputElement
  slider.type = 'range'
  slider.min = '0'
  slider.max = String(LOAD_MAX)
  slider.step = '5'
  slider.value = String(futLoad)
  slider.setAttribute('aria-label', 'assumed future daily load')
  const ctrlLab = el('span', 'tri-pmc-ctrl-lab')
  ctrl.append(el('span', 'tri-pmc-ctrl-k', 'projected load'), slider, ctrlLab)
  block.appendChild(ctrl)

  const legendRow = (cls: string, name: string, val: string): HTMLElement => {
    const row = el('div', `tri-pmc-leg ${cls}`)
    row.append(
      el('span', 'tri-pmc-dot'),
      el('span', 'tri-pmc-leg-v', val),
      el('span', 'tri-pmc-leg-k', name),
    )
    return row
  }
  const entryRow = (a: ActivitySummary): HTMLElement => {
    const row = el('div', 'tri-pmc-entry')
    row.append(
      el('span', `tri-pmc-entry-s tri-leg-${a.sport}`, a.sport),
      el('span', 'tri-pmc-entry-n', a.name || a.sport),
      el('span', 'tri-pmc-entry-d', dist(a.distanceKm, a.sport)),
    )
    return row
  }
  const renderLegend = (i: number): void => {
    const proj = i >= n
    const p = proj ? projSeries[Math.min(H - 1, i - n)] : daily[i]
    const date = proj ? projDate(i - n + 1) : daily[i].date
    const entries = proj
      ? []
      : [...(activitiesByDate.get(date) ?? [])].sort((a, b) => b.load - a.load)
    const entryList = el('div', 'tri-pmc-entries')
    if (!proj) {
      if (entries.length === 0)
        entryList.appendChild(el('div', 'tri-pmc-entry tri-pmc-entry--empty', 'no activity'))
      else for (const activity of entries.slice(0, 3)) entryList.appendChild(entryRow(activity))
    }
    const metricGrid = el('div', 'tri-pmc-leg-grid')
    metricGrid.append(
      legendRow('tri-pmc-fit', 'fitness', String(Math.round(p.ctl))),
      legendRow('tri-pmc-fat', 'fatigue', String(Math.round(p.atl))),
      legendRow('tri-pmc-form', 'form', signed(Math.round(p.tsb))),
    )
    readoutEl.replaceChildren(
      el('span', 'tri-pmc-leg-date', `${shortDate(date)}${proj ? ' · proj' : ''}`),
      metricGrid,
    )
    if (!proj) readoutEl.append(entryList)
  }
  const setCtrlLab = (): void => {
    const lp = projSeries[H - 1]
    ctrlLab.textContent = `${futLoad}/day → ${H}d: fitness ${Math.round(lp.ctl)} · form ${signed(Math.round(lp.tsb))}`
  }
  let activeIndex = n - 1
  const focusIndex = (i: number, hover: boolean): void => {
    activeIndex = Math.round(clampN(i, 0, N - 1))
    const cx = x(activeIndex).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    readoutEl.style.left = `${clampN((x(activeIndex) / ANA_W) * 100, 14, 86).toFixed(2)}%`
    renderLegend(activeIndex)
    block.classList.toggle('tri-chart--hover', hover)
  }
  setCtrlLab()
  focusIndex(n - 1, false)
  slider.addEventListener('input', () => {
    futLoad = Number(slider.value)
    projSeries = project(futLoad)
    tsbProj.setAttribute(
      'd',
      projPath(daily[n - 1].tsb, p => p.tsb, yTsb),
    )
    atlProj.setAttribute(
      'd',
      projPath(daily[n - 1].atl, p => p.atl, yFit),
    )
    ctlProj.setAttribute(
      'd',
      projPath(daily[n - 1].ctl, p => p.ctl, yFit),
    )
    setCtrlLab()
    focusIndex(activeIndex, block.classList.contains('tri-chart--hover'))
  })

  const onMove = (event: MouseEvent): void => {
    const rect = s.getBoundingClientRect()
    const i = Math.round(clampN((event.clientX - rect.left) / rect.width, 0, 1) * (N - 1))
    focusIndex(i, true)
  }
  const onLeave = (): void => {
    focusIndex(n - 1, false)
  }
  s.addEventListener('mousemove', onMove)
  s.addEventListener('mouseleave', onLeave)

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
  const yMaxC = niceUp(mx)
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const y = (v: number): number => bot - (v / yMaxC) * (bot - top)
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
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
  block.appendChild(
    axisFrame(
      s,
      [yMaxC, yMaxC / 2, 0].map(v => ({ label: String(Math.round(v)), vbY: y(v) })),
      ANA_H,
      monthTicks(
        daily.map(d => d.date),
        i => (i / (n - 1)) * ANA_W,
      ),
      70,
    ),
  )
  block.appendChild(el('div', 'tri-chart-readout'))
  const cap = el('div', 'tri-elev-cap')
  for (const sp of ['swim', 'bike', 'run'] as Sport[]) {
    const th = bySport(data.thresholds, sp)
    const label = th == null ? '—' : th.staleDays === 0 ? 'today' : `${th.staleDays}d ago`
    const leg = el('span', `tri-ana-leg tri-leg-${sp}`)
    leg.append(buildIcon(sp), el('span', 'tri-ana-k', label))
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
  const yMaxW = niceUp(mx)
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
    const h = (w.load / yMaxW) * (H - 2)
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
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' }))
  block.appendChild(
    axisFrame(
      s,
      [yMaxW, yMaxW / 2, 0].map(v => ({
        label: String(Math.round(v)),
        vbY: bot - (v / yMaxW) * (H - 2),
      })),
      H,
      monthTicks(
        wk.map(w => w.weekStart),
        i => ((i + 0.5) / n) * 100,
      ),
      56,
    ),
  )
  block.appendChild(el('div', 'tri-chart-readout'))
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

const fmtTrendVal = (sport: Sport, v: number): string =>
  sport === 'bike' ? `${Math.round(v)} km/h` : `${clock(v)}${sport === 'swim' ? ' /100m' : ' /km'}`
const fmtTrendShort = (sport: Sport, v: number): string =>
  sport === 'bike' ? String(Math.round(v)) : clock(v)

const buildTrendPanel = (data: Analytics, sport: Sport): HTMLElement => {
  const tr = bySport(data.trends, sport)
  const th = bySport(data.thresholds, sport)
  const wrap = el('div', `tri-trend-panel${tr?.stale ? ' tri-trend-stale' : ''}`)
  wrap.dataset.sport = sport
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
    const xOf = (p: number): number => p * ANA_W
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
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-trend-axis' }))
    s.appendChild(svg('line', { x1: 0, y1: ANA_H, x2: ANA_W, y2: ANA_H, class: 'tri-trend-axis' }))
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
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    const track = el('div', 'tri-trend-track')
    const dot = el('span', `tri-trend-dot tri-bg-${sport}`)
    dot.style.left = `${clampN((xOf(0) / ANA_W) * 100, 0, 98)}%`
    dot.style.top = `${clampN((Y(level) / ANA_H) * 100, 4, 96)}%`
    track.append(s, dot)
    const yax = el('div', 'tri-trend-yax')
    yax.append(
      el('span', '', fmtTrendShort(sport, tr.invert ? lo : hi)),
      el('span', '', fmtTrendShort(sport, tr.invert ? hi : lo)),
    )
    const chart = el('div', 'tri-trend-chart')
    chart.append(yax, track)
    const xax = el('div', 'tri-trend-xax')
    xax.append(el('span', '', 'now'), el('span', '', `+${Math.round(weeks)} wk`))
    wrap.append(chart, xax, el('div', 'tri-chart-readout'))
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

interface BodyDay {
  date: string
  ts: number
  samples: { date: string; ts: number; kg: number }[]
  min: number
  max: number
  first: number
  last: number
}

const groupBodyByDay = (series: { date: string; ts: number; kg: number }[]): BodyDay[] => {
  const byDay = new Map<string, { date: string; ts: number; kg: number }[]>()
  for (const p of series) {
    const arr = byDay.get(p.date)
    if (arr) arr.push(p)
    else byDay.set(p.date, [p])
  }
  return [...byDay.values()]
    .map(arr => {
      const sorted = arr.slice().sort((a, b) => a.ts - b.ts)
      let mn = Infinity
      let mx = -Infinity
      for (const q of sorted) {
        if (q.kg < mn) mn = q.kg
        if (q.kg > mx) mx = q.kg
      }
      const last = sorted[sorted.length - 1]
      return {
        date: sorted[0].date,
        ts: last.ts,
        samples: sorted,
        min: mn,
        max: mx,
        first: sorted[0].kg,
        last: last.kg,
      }
    })
    .sort((a, b) => a.ts - b.ts)
}

const buildBody = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-bodywt')
  const title = anaTitle('body weight', 'weight')
  const b: BodyBlock = data.body
  if (b.latestKg == null) {
    block.appendChild(title)
    block.appendChild(el('div', 'tri-ana-empty', 'no weight logged'))
    return block
  }
  const titleRow = el('div', 'tri-bodywt-titlerow')
  titleRow.append(title, weightSwitch())
  block.appendChild(titleRow)
  const head = el('div', 'tri-bodywt-head')
  head.append(el('span', 'tri-bodywt-kg', wFmt(b.latestKg)))
  block.appendChild(head)
  const pts = b.series
  if (pts.length >= 2) {
    let min = Infinity
    let max = -Infinity
    for (const p of pts) {
      if (p.kg < min) min = p.kg
      if (p.kg > max) max = p.kg
    }
    if (b.goalKg != null) {
      if (b.goalKg < min) min = b.goalKg
      if (b.goalKg > max) max = b.goalKg
    }
    const range = Math.max(0.5, max - min)
    const lo = min - range * 0.18
    const hi = max + range * 0.18
    const days = groupBodyByDay(pts)
    const nd = days.length
    const t0 = pts[0].ts
    const t1 = pts[pts.length - 1].ts
    const xPct = (ts: number): number => (t1 > t0 ? ((ts - t0) / (t1 - t0)) * 100 : 50)
    const yPct = (kg: number): number => (1 - (kg - lo) / (hi - lo)) * 100
    const chart = el('div', 'tri-bodywt-chart')
    const yax = el('div', 'tri-bodywt-yax')
    yax.append(el('span', '', wNum(hi)), el('span', '', wNum(lo)))
    const plot = el('div', 'tri-bodywt-plot')
    const s = svg('svg', {
      class: 'tri-bodywt-svg',
      viewBox: '0 0 100 100',
      preserveAspectRatio: 'none',
    })
    for (const gy of [0, 50, 100])
      s.appendChild(svg('line', { x1: 0, y1: gy, x2: 100, y2: gy, class: 'tri-bodywt-grid' }))
    if (b.goalKg != null)
      s.appendChild(
        svg('line', {
          x1: 0,
          y1: yPct(b.goalKg),
          x2: 100,
          y2: yPct(b.goalKg),
          class: 'tri-bodywt-goal',
        }),
      )
    for (const d of days) {
      if (d.samples.length < 2) continue
      const dx = xPct(d.ts).toFixed(2)
      s.appendChild(
        svg('line', {
          x1: dx,
          y1: yPct(d.max),
          x2: dx,
          y2: yPct(d.min),
          class: 'tri-bodywt-range',
          'data-day': d.date,
        }),
      )
    }
    s.appendChild(
      svg('path', {
        d: polyD(days.map(d => [xPct(d.ts), yPct(d.last)])),
        class: 'tri-bodywt-line',
      }),
    )
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 100, class: 'tri-ana-cursor' }))
    plot.appendChild(s)
    days.forEach((d, di) => {
      const left = `${xPct(d.ts).toFixed(2)}%`
      d.samples.forEach((sample, si) => {
        const dayLast = si === d.samples.length - 1
        const cls =
          di === nd - 1 && dayLast
            ? 'tri-bodywt-pt tri-bodywt-pt--last'
            : dayLast
              ? 'tri-bodywt-pt'
              : 'tri-bodywt-pt tri-bodywt-pt--sub'
        const m = el('span', cls)
        m.style.left = left
        m.style.top = `${yPct(sample.kg).toFixed(2)}%`
        plot.appendChild(m)
      })
    })
    let yaxR: HTMLElement | null = null
    const bmr = Array.isArray(b.bmrSeries) ? b.bmrSeries : []
    if (bmr.length >= 2) {
      const byDayB = new Map<string, { ts: number; bmr: number }>()
      for (const p of bmr) {
        const e = byDayB.get(p.date)
        if (!e || p.ts > e.ts) byDayB.set(p.date, { ts: p.ts, bmr: p.bmr })
      }
      const bd = [...byDayB.values()].sort((p, q) => p.ts - q.ts)
      let blo = Infinity
      let bhi = -Infinity
      for (const p of bd) {
        if (p.bmr < blo) blo = p.bmr
        if (p.bmr > bhi) bhi = p.bmr
      }
      const brange = Math.max(40, bhi - blo)
      const bLoP = blo - brange * 0.18
      const bHiP = bhi + brange * 0.18
      const bY = (v: number): number => (1 - (v - bLoP) / (bHiP - bLoP)) * 100
      s.appendChild(
        svg('path', { d: polyD(bd.map(p => [xPct(p.ts), bY(p.bmr)])), class: 'tri-bodywt-bmr' }),
      )
      const lastB = bd[bd.length - 1]
      const bm = el('span', 'tri-bodywt-bpt')
      bm.style.left = `${xPct(lastB.ts).toFixed(2)}%`
      bm.style.top = `${bY(lastB.bmr).toFixed(2)}%`
      plot.appendChild(bm)
      yaxR = el('div', 'tri-bodywt-yax tri-bodywt-yax-r')
      yaxR.append(el('span', '', `${Math.round(bHiP)}`), el('span', '', `${Math.round(bLoP)}`))
    }
    chart.append(yax, plot)
    if (yaxR) {
      chart.appendChild(yaxR)
      block.classList.add('tri-bodywt--bmr')
    }
    const xax = el('div', 'tri-bodywt-xax')
    xax.append(
      el('span', '', shortDate(days[0].date)),
      el('span', '', shortDate(days[nd - 1].date)),
    )
    block.append(chart, xax)
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const cap = el('div', 'tri-elev-cap')
  if (b.trendKgPerWeek != null)
    cap.appendChild(
      markGloss(
        el('span', 'tri-ana-k', `${wSigned(b.trendKgPerWeek, 2)} ${weightUnit}/wk`),
        'wtrend',
      ),
    )
  if (b.goalKg != null) {
    const delta = b.goalDeltaKg != null ? ` (${wSigned(b.goalDeltaKg, 1)} ${weightUnit})` : ''
    const eta = b.goalEtaWeeks != null ? ` · $\\approx${b.goalEtaWeeks}$ wk` : ''
    cap.appendChild(markGloss(mathK('tri-ana-k', `goal ${wFmt(b.goalKg)}${delta}${eta}`), 'wgoal'))
  }
  if (b.bodyFatPct != null)
    cap.appendChild(
      markGloss(el('span', 'tri-ana-k', `fat ${b.bodyFatPct.toFixed(1)}%`), 'bodyfat'),
    )
  if (b.bmi != null)
    cap.appendChild(markGloss(el('span', 'tri-ana-k', `bmi ${b.bmi.toFixed(1)}`), 'bmi'))
  if (b.latestBmr != null)
    cap.appendChild(markGloss(el('span', 'tri-ana-k tri-bmr-k', `BMR ${b.latestBmr} kcal`), 'bmr'))
  if (b.muscleMassKg != null)
    cap.appendChild(el('span', 'tri-ana-k', `muscle ${wFmt(b.muscleMassKg, 1, 1)}`))
  if (b.boneMassKg != null)
    cap.appendChild(el('span', 'tri-ana-k', `bone ${wFmt(b.boneMassKg, 1, 1)}`))
  if (b.bodyWaterPct != null)
    cap.appendChild(el('span', 'tri-ana-k', `water ${b.bodyWaterPct.toFixed(1)}%`))
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
  const yMaxE = niceUp(mx)
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
    const h = (w.effort / yMaxE) * (H - 2)
    s.appendChild(
      svg('rect', { x: i + 0.12, y: bot - h, width: 0.76, height: h, class: 'tri-seg--effort' }),
    )
  })
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' }))
  block.appendChild(
    axisFrame(
      s,
      [yMaxE, yMaxE / 2, 0].map(v => ({
        label: String(Math.round(v)),
        vbY: bot - (v / yMaxE) * (H - 2),
      })),
      H,
      monthTicks(
        all.map(w => w.weekStart),
        i => ((i + 0.5) / n) * 100,
      ),
      56,
    ),
  )
  block.appendChild(el('div', 'tri-chart-readout'))
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

const segRuns = <T>(
  rows: T[],
  sel: (r: T) => number | null,
  x: (i: number) => number,
  y: (v: number) => number,
): [number, number][][] => {
  const out: [number, number][][] = []
  let cur: [number, number][] = []
  rows.forEach((r, i) => {
    const v = sel(r)
    if (v == null) {
      if (cur.length > 1) out.push(cur)
      cur = []
      return
    }
    cur.push([x(i), y(v)])
  })
  if (cur.length > 1) out.push(cur)
  return out
}

const buildRecoveryChart = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-recovery')
  block.appendChild(anaTitle('recovery · hrv · rhr', 'hrv'))
  const rec = data.recovery
  if (!rec.series.length) {
    block.appendChild(el('div', 'tri-ana-empty', 'no recovery data'))
    return block
  }
  if (rec.flags.length) {
    const flags = el('div', 'tri-rec-flags')
    for (const f of rec.flags) {
      const row = el('div', `tri-rec-flag tri-flag--${f.severity}`)
      row.appendChild(el('span', 'tri-rec-dot'))
      row.appendChild(markGloss(el('span', 'tri-rec-flag-label', f.label), f.metric))
      row.appendChild(el('span', 'tri-rec-flag-detail', f.detail))
      flags.appendChild(row)
    }
    block.appendChild(flags)
  }
  const n = rec.series.length
  if (rec.status !== 'building' && n > 1) {
    const x = (i: number): number => (i / (n - 1)) * ANA_W
    const yZ = (z: number): number => 15 - (clampN(z, -3, 3) / 3) * 12
    const s = svg('svg', {
      class: 'tri-ana-svg tri-rec-svg',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    s.appendChild(
      svg('rect', { x: 0, y: yZ(1), width: ANA_W, height: yZ(-1) - yZ(1), class: 'tri-rec-band' }),
    )
    s.appendChild(svg('line', { x1: 0, y1: yZ(0), x2: ANA_W, y2: yZ(0), class: 'tri-ana-zero' }))
    s.appendChild(svg('line', { x1: 0, y1: yZ(-1), x2: ANA_W, y2: yZ(-1), class: 'tri-rec-zline' }))
    s.appendChild(
      svg('line', {
        x1: 0,
        y1: yZ(-2),
        x2: ANA_W,
        y2: yZ(-2),
        class: 'tri-rec-zline tri-rec-zline--alert',
      }),
    )
    for (const seg of segRuns(rec.series, d => d.hrvZ, x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-hrv' }))
    for (const seg of segRuns(rec.series, d => (d.rhrZ == null ? null : -d.rhrZ), x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-rhr' }))
    s.appendChild(svg('line', { x1: ANA_W, y1: 0, x2: ANA_W, y2: ANA_H, class: 'tri-pmc-now' }))
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    block.appendChild(
      axisFrame(
        s,
        [
          { label: '+2σ', vbY: yZ(2) },
          { label: '0', vbY: yZ(0) },
          { label: '-2σ', vbY: yZ(-2) },
        ],
        ANA_H,
        monthTicks(
          rec.series.map(d => d.date),
          i => (i / (n - 1)) * ANA_W,
        ),
        150,
      ),
    )
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const t = rec.thresholds
  const sevCls = (cond: boolean | null, alert: boolean | null): string =>
    alert ? 'tri-flag--alert' : cond ? 'tri-flag--watch' : ''
  const hrvCls = sevCls(
    rec.hrvZ != null && rec.hrvZ <= t.hrvWatchZ,
    rec.hrvZ != null && rec.hrvZ <= t.hrvAlertZ,
  )
  const rhrCls = sevCls(
    rec.rhrZ != null && rec.rhrZ >= t.rhrWatchZ,
    rec.rhrZ != null && rec.rhrZ >= t.rhrAlertZ,
  )
  const tmpCls = sevCls(
    rec.tempDevLatest != null && rec.tempDevLatest >= t.tempWatchC,
    rec.tempDevLatest != null && rec.tempDevLatest >= t.tempAlertC,
  )
  const rdyCls =
    rec.readinessLatest != null && rec.readinessLatest < t.readinessFloor ? 'tri-flag--watch' : ''
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    markGloss(
      el(
        'span',
        `tri-ana-k ${hrvCls}`.trim(),
        rec.hrvLatest != null ? `HRV ${Math.round(rec.hrvLatest)} ms` : 'HRV —',
      ),
      'hrv',
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${rhrCls}`.trim(),
        rec.rhrLatest != null ? `RHR ${Math.round(rec.rhrLatest)}` : 'RHR —',
      ),
      'rhr',
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${rdyCls}`.trim(),
        rec.readinessLatest != null
          ? `readiness ${Math.round(rec.readinessLatest)}`
          : 'readiness —',
      ),
      'oreadiness',
    ),
    markGloss(
      mathK(
        `tri-ana-k ${tmpCls}`.trim(),
        rec.tempDevLatest != null
          ? `temp ${rec.tempDevLatest >= 0 ? '+' : ''}${rec.tempDevLatest.toFixed(1)}$^\\circ\\mathrm{C}$`
          : 'temp —',
      ),
      'tempdev',
    ),
  )
  if (rec.status !== 'firm')
    cap.appendChild(el('span', 'tri-ana-k', `baseline ${rec.baselineDays}/14`))
  block.appendChild(cap)
  return block
}

const buildSleep = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-sleep')
  block.appendChild(anaTitle('sleep · debt', 'sleepdebt'))
  const rec = data.recovery
  const view = rec.series.slice(-28)
  if (!view.some(d => d.sleepS != null)) {
    block.appendChild(el('div', 'tri-ana-empty', 'no sleep logged'))
    return block
  }
  const n = view.length
  const H = 32
  const bot = H - 0.5
  let maxS = rec.sleepTargetS
  for (const d of view) if (d.sleepS != null && d.sleepS > maxS) maxS = d.sleepS
  maxS = Math.ceil(maxS / 3600) * 3600
  const yBar = (sec: number): number => bot - (sec / maxS) * (H - 2)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-ana-weekly-svg',
    viewBox: `0 0 ${n} ${H}`,
    preserveAspectRatio: 'none',
  })
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: yBar(rec.sleepTargetS),
      x2: n,
      y2: yBar(rec.sleepTargetS),
      class: 'tri-rec-target',
    }),
  )
  s.appendChild(
    svg('line', {
      x1: 0,
      y1: yBar(rec.thresholds.sleepFloorS),
      x2: n,
      y2: yBar(rec.thresholds.sleepFloorS),
      class: 'tri-rec-floor',
    }),
  )
  view.forEach((d, i) => {
    if (d.sleepS == null) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (d.sleepS / maxS) * (H - 2)
    s.appendChild(
      svg('rect', {
        x: i + 0.12,
        y: bot - h,
        width: 0.76,
        height: h,
        class: d.sleepS < rec.thresholds.sleepFloorS ? 'tri-seg--short' : 'tri-seg--sleep',
      }),
    )
  })
  const ys = (sc: number): number => bot - (sc / 100) * (H - 2)
  for (const seg of segRuns(
    view,
    d => d.sleepScore,
    i => i + 0.5,
    ys,
  ))
    s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-score' }))
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' }))
  const yMaxHr = Math.round(maxS / 3600)
  block.appendChild(
    axisFrame(
      s,
      [yMaxHr, yMaxHr / 2, 0].map(v => ({
        label: v === 0 ? '0' : `${v % 1 === 0 ? v : v.toFixed(1)}h`,
        vbY: yBar(v * 3600),
      })),
      H,
      monthTicks(
        view.map(d => d.date),
        i => ((i + 0.5) / n) * 100,
      ),
      56,
    ),
  )
  block.appendChild(el('div', 'tri-chart-readout'))
  const debtCls =
    rec.sleepDebtS >= rec.thresholds.sleepDebtAlertS
      ? 'tri-flag--alert'
      : rec.sleepDebtS >= rec.thresholds.sleepDebtWatchS
        ? 'tri-flag--watch'
        : ''
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el(
      'span',
      'tri-ana-k',
      rec.sleepLatestS != null ? `sleep ${hms(rec.sleepLatestS)}` : 'sleep —',
    ),
    el(
      'span',
      'tri-ana-k',
      rec.sleepBaselineS != null ? `base ${hms(rec.sleepBaselineS)}` : 'base —',
    ),
    markGloss(
      el('span', `tri-ana-k ${debtCls}`.trim(), `debt ${(rec.sleepDebtS / 3600).toFixed(1)} h`),
      'sleepdebt',
    ),
    el('span', 'tri-ana-k', `target ${hms(rec.sleepTargetS)}`),
  )
  if (rec.shortSleepStreak >= 2)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k tri-flag--${rec.shortSleepStreak >= 3 ? 'alert' : 'watch'}`,
        `${rec.shortSleepStreak} short`,
      ),
    )
  block.appendChild(cap)
  return block
}

const aceBand = (pct: number): string =>
  pct < 6
    ? 'essential'
    : pct < 14
      ? 'athlete'
      : pct < 18
        ? 'fitness'
        : pct < 25
          ? 'average'
          : 'obese'

const buildDexa = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-dexa')
  const titleRow = el('div', 'tri-dexa-titlerow')
  titleRow.appendChild(anaTitle('body composition', 'dexa'))
  const d = data.tests.dexa[data.tests.dexa.length - 1]
  if (!d) {
    block.appendChild(titleRow)
    block.appendChild(el('div', 'tri-ana-empty', 'no dexa scan logged'))
    return block
  }
  titleRow.appendChild(el('span', 'tri-dexa-date', d.date))
  block.appendChild(titleRow)

  const head = el('div', 'tri-dexa-head')
  const bf = el('div', 'tri-dexa-bf', d.bodyFat.toFixed(1))
  bf.appendChild(el('span', 'tri-dexa-unit', '% fat'))
  head.append(bf, el('span', 'tri-dexa-cat', `ACE ${aceBand(d.bodyFat)}`))
  block.appendChild(head)

  const total = d.fatLbs + d.leanLbs + d.bmcLbs
  const seg = (cls: string, lbs: number, label: string): HTMLElement => {
    const s = el('span', `tri-dexa-seg ${cls}`)
    s.style.width = `${(lbs / total) * 100}%`
    s.title = `${label} ${lbs.toFixed(1)} lb · ${((lbs / total) * 100).toFixed(0)}%`
    return s
  }
  const bar = el('div', 'tri-dexa-bar')
  bar.append(
    seg('is-lean', d.leanLbs, 'lean'),
    seg('is-fat', d.fatLbs, 'fat'),
    seg('is-bone', d.bmcLbs, 'bone'),
  )
  block.appendChild(bar)

  const legend = el('div', 'tri-dexa-legend')
  const leg = (cls: string, name: string, lbs: number): HTMLElement => {
    const w = el('span', 'tri-dexa-legitem')
    w.append(
      el('span', `tri-dexa-dot ${cls}`),
      el('span', 'tri-dexa-legname', name),
      el('span', 'tri-dexa-legval', `${lbs.toFixed(1)} lb`),
    )
    return w
  }
  legend.append(
    leg('is-lean', 'lean', d.leanLbs),
    leg('is-fat', 'fat', d.fatLbs),
    leg('is-bone', 'bone', d.bmcLbs),
  )
  block.appendChild(legend)

  const regions = [
    ['arms', d.arms],
    ['legs', d.legs],
    ['trunk', d.trunk],
  ] as const
  const reg = el('div', 'tri-dexa-regions')
  for (const [name, r] of regions) {
    if (!r) continue
    const rtot = r.fat + r.lean + r.bmc
    const row = el('div', 'tri-dexa-region')
    const rbar = el('div', 'tri-dexa-rbar')
    const rseg = (cls: string, lbs: number): HTMLElement => {
      const s = el('span', `tri-dexa-seg ${cls}`)
      s.style.width = `${(lbs / rtot) * 100}%`
      return s
    }
    rbar.append(rseg('is-lean', r.lean), rseg('is-fat', r.fat), rseg('is-bone', r.bmc))
    row.append(
      el('span', 'tri-dexa-rlabel', name),
      rbar,
      el(
        'span',
        'tri-dexa-rval',
        `${rtot.toFixed(0)} lb · ${((r.fat / rtot) * 100).toFixed(0)}% fat`,
      ),
    )
    reg.appendChild(row)
  }
  block.appendChild(reg)

  const stats = el('div', 'tri-dexa-stats')
  const stat = (label: string, val: string): void => {
    const c = el('div', 'tri-dexa-stat')
    c.append(el('span', 'tri-dexa-statv', val), el('span', 'tri-dexa-statk', label))
    stats.appendChild(c)
  }
  stat('lean', `${d.leanLbs.toFixed(1)} lb`)
  if (d.rmr != null) stat('rmr', `${d.rmr} kcal`)
  if (d.bmd != null)
    stat('bmd', `${d.bmd.toFixed(2)}${d.bmdT != null ? ` · T${signed(d.bmdT)}` : ''}`)
  if (d.vatLbs != null) stat('vat', `${d.vatLbs.toFixed(2)} lb`)
  if (d.rsmi != null) stat('rsmi', d.rsmi.toFixed(1))
  if (d.ag != null) stat('a/g', d.ag.toFixed(2))
  block.appendChild(stats)
  return block
}

const buildVo2max = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-engine-vo2')
  block.appendChild(anaTitle('vo2max · fitness age', 'vo2max'))
  const v = data.engine.vo2max
  if (v.value == null) {
    block.appendChild(el('div', 'tri-ana-empty', 'no power or hr data yet'))
    return block
  }
  const head = el('div', 'tri-engine-vo2-head')
  const num = el('div', 'tri-engine-vo2-num', v.value.toFixed(1))
  num.appendChild(el('span', 'tri-engine-vo2-unit', ' ml/kg/min'))
  head.appendChild(num)
  if (v.fitnessAge != null)
    head.appendChild(
      markGloss(
        el(
          'span',
          `tri-engine-age tri-dir-${(v.ageDeltaYears ?? 0) <= 0 ? 'up' : 'down'}`,
          `fitness age ${v.fitnessAge} (${signed(v.ageDeltaYears ?? 0)}y)`,
        ),
        'fitage',
      ),
    )
  block.appendChild(head)
  const pos = (a: number): number => clampN(((a - 20) / 60) * 100, 1.5, 98.5)
  const bar = el('div', 'tri-engine-agebar')
  if (v.fitnessAge != null) {
    const needle = el('span', 'tri-engine-agebar-needle')
    needle.style.left = `${pos(v.fitnessAge)}%`
    needle.title = `fitness age ${v.fitnessAge}`
    bar.appendChild(needle)
  }
  const chrono = el('span', 'tri-engine-agebar-chrono')
  chrono.style.left = `${pos(v.chronoAge)}%`
  chrono.title = `age ${v.chronoAge}`
  bar.appendChild(chrono)
  block.appendChild(bar)
  if (v.trend.length > 1) {
    const n = v.trend.length
    let lo = Infinity
    let hi = -Infinity
    for (const p of v.trend) {
      if (p.vo2max < lo) lo = p.vo2max
      if (p.vo2max > hi) hi = p.vo2max
    }
    if (hi - lo < 2) {
      hi += 1
      lo -= 1
    }
    const x = (i: number): number => (i / (n - 1)) * ANA_W
    const y = (val: number): number => 27 - ((val - lo) / (hi - lo)) * 24
    const s = svg('svg', {
      class: 'tri-ana-svg tri-engine-vo2-spark',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    s.appendChild(
      svg('path', {
        d: polyD(v.trend.map((p, i) => [x(i), y(p.vo2max)] as [number, number])),
        class: 'tri-elev-line tri-line-bike',
      }),
    )
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    block.appendChild(s)
    block.appendChild(el('div', 'tri-chart-readout'))
  }
  const cap = el('div', 'tri-elev-cap')
  cap.append(
    el('span', 'tri-ana-k', v.method),
    markGloss(el('span', `tri-ana-k tri-conf-${v.conf}`, v.conf), 'conf'),
  )
  if (v.percentileForAge != null)
    cap.appendChild(el('span', 'tri-ana-k', `p${v.percentileForAge} for age ${v.chronoAge}`))
  cap.appendChild(el('span', 'tri-ana-k', v.note))
  cap.appendChild(el('span', 'tri-ana-k', `hrmax ${v.hrMax} (${v.hrMaxSource})`))
  block.appendChild(cap)
  const lab = data.tests.vo2max[data.tests.vo2max.length - 1]
  if (lab) {
    const labCap = el('div', 'tri-elev-cap tri-vo2-lab')
    if (lab.vt1Hr != null)
      labCap.appendChild(
        el(
          'span',
          'tri-ana-k',
          `vt1 ${lab.vt1Hr}bpm${lab.vt1Kmh != null ? ` · ${lab.vt1Kmh}km/h` : ''}`,
        ),
      )
    if (lab.maxKmh != null) labCap.appendChild(el('span', 'tri-ana-k', `vmax ${lab.maxKmh}km/h`))
    if (lab.ve != null) labCap.appendChild(el('span', 'tri-ana-k', `ve ${lab.ve}l/min`))
    labCap.appendChild(el('span', 'tri-ana-k', `lab ${lab.date}`))
    block.appendChild(labCap)
  }
  return block
}

const buildAbilities = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-engine-radar')
  block.appendChild(anaTitle('abilities', 'radar'))
  const ab = data.engine.abilities
  if (!ab.axes.length || ab.axes.every(a => a.score == null)) {
    block.appendChild(el('div', 'tri-ana-empty', 'not enough data'))
    return block
  }
  const cx = 50
  const cy = 50
  const R = 36
  const angle = (i: number): number => ((-90 + (360 / ab.axes.length) * i) * Math.PI) / 180
  const pt = (i: number, score: number): [number, number] => {
    const th = angle(i)
    const r = (R * score) / 100
    return [cx + r * Math.cos(th), cy + r * Math.sin(th)]
  }
  const s = svg('svg', { class: 'tri-radar-svg', viewBox: '0 0 100 100' })
  for (const g of [25, 50, 75, 100])
    s.appendChild(
      svg('path', { d: `${polyD(ab.axes.map((_, i) => pt(i, g)))} Z`, class: 'tri-radar-grid' }),
    )
  ab.axes.forEach((_, i) => {
    const [px, py] = pt(i, 100)
    s.appendChild(svg('line', { x1: cx, y1: cy, x2: px, y2: py, class: 'tri-radar-spoke' }))
  })
  s.appendChild(
    svg('path', {
      d: `${polyD(ab.axes.map((a, i) => pt(i, a.score ?? 0)))} Z`,
      class: 'tri-radar-fill',
    }),
  )
  ab.axes.forEach((a, i) => {
    const [px, py] = pt(i, a.score ?? 0)
    s.appendChild(
      svg('circle', {
        cx: px,
        cy: py,
        r: 1.4,
        class: a.score == null ? 'tri-radar-dot tri-radar-dot--null' : 'tri-radar-dot',
      }),
    )
    const th = angle(i)
    const label = svg('text', {
      x: cx + (R + 8) * Math.cos(th),
      y: cy + (R + 8) * Math.sin(th) + 1.6,
      'text-anchor': Math.abs(Math.cos(th)) < 0.3 ? 'middle' : Math.cos(th) > 0 ? 'start' : 'end',
      class: a.score == null ? 'tri-radar-ax tri-radar-ax--null' : 'tri-radar-ax',
    })
    label.textContent = a.label
    s.appendChild(label)
  })
  block.appendChild(s)
  return block
}

const buildCardio = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-engine-cardio')
  block.appendChild(anaTitle('cardiovascular health', 'ef'))
  const c = data.engine.cardio
  if (!c.metrics.length || c.metrics.every(m => m.value == null)) {
    block.appendChild(el('div', 'tri-ana-empty', 'no heart data yet'))
    return block
  }
  const seriesOf = (key: string): number[] => {
    if (key === 'rhr') return c.rhrSeries.map(p => p.rhr)
    if (key === 'hrv') return c.hrvSeries.map(p => p.hrv)
    if (key === 'ef') return c.efSeries.map(p => p.ef)
    if (key === 'decoupling') return c.decouplingSeries.map(p => p.pct)
    return []
  }
  const glossOf: Record<string, string> = {
    rhr: 'rhr',
    hrv: 'hrv',
    ef: 'ef',
    decoupling: 'decouple',
  }
  for (const m of c.metrics) {
    const row = el('div', 'tri-engine-row')
    row.dataset.metric = m.key
    row.dataset.label = m.label
    row.dataset.unit = m.unit
    row.appendChild(markGloss(el('span', 'tri-engine-row-k', m.label), glossOf[m.key] ?? 'ef'))
    const ys = seriesOf(m.key)
    if (ys.length > 1) {
      const n = ys.length
      let lo = Infinity
      let hi = -Infinity
      for (const yv of ys) {
        if (yv < lo) lo = yv
        if (yv > hi) hi = yv
      }
      if (hi - lo === 0) {
        hi += 1
        lo -= 1
      }
      const s = svg('svg', {
        class: 'tri-engine-spark',
        viewBox: '0 0 100 24',
        preserveAspectRatio: 'none',
      })
      s.appendChild(
        svg('path', {
          d: polyD(
            ys.map(
              (yv, i) =>
                [(i / (n - 1)) * 100, 21 - ((yv - lo) / (hi - lo)) * 18] as [number, number],
            ),
          ),
          class: `tri-elev-line ${m.key === 'hrv' ? 'tri-line-bike' : m.key === 'ef' ? 'tri-line-swim' : m.key === 'rhr' ? 'tri-line-run' : ''}`,
        }),
      )
      s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: 24, class: 'tri-ana-cursor' }))
      row.appendChild(s)
      row.appendChild(el('div', 'tri-chart-readout'))
    } else row.appendChild(el('span', 'tri-engine-spark'))
    const val = el(
      'span',
      'tri-engine-row-v',
      m.value != null ? `${m.value}${m.unit === '%' ? '%' : ` ${m.unit}`}` : '—',
    )
    val.title = m.note
    row.appendChild(val)
    row.appendChild(
      el(
        'span',
        `tri-engine-row-dir ${m.dir === 'improving' ? 'tri-dir-up' : m.dir === 'declining' ? 'tri-dir-down' : 'tri-dir-flat'}`,
        m.dir === 'improving' ? '▲' : m.dir === 'declining' ? '▼' : m.dir === 'stable' ? '■' : '',
      ),
    )
    block.appendChild(row)
  }
  return block
}

const ANALYTICS_BUILDERS: Record<string, (data: Analytics) => HTMLElement> = {
  body: buildBody,
  dexa: buildDexa,
  gauge: buildGauge,
  recovery: buildRecoveryChart,
  sleep: buildSleep,
  vo2max: buildVo2max,
  abilities: buildAbilities,
  cardio: buildCardio,
  pmc: buildPmc,
  'ctl-sport': buildCtlSport,
  weekly: buildWeekly,
  effort: buildEffort,
  readiness: buildReadiness,
  trend: buildTrend,
  actions: buildActions,
}

const scrubBind = (
  hover: HTMLElement,
  svgEl: SVGElement,
  cursor: SVGElement,
  readout: HTMLElement,
  count: number,
  vbW: number,
  textOf: (i: number) => string,
): (() => void) => {
  if (count < 2) return () => {}
  const onMove = (event: MouseEvent) => {
    const r = svgEl.getBoundingClientRect()
    const frac = clampN((event.clientX - r.left) / r.width, 0, 1)
    const cx = (frac * vbW).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    setMath(readout, textOf(Math.round(frac * (count - 1))))
    hover.classList.add('tri-chart--hover')
  }
  const onLeave = () => hover.classList.remove('tri-chart--hover')
  svgEl.addEventListener('mousemove', onMove)
  svgEl.addEventListener('mouseleave', onLeave)
  return () => {
    svgEl.removeEventListener('mousemove', onMove)
    svgEl.removeEventListener('mouseleave', onLeave)
  }
}

type ScrubItem = {
  svgEl: SVGElement
  cursor: SVGElement
  readout: HTMLElement
  hover: HTMLElement
  textOf: (f: number) => string
}

const scrubGroup = (items: ScrubItem[], cursorXOf: (f: number) => number): (() => void) => {
  if (items.length === 0) return () => {}
  const move = (event: MouseEvent, ref: SVGElement) => {
    const r = ref.getBoundingClientRect()
    const f = clampN((event.clientX - r.left) / r.width, 0, 1)
    const cx = cursorXOf(f).toFixed(2)
    for (const it of items) {
      it.cursor.setAttribute('x1', cx)
      it.cursor.setAttribute('x2', cx)
      it.hover.classList.add('tri-chart--hover')
      setMath(it.readout, it.textOf(f))
    }
  }
  const leave = () => {
    for (const it of items) it.hover.classList.remove('tri-chart--hover')
  }
  const offs: (() => void)[] = []
  for (const it of items) {
    const onMove = (e: MouseEvent) => move(e, it.svgEl)
    it.svgEl.addEventListener('mousemove', onMove)
    it.svgEl.addEventListener('mouseleave', leave)
    offs.push(() => {
      it.svgEl.removeEventListener('mousemove', onMove)
      it.svgEl.removeEventListener('mouseleave', leave)
    })
  }
  return () => offs.forEach(f => f())
}

const wireScrub = (panel: HTMLElement, data: Analytics): (() => void) => {
  const cleanups: (() => void)[] = []
  const bind = (
    blockSel: string,
    svgSel: string,
    count: number,
    vbW: number,
    textOf: (i: number) => string,
  ) => {
    const block = panel.querySelector<HTMLElement>(blockSel)
    const svgEl = block?.querySelector<SVGElement>(svgSel)
    const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
    const readout = block?.querySelector<HTMLElement>('.tri-chart-readout')
    if (block && svgEl && cursor && readout)
      cleanups.push(scrubBind(block, svgEl, cursor, readout, count, vbW, textOf))
  }

  const daily = data.daily
  const rec = data.recovery.series
  bind('.tri-ana-recovery', '.tri-rec-svg', rec.length, ANA_W, i => {
    const d = rec[i]
    const z = d.hrvZ != null ? ` $${signed(d.hrvZ)}\\sigma$` : ''
    return `${d.date} · HRV ${d.hrv ?? '—'}${z} · RHR ${d.rhr ?? '—'} · rdy ${d.readiness ?? '—'}`
  })

  const sleepView = data.recovery.series.slice(-28)
  bind('.tri-ana-sleep', '.tri-ana-weekly-svg', sleepView.length, sleepView.length, i => {
    const d = sleepView[i]
    const debt = d.sleepDebtS != null ? `${(d.sleepDebtS / 3600).toFixed(1)}h` : '—'
    return `${d.date} · ${d.sleepS != null ? hms(d.sleepS) : '—'} · score ${d.sleepScore ?? '—'} · debt ${debt}`
  })

  const trend = data.engine.vo2max.trend
  bind('.tri-engine-vo2', '.tri-engine-vo2-spark', trend.length, ANA_W, i => {
    const p = trend[i]
    return `${p.weekStart} · ${p.vo2max.toFixed(1)} ml/kg/min`
  })

  bind('.tri-ana-ctlsport', '.tri-ana-svg', daily.length, ANA_W, i => {
    const d = daily[i]
    return `${d.date} · swim ${Math.round(d.swimCtl)} · bike ${Math.round(d.bikeCtl)} · run ${Math.round(d.runCtl)}`
  })

  const wk = data.weekly
  bind('.tri-ana-weekly', '.tri-ana-weekly-svg', wk.length, wk.length, i => {
    const w = wk[i]
    return `${w.weekStart} · load ${Math.round(w.load)}`
  })
  bind('.tri-ana-effort', '.tri-ana-weekly-svg', wk.length, wk.length, i => {
    const w = wk[i]
    return `${w.weekStart} · effort ${Math.round(w.effort)}`
  })

  const bodySeries = data.body.series
  const bodyBlock = panel.querySelector<HTMLElement>('.tri-ana-bodywt')
  const bodyPlot = bodyBlock?.querySelector<HTMLElement>('.tri-bodywt-plot')
  const bodyCursor = bodyPlot?.querySelector<SVGElement>('.tri-ana-cursor')
  const bodyReadout = bodyBlock?.querySelector<HTMLElement>('.tri-chart-readout')
  if (bodyBlock && bodyPlot && bodyCursor && bodyReadout && bodySeries.length >= 2) {
    const bdays = groupBodyByDay(bodySeries)
    const bmrByDay = new Map<string, number>()
    for (const p of Array.isArray(data.body.bmrSeries) ? data.body.bmrSeries : [])
      bmrByDay.set(p.date, p.bmr)
    const bt0 = bdays[0].ts
    const bt1 = bdays[bdays.length - 1].ts
    const bx = (ts: number): number => (bt1 > bt0 ? ((ts - bt0) / (bt1 - bt0)) * 100 : 50)
    const ranges = Array.from(bodyPlot.querySelectorAll<SVGLineElement>('.tri-bodywt-range'))
    const onMove = (event: MouseEvent) => {
      const r = bodyPlot.getBoundingClientRect()
      const fx = clampN((event.clientX - r.left) / r.width, 0, 1) * 100
      let best = bdays[0]
      let bestD = Infinity
      for (const d of bdays) {
        const dd = Math.abs(bx(d.ts) - fx)
        if (dd < bestD) {
          bestD = dd
          best = d
        }
      }
      const cx = bx(best.ts).toFixed(2)
      bodyCursor.setAttribute('x1', cx)
      bodyCursor.setAttribute('x2', cx)
      const bmrV = bmrByDay.get(best.date)
      const bmrTxt = bmrV != null ? ` · BMR ${bmrV} kcal` : ''
      if (best.samples.length > 1) {
        const delta = best.last - best.first
        setMath(
          bodyReadout,
          `${shortDate(best.date)} · $${best.samples.length}\\times$ · ${wNum(best.min)}–${wNum(best.max)} ${weightUnit} · $\\Delta${wSigned(delta, 1)}$${bmrTxt}`,
        )
      } else {
        setMath(bodyReadout, `${shortDate(best.date)} · ${wFmt(best.last)}${bmrTxt}`)
      }
      for (const ln of ranges)
        ln.classList.toggle('tri-bodywt-range--active', ln.dataset.day === best.date)
      bodyBlock.classList.add('tri-chart--hover')
    }
    const onLeave = () => {
      bodyBlock.classList.remove('tri-chart--hover')
      for (const ln of ranges) ln.classList.remove('tri-bodywt-range--active')
    }
    bodyPlot.addEventListener('mousemove', onMove)
    bodyPlot.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      bodyPlot.removeEventListener('mousemove', onMove)
      bodyPlot.removeEventListener('mouseleave', onLeave)
    })
  }

  const cardioBlock = panel.querySelector<HTMLElement>('.tri-engine-cardio')
  if (cardioBlock) {
    const ser: Record<string, { date: string; v: number }[]> = {
      rhr: data.engine.cardio.rhrSeries.map(p => ({ date: p.date, v: p.rhr })),
      hrv: data.engine.cardio.hrvSeries.map(p => ({ date: p.date, v: p.hrv })),
      ef: data.engine.cardio.efSeries.map(p => ({ date: p.date, v: p.ef })),
      decoupling: data.engine.cardio.decouplingSeries.map(p => ({ date: p.date, v: p.pct })),
    }
    for (const row of Array.from(cardioBlock.querySelectorAll<HTMLElement>('.tri-engine-row'))) {
      const points = ser[row.dataset.metric ?? '']
      const svgEl = row.querySelector<SVGElement>('.tri-engine-spark')
      const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
      const readout = row.querySelector<HTMLElement>('.tri-chart-readout')
      if (svgEl && cursor && readout && points && points.length > 1) {
        const label = row.dataset.label ?? ''
        const unit = row.dataset.unit ?? ''
        cleanups.push(
          scrubBind(row, svgEl, cursor, readout, points.length, 100, i => {
            const p = points[i]
            return `${p.date} · ${label} ${p.v}${unit ? ` ${unit}` : ''}`
          }),
        )
      }
    }
  }

  const trendBlock = panel.querySelector<HTMLElement>('.tri-ana-trend')
  if (trendBlock) {
    const items: ScrubItem[] = []
    for (const sport of ['swim', 'bike', 'run'] as Sport[]) {
      const wrap = trendBlock.querySelector<HTMLElement>(`.tri-trend-panel[data-sport="${sport}"]`)
      const svgEl = wrap?.querySelector<SVGElement>('.tri-trend-svg')
      const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
      const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
      const tr = bySport(data.trends, sport)
      if (
        !wrap ||
        !svgEl ||
        !cursor ||
        !readout ||
        !tr ||
        tr.level == null ||
        tr.forecast.length < 1
      )
        continue
      const level = tr.level
      const weeks = tr.forecast.length / 7
      const endVal = level + (tr.slopePerWeek ?? 0) * weeks
      items.push({
        svgEl,
        cursor,
        readout,
        hover: wrap,
        textOf: f =>
          `+${(f * weeks).toFixed(1)} wk · ${fmtTrendVal(sport, level + (endVal - level) * f)}`,
      })
    }
    cleanups.push(scrubGroup(items, f => f * ANA_W))
  }

  const radarSvg = panel.querySelector<SVGElement>('.tri-engine-radar .tri-radar-svg')
  if (radarSvg) {
    document.body.querySelector('.tri-radar-tip')?.remove()
    const radarTip = el('div', 'tri-gloss tri-radar-tip')
    radarTip.setAttribute('role', 'tooltip')
    document.body.appendChild(radarTip)
    const axes = data.engine.abilities.axes
    const vbPt = (i: number, score: number): [number, number] => {
      const th = ((-90 + (360 / axes.length) * i) * Math.PI) / 180
      const r = (36 * score) / 100
      return [50 + r * Math.cos(th), 50 + r * Math.sin(th)]
    }
    const onMove = (event: MouseEvent) => {
      const rect = radarSvg.getBoundingClientRect()
      let best = -1
      let bestD = Infinity
      axes.forEach((a, i) => {
        const [vx, vy] = vbPt(i, a.score ?? 0)
        const d = Math.hypot(
          rect.left + (vx / 100) * rect.width - event.clientX,
          rect.top + (vy / 100) * rect.height - event.clientY,
        )
        if (d < bestD) {
          bestD = d
          best = i
        }
      })
      const a = best >= 0 ? axes[best] : null
      if (!a) return
      const raw = a.rawValue != null ? `${a.rawValue} ${a.rawUnit}` : 'no data'
      radarTip.replaceChildren(
        el('span', 'tri-gloss-h', a.label),
        el('span', 'tri-gloss-def', `${raw} · ${a.score != null ? `${a.score}/100` : '—'}`),
      )
      radarTip.classList.add('tri-gloss--on')
      const pr = radarTip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      radarTip.style.left = `${Math.max(8, left).toFixed(0)}px`
      radarTip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const onLeave = () => radarTip.classList.remove('tri-gloss--on')
    radarSvg.addEventListener('mousemove', onMove)
    radarSvg.addEventListener('mouseleave', onLeave)
    cleanups.push(() => {
      radarSvg.removeEventListener('mousemove', onMove)
      radarSvg.removeEventListener('mouseleave', onLeave)
      radarTip.remove()
    })
  }

  return () => {
    for (const c of cleanups) c()
  }
}

const SEARCH_SECTIONS: { label: string; chart: string; hay: string }[] = [
  {
    label: 'body weight',
    chart: 'body',
    hay: 'body weight kg lbs mass cut goal fat bmi muscle bone water composition scale index',
  },
  { label: 'form · ramp', chart: 'gauge', hay: 'form ramp gauge taper peak projection' },
  {
    label: 'recovery · hrv · rhr',
    chart: 'recovery',
    hay: 'recovery hrv heart rate variability rhr resting autonomic illness temperature overreaching suppressed fatigue oura',
  },
  {
    label: 'sleep · debt',
    chart: 'sleep',
    hay: 'sleep debt duration score short streak need target rest hours oura',
  },
  {
    label: 'vo2max · fitness age',
    chart: 'vo2max',
    hay: 'vo2max vo2 max aerobic fitness age friend percentile engine ftp map',
  },
  {
    label: 'abilities',
    chart: 'abilities',
    hay: 'abilities radar sprint threshold endurance climb cadence recovery power profile vam wkg',
  },
  {
    label: 'cardiovascular health',
    chart: 'cardio',
    hay: 'cardio cardiovascular heart rhr hrv efficiency factor decoupling aerobic drift',
  },
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
  wgoal: 'body',
  bodyfat: 'body',
  bmi: 'body',
  hrv: 'recovery',
  rhr: 'recovery',
  tempdev: 'recovery',
  overreaching: 'recovery',
  oreadiness: 'recovery',
  sleepdebt: 'sleep',
  vo2max: 'vo2max',
  fitage: 'vo2max',
  vam: 'abilities',
  radar: 'abilities',
  ef: 'cardio',
  decouple: 'cardio',
}

const searchCommandTitle = (prefix: string, value?: string): HTMLElement => {
  const wrap = el('span', 'tri-search-command')
  wrap.appendChild(el('span', 'tri-search-command-token', prefix))
  if (value) wrap.appendChild(el('span', 'tri-search-command-value', value))
  return wrap
}

const modalBaseY = (): number => -50

const flipClose = (
  btn: HTMLElement,
  panel: HTMLElement,
  reduce: boolean,
  finish: () => void,
): Animation | null => {
  if (reduce) {
    finish()
    return null
  }
  const br = btn.getBoundingClientRect()
  const pr = panel.getBoundingClientRect()
  if (pr.width < 1 || pr.height < 1 || br.width < 1) {
    finish()
    return null
  }
  const dx = br.left + br.width / 2 - (pr.left + pr.width / 2)
  const dy = br.top + br.height / 2 - (pr.top + pr.height / 2)
  const sx = Math.max(0.05, br.width / pr.width)
  const sy = Math.max(0.05, br.height / pr.height)
  const anim = panel.animate(
    [
      { opacity: 1, transform: `translate(-50%, ${modalBaseY()}%) scale(1, 1)` },
      {
        opacity: 0,
        transform: `translate(-50%, ${modalBaseY()}%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
      },
    ],
    { duration: 240, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' },
  )
  const done = () => {
    finish()
    anim.cancel()
  }
  anim.finished.then(done).catch(done)
  return anim
}

const setupAnalytics = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-analytics-btn')
  const panel = root.querySelector<HTMLElement>('.tri-analytics')
  const scrim = root.querySelector<HTMLElement>('.tri-analytics-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-ana-close')
  const title = root.querySelector<HTMLElement>('.tri-ana-title')
  const search = root.querySelector<HTMLInputElement>('.tri-ana-search')
  const results = root.querySelector<HTMLElement>('.tri-ana-results')
  const pageMode = root.dataset.triView === 'analytics'
  if (!panel || (!btn && !pageMode)) return null

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
    for (const block of Array.from(panel.querySelectorAll<HTMLElement>('.tri-ana-block'))) {
      const build = ANALYTICS_BUILDERS[block.dataset.chart ?? '']
      if (build) block.replaceChildren(build(d))
    }
    scrubCleanup = wireScrub(panel, d)
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
  let closeAnim: Animation | null = null
  let closeSeq = 0
  const close = () => {
    if (!btn) return
    const seq = ++closeSeq
    closeAnim?.cancel()
    closeAnim = flipClose(btn, panel, reduce, () => {
      if (seq !== closeSeq) return
      root.classList.remove('tri-analytics-open')
      panel.setAttribute('aria-hidden', 'true')
      if (search) search.value = ''
      panel.classList.remove('tri-analytics--searching')
      closeDetail()
      if (results) results.replaceChildren()
      selIndex = -1
    })
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
        DETAIL_FTP = d.ftp ?? null
        DETAIL_GOAL_FTP = d.goalFtp ?? null
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
        const fv = t.slice(7)
        filterSport = fv === 'hike' ? 'walk' : fv
      } else if (t.startsWith('sort:')) {
        sortKey = t.slice(5)
      } else {
        if (t) tokens.push(t)
      }
    }

    const metrics: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1]
    const hints: HTMLElement[] = []

    if (
      lastToken.startsWith('filter:') &&
      !['bike', 'run', 'swim', 'walk', 'hike'].includes(lastToken.slice(7))
    ) {
      const prefix = lastToken.slice(7)
      for (const f of ['bike', 'run', 'swim', 'walk']) {
        if (f.startsWith(prefix)) {
          const it = ritem(searchCommandTitle('filter:', f), 'filter activities')
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
          const it = ritem(searchCommandTitle('sort:', s), 'sort activities')
          it.dataset.insert = `sort:${s}`
          hints.push(it)
        }
      }
    } else if (lastToken.length > 0 && 'filter:'.startsWith(lastToken) && lastToken !== 'filter:') {
      const it = ritem(searchCommandTitle('filter:'), 'filter by sport (bike, run, swim, walk)')
      it.dataset.insert = 'filter:'
      hints.push(it)
    } else if (lastToken.length > 0 && 'sort:'.startsWith(lastToken) && lastToken !== 'sort:') {
      const it = ritem(searchCommandTitle('sort:'), 'sort by distance, cadence, pace')
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
          (a.cadence ? (a.sport === 'run' ? ` · ${a.cadence * 2} spm` : ` · ${a.cadence} rpm`) : '')
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
    if (!btn) return
    closeSeq++
    closeAnim?.cancel()
    closeAnim = null
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
          transform: `translate(-50%, ${modalBaseY()}%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
        },
        { opacity: 1, transform: `translate(-50%, ${modalBaseY()}%) scale(1, 1)` },
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

  if (pageMode) {
    panel.classList.add('tri-analytics--page')
    panel.setAttribute('aria-hidden', 'false')
    load()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  search?.addEventListener('input', runSearch)
  search?.addEventListener('keydown', onSearchKey)
  results?.addEventListener('click', onResultsClick)
  detail?.addEventListener('click', onDetailToggle)
  document.addEventListener('keydown', onKey)
  const onUnitChange = () => {
    if (data) render(data)
  }
  document.addEventListener('tri-weightunit', onUnitChange)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    results?.removeEventListener('click', onResultsClick)
    detail?.removeEventListener('click', onDetailToggle)
    document.removeEventListener('keydown', onKey)
    document.removeEventListener('tri-weightunit', onUnitChange)
    scrubCleanup?.()
  }
}

type GeoFC = { type: 'FeatureCollection'; features: unknown[] }
const emptyFC = (): GeoFC => ({ type: 'FeatureCollection', features: [] })

const gpsRoute = (d: StravaActivityDetail): readonly StravaMapPoint[] =>
  d.mapRoute && d.mapRoute.length >= 2 ? d.mapRoute : d.route

const lineFeature = (route: readonly StravaMapPoint[], props: Record<string, unknown> = {}) => ({
  type: 'Feature',
  properties: props,
  geometry: { type: 'LineString', coordinates: route.map(p => [p.lng, p.lat]) },
})

type OverviewMode = 'heat' | 'w' | 'hr' | 'cad' | 'spd'

interface OverviewLegend {
  lo: string
  hi: string
}

interface Overview {
  heat: GeoFC
  traces: GeoFC
  legend: Record<OverviewMode, OverviewLegend | null>
}

const OVERVIEW_METRICS = ['w', 'hr', 'cad', 'spd'] as const
const OVERVIEW_CELL = 0.0008

const overviewCellKey = (lng: number, lat: number): string =>
  `${Math.round(lng / OVERVIEW_CELL)},${Math.round(lat / OVERVIEW_CELL)}`

const stampSegment = (a: StravaMapPoint, b: StravaMapPoint, into: Set<string>) => {
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(b.lng - a.lng), Math.abs(b.lat - a.lat)) / (OVERVIEW_CELL / 2)),
  )
  for (let s = 0; s <= steps; s++) {
    const t = s / steps
    into.add(overviewCellKey(a.lng + (b.lng - a.lng) * t, a.lat + (b.lat - a.lat) * t))
  }
}

const pctRange = (vals: number[]): [number, number] => {
  const sorted = [...vals].sort((x, y) => x - y)
  const lo = sorted[Math.floor(0.1 * (sorted.length - 1))]
  const hi = sorted[Math.ceil(0.9 * (sorted.length - 1))]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

const overviewMetric = (d: StravaActivityDetail, k: (typeof OVERVIEW_METRICS)[number]) => {
  if (k === 'w') return d.deviceWatts && d.avgWatts ? d.avgWatts : null
  if (k === 'hr') return d.avgHr
  if (k === 'cad') return d.avgCadence == null ? null : d.avgCadence * (d.sport === 'run' ? 2 : 1)
  return d.movingTimeS > 0 ? d.distanceKm / (d.movingTimeS / 3600) : null
}

const overviewFmt = (k: (typeof OVERVIEW_METRICS)[number], sport: Sport, v: number): string => {
  if (k === 'w') return `${Math.round(v)} W`
  if (k === 'hr') return `${Math.round(v)} bpm`
  if (k === 'cad') return `${Math.round(v)} ${sport === 'run' ? 'spm' : 'rpm'}`
  if (sport === 'bike') return `${(v * KM_TO_MI).toFixed(1)} mph`
  return `${clock(3600 / (v * KM_TO_MI))} /mi`
}

const buildOverview = (dp: DetailPayload | null, sportFilter: 'run' | 'bike' | null): Overview => {
  const acts: StravaActivityDetail[] = []
  const det = dp?.details ?? {}
  for (const k in det) {
    const d = det[k]
    if (sportFilter && d.sport !== sportFilter) continue
    if ((d.sport === 'run' || d.sport === 'bike') && gpsRoute(d).length >= 2) acts.push(d)
  }
  const counts = new Map<string, number>()
  for (const d of acts) {
    const cells = new Set<string>()
    const r = gpsRoute(d)
    for (let i = 0; i < r.length - 1; i++) stampSegment(r[i], r[i + 1], cells)
    for (const c of cells) counts.set(c, (counts.get(c) ?? 0) + 1)
  }
  let maxCount = 1
  for (const c of counts.values()) if (c > maxCount) maxCount = c
  const bucketOf = (c: number): number =>
    maxCount > 1
      ? Math.min(7, Math.max(1, 1 + Math.round((6 * Math.log(c)) / Math.log(maxCount))))
      : 1
  const heatFeatures: unknown[] = []
  for (const d of acts) {
    const r = gpsRoute(d)
    let runB = -1
    let coords: [number, number][] = []
    const flush = () => {
      if (coords.length >= 2)
        heatFeatures.push({
          type: 'Feature',
          properties: { id: d.id, heat: runB },
          geometry: { type: 'LineString', coordinates: coords },
        })
    }
    for (let i = 0; i < r.length - 1; i++) {
      const mid = overviewCellKey((r[i].lng + r[i + 1].lng) / 2, (r[i].lat + r[i + 1].lat) / 2)
      const b = bucketOf(counts.get(mid) ?? 1)
      if (b !== runB) {
        flush()
        runB = b
        coords = [[r[i].lng, r[i].lat]]
      }
      coords.push([r[i + 1].lng, r[i + 1].lat])
    }
    flush()
  }
  const ranges = new Map<string, [number, number]>()
  for (const k of OVERVIEW_METRICS)
    for (const sport of ['run', 'bike'] as const) {
      const vals: number[] = []
      for (const d of acts)
        if (d.sport === sport) {
          const v = overviewMetric(d, k)
          if (v != null && v > 0) vals.push(v)
        }
      if (vals.length) ranges.set(`${k}:${sport}`, pctRange(vals))
    }
  const traceFeatures: unknown[] = []
  for (const d of acts) {
    const props: Record<string, unknown> = { id: d.id }
    for (const k of OVERVIEW_METRICS) {
      const v = overviewMetric(d, k)
      const range = ranges.get(`${k}:${d.sport}`)
      props[k] =
        v != null && v > 0 && range
          ? Math.min(1, Math.max(0, (v - range[0]) / (range[1] - range[0])))
          : -1
    }
    traceFeatures.push(lineFeature(gpsRoute(d), props))
  }
  const legend: Record<OverviewMode, OverviewLegend | null> = {
    heat: { lo: '$1\\times$', hi: `$${maxCount}\\times$` },
    w: null,
    hr: null,
    cad: null,
    spd: null,
  }
  for (const k of OVERVIEW_METRICS) {
    const present = (['run', 'bike'] as const).filter(s => ranges.has(`${k}:${s}`))
    if (present.length === 1) {
      const [lo, hi] = ranges.get(`${k}:${present[0]}`)!
      legend[k] = { lo: overviewFmt(k, present[0], lo), hi: overviewFmt(k, present[0], hi) }
    }
  }
  return {
    heat: { type: 'FeatureCollection', features: heatFeatures },
    traces: { type: 'FeatureCollection', features: traceFeatures },
    legend,
  }
}

const heatColorExpr: unknown[] = (() => {
  const e: unknown[] = ['interpolate', ['linear'], ['get', 'heat']]
  HEAT_RAMP.forEach((c, i) => e.push(i + 1, c))
  return e
})()

const heatOpacityExpr: unknown[] = ['interpolate', ['linear'], ['get', 'heat'], 1, 0.25, 7, 0.9]

const heatWidthExpr: unknown[] = (() => {
  const w = (base: number, k: number) => ['+', base, ['*', k, ['-', ['get', 'heat'], 1]]]
  return ['interpolate', ['linear'], ['zoom'], 10, w(0.8, 0.12), 14, w(1.3, 0.22), 16, w(1.9, 0.3)]
})()

const overviewRamp = (m: OverviewMode): string[] =>
  m === 'hr' ? HR_RAMP : m === 'cad' ? CAD_RAMP : m === 'spd' ? SPD_RAMP : HEAT_RAMP

const traceColorExpr = (k: OverviewMode): unknown[] => {
  const ramp: unknown[] = ['interpolate', ['linear'], ['get', k]]
  overviewRamp(k).forEach((c, i) => ramp.push(i / 6, c))
  return ['case', ['<', ['get', k], 0], '#b7b3ac', ramp]
}

const traceOpacityExpr = (k: OverviewMode): unknown[] => ['case', ['<', ['get', k], 0], 0.12, 0.6]

const routeFC = (d: StravaActivityDetail): GeoFC => ({
  type: 'FeatureCollection',
  features: [lineFeature(gpsRoute(d))],
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
  colors: string[] = HEAT_RAMP,
  zeroGap = false,
): unknown[] => {
  const vals = d.route.map((p, i) => pick(p, i))
  const pool = zeroGap ? vals.filter(v => v > 0) : vals
  let lo = Infinity
  let hi = -Infinity
  for (const v of pool.length ? pool : vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const range = hi > lo ? hi - lo : 1
  const dN = d.route[d.route.length - 1].d || 1
  const pairs: [number, string][] = []
  let lastT = -1
  d.route.forEach((p, i) => {
    const v = vals[i]
    if (zeroGap && v <= 0 && (vals[i - 1] ?? 0) > 0 && (vals[i + 1] ?? 0) > 0) return
    const t = Math.min(1, Math.max(0, p.d / dN))
    if (t <= lastT) return
    lastT = t
    const bucket =
      zeroGap && v <= 0 ? 1 : Math.min(7, Math.max(1, Math.ceil(((v - lo) / range) * 7) || 1))
    pairs.push([t, colors[bucket - 1]])
  })
  if (pairs.length === 0) pairs.push([0, colors[3]])
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
  const pageMode = root.dataset.triView === 'maps'
  if (!panel || (!btn && !pageMode)) return null

  const body = root.querySelector<HTMLElement>('.tri-map-body')
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let loaded = false
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailLoaded = false
  let selIndex = -1
  const canvas = root.querySelector<HTMLElement>('.tri-map-canvas')
  const overlay = root.querySelector<HTMLElement>('.tri-map-overlay')
  const tip = root.querySelector<HTMLElement>('.tri-map-tip')
  const legendLo = overlay?.querySelector<HTMLElement>('.tri-map-legend-lo') ?? null
  const legendHi = overlay?.querySelector<HTMLElement>('.tri-map-legend-hi') ?? null
  const legendBar = overlay?.querySelector<HTMLElement>('.tri-map-legend-bar') ?? null
  const modeBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-map-mode'))
  let mode: OverviewMode = 'heat'
  let sportFilter: 'run' | 'bike' | null = null
  const overviewCache = new Map<string, Overview>()
  const getOverview = () => {
    const key = sportFilter ?? 'all'
    let ov = overviewCache.get(key)
    if (!ov) {
      ov = buildOverview(detailData, sportFilter)
      overviewCache.set(key, ov)
    }
    return ov
  }

  const mapCtl = (() => {
    let map: any = null
    let started = false
    let okFlag = false
    let hoverId: string | null = null
    const clearHover = () => {
      hoverId = null
      map?.getSource('tri-hov')?.setData(emptyFC())
      if (map) map.getCanvas().style.cursor = ''
      tip?.classList.remove('tri-map-tip--on')
    }
    const applyMode = () => {
      if (!map) return
      map.setLayoutProperty('tri-heat', 'visibility', mode === 'heat' ? 'visible' : 'none')
      map.setLayoutProperty('tri-traces', 'visibility', mode === 'heat' ? 'none' : 'visible')
      map.setPaintProperty('tri-heat', 'line-opacity', heatOpacityExpr)
      if (mode !== 'heat') {
        map.setPaintProperty('tri-traces', 'line-color', traceColorExpr(mode))
        map.setPaintProperty('tri-traces', 'line-opacity', traceOpacityExpr(mode))
      }
      const lg = getOverview().legend[mode]
      if (legendLo) setMath(legendLo, lg?.lo ?? 'low')
      if (legendHi) setMath(legendHi, lg?.hi ?? 'high')
      if (legendBar) legendBar.style.background = rampGradient(overviewRamp(mode))
    }
    const recolor = (d: StravaActivityDetail, i: number) => {
      if (!map) return
      const spec = metricSpecs(d)[i]
      if (spec)
        map.setPaintProperty(
          'tri-sel',
          'line-gradient',
          gradientExpr(d, spec.pick, spec.ramp, spec.zeroGap),
        )
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
          'line-color': heatColorExpr,
          'line-opacity': heatOpacityExpr,
          'line-width': heatWidthExpr,
        },
      })
      map.addSource('tri-traces', { type: 'geojson', data: emptyFC() })
      map.addLayer({
        id: 'tri-traces',
        type: 'line',
        source: 'tri-traces',
        layout: { 'line-cap': 'round', 'line-join': 'round', visibility: 'none' },
        paint: {
          'line-color': '#fc4c02',
          'line-opacity': 0.5,
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 1.2, 14, 2, 16, 3],
        },
      })
      map.addLayer({
        id: 'tri-hit',
        type: 'line',
        source: 'tri-traces',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 12 },
      })
      map.addSource('tri-hov', { type: 'geojson', data: emptyFC() })
      map.addLayer({
        id: 'tri-hov-casing',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fff9f3', 'line-width': 4.6 },
      })
      map.addLayer({
        id: 'tri-hov',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fc4c02', 'line-width': 2.8 },
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
      map.on('mousemove', 'tri-hit', (e: any) => {
        if (panel.classList.contains('tri-map--detail')) return
        const id = e.features?.[0]?.properties?.id
        if (id == null) return
        map.getCanvas().style.cursor = 'pointer'
        const key = String(id)
        if (hoverId !== key) {
          hoverId = key
          const d = detailData?.details?.[key]
          if (!d) return
          map.getSource('tri-hov')?.setData(routeFC(d))
          if (tip)
            tip.textContent = `${d.name || d.sport} · ${shortDate(d.date)} · ${dist(d.distanceKm, d.sport)}`
        }
        if (tip) {
          tip.classList.add('tri-map-tip--on')
          const bound = canvas.clientWidth - tip.offsetWidth - 8
          tip.style.left = `${Math.min(e.point.x + 14, Math.max(8, bound))}px`
          tip.style.top = `${e.point.y + 14}px`
        }
      })
      map.on('mouseleave', 'tri-hit', clearHover)
      map.on('click', 'tri-hit', (e: any) => {
        if (panel.classList.contains('tri-map--detail')) return
        const id = e.features?.[0]?.properties?.id
        if (id != null) showRoute(String(id))
      })
      applyMode()
      okFlag = true
    }
    const drawOverview = () => {
      if (!map) return
      const ov = getOverview()
      map.getSource('tri-heat')?.setData(ov.heat)
      map.getSource('tri-traces')?.setData(ov.traces)
      applyMode()
      const b = fcBounds(ov.traces)
      if (b) map.fitBounds(b, { padding: 48, maxZoom: 13, duration: reduce ? 0 : 600 })
    }
    const select = (d: StravaActivityDetail, i: number) => {
      if (!map) return
      clearHover()
      map.getSource('tri-sel')?.setData(routeFC(d))
      recolor(d, i)
      map.setPaintProperty('tri-heat', 'line-opacity', 0.06)
      map.setPaintProperty('tri-traces', 'line-opacity', 0.06)
      const b = fcBounds(routeFC(d))
      if (b) map.fitBounds(b, { padding: 40, maxZoom: 15, duration: reduce ? 0 : 600 })
    }
    const moveDot = (lng: number, lat: number) =>
      map?.getSource('tri-dot')?.setData(pointFC(lng, lat))
    const clearSelection = () => {
      if (!map) return
      clearHover()
      map.getSource('tri-sel')?.setData(emptyFC())
      map.getSource('tri-dot')?.setData(emptyFC())
      drawOverview()
    }
    const resize = () => map?.resize()
    const dispose = () => {
      clearHover()
      if (map?.remove) map.remove()
      map = null
      started = false
      okFlag = false
      if (canvas) (canvas as unknown as { _mapInstance: unknown })._mapInstance = null
    }
    return {
      init,
      ok: () => okFlag,
      drawOverview,
      applyMode,
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
        DETAIL_FTP = d.ftp ?? null
        DETAIL_GOAL_FTP = d.goalFtp ?? null
        overviewCache.clear()
        mapCtl.drawOverview()
        if (search?.value) runSearch()
      })
      .catch(() => {})
  }
  const setSportFilter = (f: 'run' | 'bike' | null) => {
    if (f === sportFilter) return
    sportFilter = f
    mapCtl.drawOverview()
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
    setSportFilter(null)
  }
  let closeAnim: Animation | null = null
  let closeSeq = 0
  const close = () => {
    if (!btn) return
    const seq = ++closeSeq
    closeAnim?.cancel()
    closeAnim = flipClose(btn, panel, reduce, () => {
      if (seq !== closeSeq) return
      root.classList.remove('tri-map-open')
      panel.setAttribute('aria-hidden', 'true')
      toMain()
    })
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
      panel.classList.remove('tri-map--searching')
      results?.setAttribute('aria-hidden', 'true')
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
      setSportFilter(null)
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
    setSportFilter(filterSport === 'run' || filterSport === 'bike' ? filterSport : null)
    const hints: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1]
    if (lastToken.startsWith('filter:') && !['bike', 'run'].includes(lastToken.slice(7))) {
      const prefix = lastToken.slice(7)
      for (const f of ['bike', 'run'])
        if (f.startsWith(prefix)) {
          const it = ritem(searchCommandTitle('filter:', f), 'filter routes')
          it.dataset.insert = `filter:${f}`
          hints.push(it)
        }
    } else if (lastToken.length > 0 && 'filter:'.startsWith(lastToken) && lastToken !== 'filter:') {
      const it = ritem(searchCommandTitle('filter:'), 'filter by sport (bike, run)')
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
      mapCtl.drawOverview()
    })
  const open = () => {
    if (!btn) return
    closeSeq++
    closeAnim?.cancel()
    closeAnim = null
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
          transform: `translate(-50%, ${modalBaseY()}%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
        },
        { opacity: 1, transform: `translate(-50%, ${modalBaseY()}%) scale(1, 1)` },
      ],
      { duration: 300, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
    )
    anim.finished.then(startMap).catch(startMap)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (panel.classList.contains('tri-map--searching') && search?.value) {
      search.value = ''
      runSearch()
      return
    }
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
  const onPanelClick = (event: MouseEvent) => {
    if (!panel.classList.contains('tri-map--searching')) return
    if ((event.target as HTMLElement | null)?.closest('.tri-map-search-wrap')) return
    panel.classList.remove('tri-map--searching')
    results?.setAttribute('aria-hidden', 'true')
  }
  const onModeClick = (event: MouseEvent) => {
    const b = (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-map-mode')
    const m = b?.dataset.mode as OverviewMode | undefined
    if (!m || m === mode) return
    mode = m
    const accent = overviewRamp(m)[6]
    for (const it of modeBtns) {
      const on = it.dataset.mode === m
      it.setAttribute('aria-pressed', String(on))
      it.style.background = on ? accent : ''
      it.style.borderColor = on ? accent : ''
    }
    mapCtl.applyMode()
  }

  if (pageMode) {
    panel.classList.add('tri-map--page')
    panel.setAttribute('aria-hidden', 'false')
    load()
    void loadDetails()
    startMap()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  search?.addEventListener('input', runSearch)
  search?.addEventListener('focus', runSearch)
  search?.addEventListener('keydown', onSearchKey)
  results?.addEventListener('click', onResultsClick)
  panel.addEventListener('click', onPanelClick)
  overlay?.addEventListener('click', onModeClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('focus', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    results?.removeEventListener('click', onResultsClick)
    panel.removeEventListener('click', onPanelClick)
    overlay?.removeEventListener('click', onModeClick)
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

  const dists = root.querySelectorAll<HTMLElement>('.tri-dist[data-km]')
  let mi = false
  const onClick = () => {
    mi = !mi
    unit.textContent = mi ? 'mi' : 'km'
    setDistanceUnit(mi)
    for (const c of cells) {
      if (!mi) {
        c.textContent = c.dataset.km ?? ''
      } else {
        const v = Number(c.dataset.km) * KM_TO_MI
        c.textContent = v < 10 ? v.toFixed(2) : v.toFixed(1)
      }
    }
    for (const d of dists) {
      const km = Number(d.dataset.km)
      const kind = d.dataset.kind ?? 'combined'
      d.textContent = kind === 'combined' ? distCombined(km) : dist(km, kind as ActivityKind)
    }
    window.dispatchEvent(new CustomEvent('tri:unit'))
  }
  unit.addEventListener('click', onClick)
  return () => {
    window.clearTimeout(showTimer)
    unit.removeEventListener('click', onClick)
    ann?.remove()
    setDistanceUnit(false)
  }
}

const setupTraining = (root: HTMLElement): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-training-btn')
  const panel = root.querySelector<HTMLElement>('.tri-training')
  const scrim = root.querySelector<HTMLElement>('.tri-training-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-training-close')
  const title = root.querySelector<HTMLElement>('.tri-training-title')
  const search = root.querySelector<HTMLInputElement>('.tri-training-search')
  const results = root.querySelector<HTMLElement>('.tri-training-results')
  const list = root.querySelector<HTMLElement>('.tri-training-plans')
  const tree = root.querySelector<HTMLElement>('.tri-training-tree')
  const preview = root.querySelector<HTMLElement>('.tri-training-doc')
  const pageMode = root.dataset.triView === 'training'
  if (!panel || (!btn && !pageMode)) return null

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let loaded = false
  let data: TrainingPayload | null = null
  let selIndex = -1

  const showPlan = (plan: TrainingPlan) => {
    if (!preview) return
    const head = el('div', 'tri-pop-head tri-training-head')
    head.appendChild(el('span', 'tri-pop-date tri-training-meta-name', plan.meta))
    const meta = el('ul', 'tri-training-meta')
    const metaRow = (label: string, value: string) => {
      if (!value) return
      const li = el('li')
      li.append(el('span', 'tri-training-meta-k', label), el('span', 'tri-training-meta-v', value))
      meta.appendChild(li)
    }
    metaRow('distance', plan.distance)
    metaRow('date', plan.date)
    metaRow('objectif', plan.target)
    metaRow(
      'avec',
      plan.author
        ? plan.author
            .split(',')
            .map(s => s.trim())
            .join(', ')
        : '',
    )
    if (meta.childElementCount) head.appendChild(meta)
    const body = el('div', 'tri-training-render')
    body.innerHTML = plan.html
    preview.replaceChildren(head, body)
    preview.scrollTo({ top: 0 })
    buildTree(plan, body)
    document.dispatchEvent(
      new CustomEvent('contentdecrypted', { detail: { article: preview, content: preview } }),
    )
  }

  const buildTree = (plan: TrainingPlan, body: HTMLElement) => {
    if (!tree) return
    const heads = Array.from(body.querySelectorAll<HTMLElement>('h2, h3, h4')).filter(
      h => !h.closest('.footnotes'),
    )
    tree.replaceChildren()
    if (!heads.length) return
    type TNode = { id: string; label: string; level: number; children: TNode[] }
    const roots: TNode[] = []
    const stack: TNode[] = []
    heads.forEach((h, i) => {
      const id = `tri-h-${plan.id}-${i}`
      h.id = id
      const node: TNode = {
        id,
        label: (h.textContent ?? '').trim(),
        level: Number(h.tagName[1]),
        children: [],
      }
      while (stack.length && stack[stack.length - 1].level >= node.level) stack.pop()
      ;(stack.length ? stack[stack.length - 1].children : roots).push(node)
      stack.push(node)
    })
    const nb = ' '
    const seg = (bar: boolean) => (bar ? `│${nb.repeat(3)}` : nb.repeat(4))
    const lines: HTMLElement[] = []
    const walk = (node: TNode, last: boolean, anc: boolean[]) => {
      const prefix = anc.map(seg).join('') + (last ? '└── ' : '├── ')
      const line = el('div', 'tri-training-tree-line')
      line.appendChild(el('span', 'tri-training-tree-prefix', prefix))
      const link = el('button', 'tri-training-tree-link', node.label)
      link.setAttribute('type', 'button')
      link.dataset.target = node.id
      line.appendChild(link)
      lines.push(line)
      node.children.forEach((c, i) => walk(c, i === node.children.length - 1, [...anc, !last]))
    }
    roots.forEach((n, i) => walk(n, i === roots.length - 1, []))
    tree.replaceChildren(...lines)
  }

  const select = (idx: number) => {
    if (!data || !data.plans.length) return
    const items = list ? Array.from(list.querySelectorAll<HTMLElement>('.tri-ana-ritem')) : []
    selIndex = clampN(idx, 0, data.plans.length - 1)
    items.forEach((it, i) => it.classList.toggle('tri-ana-ritem--sel', i === selIndex))
    showPlan(data.plans[selIndex])
  }

  const ritem = (plan: TrainingPlan, i: number): HTMLElement => {
    const it = el('button', 'tri-ana-ritem')
    it.setAttribute('type', 'button')
    it.dataset.plan = String(i)
    it.append(
      el('span', 'tri-ana-ritem-t', plan.meta),
      el('span', 'tri-ana-ritem-s', [plan.distance, plan.target].filter(Boolean).join(' · ')),
    )
    return it
  }

  const renderList = () => {
    if (!list || !data) return
    list.replaceChildren(...data.plans.map(ritem))
    if (data.plans.length) select(0)
    else preview?.replaceChildren(el('div', 'tri-ana-empty', 'no plan'))
  }

  const load = () => {
    if (loaded) return
    loaded = true
    const path = root.dataset.trainingPath
    if (!path) return
    fetch(path)
      .then(res => res.json())
      .then((d: TrainingPayload) => {
        data = d
        renderList()
      })
      .catch(() => {})
  }

  let closeAnim: Animation | null = null
  let closeSeq = 0
  const toMain = () => {
    if (search) search.value = ''
    panel.classList.remove('tri-training--searching')
    results?.replaceChildren()
  }
  const open = () => {
    if (!btn) return
    closeSeq++
    closeAnim?.cancel()
    closeAnim = null
    root.classList.add('tri-training-open')
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
          transform: `translate(-50%, ${modalBaseY()}%) translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`,
        },
        { opacity: 1, transform: `translate(-50%, ${modalBaseY()}%) scale(1, 1)` },
      ],
      { duration: 300, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
    )
  }
  const close = () => {
    if (!btn) return
    const seq = ++closeSeq
    closeAnim?.cancel()
    closeAnim = flipClose(btn, panel, reduce, () => {
      if (seq !== closeSeq) return
      root.classList.remove('tri-training-open')
      panel.setAttribute('aria-hidden', 'true')
      toMain()
    })
  }

  const runSearch = () => {
    if (!search || !results || !data) return
    const q = search.value.trim().toLowerCase()
    if (!q) {
      panel.classList.remove('tri-training--searching')
      results.replaceChildren()
      return
    }
    panel.classList.add('tri-training--searching')
    const hits = data.plans
      .map((p, i) => ({ p, i }))
      .filter(({ p }) => `${p.meta} ${p.distance} ${p.target} ${p.date}`.toLowerCase().includes(q))
    results.replaceChildren(
      ...(hits.length
        ? hits.map(({ p, i }) => ritem(p, i))
        : [el('div', 'tri-ana-empty', 'no matches')]),
    )
  }

  const activate = (it?: HTMLElement | null) => {
    if (!it || it.dataset.plan == null) return
    select(Number(it.dataset.plan))
    toMain()
  }
  const onListClick = (event: MouseEvent) =>
    activate((event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-ana-ritem'))
  const onResultsClick = (event: MouseEvent) =>
    activate((event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-ana-ritem'))
  const onTreeClick = (event: MouseEvent) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-target]')
    if (!t?.dataset.target || !preview) return
    const target = preview.querySelector<HTMLElement>(`[id="${CSS.escape(t.dataset.target)}"]`)
    if (!target) return
    preview.scrollTo({
      top:
        preview.scrollTop +
        target.getBoundingClientRect().top -
        preview.getBoundingClientRect().top -
        8,
      behavior: 'smooth',
    })
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || !root.classList.contains('tri-training-open')) return
    if (search && search.value) {
      search.value = ''
      runSearch()
      return
    }
    close()
  }

  if (pageMode) {
    panel.classList.add('tri-training--page')
    panel.setAttribute('aria-hidden', 'false')
    load()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  search?.addEventListener('input', runSearch)
  results?.addEventListener('click', onResultsClick)
  list?.addEventListener('click', onListClick)
  tree?.addEventListener('click', onTreeClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    results?.removeEventListener('click', onResultsClick)
    list?.removeEventListener('click', onListClick)
    tree?.removeEventListener('click', onTreeClick)
    document.removeEventListener('keydown', onKey)
  }
}

const renderGlossDef = (def: string): HTMLElement => {
  const span = el('span', 'tri-gloss-def')
  span.replaceChildren(...mathFrag(def))
  return span
}

const LEG_GLOSS: Record<string, { term: string; def: string }> = {
  legdist: {
    term: 'total distance',
    def: 'Distance covered in this discipline over the season window. Swims read in metres; ride and run follow the km/mi toggle.',
  },
  legcount: {
    term: 'sessions',
    def: 'Number of workouts logged in this discipline over the window.',
  },
  legtime: { term: 'total time', def: 'Total moving time across these sessions.' },
  herodist: {
    term: 'season total',
    def: 'Combined swim, bike, and run distance over the window. Follows the km/mi toggle.',
  },
}

const setupGloss = (root: HTMLElement): (() => void) | null => {
  const zones = ['.tri-analytics', '.tri-foot', '.tri-head']
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
    const key = term.dataset.gloss ?? ''
    const g = GLOSS[key] ?? LEG_GLOSS[key]
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

const setupAxisLabels = (root: HTMLElement): (() => void) | null => {
  const axis = root.querySelector<HTMLElement>('.tri-axis')
  if (!axis) return null
  const labels = [...axis.querySelectorAll<HTMLElement>('.tri-axis-year')]
  if (labels.length === 0) return null
  const visible = new Map<Element, boolean>()
  let frame: number | null = null
  const apply = () => {
    frame = null
    const viewportBottom = window.innerHeight - 1
    const clippedByRect = labels.some(label => {
      const r = label.getBoundingClientRect()
      return r.top < 0 || r.bottom > viewportBottom
    })
    root.classList.toggle(
      'tri-axis-labels-hidden',
      clippedByRect || [...visible.values()].some(ok => !ok),
    )
  }
  const schedule = () => {
    if (frame == null) frame = window.requestAnimationFrame(apply)
  }
  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries)
        visible.set(entry.target, entry.isIntersecting && entry.intersectionRatio >= 0.98)
      schedule()
    },
    { threshold: [0, 0.98, 1] },
  )
  for (const label of labels) {
    visible.set(label, true)
    observer.observe(label)
  }
  const resize = new ResizeObserver(schedule)
  resize.observe(root)
  resize.observe(axis)
  window.addEventListener('resize', schedule, { passive: true })
  schedule()
  return () => {
    observer.disconnect()
    resize.disconnect()
    window.removeEventListener('resize', schedule)
    if (frame != null) window.cancelAnimationFrame(frame)
    root.classList.remove('tri-axis-labels-hidden')
  }
}

const setupShortcuts = (root: HTMLElement): (() => void) => {
  let waitingForG = false
  let gTimeout: number | null = null

  const modalChords: Record<string, { btn: string; openClass: string; close: string }> = {
    a: { btn: '.tri-analytics-btn', openClass: 'tri-analytics-open', close: '.tri-ana-close' },
    c: { btn: '.tri-calc-btn', openClass: 'tri-calc-open', close: '.tri-calc-close' },
    m: { btn: '.tri-map-btn', openClass: 'tri-map-open', close: '.tri-map-close' },
    t: { btn: '.tri-training-btn', openClass: 'tri-training-open', close: '.tri-training-close' },
  }
  const closeOpenModals = (except?: string) => {
    for (const k in modalChords) {
      if (k === except) continue
      const mc = modalChords[k]
      if (root.classList.contains(mc.openClass)) root.querySelector<HTMLElement>(mc.close)?.click()
    }
  }
  const toggleModal = (key: string) => {
    const mc = modalChords[key]
    if (root.classList.contains(mc.openClass)) {
      root.querySelector<HTMLElement>(mc.close)?.click()
      return
    }
    closeOpenModals(key)
    root.querySelector<HTMLElement>(mc.btn)?.click()
  }
  const toggleSearchFocus = (): boolean => {
    const search =
      (root.classList.contains('tri-map-open')
        ? root.querySelector<HTMLInputElement>('.tri-map-search')
        : null) ??
      (root.classList.contains('tri-analytics-open')
        ? root.querySelector<HTMLInputElement>('.tri-analytics .tri-ana-search')
        : null)
    if (!search) return false
    if (document.activeElement === search) {
      search.blur()
    } else {
      search.focus()
      search.select()
    }
    return true
  }

  const onKey = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'k') {
      if (toggleSearchFocus()) {
        e.preventDefault()
        e.stopImmediatePropagation()
      }
      return
    }

    if (e.shiftKey && (e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'g') {
      e.preventDefault()
      e.stopImmediatePropagation()
      const url = new URL('/triathlon/tools', window.location.toString())
      if (window.spaNavigate) window.spaNavigate(url)
      else window.location.href = url.toString()
      return
    }

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
      if (key === 'a' || key === 'c' || key === 'm' || key === 't') {
        toggleModal(key)
        handled = true
      } else if (key === 'g') {
        closeOpenModals()
        root.querySelector<HTMLElement>('.tri-gear-btn')?.click()
        handled = true
      } else if (key === 'p') {
        closeOpenModals()
        root.querySelector<HTMLElement>('.tri-pace-btn')?.click()
        handled = true
      } else if (key === 's') {
        root.querySelector<HTMLElement>('.tri-total')?.click()
        handled = true
      } else if (key === 'h') {
        if (window.spaNavigate) {
          window.spaNavigate(new URL('/', window.location.toString()))
        } else {
          window.location.href = '/'
        }
        handled = true
      } else if (key === 'f') {
        root.querySelector<HTMLElement>('.tri-fuel-btn')?.click()
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

window.quartzTriathlon = {
  dayCard: async (date, detailPath, extras) => {
    const data = await loadDetailPayload(detailPath)
    if (!data) return null
    return buildDayCard(date, data, extras ?? {})
  },
}

document.addEventListener('nav', () => {
  const embedCleanup = setupDayEmbeds()
  if (embedCleanup) window.addCleanup?.(embedCleanup)
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
  const trainingCleanup = setupTraining(root)
  if (trainingCleanup) window.addCleanup?.(trainingCleanup)
  const mapCleanup = setupMap(root)
  if (mapCleanup) window.addCleanup?.(mapCleanup)
  const glossCleanup = setupGloss(root)
  if (glossCleanup) window.addCleanup?.(glossCleanup)
  const axisCleanup = setupAxisLabels(root)
  if (axisCleanup) window.addCleanup?.(axisCleanup)
  const shortcutsCleanup = setupShortcuts(root)
  if (shortcutsCleanup) window.addCleanup?.(shortcutsCleanup)
  const hashDate = /^#(\d{4}-\d{2}-\d{2})$/.exec(window.location.hash)?.[1]
  if (hashDate)
    window.dispatchEvent(new CustomEvent('tri:focus-day', { detail: { date: hashDate } }))
})
