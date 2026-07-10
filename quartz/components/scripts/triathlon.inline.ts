import type { RoughAnnotation } from 'rough-notation/lib/model'
import katex from 'katex'
import { annotate } from 'rough-notation'
import type {
  ActivitySummary,
  Analytics,
  BodyBlock,
  DailyPoint,
  RaceLeg,
  SportTrend,
  Vo2LabProfile,
  Vo2LabProfileSample,
  Vo2LabProfileStats,
  Vo2LabRecord,
  Vo2LabTargetStep,
} from '../../plugins/stores/analytics'
import type { OuraDayDetail, OuraSeries } from '../../plugins/stores/oura'
import {
  type ActivityHealth,
  type ActivityKind,
  type StravaMapPoint,
  type PowerCurvePoint,
  type Sport,
  type StravaActivityDetail,
  type StravaZones,
  type SwimTrendPoint,
} from '../../plugins/stores/strava'
import {
  CALC_ANCHOR_PREFIX,
  type CalcShare,
  computeTriathlonCalcTimes,
  decodeCalcShare,
  deriveZoneBands,
  encodeCalcShare,
  formatDurationClock,
  parseClockSeconds,
  type ProjectedLeg,
  projectZoneTimes,
  type SportThresholdVel,
  solveTriathlonCalcLeg,
  solveTriathlonCalcTarget,
  type TriathlonCalcInput,
  type TriathlonCalcLeg,
  type Vo2LabZones,
  type ZoneBand,
} from '../../util/triathlon-calculator'
import {
  axisFrame,
  buildActivity as buildActivityNode,
  buildCyclingBestEfforts as buildCyclingBestEffortsNode,
  buildHrZones as hrZonesNode,
  buildPowerCurve as powerCurveNode,
  buildPowerHist as powerHistNode,
  buildPowerZones as powerZonesNode,
  zoneClock,
  buildDayCard as buildDayCardNode,
  buildElevation as buildElevationNode,
  buildIcon as buildIconNode,
  buildLayers as buildLayersNode,
  buildPool as buildPoolNode,
  buildRecovery as buildRecoveryNode,
  clock,
  dist,
  distCombined,
  decodePowerCurve,
  dur,
  formatAltitude,
  gradeAt,
  isImperialUnit,
  KM_TO_MI,
  M_TO_FT,
  powerCurveFraction,
  powerCurveHoverAt,
  rate,
  routeStreamFlags,
  scrubDist,
  setDistanceUnit,
  shortDate,
  statRow as statRowNode,
  swimTrendHoverAt,
  buildTrace as buildTraceNode,
  zoneDuo as zoneDuoNode,
  type AxisXTick,
  type DayCardExtras,
  type DetailCtx,
  type SwimTrendChartPoint,
  type SwimTrendMode,
  type TriNodeFactory,
} from '../../util/triathlon-card'
import {
  applyTriLocale,
  glossFor,
  glossKeys,
  initTriLocale,
  powerCurveReferenceLabel,
  swimActivityComparisonText,
  swimActivityDistanceText,
  swimActivityDisplayValue,
  swimActivityHeaderValue,
  swimActivityPointText,
  swimActivityValueText,
  tl,
  triLocale,
} from '../../util/triathlon-i18n'

const applyI18n = (root: HTMLElement): void => {
  for (const node of root.querySelectorAll<HTMLElement>('[data-i18n]')) {
    const key = node.dataset.i18n
    if (key) node.textContent = tl(key)
  }
  for (const node of root.querySelectorAll<HTMLElement>('[data-i18n-aria-label]')) {
    const key = node.dataset.i18nAriaLabel
    if (key) node.setAttribute('aria-label', tl(key))
  }
}
import {
  type PaceDayState,
  type PaceLegSpec,
  type PaceSport,
  isPaceSport,
} from '../../util/pace-features'
import { PaceForecaster, Z80 } from '../../util/pace-forecast'
import { applyMonochromeMapPalette, loadMapbox } from './mapbox-client'

export {}

type DetailPayload = {
  details: Record<string, StravaActivityDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
  powerCurveYearRef?: PowerCurvePoint[]
  powerCurveYear?: number | null
  ftp?: number | null
  goalFtp?: number | null
  vt1Hr?: number | null
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
let DETAIL_CURVE_YEAR_REF: PowerCurvePoint[] = []
let DETAIL_CURVE_YEAR: number | null = null
let DETAIL_FTP: number | null = null
let DETAIL_GOAL_FTP: number | null = null
let DETAIL_VT1: number | null = null
let DETAIL_PAYLOAD: Promise<DetailPayload | null> | null = null

const loadDetailPayload = (path: string): Promise<DetailPayload | null> => {
  DETAIL_PAYLOAD ??= fetch(path)
    .then(res => res.json())
    .then((data: DetailPayload) => {
      DETAIL_ZONES = data.zones ?? null
      DETAIL_CURVE_REF = data.powerCurveRef ?? []
      DETAIL_CURVE_YEAR_REF = data.powerCurveYearRef ?? []
      DETAIL_CURVE_YEAR = data.powerCurveYear ?? null
      DETAIL_FTP = data.ftp ?? null
      DETAIL_GOAL_FTP = data.goalFtp ?? null
      DETAIL_VT1 = data.vt1Hr ?? null
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
  tick: (value: number) => string,
): HTMLElement => buildTraceNode(domF, d, pick, title, cap, tick) as HTMLElement

const zoneDuo = (a: HTMLElement | null, b: HTMLElement | null): HTMLElement | null =>
  zoneDuoNode(domF, a, b) as HTMLElement | null

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
    const fraction = Math.min(1, Math.max(0, (clientX - r.left) / r.width))
    const targetD = fraction * maxD
    let index = 0
    let best = Infinity
    for (let k = 0; k < route.length; k++) {
      const dd = Math.abs(route[k].d - targetD)
      if (dd < best) {
        best = dd
        index = k
      }
    }
    return index
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

const clientCtx = (): DetailCtx => ({
  zones: DETAIL_ZONES,
  curveRef: DETAIL_CURVE_REF,
  curveYearRef: DETAIL_CURVE_YEAR_REF,
  curveYear: DETAIL_CURVE_YEAR,
  ftp: DETAIL_FTP,
  goalFtp: DETAIL_GOAL_FTP,
  vt1: DETAIL_VT1,
})

const buildHrZones = (d: StravaActivityDetail): HTMLElement | null => {
  const n = hrZonesNode(domF, d, clientCtx())
  if (n) applyI18n(n as HTMLElement)
  return n as HTMLElement | null
}

const buildPowerZones = (d: StravaActivityDetail): HTMLElement | null => {
  const n = powerZonesNode(domF, d, clientCtx())
  if (n) applyI18n(n as HTMLElement)
  return n as HTMLElement | null
}

const buildPowerHist = (d: StravaActivityDetail): HTMLElement | null => {
  const n = powerHistNode(domF, d)
  if (n) applyI18n(n as HTMLElement)
  return n as HTMLElement | null
}

const buildPowerCurve = (d: StravaActivityDetail): HTMLElement | null => {
  const n = powerCurveNode(domF, d, clientCtx())
  if (n) applyI18n(n as HTMLElement)
  return n as HTMLElement | null
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
  const paceTick = (kmh: number): string => {
    if (kmh <= 0) return '0'
    if (d.sport === 'bike') return `${(kmh * KM_TO_MI).toFixed(0)}mph`
    if (d.sport === 'swim') return clock(3600 / (kmh * 10))
    return clock(3600 / (kmh * KM_TO_MI))
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
        paceTick,
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
        v => `${Math.round(v)}w`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.w} W`,
    extra: () => [zoneDuo(buildPowerCurve(d), buildPowerHist(d)), buildPowerZones(d)],
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
        v => `${Math.round(v)}bpm`,
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
        v => `${Math.round(v)}${cadUnit}`,
      ),
    readout: p => `${scrubDist(p.d, d.sport)} · ${p.cad * cadScale} ${cadUnit}`,
  }
  const elevSpec: MapMetric = {
    label: 'elevation',
    ramp: ELEV_RAMP,
    pick: p => p.alt,
    fmt: formatAltitude,
    profile: () => buildElevation(d),
    readout: (p, i) => {
      const g = Math.round(gradeAt(route, i) * 10) / 10
      return `${scrubDist(p.d, d.sport)} · ${formatAltitude(p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%`
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
  initialMetric?: number
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
  if (d.sport === 'strength' || d.sport === 'treatment' || d.sport === 'yoga') {
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
    const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
    for (const z of [
      zoneDuo(buildHrZones(d), buildPowerZones(d)),
      zoneDuo(buildPowerCurve(d), buildPowerHist(d)),
    ])
      if (z) more.appendChild(z)
    if (bestEfforts) more.appendChild(bestEfforts)
    if (more.childElementCount > 0) wrap.appendChild(more)
    return wrap
  }

  const tablist = el('div', 'tri-map-tablist')
  tablist.setAttribute('role', 'tablist')
  const figs = el('div', 'tri-act-figs tri-map-figs')
  const profileBox = el('div', 'tri-map-profile')
  const zoneBox = el('div', 'tri-act-more')
  const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
  wrap.append(tablist, figs, profileBox, zoneBox)

  let active = Math.min(specs.length - 1, Math.max(0, opts?.initialMetric ?? 0))
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
    if (bestEfforts) zoneBox.appendChild(bestEfforts)
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

const renderDetail = (d: StravaActivityDetail, payload?: DetailPayload | null): HTMLElement => {
  const wrap = buildActivityNode(domF, d, false, undefined, payload?.swimTrend ?? []) as HTMLElement
  const surfaces: ScrubSurface[] = []
  const elev = wrap.querySelector<HTMLElement>('.tri-act-figs .tri-elev-wrap')
  if (elev) {
    surfaces.push({
      wrap: elev,
      fmt: (p, i) => {
        const g = Math.round(gradeAt(d.route, i) * 10) / 10
        return (
          `${scrubDist(p.d, d.sport)} · ${formatAltitude(p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%` +
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
        v => `${Math.round(v)}bpm`,
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
        v => `${Math.round(v)}w`,
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
        v => `${Math.round(v)}${cadUnit}`,
      )
      more.appendChild(t)
      surfaces.push({
        wrap: t,
        fmt: p => `${scrubDist(p.d, d.sport)} · ${p.cad * cadScale} ${cadUnit}`,
      })
    }
    const zones = zoneDuo(buildHrZones(d), buildPowerZones(d))
    if (zones) more.appendChild(zones)
    const charts = zoneDuo(buildPowerCurve(d), buildPowerHist(d))
    if (charts) more.appendChild(charts)
    const bestEfforts = more.querySelector<HTMLElement>(':scope > .tri-efforts')
    if (bestEfforts) more.appendChild(bestEfforts)
  }
  if (surfaces.length > 0 && d.route.length >= 2) {
    const routeMarker = wrap.querySelector<SVGElement>('.tri-route-cursor')
    linkScrub(wrap, routeMarker, surfaces, d.route)
  }
  return wrap
}

const setActivityExpanded = (activity: HTMLElement, expanded: boolean): void => {
  activity.classList.toggle('tri-act--expanded', expanded)
  const toggle = activity.querySelector<HTMLButtonElement>(':scope > .tri-act-toggle')
  if (!toggle) return
  toggle.setAttribute('aria-expanded', String(expanded))
  toggle.textContent = expanded ? '− see less' : '+ see more'
}

const onCardToggle = (event: Event): void => {
  const toggle = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('.tri-act-toggle')
  const activity = toggle?.closest<HTMLElement>('.tri-act')
  if (activity) setActivityExpanded(activity, !activity.classList.contains('tri-act--expanded'))
}

const buildDayCard = (
  dateIso: string,
  payload: DetailPayload | null,
  extras: DayCardExtras = {},
): HTMLElement => {
  const card = buildDayCardNode(domF, dateIso, payload, extras, detail =>
    renderDetail(detail, payload),
  ) as HTMLElement
  if (extras.expanded) {
    card
      .querySelectorAll<HTMLElement>('.tri-act')
      .forEach(activity => setActivityExpanded(activity, true))
  }
  card.addEventListener('click', onCardToggle)
  return card
}

const dayExtrasFromDataset = (data: DOMStringMap): DayCardExtras => ({
  location: data.triathlonLoc,
  event: data.triathlonEvent,
  sport: data.triathlonSport as DayCardExtras['sport'],
  expanded: data.triathlonExpanded === '1',
  dateHref: data.triathlonDateHref,
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
    let payload: DetailPayload | null = null
    let pendingSwimMode: { index: number; mode: SwimTrendMode } | null = null
    const setPendingSwimMode = (target: EventTarget | null): void => {
      if (!(target instanceof Element)) return
      const button = target.closest<HTMLButtonElement>('.tri-swim-mode')
      const toggle = button?.closest<HTMLElement>('.tri-swim-mode-toggle')
      const section = toggle?.closest<HTMLElement>('.tri-swim-trends')
      if (!button || !section) return
      const index = Array.from(embed.querySelectorAll<HTMLElement>('.tri-swim-trends')).indexOf(
        section,
      )
      if (index < 0) return
      pendingSwimMode = { index, mode: button.dataset.swimMode === '100m' ? '100m' : 'lengths' }
    }
    const onSwimPointerDown = (event: PointerEvent): void => setPendingSwimMode(event.target)
    const onSwimKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Enter' || event.key === ' ') setPendingSwimMode(event.target)
    }
    const clearPendingSwimMode = (): void => {
      pendingSwimMode = null
    }
    embed.addEventListener('pointerdown', onSwimPointerDown, { passive: true })
    embed.addEventListener('keydown', onSwimKeyDown)
    embed.addEventListener('click', clearPendingSwimMode)
    embed.addEventListener('pointercancel', clearPendingSwimMode)
    teardowns.push(() => {
      embed.removeEventListener('pointerdown', onSwimPointerDown)
      embed.removeEventListener('keydown', onSwimKeyDown)
      embed.removeEventListener('click', clearPendingSwimMode)
      embed.removeEventListener('pointercancel', clearPendingSwimMode)
    })
    const render = (data: DetailPayload) => {
      const swimStates: {
        mode: SwimTrendMode
        focusedMode: SwimTrendMode | null
        charts: { kind: 'pace' | 'stroke'; distanceM: number; active: boolean; focused: boolean }[]
      }[] = Array.from(embed.querySelectorAll<HTMLElement>('.tri-swim-trends'), section => {
        const toggle = section.querySelector<HTMLElement>('.tri-swim-mode-toggle')
        const active = document.activeElement
        const charts: {
          kind: 'pace' | 'stroke'
          distanceM: number
          active: boolean
          focused: boolean
        }[] = []
        for (const chart of section.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
          const distanceM = Number(chart.getAttribute('aria-valuenow'))
          if (!Number.isFinite(distanceM)) continue
          charts.push({
            kind: chart.dataset.swimKind === 'stroke' ? 'stroke' : 'pace',
            distanceM,
            active: chart.closest('.tri-zone')?.classList.contains('tri-chart--hover') ?? false,
            focused: active === chart,
          })
        }
        return {
          mode: toggle?.dataset.swimMode === '100m' ? '100m' : 'lengths',
          focusedMode:
            active instanceof HTMLButtonElement && toggle?.contains(active)
              ? active.dataset.swimMode === '100m'
                ? '100m'
                : 'lengths'
              : null,
          charts,
        }
      })
      if (pendingSwimMode && swimStates[pendingSwimMode.index])
        swimStates[pendingSwimMode.index].mode = pendingSwimMode.mode
      pendingSwimMode = null
      const fresh = buildDayCard(date, data, extras)
      const expanded = Array.from(embed.querySelectorAll('.tri-act'), activity =>
        activity.classList.contains('tri-act--expanded'),
      )
      fresh.querySelectorAll('.tri-act').forEach((activity, index) => {
        if (index < expanded.length) setActivityExpanded(activity as HTMLElement, expanded[index])
      })
      embed.replaceChildren(fresh)
      applyI18n(fresh)
      fresh.querySelectorAll<HTMLElement>('.tri-swim-trends').forEach((section, index) => {
        const state = swimStates[index]
        if (!state) return
        const selected = section.querySelector<HTMLButtonElement>(
          `.tri-swim-mode[data-swim-mode="${state.mode}"]`,
        )
        if (state.mode === '100m') selected?.click()
        if (state.focusedMode)
          section
            .querySelector<HTMLButtonElement>(
              `.tri-swim-mode[data-swim-mode="${state.focusedMode}"]`,
            )
            ?.focus({ preventScroll: true })
        for (const chartState of state.charts) {
          const chart = section.querySelector<SVGSVGElement>(
            `.tri-swim-trend-svg[data-swim-kind="${chartState.kind}"]`,
          )
          if (chart) {
            chart.dataset.swimRestoreDistance = chartState.distanceM.toString()
            chart.dataset.swimRestoreActive = String(chartState.active)
            chart.dispatchEvent(new Event('tri:swim-restore', { bubbles: true }))
            if (chartState.focused) chart.focus({ preventScroll: true })
          }
        }
      })
      window.dispatchEvent(new CustomEvent('tri:locale'))
    }
    const upgrade = () => {
      if (upgraded) return
      upgraded = true
      void loadDetailPayload(detailPath).then(data => {
        if (!live || !embed.isConnected || !data) return
        payload = data
        render(data)
      })
    }
    const onUnit = () => (payload ? render(payload) : upgrade())
    window.addEventListener('tri:unit', onUnit)
    teardowns.push(() => window.removeEventListener('tri:unit', onUnit))
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
    for (const activity of pop.querySelectorAll<HTMLElement>('.tri-act'))
      setActivityExpanded(activity, on)
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
  const pageMode = root.dataset.triView === 'tools' || root.dataset.triView === 'calc'
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
    bikeMph: (Number(inputVal('bike')) || 0) * (isImperialUnit() ? 1 : KM_TO_MI),
    t2Sec: parseClockSeconds(inputVal('t2')),
    runPaceSec: parseClockSeconds(inputVal('run')) / (isImperialUnit() ? 1 : KM_TO_MI),
  })

  const compute = (forceTarget = false): void => {
    const times = computeTriathlonCalcTimes(readCalcInput())
    setResult('swim', times.swimSec)
    setResult('t1', times.t1Sec)
    setResult('bike', times.bikeSec)
    setResult('t2', times.t2Sec)
    setResult('run', times.runSec)
    setResult('total', times.totalSec, forceTarget)
    if (projActive) renderProjection()
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
    setInputVal('bike', bikeToDisp(paces.bikeMph))
    setInputVal('run', runToDisp(paces.runPaceSec))
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
    if (solved.bikeMph != null) setInputVal('bike', bikeToDisp(solved.bikeMph))
    if (solved.runPaceSec != null) setInputVal('run', runToDisp(solved.runPaceSec))
    compute(true)
  }

  let analytics: Analytics | null = null
  let userEdited = false
  const source = calc.querySelector<HTMLElement>('.tri-calc-source')
  const projPanel = calc.querySelector<HTMLElement>('.tri-calc-proj')
  const projZonesWrap = projPanel?.querySelector<HTMLElement>('.tri-calc-proj-zones') ?? null
  const projOut = projPanel?.querySelector<HTMLElement>('.tri-calc-proj-out') ?? null
  const projTab = calc.querySelector<HTMLElement>('.tri-calc-src--proj')
  let projBands: ZoneBand[] = []
  let projZoneIdx = 2
  let projActive = false
  let projModelKey = ''
  let projModel: { mu: number; sigma: number } | null = null
  const paceHuman = (which: 'avg' | 'pred', sport: Sport): number | null => {
    if (!analytics) return null
    const cal = bySport(analytics.calibration.paces, sport)
    const calibrated = which === 'avg' ? cal?.average : cal?.projected
    if (calibrated != null && Number.isFinite(calibrated) && calibrated > 0) return calibrated
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
    sport === 'bike'
      ? (isImperialUnit() ? v * KM_TO_MI : v).toFixed(1)
      : clock(sport === 'run' && isImperialUnit() ? v / KM_TO_MI : v)
  const bikeToDisp = (mph: number): string => (isImperialUnit() ? mph : mph / KM_TO_MI).toFixed(1)
  const runToDisp = (miSec: number): string => clock(isImperialUnit() ? miSec : miSec * KM_TO_MI)
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
    setProjActive(false)
    compute()
  }

  const readThresholds = (): SportThresholdVel | null => {
    if (!analytics) return null
    const get = (s: Sport): number => bySport(analytics!.thresholds, s)?.vThr ?? 0
    return { swim: get('swim'), bike: get('bike'), run: get('run') }
  }
  const latestLab = (): Vo2LabZones | null => {
    const labs = analytics?.tests?.vo2max
    const r = labs && labs.length ? labs[labs.length - 1] : null
    return r ? { zonesKmh: r.zonesKmh, zonesHr: r.zonesHr, maxKmh: r.maxKmh, hrMax: r.hrMax } : null
  }
  const zoneHrLabel = (band: ZoneBand): string =>
    band.hrLo != null ? `${band.hrLo}${band.hrHi != null ? `–${band.hrHi}` : '+'}` : ''
  const legBand = (sport: Sport, leg: ProjectedLeg): { pace: string; split: string } => {
    const split = `${formatDurationClock(leg.fastSec)}–${formatDurationClock(leg.slowSec)}`
    if (sport === 'swim')
      return { pace: `${clock(360 / leg.vMaxKmh)}–${clock(360 / leg.vMinKmh)}`, split }
    if (sport === 'bike') {
      const lo = isImperialUnit() ? leg.vMinKmh * KM_TO_MI : leg.vMinKmh
      const hi = isImperialUnit() ? leg.vMaxKmh * KM_TO_MI : leg.vMaxKmh
      return { pace: `${lo.toFixed(1)}–${hi.toFixed(1)}`, split }
    }
    const fast = 3600 / leg.vMaxKmh
    const slow = 3600 / leg.vMinKmh
    const conv = (s: number): number => (isImperialUnit() ? s / KM_TO_MI : s)
    return { pace: `${clock(conv(fast))}–${clock(conv(slow))}`, split }
  }
  const projUnit = (sport: Sport): string =>
    sport === 'swim'
      ? '/100m'
      : sport === 'bike'
        ? isImperialUnit()
          ? 'mph'
          : 'km/h'
        : isImperialUnit()
          ? '/mi'
          : '/km'
  const buildZoneSelector = (): void => {
    if (!projZonesWrap) return
    projZonesWrap.replaceChildren()
    for (const band of projBands) {
      const on = band.index === projZoneIdx
      const btn = el('button', `tri-calc-zone${on ? ' tri-calc-zone--on' : ''}`, `Z${band.index}`, {
        type: 'button',
        role: 'tab',
        'data-zone': String(band.index),
        'aria-selected': String(on),
      })
      const hr = zoneHrLabel(band)
      btn.title = hr ? `${tl(band.key)} · ${hr} bpm` : tl(band.key)
      projZonesWrap.appendChild(btn)
    }
  }
  const renderProjection = (): void => {
    if (!projOut) return
    const band = projBands.find(b => b.index === projZoneIdx) ?? projBands[0]
    const thr = readThresholds()
    const input = readCalcInput()
    const proj = band && thr ? projectZoneTimes(input, band, thr) : null
    if (!band || !proj) {
      projOut.replaceChildren(el('div', 'tri-ana-empty', tl('no vo2 test logged')))
      return
    }
    const ifPct = `${Math.round(proj.ifMin * 100)}–${Math.round(proj.ifMax * 100)}%`
    const frag = document.createDocumentFragment()

    const cap = el('div', 'tri-calc-proj-cap')
    cap.appendChild(el('span', 'tri-calc-proj-cap-z', tl(band.key)))
    const hr = zoneHrLabel(band)
    if (hr) cap.appendChild(el('span', 'tri-calc-proj-cap-k', `HR ${hr}`))
    cap.appendChild(el('span', 'tri-calc-proj-cap-k', `${ifPct} ${tl('threshold')}`))
    frag.appendChild(cap)

    const table = el('table', 'tri-calc-proj-io')
    const tbody = el('tbody')
    const legs: [Sport, string, ProjectedLeg][] = [
      ['swim', tl('swim'), proj.swim],
      ['bike', tl('bike'), proj.bike],
      ['run', tl('run'), proj.run],
    ]
    for (const [sport, label, leg] of legs) {
      const b = legBand(sport, leg)
      const tr = el('tr', 'tri-calc-proj-row')
      tr.append(
        el('th', 'tri-calc-proj-k', label),
        el('td', 'tri-calc-proj-pace', b.pace),
        el('td', 'tri-calc-proj-u', projUnit(sport)),
        el('td', 'tri-calc-proj-split', b.split),
      )
      tbody.appendChild(tr)
    }
    const finishTr = el('tr', 'tri-calc-proj-row tri-calc-proj-finish')
    finishTr.append(
      el('th', 'tri-calc-proj-k', tl('finish')),
      el('td', 'tri-calc-proj-pace', ifPct),
      el('td', 'tri-calc-proj-u', ''),
      el(
        'td',
        'tri-calc-proj-split',
        `${formatDurationClock(proj.fastSec)}–${formatDurationClock(proj.slowSec)}`,
      ),
    )
    tbody.appendChild(finishTr)
    table.appendChild(tbody)
    frag.appendChild(table)

    const currentSec = computeTriathlonCalcTimes(input).totalSec
    const deltaSec = (proj.fastSec + proj.slowSec) / 2 - currentSec
    const delta = el('div', 'tri-calc-proj-delta')
    const deltaD = el(
      'span',
      `tri-calc-proj-delta-d${deltaSec < 0 ? ' tri-calc-proj-delta-d--fast' : ''}`,
    )
    setMath(
      deltaD,
      `$\\Delta$ $${deltaSec >= 0 ? '+' : '-'}$${formatDurationClock(Math.abs(deltaSec))}`,
    )
    delta.append(
      el('span', 'tri-calc-proj-delta-k', tl('vs current')),
      el('span', 'tri-calc-proj-delta-v', formatDurationClock(currentSec)),
      deltaD,
    )
    frag.appendChild(delta)

    if (projModel && projModelKey === raceKey(input)) {
      const p =
        normCdf((proj.slowSec - projModel.mu) / projModel.sigma) -
        normCdf((proj.fastSec - projModel.mu) / projModel.sigma)
      const likely = el('div', 'tri-calc-proj-likely', undefined, {
        title: 'model probability the actual finish lands in this projected range',
      })
      likely.append(
        el('span', 'tri-calc-proj-likely-k', 'model'),
        el('span', 'tri-calc-proj-likely-v', `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`),
      )
      frag.appendChild(likely)
    } else {
      void ensureProjModel(input)
    }

    projOut.replaceChildren(frag)
  }
  const raceKey = (i: TriathlonCalcInput): string => `${i.swimKm}-${i.bikeKm}-${i.runKm}`
  const ensureProjModel = async (i: TriathlonCalcInput): Promise<void> => {
    const key = raceKey(i)
    if (key === projModelKey && projModel) return
    const f = paceForecaster
    if (!f?.ready) return
    const legs: PaceLegSpec[] = [
      { sport: 'swim', distanceKm: i.swimKm, elevationM: 0, tempC: null, windKph: null },
      { sport: 'bike', distanceKm: i.bikeKm, elevationM: 0, tempC: null, windKph: null },
      { sport: 'run', distanceKm: i.runKm, elevationM: 0, tempC: null, windKph: null },
    ]
    const fin = await f.forecastFinish(legs, i.t1Sec + i.t2Sec)
    if (!fin || fin.slowSec <= fin.fastSec) return
    projModel = { mu: fin.midSec, sigma: (fin.slowSec - fin.fastSec) / (2 * 1.2816) }
    projModelKey = key
    if (projActive) renderProjection()
  }
  const pinModalTop = (): void => {
    if (pageMode) return
    const top = Math.round(calc.getBoundingClientRect().top)
    calc.style.transition = 'none'
    calc.style.top = `${top}px`
    calc.style.transform = 'translateX(-50%) scale(1)'
    void calc.offsetHeight
    calc.style.transition = ''
  }
  const unpinModalTop = (): void => {
    if (pageMode) return
    calc.style.top = ''
    calc.style.transform = ''
    calc.style.transition = ''
  }
  const setProjActive = (on: boolean): void => {
    projActive = on
    if (!projPanel) return
    if (on && projPanel.hidden) pinModalTop()
    projPanel.hidden = !on
  }
  const selectProjection = (): void => {
    for (const b of calc.querySelectorAll<HTMLElement>('.tri-calc-src')) {
      const on = b.dataset.src === 'proj'
      b.classList.toggle('tri-calc-src--on', on)
      b.setAttribute('aria-selected', String(on))
    }
    setProjActive(true)
    renderProjection()
  }

  const open = () => {
    unpinModalTop()
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
    if (src?.dataset.src === 'proj') {
      selectProjection()
      return
    }
    const zoneBtn = targetElement?.closest<HTMLElement>('.tri-calc-zone')
    if (zoneBtn?.dataset.zone) {
      projZoneIdx = Number(zoneBtn.dataset.zone)
      for (const b of projZonesWrap?.querySelectorAll<HTMLElement>('.tri-calc-zone') ?? []) {
        const on = b === zoneBtn
        b.classList.toggle('tri-calc-zone--on', on)
        b.setAttribute('aria-selected', String(on))
      }
      renderProjection()
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
  const bikeUnitCell = calc.querySelector<HTMLElement>('.tri-calc-u[data-u="bike"]')
  const runUnitCell = calc.querySelector<HTMLElement>('.tri-calc-u[data-u="run"]')
  const syncUnitLabels = () => {
    if (bikeUnitCell) bikeUnitCell.textContent = isImperialUnit() ? 'mph' : 'km/h'
    if (runUnitCell) runUnitCell.textContent = isImperialUnit() ? '/mi' : '/km'
  }
  const onUnit = () => {
    const bikeRaw = Number(inputVal('bike')) || 0
    if (bikeRaw > 0)
      setInputVal('bike', (isImperialUnit() ? bikeRaw * KM_TO_MI : bikeRaw / KM_TO_MI).toFixed(1))
    const runRaw = parseClockSeconds(inputVal('run'))
    if (runRaw > 0)
      setInputVal('run', clock(isImperialUnit() ? runRaw / KM_TO_MI : runRaw * KM_TO_MI))
    syncUnitLabels()
    compute()
  }
  window.addEventListener('tri:unit', onUnit)
  syncUnitLabels()
  if (!isImperialUnit()) onUnit()
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
        const lab = latestLab()
        projBands = lab ? deriveZoneBands(lab) : []
        if (projBands.length) {
          if (!projBands.some(b => b.index === projZoneIdx)) projZoneIdx = projBands[0].index
          buildZoneSelector()
          if (projTab) projTab.hidden = false
          if (source) source.hidden = false
        }
      })
      .catch(() => {})

  const copyBtn = calc.querySelector<HTMLElement>('.tri-calc-copy')
  const currentShare = (): CalcShare => {
    const presets = Array.from(calc.querySelectorAll('.tri-calc-preset'))
    const idx = presets.findIndex(p => p.classList.contains('tri-calc-preset--on'))
    const mode: 'a' | 'p' =
      calc.querySelector('.tri-calc-src--on')?.getAttribute('data-src') === 'pred' ? 'p' : 'a'
    const ci = readCalcInput()
    return {
      presetIdx: idx >= 0 ? idx : 1,
      mode,
      unit: isImperialUnit() ? 'i' : 'm',
      swimPaceSec: ci.swimPaceSec,
      t1Sec: ci.t1Sec,
      bikeMph: ci.bikeMph,
      t2Sec: ci.t2Sec,
      runPaceSec: ci.runPaceSec,
    }
  }
  let copyTimer: number | null = null
  const onCopy = () => {
    const text = `![[triathlon#${CALC_ANCHOR_PREFIX}${encodeCalcShare(currentShare())}]]`
    void navigator.clipboard?.writeText(text).then(() => {
      if (!copyBtn) return
      copyBtn.classList.add('check')
      if (copyTimer) clearTimeout(copyTimer)
      copyTimer = window.setTimeout(() => copyBtn.classList.remove('check'), 2000)
    })
  }
  copyBtn?.addEventListener('click', onCopy)

  const onCalcFill = (event: Event): void => {
    const share = (event as CustomEvent).detail?.share as CalcShare | undefined
    if (!share) return
    if ((share.unit === 'i') !== isImperialUnit()) toggleTriUnit()
    const presets = Array.from(calc.querySelectorAll<HTMLElement>('.tri-calc-preset'))
    const preset = presets[share.presetIdx]
    if (preset) {
      calc.dataset.swim = preset.dataset.swim ?? ''
      calc.dataset.bike = preset.dataset.bike ?? ''
      calc.dataset.run = preset.dataset.run ?? ''
      for (const p of presets) p.classList.toggle('tri-calc-preset--on', p === preset)
    }
    const srcKey = share.mode === 'p' ? 'pred' : 'avg'
    for (const b of calc.querySelectorAll<HTMLElement>('.tri-calc-src')) {
      const on = b.dataset.src === srcKey
      b.classList.toggle('tri-calc-src--on', on)
      b.setAttribute('aria-selected', String(on))
    }
    setInputVal('swim', clock(share.swimPaceSec))
    setInputVal('t1', clock(share.t1Sec))
    setInputVal('bike', bikeToDisp(share.bikeMph))
    setInputVal('t2', clock(share.t2Sec))
    setInputVal('run', runToDisp(share.runPaceSec))
    userEdited = true
    compute(true)
    open()
  }
  window.addEventListener('tri:calc-fill', onCalcFill)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    calc.removeEventListener('click', onCalcClick)
    calc.removeEventListener('input', onInput)
    calc.removeEventListener('change', onChange)
    calc.removeEventListener('keydown', onCalcKey)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('tri:unit', onUnit)
    copyBtn?.removeEventListener('click', onCopy)
    window.removeEventListener('tri:calc-fill', onCalcFill)
    if (copyTimer) clearTimeout(copyTimer)
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

const KG_PER_LB = 0.45359237
const weightUnitLabel = (): string => (isImperialUnit() ? 'lb' : 'kg')
const wConv = (kg: number): number => (isImperialUnit() ? kg / KG_PER_LB : kg)
const wNum = (kg: number, kgDp = 1, lbDp = 0): string =>
  wConv(kg).toFixed(isImperialUnit() ? lbDp : kgDp)
const wFmt = (kg: number, kgDp = 1, lbDp = 0): string =>
  `${wNum(kg, kgDp, lbDp)} ${weightUnitLabel()}`
const wSigned = (kg: number, dp: number): string => {
  const v = wConv(kg)
  return `${v > 0 ? '+' : ''}${v.toFixed(dp)}`
}
const weightSwitch = (): HTMLElement => {
  const g = el('div', 'tri-unit-switch', undefined, { role: 'group', 'aria-label': 'weight unit' })
  for (const u of ['kg', 'lb'] as const) {
    const on = (u === 'lb') === isImperialUnit()
    const opt = el('button', on ? 'tri-unit-opt tri-unit-opt--on' : 'tri-unit-opt', u, {
      type: 'button',
      'aria-pressed': String(on),
    })
    opt.addEventListener('click', () => {
      if ((u === 'lb') !== isImperialUnit()) toggleTriUnit()
    })
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
const signedFixed = (n: number, dp: number): string => `${n > 0 ? '+' : ''}${n.toFixed(dp)}`
const hms = (sec: number): string => {
  const t = Math.max(0, Math.round(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`
}
const speedUnitLabel = (): 'mph' | 'km/h' => (isImperialUnit() ? 'mph' : 'km/h')
const speedFromKmh = (kmh: number): number => (isImperialUnit() ? kmh * KM_TO_MI : kmh)
const fmtSpeedKmh = (kmh: number, dp = 1, gap = ' '): string =>
  `${speedFromKmh(kmh).toFixed(dp)}${gap}${speedUnitLabel()}`
type RaceLegSplit = Pick<RaceLeg, 'sport' | 'legKm' | 'splitS'>
const raceLegDistance = (leg: RaceLegSplit): string =>
  leg.sport === 'swim'
    ? `${Math.round(leg.legKm * 1000).toLocaleString('en-US')} m`
    : fmtKm(leg.legKm)
const raceLegPace = (leg: RaceLegSplit): string => {
  if (leg.legKm <= 0 || leg.splitS <= 0) return '—'
  if (leg.sport === 'swim') return `${clock(leg.splitS / (leg.legKm * 10))} /100m`
  if (leg.sport === 'bike') {
    const kmh = leg.legKm / (leg.splitS / 3600)
    return fmtSpeedKmh(kmh)
  }
  const secKm = leg.splitS / leg.legKm
  return isImperialUnit() ? `${clock(secKm / KM_TO_MI)} /mi` : `${clock(secKm)} /km`
}
const raceLegTip = (leg: RaceLegSplit): string =>
  `${tl(leg.sport)} · ${hms(leg.splitS)} · ${raceLegDistance(leg)} · ${raceLegPace(leg)}`
const markGloss = (e: HTMLElement, key: string): HTMLElement => {
  e.dataset.gloss = key
  e.tabIndex = 0
  return e
}

const anaTitle = (text: string, key?: string): HTMLElement => {
  const e = el('div', 'tri-ana-block-title', tl(text))
  if (key) markGloss(e, key)
  return e
}
const bySport = <T extends { sport: Sport }>(arr: T[], sport: Sport): T | undefined =>
  arr.find(x => x.sport === sport)
const thLabel = (th: { paceLabel: string; unit: string; vThr?: number }): string => {
  if (th.unit === 'km/h' && th.vThr != null) return fmtSpeedKmh(th.vThr * 3.6, 0)
  if (th.unit === 's/km' && th.vThr != null && isImperialUnit())
    return `${clock(1609.344 / th.vThr)} /mi`
  return th.unit === 'km/h' ? `${th.paceLabel} km/h` : `${th.paceLabel}${th.unit.slice(1)}`
}
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
      el('span', 'tri-ana-chip', `${tl('ramp')} ${signed(Math.round((r.rampWeek || 0) * 100))}%`),
      'ramp',
    ),
    markGloss(
      el(
        'span',
        'tri-ana-chip',
        r.monotony != null ? `${tl('monotony')} ${r.monotony.toFixed(2)}` : `${tl('monotony')} —`,
      ),
      'monotony',
    ),
    markGloss(
      el(
        'span',
        'tri-ana-chip',
        r.strain != null ? `${tl('strain')} ${Math.round(r.strain)}` : `${tl('strain')} —`,
      ),
      'strain',
    ),
  )
  if (r.acwr == null) {
    block.appendChild(el('div', 'tri-ana-empty', tl('building base — ACWR needs ~4 weeks')))
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

const PMC_PROJ_DAYS = 14
const K42 = 1 - Math.exp(-1 / 42)
const K7 = 1 - Math.exp(-1 / 7)

const buildPmc = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-pmc')
  const daily = data.daily
  const n = daily.length
  if (n < 2) {
    block.appendChild(el('div', 'tri-ana-empty', tl('not enough data')))
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
    head.append(el('span', 'tri-pmc-dot'), el('span', 'tri-pmc-stat-k', tl(label)))
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

  const frame = axisFrame(
    domF,
    s,
    [maxFit, maxFit / 2, 0].map(gv => ({ label: String(Math.round(gv)), vbY: yFit(gv) })),
    PMC_H,
    [
      ...monthTicks(
        daily.map(d => d.date),
        i => x(i),
      ),
      { label: tl('today'), pct: nowX, cls: 'tri-pmc-xt--now' },
      { label: `+${H}d`, pct: 100, cls: 'tri-pmc-xt--end' },
    ],
    false,
  ) as HTMLElement
  const readoutEl = el('div', 'tri-chart-readout')
  frame.querySelector('.tri-cax-stage')?.appendChild(readoutEl)
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
  ctrl.append(el('span', 'tri-pmc-ctrl-k', tl('projected load')), slider, ctrlLab)
  block.appendChild(ctrl)

  const sportSeries: { sp: Sport; get: (d: DailyPoint) => number }[] = [
    { sp: 'swim', get: d => d.swimCtl },
    { sp: 'bike', get: d => d.bikeCtl },
    { sp: 'run', get: d => d.runCtl },
  ]
  const sportCap = el('div', 'tri-elev-cap')
  sportCap.appendChild(el('span', 'tri-ana-k', tl('fitness')))
  for (const { sp, get } of sportSeries) {
    const th = bySport(data.thresholds, sp)
    const stale = th == null ? '—' : th.staleDays === 0 ? 'today' : `${th.staleDays}d ago`
    const leg = el('span', `tri-ana-leg tri-leg-${sp}`)
    leg.append(
      buildIcon(sp),
      el('span', 'tri-ana-k', `${Math.round(get(daily[n - 1]))} · ${stale}`),
    )
    sportCap.appendChild(leg)
  }
  block.appendChild(sportCap)

  const legendRow = (cls: string, name: string, val: string): HTMLElement => {
    const row = el('div', `tri-pmc-leg ${cls}`)
    row.append(
      el('span', 'tri-pmc-dot'),
      el('span', 'tri-pmc-leg-v', val),
      el('span', 'tri-pmc-leg-k', tl(name)),
    )
    return row
  }
  const entryRow = (a: ActivitySummary): HTMLElement => {
    const row = el('div', 'tri-pmc-entry')
    row.append(
      el('span', 'tri-pmc-entry-n', a.name || a.sport),
      el('span', `tri-pmc-entry-s tri-leg-${a.sport}`, tl(a.sport)),
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
        entryList.appendChild(el('div', 'tri-pmc-entry tri-pmc-entry--empty', tl('no activity')))
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
    if (!proj) {
      const d = daily[i]
      readoutEl.append(
        el(
          'span',
          'tri-pmc-leg-load',
          `${tl('swim')} ${Math.round(d.swimCtl)} · ${tl('bike')} ${Math.round(d.bikeCtl)} · ${tl('run')} ${Math.round(d.runCtl)}`,
        ),
        el('span', 'tri-pmc-leg-load', `${tl('training impulse')} ${Math.round(d.load)}`),
      )
      readoutEl.append(entryList)
    }
  }
  const setCtrlLab = (): void => {
    const lp = projSeries[H - 1]
    ctrlLab.textContent = `${futLoad}/day → ${H}d: ${tl('fitness')} ${Math.round(lp.ctl)} · ${tl('form')} ${signed(Math.round(lp.tsb))}`
  }
  let activeIndex = n - 1
  let locked = false
  const focusIndex = (i: number, hover: boolean): void => {
    activeIndex = Math.round(clampN(i, 0, N - 1))
    const cx = x(activeIndex).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    readoutEl.style.left = `${clampN((x(activeIndex) / ANA_W) * 100, 14, 86).toFixed(2)}%`
    renderLegend(activeIndex)
    block.classList.toggle('tri-chart--hover', hover)
  }
  const indexAt = (event: MouseEvent): number => {
    const rect = s.getBoundingClientRect()
    return Math.round(clampN((event.clientX - rect.left) / rect.width, 0, 1) * (N - 1))
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
    if (locked) return
    focusIndex(indexAt(event), true)
  }
  const onLeave = (): void => {
    if (locked) return
    focusIndex(n - 1, false)
  }
  const onClick = (event: MouseEvent): void => {
    const i = indexAt(event)
    if (locked && i === activeIndex) {
      locked = false
      block.classList.remove('tri-chart--locked')
      focusIndex(n - 1, false)
      return
    }
    locked = true
    block.classList.add('tri-chart--locked')
    focusIndex(i, true)
  }
  s.addEventListener('mousemove', onMove)
  s.addEventListener('mouseleave', onLeave)
  s.addEventListener('click', onClick)

  return block
}

type WkKind = 'load' | 'effort'
const WKT_H = 34
const WKT_TOP = 4
const WKT_BOT = WKT_H - 4
const WKT_CHRONIC = 4
const WKT_ALPHA = 1 - Math.exp(-1 / WKT_CHRONIC)
const WKT_Z = 1.28
const WKT_ACTS = 4

const wkVal = (w: Analytics['weekly'][number], kind: WkKind): number =>
  kind === 'load' ? w.load : w.effort

const wkBands = (vals: number[]): ([number, number] | null)[] => {
  const out: ([number, number] | null)[] = []
  let m = 0
  let v = 0
  for (let i = 0; i < vals.length; i++) {
    const s = Math.sqrt(v)
    out.push(i >= WKT_CHRONIC && m > 0 ? [Math.max(0, m - WKT_Z * s), m + WKT_Z * s] : null)
    if (i === 0) {
      m = vals[0]
    } else {
      const d = vals[i] - m
      m += WKT_ALPHA * d
      v = (1 - WKT_ALPHA) * (v + WKT_ALPHA * d * d)
    }
  }
  return out
}

const wkDates = (weekStart: string): string[] =>
  Array.from({ length: 7 }, (_, k) =>
    new Date(Date.parse(`${weekStart}T00:00:00Z`) + k * 86400000).toISOString().slice(0, 10),
  )

const wkDayLetter = (iso: string): string =>
  new Date(`${iso}T00:00:00Z`)
    .toLocaleDateString(triLocale() === 'fr' ? 'fr-CA' : 'en-US', {
      weekday: 'narrow',
      timeZone: 'UTC',
    })
    .toUpperCase()

const renderWkDetail = (block: HTMLElement, data: Analytics, kind: WkKind, i: number): void => {
  const host = block.querySelector<HTMLElement>('.tri-wkdetail')
  const w = data.weekly[i]
  if (!host || !w || host.dataset.week === String(i)) return
  host.dataset.week = String(i)
  const vals = data.weekly.map(x => wkVal(x, kind))
  const band = wkBands(vals)[i]
  const days = wkDates(w.weekStart)
  const head = el('div', 'tri-wkdetail-head')
  head.append(
    el('span', 'tri-wkdetail-num', String(Math.round(vals[i]))),
    el('span', 'tri-wkdetail-range', `${shortDate(days[0])} – ${shortDate(days[6])}`),
  )
  if (band) {
    const state = vals[i] > band[1] ? 'above' : vals[i] < band[0] ? 'below' : 'in'
    head.appendChild(
      el('span', `tri-wkdetail-state tri-wkdetail-state--${state}`, tl(`${state} range`)),
    )
  }
  const byDate = new Map<string, DailyPoint>()
  for (const d of data.daily) byDate.set(d.date, d)
  const dayVals = days.map(date => {
    const d = byDate.get(date)
    return d ? (kind === 'load' ? d.load : d.effort) : 0
  })
  let dayMax = 1
  for (const dv of dayVals) if (dv > dayMax) dayMax = dv
  const grid = el('div', 'tri-wkdetail-days')
  days.forEach((date, k) => {
    const col = el('span', 'tri-wkdetail-day')
    const track = el('span', 'tri-wkdetail-track')
    if (dayVals[k] > 0) {
      const fill = el('span', 'tri-wkdetail-fill')
      fill.style.height = `${Math.max(6, (dayVals[k] / dayMax) * 100).toFixed(1)}%`
      track.appendChild(fill)
    }
    col.append(track, el('span', 'tri-wkdetail-dl', wkDayLetter(date)))
    grid.appendChild(col)
  })
  const stats = mathK(
    'tri-ana-k tri-wkdetail-stats',
    `${w.sessions}$\\times$ $\\cdot$ ${fmtKm(w.km)} $\\cdot$ ${w.hours.toFixed(1)}h`,
  )
  const actsBox = el('div', 'tri-wkdetail-acts')
  const acts = data.activities
    .filter(a => a.date >= days[0] && a.date <= days[6])
    .map(a => ({ a, v: kind === 'load' ? a.load : a.effort }))
    .sort((p, q) => (q.v ?? -1) - (p.v ?? -1))
  for (const { a, v } of acts.slice(0, WKT_ACTS)) {
    const row = el('div', 'tri-wkdetail-act')
    row.append(
      buildIconLeg(a.sport),
      el('span', 'tri-wkdetail-act-name', a.name || a.sport),
      el('span', 'tri-wkdetail-act-t', hms(a.movingTimeS)),
      el('span', 'tri-wkdetail-act-v', v != null && v > 0 ? String(Math.round(v)) : '—'),
    )
    actsBox.appendChild(row)
  }
  if (acts.length > WKT_ACTS)
    actsBox.appendChild(
      el('div', 'tri-wkdetail-act tri-wkdetail-act--more', `+${acts.length - WKT_ACTS}`),
    )
  host.replaceChildren(head, grid, stats, actsBox)
}

const buildWeekTrend = (data: Analytics, kind: WkKind): HTMLElement => {
  const block = el('div', kind === 'load' ? 'tri-ana-weekly' : 'tri-ana-effort')
  block.appendChild(
    kind === 'load' ? anaTitle('weekly load', 'load') : anaTitle('relative effort', 'effort'),
  )
  const wk = data.weekly
  if (kind === 'load' ? !wk.length : !wk.some(w => w.effort > 0)) {
    block.appendChild(
      el('div', 'tri-ana-empty', tl(kind === 'load' ? 'no weeks' : 'no effort logged')),
    )
    return block
  }
  const n = wk.length
  const vals = wk.map(w => wkVal(w, kind))
  const bands = wkBands(vals)
  let mx = 1
  for (const v of vals) if (v > mx) mx = v
  for (const b of bands) if (b && b[1] > mx) mx = b[1]
  const yMax = niceUp(mx)
  const x = (i: number): number => ((i + 0.5) / n) * ANA_W
  const y = (v: number): number => WKT_BOT - (v / yMax) * (WKT_BOT - WKT_TOP)
  const s = svg('svg', {
    class: 'tri-ana-svg tri-wkt-svg',
    viewBox: `0 0 ${ANA_W} ${WKT_H}`,
    preserveAspectRatio: 'none',
  })
  let run: number[] = []
  const flushBand = (): void => {
    if (run.length >= 2) {
      const top = run.map(i => [x(i), y(bands[i]![1])] as [number, number])
      const btm = [...run].reverse().map(i => [x(i), y(bands[i]![0])] as [number, number])
      s.appendChild(svg('path', { d: `${polyD([...top, ...btm])} Z`, class: 'tri-wkt-band' }))
    }
    run = []
  }
  bands.forEach((b, i) => {
    if (b) run.push(i)
    else flushBand()
  })
  flushBand()
  const pred = bands[n - 1]
  if (pred)
    s.appendChild(
      svg('line', {
        x1: x(n - 1).toFixed(2),
        y1: y(pred[1]).toFixed(2),
        x2: x(n - 1).toFixed(2),
        y2: y(pred[0]).toFixed(2),
        class: 'tri-wkt-whisk',
      }),
    )
  s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: WKT_H, class: 'tri-ana-cursor' }))
  vals.forEach((v, i) => {
    const d = `M ${x(i).toFixed(2)} ${y(v).toFixed(2)} l 0.01 0`
    const g = svg('g', {
      class: i === n - 1 ? 'tri-wkt-pt tri-wkt-pt--now' : 'tri-wkt-pt',
      'data-week': i,
    })
    g.appendChild(svg('path', { d, class: 'tri-wkt-halo' }))
    g.appendChild(svg('path', { d, class: 'tri-wkt-o' }))
    if (i !== n - 1) g.appendChild(svg('path', { d, class: 'tri-wkt-i' }))
    s.appendChild(g)
  })
  const frame = axisFrame(
    domF,
    s,
    [yMax, yMax / 2, 0].map(v => ({ label: String(Math.round(v)), vbY: y(v) })),
    WKT_H,
    monthTicks(
      wk.map(w => w.weekStart),
      i => ((i + 0.5) / n) * 100,
    ),
  ) as HTMLElement
  if (pred) {
    const stage = frame.querySelector<HTMLElement>('.tri-cax-stage')
    for (const bound of [pred[1], pred[0]])
      stage?.appendChild(
        el('span', 'tri-wkt-pred', String(Math.round(bound)), {
          style: `top:${((y(bound) / WKT_H) * 100).toFixed(1)}%`,
        }),
      )
  }
  block.appendChild(frame)
  const wrap = el('div', 'tri-wkdetail-wrap')
  wrap.appendChild(el('div', 'tri-wkdetail'))
  block.appendChild(wrap)
  return block
}

const buildWeekly = (data: Analytics): HTMLElement => {
  const block = buildWeekTrend(data, 'load')
  const wk = data.weekly
  if (!wk.length) return block
  let mx = 1
  for (const w of wk) if (w.load > mx) mx = w.load
  const active = wk.filter(w => w.load > 0).length
  const last = wk[wk.length - 1]
  const prev = wk.length >= 2 ? wk[wk.length - 2] : null
  const vol = data.calibration.volume
  const deltaClass =
    vol.deltaLoad > 0 ? 'tri-dir-up' : vol.deltaLoad < 0 ? 'tri-dir-down' : 'tri-dir-flat'
  const cap = el('div', 'tri-elev-cap tri-wk-cap')
  const statRow = el('div', 'tri-wk-cap-row')
  statRow.append(
    el('span', 'tri-ana-k', `${active} ${tl('active wk')}`),
    el('span', 'tri-ana-k', `${tl('peak')} ${Math.round(mx)}/wk`),
    mathK('tri-ana-k', `28d ${fmtKm(vol.currentKm)} $\\cdot$ ${vol.currentHours.toFixed(1)}h`),
  )
  cap.appendChild(statRow)
  const deltaRow = el('div', 'tri-wk-cap-row')
  deltaRow.appendChild(
    mathK(
      `tri-ana-k ${deltaClass}`,
      `$\\Delta$ ${fmtSignedKm(vol.deltaKm)} $\\cdot$ ${signedFixed(vol.deltaHours, 1)}h $\\cdot$ ${signedFixed(vol.deltaLoad, 0)} load`,
    ),
  )
  if (prev)
    deltaRow.appendChild(
      mathK(
        'tri-ana-k',
        `wk $\\Delta$ ${fmtSignedKm(last.km - prev.km)} $\\cdot$ ${signedFixed(last.hours - prev.hours, 1)}h`,
      ),
    )
  cap.appendChild(deltaRow)
  const legRow = el('div', 'tri-wk-cap-row tri-wk-cap-legs')
  for (const sport of vol.sports) {
    if (sport.currentKm <= 0 && sport.previousKm <= 0) continue
    const leg = el('span', `tri-ana-leg tri-leg-${sport.sport}`)
    leg.append(
      buildIcon(sport.sport),
      el('span', 'tri-ana-k', `${fmtKm(sport.currentKm)} (${fmtSignedKm(sport.deltaKm)})`),
    )
    legRow.appendChild(leg)
  }
  if (legRow.childElementCount) cap.appendChild(legRow)
  block.appendChild(cap)
  return block
}

const renderLegSegments = (track: HTMLElement, legs: RaceLegSplit[]): void => {
  for (const old of track.querySelectorAll('.tri-rdy-leg')) old.remove()
  const legTotalS = legs.reduce((sum, leg) => sum + Math.max(0, leg.splitS), 0)
  if (legTotalS <= 0) return
  let legOffsetS = 0
  for (const leg of legs) {
    const splitS = Math.max(0, leg.splitS)
    if (splitS <= 0) continue
    const hit = el('button', `tri-rdy-leg tri-rdy-leg-${leg.sport}`, undefined, {
      type: 'button',
      'aria-label': raceLegTip(leg),
      'data-tip': raceLegTip(leg),
    })
    hit.style.left = `${((legOffsetS / legTotalS) * 100).toFixed(2)}%`
    hit.style.width = `${((splitS / legTotalS) * 100).toFixed(2)}%`
    track.appendChild(hit)
    legOffsetS += splitS
  }
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
    renderLegSegments(track, r.legs)
    row.appendChild(track)
    const meta = el('span', 'tri-rdy-meta')
    meta.append(
      markGloss(el('span', `tri-rdy-bind tri-leg-${r.bindingLeg}`, tl(r.bindingLeg)), 'binding'),
      markGloss(el('span', 'tri-rdy-time', hms(r.predictedTotalS)), 'predtime'),
    )
    const gain = r.currentTotalS - r.predictedTotalS
    const showGain = r.projected && Math.abs(gain) >= 1
    meta.appendChild(
      el(
        'span',
        `tri-rdy-delta${showGain ? ` tri-dir-${gain > 0 ? 'up' : 'down'}` : ''}`,
        showGain ? `${gain > 0 ? '−' : '+'}${hms(Math.abs(gain))}` : '',
      ),
    )
    const rangeTxt =
      r.predictedFastS < r.predictedSlowS ? `${hms(r.predictedFastS)}–${hms(r.predictedSlowS)}` : ''
    meta.appendChild(
      markGloss(
        el('span', 'tri-rdy-forecast', rangeTxt, {
          title: tl('trend-projected finish · 80% range · incl. T1+T2'),
        }),
        'predtime',
      ),
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
  a.textContent = tl(method)
  span.append(a, ` · n=${n}`)
  return span
}

const fmtTrendVal = (sport: Sport, v: number): string =>
  sport === 'bike'
    ? fmtSpeedKmh(v, 0)
    : `${clock(sport === 'run' && isImperialUnit() ? v / KM_TO_MI : v)}${sport === 'swim' ? ' /100m' : isImperialUnit() ? ' /mi' : ' /km'}`
const fmtTrendShort = (sport: Sport, v: number): string =>
  sport === 'bike'
    ? String(Math.round(speedFromKmh(v)))
    : clock(sport === 'run' && isImperialUnit() ? v / KM_TO_MI : v)
const fmtKm = (km: number): string =>
  isImperialUnit() ? `${(km * KM_TO_MI).toFixed(1)} mi` : `${km.toFixed(1)} km`
const fmtSignedKm = (km: number): string =>
  isImperialUnit() ? `${signedFixed(km * KM_TO_MI, 1)} mi` : `${signedFixed(km, 1)} km`

type TrendSamples = { centers: number[]; los: number[]; his: number[]; days: number }
const trendSamples = (tr: SportTrend): TrendSamples | null => {
  if (tr.level == null || tr.forecast.length < 1) return null
  const lvl = tr.level
  return {
    centers: [lvl, ...tr.forecast.map(p => p.value)],
    los: [lvl, ...tr.forecast.map(p => p.lo)],
    his: [lvl, ...tr.forecast.map(p => p.hi)],
    days: tr.forecast.length,
  }
}
const sampleTrend = (
  s: TrendSamples,
  f: number,
): { value: number; lo: number; hi: number; days: number } => {
  const q = clampN(f, 0, 1) * s.days
  const i0 = Math.floor(q)
  const i1 = Math.min(s.days, i0 + 1)
  const t = q - i0
  const at = (a: number[]): number => a[i0] + (a[i1] - a[i0]) * t
  return { value: at(s.centers), lo: at(s.los), hi: at(s.his), days: q }
}

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
  const samples = trendSamples(tr)
  if (samples) {
    const { centers, los, his, days: M } = samples
    const level = centers[0]
    const weeks = M / 7
    let cLo = Infinity
    let cHi = -Infinity
    for (let i = 0; i <= M; i++) {
      if (centers[i] > cHi) cHi = centers[i]
      if (centers[i] < cLo) cLo = centers[i]
    }
    const scale = Math.max(cHi - cLo, Math.abs(level) * 0.05, 1e-6)
    const coneMax = scale * 0.5
    const halfAt = (i: number): number => Math.min((his[i] - los[i]) / 2, coneMax)
    let lo = cLo
    let hi = cHi
    for (let i = 0; i <= M; i++) {
      const h = halfAt(i)
      if (centers[i] + h > hi) hi = centers[i] + h
      if (centers[i] - h < lo) lo = centers[i] - h
    }
    const pad = scale * 0.3
    lo -= pad
    hi += pad
    const span = Math.max(1e-6, hi - lo)
    const top = 4
    const bot = 24
    const xOf = (i: number): number => (i / M) * ANA_W
    const Y = (v: number): number => {
      const t = (v - lo) / span
      return tr.invert ? top + t * (bot - top) : bot - t * (bot - top)
    }
    const Yc = (v: number): number => clampN(Y(v), 0.5, ANA_H - 0.5)
    const s = svg('svg', {
      class: 'tri-ana-svg tri-trend-svg',
      viewBox: `0 0 ${ANA_W} ${ANA_H}`,
      preserveAspectRatio: 'none',
    })
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-trend-axis' }))
    s.appendChild(svg('line', { x1: 0, y1: ANA_H, x2: ANA_W, y2: ANA_H, class: 'tri-trend-axis' }))
    const hiPts: [number, number][] = []
    const loPts: [number, number][] = []
    const midPts: [number, number][] = []
    for (let i = 0; i <= M; i++) {
      const h = halfAt(i)
      hiPts.push([xOf(i), Yc(centers[i] + h)])
      loPts.push([xOf(i), Yc(centers[i] - h)])
      midPts.push([xOf(i), Yc(centers[i])])
    }
    s.appendChild(
      svg('path', {
        d: `${polyD([...hiPts, ...loPts.reverse()])} Z`,
        class: `tri-trend-band tri-fill-${sport}`,
      }),
    )
    s.appendChild(svg('path', { d: polyD(midPts), class: `tri-trend-proj tri-line-${sport}` }))
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    const track = el('div', 'tri-trend-track')
    const dot = el('span', `tri-trend-dot tri-bg-${sport}`)
    dot.style.left = '0%'
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
    xax.append(el('span', '', tl('now')), el('span', '', `+${Math.round(weeks)} wk`))
    wrap.append(chart, xax, el('div', 'tri-chart-readout'))
  }
  const cal = bySport(data.calibration.paces, sport)
  if (cal) {
    const cap = el('div', 'tri-elev-cap tri-trend-cap')
    if (cal.average != null)
      cap.appendChild(el('span', 'tri-ana-k', `${tl('avg')} ${fmtTrendVal(sport, cal.average)}`))
    if (cal.projected != null)
      cap.appendChild(el('span', 'tri-ana-k', `proj ${fmtTrendVal(sport, cal.projected)}`))
    if (cal.deltaPct != null) {
      const cls =
        cal.direction === 'faster'
          ? 'tri-dir-up'
          : cal.direction === 'slower'
            ? 'tri-dir-down'
            : 'tri-dir-flat'
      cap.appendChild(
        el(
          'span',
          `tri-ana-k ${cls}`,
          `${signedFixed(cal.deltaPct, 1)}% vs prev ${data.calibration.windowDays}d`,
        ),
      )
    }
    if (cal.projectedDeltaPct != null)
      cap.appendChild(
        el(
          'span',
          'tri-ana-k',
          `${signedFixed(cal.projectedDeltaPct, 1)}% next ${data.calibration.projectionDays}d`,
        ),
      )
    cap.appendChild(el('span', 'tri-ana-k', `n ${cal.sampleSize}/${cal.previousSampleSize}`))
    if (cal.latestDate)
      cap.appendChild(el('span', 'tri-ana-k', `${tl('latest')} ${shortDate(cal.latestDate)}`))
    wrap.appendChild(cap)
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
  block.appendChild(buildDistancePredictor())
  return block
}

const buildActions = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-actions')
  block.appendChild(anaTitle('things to improve'))
  const banner = el('div', 'tri-actions-head')
  banner.append(
    el('span', 'tri-actions-weak', tl('weakest')),
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
    block.appendChild(el('div', 'tri-ana-empty', tl('no weight logged')))
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
      const goalBmr = b.goalBmr ?? b.goalLeanBmr
      if (goalBmr != null) {
        if (goalBmr < blo) blo = goalBmr
        if (goalBmr > bhi) bhi = goalBmr
      }
      const brange = Math.max(40, bhi - blo)
      let bLoP = blo - brange * 0.18
      let bHiP = bhi + brange * 0.18
      if (goalBmr != null && b.goalKg != null) {
        const weightGoalY = yPct(b.goalKg)
        const goalY = (1 - (goalBmr - bLoP) / (bHiP - bLoP)) * 100
        if (Math.abs(goalY - weightGoalY) < 10) {
          const targetY = clampN(weightGoalY - 10, 12, 88)
          bLoP = Math.min(bLoP, bHiP - ((bHiP - goalBmr) * 100) / targetY)
        }
      }
      const bY = (v: number): number => (1 - (v - bLoP) / (bHiP - bLoP)) * 100
      if (goalBmr != null)
        s.appendChild(
          svg('line', {
            x1: 0,
            y1: bY(goalBmr),
            x2: 100,
            y2: bY(goalBmr),
            class: 'tri-bodywt-bmr-goal',
          }),
        )
      const firstB = bd[0]
      const firstBx = xPct(firstB.ts)
      if (firstBx > 0)
        s.appendChild(
          svg('line', {
            x1: 0,
            y1: bY(firstB.bmr),
            x2: firstBx.toFixed(2),
            y2: bY(firstB.bmr),
            class: 'tri-bodywt-bmr-missing',
          }),
        )
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
        el('span', 'tri-ana-k', `${wSigned(b.trendKgPerWeek, 2)} ${weightUnitLabel()}/wk`),
        'wtrend',
      ),
    )
  if (b.goalKg != null) {
    const delta =
      b.goalDeltaKg != null ? ` (${wSigned(b.goalDeltaKg, 1)} ${weightUnitLabel()})` : ''
    const eta = b.goalEtaWeeks != null ? ` · $\\approx${b.goalEtaWeeks}$ wk` : ''
    cap.appendChild(
      markGloss(mathK('tri-ana-k', `${tl('goal')} ${wFmt(b.goalKg)}${delta}${eta}`), 'wgoal'),
    )
  }
  if (b.goalBmr != null || b.goalLeanBmr != null) {
    const lean = b.goalLeanBmr != null ? ` · FFM ${b.goalLeanBmr} kcal` : ''
    const text =
      b.goalBmr != null ? `goal BMR ${b.goalBmr} kcal${lean}` : `goal BMR ${b.goalLeanBmr} kcal`
    cap.appendChild(markGloss(el('span', 'tri-ana-k tri-bmr-k', text), 'bmr'))
  }
  if (b.bodyFatPct != null)
    cap.appendChild(
      markGloss(el('span', 'tri-ana-k', `${tl('fat')} ${b.bodyFatPct.toFixed(1)}%`), 'bodyfat'),
    )
  if (b.bmi != null)
    cap.appendChild(markGloss(el('span', 'tri-ana-k', `${tl('bmi')} ${b.bmi.toFixed(1)}`), 'bmi'))
  if (b.latestBmr != null)
    cap.appendChild(markGloss(el('span', 'tri-ana-k tri-bmr-k', `BMR ${b.latestBmr} kcal`), 'bmr'))
  if (b.muscleMassKg != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('muscle')} ${wFmt(b.muscleMassKg, 1, 1)}`))
  if (b.boneMassKg != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('bone')} ${wFmt(b.boneMassKg, 1, 1)}`))
  if (b.bodyWaterPct != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('water')} ${b.bodyWaterPct.toFixed(1)}%`))
  const next = (data.events ?? [])
    .filter(e => e.date >= data.meta.today)
    .sort((a, b2) => a.date.localeCompare(b2.date))[0]
  if (next) cap.appendChild(el('span', 'tri-ana-k', `${next.event ?? tl('race')} · ${next.date}`))
  block.appendChild(cap)
  return block
}

const buildEffort = (data: Analytics): HTMLElement => {
  const block = buildWeekTrend(data, 'effort')
  const all = data.weekly
  if (!all.some(w => w.effort > 0)) return block
  let mx = 1
  for (const w of all) if (w.effort > mx) mx = w.effort
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k', `${tl('peak')} ${Math.round(mx)}`))
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

const missingRuns = <T>(
  rows: T[],
  sel: (r: T) => number | null,
  x: (i: number) => number,
  y: (v: number) => number,
): [number, number][][] => {
  const out: [number, number][][] = []
  let previous: { i: number; value: number } | null = null
  for (const [i, row] of rows.entries()) {
    const value = sel(row)
    if (value == null) continue
    if (previous == null) {
      if (i > 0)
        out.push([
          [x(0), y(value)],
          [x(i), y(value)],
        ])
    } else if (i > previous.i + 1) {
      out.push([
        [x(previous.i), y(previous.value)],
        [x(i), y(value)],
      ])
    }
    previous = { i, value }
  }
  if (previous != null && previous.i < rows.length - 1)
    out.push([
      [x(previous.i), y(previous.value)],
      [x(rows.length - 1), y(previous.value)],
    ])
  return out
}

const buildRecoveryChart = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-recovery')
  block.appendChild(anaTitle('recovery · hrv · rhr', 'hrv'))
  const rec = data.recovery
  if (!rec.series.length) {
    block.appendChild(el('div', 'tri-ana-empty', tl('no recovery data')))
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
    const recLeg = (cls: string, name: string): HTMLElement => {
      const w = el('span', 'tri-rec-legitem')
      w.append(el('span', `tri-rec-legdot ${cls}`), el('span', 'tri-rec-legname', tl(name)))
      return w
    }
    const legend = el('div', 'tri-rec-legend')
    legend.append(recLeg('tri-rec-leg-hrv', 'hrv'), recLeg('tri-rec-leg-rhr', 'rhr'))
    block.appendChild(legend)
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
    for (const seg of missingRuns(rec.series, d => d.hrvZ, x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-hrv tri-rec-missing' }))
    for (const seg of missingRuns(rec.series, d => (d.rhrZ == null ? null : -d.rhrZ), x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-rhr tri-rec-missing' }))
    for (const seg of segRuns(rec.series, d => d.hrvZ, x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-hrv' }))
    for (const seg of segRuns(rec.series, d => (d.rhrZ == null ? null : -d.rhrZ), x, yZ))
      s.appendChild(svg('path', { d: polyD(seg), class: 'tri-rec-rhr' }))
    s.appendChild(svg('line', { x1: ANA_W, y1: 0, x2: ANA_W, y2: ANA_H, class: 'tri-pmc-now' }))
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    block.appendChild(
      axisFrame(
        domF,
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
        false,
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
    cap.appendChild(el('span', 'tri-ana-k', `${tl('baseline')} ${rec.baselineDays}/14`))
  block.appendChild(cap)
  return block
}

let OURA_DETAIL_PATH: string | null = null
let OURA_DETAIL: Promise<Record<string, OuraDayDetail> | null> | null = null
let OURA_OPEN_DATE: string | null = null
let OURA_SCRUB_OFF: (() => void)[] = []

const loadOuraDetail = (): Promise<Record<string, OuraDayDetail> | null> => {
  if (!OURA_DETAIL_PATH) return Promise.resolve(null)
  OURA_DETAIL ??= fetch(OURA_DETAIL_PATH)
    .then(res => res.json() as Promise<Record<string, OuraDayDetail>>)
    .catch(() => null)
  return OURA_DETAIL
}

const flushOuraScrub = (): void => {
  for (const off of OURA_SCRUB_OFF) off()
  OURA_SCRUB_OFF = []
}

const wallMin = (iso: string): number => Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16))
const wallClock = (min: number): string => {
  const m = ((Math.round(min) % 1440) + 1440) % 1440
  return `${Math.floor(m / 60)
    .toString()
    .padStart(2, '0')}:${(m % 60).toString().padStart(2, '0')}`
}

const hourTicks = (
  startIso: string,
  intervalS: number,
  count: number,
  pctOf: (i: number) => number,
): AxisXTick[] => {
  const out: AxisXTick[] = []
  const startS = wallMin(startIso) * 60
  let bucket = Math.floor(startS / 7200)
  for (let i = 1; i < count; i++) {
    const b = Math.floor((startS + i * intervalS) / 7200)
    if (b === bucket) continue
    bucket = b
    out.push({ label: wallClock((b * 7200) / 60), pct: pctOf(i) })
  }
  return out
}

const OURA_STAGE: Record<string, { key: string; lane: number }> = {
  '4': { key: 'awake', lane: 0 },
  '3': { key: 'rem', lane: 1 },
  '2': { key: 'light', lane: 2 },
  '1': { key: 'deep', lane: 3 },
}

const ouraScoreCls = (v: number): string =>
  v < 70 ? 'tri-flag--alert' : v < 85 ? 'tri-flag--watch' : 'tri-flag--info'

const ouraContribGroup = (
  title: string,
  contrib: Record<string, number | null> | null,
): HTMLElement | null => {
  if (!contrib) return null
  const rows = Object.entries(contrib).filter((e): e is [string, number] => e[1] != null)
  if (!rows.length) return null
  const g = el('div', 'tri-sleep-contrib')
  g.appendChild(el('div', 'tri-ana-block-title', tl(title)))
  for (const [key, v] of rows) {
    const row = el('div', 'tri-sleep-contrib-row')
    const bar = el('div', 'tri-sleep-contrib-bar')
    const fill = el(
      'div',
      v >= 70 ? 'tri-sleep-contrib-fill' : 'tri-sleep-contrib-fill tri-sleep-contrib-fill--low',
    )
    fill.style.width = `${clampN(v, 0, 100)}%`
    bar.appendChild(fill)
    row.append(
      el('span', 'tri-sleep-contrib-label', tl(key.replace(/_/g, ' '))),
      bar,
      el('span', 'tri-sleep-contrib-val', String(Math.round(v))),
    )
    g.appendChild(row)
  }
  return g
}

const buildHypnogram = (d: OuraDayDetail): HTMLElement | null => {
  const phase = d.phase5Min
  if (!phase || !phase.length || !d.bedtimeStart) return null
  const len = phase.length
  const H = 16
  const wrap = el('div', 'tri-sleep-chart tri-sleep-hyp')
  wrap.appendChild(el('div', 'tri-ana-block-title', tl('sleep stages')))
  const s = svg('svg', {
    class: 'tri-ana-svg tri-hyp-svg',
    viewBox: `0 0 ${len} ${H}`,
    preserveAspectRatio: 'none',
  })
  let i = 0
  while (i < len) {
    const c = phase[i]
    let j = i + 1
    while (j < len && phase[j] === c) j++
    const st = OURA_STAGE[c]
    if (st)
      s.appendChild(
        svg('rect', {
          x: i,
          y: st.lane * 4 + 0.3,
          width: j - i,
          height: 3.4,
          class: `tri-hyp--${st.key}`,
        }),
      )
    i = j
  }
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  wrap.appendChild(
    axisFrame(
      domF,
      s,
      [
        { label: tl('awake'), vbY: 2 },
        { label: tl('rem'), vbY: 6 },
        { label: tl('light'), vbY: 10 },
        { label: tl('deep'), vbY: 14 },
      ],
      H,
      hourTicks(d.bedtimeStart, 300, len, k => (k / len) * 100),
      false,
    ),
  )
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  const cap = el('div', 'tri-elev-cap')
  const durs: [string, number | null][] = [
    ['deep', d.deepS],
    ['light', d.lightS],
    ['rem', d.remS],
    ['awake', d.awakeS],
  ]
  for (const [name, sec] of durs)
    if (sec != null) cap.appendChild(el('span', 'tri-ana-k', `${tl(name)} ${hms(sec)}`))
  wrap.appendChild(cap)
  const startMin = wallMin(d.bedtimeStart)
  OURA_SCRUB_OFF.push(
    scrubBind(wrap, s, cursor, readout, len, len, k => {
      const st = OURA_STAGE[phase[k]]
      return `${wallClock(startMin + k * 5)} · ${st ? tl(st.key) : '—'}`
    }),
  )
  return wrap
}

const buildOuraSeriesChart = (
  title: string,
  series: OuraSeries | null,
  unit: string,
  strokeCls: string,
): HTMLElement | null => {
  if (!series || series.items.length < 2) return null
  const items = series.items
  const n = items.length
  const vals = items.filter((v): v is number => v != null)
  if (vals.length < 2) return null
  let lo = Infinity
  let hi = -Infinity
  for (const v of vals) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const pad = Math.max((hi - lo) * 0.1, 1)
  const mn = lo - pad
  const mx = hi + pad
  const x = (i: number): number => (i / (n - 1)) * ANA_W
  const y = (v: number): number => ANA_H - 2 - ((v - mn) / (mx - mn)) * (ANA_H - 4)
  const wrap = el('div', 'tri-sleep-chart')
  wrap.appendChild(el('div', 'tri-ana-block-title', tl(title)))
  const s = svg('svg', {
    class: 'tri-ana-svg tri-sleep-line-svg',
    viewBox: `0 0 ${ANA_W} ${ANA_H}`,
    preserveAspectRatio: 'none',
  })
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length
  s.appendChild(svg('line', { x1: 0, y1: y(avg), x2: ANA_W, y2: y(avg), class: 'tri-rec-target' }))
  for (const seg of segRuns(items, v => v, x, y))
    s.appendChild(svg('path', { d: polyD(seg), class: strokeCls }))
  const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' })
  s.appendChild(cursor)
  wrap.appendChild(
    axisFrame(
      domF,
      s,
      [
        { label: String(Math.round(hi)), vbY: y(hi) },
        { label: String(Math.round(lo)), vbY: y(lo) },
      ],
      ANA_H,
      hourTicks(series.startTs, series.intervalS, n, k => (k / (n - 1)) * 100),
    ),
  )
  const readout = el('div', 'tri-chart-readout')
  wrap.appendChild(readout)
  const startMin = wallMin(series.startTs)
  OURA_SCRUB_OFF.push(
    scrubBind(wrap, s, cursor, readout, n, ANA_W, k => {
      const v = items[k]
      const t = wallClock(startMin + (k * series.intervalS) / 60)
      return `${t} · ${v != null ? Math.round(v) : '—'} ${unit}`
    }),
  )
  return wrap
}

const SLEEPLESS_ROCKY_FRAMES = [1, 2, 3, 0].map(
  c => `/static/landing/rocky-monomyth/frames/rocky-monomyth-r5-c${c}.webp`,
)

const buildSleeplessRock = (caption: string): HTMLElement => {
  const wrap = el('div', 'tri-sleep-empty')
  const stage = el('div', 'tri-sleep-rocky', undefined, { 'aria-hidden': 'true' })
  for (const src of SLEEPLESS_ROCKY_FRAMES)
    stage.appendChild(
      el('img', 'tri-rocky-frame', undefined, {
        src,
        alt: '',
        width: '192',
        height: '208',
        decoding: 'async',
        draggable: 'false',
      }),
    )
  wrap.append(stage, el('div', 'tri-ana-empty', caption))
  return wrap
}

const buildOuraDayDetail = (d: OuraDayDetail, close: () => void): HTMLElement => {
  const wrap = el('div', 'tri-sleep-day-body')
  const head = el('div', 'tri-sleep-day-head')
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k tri-sleep-day-date', shortDate(d.date)))
  if (d.bedtimeStart)
    cap.appendChild(
      el('span', 'tri-ana-k', `${tl('bedtime')} ${wallClock(wallMin(d.bedtimeStart))}`),
    )
  if (d.bedtimeEnd)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('wake-up')} ${wallClock(wallMin(d.bedtimeEnd))}`))
  if (d.totalSleepS != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('sleep')} ${hms(d.totalSleepS)}`))
  if (d.efficiency != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('efficiency')} ${Math.round(d.efficiency)}%`))
  if (d.latencyS != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('latency')} ${hms(d.latencyS)}`))
  if (d.lowestHr != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('lowest hr')} ${Math.round(d.lowestHr)}`))
  if (d.avgBreath != null)
    cap.appendChild(el('span', 'tri-ana-k', `${tl('breath')} ${d.avgBreath.toFixed(1)}`))
  if (d.sleepScore != null)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k ${ouraScoreCls(d.sleepScore)}`,
        `${tl('sleep score')} ${Math.round(d.sleepScore)}`,
      ),
    )
  if (d.readinessScore != null)
    cap.appendChild(
      el(
        'span',
        `tri-ana-k ${ouraScoreCls(d.readinessScore)}`,
        `${tl('readiness')} ${Math.round(d.readinessScore)}`,
      ),
    )
  head.appendChild(cap)
  const closeBtn = el('button', 'tri-sleep-day-close', '×', {
    type: 'button',
    'aria-label': tl('Close'),
  })
  closeBtn.addEventListener('click', close)
  head.appendChild(closeBtn)
  wrap.appendChild(head)
  const hyp = buildHypnogram(d)
  const hrv = buildOuraSeriesChart('hrv', d.hrv, 'ms', 'tri-rec-hrv')
  const hr = buildOuraSeriesChart('resting heart rate', d.hr, 'bpm', 'tri-rec-rhr')
  if (!hyp && !hrv && !hr)
    wrap.appendChild(buildSleeplessRock(tl('rock bottom — no sleep recorded')))
  const sleepContrib = ouraContribGroup('sleep score', d.sleepContrib)
  if (sleepContrib) wrap.appendChild(sleepContrib)
  const readyContrib = ouraContribGroup('readiness', d.readinessContrib)
  if (readyContrib) wrap.appendChild(readyContrib)
  if (hyp) wrap.appendChild(hyp)
  if (hrv) wrap.appendChild(hrv)
  if (hr) wrap.appendChild(hr)
  return wrap
}

const buildSleep = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ana-sleep')
  block.appendChild(anaTitle('sleep · debt', 'sleepdebt'))
  const rec = data.recovery
  const view = rec.series
  if (!view.some(d => d.sleepS != null)) {
    block.appendChild(el('div', 'tri-ana-empty', tl('no sleep logged')))
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
    class: 'tri-ana-svg tri-sleep-svg',
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
  const barByDate = new Map<string, SVGElement>()
  view.forEach((d, i) => {
    if (d.sleepS == null) {
      s.appendChild(
        svg('rect', { x: i + 0.35, y: bot - 0.5, width: 0.3, height: 0.5, class: 'tri-seg--rest' }),
      )
      return
    }
    const h = (d.sleepS / maxS) * (H - 2)
    const base = d.sleepS < rec.thresholds.sleepFloorS ? 'tri-seg--short' : 'tri-seg--sleep'
    const bar = svg('rect', {
      x: i + 0.2,
      y: bot - h,
      width: 0.6,
      height: h,
      class: d.date === OURA_OPEN_DATE ? `${base} tri-seg--active` : base,
    })
    barByDate.set(d.date, bar)
    s.appendChild(bar)
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
      domF,
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
      rec.sleepLatestS != null ? `${tl('sleep')} ${hms(rec.sleepLatestS)}` : `${tl('sleep')} —`,
    ),
    el(
      'span',
      'tri-ana-k',
      rec.sleepBaselineS != null ? `${tl('base')} ${hms(rec.sleepBaselineS)}` : `${tl('base')} —`,
    ),
    markGloss(
      el(
        'span',
        `tri-ana-k ${debtCls}`.trim(),
        `${tl('debt')} ${(rec.sleepDebtS / 3600).toFixed(1)} h`,
      ),
      'sleepdebt',
    ),
    el('span', 'tri-ana-k', `${tl('target')} ${hms(rec.sleepTargetS)}`),
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
  const dayWrap = el('div', 'tri-sleep-day')
  const dayInner = el('div', 'tri-sleep-day-inner')
  dayWrap.appendChild(dayInner)
  block.appendChild(dayWrap)
  const setActive = (date: string | null): void => {
    for (const [dt, bar] of barByDate) bar.classList.toggle('tri-seg--active', dt === date)
  }
  const closeDay = (): void => {
    OURA_OPEN_DATE = null
    setActive(null)
    dayWrap.classList.remove('tri-sleep-day--open')
  }
  const showDay = (date: string, onReady?: () => void): void => {
    flushOuraScrub()
    void loadOuraDetail().then(details => {
      if (OURA_OPEN_DATE !== date || !dayInner.isConnected) return
      const d = details?.[date]
      if (!d) {
        dayInner.replaceChildren(buildSleeplessRock(tl('no detail for this night')))
        onReady?.()
        return
      }
      flushOuraScrub()
      dayInner.replaceChildren(buildOuraDayDetail(d, closeDay))
      onReady?.()
    })
  }
  const openDay = (date: string): void => {
    OURA_OPEN_DATE = date
    setActive(date)
    showDay(date, () => {
      requestAnimationFrame(() => dayWrap.classList.add('tri-sleep-day--open'))
    })
  }
  s.addEventListener('click', ev => {
    const r = s.getBoundingClientRect()
    if (r.width <= 0) return
    const frac = clampN((ev.clientX - r.left) / r.width, 0, 1)
    const date = view[Math.min(n - 1, Math.floor(frac * n))]?.date
    if (!date) return
    if (OURA_OPEN_DATE === date) closeDay()
    else openDay(date)
  })
  if (OURA_OPEN_DATE && view.some(d => d.date === OURA_OPEN_DATE)) {
    dayWrap.classList.add('tri-sleep-day--open')
    showDay(OURA_OPEN_DATE)
  }
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
    block.appendChild(el('div', 'tri-ana-empty', tl('no dexa scan logged')))
    return block
  }
  titleRow.appendChild(el('span', 'tri-dexa-date', d.date))
  block.appendChild(titleRow)

  const head = el('div', 'tri-dexa-head')
  const bf = el('div', 'tri-dexa-bf', d.bodyFat.toFixed(1))
  bf.appendChild(el('span', 'tri-dexa-unit', tl('% fat')))
  head.append(bf, el('span', 'tri-dexa-cat', `ACE ${aceBand(d.bodyFat)}`))
  block.appendChild(head)

  const total = d.fatLbs + d.leanLbs + d.bmcLbs
  const seg = (cls: string, lbs: number, label: string): HTMLElement => {
    const s = el('span', `tri-dexa-seg ${cls}`)
    s.style.width = `${(lbs / total) * 100}%`
    s.title = `${tl(label)} ${wFmt(lbs * KG_PER_LB, 1, 1)} · ${((lbs / total) * 100).toFixed(0)}%`
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
      el('span', 'tri-dexa-legname', tl(name)),
      el('span', 'tri-dexa-legval', wFmt(lbs * KG_PER_LB, 1, 1)),
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
      el('span', 'tri-dexa-rlabel', tl(name)),
      rbar,
      el(
        'span',
        'tri-dexa-rval',
        `${wFmt(rtot * KG_PER_LB, 0, 0)} · ${((r.fat / rtot) * 100).toFixed(0)}% fat`,
      ),
    )
    reg.appendChild(row)
  }
  block.appendChild(reg)

  const stats = el('div', 'tri-dexa-stats')
  const stat = (label: string, val: string): void => {
    const c = el('div', 'tri-dexa-stat')
    c.append(el('span', 'tri-dexa-statv', val), el('span', 'tri-dexa-statk', tl(label)))
    stats.appendChild(c)
  }
  stat('lean', wFmt(d.leanLbs * KG_PER_LB, 1, 1))
  if (d.rmr != null) stat('rmr', `${d.rmr} kcal`)
  if (d.bmd != null)
    stat('bmd', `${d.bmd.toFixed(2)}${d.bmdT != null ? ` · T${signed(d.bmdT)}` : ''}`)
  if (d.vatLbs != null) stat('vat', wFmt(d.vatLbs * KG_PER_LB, 2, 2))
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
    block.appendChild(el('div', 'tri-ana-empty', tl('no power or hr data yet')))
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
          `${tl('fitness age')} ${v.fitnessAge} (${signed(v.ageDeltaYears ?? 0)}y)`,
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
    needle.title = `${tl('fitness age')} ${v.fitnessAge}`
    bar.appendChild(needle)
  }
  const chrono = el('span', 'tri-engine-agebar-chrono')
  chrono.style.left = `${pos(v.chronoAge)}%`
  chrono.title = `${tl('age')} ${v.chronoAge}`
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
    const pts = v.trend.map((p, i) => [x(i), y(p.vo2max)] as [number, number])
    const proj = (i: number): boolean =>
      v.trend[i - 1].method === 'bike' || v.trend[i].method === 'bike'
    let from = 0
    for (let i = 1; i < n; i++) {
      if (i === n - 1 || proj(i + 1) !== proj(i)) {
        s.appendChild(
          svg('path', {
            d: polyD(pts.slice(from, i + 1)),
            class: `tri-elev-line tri-line-bike${proj(i) ? ' tri-vo2-proj' : ''}`,
          }),
        )
        from = i
      }
    }
    s.appendChild(svg('line', { x1: 0, y1: 0, x2: 0, y2: ANA_H, class: 'tri-ana-cursor' }))
    const frame = axisFrame(
      domF,
      s,
      [hi, (hi + lo) / 2, lo].map(val => ({ label: val.toFixed(1), vbY: y(val) })),
      ANA_H,
      [
        { label: shortDate(v.trend[0].weekStart), pct: 0, cls: 'tri-cax-xt--first' },
        {
          label: shortDate(v.trend[v.trend.length - 1].weekStart),
          pct: 100,
          cls: 'tri-cax-xt--last',
        },
      ],
    )
    frame.classList.add('tri-engine-vo2-axis')
    block.appendChild(frame)
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
  if (v.trend.some(p => p.method === 'bike'))
    cap.appendChild(
      el('span', 'tri-ana-k tri-vo2-proj-note', tl('dashes = projected from bike power')),
    )
  block.appendChild(cap)
  const lab = data.tests.vo2max[data.tests.vo2max.length - 1]
  if (lab) {
    const labCap = el('div', 'tri-elev-cap tri-vo2-lab')
    if (lab.vt1Hr != null)
      labCap.appendChild(
        el(
          'span',
          'tri-ana-k',
          `vt1 ${lab.vt1Hr}bpm${lab.vt1Kmh != null ? ` · ${fmtSpeedKmh(lab.vt1Kmh, 1, '')}` : ''}`,
        ),
      )
    if (lab.maxKmh != null)
      labCap.appendChild(el('span', 'tri-ana-k', `vmax ${fmtSpeedKmh(lab.maxKmh, 1, '')}`))
    if (lab.ve != null) labCap.appendChild(el('span', 'tri-ana-k', `ve ${lab.ve}l/min`))
    labCap.appendChild(el('span', 'tri-ana-k', `${tl('lab')} ${lab.date}`))
    block.appendChild(labCap)
  }
  return block
}

const VO2P_W = 800
const VO2P_H = 430
const VO2P_L = 82
const VO2P_R = 690
const VO2P_T = 48
const VO2P_B = 360
const VO2P_PW = VO2P_R - VO2P_L
const VO2P_PH = VO2P_B - VO2P_T

type Vo2ProfileMetric = 'vo2' | 'hr' | 've' | 'rf' | 'tv'
type Vo2ProfileChartKind = 'metabolic' | 'ventilation'

const vo2ProfileChartLabel = (kind: Vo2ProfileChartKind): string =>
  kind === 'metabolic' ? 'Metabolic' : 'Ventilation'
const vo2ProfileTargetLegend = (): string => `Target[${speedUnitLabel()}]`
const vo2ProfileTargetTick = (kmh: number): string =>
  isImperialUnit() ? speedFromKmh(kmh).toFixed(kmh === 0 ? 0 : 1) : kmh.toFixed(0)

const vo2ProfileText = (
  cls: string,
  text: string,
  attrs: Record<string, string | number>,
): SVGElement => {
  const t = svg('text', { ...attrs, class: cls })
  t.textContent = text
  return t
}

const vo2ProfileTime = (seconds: number): string => {
  const sec = Math.max(0, Math.round(seconds))
  const min = Math.floor(sec / 60)
  return `${min}:${String(sec - min * 60).padStart(2, '0')}`
}

const vo2ProfileWithTip = <T extends SVGElement>(node: T, heading: string, detail: string): T => {
  node.setAttribute('data-tip-h', heading)
  node.setAttribute('data-tip-d', detail)
  node.setAttribute('aria-label', `${heading} ${detail}`)
  return node
}

const vo2ProfileSampleValue = (
  sample: Vo2LabProfileSample,
  metric: Vo2ProfileMetric,
): number | null => {
  if (metric === 'vo2') return sample.vo2
  if (metric === 'hr') return sample.hr
  if (metric === 've') return sample.ve
  if (metric === 'rf') return sample.rf
  return sample.tv
}

const vo2ProfileX = (profile: Vo2LabProfile, t: number): number =>
  VO2P_L + clampN(t / profile.durationSec, 0, 1) * VO2P_PW

const vo2ProfileY = (value: number, lo: number, hi: number): number =>
  VO2P_B - clampN((value - lo) / (hi - lo), 0, 1) * VO2P_PH

const vo2ProfilePath = (
  profile: Vo2LabProfile,
  metric: Vo2ProfileMetric,
  lo: number,
  hi: number,
): string => {
  const pts: [number, number][] = []
  for (const sample of profile.samples) {
    const v = vo2ProfileSampleValue(sample, metric)
    if (v != null) pts.push([vo2ProfileX(profile, sample.t), vo2ProfileY(v, lo, hi)])
  }
  return polyD(pts)
}

const vo2ProfileTargetPath = (
  profile: Vo2LabProfile,
  steps: Vo2LabTargetStep[],
  area: boolean,
): string => {
  if (!steps.length) return ''
  let d = `M ${vo2ProfileX(profile, 0).toFixed(2)} ${vo2ProfileY(steps[0].kmh, 0, 20).toFixed(2)}`
  for (let i = 1; i < steps.length; i++) {
    const prev = steps[i - 1]
    const curr = steps[i]
    const x = vo2ProfileX(profile, curr.t).toFixed(2)
    d += ` L ${x} ${vo2ProfileY(prev.kmh, 0, 20).toFixed(2)}`
    d += ` L ${x} ${vo2ProfileY(curr.kmh, 0, 20).toFixed(2)}`
  }
  const last = steps[steps.length - 1]
  d += ` L ${vo2ProfileX(profile, profile.durationSec).toFixed(2)} ${vo2ProfileY(last.kmh, 0, 20).toFixed(2)}`
  if (area)
    d += ` L ${vo2ProfileX(profile, profile.durationSec).toFixed(2)} ${VO2P_B} L ${vo2ProfileX(profile, 0).toFixed(2)} ${VO2P_B} Z`
  return d
}

const vo2ProfileStat = (
  label: string,
  stats: Vo2LabProfileStats | null,
  cls: string,
  dp: number,
): HTMLElement | null => {
  if (!stats) return null
  const item = el('span', 'tri-vo2p-stat')
  item.append(
    el('span', `tri-vo2p-stat-name ${cls}`, label),
    el('span', 'tri-vo2p-stat-k', 'Min:'),
    el('span', `tri-vo2p-stat-v ${cls}`, stats.min.toFixed(dp)),
    el('span', 'tri-vo2p-stat-k', 'Max:'),
    el('span', `tri-vo2p-stat-v ${cls}`, stats.max.toFixed(dp)),
    el('span', 'tri-vo2p-stat-k', 'Avg:'),
    el('span', `tri-vo2p-stat-v ${cls}`, stats.avg.toFixed(dp)),
  )
  return item
}

const vo2ProfileLegendItem = (label: string, cls: string, area = false): HTMLElement => {
  const item = el('span', 'tri-vo2p-leg')
  item.append(el('span', `tri-vo2p-leg-mark ${cls}${area ? ' tri-vo2p-leg-mark--area' : ''}`))
  item.appendChild(el('span', 'tri-vo2p-leg-text', label))
  return item
}

const vo2ProfileTicks = (
  s: SVGElement,
  values: number[],
  lo: number,
  hi: number,
  xText: number,
  xTick0: number,
  xTick1: number,
  cls: string,
  anchor: 'start' | 'end',
  dp = 0,
): void => {
  for (const v of values) {
    const y = vo2ProfileY(v, lo, hi)
    s.appendChild(svg('line', { x1: xTick0, y1: y, x2: xTick1, y2: y, class: 'tri-vo2p-tick' }))
    s.appendChild(
      vo2ProfileText('tri-vo2p-ytext ' + cls, v.toFixed(dp), {
        x: xText,
        y: y + 4,
        'text-anchor': anchor,
      }),
    )
  }
}

const vo2ProfileTargetTicks = (s: SVGElement): void => {
  for (const kmh of [0, 5, 10, 15, 20]) {
    const y = vo2ProfileY(kmh, 0, 20)
    s.appendChild(
      svg('line', { x1: VO2P_R, y1: y, x2: VO2P_R + 10, y2: y, class: 'tri-vo2p-tick' }),
    )
    s.appendChild(
      vo2ProfileText('tri-vo2p-ytext tri-vo2p-target', vo2ProfileTargetTick(kmh), {
        x: VO2P_R + 52,
        y: y + 4,
        'text-anchor': 'start',
      }),
    )
  }
}

const vo2ProfilePhase = (s: SVGElement, profile: Vo2LabProfile): void => {
  if (profile.warmupEndSec != null)
    s.appendChild(
      svg('rect', {
        x: vo2ProfileX(profile, 0),
        y: VO2P_T,
        width: vo2ProfileX(profile, profile.warmupEndSec) - VO2P_L,
        height: VO2P_PH,
        class: 'tri-vo2p-phase',
      }),
    )
  if (profile.cooldownStartSec != null)
    s.appendChild(
      svg('rect', {
        x: vo2ProfileX(profile, profile.cooldownStartSec),
        y: VO2P_T,
        width:
          vo2ProfileX(profile, profile.durationSec) -
          vo2ProfileX(profile, profile.cooldownStartSec),
        height: VO2P_PH,
        class: 'tri-vo2p-phase',
      }),
    )
}

const vo2ProfileZoneHit = (
  s: SVGElement,
  profile: Vo2LabProfile,
  start: number,
  end: number,
  label: string,
): void => {
  if (end <= start) return
  s.appendChild(
    vo2ProfileWithTip(
      svg('rect', {
        x: vo2ProfileX(profile, start),
        y: VO2P_T,
        width: vo2ProfileX(profile, end) - vo2ProfileX(profile, start),
        height: VO2P_PH,
        class: 'tri-vo2p-zone-hit',
      }),
      label,
      `${vo2ProfileTime(start)}-${vo2ProfileTime(end)}`,
    ),
  )
}

const vo2ProfileZoneHits = (s: SVGElement, profile: Vo2LabProfile): void => {
  const warmupEnd = profile.warmupEndSec ?? 0
  const cooldownStart = profile.cooldownStartSec ?? profile.durationSec
  if (profile.warmupEndSec != null) vo2ProfileZoneHit(s, profile, 0, warmupEnd, 'Warm-Up')
  vo2ProfileZoneHit(s, profile, warmupEnd, cooldownStart, 'Test')
  if (profile.cooldownStartSec != null)
    vo2ProfileZoneHit(s, profile, cooldownStart, profile.durationSec, 'Cool-Down')
}

const vo2ProfileMarker = (
  s: SVGElement,
  profile: Vo2LabProfile,
  t: number | null,
  label: string,
): void => {
  if (t == null) return
  const x = vo2ProfileX(profile, t)
  s.appendChild(svg('line', { x1: x, y1: VO2P_T, x2: x, y2: VO2P_B, class: 'tri-vo2p-marker' }))
  s.appendChild(
    vo2ProfileWithTip(
      svg('rect', {
        x: x - 5,
        y: VO2P_T,
        width: 10,
        height: VO2P_PH,
        class: 'tri-vo2p-marker-hit',
      }),
      label,
      vo2ProfileTime(t),
    ),
  )
}

const vo2ProfileBaseSvg = (profile: Vo2LabProfile, kind: Vo2ProfileChartKind): SVGElement => {
  const s = svg('svg', {
    class: 'tri-vo2p-svg',
    viewBox: `0 0 ${VO2P_W} ${VO2P_H}`,
    preserveAspectRatio: 'xMidYMid meet',
  })
  s.appendChild(svg('rect', { x: 0, y: 0, width: VO2P_W, height: VO2P_H, class: 'tri-vo2p-bg' }))
  vo2ProfilePhase(s, profile)
  const targetD = vo2ProfileTargetPath(profile, profile.targetKmh, false)
  s.appendChild(
    svg('path', {
      d: vo2ProfileTargetPath(profile, profile.targetKmh, true),
      class: 'tri-vo2p-target-area',
    }),
  )
  for (let t = 0; t <= 720; t += 30) {
    const x = vo2ProfileX(profile, t)
    const major = t % 120 === 0
    s.appendChild(
      svg('line', {
        x1: x,
        y1: VO2P_B,
        x2: x,
        y2: VO2P_B + (major ? 14 : 8),
        class: 'tri-vo2p-xtick',
      }),
    )
    if (major)
      s.appendChild(
        vo2ProfileText('tri-vo2p-xtext', `${Math.floor(t / 60)}:00`, {
          x,
          y: VO2P_B + 34,
          'text-anchor': 'middle',
        }),
      )
  }
  for (const gy of [0, 0.25, 0.5, 0.75, 1]) {
    const y = VO2P_T + gy * VO2P_PH
    s.appendChild(svg('line', { x1: VO2P_L, y1: y, x2: VO2P_R, y2: y, class: 'tri-vo2p-grid' }))
  }
  s.appendChild(svg('path', { d: targetD, class: 'tri-vo2p-target-line' }))
  vo2ProfileZoneHits(s, profile)
  vo2ProfileMarker(s, profile, profile.vt1Sec, 'VT 1')
  vo2ProfileMarker(s, profile, profile.vo2maxSec, 'VO2 max')
  s.appendChild(
    svg('rect', { x: VO2P_L, y: VO2P_T, width: VO2P_PW, height: VO2P_PH, class: 'tri-vo2p-frame' }),
  )
  if (kind === 'metabolic') {
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'vo2', 0, 60),
        class: 'tri-vo2p-line tri-vo2p-line--vo2',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'hr', 60, 200),
        class: 'tri-vo2p-line tri-vo2p-line--hr',
      }),
    )
    vo2ProfileTicks(
      s,
      [60, 80, 100, 120, 140, 160, 180, 200],
      60,
      200,
      VO2P_L - 17,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-red',
      'end',
    )
    vo2ProfileTicks(
      s,
      [0, 10, 20, 30, 40, 50, 60],
      0,
      60,
      VO2P_R + 16,
      VO2P_R,
      VO2P_R + 10,
      'tri-vo2p-blue',
      'start',
    )
  } else {
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 've', 0, 160),
        class: 'tri-vo2p-line tri-vo2p-line--ve',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'rf', 0, 80),
        class: 'tri-vo2p-line tri-vo2p-line--rf',
      }),
    )
    s.appendChild(
      svg('path', {
        d: vo2ProfilePath(profile, 'tv', 0, 4),
        class: 'tri-vo2p-line tri-vo2p-line--tv',
      }),
    )
    vo2ProfileTicks(
      s,
      [0, 20, 40, 60, 80],
      0,
      80,
      VO2P_L - 16,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-cyan',
      'end',
    )
    vo2ProfileTicks(
      s,
      [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
      0,
      4,
      VO2P_L - 50,
      VO2P_L - 10,
      VO2P_L,
      'tri-vo2p-orange',
      'end',
      1,
    )
    vo2ProfileTicks(
      s,
      [0, 20, 40, 60, 80, 100, 120, 140, 160],
      0,
      160,
      VO2P_R + 16,
      VO2P_R,
      VO2P_R + 10,
      'tri-vo2p-green',
      'start',
    )
  }
  vo2ProfileTargetTicks(s)
  return s
}

const buildVo2ProfileChart = (profile: Vo2LabProfile, kind: Vo2ProfileChartKind): HTMLElement => {
  const panel = el('div', 'tri-vo2p-panel')
  const head = el('div', 'tri-vo2p-panel-head')
  head.appendChild(el('span', 'tri-vo2p-panel-heading', vo2ProfileChartLabel(kind)))
  const stats = el('div', 'tri-vo2p-stats')
  const profileStats =
    kind === 'metabolic'
      ? [
          vo2ProfileStat('VO2', profile.stats.vo2, 'tri-vo2p-blue', 1),
          vo2ProfileStat('HR', profile.stats.hr, 'tri-vo2p-red', 0),
        ]
      : [
          vo2ProfileStat('Tv', profile.stats.tv, 'tri-vo2p-orange', 1),
          vo2ProfileStat('Rf', profile.stats.rf, 'tri-vo2p-cyan', 1),
          vo2ProfileStat('Ve', profile.stats.ve, 'tri-vo2p-green', 1),
        ]
  for (const stat of profileStats) if (stat) stats.appendChild(stat)
  const legend = el('div', 'tri-vo2p-legend')
  legend.appendChild(vo2ProfileLegendItem(vo2ProfileTargetLegend(), 'tri-vo2p-target', true))
  if (kind === 'metabolic') {
    legend.appendChild(vo2ProfileLegendItem('VO2[mL/kg/min]', 'tri-vo2p-blue'))
    legend.appendChild(vo2ProfileLegendItem('HR[bpm]', 'tri-vo2p-red'))
  } else {
    legend.appendChild(vo2ProfileLegendItem('Ve[L/min]', 'tri-vo2p-green'))
    legend.appendChild(vo2ProfileLegendItem('Rf[bpm]', 'tri-vo2p-cyan'))
    legend.appendChild(vo2ProfileLegendItem('Tv[L]', 'tri-vo2p-orange'))
  }
  const fig = el('div', 'tri-vo2p-fig')
  fig.append(legend, vo2ProfileBaseSvg(profile, kind))
  panel.append(head, stats, fig)
  return panel
}

const buildVo2Profile = (profile: Vo2LabProfile): HTMLElement => {
  const wrap = el('div', 'tri-vo2p')
  wrap.append(
    buildVo2ProfileChart(profile, 'metabolic'),
    buildVo2ProfileChart(profile, 'ventilation'),
  )
  return wrap
}

const appendVo2TestCap = (block: HTMLElement, r: Vo2LabRecord): void => {
  const cap = el('div', 'tri-elev-cap')
  cap.appendChild(el('span', 'tri-ana-k', `vo2max ${r.value.toFixed(1)} ml/kg/min`))
  if (r.percentile != null) cap.appendChild(el('span', 'tri-ana-k', `p${r.percentile}`))
  if (r.vt1Hr != null)
    cap.appendChild(
      el(
        'span',
        'tri-ana-k',
        `vt1 ${r.vt1Hr}bpm${r.vt1Kmh != null ? ` · ${fmtSpeedKmh(r.vt1Kmh, 1, '')}` : ''}${r.caloriesAtVt1 != null ? ` · ${r.caloriesAtVt1}kcal/h` : ''}`,
      ),
    )
  if (r.ve != null) cap.appendChild(el('span', 'tri-ana-k', `ve ${r.ve}l/min`))
  if (r.hrMax != null) cap.appendChild(el('span', 'tri-ana-k', `hrmax ${r.hrMax}`))
  block.appendChild(cap)
}

const buildVo2test = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-vo2t')
  const titleRow = el('div', 'tri-vo2t-titlerow')
  titleRow.appendChild(anaTitle('vo2 test profile', 'vo2test'))
  const r = data.tests.vo2max[data.tests.vo2max.length - 1]
  if (r?.profile) {
    titleRow.appendChild(el('span', 'tri-vo2t-date', r.date))
    block.appendChild(titleRow)
    block.appendChild(buildVo2Profile(r.profile))
    appendVo2TestCap(block, r)
    return block
  }
  const anchors = r
    ? r.zonesKmh.map((s, i) => ({ s, hr: r.zonesHr[i] })).filter(p => p.hr != null)
    : []
  if (!r || anchors.length < 2 || r.maxKmh == null) {
    block.appendChild(titleRow)
    block.appendChild(el('div', 'tri-ana-empty', tl('no vo2 test logged')))
    return block
  }
  titleRow.appendChild(el('span', 'tri-vo2t-date', r.date))
  block.appendChild(titleRow)

  const maxHr = r.hrAtVo2max ?? r.hrMax ?? anchors[anchors.length - 1].hr
  const bandTop = r.hrMax ?? maxHr
  const curve = [...anchors, { s: r.maxKmh, hr: maxHr }].sort((a, b) => a.s - b.s)
  const speeds = curve.map(p => p.s)
  const hrs = [...curve.map(p => p.hr), bandTop]
  if (r.vt1Hr != null) hrs.push(r.vt1Hr)
  const x0 = Math.min(...speeds) - 0.6
  const x1 = Math.max(...speeds) + 0.6
  const y0 = Math.min(...hrs) - 6
  const y1 = Math.max(...hrs) + 6
  const xP = (s: number): number => ((s - x0) / (x1 - x0)) * 100
  const yP = (hr: number): number => (1 - (hr - y0) / (y1 - y0)) * 100

  const chart = el('div', 'tri-vo2t-chart')
  const yax = el('div', 'tri-vo2t-yax')
  yax.append(el('span', '', String(Math.round(y1))), el('span', '', String(Math.round(y0))))
  const plot = el('div', 'tri-vo2t-plot')
  const s = svg('svg', {
    class: 'tri-vo2t-svg',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'none',
  })

  const zoneNames = ['warm up', 'fat burning', 'endurance', 'vigorous', 'maximal']
  const lows = [...r.zonesHr, bandTop]
  lows.forEach((lo, i) => {
    const hi = i + 1 < lows.length ? lows[i + 1] : Math.ceil(y1)
    const visLo = i === 0 ? y0 : lo
    const last = i + 1 >= lows.length
    const spd = r.zonesKmh[i] != null ? ` · ${fmtSpeedKmh(r.zonesKmh[i])}` : ''
    const kcal = r.zonesKcal[i] != null ? ` · ${r.zonesKcal[i]} kcal/h` : ''
    s.appendChild(
      svg('rect', {
        x: 0,
        y: yP(hi),
        width: 100,
        height: yP(visLo) - yP(hi),
        class: `tri-vo2t-zone tri-vo2t-zone--${i + 1}`,
        'data-tip-h': tl(zoneNames[i] ?? `zone ${i + 1}`),
        'data-tip-d': `${last ? `${lo}+ bpm` : `${lo}–${hi - 1} bpm`}${spd}${kcal}`,
      }),
    )
  })

  for (const gy of [0, 50, 100])
    s.appendChild(svg('line', { x1: 0, y1: gy, x2: 100, y2: gy, class: 'tri-vo2t-grid' }))
  if (r.vt1Kmh != null)
    s.appendChild(
      svg('line', { x1: xP(r.vt1Kmh), y1: 0, x2: xP(r.vt1Kmh), y2: 100, class: 'tri-vo2t-vt' }),
    )
  s.appendChild(
    svg('line', { x1: xP(r.maxKmh), y1: 0, x2: xP(r.maxKmh), y2: 100, class: 'tri-vo2t-vt' }),
  )
  s.appendChild(
    svg('path', { d: polyD(curve.map(p => [xP(p.s), yP(p.hr)])), class: 'tri-vo2t-line' }),
  )
  plot.appendChild(s)
  for (const p of curve) {
    const m = el('span', 'tri-vo2t-pt')
    m.style.left = `${xP(p.s).toFixed(1)}%`
    m.style.top = `${yP(p.hr).toFixed(1)}%`
    plot.appendChild(m)
  }
  const marker = (
    spd: number,
    hr: number,
    mod: string,
    label: string,
    tipH: string,
    tipD: string,
  ): void => {
    const pt = el('span', `tri-vo2t-pt tri-vo2t-pt--${mod}`)
    pt.style.left = `${xP(spd).toFixed(1)}%`
    pt.style.top = `${yP(hr).toFixed(1)}%`
    pt.dataset.tipH = tipH
    pt.dataset.tipD = tipD
    plot.appendChild(pt)
    if (label) {
      const lbl = el('span', `tri-vo2t-lbl tri-vo2t-lbl--${mod}`, label)
      lbl.style.left = `${xP(spd).toFixed(1)}%`
      lbl.style.top = `${yP(hr).toFixed(1)}%`
      plot.appendChild(lbl)
    }
  }
  if (r.vt1Kmh != null && r.vt1Hr != null)
    marker(
      r.vt1Kmh,
      r.vt1Hr,
      'vt',
      'vt1',
      'vt1 · aerobic threshold',
      `${r.vt1Hr} bpm · ${fmtSpeedKmh(r.vt1Kmh)}${r.caloriesAtVt1 != null ? ` · ${r.caloriesAtVt1} kcal/h` : ''}`,
    )
  marker(
    r.maxKmh,
    maxHr,
    'max',
    '',
    'vo2max',
    `${r.value.toFixed(1)} ml/kg/min · ${maxHr} bpm · ${fmtSpeedKmh(r.maxKmh)}`,
  )
  chart.append(yax, plot)
  block.appendChild(chart)
  const xax = el('div', 'tri-vo2t-xax')
  xax.append(el('span', '', fmtSpeedKmh(speeds[0])), el('span', '', fmtSpeedKmh(r.maxKmh)))
  block.appendChild(xax)

  appendVo2TestCap(block, r)
  return block
}

const buildFtpHypothesis = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-ftp')
  block.appendChild(anaTitle('ftp hypothesis', 'ftp'))
  const h = data.engine.ftpHypothesis
  if (!h) {
    block.appendChild(el('div', 'tri-ana-empty', tl('no vo2-derived ftp estimate')))
    return block
  }

  const head = el('div', 'tri-ftp-head')
  const headline = el('div', 'tri-ftp-main')
  headline.append(
    el('span', 'tri-ftp-num', String(h.ftp), { 'data-ftp-out': 'headline' }),
    el('span', 'tri-ftp-unit', ' W'),
  )
  const meta = el('div', 'tri-ftp-meta')
  meta.append(
    el('span', 'tri-ftp-pill', `${h.low}-${h.high} W`, { 'data-ftp-out': 'band' }),
    el('span', 'tri-ftp-pill', `${h.wattsPerKg.toFixed(2)} W/kg`, { 'data-ftp-out': 'wkg' }),
    markGloss(el('span', `tri-ftp-pill tri-conf-${h.conf}`, h.conf), 'conf'),
  )
  head.append(headline, meta)
  block.appendChild(head)

  const methods = el('div', 'tri-ftp-methods')
  const methodRow = (label: string, key: string, value: number, cls: string): HTMLElement => {
    const row = el('div', 'tri-ftp-method')
    row.appendChild(el('span', 'tri-ftp-method-k', tl(label)))
    const track = el('span', 'tri-ftp-method-track')
    const fill = el('span', `tri-ftp-method-fill ${cls}`, undefined, { 'data-ftp-bar': key })
    fill.style.width = `${clampN((value / 350) * 100, 4, 100)}%`
    track.appendChild(fill)
    row.append(
      track,
      el('span', 'tri-ftp-method-v', `${Math.round(value)} W`, { 'data-ftp-out': key }),
    )
    return row
  }
  methods.append(
    methodRow('efficiency chain', 'efficiencyFtp', h.efficiencyFtp, 'tri-ftp-method-fill--eff'),
    methodRow('ACSM inverse', 'acsmFtp', h.acsmFtp, 'tri-ftp-method-fill--acsm'),
  )
  block.appendChild(methods)

  const chain = el('div', 'tri-ftp-chain')
  const chainRow = (label: string, key: string, value: string): HTMLElement => {
    const row = el('div', 'tri-ftp-chain-row')
    row.append(
      el('span', 'tri-ftp-chain-k', tl(label)),
      el('span', 'tri-ftp-chain-v', value, { 'data-ftp-out': key }),
    )
    return row
  }
  chain.append(
    chainRow('absolute vo2max', 'absoluteRunningVo2', `${h.absoluteRunningVo2.toFixed(2)} L/min`),
    chainRow('cycling vo2max', 'cyclingVo2max', `${h.cyclingVo2max.toFixed(2)} L/min`),
    chainRow('vo2 at threshold', 'thresholdVo2', `${h.thresholdVo2.toFixed(2)} L/min`),
    chainRow('metabolic power', 'metabolicWatts', `${Math.round(h.metabolicWatts)} W`),
    chainRow('MAP', 'acsmMapWatts', `${Math.round(h.acsmMapWatts)} W`),
  )
  block.appendChild(chain)

  const controls = el('div', 'tri-ftp-controls')
  const control = (
    key: string,
    label: string,
    min: number,
    max: number,
    step: number,
    value: number,
    unit: string,
    note: string,
    editable = false,
    display: string = String(value),
  ): HTMLElement => {
    const wrap = el(editable ? 'div' : 'label', 'tri-ftp-ctrl')
    const row = el('span', 'tri-ftp-ctrl-row')
    let valEl: HTMLElement
    if (editable) {
      valEl = el('span', 'tri-ftp-ctrl-val tri-ftp-ctrl-val--edit', undefined, {
        'data-ftp-val': key,
      })
      const numIn = document.createElement('input')
      numIn.className = 'tri-ftp-ctrl-num'
      numIn.type = 'text'
      numIn.inputMode = 'decimal'
      numIn.dataset.ftpNum = key
      numIn.value = display
      numIn.setAttribute('aria-label', label)
      valEl.append(numIn, el('span', 'tri-ftp-ctrl-unit', unit, { 'data-ftp-unit': key }))
    } else {
      valEl = el('span', 'tri-ftp-ctrl-val', `${display}${unit}`, { 'data-ftp-val': key })
    }
    row.append(el('span', 'tri-ftp-ctrl-label', tl(label)), valEl)
    const input = document.createElement('input')
    input.className = 'tri-ftp-range'
    input.type = 'range'
    input.dataset.ftpParam = key
    input.dataset.ftpDefault = String(value)
    input.min = String(min)
    input.max = String(max)
    input.step = String(step)
    input.value = String(value)
    input.setAttribute('aria-label', label)
    wrap.append(row, input, el('span', 'tri-ftp-note', note))
    return wrap
  }
  controls.append(
    control(
      'mass',
      'bodyweight',
      60,
      110,
      0.1,
      h.massKg,
      ` ${weightUnitLabel()}`,
      'vo2 report input',
      true,
      wNum(h.massKg, 1, 0),
    ),
    control(
      'vo2',
      'running vo2max',
      30,
      70,
      0.1,
      h.runningVo2max,
      '',
      'measured treadmill value',
      true,
      h.runningVo2max.toFixed(1),
    ),
    control(
      'discount',
      'cross-modal discount',
      0,
      15,
      0.5,
      h.crossModalDiscountPct,
      '%',
      'running to cycling haircut',
    ),
    control('threshold', 'LT2 fraction', 70, 92, 0.5, h.thresholdPct, '%', 'VT2 was not detected'),
    control(
      'efficiency',
      'gross efficiency',
      18,
      25,
      0.5,
      h.grossEfficiencyPct,
      '%',
      'cycling mechanical yield',
    ),
  )
  block.appendChild(controls)
  const foot = el('div', 'tri-ftp-foot')
  foot.append(
    el('span', 'tri-ftp-source', `${tl('lab')} ${h.date}`),
    el('span', 'tri-ftp-source', h.note),
    el('button', 'tri-ftp-reset', 'reset', { type: 'button' }),
  )
  block.appendChild(foot)
  return block
}

type SportAbility = Analytics['engine']['abilities']['sports'][number]
type AbilityAxis = SportAbility['axes'][number]

const radarUnitText = (): string => tl(isImperialUnit() ? 'feet' : 'metres')
const radarDefinition = (key: string): string => tl(key).replace('{unit}', radarUnitText())

const radarAxisLabel = (sports: readonly SportAbility[], index: number): string => {
  const labels = sports
    .map(sport => sport.axes[index]?.label)
    .filter((label): label is string => label != null)
    .map(label => tl(label))
  return [...new Set(labels)].join(' / ')
}

const radarAxisDefinition = (sport: Sport, axis: AbilityAxis): string => {
  switch (axis.key) {
    case 'sprint':
      if (sport === 'bike') return radarDefinition('radar sprint bike definition')
      if (sport === 'run') return radarDefinition('radar sprint run definition')
      return radarDefinition('radar sprint swim definition')
    case 'threshold':
      if (sport === 'bike') return radarDefinition('radar threshold bike definition')
      if (sport === 'run') return radarDefinition('radar threshold run definition')
      return radarDefinition('radar threshold swim definition')
    case 'endurance':
      return radarDefinition('radar endurance definition')
    case 'climb':
      if (sport === 'swim') return radarDefinition('radar pace swim definition')
      if (sport === 'run') return radarDefinition('radar climb run definition')
      return radarDefinition('radar climb bike definition')
    case 'cadence':
      if (sport === 'bike') return radarDefinition('radar cadence bike definition')
      if (sport === 'run') return radarDefinition('radar cadence run definition')
      return radarDefinition('radar stroke rate swim definition')
    case 'recovery':
      return radarDefinition('radar recovery definition')
  }
}

const radarNotationDefinition = (axis: AbilityAxis): string => {
  switch (axis.rawUnit) {
    case 'w/kg':
      return radarDefinition('radar unit wkg definition')
    case 'ctl':
      return radarDefinition('radar unit ctl definition')
    case 'm/h':
      return isImperialUnit()
        ? radarDefinition('radar unit fth definition')
        : radarDefinition('radar unit mh definition')
    case 'm/s':
      return radarDefinition('radar unit mspeed definition')
    case 's/100m':
      return radarDefinition('radar unit s100m definition')
    case 'rpm':
      return radarDefinition('radar unit rpm definition')
    case 'spm':
      return radarDefinition('radar unit spm definition')
    case 'str/min':
      return radarDefinition('radar unit strmin definition')
    case 'readiness':
      return radarDefinition('radar unit readiness definition')
    case 'ms':
      return radarDefinition('radar unit ms definition')
    default:
      return radarDefinition('radar unit default definition')
  }
}

const radarPaceHint = (sport: Sport, axis: AbilityAxis): string | null => {
  if (axis.rawUnit !== 'm/s' || axis.rawValue == null || axis.rawValue <= 0) return null
  if (sport === 'swim') return `${clock(100 / axis.rawValue)} /100m`
  return isImperialUnit()
    ? `${clock(1609.344 / axis.rawValue)} /mi`
    : `${clock(1000 / axis.rawValue)} /km`
}

const buildAbilities = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-engine-radar')
  block.appendChild(anaTitle('abilities', 'radar'))
  const sports = data.engine.abilities.sports.filter(sp => sp.axes.length > 0)
  if (!sports.length || sports.every(sp => sp.axes.every(a => a.score == null))) {
    block.appendChild(el('div', 'tri-ana-empty', tl('not enough data')))
    return block
  }
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const pressed = new Set<Sport>([
    sports.some(sp => sp.sport === 'bike') ? 'bike' : sports[0].sport,
  ])
  let avg = false
  const singleOf = (): SportAbility | null =>
    !avg && pressed.size === 1 ? (sports.find(sp => pressed.has(sp.sport)) ?? null) : null

  const tabs = el('div', 'tri-radar-sports', undefined, {
    role: 'group',
    'aria-label': 'radar sports',
  })
  const avgTab = el('button', 'tri-radar-sport tri-radar-sport--avg', undefined, {
    type: 'button',
    'aria-pressed': 'false',
    'aria-label': tl('average'),
    title: tl('average'),
  })
  avgTab.appendChild(buildLayersNode(domF) as SVGElement)
  avgTab.addEventListener('click', () => toggleAvg())
  tabs.appendChild(avgTab)
  const tabOf = new Map<Sport, HTMLElement>()
  for (const sp of sports) {
    const tab = el('button', `tri-radar-sport tri-radar-sport--${sp.sport}`, undefined, {
      type: 'button',
      'aria-pressed': pressed.has(sp.sport) ? 'true' : 'false',
      'aria-label': sp.sport,
      title: sp.sport,
    })
    tab.appendChild(buildIcon(sp.sport))
    tab.addEventListener('click', () => toggleSport(sp.sport))
    tabOf.set(sp.sport, tab)
    tabs.appendChild(tab)
  }
  block.appendChild(tabs)

  const axesRef = sports[0].axes
  const axesN = axesRef.length
  const cx = 50
  const cy = 50
  const R = 36
  const angle = (i: number): number => ((-90 + (360 / axesN) * i) * Math.PI) / 180
  const pt = (i: number, score: number): [number, number] => {
    const th = angle(i)
    const r = (R * score) / 100
    return [cx + r * Math.cos(th), cy + r * Math.sin(th)]
  }
  const zeros = (): number[] => axesRef.map(() => 0)
  const ringD = (vals: number[]): string => `${polyD(vals.map((v, i) => pt(i, v)))} Z`
  const s = svg('svg', { class: 'tri-radar-svg', viewBox: '0 0 100 100' })
  for (const g of [25, 50, 75, 100])
    s.appendChild(svg('path', { d: ringD(axesRef.map(() => g)), class: 'tri-radar-grid' }))
  axesRef.forEach((_, i) => {
    const [px, py] = pt(i, 100)
    s.appendChild(svg('line', { x1: cx, y1: cy, x2: px, y2: py, class: 'tri-radar-spoke' }))
  })
  type RadarKey = Sport | 'avg'
  const solidOf = new Map<RadarKey, SVGElement>()
  const projPathOf = new Map<RadarKey, SVGElement>()
  const radarKeys: RadarKey[] = [...sports.map(sp => sp.sport), 'avg']
  for (const k of radarKeys) {
    const path = svg('path', { d: ringD(zeros()), class: `tri-radar-fill tri-radar-fill--${k}` })
    s.appendChild(path)
    solidOf.set(k, path)
  }
  for (const k of radarKeys) {
    const path = svg('path', { d: ringD(zeros()), class: `tri-radar-proj tri-radar-proj--${k}` })
    s.appendChild(path)
    projPathOf.set(k, path)
  }
  const dots = axesRef.map((_, i) => {
    const [px, py] = pt(i, 0)
    const dot = svg('circle', { cx: px, cy: py, r: 1.4, class: 'tri-radar-dot' })
    s.appendChild(dot)
    return dot
  })
  const labels = axesRef.map((a, i) => {
    const th = angle(i)
    const label = svg('text', {
      x: cx + (R + 8) * Math.cos(th),
      y: cy + (R + 8) * Math.sin(th) + 1.6,
      'text-anchor': Math.abs(Math.cos(th)) < 0.3 ? 'middle' : Math.cos(th) > 0 ? 'start' : 'end',
      class: 'tri-radar-ax',
    })
    label.textContent = tl(a.label)
    s.appendChild(label)
    return label
  })
  block.appendChild(s)

  const keyCap = el('div', 'tri-radar-key')
  const nowKey = el('span', 'tri-radar-key-item')
  nowKey.append(
    el('span', 'tri-radar-swatch tri-radar-swatch--now'),
    el('span', undefined, tl('now')),
  )
  const projKey = el('span', 'tri-radar-key-item')
  projKey.append(
    el('span', 'tri-radar-swatch tri-radar-swatch--proj'),
    el('span', undefined, `${tl('projected')} +28d`),
  )
  keyCap.append(nowKey, projKey)
  block.appendChild(keyCap)

  const shown = new Map<RadarKey, { solid: number[]; proj: number[] }>()
  for (const k of radarKeys) shown.set(k, { solid: zeros(), proj: zeros() })
  let raf = 0
  const apply = (): void => {
    for (const [k, st] of shown) {
      solidOf.get(k)!.setAttribute('d', ringD(st.solid))
      projPathOf.get(k)!.setAttribute('d', ringD(st.proj))
    }
    const single = singleOf()
    const focus = avg ? shown.get('avg') : single ? shown.get(single.sport) : null
    if (focus) {
      focus.solid.forEach((v, i) => {
        const [px, py] = pt(i, v)
        dots[i].setAttribute('cx', px.toFixed(2))
        dots[i].setAttribute('cy', py.toFixed(2))
      })
    }
  }
  const avgAxis = (pick: (a: AbilityAxis) => number | null | undefined): number[] =>
    axesRef.map((_, i) => {
      const xs = sports.map(sp => pick(sp.axes[i])).filter((v): v is number => v != null)
      return xs.length ? xs.reduce((acc, v) => acc + v, 0) / xs.length : 0
    })
  const targetOf = (sp: SportAbility): { solid: number[]; proj: number[] } =>
    !avg && pressed.has(sp.sport)
      ? { solid: sp.axes.map(a => a.score ?? 0), proj: sp.axes.map(a => a.proj ?? a.score ?? 0) }
      : { solid: zeros(), proj: zeros() }
  const avgTarget = (): { solid: number[]; proj: number[] } =>
    avg
      ? { solid: avgAxis(a => a.score), proj: avgAxis(a => a.proj ?? a.score) }
      : { solid: zeros(), proj: zeros() }
  const morphAll = (animate: boolean): void => {
    window.cancelAnimationFrame(raf)
    const targets = new Map<RadarKey, { solid: number[]; proj: number[] }>(
      sports.map(sp => [sp.sport, targetOf(sp)] as const),
    )
    targets.set('avg', avgTarget())
    if (!animate || reduced) {
      for (const [k, g] of targets) shown.set(k, g)
      apply()
      return
    }
    const from = new Map(
      [...shown].map(([k, v]) => [k, { solid: [...v.solid], proj: [...v.proj] }] as const),
    )
    const t0 = performance.now()
    const tick = (now: number): void => {
      const t = Math.min(1, (now - t0) / 450)
      const e = 1 - (1 - t) ** 3
      for (const [k, g] of targets) {
        const f = from.get(k)!
        shown.set(k, {
          solid: f.solid.map((v, i) => v + (g.solid[i] - v) * e),
          proj: f.proj.map((v, i) => v + (g.proj[i] - v) * e),
        })
      }
      apply()
      if (t < 1) raf = window.requestAnimationFrame(tick)
    }
    raf = window.requestAnimationFrame(tick)
  }
  const applyAxisClasses = (): void => {
    const single = singleOf()
    axesRef.forEach((_, i) => {
      const isNull = avg
        ? sports.every(sp => sp.axes[i].score == null)
        : single != null && single.axes[i].score == null
      dots[i].setAttribute('class', isNull ? 'tri-radar-dot tri-radar-dot--null' : 'tri-radar-dot')
      labels[i].setAttribute('class', isNull ? 'tri-radar-ax tri-radar-ax--null' : 'tri-radar-ax')
    })
  }

  const devBox = el('div', 'tri-dev-slot')
  const legendOn = new Set<string>(['endurance', 'recovery'])
  let revealDev: (() => void) | null = null
  const DEV_KEYS = ['endurance', 'recovery', 'cadence', 'sprint', 'threshold', 'climb'] as const
  type DevSeries = {
    key: string
    cls: string
    dotCls: string
    label: string
    vals: (number | null)[]
    toggle: boolean
  }

  const renderDev = (draw: 'defer' | 'animate' | 'none'): void => {
    revealDev = null
    const single = singleOf()
    const hist = (single ?? sports[0]).history ?? []
    const meanAt = (sp: SportAbility, i: number): number | null => {
      const h = sp.history[i]
      if (!h) return null
      const xs = DEV_KEYS.map(k => h[k]).filter((v): v is number => v != null)
      return xs.length ? xs.reduce((acc, v) => acc + v, 0) / xs.length : null
    }
    let series: DevSeries[]
    if (single) {
      series = DEV_KEYS.filter(k => hist.filter(h => h[k] != null).length >= 2).map(k => ({
        key: k,
        cls: `tri-dev-line--${k}`,
        dotCls: `tri-dev-dot--${k}`,
        label: tl(single.axes.find(axis => axis.key === k)?.label ?? k),
        vals: hist.map(h => h[k]),
        toggle: true,
      }))
    } else if (avg) {
      series = [
        {
          key: 'avg',
          cls: 'tri-line-avg',
          dotCls: 'tri-dev-dot--sp-avg',
          label: tl('average'),
          vals: hist.map((_, i) => {
            const xs = sports.map(sp => meanAt(sp, i)).filter((v): v is number => v != null)
            return xs.length ? Math.round(xs.reduce((acc, v) => acc + v, 0) / xs.length) : null
          }),
          toggle: false,
        },
      ].filter(sr => sr.vals.filter(v => v != null).length >= 2)
    } else {
      series = sports
        .filter(sp => pressed.has(sp.sport))
        .map(sp => ({
          key: sp.sport as string,
          cls: `tri-line-${sp.sport}`,
          dotCls: `tri-dev-dot--sp-${sp.sport}`,
          label: tl(sp.sport),
          vals: sp.history.map((_, i) => {
            const v = meanAt(sp, i)
            return v == null ? null : Math.round(v)
          }),
          toggle: false,
        }))
        .filter(sr => sr.vals.filter(v => v != null).length >= 2)
    }
    if (hist.length < 2 || !series.length) {
      devBox.replaceChildren()
      return
    }
    const dev = el('div', 'tri-dev')
    const W = 100
    const H = 30
    const xAt = (i: number): number => (i / (hist.length - 1)) * W
    const yAt = (v: number): number => H - (v / 100) * H
    const frame = el('div', 'tri-dev-frame')
    const yax = el('div', 'tri-dev-yax')
    for (const gv of [100, 75, 50, 25, 0]) {
      const lab = el('span', 'tri-dev-yt', String(gv))
      lab.style.top = `${100 - gv}%`
      yax.appendChild(lab)
    }
    frame.appendChild(yax)
    const plot = el('div', 'tri-dev-plot')
    const sv = svg('svg', {
      class: 'tri-dev-svg',
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'none',
    })
    for (const gv of [100, 75, 50, 25, 0])
      sv.appendChild(
        svg('line', {
          x1: 0,
          y1: yAt(gv),
          x2: W,
          y2: yAt(gv),
          class: gv === 50 ? 'tri-dev-grid tri-dev-grid--mid' : 'tri-dev-grid',
        }),
      )
    const linesG = svg('g', { class: 'tri-dev-lines' }) as SVGGElement
    sv.appendChild(linesG)
    const paths = new Map<string, SVGElement>()
    for (const sr of series) {
      const d = sr.vals
        .map((v, i) => ({ x: xAt(i), v }))
        .filter((p): p is { x: number; v: number } => p.v != null)
        .map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(2)} ${yAt(p.v).toFixed(2)}`)
        .join(' ')
      const off = sr.toggle && !legendOn.has(sr.key)
      const path = svg('path', {
        d,
        class: `tri-dev-line ${sr.cls}${off ? ' tri-dev-line--off' : ''}`,
      })
      linesG.appendChild(path)
      paths.set(sr.key, path)
    }
    const cursor = svg('line', { x1: 0, y1: 0, x2: 0, y2: H, class: 'tri-chart-cursor' })
    sv.appendChild(cursor)
    plot.appendChild(sv)
    const readoutEl = el('div', 'tri-chart-readout tri-dev-read')
    plot.appendChild(readoutEl)
    const renderRead = (i: number): void => {
      const rows: HTMLElement[] = [el('span', 'tri-dev-read-date', shortDate(hist[i].date))]
      for (const sr of series) {
        const v = sr.vals[i]
        if (paths.get(sr.key)!.classList.contains('tri-dev-line--off') || v == null) continue
        const row = el('div', 'tri-dev-read-row')
        row.append(
          el('span', `tri-dev-dot ${sr.dotCls}`),
          el('span', 'tri-dev-read-k', sr.label),
          el('span', 'tri-dev-read-v', String(v)),
        )
        rows.push(row)
      }
      readoutEl.replaceChildren(...rows)
    }
    const focusIndex = (i: number, hover: boolean): void => {
      const idx = Math.round(clampN(i, 0, hist.length - 1))
      const cxAttr = xAt(idx).toFixed(2)
      cursor.setAttribute('x1', cxAttr)
      cursor.setAttribute('x2', cxAttr)
      readoutEl.style.left = `${clampN((xAt(idx) / W) * 100, 6, 80).toFixed(2)}%`
      readoutEl.style.right = 'auto'
      renderRead(idx)
      dev.classList.toggle('tri-chart--hover', hover)
    }
    const indexAt = (event: MouseEvent): number => {
      const rect = sv.getBoundingClientRect()
      return Math.round(clampN((event.clientX - rect.left) / rect.width, 0, 1) * (hist.length - 1))
    }
    sv.addEventListener('mousemove', (event: MouseEvent) => focusIndex(indexAt(event), true))
    sv.addEventListener('mouseleave', () => dev.classList.remove('tri-chart--hover'))
    const xax = el('div', 'tri-dev-xax')
    for (const t of monthTicks(
      hist.map(h => h.date),
      i => xAt(i),
    )) {
      const lab = el('span', `tri-dev-xt${t.cls ? ' tri-dev-xt--first' : ''}`, t.label)
      lab.style.left = `${t.pct.toFixed(2)}%`
      xax.appendChild(lab)
    }
    frame.appendChild(plot)
    dev.appendChild(frame)
    dev.appendChild(xax)
    const legend = el('div', 'tri-dev-legend')
    for (const sr of series) {
      if (!sr.toggle) {
        const item = el('span', 'tri-dev-leg tri-dev-leg--static')
        item.append(
          el('span', `tri-dev-dot ${sr.dotCls}`),
          el('span', 'tri-dev-leg-name', sr.label),
        )
        legend.appendChild(item)
        continue
      }
      const item = el(
        'button',
        `tri-dev-leg${legendOn.has(sr.key) ? '' : ' tri-dev-leg--off'}`,
        undefined,
        { type: 'button' },
      )
      item.append(el('span', `tri-dev-dot ${sr.dotCls}`), el('span', 'tri-dev-leg-name', sr.label))
      item.addEventListener('click', () => {
        const hidden = paths.get(sr.key)!.classList.toggle('tri-dev-line--off')
        item.classList.toggle('tri-dev-leg--off', hidden)
        if (hidden) legendOn.delete(sr.key)
        else legendOn.add(sr.key)
      })
      legend.appendChild(item)
    }
    dev.appendChild(legend)
    if (draw !== 'none') {
      linesG.style.clipPath = 'inset(0 100% 0 0)'
      const reveal = (): void => {
        linesG.style.clipPath = 'inset(0 0 0 0)'
      }
      if (draw === 'animate')
        window.requestAnimationFrame(() => window.requestAnimationFrame(reveal))
      else revealDev = reveal
    }
    devBox.replaceChildren(dev)
  }
  block.appendChild(devBox)

  let revealed = reduced
  const syncChrome = (): void => {
    block.dataset.sport = avg ? 'avg' : (singleOf()?.sport ?? (pressed.size ? 'all' : 'none'))
    block.dataset.pressed = avg ? sports.map(sp => sp.sport).join(',') : [...pressed].join(',')
    block.classList.toggle('tri-engine-radar--multi', !avg && pressed.size !== 1)
    block.classList.toggle('tri-engine-radar--avg', avg)
    for (const sp of sports)
      block.classList.toggle(
        `tri-engine-radar--${sp.sport}`,
        !avg && pressed.size === 1 && pressed.has(sp.sport),
      )
    avgTab.setAttribute('aria-pressed', avg ? 'true' : 'false')
    for (const [k, tab] of tabOf)
      tab.setAttribute('aria-pressed', !avg && pressed.has(k) ? 'true' : 'false')
    const activeSports =
      avg || pressed.size === 0 ? sports : sports.filter(sport => pressed.has(sport.sport))
    labels.forEach((label, index) => {
      label.textContent = radarAxisLabel(activeSports, index)
    })
    applyAxisClasses()
  }
  const rerender = (): void => {
    revealed = true
    syncChrome()
    morphAll(!reduced)
    renderDev(reduced ? 'none' : 'animate')
  }
  const toggleSport = (sport: Sport): void => {
    if (avg) {
      avg = false
      pressed.clear()
      pressed.add(sport)
    } else if (pressed.has(sport)) pressed.delete(sport)
    else pressed.add(sport)
    rerender()
  }
  const toggleAvg = (): void => {
    avg = !avg
    rerender()
  }

  syncChrome()
  renderDev(reduced ? 'none' : 'defer')
  if (reduced) morphAll(false)
  else {
    apply()
    const io = new IntersectionObserver(
      entries => {
        if (!entries.some(en => en.isIntersecting)) return
        io.disconnect()
        if (revealed) return
        revealed = true
        morphAll(true)
        revealDev?.()
        revealDev = null
      },
      { threshold: 0.15 },
    )
    io.observe(block)
  }

  return block
}

const buildCardio = (data: Analytics): HTMLElement => {
  const block = el('div', 'tri-engine-cardio')
  block.appendChild(anaTitle('cardiovascular health', 'ef'))
  const c = data.engine.cardio
  if (!c.metrics.length || c.metrics.every(m => m.value == null)) {
    block.appendChild(el('div', 'tri-ana-empty', tl('no heart data yet')))
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
    row.appendChild(markGloss(el('span', 'tri-engine-row-k', tl(m.label)), glossOf[m.key] ?? 'ef'))
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
  vo2test: buildVo2test,
  abilities: buildAbilities,
  cardio: buildCardio,
  pmc: buildPmc,
  weekly: buildWeekly,
  effort: buildEffort,
  readiness: buildReadiness,
  trend: buildTrend,
  actions: buildActions,
  ftp: buildFtpHypothesis,
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

  const rec = data.recovery.series
  bind('.tri-ana-recovery', '.tri-rec-svg', rec.length, ANA_W, i => {
    const d = rec[i]
    const z = d.hrvZ != null ? ` $${signed(d.hrvZ)}\\sigma$` : ''
    return `${d.date} · HRV ${d.hrv ?? '—'}${z} · RHR ${d.rhr ?? '—'} · rdy ${d.readiness ?? '—'}`
  })

  const sleepView = data.recovery.series
  bind('.tri-ana-sleep', '.tri-sleep-svg', sleepView.length, sleepView.length, i => {
    const d = sleepView[i]
    const debt = d.sleepDebtS != null ? `${(d.sleepDebtS / 3600).toFixed(1)}h` : '—'
    return `${d.date} · ${d.sleepS != null ? hms(d.sleepS) : '—'} · score ${d.sleepScore ?? '—'} · debt ${debt}`
  })

  const trend = data.engine.vo2max.trend
  bind('.tri-engine-vo2', '.tri-engine-vo2-spark', trend.length, ANA_W, i => {
    const p = trend[i]
    const src = p.method === 'bike' ? `bike (${tl('projected')})` : p.method
    return `${p.weekStart} · ${p.vo2max.toFixed(1)} ml/kg/min · ${src}`
  })

  const wk = data.weekly
  const bindWkTrend = (blockSel: string, kind: WkKind): void => {
    const block = panel.querySelector<HTMLElement>(blockSel)
    const svgEl = block?.querySelector<SVGElement>('.tri-wkt-svg')
    const cursor = svgEl?.querySelector<SVGElement>('.tri-ana-cursor')
    const wrap = block?.querySelector<HTMLElement>('.tri-wkdetail-wrap')
    if (!block || !svgEl || !cursor || !wrap || !wk.length) return
    const pts = Array.from(svgEl.querySelectorAll<SVGElement>('.tri-wkt-pt'))
    const mark = (cls: string, idx: number | null): void => {
      for (const p of pts) p.classList.toggle(cls, p.dataset.week === String(idx))
    }
    const idxAt = (event: MouseEvent): number => {
      const r = svgEl.getBoundingClientRect()
      const f = clampN((event.clientX - r.left) / r.width, 0, 1)
      return Math.min(wk.length - 1, Math.floor(f * wk.length))
    }
    let sel: number | null = null
    const onMove = (event: MouseEvent): void => {
      const i = idxAt(event)
      const cx = (((i + 0.5) / wk.length) * ANA_W).toFixed(2)
      cursor.setAttribute('x1', cx)
      cursor.setAttribute('x2', cx)
      mark('tri-wkt-pt--hot', i)
      block.classList.add('tri-chart--hover')
    }
    const onLeave = (): void => {
      block.classList.remove('tri-chart--hover')
      mark('tri-wkt-pt--hot', null)
    }
    const onClick = (event: MouseEvent): void => {
      const i = idxAt(event)
      sel = sel === i ? null : i
      if (sel != null) renderWkDetail(block, data, kind, sel)
      mark('tri-wkt-pt--sel', sel)
      wrap.classList.toggle('tri-wkdetail-wrap--open', sel != null)
    }
    svgEl.addEventListener('mousemove', onMove)
    svgEl.addEventListener('mouseleave', onLeave)
    svgEl.addEventListener('click', onClick)
    cleanups.push(() => {
      svgEl.removeEventListener('mousemove', onMove)
      svgEl.removeEventListener('mouseleave', onLeave)
      svgEl.removeEventListener('click', onClick)
    })
  }
  bindWkTrend('.tri-ana-weekly', 'load')
  bindWkTrend('.tri-ana-effort', 'effort')

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
          `${shortDate(best.date)} · $${best.samples.length}\\times$ · ${wNum(best.min)}–${wNum(best.max)} ${weightUnitLabel()} · $\\Delta${wSigned(delta, 1)}$${bmrTxt}`,
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
      const samples = tr ? trendSamples(tr) : null
      if (!wrap || !svgEl || !cursor || !readout || !tr || !samples) continue
      items.push({
        svgEl,
        cursor,
        readout,
        hover: wrap,
        textOf: f => {
          const at = sampleTrend(samples, f)
          const band = `${fmtTrendShort(sport, Math.min(at.lo, at.hi))}–${fmtTrendShort(sport, Math.max(at.lo, at.hi))}`
          return `+${(at.days / 7).toFixed(1)} wk · ${fmtTrendVal(sport, at.value)} · ${band}`
        },
      })
    }
    cleanups.push(scrubGroup(items, f => f * ANA_W))
  }

  const radarBlock = panel.querySelector<HTMLElement>('.tri-engine-radar')
  const radarSvg = radarBlock?.querySelector<SVGElement>('.tri-radar-svg')
  if (radarBlock && radarSvg) {
    document.body.querySelector('.tri-radar-tip')?.remove()
    const radarTip = el('div', 'tri-gloss tri-radar-tip')
    radarTip.setAttribute('role', 'tooltip')
    document.body.appendChild(radarTip)
    const abilities = data.engine.abilities.sports
    const rawOf = (sp: SportAbility, a: AbilityAxis): string => {
      const pace = radarPaceHint(sp.sport, a)
      if (a.rawValue == null) return tl('no data')
      if (a.rawUnit === 's/100m') return `${clock(a.rawValue)} /100m`
      const vamFt = a.rawUnit === 'm/h' && isImperialUnit()
      const value = vamFt ? Math.round(a.rawValue * M_TO_FT) : a.rawValue
      return `${value} ${vamFt ? 'ft/h' : a.rawUnit}${pace ? ` (${pace})` : ''}`
    }
    const projTxtOf = (a: AbilityAxis): string =>
      a.proj != null && a.proj !== a.score ? ` → ${a.proj}/100` : ''
    const onMove = (event: MouseEvent) => {
      const n = abilities[0]?.axes.length ?? 0
      if (!n) return
      const rect = radarSvg.getBoundingClientRect()
      const dx = event.clientX - (rect.left + rect.width / 2)
      const dy = event.clientY - (rect.top + rect.height / 2)
      const deg = (Math.atan2(dy, dx) * 180) / Math.PI
      const idx = ((Math.round(((deg + 90) / 360) * n) % n) + n) % n
      const pressedSports = (radarBlock.dataset.pressed ?? '').split(',').filter(Boolean)
      if (!pressedSports.length) {
        radarTip.classList.remove('tri-gloss--on')
        return
      }
      const single =
        pressedSports.length === 1 ? abilities.find(sp => sp.sport === pressedSports[0]) : undefined
      if (single) {
        const a = single.axes[idx]
        radarTip.replaceChildren(
          el('span', 'tri-gloss-h', `${tl(single.sport)} · ${tl(a.label)}`),
          el(
            'span',
            'tri-gloss-def',
            `${rawOf(single, a)} · ${a.score != null ? `${a.score}/100` : '—'}${projTxtOf(a)}`,
          ),
          renderGlossDef(radarAxisDefinition(single.sport, a)),
          renderGlossDef(radarNotationDefinition(a)),
        )
      } else {
        const activeAbilities = abilities.filter(sp => pressedSports.includes(sp.sport))
        const rows: HTMLElement[] = [
          el(
            'span',
            'tri-gloss-h',
            radarAxisLabel(activeAbilities, idx) || tl(abilities[0].axes[idx].label),
          ),
        ]
        if (radarBlock.dataset.sport === 'avg') {
          const xs = abilities
            .filter(sp => pressedSports.includes(sp.sport))
            .map(sp => sp.axes[idx].score)
            .filter((v): v is number => v != null)
          if (xs.length)
            rows.push(
              el(
                'span',
                'tri-gloss-def',
                `${tl('average')}: ${Math.round(xs.reduce((acc, v) => acc + v, 0) / xs.length)}/100`,
              ),
            )
        }
        for (const sp of abilities) {
          if (!pressedSports.includes(sp.sport)) continue
          const a = sp.axes[idx]
          rows.push(
            el(
              'span',
              'tri-gloss-def',
              `${tl(sp.sport)}: ${a.score != null ? `${a.score}/100` : '—'}${projTxtOf(a)} · ${rawOf(sp, a)}`,
            ),
          )
        }
        radarTip.replaceChildren(...rows)
      }
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

  const vo2t = panel.querySelector<HTMLElement>('.tri-vo2t')
  if (vo2t) {
    document.body.querySelector('.tri-vo2t-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-vo2t-tip')
    tip.setAttribute('role', 'tooltip')
    document.body.appendChild(tip)
    const place = (event: MouseEvent): void => {
      const pr = tip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const bound: Array<[HTMLElement, (e: MouseEvent) => void, () => void]> = []
    for (const t of Array.from(vo2t.querySelectorAll<HTMLElement>('[data-tip-h]'))) {
      const move = (e: MouseEvent): void => {
        tip.replaceChildren(
          el('span', 'tri-gloss-h', t.dataset.tipH ?? ''),
          el('span', 'tri-gloss-def', t.dataset.tipD ?? ''),
        )
        tip.classList.add('tri-gloss--on')
        place(e)
      }
      const leave = (): void => tip.classList.remove('tri-gloss--on')
      t.addEventListener('mousemove', move)
      t.addEventListener('mouseleave', leave)
      bound.push([t, move, leave])
    }
    cleanups.push(() => {
      for (const [t, move, leave] of bound) {
        t.removeEventListener('mousemove', move)
        t.removeEventListener('mouseleave', leave)
      }
      tip.remove()
    })
  }

  const vo2p = panel.querySelector<HTMLElement>('.tri-vo2p')
  if (vo2p) {
    document.body.querySelector('.tri-vo2p-tip')?.remove()
    const tip = el('div', 'tri-gloss tri-vo2p-tip')
    tip.setAttribute('role', 'tooltip')
    document.body.appendChild(tip)
    const place = (event: MouseEvent): void => {
      const pr = tip.getBoundingClientRect()
      const left =
        event.clientX + 14 + pr.width > window.innerWidth - 8
          ? event.clientX - 14 - pr.width
          : event.clientX + 14
      const top =
        event.clientY + 14 + pr.height > window.innerHeight - 8
          ? event.clientY - 14 - pr.height
          : event.clientY + 14
      tip.style.left = `${Math.max(8, left).toFixed(0)}px`
      tip.style.top = `${Math.max(8, top).toFixed(0)}px`
    }
    const bound: Array<[SVGElement, (e: MouseEvent) => void, () => void]> = []
    for (const t of Array.from(vo2p.querySelectorAll<SVGElement>('[data-tip-h]'))) {
      const move = (e: MouseEvent): void => {
        tip.replaceChildren(
          el('span', 'tri-gloss-h', t.getAttribute('data-tip-h') ?? ''),
          el('span', 'tri-gloss-def', t.getAttribute('data-tip-d') ?? ''),
        )
        tip.classList.add('tri-gloss--on')
        place(e)
      }
      const leave = (): void => tip.classList.remove('tri-gloss--on')
      t.addEventListener('mousemove', move)
      t.addEventListener('mouseleave', leave)
      bound.push([t, move, leave])
    }
    cleanups.push(() => {
      for (const [t, move, leave] of bound) {
        t.removeEventListener('mousemove', move)
        t.removeEventListener('mouseleave', leave)
      }
      tip.remove()
    })
  }

  const ftpBlock = panel.querySelector<HTMLElement>('.tri-ftp')
  if (ftpBlock) {
    const inputs = Array.from(ftpBlock.querySelectorAll<HTMLInputElement>('[data-ftp-param]'))
    const out = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-out="${key}"]`)
    const valOut = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-val="${key}"]`)
    const bar = (key: string): HTMLElement | null =>
      ftpBlock.querySelector<HTMLElement>(`[data-ftp-bar="${key}"]`)
    const inputFor = (key: string): HTMLInputElement | undefined =>
      inputs.find(input => input.dataset.ftpParam === key)
    const num = (key: string): number => Number(inputFor(key)?.value ?? 0)
    const setText = (key: string, value: string): void => {
      const node = out(key)
      if (node) node.textContent = value
    }
    const setBar = (key: string, value: number): void => {
      const node = bar(key)
      if (node) node.style.width = `${clampN((value / 350) * 100, 4, 100)}%`
    }
    const setNum = (key: string, display: string, unit: string): void => {
      const numIn = ftpBlock.querySelector<HTMLInputElement>(`[data-ftp-num="${key}"]`)
      if (numIn && document.activeElement !== numIn) numIn.value = display
      const unitNode = ftpBlock.querySelector<HTMLElement>(`[data-ftp-unit="${key}"]`)
      if (unitNode) unitNode.textContent = unit
    }
    const renderFtp = (): void => {
      const mass = num('mass')
      const vo2 = num('vo2')
      const discountPct = num('discount')
      const thresholdPct = num('threshold')
      const efficiencyPct = num('efficiency')
      const discount = discountPct / 100
      const threshold = thresholdPct / 100
      const efficiency = efficiencyPct / 100
      const absRun = (vo2 * mass) / 1000
      const absCyc = absRun * (1 - discount)
      const thr = absCyc * threshold
      const met = (thr * 20.9 * 1000) / 60
      const ftpEff = met * efficiency
      const map = (Math.max(0, vo2 * (1 - discount) - 7) * mass) / 1.8 / 6.12
      const ftpAcsm = map * 0.75
      const ftp = (ftpEff + ftpAcsm) / 2
      const ftpShown = Math.round(ftp / 10) * 10
      const low = Math.round((ftp - 25) / 5) * 5
      const high = Math.round((ftp + 25) / 5) * 5
      const writeVal = (key: string, value: string): void => {
        const node = valOut(key)
        if (node) node.textContent = value
      }
      setNum('mass', wNum(mass, 1, 0), ` ${weightUnitLabel()}`)
      setNum('vo2', vo2.toFixed(1), '')
      writeVal('discount', `${discountPct}%`)
      writeVal('threshold', `${thresholdPct}%`)
      writeVal('efficiency', `${efficiencyPct}%`)
      setText('headline', String(ftpShown))
      setText('band', `${low}-${high} W`)
      setText('wkg', `${(ftpShown / mass).toFixed(2)} W/kg`)
      setText('efficiencyFtp', `${Math.round(ftpEff)} W`)
      setText('acsmFtp', `${Math.round(ftpAcsm)} W`)
      setText('absoluteRunningVo2', `${absRun.toFixed(2)} L/min`)
      setText('cyclingVo2max', `${absCyc.toFixed(2)} L/min`)
      setText('thresholdVo2', `${thr.toFixed(2)} L/min`)
      setText('metabolicWatts', `${Math.round(met)} W`)
      setText('acsmMapWatts', `${Math.round(map)} W`)
      setBar('efficiencyFtp', ftpEff)
      setBar('acsmFtp', ftpAcsm)
    }
    const onInput = () => renderFtp()
    for (const input of inputs) input.addEventListener('input', onInput)
    const numInputs = Array.from(ftpBlock.querySelectorAll<HTMLInputElement>('[data-ftp-num]'))
    const onNum = (e: Event): void => {
      const numIn = e.target as HTMLInputElement
      const key = numIn.dataset.ftpNum
      const range = key ? inputFor(key) : undefined
      if (!key || !range) return
      let v = Number(numIn.value)
      if (!Number.isFinite(v)) return renderFtp()
      if (key === 'mass' && isImperialUnit()) v *= KG_PER_LB
      range.value = String(clampN(v, Number(range.min), Number(range.max)))
      renderFtp()
    }
    const onNumKey = (e: KeyboardEvent): void => {
      if (e.key === 'Enter') {
        e.preventDefault()
        ;(e.target as HTMLInputElement).blur()
      }
    }
    const onNumFocus = (e: Event): void => (e.target as HTMLInputElement).select()
    for (const numIn of numInputs) {
      numIn.addEventListener('change', onNum)
      numIn.addEventListener('keydown', onNumKey)
      numIn.addEventListener('focus', onNumFocus)
    }
    const reset = ftpBlock.querySelector<HTMLButtonElement>('.tri-ftp-reset')
    const onReset = (): void => {
      for (const input of inputs)
        if (input.dataset.ftpDefault) input.value = input.dataset.ftpDefault
      renderFtp()
    }
    reset?.addEventListener('click', onReset)
    renderFtp()
    cleanups.push(() => {
      for (const input of inputs) input.removeEventListener('input', onInput)
      for (const numIn of numInputs) {
        numIn.removeEventListener('change', onNum)
        numIn.removeEventListener('keydown', onNumKey)
        numIn.removeEventListener('focus', onNumFocus)
      }
      reset?.removeEventListener('click', onReset)
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
    label: 'ftp hypothesis',
    chart: 'ftp',
    hay: 'ftp watts power vo2 hypothesis slider acsm efficiency threshold lt2 vt2 cycling',
  },
  {
    label: 'abilities',
    chart: 'abilities',
    hay: 'abilities radar sprint threshold endurance climb cadence recovery power profile vam wkg swim bike run pace css stroke average',
  },
  {
    label: 'cardiovascular health',
    chart: 'cardio',
    hay: 'cardio cardiovascular heart rhr hrv efficiency factor decoupling aerobic drift',
  },
  {
    label: 'fitness · fatigue · form',
    chart: 'pmc',
    hay: 'pmc fitness fatigue form ctl atl tsb discipline swim bike run',
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
  dexa: 'dexa',
  bmi: 'body',
  hrv: 'recovery',
  rhr: 'recovery',
  tempdev: 'recovery',
  overreaching: 'recovery',
  oreadiness: 'recovery',
  sleepdebt: 'sleep',
  vo2max: 'vo2max',
  ftp: 'ftp',
  watts: 'ftp',
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

const ACTIVITY_FILTER_SPORTS: readonly string[] = ['bike', 'run', 'swim', 'walk']
const ACTIVITY_SORT_KEYS: readonly string[] = ['distance', 'cadence', 'pace']

interface ActivityQuery {
  filterSport: string | null
  sortKey: string | null
  tokens: string[]
}

const parseActivityQuery = (rawTokens: string[]): ActivityQuery => {
  let filterSport: string | null = null
  let sortKey: string | null = null
  const tokens: string[] = []
  for (const t of rawTokens) {
    if (t.startsWith('filter:')) {
      const fv = t.slice(7)
      filterSport = fv === 'hike' ? 'walk' : fv
    } else if (t.startsWith('sort:')) {
      sortKey = t.slice(5)
    } else if (t) tokens.push(t)
  }
  return { filterSport, sortKey, tokens }
}

const sortActivitiesBy = <
  T extends Pick<ActivitySummary, 'distanceKm' | 'cadence' | 'movingTimeS'>,
>(
  acts: T[],
  sortKey: string | null,
): T[] => {
  if (!sortKey) return acts
  return acts.sort((a, b) => {
    if (sortKey === 'distance') return b.distanceKm - a.distanceKm
    if (sortKey === 'cadence') return (b.cadence ?? 0) - (a.cadence ?? 0)
    if (sortKey === 'pace') {
      const sa = a.movingTimeS > 0 ? a.distanceKm / a.movingTimeS : 0
      const sb = b.movingTimeS > 0 ? b.distanceKm / b.movingTimeS : 0
      return sb - sa
    }
    return 0
  })
}

const activityCommandHints = (
  lastToken: string,
  ritem: (title: HTMLElement | string, sub: string) => HTMLElement,
  noun: string,
): HTMLElement[] => {
  const hints: HTMLElement[] = []
  const filterValue = lastToken.startsWith('filter:') ? lastToken.slice(7) : null
  const sortValue = lastToken.startsWith('sort:') ? lastToken.slice(5) : null
  if (filterValue !== null && ![...ACTIVITY_FILTER_SPORTS, 'hike'].includes(filterValue)) {
    for (const f of ACTIVITY_FILTER_SPORTS)
      if (f.startsWith(filterValue)) {
        const it = ritem(searchCommandTitle('filter:', f), `filter ${noun}`)
        it.dataset.insert = `filter:${f}`
        hints.push(it)
      }
  } else if (sortValue !== null && !ACTIVITY_SORT_KEYS.includes(sortValue)) {
    for (const s of ACTIVITY_SORT_KEYS)
      if (s.startsWith(sortValue)) {
        const it = ritem(searchCommandTitle('sort:', s), `sort ${noun}`)
        it.dataset.insert = `sort:${s}`
        hints.push(it)
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
  return hints
}

const marqueeCtl = (): { run: (name: HTMLElement) => void; stop: () => void } => {
  let host: HTMLElement | null = null
  let raf = 0
  let start = 0
  const stop = () => {
    if (raf) {
      cancelAnimationFrame(raf)
      raf = 0
    }
    if (host) {
      host.scrollLeft = 0
      host.style.textOverflow = ''
      host = null
    }
    start = 0
  }
  const run = (name: HTMLElement) => {
    if (name === host) return
    stop()
    const max = name.scrollWidth - name.clientWidth
    if (max <= 2) return
    host = name
    name.style.textOverflow = 'clip'
    const leg = (max / 36) * 1000
    const cycle = leg * 2 + 1400
    const step = (ts: number) => {
      if (!name.isConnected) {
        stop()
        return
      }
      if (!start) start = ts
      const t = (ts - start) % cycle
      name.scrollLeft =
        t < leg
          ? (t / leg) * max
          : t < leg + 700
            ? max
            : t < leg * 2 + 700
              ? max - ((t - leg - 700) / leg) * max
              : 0
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
  }
  return { run, stop }
}

const detailHead = (date: string, title: string): { head: HTMLElement; back: HTMLElement } => {
  const head = el('div', 'tri-pop-head tri-pop-head--detail')
  const row = el('div', 'tri-pop-head-row')
  const back = el('button', 'tri-ana-back tri-ana-back--ico')
  back.setAttribute('type', 'button')
  back.setAttribute('aria-label', tl('go back'))
  const ico = svg('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' })
  ico.appendChild(svg('path', { d: 'M19 12H5M11 6l-6 6 6 6' }))
  back.appendChild(ico)
  row.append(el('span', 'tri-pop-date', date), back)
  const titleEl = el('span', 'tri-pop-title', title)
  const marquee = marqueeCtl()
  titleEl.addEventListener('mouseenter', () => marquee.run(titleEl))
  titleEl.addEventListener('mouseleave', marquee.stop)
  head.append(row, titleEl)
  return { head, back }
}

const setupAnalytics = (root: HTMLElement): (() => void) | null => {
  if (root.dataset.ouraDetailPath) OURA_DETAIL_PATH = root.dataset.ouraDetailPath
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
  let loaded = false
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailLoaded = false
  let scrubCleanup: (() => void) | null = null
  let selIndex = -1

  const render = (d: Analytics) => {
    data = d
    scrubCleanup?.()
    scrubCleanup = null
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
  const close = () => {
    root.classList.remove('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'true')
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
        DETAIL_CURVE_YEAR_REF = d.powerCurveYearRef ?? []
        DETAIL_CURVE_YEAR = d.powerCurveYear ?? null
        DETAIL_FTP = d.ftp ?? null
        DETAIL_GOAL_FTP = d.goalFtp ?? null
        DETAIL_VT1 = d.vt1Hr ?? null
      })
      .catch(() => {})
  }
  const showActivity = (id: string) => {
    if (!detail) return
    void loadDetails().then(() => {
      const d = detailData?.details?.[id]
      if (!d) return
      const card = el('div', 'tri-pop-card')
      const { head, back } = detailHead(shortDate(d.date), d.name || d.sport)
      card.appendChild(head)
      const act = renderDetail(d, detailData)
      setActivityExpanded(act, true)
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
    const { filterSport, sortKey, tokens } = parseActivityQuery(rawTokens)

    const metrics: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1]
    const hints = activityCommandHints(lastToken, ritem, 'activities')

    if (!filterSport && !sortKey) {
      for (const s of SEARCH_SECTIONS)
        if (matchHay(`${s.label} ${tl(s.label)} ${s.hay}`.toLowerCase(), tokens)) {
          const it = ritem(tl(s.label), 'section')
          it.dataset.chart = s.chart
          metrics.push(it)
        }
      for (const key of glossKeys()) {
        const g = glossFor(key)
        if (g && matchHay(`${key} ${g.term} ${g.def}`.toLowerCase(), tokens)) {
          const it = ritem(g.term, g.def)
          it.dataset.chart = GLOSS_CHART[key] ?? 'pmc'
          metrics.push(it)
        }
      }
    }

    const acts = sortActivitiesBy(
      (data?.activities ?? []).filter(a => {
        if (filterSport && a.sport !== filterSport) return false
        return (
          tokens.length === 0 || matchHay(`${a.name} ${a.sport} ${a.date}`.toLowerCase(), tokens)
        )
      }),
      sortKey,
    )

    if (hints.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
      for (const it of hints) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (metrics.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', tl('metrics & terms')))
      for (const it of metrics.slice(0, 8)) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', tl('activities')))
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
      results.appendChild(el('div', 'tri-ana-empty', tl('no matches')))
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
    toMain()
    root.classList.add('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
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
  detail?.addEventListener('click', onCardToggle)
  document.addEventListener('keydown', onKey)
  const onUnitChange = () => {
    if (data) {
      render(data)
    }
  }
  window.addEventListener('tri:unit', onUnitChange)
  window.addEventListener('tri:locale', onUnitChange)

  return () => {
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    results?.removeEventListener('click', onResultsClick)
    detail?.removeEventListener('click', onCardToggle)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('tri:unit', onUnitChange)
    window.removeEventListener('tri:locale', onUnitChange)
    scrubCleanup?.()
  }
}

type GeoFC = { type: 'FeatureCollection'; features: unknown[] }
const emptyFC = (): GeoFC => ({ type: 'FeatureCollection', features: [] })
type GeoCoord = [number, number]

const gpsRoute = (d: StravaActivityDetail): readonly StravaMapPoint[] =>
  d.mapRoute && d.mapRoute.length >= 2 ? d.mapRoute : d.route

const gpsSegments = (d: StravaActivityDetail): readonly (readonly StravaMapPoint[])[] =>
  gpsRoute(d).length >= 2 ? [gpsRoute(d)] : []

const lineFeatures = (route: readonly StravaMapPoint[], props: Record<string, unknown> = {}) =>
  route.length >= 2
    ? [
        {
          type: 'Feature',
          properties: props,
          geometry: { type: 'LineString', coordinates: route.map(p => [p.lng, p.lat] as GeoCoord) },
        },
      ]
    : []

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
const ROUTE_SPORTS = ['bike', 'run', 'walk'] as const
const OVERVIEW_CELL = 0.0008

const overviewCellKey = (lng: number, lat: number): string =>
  `${Math.round(lng / OVERVIEW_CELL)},${Math.round(lat / OVERVIEW_CELL)}`

const HEAT_SNAP = 0.0003
const HEAT_PRUNE_DENSITY = 4

const heatKey = (lng: number, lat: number): string =>
  `${Math.round(lng / HEAT_SNAP)},${Math.round(lat / HEAT_SNAP)}`

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
  if (k === 'cad') return d.avgCadence == null ? null : d.avgCadence * (d.sport === 'bike' ? 1 : 2)
  return d.movingTimeS > 0 ? d.distanceKm / (d.movingTimeS / 3600) : null
}

const overviewFmt = (
  k: (typeof OVERVIEW_METRICS)[number],
  sport: ActivityKind,
  v: number,
): string => {
  if (k === 'w') return `${Math.round(v)} W`
  if (k === 'hr') return `${Math.round(v)} bpm`
  if (k === 'cad') return `${Math.round(v)} ${sport === 'bike' ? 'rpm' : 'spm'}`
  if (sport === 'bike') return `${(v * KM_TO_MI).toFixed(1)} mph`
  return `${clock(3600 / (v * KM_TO_MI))} /mi`
}

const buildOverview = (dp: DetailPayload | null, enabled: ReadonlySet<ActivityKind>): Overview => {
  const acts: StravaActivityDetail[] = []
  const det = dp?.details ?? {}
  for (const k in det) {
    const d = det[k]
    if (enabled.has(d.sport) && gpsRoute(d).length >= 2) acts.push(d)
  }
  const counts = new Map<string, number>()
  for (const d of acts) {
    const cells = new Set<string>()
    for (const r of gpsSegments(d))
      for (let i = 0; i < r.length - 1; i++) stampSegment(r[i], r[i + 1], cells)
    for (const c of cells) counts.set(c, (counts.get(c) ?? 0) + 1)
  }
  let maxCount = 1
  for (const c of counts.values()) if (c > maxCount) maxCount = c
  const bucketOf = (c: number): number =>
    maxCount > 1
      ? Math.min(7, Math.max(1, 1 + Math.round((6 * Math.log(c)) / Math.log(maxCount))))
      : 1
  const nodeAcc = new Map<string, [number, number, number]>()
  for (const d of acts)
    for (const r of gpsSegments(d))
      for (const p of r) {
        const k = heatKey(p.lng, p.lat)
        const a = nodeAcc.get(k)
        if (a) {
          a[0] += p.lng
          a[1] += p.lat
          a[2] += 1
        } else nodeAcc.set(k, [p.lng, p.lat, 1])
      }
  const nodeCoord = (k: string): GeoCoord => {
    const a = nodeAcc.get(k)!
    return [a[0] / a[2], a[1] / a[2]]
  }
  const edgeKey = (ka: string, kb: string): string => (ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`)
  const seqs: string[][] = []
  const edgeCount = new Map<string, number>()
  for (const d of acts) {
    const seen = new Set<string>()
    for (const r of gpsSegments(d)) {
      const seq: string[] = []
      for (const p of r) {
        const k = heatKey(p.lng, p.lat)
        if (seq.length && seq[seq.length - 1] === k) continue
        seq.push(k)
      }
      if (seq.length >= 2) seqs.push(seq)
      for (let i = 0; i < seq.length - 1; i++) seen.add(edgeKey(seq[i], seq[i + 1]))
    }
    for (const e of seen) edgeCount.set(e, (edgeCount.get(e) ?? 0) + 1)
  }
  const heatFeatures: unknown[] = []
  const drawnEdges = new Set<string>()
  for (const seq of seqs) {
    let runB = -1
    let coords: GeoCoord[] = []
    const flush = () => {
      if (coords.length >= 2)
        heatFeatures.push({
          type: 'Feature',
          properties: { heat: runB },
          geometry: { type: 'LineString', coordinates: coords },
        })
      coords = []
      runB = -1
    }
    for (let i = 0; i < seq.length - 1; i++) {
      const edge = edgeKey(seq[i], seq[i + 1])
      if (drawnEdges.has(edge)) {
        flush()
        continue
      }
      const ca = nodeCoord(seq[i])
      const cb = nodeCoord(seq[i + 1])
      const cell = counts.get(overviewCellKey((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)) ?? 1
      if ((edgeCount.get(edge) ?? 0) < 2 && cell >= HEAT_PRUNE_DENSITY) {
        flush()
        continue
      }
      drawnEdges.add(edge)
      const bucket = bucketOf(cell)
      if (bucket !== runB) {
        flush()
        coords = [ca]
        runB = bucket
      }
      coords.push(cb)
    }
    flush()
  }
  const ranges = new Map<string, [number, number]>()
  for (const k of OVERVIEW_METRICS)
    for (const sport of ROUTE_SPORTS) {
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
    traceFeatures.push(...lineFeatures(gpsRoute(d), props))
  }
  const legend: Record<OverviewMode, OverviewLegend | null> = {
    heat: { lo: '$1\\times$', hi: `$${maxCount}\\times$` },
    w: null,
    hr: null,
    cad: null,
    spd: null,
  }
  for (const k of OVERVIEW_METRICS) {
    const present = ROUTE_SPORTS.filter(s => ranges.has(`${k}:${s}`))
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

const heatOpacityExpr: unknown[] = ['interpolate', ['linear'], ['get', 'heat'], 1, 0.5, 7, 0.95]

const heatWidthExpr: unknown[] = (() => {
  const w = (base: number, k: number) => ['+', base, ['*', k, ['-', ['get', 'heat'], 1]]]
  return ['interpolate', ['linear'], ['zoom'], 10, w(0.55, 0.08), 14, w(0.9, 0.14), 16, w(1.3, 0.2)]
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
  features: lineFeatures(gpsRoute(d)),
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
    const coords = (f as { geometry: { coordinates: GeoCoord[] } }).geometry.coordinates
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
  let activeRouteId: string | null = null
  let activeRouteMetric = 0
  const canvas = root.querySelector<HTMLElement>('.tri-map-canvas')
  const overlay = root.querySelector<HTMLElement>('.tri-map-overlay')
  const tip = root.querySelector<HTMLElement>('.tri-map-tip')
  const legendLo = overlay?.querySelector<HTMLElement>('.tri-map-legend-lo') ?? null
  const legendHi = overlay?.querySelector<HTMLElement>('.tri-map-legend-hi') ?? null
  const legendBar = overlay?.querySelector<HTMLElement>('.tri-map-legend-bar') ?? null
  const modeBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-map-mode'))
  const sportBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-map-sport'))
  const side = root.querySelector<HTMLElement>('.tri-map-side')
  const sideFold = side?.querySelector<HTMLButtonElement>('.tri-map-side-fold') ?? null
  const styleBtn = side?.querySelector<HTMLButtonElement>('.tri-map-style') ?? null
  let mode: OverviewMode = 'heat'
  const enabledSports = new Set<ActivityKind>(ROUTE_SPORTS)
  const overviewCache = new Map<string, Overview>()
  const getOverview = () => {
    const key = ROUTE_SPORTS.filter(s => enabledSports.has(s)).join(',') || 'none'
    let ov = overviewCache.get(key)
    if (!ov) {
      ov = buildOverview(detailData, enabledSports)
      overviewCache.set(key, ov)
    }
    return ov
  }

  const mapCtl = (() => {
    let map: any = null
    let started = false
    let okFlag = false
    let hoverId: string | null = null
    let styleSeq = 0
    let eventsBound = false
    let selection: { d: StravaActivityDetail; i: number } | null = null
    const ready = () => Boolean(map && okFlag)
    const clearHover = () => {
      hoverId = null
      if (ready()) map.getSource('tri-hov')?.setData(emptyFC())
      if (map) map.getCanvas().style.cursor = ''
      tip?.classList.remove('tri-map-tip--on')
    }
    const applyMode = () => {
      if (!ready()) return
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
      if (!ready()) return
      const spec = metricSpecs(d)[i]
      if (spec)
        map.setPaintProperty(
          'tri-sel',
          'line-gradient',
          gradientExpr(d, spec.pick, spec.ramp, spec.zeroGap),
        )
    }
    const addSource = (id: string, source: Record<string, unknown>) => {
      if (!map.getSource(id)) map.addSource(id, source)
    }
    const addLayer = (layer: Record<string, unknown>) => {
      const id = layer.id
      if (typeof id === 'string' && !map.getLayer(id)) map.addLayer(layer)
    }
    const onTraceMove = (e: any) => {
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
        if (!canvas) return
        const bound = canvas.clientWidth - tip.offsetWidth - 8
        tip.style.left = `${Math.min(e.point.x + 14, Math.max(8, bound))}px`
        tip.style.top = `${e.point.y + 14}px`
      }
    }
    const onTraceClick = (e: any) => {
      if (panel.classList.contains('tri-map--detail')) return
      const id = e.features?.[0]?.properties?.id
      if (id != null) showRoute(String(id))
    }
    const bindEvents = () => {
      if (eventsBound) return
      map.on('mousemove', 'tri-hit', onTraceMove)
      map.on('mouseleave', 'tri-hit', clearHover)
      map.on('click', 'tri-hit', onTraceClick)
      eventsBound = true
    }
    const installLayers = () => {
      if (!map) return
      if (readTriMapStyle() === 'mono') applyMonochromeMapPalette(map)
      addSource('tri-heat', { type: 'geojson', data: emptyFC() })
      addLayer({
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
      addSource('tri-traces', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-traces',
        type: 'line',
        source: 'tri-traces',
        layout: { 'line-cap': 'round', 'line-join': 'round', visibility: 'none' },
        paint: {
          'line-color': '#fc4c02',
          'line-opacity': 0.44,
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 0.75, 14, 1.2, 16, 1.8],
        },
      })
      addLayer({
        id: 'tri-hit',
        type: 'line',
        source: 'tri-traces',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#000', 'line-opacity': 0, 'line-width': 12 },
      })
      addSource('tri-hov', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-hov-casing',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fff9f3', 'line-width': 3.2 },
      })
      addLayer({
        id: 'tri-hov',
        type: 'line',
        source: 'tri-hov',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fc4c02', 'line-width': 2 },
      })
      addSource('tri-sel', { type: 'geojson', lineMetrics: true, data: emptyFC() })
      addLayer({
        id: 'tri-sel-casing',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: { 'line-color': '#fff9f3', 'line-width': 3.4 },
      })
      addLayer({
        id: 'tri-sel',
        type: 'line',
        source: 'tri-sel',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-width': 2.1,
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
      addSource('tri-dot', { type: 'geojson', data: emptyFC() })
      addLayer({
        id: 'tri-dot',
        type: 'circle',
        source: 'tri-dot',
        paint: {
          'circle-radius': 3.5,
          'circle-color': '#fc4c02',
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#fff9f3',
        },
      })
      bindEvents()
      okFlag = true
      applyMode()
    }
    const refreshMapData = () => {
      if (selection) select(selection.d, selection.i)
      else drawOverview()
    }
    const applyMapStyle = () => {
      if (!map) return
      const seq = ++styleSeq
      okFlag = false
      clearHover()
      map.setStyle(mapboxStyleUrl(readTriMapStyle()))
      map.once('style.load', () => {
        if (!map || seq !== styleSeq) return
        installLayers()
        refreshMapData()
      })
    }
    const init = async (): Promise<void> => {
      if (started) return
      started = true
      if (!canvas) return
      const mapboxgl = await loadMapbox()
      if (!mapboxgl) {
        started = false
        canvas.classList.add('tri-map-canvas--down')
        canvas.textContent = tl('map unavailable')
        return
      }
      canvas.classList.remove('tri-map-canvas--down')
      canvas.textContent = ''
      map = new mapboxgl.Map({
        container: canvas,
        style: mapboxStyleUrl(readTriMapStyle()),
        center: [-79.4, 43.7],
        zoom: 9,
        attributionControl: false,
      })
      ;(canvas as unknown as { _mapInstance: unknown })._mapInstance = map
      await new Promise<void>(resolve => map.once('load', () => resolve()))
      installLayers()
    }
    const setOverviewData = () => {
      if (!ready()) return
      const ov = getOverview()
      map.getSource('tri-heat')?.setData(ov.heat)
      map.getSource('tri-traces')?.setData(ov.traces)
      applyMode()
    }
    const drawOverview = () => {
      if (!ready()) return
      setOverviewData()
      const b = fcBounds(getOverview().traces)
      if (b) map.fitBounds(b, { padding: 48, maxZoom: 13, duration: reduce ? 0 : 600 })
    }
    const select = (d: StravaActivityDetail, i: number) => {
      selection = { d, i }
      if (!ready()) return
      clearHover()
      map.getSource('tri-sel')?.setData(routeFC(d))
      recolor(d, i)
      map.setPaintProperty('tri-heat', 'line-opacity', 0.06)
      map.setPaintProperty('tri-traces', 'line-opacity', 0.06)
      const b = fcBounds(routeFC(d))
      if (b) map.fitBounds(b, { padding: 40, maxZoom: 15, duration: reduce ? 0 : 600 })
    }
    const moveDot = (lng: number, lat: number) =>
      ready() ? map.getSource('tri-dot')?.setData(pointFC(lng, lat)) : undefined
    const clearSelection = () => {
      selection = null
      if (!ready()) return
      clearHover()
      map.getSource('tri-sel')?.setData(emptyFC())
      map.getSource('tri-dot')?.setData(emptyFC())
      setOverviewData()
    }
    const resize = () => map?.resize()
    const dispose = () => {
      clearHover()
      if (map?.remove) map.remove()
      map = null
      started = false
      okFlag = false
      eventsBound = false
      selection = null
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
      applyMapStyle,
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
        DETAIL_CURVE_YEAR_REF = d.powerCurveYearRef ?? []
        DETAIL_CURVE_YEAR = d.powerCurveYear ?? null
        DETAIL_FTP = d.ftp ?? null
        DETAIL_GOAL_FTP = d.goalFtp ?? null
        DETAIL_VT1 = d.vt1Hr ?? null
        overviewCache.clear()
        mapCtl.drawOverview()
        if (search?.value) runSearch()
      })
      .catch(() => {})
  }
  const syncSportBtns = () => {
    for (const b of sportBtns)
      b.setAttribute('aria-pressed', String(enabledSports.has(b.dataset.sport as ActivityKind)))
  }
  const setEnabledSports = (next: ReadonlySet<ActivityKind>) => {
    if (next.size === enabledSports.size && [...next].every(s => enabledSports.has(s))) return
    enabledSports.clear()
    for (const s of next) enabledSports.add(s)
    syncSportBtns()
    mapCtl.drawOverview()
  }
  const toggleSport = (sport: ActivityKind) => {
    const next = new Set(enabledSports)
    if (next.has(sport)) next.delete(sport)
    else next.add(sport)
    setEnabledSports(next)
  }
  let detailAnim: Animation | null = null
  const finishCloseDetail = () => {
    panel.classList.remove('tri-map--detail')
    if (detail) detail.replaceChildren()
    activeRouteId = null
    activeRouteMetric = 0
    mapCtl.clearSelection()
    requestAnimationFrame(() => mapCtl.resize())
  }
  const closeDetail = (animate = false) => {
    detailAnim?.cancel()
    detailAnim = null
    const card = detail?.querySelector<HTMLElement>('.tri-pop-card')
    if (!animate || reduce || !card) {
      finishCloseDetail()
      return
    }
    const anim = card.animate(
      [
        { opacity: 1, transform: 'translateX(0)' },
        { opacity: 0, transform: 'translateX(1.25rem)' },
      ],
      { duration: 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' },
    )
    detailAnim = anim
    const done = () => {
      if (detailAnim !== anim) return
      detailAnim = null
      finishCloseDetail()
    }
    anim.finished.then(done).catch(done)
  }
  const toMain = () => {
    closeDetail()
    if (search) search.value = ''
    panel.classList.remove('tri-map--searching')
    if (results) results.replaceChildren()
    selIndex = -1
    setEnabledSports(new Set(ROUTE_SPORTS))
  }
  const close = () => {
    root.classList.remove('tri-map-open')
    panel.setAttribute('aria-hidden', 'true')
  }
  const showRoute = (id: string, initialMetric = 0, selectMap = true) => {
    if (!detail) return
    void loadDetails().then(() => {
      const d = detailData?.details?.[id]
      if (!d) return
      activeRouteId = id
      activeRouteMetric = initialMetric
      detailAnim?.cancel()
      detailAnim = null
      const card = el('div', 'tri-pop-card')
      const { head, back } = detailHead(shortDate(d.date), d.name || d.sport)
      card.appendChild(head)
      const mapMode = mapCtl.ok()
      card.appendChild(
        renderMapDetail(d, {
          mapMode,
          initialMetric,
          onMetric: i => {
            activeRouteMetric = i
            if (mapMode) mapCtl.recolor(d, i)
          },
          onHover: mapMode ? p => mapCtl.moveDot(p.lng, p.lat) : undefined,
        }),
      )
      detail.replaceChildren(card)
      panel.classList.add('tri-map--detail')
      panel.classList.remove('tri-map--searching')
      results?.setAttribute('aria-hidden', 'true')
      back.addEventListener('click', () => closeDetail(true), { once: true })
      if (mapMode && selectMap) {
        requestAnimationFrame(() => {
          mapCtl.resize()
          mapCtl.select(d, initialMetric)
        })
      } else if (mapMode) {
        requestAnimationFrame(() => mapCtl.resize())
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
      setEnabledSports(new Set(ROUTE_SPORTS))
      return
    }
    panel.classList.add('tri-map--searching')
    results.setAttribute('aria-hidden', 'false')
    const rawTokens = q.split(/\s+/)
    const { filterSport, sortKey, tokens } = parseActivityQuery(rawTokens)
    const routeSport = ROUTE_SPORTS.find(s => s === filterSport)
    setEnabledSports(routeSport ? new Set([routeSport]) : new Set(ROUTE_SPORTS))
    const lastToken = rawTokens[rawTokens.length - 1]
    const hints = activityCommandHints(lastToken, ritem, 'routes')
    const ids = drawableIds()
    const acts = sortActivitiesBy(
      (data?.activities ?? []).filter(a => {
        if (!ids.has(String(a.id))) return false
        if (filterSport && a.sport !== filterSport) return false
        return (
          tokens.length === 0 || matchHay(`${a.name} ${a.sport} ${a.date}`.toLowerCase(), tokens)
        )
      }),
      sortKey,
    )
    if (hints.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
      for (const it of hints) grp.appendChild(it)
      results.appendChild(grp)
    }
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', tl('routes')))
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
    toMain()
    root.classList.add('tri-map-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    void loadDetails()
    startMap()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (panel.classList.contains('tri-map--searching') && search?.value) {
      search.value = ''
      runSearch()
      return
    }
    if (panel.classList.contains('tri-map--detail')) {
      closeDetail(true)
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
  const onSportClick = (event: MouseEvent) => {
    const b = (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-map-sport')
    const s = b?.dataset.sport as ActivityKind | undefined
    if (s) toggleSport(s)
  }
  const syncStyleBtn = () =>
    styleBtn?.setAttribute('aria-pressed', String(readTriMapStyle() === 'satellite'))
  const onStyleClick = () =>
    setTriMapStyle(readTriMapStyle() === 'satellite' ? 'mono' : 'satellite')
  const onFold = () => {
    if (!side) return
    const folded = side.classList.toggle('tri-map-side--folded')
    sideFold?.setAttribute('aria-expanded', String(!folded))
    sideFold?.setAttribute('aria-label', folded ? 'Expand map controls' : 'Collapse map controls')
  }
  const onMapStyle = () => {
    syncStyleBtn()
    mapCtl.applyMapStyle()
  }
  const onUnit = () => {
    if (activeRouteId) showRoute(activeRouteId, activeRouteMetric, false)
  }
  syncStyleBtn()

  if (pageMode) {
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
  side?.addEventListener('click', onSportClick)
  sideFold?.addEventListener('click', onFold)
  styleBtn?.addEventListener('click', onStyleClick)
  document.addEventListener('keydown', onKey)
  window.addEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
  window.addEventListener('tri:unit', onUnit)

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
    side?.removeEventListener('click', onSportClick)
    sideFold?.removeEventListener('click', onFold)
    styleBtn?.removeEventListener('click', onStyleClick)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener(TRI_MAP_STYLE_EVENT, onMapStyle)
    window.removeEventListener('tri:unit', onUnit)
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
  const sync = () => {
    const mi = isImperialUnit()
    unit.textContent = mi ? 'mi' : 'km'
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
  }
  const onClick = () => toggleTriUnit()
  unit.addEventListener('click', onClick)
  window.addEventListener('tri:unit', sync)
  sync()
  return () => {
    window.clearTimeout(showTimer)
    unit.removeEventListener('click', onClick)
    window.removeEventListener('tri:unit', sync)
    ann?.remove()
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
    else preview?.replaceChildren(el('div', 'tri-ana-empty', tl('no plan')))
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

  const toMain = () => {
    if (search) search.value = ''
    panel.classList.remove('tri-training--searching')
    results?.replaceChildren()
  }
  const open = () => {
    toMain()
    root.classList.add('tri-training-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
  }
  const close = () => {
    root.classList.remove('tri-training-open')
    panel.setAttribute('aria-hidden', 'true')
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
        : [el('div', 'tri-ana-empty', tl('no matches'))]),
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
    const g = glossFor(key)
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

const TRI_UNIT_KEY = 'tri-dist-unit'
const TRI_MAP_STYLE_KEY = 'tri-map-style'
const TRI_MAP_STYLE_EVENT = 'tri:mapstyle'
const TRI_MAP_STYLES = ['mono', 'streets', 'satellite'] as const
type TriMapStyle = (typeof TRI_MAP_STYLES)[number]

const readTriMapStyle = (): TriMapStyle => {
  try {
    const stored = localStorage.getItem(TRI_MAP_STYLE_KEY)
    if (stored === 'streets' || stored === 'satellite') return stored
  } catch {
    return 'mono'
  }
  return 'mono'
}

const mapboxStyleUrl = (style: TriMapStyle): string =>
  style === 'streets'
    ? 'mapbox://styles/mapbox/streets-v12'
    : style === 'satellite'
      ? 'mapbox://styles/mapbox/satellite-streets-v12'
      : 'mapbox://styles/mapbox/light-v11'

const setTriMapStyle = (next: TriMapStyle): void => {
  try {
    localStorage.setItem(TRI_MAP_STYLE_KEY, next)
  } catch {
    void 0
  }
  window.dispatchEvent(new CustomEvent(TRI_MAP_STYLE_EVENT, { detail: { style: next } }))
}

const nextTriMapStyle = (): TriMapStyle =>
  TRI_MAP_STYLES[(TRI_MAP_STYLES.indexOf(readTriMapStyle()) + 1) % TRI_MAP_STYLES.length]

const toggleTriMapStyle = (): void => setTriMapStyle(nextTriMapStyle())

const toggleTriUnit = (): void => {
  const next = !isImperialUnit()
  setDistanceUnit(next)
  try {
    localStorage.setItem(TRI_UNIT_KEY, next ? 'mi' : 'km')
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent('tri:unit'))
}

const TRI_PAGES: { path: string; label: string; hint: string }[] = [
  { path: '/triathlon', label: 'triathlon', hint: 'overview' },
  { path: '/triathlon/tools', label: 'tools', hint: 'gears' },
  { path: '/triathlon/calc', label: 'calculator', hint: 'race calc' },
  { path: '/triathlon/analytics', label: 'analytics', hint: 'charts' },
  { path: '/triathlon/maps', label: 'maps', hint: 'routes' },
  { path: '/triathlon/training', label: 'training', hint: 'plans' },
  { path: '/triathlon/feed', label: 'feed', hint: 'all activities' },
  { path: '/triathlon/on', label: 'on', hint: 'by date' },
]

type SearchShortcut = { view: string; openClass: string; search: string }

const TRI_SEARCH_SHORTCUTS: SearchShortcut[] = [
  { view: 'analytics', openClass: 'tri-analytics-open', search: '.tri-analytics .tri-ana-search' },
  { view: 'maps', openClass: 'tri-map-open', search: '.tri-map .tri-map-search' },
  {
    view: 'training',
    openClass: 'tri-training-open',
    search: '.tri-training .tri-training-search',
  },
]

const isEditable = (el: HTMLElement): boolean => {
  const tag = el.tagName.toLowerCase()
  return (
    tag === 'input' ||
    tag === 'textarea' ||
    tag === 'select' ||
    el.isContentEditable ||
    el.closest('.search-container') !== null
  )
}

const currentSearchShortcut = (root: HTMLElement): SearchShortcut | undefined => {
  const subView = root.dataset.triView
  if (subView) return TRI_SEARCH_SHORTCUTS.find(shortcut => shortcut.view === subView)
  return TRI_SEARCH_SHORTCUTS.find(shortcut => root.classList.contains(shortcut.openClass))
}

const toggleSearchFocus = (root: HTMLElement, target: HTMLElement | null): boolean => {
  const shortcut = currentSearchShortcut(root)
  if (!shortcut) return false
  const search = root.querySelector<HTMLInputElement>(shortcut.search)
  if (!search) return false
  if (target && isEditable(target) && target !== search) return false
  if (document.activeElement === search) search.blur()
  else {
    search.focus()
    search.select()
  }
  return true
}

const setupCommandPalette = (root: HTMLElement): (() => void) => {
  const overlay = el('div', 'tri-cmdk', undefined, { 'aria-hidden': 'true' })
  const box = el('div', 'tri-cmdk-box', undefined, {
    role: 'dialog',
    'aria-label': 'command palette',
  })
  const input = el('input', 'tri-cmdk-input', undefined, {
    type: 'text',
    placeholder: 'go to page · toggle units...',
    'aria-label': 'command',
    autocomplete: 'off',
    spellcheck: 'false',
  }) as HTMLInputElement
  const list = el('div', 'tri-cmdk-list', undefined, { role: 'listbox' })
  box.append(input, list)
  overlay.appendChild(box)
  root.appendChild(overlay)

  interface Cmd {
    label: () => string
    hint: string
    keys: string
    run: () => void
  }
  const navTo = (path: string) => (): void => {
    close()
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }
  const cmds: Cmd[] = [
    ...TRI_PAGES.map(p => ({
      label: () => `${p.label}`,
      hint: p.hint,
      keys: `go ${p.label} ${p.path}`,
      run: navTo(p.path),
    })),
    {
      label: () => `${isImperialUnit() ? 'imperial → metric' : 'metric → imperial'}`,
      hint: 'units',
      keys: 'toggle units km mi miles kg lb imperial metric pace distance speed weight',
      run: () => {
        toggleTriUnit()
        render()
      },
    },
    {
      label: () => (triLocale() === 'fr' ? 'langue · english' : 'language · français'),
      hint: 'locale',
      keys: 'language langue locale english french francais français en fr i18n',
      run: () => {
        applyTriLocale(triLocale() === 'fr' ? 'en' : 'fr')
        render()
      },
    },
    {
      label: () => {
        const next = nextTriMapStyle()
        return `map style · ${next === 'mono' ? 'monochrome' : next}`
      },
      hint: 'map',
      keys: 'map style roads streets monochrome mono satellite imagery mapbox route road',
      run: () => {
        toggleTriMapStyle()
        render()
      },
    },
  ]

  let items: Cmd[] = cmds
  let sel = 0
  let isOpen = false

  const paint = (): void => {
    const rows = list.querySelectorAll<HTMLElement>('.tri-cmdk-row')
    rows.forEach((r, i) => {
      r.classList.toggle('tri-cmdk-row--on', i === sel)
      r.setAttribute('aria-selected', String(i === sel))
    })
    rows[sel]?.scrollIntoView({ block: 'nearest' })
  }
  const render = (): void => {
    const q = input.value.trim().toLowerCase()
    items = q
      ? cmds.filter(c => `${c.label()} ${c.hint} ${c.keys}`.toLowerCase().includes(q))
      : cmds
    if (sel >= items.length) sel = Math.max(0, items.length - 1)
    list.replaceChildren(
      ...items.map((c, i) => {
        const row = el(
          'div',
          i === sel ? 'tri-cmdk-row tri-cmdk-row--on' : 'tri-cmdk-row',
          undefined,
          { role: 'option', 'aria-selected': String(i === sel) },
        )
        row.append(
          el('span', 'tri-cmdk-row-label', c.label()),
          el('span', 'tri-cmdk-row-hint', c.hint),
        )
        row.addEventListener('mousemove', () => {
          if (sel !== i) {
            sel = i
            paint()
          }
        })
        row.addEventListener('click', () => c.run())
        return row
      }),
    )
    if (!items.length) list.appendChild(el('div', 'tri-cmdk-empty', tl('no commands')))
  }
  const openPalette = (): void => {
    if (isOpen) return
    isOpen = true
    input.value = ''
    sel = 0
    render()
    overlay.classList.add('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'false')
    input.focus()
  }
  function close(): void {
    if (!isOpen) return
    isOpen = false
    overlay.classList.remove('tri-cmdk--on')
    overlay.setAttribute('aria-hidden', 'true')
    input.blur()
  }

  const onInput = (): void => {
    sel = 0
    render()
  }
  const onInputKey = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    } else if (e.key === 'Enter') {
      e.preventDefault()
      items[sel]?.run()
    } else if (e.key === 'ArrowDown' || (e.ctrlKey && e.key.toLowerCase() === 'n')) {
      e.preventDefault()
      if (items.length) sel = (sel + 1) % items.length
      paint()
    } else if (e.key === 'ArrowUp' || (e.ctrlKey && e.key.toLowerCase() === 'p')) {
      e.preventDefault()
      if (items.length) sel = (sel - 1 + items.length) % items.length
      paint()
    }
  }
  const onDocKey = (e: KeyboardEvent): void => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      e.stopImmediatePropagation()
      if (toggleSearchFocus(root, null) || currentSearchShortcut(root)) return
      if (root.matches('.tri-analytics-open, .tri-map-open, .tri-training-open, .tri-calc-open'))
        return
      if (isOpen) close()
      else openPalette()
    }
  }
  const onScrim = (e: MouseEvent): void => {
    if (e.target === overlay) close()
  }
  input.addEventListener('input', onInput)
  input.addEventListener('keydown', onInputKey)
  overlay.addEventListener('mousedown', onScrim)
  document.addEventListener('keydown', onDocKey, true)
  return () => {
    document.removeEventListener('keydown', onDocKey, true)
    overlay.remove()
  }
}

const setupShortcuts = (root: HTMLElement): (() => void) => {
  let waitingForG = false
  let gTimeout: number | null = null

  const clearG = (): void => {
    waitingForG = false
    if (gTimeout) {
      clearTimeout(gTimeout)
      gTimeout = null
    }
  }
  const go = (path: string) => {
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }
  const subView = root.dataset.triView
  const subpageNav: Record<string, string> = {
    g: '/triathlon/tools',
    c: '/triathlon/calc',
    a: '/triathlon/analytics',
    m: '/triathlon/maps',
    t: '/triathlon/training',
    r: '/triathlon/training',
    f: '/triathlon/feed',
    o: '/triathlon/on',
    h: '/triathlon',
  }

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
  const runChord = (key: string): boolean => {
    if (subView) {
      const path = subpageNav[key]
      if (!path) return false
      go(path)
      return true
    }
    if (key === 'a' || key === 'c' || key === 'm' || key === 't') {
      toggleModal(key)
      return true
    }
    if (key === 'g') {
      closeOpenModals()
      root.querySelector<HTMLElement>('.tri-gear-btn')?.click()
      return true
    }
    if (key === 'p') {
      closeOpenModals()
      root.querySelector<HTMLElement>('.tri-pace-btn')?.click()
      return true
    }
    if (key === 's') {
      root.querySelector<HTMLElement>('.tri-total')?.click()
      return true
    }
    if (key === 'h') {
      go('/')
      return true
    }
    if (key === 'f') {
      root.querySelector<HTMLElement>('.tri-fuel-btn')?.click()
      return true
    }
    return false
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.shiftKey && (e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'g') {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      go('/triathlon/tools')
      return
    }

    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key === '\\') {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      runChord('h')
      return
    }

    const el = e.target instanceof HTMLElement ? e.target : null
    if (
      e.key === '/' &&
      !e.shiftKey &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey &&
      toggleSearchFocus(root, el)
    ) {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      return
    }

    if (el && isEditable(el)) {
      clearG()
      return
    }

    if (e.ctrlKey || e.metaKey || e.altKey) {
      clearG()
      return
    }

    if (waitingForG) {
      if (runChord(e.key.toLowerCase())) {
        e.preventDefault()
        e.stopImmediatePropagation()
      }
      clearG()
    } else if (e.key.toLowerCase() === 'g') {
      waitingForG = true
      gTimeout = window.setTimeout(() => {
        waitingForG = false
        gTimeout = null
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
  const sync = () => {
    const mph = isImperialUnit()
    for (const b of buttons) b.textContent = mph ? 'mph' : 'km/h'
    for (const c of cells) c.textContent = (mph ? c.dataset.mph : c.dataset.kph) ?? ''
  }
  const onClick = () => toggleTriUnit()
  for (const b of buttons) b.addEventListener('click', onClick)
  window.addEventListener('tri:unit', sync)
  sync()
  return () => {
    for (const b of buttons) b.removeEventListener('click', onClick)
    window.removeEventListener('tri:unit', sync)
  }
}

const FEED_SORTS: Record<string, (a: ActivitySummary, b: ActivitySummary) => number> = {
  date: (a, b) => b.date.localeCompare(a.date),
  distance: (a, b) => b.distanceKm - a.distanceKm,
  pace: (a, b) =>
    (b.movingTimeS > 0 ? b.distanceKm / b.movingTimeS : 0) -
    (a.movingTimeS > 0 ? a.distanceKm / a.movingTimeS : 0),
}

const setupFeed = (root: HTMLElement): (() => void) | null => {
  if (root.dataset.triView !== 'feed') return null
  const list = root.querySelector<HTMLElement>('.tri-feed-list')
  const search = root.querySelector<HTMLInputElement>('.tri-feed-search')
  const countEl = root.querySelector<HTMLElement>('.tri-feed-count')
  const analyticsPath = root.dataset.analyticsPath
  const detailPath = root.dataset.detailPath
  const datePrefix = root.dataset.feedPrefix ?? ''
  if (!list || !analyticsPath) return null

  let acts: ActivitySummary[] = []
  let openId: string | null = null
  const detailCache = new Map<string, HTMLElement>()

  const buildSub = (a: ActivitySummary): HTMLElement => {
    const sub = el('span', 'tri-feed-sub')
    const cell = (cls: string, val: string): void => {
      sub.appendChild(el('span', `tri-feed-c ${cls}`, val || '-'))
    }
    cell('tri-feed-c--date', a.date)
    cell('tri-feed-c--dist', a.distanceKm > 0 ? dist(a.distanceKm, a.sport) : '')
    cell('tri-feed-c--time', a.movingTimeS > 0 ? dur(a.movingTimeS) : '')
    cell(
      'tri-feed-c--pace',
      a.distanceKm > 0 && a.movingTimeS > 0 ? rate(a.sport, a.distanceKm, a.movingTimeS) : '',
    )
    return sub
  }

  const buildDetail = (id: string): HTMLElement => {
    const cached = detailCache.get(id)
    if (cached) return cached
    const wrap = el('div', 'tri-feed-detail')
    wrap.appendChild(el('div', 'tri-ana-empty', tl('loading…')))
    detailCache.set(id, wrap)
    if (detailPath)
      void loadDetailPayload(detailPath).then(payload => {
        const d = payload?.details?.[id]
        if (!d) {
          wrap.replaceChildren(el('div', 'tri-ana-empty', tl('no detail')))
          return
        }
        const act = renderDetail(d, payload)
        setActivityExpanded(act, true)
        wrap.replaceChildren(act)
        const h = payload?.health?.[d.date]
        if (h) {
          const rec = buildRecovery(h)
          if (rec) wrap.appendChild(rec)
        }
      })
    return wrap
  }

  const collapse = () => {
    if (openId == null) return
    const row = list.querySelector<HTMLElement>(`.tri-feed-row[data-id="${openId}"]`)
    row?.querySelector('.tri-feed-detail')?.remove()
    row?.querySelector('.tri-feed-head')?.setAttribute('aria-expanded', 'false')
    row?.classList.remove('tri-feed-row--open')
    openId = null
  }

  const expand = (id: string) => {
    if (openId === id) {
      collapse()
      return
    }
    collapse()
    const row = list.querySelector<HTMLElement>(`.tri-feed-row[data-id="${id}"]`)
    if (!row) return
    openId = id
    row.classList.add('tri-feed-row--open')
    row.querySelector('.tri-feed-head')?.setAttribute('aria-expanded', 'true')
    row.appendChild(buildDetail(id))
  }

  const renderList = () => {
    const q = (search?.value ?? '').trim().toLowerCase()
    const sortKey = /sort:(distance|pace|date)/.exec(q)?.[1] ?? 'date'
    const term = q
      .replace(/sort:\w+/g, '')
      .replace(/filter:/g, '')
      .trim()
    const filtered = (
      term
        ? acts.filter(a => `${a.sport} ${a.name} ${a.date}`.toLowerCase().includes(term))
        : acts.slice()
    ).sort(FEED_SORTS[sortKey] ?? FEED_SORTS.date)
    list.replaceChildren(
      ...filtered.map(a => {
        const row = el('div', 'tri-feed-row', undefined, { role: 'listitem' })
        row.dataset.id = String(a.id)
        const head = el('button', 'tri-feed-head', undefined, {
          type: 'button',
          'aria-expanded': 'false',
        })
        head.append(buildIcon(a.sport), el('span', 'tri-feed-name', a.name || a.sport), buildSub(a))
        row.appendChild(head)
        return row
      }),
    )
    if (!filtered.length) list.appendChild(el('div', 'tri-ana-empty', tl('no activities')))
    if (countEl) countEl.textContent = String(filtered.length)
    list.setAttribute('aria-busy', 'false')
  }

  const onListClick = (e: MouseEvent) => {
    const head = (e.target as HTMLElement).closest<HTMLElement>('.tri-feed-head')
    const id = head?.closest<HTMLElement>('.tri-feed-row')?.dataset.id
    if (id) expand(id)
  }
  const onSearch = () => {
    collapse()
    renderList()
  }
  const onUnit = () => {
    detailCache.clear()
    const reopen = openId
    openId = null
    renderList()
    if (reopen) expand(reopen)
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && openId != null) collapse()
  }

  const marquee = marqueeCtl()
  const onOver = (e: MouseEvent) => {
    const name = (e.target as HTMLElement)
      .closest<HTMLElement>('.tri-feed-head')
      ?.querySelector<HTMLElement>('.tri-feed-name')
    if (name) marquee.run(name)
  }
  const onOut = (e: MouseEvent) => {
    const head = (e.target as HTMLElement).closest<HTMLElement>('.tri-feed-head')
    const to = e.relatedTarget as Node | null
    if (head && to && head.contains(to)) return
    marquee.stop()
  }

  list.addEventListener('click', onListClick)
  list.addEventListener('mouseover', onOver)
  list.addEventListener('mouseout', onOut)
  search?.addEventListener('input', onSearch)
  window.addEventListener('tri:unit', onUnit)
  document.addEventListener('keydown', onKey)

  fetch(analyticsPath)
    .then(res => res.json())
    .then((d: Analytics) => {
      acts = (d.activities ?? []).filter(activity => activity.date.startsWith(datePrefix))
      renderList()
    })
    .catch(() => {
      list.setAttribute('aria-busy', 'false')
      list.replaceChildren(el('div', 'tri-ana-empty', tl('no data')))
    })

  return () => {
    marquee.stop()
    list.removeEventListener('click', onListClick)
    list.removeEventListener('mouseover', onOver)
    list.removeEventListener('mouseout', onOut)
    search?.removeEventListener('input', onSearch)
    window.removeEventListener('tri:unit', onUnit)
    document.removeEventListener('keydown', onKey)
  }
}

window.quartzTriathlon = {
  dayCard: async (date, detailPath, extras) => {
    const data = await loadDetailPayload(detailPath)
    if (!data) return null
    return buildDayCard(date, data, extras ?? {})
  },
}

const setupI18n = (root: HTMLElement): (() => void) => {
  const apply = (): void => applyI18n(root)
  apply()
  window.addEventListener('tri:locale', apply)
  return () => window.removeEventListener('tri:locale', apply)
}

const setupChartScrub = (scope: HTMLElement): (() => void) => {
  type CurveRange = 'six-weeks' | 'year'
  let activeWrap: HTMLElement | null = null
  let activeBar: Element | null = null
  let focusedSvg: SVGSVGElement | null = null
  const curveCache = new WeakMap<
    SVGSVGElement,
    { curve: PowerCurvePoint[]; sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] }
  >()
  const swimCache = new WeakMap<
    SVGSVGElement,
    { lengths: SwimTrendChartPoint[]; '100m': SwimTrendChartPoint[] }
  >()
  const swimAnimations = new Map<SVGGElement, Animation>()
  const curveData = (
    svg: SVGSVGElement,
  ): { curve: PowerCurvePoint[]; sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] } => {
    const cached = curveCache.get(svg)
    if (cached) return cached
    const value = {
      curve: decodePowerCurve(svg.dataset.curve),
      sixWeeks: decodePowerCurve(svg.dataset.curveRefSixWeeks),
      year: decodePowerCurve(svg.dataset.curveRefYear),
    }
    curveCache.set(svg, value)
    return value
  }
  const isSwimTrendPoint = (value: unknown): value is SwimTrendChartPoint =>
    typeof value === 'object' &&
    value !== null &&
    'elapsedS' in value &&
    typeof value.elapsedS === 'number' &&
    Number.isFinite(value.elapsedS) &&
    'cumulativeDistanceM' in value &&
    typeof value.cumulativeDistanceM === 'number' &&
    Number.isFinite(value.cumulativeDistanceM) &&
    'value' in value &&
    typeof value.value === 'number' &&
    Number.isFinite(value.value) &&
    'xPct' in value &&
    typeof value.xPct === 'number' &&
    Number.isFinite(value.xPct) &&
    'yPct' in value &&
    typeof value.yPct === 'number' &&
    Number.isFinite(value.yPct) &&
    (!('windowStartDistanceM' in value) ||
      (typeof value.windowStartDistanceM === 'number' &&
        Number.isFinite(value.windowStartDistanceM)))
  const swimMode = (svg: SVGSVGElement): SwimTrendMode =>
    svg.dataset.swimMode === '100m' ? '100m' : 'lengths'
  const decodeSwimData = (value: string | undefined): SwimTrendChartPoint[] => {
    const parsed: unknown = JSON.parse(value ?? '[]')
    return Array.isArray(parsed) ? parsed.filter(isSwimTrendPoint) : []
  }
  const swimData = (
    svg: SVGSVGElement,
    mode: SwimTrendMode = swimMode(svg),
  ): SwimTrendChartPoint[] => {
    const cached = swimCache.get(svg)
    if (cached) return cached[mode]
    const value = {
      lengths: decodeSwimData(svg.dataset.swimSeriesLengths),
      '100m': decodeSwimData(svg.dataset.swimSeriesHundred),
    }
    swimCache.set(svg, value)
    return value[mode]
  }
  const curveRange = (svg: SVGSVGElement): CurveRange =>
    svg.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
  const curveReference = (
    svg: SVGSVGElement,
    data: { sixWeeks: PowerCurvePoint[]; year: PowerCurvePoint[] },
  ): PowerCurvePoint[] => (curveRange(svg) === 'year' ? data.year : data.sixWeeks)
  const curveReferenceYear = (svg: SVGSVGElement): number | null => {
    if (curveRange(svg) !== 'year') return null
    const year = Number(svg.dataset.curveYear)
    return Number.isInteger(year) ? year : null
  }
  const selectedCurveIndex = (svg: SVGSVGElement): number => {
    const value = Number(svg.dataset.curveIndex ?? 0)
    return Number.isInteger(value) ? value : 0
  }
  const selectedSwimIndex = (svg: SVGSVGElement): number => {
    const value = Number(svg.dataset.swimIndex ?? 0)
    return Number.isInteger(value) ? value : 0
  }
  const curveValueText = (
    svg: SVGSVGElement,
    point: PowerCurvePoint,
    referenceWatts: number | null,
  ): string =>
    `${zoneClock(point.s)}, ${tl('this ride')} ${point.w.toLocaleString()} watts${referenceWatts == null ? '' : `, ${powerCurveReferenceLabel(curveReferenceYear(svg))} ${referenceWatts.toLocaleString()} watts`}`
  const swimKind = (svg: SVGSVGElement): 'pace' | 'stroke' =>
    svg.dataset.swimKind === 'stroke' ? 'stroke' : 'pace'
  const swimAriaLabel = (svg: SVGSVGElement): string =>
    `${tl('swim')} ${tl(swimKind(svg) === 'pace' ? 'pace' : 'stroke rate')} · ${tl(swimMode(svg) === '100m' ? '100 m' : 'lengths')}`
  const swimDisplayValue = (kind: 'pace' | 'stroke', value: number): string =>
    swimActivityDisplayValue(kind, value, clock(value))
  const swimTextPoint = (point: SwimTrendChartPoint) => ({
    elapsed: clock(point.elapsedS),
    cumulativeDistanceM: point.cumulativeDistanceM,
    ...(point.windowStartDistanceM == null
      ? {}
      : { windowStartDistanceM: point.windowStartDistanceM }),
  })
  const swimValueText = (kind: 'pace' | 'stroke', point: SwimTrendChartPoint): string =>
    swimActivityValueText(kind, swimTextPoint(point), point.value, clock(point.value))
  const deactivate = (wrap: HTMLElement): void => {
    wrap.classList.remove('tri-chart--hover')
    const point = wrap.querySelector<HTMLElement>('.tri-swim-trend-hover')
    if (point) point.hidden = true
  }
  const activate = (wrap: HTMLElement): void => {
    if (activeWrap && activeWrap !== wrap) deactivate(activeWrap)
    activeWrap = wrap
    wrap.classList.add('tri-chart--hover')
  }
  const clear = (): void => {
    if (activeWrap) deactivate(activeWrap)
    activeBar?.classList.remove('tri-hist-bar--on')
    activeWrap = null
    activeBar = null
  }
  const placeCurvePoint = (
    point: HTMLElement | null,
    x: number,
    watts: number | null,
    maxWatts: number,
    height: number,
  ): void => {
    if (!point) return
    point.hidden = watts == null
    if (watts == null) return
    const y = height - (Math.min(maxWatts, Math.max(0, watts)) / maxWatts) * (height - 1)
    point.style.left = `${x.toFixed(2)}%`
    point.style.top = `${((y / height) * 100).toFixed(2)}%`
  }
  const showCurve = (svg: SVGSVGElement, fraction: number, activateChart = true): void => {
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const readout = wrap?.querySelector<HTMLElement>('.tri-curve-readout')
    if (!wrap || !readout) return
    const data = curveData(svg)
    const curve = data.curve
    const reference = curveReference(svg, data)
    const hover = powerCurveHoverAt(curve, reference, fraction)
    if (!hover) return
    if (svg.dataset.curveIndex !== String(hover.index)) {
      cursor?.setAttribute('x1', hover.xPct.toFixed(2))
      cursor?.setAttribute('x2', hover.xPct.toFixed(2))
      const duration = readout.querySelector<HTMLElement>('.tri-curve-readout-duration')
      const ride = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ride')
      const referenceRow = readout.querySelector<HTMLElement>('.tri-curve-readout-row--ref')
      const referenceValue = readout.querySelector<HTMLElement>('.tri-curve-readout-value--ref')
      const referenceLabel = readout.querySelector<HTMLElement>('.tri-curve-readout-label--ref')
      const maxWatts = Number(svg.dataset.curveDomainMax)
      const height = svg.viewBox.baseVal.height
      if (Number.isFinite(maxWatts) && maxWatts > 0 && height > 0) {
        placeCurvePoint(
          wrap.querySelector<HTMLElement>('.tri-curve-point--ride'),
          hover.xPct,
          hover.watts,
          maxWatts,
          height,
        )
        placeCurvePoint(
          wrap.querySelector<HTMLElement>('.tri-curve-point--ref'),
          hover.xPct,
          hover.referenceWatts,
          maxWatts,
          height,
        )
      }
      if (duration) duration.textContent = zoneClock(hover.durationS)
      if (ride) ride.textContent = `${hover.watts.toLocaleString()} W`
      if (referenceRow) referenceRow.hidden = hover.referenceWatts == null
      if (referenceValue && hover.referenceWatts != null)
        referenceValue.textContent = `${hover.referenceWatts.toLocaleString()} W`
      if (referenceLabel)
        referenceLabel.textContent = powerCurveReferenceLabel(curveReferenceYear(svg))
      svg.dataset.curveIndex = String(hover.index)
      svg.setAttribute('aria-valuenow', String(hover.durationS))
      svg.setAttribute(
        'aria-valuetext',
        curveValueText(svg, { s: hover.durationS, w: hover.watts }, hover.referenceWatts),
      )
    }
    if (activateChart) {
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = null
      activate(wrap)
    }
  }
  const showCurveIndex = (
    svg: SVGSVGElement,
    requestedIndex: number,
    activateChart = true,
  ): void => {
    const { curve } = curveData(svg)
    if (curve.length < 2) return
    const index = Math.min(curve.length - 1, Math.max(0, requestedIndex))
    showCurve(
      svg,
      powerCurveFraction(curve[index].s, curve[0].s, curve[curve.length - 1].s),
      activateChart,
    )
  }
  const showSwim = (svg: SVGSVGElement, fraction: number, activateChart = true): void => {
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const point = wrap?.querySelector<HTMLElement>('.tri-swim-trend-hover')
    const readoutPosition = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-position')
    const readoutValue = wrap?.querySelector<HTMLElement>('.tri-swim-trend-readout-value')
    if (!wrap || !point || !readoutPosition || !readoutValue) return
    const kind = swimKind(svg)
    const hover = swimTrendHoverAt(swimData(svg), fraction)
    if (!hover) return
    cursor?.setAttribute('x1', hover.xPct.toFixed(2))
    cursor?.setAttribute('x2', hover.xPct.toFixed(2))
    point.style.left = `${hover.xPct.toFixed(2)}%`
    point.style.top = `${hover.yPct.toFixed(2)}%`
    point.hidden = !activateChart
    readoutPosition.textContent = swimActivityPointText(swimTextPoint(hover))
    readoutValue.textContent = swimDisplayValue(kind, hover.value)
    svg.dataset.swimIndex = String(hover.index)
    svg.setAttribute('aria-label', swimAriaLabel(svg))
    svg.setAttribute('aria-valuenow', String(Math.round(hover.cumulativeDistanceM)))
    svg.setAttribute('aria-valuetext', swimValueText(kind, hover))
    if (activateChart) {
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = null
      activate(wrap)
    }
  }
  const showSwimIndex = (
    svg: SVGSVGElement,
    requestedIndex: number,
    activateChart = true,
  ): void => {
    const points = swimData(svg)
    if (points.length === 0) return
    const index = Math.min(points.length - 1, Math.max(0, requestedIndex))
    showSwim(svg, points[index].xPct / 100, activateChart)
  }
  const onSwimRestore = (event: Event): void => {
    if (
      !(event.target instanceof SVGSVGElement) ||
      !event.target.classList.contains('tri-swim-trend-svg')
    )
      return
    const svg = event.target
    const distanceM = Number(svg.dataset.swimRestoreDistance)
    const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
    const activateChart = svg.dataset.swimRestoreActive === 'true'
    delete svg.dataset.swimRestoreDistance
    delete svg.dataset.swimRestoreActive
    if (!Number.isFinite(distanceM) || !Number.isFinite(totalDistanceM) || totalDistanceM <= 0)
      return
    showSwim(svg, distanceM / totalDistanceM, activateChart)
  }
  const showFocused = (): void => {
    if (!focusedSvg) {
      clear()
      return
    }
    if (focusedSvg.classList.contains('tri-curve-svg'))
      showCurveIndex(focusedSvg, selectedCurveIndex(focusedSvg))
    else showSwimIndex(focusedSvg, selectedSwimIndex(focusedSvg))
  }
  const onPointer = (event: PointerEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>(
      '.tri-curve-svg, .tri-hist-svg, .tri-swim-trend-svg',
    )
    if (!svg) {
      if (activeWrap) showFocused()
      return
    }
    const wrap = svg.closest<HTMLElement>('.tri-zone')
    const cursor = svg.querySelector<SVGElement>('.tri-chart-cursor')
    const readout = wrap?.querySelector<HTMLElement>('.tri-chart-readout')
    const r = svg.getBoundingClientRect()
    const frac = r.width > 0 ? Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) : 0
    if (svg.classList.contains('tri-curve-svg')) {
      showCurve(svg, frac)
      return
    }
    if (svg.classList.contains('tri-swim-trend-svg')) {
      showSwim(svg, frac)
      return
    } else {
      const hist = JSON.parse(svg.dataset.hist ?? '[]') as number[]
      const n = hist.length
      if (n < 2) return
      const total = hist.reduce((a, b) => a + b, 0) || 1
      const bin = Math.max(0, Math.min(n - 1, Math.floor(frac * n)))
      cursor?.setAttribute('x1', `${bin + 0.5}`)
      cursor?.setAttribute('x2', `${bin + 0.5}`)
      activeBar?.classList.remove('tri-hist-bar--on')
      activeBar = svg.querySelector(`.tri-hist-bar[data-bin="${bin}"]`)
      activeBar?.classList.add('tri-hist-bar--on')
      if (readout)
        readout.textContent = `${bin * 25}–${bin * 25 + 24} W · ${zoneClock(hist[bin])} (${((hist[bin] / total) * 100).toFixed(1)}%)`
    }
    if (wrap) activate(wrap)
  }
  const onFocus = (event: FocusEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    focusedSvg = svg
    if (svg.classList.contains('tri-curve-svg')) showCurveIndex(svg, selectedCurveIndex(svg))
    else showSwimIndex(svg, selectedSwimIndex(svg))
  }
  const onBlur = (event: FocusEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    if (focusedSvg === svg) focusedSvg = null
    clear()
  }
  const onKey = (event: KeyboardEvent): void => {
    if (!(event.target instanceof Element)) return
    const svg = event.target.closest<SVGSVGElement>('.tri-curve-svg, .tri-swim-trend-svg')
    if (!svg) return
    const isCurve = svg.classList.contains('tri-curve-svg')
    const length = isCurve ? curveData(svg).curve.length : swimData(svg).length
    if (length < 2) return
    const current = isCurve ? selectedCurveIndex(svg) : selectedSwimIndex(svg)
    let next: number | null = null
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = current - 1
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = current + 1
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = length - 1
    else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      focusedSvg = null
      svg.blur()
      clear()
      return
    }
    if (next == null) return
    event.preventDefault()
    focusedSvg = svg
    if (isCurve) showCurveIndex(svg, next)
    else showSwimIndex(svg, next)
  }
  const setSwimLayer = (svg: SVGSVGElement, mode: SwimTrendMode, animate: boolean): void => {
    const previousMode = swimMode(svg)
    if (previousMode === mode) return
    const previous = svg.querySelector<SVGGElement>(
      `.tri-swim-series[data-swim-mode="${previousMode}"]`,
    )
    const next = svg.querySelector<SVGGElement>(`.tri-swim-series[data-swim-mode="${mode}"]`)
    if (!previous || !next) return
    const previousOpacity = getComputedStyle(previous).opacity
    const nextOpacity = getComputedStyle(next).opacity
    swimAnimations.get(previous)?.cancel()
    swimAnimations.get(next)?.cancel()
    swimAnimations.delete(previous)
    swimAnimations.delete(next)
    previous.classList.remove('tri-swim-series--active')
    previous.setAttribute('aria-hidden', 'true')
    next.classList.add('tri-swim-series--active')
    next.setAttribute('aria-hidden', 'false')
    svg.dataset.swimMode = mode
    if (!animate || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timing: KeyframeAnimationOptions = {
      duration: 180,
      easing: 'cubic-bezier(0.77, 0, 0.175, 1)',
    }
    swimAnimations.set(
      previous,
      previous.animate([{ opacity: previousOpacity }, { opacity: 0 }], timing),
    )
    swimAnimations.set(next, next.animate([{ opacity: nextOpacity }, { opacity: 1 }], timing))
  }
  const setSwimMode = (section: HTMLElement, mode: SwimTrendMode, animate: boolean): void => {
    const toggle = section.querySelector<HTMLElement>('.tri-swim-mode-toggle')
    if (!toggle || toggle.dataset.swimMode === mode) return
    toggle.dataset.swimMode = mode
    for (const option of toggle.querySelectorAll<HTMLButtonElement>('.tri-swim-mode'))
      option.setAttribute('aria-pressed', String(option.dataset.swimMode === mode))
    for (const svg of section.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
      const previous = swimData(svg)
      const selected = Math.min(previous.length - 1, Math.max(0, selectedSwimIndex(svg)))
      const fraction = previous[selected]?.xPct != null ? previous[selected].xPct / 100 : 1
      const wrap = svg.closest<HTMLElement>('.tri-zone')
      const wasActive = wrap?.classList.contains('tri-chart--hover') ?? false
      setSwimLayer(svg, mode, animate)
      delete svg.dataset.swimIndex
      showSwim(svg, fraction, wasActive)
    }
  }
  const onChartClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const swimButton = event.target.closest<HTMLButtonElement>('.tri-swim-mode')
    const swimSection = swimButton?.closest<HTMLElement>('.tri-swim-trends')
    if (swimButton && swimSection) {
      const mode: SwimTrendMode = swimButton.dataset.swimMode === '100m' ? '100m' : 'lengths'
      setSwimMode(swimSection, mode, event.detail > 0)
      return
    }
    const button = event.target.closest<HTMLButtonElement>('.tri-curve-range')
    const wrap = button?.closest<HTMLElement>('.tri-zone')
    const svg = wrap?.querySelector<SVGSVGElement>('.tri-curve-svg')
    if (!button || button.disabled || !wrap || !svg) return
    const range: CurveRange = button.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
    const data = curveData(svg)
    const reference = range === 'year' ? data.year : data.sixWeeks
    if (reference.length === 0) return
    const index = selectedCurveIndex(svg)
    const wasActive = wrap.classList.contains('tri-chart--hover')
    svg.dataset.curveRange = range
    for (const option of wrap.querySelectorAll<HTMLButtonElement>('.tri-curve-range'))
      option.setAttribute('aria-pressed', String(option.dataset.curveRange === range))
    for (const path of svg.querySelectorAll<SVGElement>('.tri-curve-ref[data-curve-range]'))
      path.toggleAttribute('hidden', path.dataset.curveRange !== range)
    delete svg.dataset.curveIndex
    showCurveIndex(svg, index, wasActive)
  }
  const onLocale = (): void => {
    for (const delta of scope.querySelectorAll<HTMLElement>('.tri-swim-trend-delta')) {
      const kind = delta.dataset.swimComparisonKind === 'stroke' ? 'stroke' : 'pace'
      const rawDelta = delta.dataset.swimComparisonDelta
      const rawPrior = delta.dataset.swimComparisonPrior
      const comparisonDelta = rawDelta == null ? null : Number(rawDelta)
      const comparisonPrior = rawPrior == null ? null : Number(rawPrior)
      delta.textContent = swimActivityComparisonText(
        kind,
        comparisonDelta != null && Number.isFinite(comparisonDelta) ? comparisonDelta : null,
        comparisonPrior != null && Number.isInteger(comparisonPrior) ? comparisonPrior : null,
      )
    }
    for (const average of scope.querySelectorAll<HTMLElement>('.tri-swim-trend-value')) {
      const kind = average.dataset.swimAverageKind === 'stroke' ? 'stroke' : 'pace'
      const value = Number(average.dataset.swimAverageValue)
      if (Number.isFinite(value))
        average.textContent = swimActivityHeaderValue(kind, value, clock(value))
    }
    for (const svg of scope.querySelectorAll<SVGSVGElement>('.tri-curve-svg')) {
      const data = curveData(svg)
      const curve = data.curve
      const reference = curveReference(svg, data)
      if (curve.length < 2) continue
      const index = Math.min(curve.length - 1, Math.max(0, selectedCurveIndex(svg)))
      const point = curve[index]
      const referenceWatts = reference.find(candidate => candidate.s === point.s)?.w ?? null
      const referenceLabel = svg
        .closest<HTMLElement>('.tri-zone')
        ?.querySelector<HTMLElement>('.tri-curve-readout-label--ref')
      if (referenceLabel)
        referenceLabel.textContent = powerCurveReferenceLabel(curveReferenceYear(svg))
      svg.setAttribute('aria-label', tl('power curve'))
      svg.setAttribute('aria-valuenow', String(point.s))
      svg.setAttribute('aria-valuetext', curveValueText(svg, point, referenceWatts))
    }
    for (const svg of scope.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
      svg.setAttribute('aria-label', swimAriaLabel(svg))
      const wrap = svg.closest<HTMLElement>('.tri-zone')
      const totalDistanceM = Number(svg.getAttribute('aria-valuemax'))
      if (wrap && Number.isFinite(totalDistanceM)) {
        const distances = [0, totalDistanceM / 2, totalDistanceM]
        wrap.querySelectorAll<HTMLElement>('.tri-cax-xt').forEach((tick, index) => {
          const distanceM = distances[index]
          if (distanceM != null) tick.textContent = swimActivityDistanceText(distanceM)
        })
      }
      showSwimIndex(
        svg,
        selectedSwimIndex(svg),
        wrap?.classList.contains('tri-chart--hover') ?? false,
      )
    }
  }
  const onPointerLeave = (): void => showFocused()
  scope.addEventListener('pointermove', onPointer)
  scope.addEventListener('pointerdown', onPointer)
  scope.addEventListener('pointerleave', onPointerLeave)
  scope.addEventListener('pointercancel', onPointerLeave)
  scope.addEventListener('focusin', onFocus)
  scope.addEventListener('focusout', onBlur)
  scope.addEventListener('keydown', onKey)
  scope.addEventListener('click', onChartClick)
  scope.addEventListener('tri:swim-restore', onSwimRestore)
  window.addEventListener('tri:locale', onLocale)
  onLocale()
  return () => {
    clear()
    for (const animation of swimAnimations.values()) animation.cancel()
    swimAnimations.clear()
    scope.removeEventListener('pointermove', onPointer)
    scope.removeEventListener('pointerdown', onPointer)
    scope.removeEventListener('pointerleave', onPointerLeave)
    scope.removeEventListener('pointercancel', onPointerLeave)
    scope.removeEventListener('focusin', onFocus)
    scope.removeEventListener('focusout', onBlur)
    scope.removeEventListener('keydown', onKey)
    scope.removeEventListener('click', onChartClick)
    scope.removeEventListener('tri:swim-restore', onSwimRestore)
    window.removeEventListener('tri:locale', onLocale)
  }
}

function normCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z))
  const d = 0.3989423 * Math.exp((-z * z) / 2)
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
  return z >= 0 ? 1 - p : p
}

let paceForecaster: PaceForecaster | null = null
let paceForecastUnavailable = false

const PRED_SPORTS: { sport: PaceSport; dists: { km: number; label: string }[] }[] = [
  {
    sport: 'swim',
    dists: [
      { km: 0.75, label: '750m' },
      { km: 1.5, label: '1.5K' },
      { km: 1.9, label: '1.9K' },
      { km: 3.8, label: '3.8K' },
    ],
  },
  {
    sport: 'bike',
    dists: [
      { km: 20, label: '20K' },
      { km: 40, label: '40K' },
      { km: 90, label: '90K' },
      { km: 180, label: '180K' },
    ],
  },
  {
    sport: 'run',
    dists: [
      { km: 5, label: '5K' },
      { km: 10, label: '10K' },
      { km: 21.0975, label: 'half' },
      { km: 42.195, label: 'marathon' },
    ],
  },
]

const PRED_COMPARE_OPTIONS = [
  { key: '7', label: '7d', days: 7 },
  { key: '14', label: '14d', days: 14 },
  { key: '30', label: '30d', days: 30 },
  { key: '60', label: '60d', days: 60 },
  { key: 'custom', label: 'custom', days: null },
] as const

type PredCompareKey = (typeof PRED_COMPARE_OPTIONS)[number]['key']

interface PredComparison {
  day: PaceDayState | null
  label: string
}

interface PredResult {
  card: HTMLElement
  nowSec: number
  fastSec: number
  slowSec: number
  delta: number | null
  compareLabel: string
}

const PRED_AXIS_FRACS = [0, 0.25, 0.5, 0.75, 1]
const PRED_DEFAULT_COMPARE: PredCompareKey = '7'
const PRED_CALENDAR_WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S']
const PRED_CALENDAR_MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
]
let predRunSeq = 0

interface PredDateParts {
  year: number
  month: number
  day: number
}

interface PredMonthParts {
  year: number
  month: number
}

interface PredDatePicker {
  wrap: HTMLElement
  trigger: HTMLButtonElement
  panel: HTMLElement
  render: () => void
  close: () => void
}

const predDatePad = (value: number): string => String(value).padStart(2, '0')

const predDateValue = (year: number, month: number, day: number): string =>
  `${year}-${predDatePad(month)}-${predDatePad(day)}`

const predDateFromLocal = (date: Date): string =>
  predDateValue(date.getFullYear(), date.getMonth() + 1, date.getDate())

const parsePredDate = (value: string | undefined): PredDateParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const date = new Date(year, month - 1, day)
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day)
    return null
  return { year, month, day }
}

const predMonthValue = (parts: PredMonthParts): string =>
  `${parts.year}-${predDatePad(parts.month)}`

const parsePredMonth = (value: string | undefined): PredMonthParts | null => {
  if (!value) return null
  const match = /^(\d{4})-(\d{2})$/.exec(value)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  if (month < 1 || month > 12) return null
  return { year, month }
}

const predMonthFromDate = (parts: PredDateParts): PredMonthParts => ({
  year: parts.year,
  month: parts.month,
})

const addPredMonths = (parts: PredMonthParts, delta: number): PredMonthParts => {
  const date = new Date(parts.year, parts.month - 1 + delta, 1)
  return { year: date.getFullYear(), month: date.getMonth() + 1 }
}

const predTodayParts = (): PredDateParts => {
  const date = new Date()
  return { year: date.getFullYear(), month: date.getMonth() + 1, day: date.getDate() }
}

const clampPredDate = (value: string, min: string | undefined, max: string | undefined): string => {
  if (min && value < min) return min
  if (max && value > max) return max
  return value
}

const predButton = (
  cls: string,
  text?: string,
  attrs?: Record<string, string>,
): HTMLButtonElement => {
  const button = document.createElement('button')
  button.className = cls
  button.type = 'button'
  if (text !== undefined) button.textContent = text
  if (attrs) for (const k in attrs) button.setAttribute(k, attrs[k])
  return button
}

const buildPredCalendarIcon = (): SVGElement => {
  const icon = svg('svg', {
    class: 'tri-pred-date-ico',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  icon.append(
    svg('path', {
      d: 'M4.5 2v2M11.5 2v2M3.5 5.5h9M4 3.5h8a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Z',
      stroke: 'currentColor',
      'stroke-width': '1.35',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  return icon
}

const buildPredCalendarArrow = (direction: -1 | 1): SVGElement => {
  const icon = svg('svg', {
    class: 'tri-pred-cal-arrow',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  icon.append(
    svg('path', {
      d: direction < 0 ? 'M10 3.5 5.5 8l4.5 4.5' : 'M6 3.5 10.5 8 6 12.5',
      stroke: 'currentColor',
      'stroke-width': '1.6',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  return icon
}

const positionPredCalendar = (trigger: HTMLElement, panel: HTMLElement): void => {
  const rect = trigger.getBoundingClientRect()
  const width = Math.min(Math.max(238, rect.width), window.innerWidth - 16)
  const height = 304
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))
  const below = rect.bottom + 6
  const top = below + height > window.innerHeight ? Math.max(8, rect.top - height - 6) : below
  panel.style.inlineSize = `${width}px`
  panel.style.insetInlineStart = `${left}px`
  panel.style.insetBlockStart = `${top}px`
}

const focusPredCalendarSelection = (panel: HTMLElement): void => {
  const selected =
    panel.querySelector<HTMLButtonElement>('.tri-pred-cal-day--selected:not(:disabled)') ??
    panel.querySelector<HTMLButtonElement>('.tri-pred-cal-day:not(:disabled)')
  selected?.focus()
}

const movePredCalendarFocus = (panel: HTMLElement, offset: number): void => {
  const active = document.activeElement
  if (!(active instanceof HTMLButtonElement) || !active.classList.contains('tri-pred-cal-day'))
    return
  const current = parsePredDate(active.dataset.date)
  if (!current) return
  const date = new Date(current.year, current.month - 1, current.day + offset)
  const next = clampPredDate(predDateFromLocal(date), panel.dataset.minDate, panel.dataset.maxDate)
  const nextParts = parsePredDate(next)
  if (!nextParts) return
  panel.dataset.viewMonth = predMonthValue(predMonthFromDate(nextParts))
  panel.dispatchEvent(new CustomEvent('tri:pred-date-render'))
  panel.querySelector<HTMLButtonElement>(`.tri-pred-cal-day[data-date="${next}"]`)?.focus()
}

const buildPredDatePicker = (
  block: HTMLElement,
  onOpen: () => void,
  onSelect: (date: string) => void,
  onClear: () => void,
): PredDatePicker => {
  const wrap = el('div', 'tri-pred-date-wrap')
  const trigger = predButton('tri-pred-date', undefined, {
    'aria-label': 'comparison date',
    'aria-haspopup': 'dialog',
    'aria-expanded': 'false',
  })
  const text = el('span', 'tri-pred-date-text')
  trigger.append(text, buildPredCalendarIcon())
  const panel = el('div', 'tri-pred-calendar', undefined, {
    role: 'dialog',
    'aria-label': 'comparison date picker',
    popover: 'auto',
  })
  panel.id = `tri-pred-calendar-${Math.random().toString(36).slice(2)}`
  panel.tabIndex = -1
  trigger.setAttribute('aria-controls', panel.id)
  wrap.append(trigger, panel)

  const close = (): void => {
    if (panel.matches(':popover-open') && typeof panel.hidePopover === 'function')
      panel.hidePopover()
    panel.removeAttribute('data-open')
    trigger.setAttribute('aria-expanded', 'false')
  }

  const render = (): void => {
    const min = block.dataset.compareMin
    const max = block.dataset.compareMax
    const today = predDateFromLocal(new Date())
    const selected =
      block.dataset.compareDate ??
      max ??
      min ??
      predDateValue(predTodayParts().year, predTodayParts().month, predTodayParts().day)
    const selectedParts =
      parsePredDate(selected) ?? parsePredDate(max) ?? parsePredDate(min) ?? predTodayParts()
    const view = parsePredMonth(panel.dataset.viewMonth) ?? predMonthFromDate(selectedParts)
    panel.dataset.viewMonth = predMonthValue(view)
    if (min) panel.dataset.minDate = min
    else delete panel.dataset.minDate
    if (max) panel.dataset.maxDate = max
    else delete panel.dataset.maxDate

    const minMonth = parsePredDate(min)
    const maxMonth = parsePredDate(max)
    const prevMonth = addPredMonths(view, -1)
    const nextMonth = addPredMonths(view, 1)
    const monthTitle = `${PRED_CALENDAR_MONTHS[view.month - 1]} ${view.year}`
    const head = el('div', 'tri-pred-cal-head')
    const title = el('span', 'tri-pred-cal-title', monthTitle)
    const prev = predButton('tri-pred-cal-nav', undefined, { 'aria-label': 'previous month' })
    const next = predButton('tri-pred-cal-nav', undefined, { 'aria-label': 'next month' })
    prev.appendChild(buildPredCalendarArrow(-1))
    next.appendChild(buildPredCalendarArrow(1))
    if (minMonth && predMonthValue(prevMonth) < predMonthValue(predMonthFromDate(minMonth)))
      prev.disabled = true
    if (maxMonth && predMonthValue(nextMonth) > predMonthValue(predMonthFromDate(maxMonth)))
      next.disabled = true
    prev.addEventListener('click', () => {
      panel.dataset.viewMonth = predMonthValue(prevMonth)
      render()
      focusPredCalendarSelection(panel)
    })
    next.addEventListener('click', () => {
      panel.dataset.viewMonth = predMonthValue(nextMonth)
      render()
      focusPredCalendarSelection(panel)
    })
    head.append(title, prev, next)

    const week = el('div', 'tri-pred-cal-week')
    for (const day of PRED_CALENDAR_WEEKDAYS)
      week.appendChild(el('span', 'tri-pred-cal-weekday', day))

    const grid = el('div', 'tri-pred-cal-grid')
    const monthStart = new Date(view.year, view.month - 1, 1)
    const gridStart = new Date(view.year, view.month - 1, 1 - monthStart.getDay())
    for (let i = 0; i < 42; i += 1) {
      const date = new Date(gridStart.getFullYear(), gridStart.getMonth(), gridStart.getDate() + i)
      const value = predDateFromLocal(date)
      const day = predButton('tri-pred-cal-day', String(date.getDate()), {
        'data-date': value,
        'aria-label': value,
      })
      if (date.getMonth() !== view.month - 1) day.classList.add('tri-pred-cal-day--muted')
      if (value === selected) {
        day.classList.add('tri-pred-cal-day--selected')
        day.setAttribute('aria-current', 'date')
      }
      if (value === today) day.classList.add('tri-pred-cal-day--today')
      if ((min && value < min) || (max && value > max)) day.disabled = true
      else
        day.addEventListener('click', () => {
          onSelect(value)
          close()
        })
      grid.appendChild(day)
    }

    const foot = el('div', 'tri-pred-cal-foot')
    const clear = predButton('tri-pred-cal-action', 'clear')
    const now = predButton('tri-pred-cal-action', 'today')
    clear.addEventListener('click', () => {
      onClear()
      close()
    })
    now.addEventListener('click', () => {
      onSelect(clampPredDate(today, min, max))
      close()
    })
    foot.append(clear, now)
    panel.replaceChildren(head, week, grid, foot)
  }

  trigger.addEventListener('click', () => {
    if (panel.matches(':popover-open') || panel.dataset.open === 'true') {
      close()
      return
    }
    onOpen()
    const selected = parsePredDate(block.dataset.compareDate)
    if (selected) panel.dataset.viewMonth = predMonthValue(predMonthFromDate(selected))
    render()
    positionPredCalendar(trigger, panel)
    if (typeof panel.showPopover === 'function') panel.showPopover()
    else panel.dataset.open = 'true'
    trigger.setAttribute('aria-expanded', 'true')
    focusPredCalendarSelection(panel)
  })
  panel.addEventListener('toggle', () => {
    trigger.setAttribute('aria-expanded', String(panel.matches(':popover-open')))
  })
  panel.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      close()
      trigger.focus()
      return
    }
    const offsets: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowUp: -7,
      ArrowDown: 7,
      Home: -42,
      End: 42,
      PageUp: -31,
      PageDown: 31,
    }
    const offset = offsets[event.key]
    if (offset === undefined) return
    event.preventDefault()
    movePredCalendarFocus(panel, offset)
  })
  panel.addEventListener('tri:pred-date-render', () => render())

  return { wrap, trigger, panel, render, close }
}

const isPredCompareKey = (value: string | undefined): value is PredCompareKey =>
  PRED_COMPARE_OPTIONS.some(option => option.key === value)

const predCompareKey = (block: HTMLElement): PredCompareKey => {
  const key = block.dataset.compareMode
  return isPredCompareKey(key) ? key : PRED_DEFAULT_COMPARE
}

const predCompareOption = (key: PredCompareKey): (typeof PRED_COMPARE_OPTIONS)[number] =>
  PRED_COMPARE_OPTIONS.find(option => option.key === key) ?? PRED_COMPARE_OPTIONS[0]

const syncPredDateControl = (block: HTMLElement, f: PaceForecaster): void => {
  const trigger = block.querySelector<HTMLButtonElement>('.tri-pred-date')
  const text = block.querySelector<HTMLElement>('.tri-pred-date-text')
  const bounds = f.dayBounds()
  if (!trigger || !text || !bounds) return
  const selected = block.dataset.compareDate
  const fallback = f.dayStateAgo(30)?.date ?? bounds.min
  const date = selected && selected >= bounds.min && selected <= bounds.max ? selected : fallback
  block.dataset.compareMin = bounds.min
  block.dataset.compareMax = bounds.max
  block.dataset.compareDate = date
  trigger.dataset.value = date
  text.textContent = date
  const panel = block.querySelector<HTMLElement>('.tri-pred-calendar')
  if (panel?.matches(':popover-open')) panel.dispatchEvent(new CustomEvent('tri:pred-date-render'))
}

const predComparison = (f: PaceForecaster, block: HTMLElement): PredComparison => {
  const key = predCompareKey(block)
  if (key === 'custom') {
    const day = f.dayStateOnOrBefore(block.dataset.compareDate ?? '')
    return { day, label: day?.date ? `vs ${shortDate(day.date)}` : tl('custom date missing') }
  }
  const days = predCompareOption(key).days ?? 30
  const day = f.dayStateAgo(days)
  return {
    day,
    label: day?.date ? `vs ${days}d (${shortDate(day.date)})` : `vs ${days}d (${tl('no data')})`,
  }
}

const renderPredAxis = (track: HTMLElement, maxSec: number): void => {
  const ticks = track.querySelectorAll<HTMLElement>('.tri-pred-axis-tick')
  if (ticks.length === PRED_AXIS_FRACS.length) {
    PRED_AXIS_FRACS.forEach((fr, i) => (ticks[i].textContent = hms(maxSec * fr)))
    return
  }
  track.replaceChildren(
    ...PRED_AXIS_FRACS.map(fr => {
      const tick = el('span', 'tri-pred-axis-tick', hms(maxSec * fr))
      tick.style.left = `${fr * 100}%`
      if (fr === 0) tick.dataset.edge = 'start'
      else if (fr === 1) tick.dataset.edge = 'end'
      return tick
    }),
  )
}

const resetPredCard = (card: HTMLElement, preserveVisual: boolean): void => {
  const stale = card.dataset.stale === '1'
  delete card.dataset.filled
  delete card.dataset.error
  card.dataset.pending = '1'
  if (preserveVisual || stale) {
    card.dataset.tipH = card.dataset.label ?? ''
    card.dataset.tipD = 'updating'
    return
  }
  delete card.dataset.tipD
  delete card.dataset.tipH
  const bar = card.querySelector<HTMLElement>('.tri-pred-bar')
  if (bar) bar.style.width = '0%'
  const range = card.querySelector<HTMLElement>('.tri-pred-bar-range')
  if (range) {
    range.style.left = '0%'
    range.style.width = '0%'
  }
  const timeEl = card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = '…'
  const deltaEl = card.querySelector('.tri-pred-delta')
  if (deltaEl) {
    deltaEl.textContent = ''
    deltaEl.classList.remove('tri-pred-delta--up', 'tri-pred-delta--down', 'tri-pred-delta--na')
  }
}

const failPredCard = (card: HTMLElement): void => {
  delete card.dataset.pending
  delete card.dataset.stale
  card.dataset.error = '1'
  const timeEl = card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = '—'
  const deltaEl = card.querySelector('.tri-pred-delta')
  if (deltaEl) deltaEl.textContent = ''
  card.dataset.tipH = card.dataset.label ?? ''
  card.dataset.tipD = 'model unavailable'
}

const applyPredResult = (r: PredResult, maxSec: number): void => {
  const pct = (s: number): number => (s / maxSec) * 100
  delete r.card.dataset.pending
  delete r.card.dataset.stale
  delete r.card.dataset.error
  r.card.dataset.filled = '1'
  const badge = r.card.querySelector('.tri-pred-badge')
  if (badge && r.card.dataset.label) badge.textContent = r.card.dataset.label
  const bar = r.card.querySelector<HTMLElement>('.tri-pred-bar')
  if (bar) bar.style.width = `${Math.max(2, pct(r.nowSec))}%`
  const range = r.card.querySelector<HTMLElement>('.tri-pred-bar-range')
  if (range) {
    range.style.left = `${pct(r.fastSec)}%`
    range.style.width = `${Math.max(0, pct(r.slowSec) - pct(r.fastSec))}%`
  }
  const timeEl = r.card.querySelector('.tri-pred-time')
  if (timeEl) timeEl.textContent = hms(r.nowSec)
  let tip = `${hms(r.nowSec)} · ${hms(r.fastSec)}–${hms(r.slowSec)} band`
  const deltaEl = r.card.querySelector('.tri-pred-delta')
  if (deltaEl) {
    deltaEl.classList.remove('tri-pred-delta--up', 'tri-pred-delta--down', 'tri-pred-delta--na')
    if (r.delta == null) {
      deltaEl.textContent = '—'
      deltaEl.classList.add('tri-pred-delta--na')
      tip += ` · ${r.compareLabel}`
    } else if (Math.abs(r.delta) >= 1) {
      const faster = r.delta < 0
      deltaEl.textContent = `${faster ? '▾' : '▴'}${hms(Math.abs(r.delta))}`
      deltaEl.classList.add(faster ? 'tri-pred-delta--up' : 'tri-pred-delta--down')
      tip += ` · ${faster ? '▾' : '▴'}${hms(Math.abs(r.delta))} ${r.compareLabel}`
    } else {
      deltaEl.textContent = ''
      tip += ` · ${r.compareLabel}`
    }
  }
  r.card.dataset.tipH = r.card.dataset.label ?? ''
  r.card.dataset.tipD = tip
}

const inferPredCard = async (
  f: PaceForecaster,
  day: PaceDayState,
  comparison: PredComparison,
  card: HTMLElement,
): Promise<PredResult | null> => {
  const km = Number(card.dataset.km)
  const sport = card.dataset.sport
  if (!km || !isPaceSport(sport)) return null
  const leg: PaceLegSpec = { sport, distanceKm: km, elevationM: 0, tempC: null, windKph: null }
  const [now, then] = await Promise.all([
    f.forecastLegAt(day, leg),
    comparison.day ? f.forecastLegAt(comparison.day, leg) : Promise.resolve(null),
  ])
  if (!now || now.mu <= 0) return null
  const meters = km * 1000
  const nowSec = meters / now.mu
  const timeSd = (meters / (now.mu * now.mu)) * now.sigma
  const fastSec = Math.max(0, nowSec - Z80 * timeSd)
  const slowSec = nowSec + Z80 * timeSd
  const delta = then && then.mu > 0 ? nowSec - meters / then.mu : null
  return { card, nowSec, fastSec, slowSec, delta, compareLabel: comparison.label }
}

async function fillDistancePredictor(scope: ParentNode): Promise<void> {
  const f = paceForecaster
  const block =
    scope instanceof HTMLElement
      ? (scope.closest<HTMLElement>('.tri-pred') ?? scope.querySelector<HTMLElement>('.tri-pred'))
      : null
  if (!block) return
  if (!f?.ready || !f.day) return
  syncPredDateControl(block, f)
  const day = f.day
  const comparison = predComparison(f, block)
  const cards = Array.from(scope.querySelectorAll<HTMLElement>('.tri-pred-card')).filter(
    c => !c.dataset.filled,
  )
  if (!cards.length) return
  const runId = String(++predRunSeq)
  block.dataset.predRun = runId
  const hasStale = cards.some(card => card.dataset.stale === '1')
  if (!hasStale) block.querySelector<HTMLElement>('.tri-pred-axis-track')?.replaceChildren()
  for (const card of cards) resetPredCard(card, hasStale)
  const results: PredResult[] = []
  const render = (): void => {
    if (block.dataset.predRun !== runId) return
    const maxSec = Math.max(...results.map(r => r.slowSec), 1)
    for (const result of results) applyPredResult(result, maxSec)
    const axis = block.querySelector<HTMLElement>('.tri-pred-axis-track')
    if (axis && results.length) renderPredAxis(axis, maxSec)
  }
  render()
  await Promise.all(
    cards.map(async card => {
      const result = await inferPredCard(f, day, comparison, card)
      if (block.dataset.predRun !== runId) return
      if (result) results.push(result)
      else failPredCard(card)
      render()
    }),
  )
}

const buildDistancePredictor = (): HTMLElement => {
  const block = el('div', 'tri-pred')
  block.dataset.compareMode = PRED_DEFAULT_COMPARE
  const head = el('div', 'tri-pred-head')
  const headMain = el('div', 'tri-pred-head-main')
  headMain.append(el('span', 'tri-pred-title', 'pace predictor'))
  const controls = el('div', 'tri-pred-controls')
  const compare = el('div', 'tri-pred-compare', undefined, {
    role: 'tablist',
    'aria-label': 'comparison range',
  })
  let activeSport: PaceSport = 'run'
  const updateCompareControls = (): void => {
    const mode = predCompareKey(block)
    for (const button of compare.querySelectorAll<HTMLButtonElement>('.tri-pred-compare-btn')) {
      const active = button.dataset.compareMode === mode
      button.classList.toggle('tri-pred-compare-btn--on', active)
      button.setAttribute('aria-selected', String(active))
    }
    if (paceForecaster?.ready) syncPredDateControl(block, paceForecaster)
  }
  head.append(headMain, controls)
  block.appendChild(head)
  const tabs = el('div', 'tri-pred-tabs', undefined, {
    role: 'group',
    'aria-label': 'predictor sport',
  })
  const grid = el('div', 'tri-pred-grid')
  const renderSport = (sport: PaceSport): void => {
    activeSport = sport
    const cfg = PRED_SPORTS.find(s => s.sport === sport)
    if (!cfg) return
    let cards = Array.from(grid.querySelectorAll<HTMLElement>('.tri-pred-card'))
    if (cards.length !== cfg.dists.length) {
      cards = cfg.dists.map(d => {
        const card = el('div', 'tri-pred-card')
        const track = el('div', 'tri-pred-bar-track')
        track.append(el('div', 'tri-pred-bar-range'), el('div', 'tri-pred-bar'))
        card.append(
          el('span', 'tri-pred-badge', d.label),
          track,
          el('span', 'tri-pred-time', '—'),
          el('span', 'tri-pred-delta'),
        )
        return card
      })
      grid.replaceChildren(...cards)
    }
    cfg.dists.forEach((d, i) => {
      const card = cards[i]
      card.dataset.km = String(d.km)
      card.dataset.sport = sport
      card.dataset.label = d.label
      if (card.dataset.filled) card.dataset.stale = '1'
      delete card.dataset.filled
      delete card.dataset.pending
      delete card.dataset.error
      if (!card.dataset.stale) {
        delete card.dataset.tipD
        delete card.dataset.tipH
      }
    })
    if (paceForecaster?.ready) queueMicrotask(() => void fillDistancePredictor(grid))
    else {
      if (paceForecastUnavailable) for (const card of cards) failPredCard(card)
    }
  }
  for (const option of PRED_COMPARE_OPTIONS) {
    const button = el(
      'button',
      `tri-pred-compare-btn${option.key === PRED_DEFAULT_COMPARE ? ' tri-pred-compare-btn--on' : ''}`,
      option.label,
      {
        type: 'button',
        role: 'tab',
        'aria-selected': String(option.key === PRED_DEFAULT_COMPARE),
        'data-compare-mode': option.key,
      },
    )
    button.addEventListener('click', () => {
      block.dataset.compareMode = option.key
      updateCompareControls()
      renderSport(activeSport)
    })
    compare.appendChild(button)
  }
  const activateCustomCompare = (): void => {
    if (block.dataset.compareMode === 'custom') return
    block.dataset.compareMode = 'custom'
    updateCompareControls()
    renderSport(activeSport)
  }
  const selectPredDate = (date: string): void => {
    block.dataset.compareMode = 'custom'
    block.dataset.compareDate = date
    updateCompareControls()
    renderSport(activeSport)
  }
  const clearPredDate = (): void => {
    block.dataset.compareMode = PRED_DEFAULT_COMPARE
    updateCompareControls()
    renderSport(activeSport)
  }
  const datePicker = buildPredDatePicker(
    block,
    activateCustomCompare,
    selectPredDate,
    clearPredDate,
  )
  for (const s of PRED_SPORTS) {
    const on = s.sport === 'run'
    const tab = el(
      'button',
      `tri-pred-tab tri-pred-tab--${s.sport}${on ? ' tri-pred-tab--on' : ''}`,
      undefined,
      {
        type: 'button',
        'aria-label': s.sport,
        'aria-pressed': String(on),
        'aria-selected': String(on),
        title: s.sport,
      },
    )
    tab.appendChild(buildIcon(s.sport))
    tab.addEventListener('click', () => {
      for (const t of tabs.querySelectorAll<HTMLElement>('.tri-pred-tab')) {
        const active = t === tab
        t.classList.toggle('tri-pred-tab--on', active)
        t.setAttribute('aria-pressed', String(active))
        t.setAttribute('aria-selected', String(active))
      }
      renderSport(s.sport)
    })
    tabs.appendChild(tab)
  }
  controls.append(tabs, compare, datePicker.wrap)
  const axis = el('div', 'tri-pred-axis')
  axis.append(
    el('span', 'tri-pred-axis-pad'),
    el('div', 'tri-pred-axis-track'),
    el('span', 'tri-pred-axis-end'),
  )
  block.append(grid, axis)
  updateCompareControls()
  renderSport('run')
  return block
}

const wirePredTip = (root: HTMLElement): (() => void) => {
  document.body.querySelector('.tri-pred-tip')?.remove()
  const tip = el('div', 'tri-gloss tri-pred-tip')
  tip.setAttribute('role', 'tooltip')
  document.body.appendChild(tip)
  let cur: HTMLElement | null = null
  const move = (e: MouseEvent): void => {
    const card = (e.target as HTMLElement | null)?.closest<HTMLElement>('.tri-pred-card')
    if (!card?.dataset.tipD) {
      cur = null
      tip.classList.remove('tri-gloss--on')
      return
    }
    if (card !== cur) {
      cur = card
      tip.replaceChildren(
        el('span', 'tri-gloss-h', card.dataset.tipH ?? ''),
        el('span', 'tri-gloss-def', card.dataset.tipD ?? ''),
      )
      tip.classList.add('tri-gloss--on')
    }
    const pr = tip.getBoundingClientRect()
    const left =
      e.clientX + 14 + pr.width > window.innerWidth - 8 ? e.clientX - 14 - pr.width : e.clientX + 14
    const top =
      e.clientY + 14 + pr.height > window.innerHeight - 8
        ? e.clientY - 14 - pr.height
        : e.clientY + 14
    tip.style.left = `${Math.max(8, left).toFixed(0)}px`
    tip.style.top = `${Math.max(8, top).toFixed(0)}px`
  }
  const leave = (): void => {
    cur = null
    tip.classList.remove('tri-gloss--on')
  }
  root.addEventListener('mousemove', move)
  root.addEventListener('mouseleave', leave)
  return () => {
    root.removeEventListener('mousemove', move)
    root.removeEventListener('mouseleave', leave)
    tip.remove()
  }
}

const paceModelBaseCandidates = (): string[] => {
  const bases = [location.origin]
  const port = Number(location.port)
  if (
    (location.hostname === 'localhost' || location.hostname === '127.0.0.1') &&
    Number.isInteger(port) &&
    port > 0 &&
    port + 707 <= 65535
  )
    bases.push(`${location.protocol}//${location.hostname}:${port + 707}`)
  return Array.from(new Set(bases))
}

const initPaceForecaster = async (forecaster: PaceForecaster): Promise<boolean> => {
  for (const base of paceModelBaseCandidates())
    if (await forecaster.init(base, 'pace', '/triathlon/data.jsonl')) return true
  return false
}

const markDistancePredictorUnavailable = (root: HTMLElement): void => {
  for (const block of root.querySelectorAll<HTMLElement>('.tri-pred')) {
    for (const card of block.querySelectorAll<HTMLElement>('.tri-pred-card'))
      if (!card.dataset.filled) failPredCard(card)
  }
}

const setupPaceForecast = (root: HTMLElement): (() => void) | null => {
  if (!root.dataset.analyticsPath && !root.querySelector('.tri-calc')) return null
  let worker: Worker
  try {
    worker = new Worker(new URL('/pace.worker.js', import.meta.url), { type: 'module' })
  } catch {
    return null
  }
  const forecaster = new PaceForecaster(worker)
  paceForecaster = forecaster
  paceForecastUnavailable = false
  initPaceForecaster(forecaster)
    .then(ok => {
      if (paceForecaster !== forecaster) return
      paceForecastUnavailable = !ok
      if (!ok) {
        markDistancePredictorUnavailable(root)
        return
      }
      void fillDistancePredictor(root)
    })
    .catch(() => {
      if (paceForecaster !== forecaster) return
      paceForecastUnavailable = true
      markDistancePredictorUnavailable(root)
    })
  const tipCleanup = wirePredTip(root)
  return () => {
    tipCleanup()
    forecaster.dispose()
    if (paceForecaster === forecaster) paceForecaster = null
  }
}

const mountTriathlon = (): (() => void) => {
  const cleanups: (() => void)[] = []
  const addCleanup = (cleanup: (() => void) | null | undefined): void => {
    if (cleanup) cleanups.push(cleanup)
  }
  try {
    setDistanceUnit(localStorage.getItem(TRI_UNIT_KEY) === 'mi')
  } catch {}
  const root = document.querySelector<HTMLElement>('.triathlon')
  if (root) initTriLocale()
  const embedCleanup = setupDayEmbeds()
  addCleanup(embedCleanup)
  addCleanup(setupChartScrub(document.body))
  if (root) {
    addCleanup(setupI18n(root))
    addCleanup(setupCommandPalette(root))
    addCleanup(setup(root))
    addCleanup(setupCalc(root))
    addCleanup(setupPaceForecast(root))
    addCleanup(setupDropdown(root, '.tri-gear-wrap', '.tri-gear-btn', '.tri-gear', 'tri-gear-open'))
    addCleanup(setupDropdown(root, '.tri-pace-wrap', '.tri-pace-btn', '.tri-pace', 'tri-pace-open'))
    addCleanup(setupPaceUnit(root))
    addCleanup(setupCheat(root))
    addCleanup(setupAnalytics(root))
    addCleanup(setupFeed(root))
    addCleanup(setupTraining(root))
    addCleanup(setupMap(root))
    addCleanup(setupGloss(root))
    addCleanup(setupAxisLabels(root))
    addCleanup(setupShortcuts(root))
    const hashDate = /^#(\d{4}-\d{2}-\d{2})$/.exec(window.location.hash)?.[1]
    if (hashDate)
      window.dispatchEvent(new CustomEvent('tri:focus-day', { detail: { date: hashDate } }))
    const calcHash = /^#(calculator-[a-z0-9-]+)$/.exec(window.location.hash)?.[1]
    const calcShare = calcHash ? decodeCalcShare(calcHash) : null
    if (calcShare)
      window.dispatchEvent(new CustomEvent('tri:calc-fill', { detail: { share: calcShare } }))
  }
  return () => {
    for (let i = cleanups.length - 1; i >= 0; i--) cleanups[i]()
  }
}

let triathlonCleanup: (() => void) | null = null

const unmountTriathlon = (): void => {
  triathlonCleanup?.()
  triathlonCleanup = null
}

document.addEventListener('prenav', unmountTriathlon)
document.addEventListener('nav', () => {
  unmountTriathlon()
  triathlonCleanup = mountTriathlon()
})
