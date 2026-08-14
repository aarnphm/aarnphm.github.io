import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { DetailCtx } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { ActivityAnalysisRange } from './analysis'
import type { ActivityAnalysisController } from './analysis'
import type { ActivityRangeChange } from './analysis'
import type { ScrubSurface } from './analysis'
import type { DetailPayload } from './data'
import { activityStatRows } from '../../../util/triathlon-card'
import { buildActivity as buildActivityNode } from '../../../util/triathlon-card'
import { buildCyclingBestEfforts as buildCyclingBestEffortsNode } from '../../../util/triathlon-card'
import { buildRespirationTrace as buildRespirationTraceNode } from '../../../util/triathlon-card'
import { buildRunGroundContactTrace as buildRunGroundContactTraceNode } from '../../../util/triathlon-card'
import { buildRunLapSplits as buildRunLapSplitsNode } from '../../../util/triathlon-card'
import { buildRunStrideTrace as buildRunStrideTraceNode } from '../../../util/triathlon-card'
import { buildRunVerticalOscillationTrace as buildRunVerticalOscillationTraceNode } from '../../../util/triathlon-card'
import { buildTrainingEffectDetails as buildTrainingEffectDetailsNode } from '../../../util/triathlon-card'
import { clock } from '../../../util/triathlon-card'
import { dominantTrainingEffectGroup } from '../../../util/triathlon-card'
import { formatAltitude } from '../../../util/triathlon-card'
import { formatGroundContactTime } from '../../../util/triathlon-card'
import { formatRespirationRate } from '../../../util/triathlon-card'
import { formatStrideLength } from '../../../util/triathlon-card'
import { formatTemperature } from '../../../util/triathlon-card'
import { formatThermalTemperature } from '../../../util/triathlon-card'
import { formatVerticalOscillation } from '../../../util/triathlon-card'
import { gearShiftAtFraction } from '../../../util/triathlon-card'
import { gradeAt } from '../../../util/triathlon-card'
import { interpolatePositiveMetricSeries } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
import { positiveMetricDomain } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { powerBalanceText } from '../../../util/triathlon-card'
import { runGroundContactTimeMs } from '../../../util/triathlon-card'
import { runStrideLengthLabel } from '../../../util/triathlon-card'
import { runStrideLengthValue } from '../../../util/triathlon-card'
import { runVerticalOscillationCm } from '../../../util/triathlon-card'
import { scrubDist } from '../../../util/triathlon-card'
import { zoneClock } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import {
  triathlonTraceEnabled,
  type TriathlonTraceSettings,
} from '../../../util/triathlon-trace-settings'
import { buildMatchedRideGroup } from '../analytics/panels/matched'
import { buildMatchedRunGroup } from '../analytics/panels/matched'
import {
  CAD_RAMP,
  ELEV_RAMP,
  HEAT_RAMP,
  HR_RAMP,
  rampGradient,
  RESP_RAMP,
  SPD_RAMP,
  STRIDE_RAMP,
} from '../maps/palette'
import { createDomFactory } from '../runtime/dom'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import { analysisRate } from './analysis'
import { linkScrub } from './analysis'
import { detailContextFromPayload } from './data'
import {
  buildElevation,
  buildHrZones,
  buildIcon,
  buildPool,
  buildPowerCurve,
  buildPowerHist,
  buildPowerZones,
  buildTrace,
  statRow,
  zoneDuo,
} from './primitives'

export const buildHeatRoute = (
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
    if (zeroGap && (vals[i] <= 0 || vals[i + 1] <= 0)) continue
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
  s.appendChild(svg('path', { class: 'tri-route-selected', d: '' }))
  s.appendChild(svg('circle', { class: 'tri-route-cursor', cx: -10, cy: -10, r: 2.6 }))
  return s
}

export const buildHeatLegend = (
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

export interface MapMetric {
  label: string
  shortLabel: string
  ramp: string[]
  zeroGap?: boolean
  pick: (p: StravaActivityDetail['route'][number], i: number) => number
  fmt: (v: number) => string
  profile: (graphDomain?: ActivityAnalysisRange | null) => HTMLElement
  readout: (p: StravaActivityDetail['route'][number], i: number) => string
  extra?: () => (HTMLElement | null)[]
}

export const metricSpecs = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
  detailContext: DetailCtx,
): MapMetric[] => {
  const domF = createDomFactory(presentation)
  const route = d.route
  const imperial = presentation.distance === 'imperial'
  const filterPowerZeros = d.sport === 'bike' && presentation.powerSamples === 'exclude-zero'
  const powerValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.w)
    : route.map(point => point.w)
  const powerDomain = filterPowerZeros ? positiveMetricDomain(powerValues) : undefined
  const pace = route.map(point => point.speedKph)
  const hasPower = d.deviceWatts && route.some(p => p.w > 0)
  const hasHr = route.some(p => p.hr > 0)
  const hasCad = route.some(p => p.cad > 0)
  const strideLabel = runStrideLengthLabel(d)
  const hasStride =
    d.sport === 'run' && route.filter(p => runStrideLengthValue(d, p) != null).length >= 2
  const hasGroundContact =
    d.sport === 'run' && route.filter(p => runGroundContactTimeMs(p) != null).length >= 2
  const hasVerticalOscillation =
    d.sport === 'run' && route.filter(p => runVerticalOscillationCm(p) != null).length >= 2
  const hasResp = route.some(p => p.resp != null && p.resp > 0)
  const hasElev = d.maxAlt > d.minAlt
  const cadUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const paceFmt = (kmh: number): string => {
    return analysisRate(presentation, d.sport, kmh)
  }
  const paceTick = (kmh: number): string => {
    if (kmh <= 0) return '0'
    if (d.sport === 'bike')
      return imperial ? `${(kmh * KM_TO_MI).toFixed(0)}mph` : `${kmh.toFixed(0)}km/h`
    if (d.sport === 'swim') return clock(3600 / (kmh * 10))
    return clock(3600 / (kmh * (imperial ? KM_TO_MI : 1)))
  }
  const paceSpec: MapMetric = {
    label: d.sport === 'bike' ? 'speed' : 'pace',
    shortLabel: d.sport === 'bike' ? 'S' : 'P',
    ramp: SPD_RAMP,
    pick: (_p, i) => pace[i],
    fmt: paceFmt,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (_p, i) => pace[i],
        d.sport === 'bike' ? 'speed' : 'pace',
        () => '',
        paceTick,
        graphDomain,
      ),
    readout: (p, i) => `${scrubDist(presentation, p.d, d.sport)} · ${paceFmt(pace[i])}`,
  }
  const powerSpec: MapMetric = {
    label: 'power',
    shortLabel: 'W',
    ramp: HEAT_RAMP,
    pick: (p, i) => powerValues[i] ?? p.w,
    fmt: v => `${Math.round(v)} W`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => powerValues[i] ?? p.w,
        'power',
        m => `${m} W peak`,
        v => `${Math.round(v)}w`,
        graphDomain,
        powerDomain,
        d.sport === 'bike' && detailContext.criticalPower
          ? {
              value: detailContext.criticalPower.criticalPowerWatts,
              label: `eCP ${detailContext.criticalPower.criticalPowerWatts} W`,
            }
          : null,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(powerValues[i] ?? p.w)} W`,
    extra: () => [
      zoneDuo(
        presentation,
        buildPowerCurve(presentation, d, detailContext),
        buildPowerHist(presentation, d),
      ),
      buildPowerZones(presentation, d, detailContext),
    ],
  }
  const hrSpec: MapMetric = {
    label: 'heart rate',
    shortLabel: 'HR',
    ramp: HR_RAMP,
    pick: p => p.hr,
    fmt: v => `${Math.round(v)} bpm`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        p => p.hr,
        'hr',
        m => `${m} bpm peak`,
        v => `${Math.round(v)}bpm`,
        graphDomain,
      ),
    readout: p => `${scrubDist(presentation, p.d, d.sport)} · ${p.hr} bpm`,
    extra: () => [
      buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null,
      buildHrZones(presentation, d, detailContext),
    ],
  }
  const cadScale = d.sport === 'run' ? 2 : 1
  const cadenceValues = filterPowerZeros
    ? interpolatePositiveMetricSeries(route, point => point.cad * cadScale)
    : route.map(point => point.cad * cadScale)
  const cadenceDomain = filterPowerZeros ? positiveMetricDomain(cadenceValues) : undefined
  const cadSpec: MapMetric = {
    label: 'cadence',
    shortLabel: 'C',
    ramp: CAD_RAMP,
    zeroGap: !filterPowerZeros,
    pick: (p, i) => cadenceValues[i] ?? p.cad * cadScale,
    fmt: v => `${Math.round(v)} ${cadUnit}`,
    profile: graphDomain =>
      buildTrace(
        presentation,
        d,
        (p, i) => cadenceValues[i] ?? p.cad * cadScale,
        'cadence',
        m => `${m} ${cadUnit} peak`,
        v => `${Math.round(v)}${cadUnit}`,
        graphDomain,
        cadenceDomain,
      ),
    readout: (p, i) =>
      `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(cadenceValues[i] ?? p.cad * cadScale)} ${cadUnit}`,
  }
  const strideSpec: MapMetric = {
    label: strideLabel,
    shortLabel: 'SL',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runStrideLengthValue(d, p) ?? 0,
    fmt: value => formatStrideLength(presentation, value),
    profile: graphDomain => buildRunStrideTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const stride = runStrideLengthValue(d, p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
    },
  }
  const groundContactSpec: MapMetric = {
    label: 'ground contact time',
    shortLabel: 'GCT',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runGroundContactTimeMs(p) ?? 0,
    fmt: formatGroundContactTime,
    profile: graphDomain =>
      buildRunGroundContactTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const groundContact = runGroundContactTimeMs(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${groundContact == null ? '—' : formatGroundContactTime(groundContact)}`
    },
  }
  const verticalOscillationSpec: MapMetric = {
    label: 'vertical oscillation',
    shortLabel: 'VO',
    ramp: STRIDE_RAMP,
    zeroGap: true,
    pick: p => runVerticalOscillationCm(p) ?? 0,
    fmt: value => formatVerticalOscillation(presentation, value),
    profile: graphDomain =>
      buildRunVerticalOscillationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p => {
      const verticalOscillation = runVerticalOscillationCm(p)
      return `${scrubDist(presentation, p.d, d.sport)} · ${verticalOscillation == null ? '—' : formatVerticalOscillation(presentation, verticalOscillation)}`
    },
  }
  const respirationSpec: MapMetric = {
    label: 'respiration',
    shortLabel: 'R',
    ramp: RESP_RAMP,
    pick: p => p.resp ?? 0,
    fmt: formatRespirationRate,
    profile: graphDomain => buildRespirationTraceNode(domF, d, null, graphDomain) as HTMLElement,
    readout: p =>
      `${scrubDist(presentation, p.d, d.sport)} · ${p.resp == null ? '—' : formatRespirationRate(p.resp)}`,
  }
  const elevSpec: MapMetric = {
    label: 'elevation',
    shortLabel: 'E',
    ramp: ELEV_RAMP,
    pick: p => p.alt,
    fmt: value => formatAltitude(presentation, value),
    profile: graphDomain => buildElevation(presentation, d, graphDomain),
    readout: (p, i) => {
      const g = Math.round(gradeAt(route, i) * 10) / 10
      return `${scrubDist(presentation, p.d, d.sport)} · ${formatAltitude(presentation, p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%`
    },
  }
  const specs: MapMetric[] = []
  if (d.sport === 'bike') {
    if (hasPower) specs.push(powerSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (hasResp) specs.push(respirationSpec)
    specs.push(paceSpec)
    if (hasElev) specs.push(elevSpec)
  } else if (d.sport === 'run') {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasCad) specs.push(cadSpec)
    if (hasStride) specs.push(strideSpec)
    if (hasGroundContact) specs.push(groundContactSpec)
    if (hasVerticalOscillation) specs.push(verticalOscillationSpec)
    if (hasResp) specs.push(respirationSpec)
    if (hasElev) specs.push(elevSpec)
    if (hasPower) specs.push(powerSpec)
  } else {
    specs.push(paceSpec)
    if (hasHr) specs.push(hrSpec)
    if (hasResp) specs.push(respirationSpec)
  }
  return specs
}

export interface MapDetailOpts {
  mapMode?: boolean
  initialMetric?: number
  onMetric?: (i: number) => void
  onHover?: (p: StravaActivityDetail['route'][number], i: number) => void
  analysis?: HTMLElement | null
  onRange?: ActivityRangeChange
  detailContext?: DetailCtx
}

export interface ActivityView {
  element: HTMLElement
  mount: () => () => void
}

export const renderMapDetail = (
  presentation: TriathlonPresentation,
  source: StravaActivityDetail,
  opts?: MapDetailOpts,
): ActivityView => {
  const domF = createDomFactory(presentation)
  const d = powerViewActivity(presentation, source)
  const wrap = el('section', 'tri-act tri-act--expanded')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildIcon(presentation, d.sport))
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const sbody = document.createElement('tbody')
  const summaryTrainingEffectGroup = d.garmin?.trainingEffectLabel
    ? dominantTrainingEffectGroup(d.garmin.trainingEffectLabel)
    : null
  for (const [label, value] of activityStatRows(presentation, d))
    sbody.appendChild(
      statRow(
        presentation,
        label,
        value,
        label === 'training effect' && summaryTrainingEffectGroup
          ? { 'data-training-effect-group': summaryTrainingEffectGroup }
          : undefined,
      ),
    )
  stats.appendChild(sbody)
  wrap.appendChild(stats)

  const specs =
    d.route.length >= 2
      ? metricSpecs(presentation, d, opts?.detailContext ?? detailContextFromPayload())
      : []
  if (specs.length === 0) {
    const figs = el('div', 'tri-act-figs')
    if (d.sport === 'swim') figs.appendChild(buildPool(presentation, d))
    if (figs.childElementCount > 0) wrap.appendChild(figs)
    const more = el('div', 'tri-act-more')
    const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
    const trainingEffect = buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null
    for (const z of [
      trainingEffect,
      zoneDuo(
        presentation,
        buildHrZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
        buildPowerZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
      ),
      zoneDuo(
        presentation,
        buildPowerCurve(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
        buildPowerHist(presentation, d),
      ),
    ])
      if (z) more.appendChild(z)
    if (bestEfforts) more.appendChild(bestEfforts)
    if (more.childElementCount > 0) wrap.appendChild(more)
    return { element: wrap, mount: () => () => {} }
  }

  const tablist = el('div', 'tri-map-tablist')
  tablist.setAttribute('role', 'tablist')
  const figs = el('div', 'tri-act-figs tri-map-figs')
  const profileBox = el('div', 'tri-map-profile')
  const runSplits = buildRunLapSplitsNode(domF, d) as HTMLElement | null
  const zoneBox = el('div', 'tri-act-more')
  const bestEfforts = buildCyclingBestEffortsNode(domF, d) as HTMLElement | null
  wrap.append(tablist, figs)
  if (runSplits) wrap.appendChild(runSplits)
  wrap.appendChild(profileBox)
  wrap.appendChild(zoneBox)

  let active = Math.min(specs.length - 1, Math.max(0, opts?.initialMetric ?? 0))
  let graphDomain: ActivityAnalysisRange | null = null
  let routeMarker: SVGElement | null = null
  let analysisController: ActivityAnalysisController | null = null
  let mounted = false
  const sameGraphDomain = (
    left: ActivityAnalysisRange | null,
    right: ActivityAnalysisRange | null,
  ): boolean =>
    left === right ||
    (left != null &&
      right != null &&
      left.startDistanceKm === right.startDistanceKm &&
      left.endDistanceKm === right.endDistanceKm)
  const renderProfile = (): void => {
    analysisController?.dispose()
    analysisController = null
    const spec = specs[active]
    const profile = spec.profile(graphDomain)
    profileBox.replaceChildren(profile)
    if (mounted) mountProfile(profile)
  }
  const mountProfile = (profile: HTMLElement): void => {
    const spec = specs[active]
    analysisController = linkScrub(
      presentation,
      wrap,
      routeMarker,
      [
        {
          wrap: profile,
          fmt: (p, i) => {
            opts?.onHover?.(p, i)
            return spec.readout(p, i)
          },
        },
      ],
      d.route,
      d,
      opts?.analysis,
      onRange,
    )
  }
  const onRange: ActivityRangeChange = (range, committed) => {
    if (committed && !sameGraphDomain(graphDomain, range)) {
      graphDomain = range
      renderProfile()
    }
    opts?.onRange?.(range, committed)
  }
  const draw = (animateTabs = true, notify = true) => {
    const spec = specs[active]
    const vals = d.route.map((p, i) => spec.pick(p, i))
    const pool = spec.zeroGap ? vals.filter(v => v > 0) : vals
    let lo = Infinity
    let hi = -Infinity
    for (const v of pool.length ? pool : vals) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    routeMarker = null
    if (opts?.mapMode) {
      figs.replaceChildren(buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
    } else {
      const heat = buildHeatRoute(d.route, spec.pick, spec.ramp, spec.zeroGap)
      figs.replaceChildren(heat, buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
      routeMarker = heat.querySelector<SVGElement>('.tri-route-cursor')
    }
    renderProfile()
    zoneBox.replaceChildren()
    if (spec.extra) for (const z of spec.extra()) if (z) zoneBox.appendChild(z)
    if (bestEfforts) zoneBox.appendChild(bestEfforts)
    const tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab'))
    if (mounted && !animateTabs) tablist.dataset.motion = 'instant'
    tabs.forEach((tab, i) => {
      const on = i === active
      tab.setAttribute('aria-selected', on ? 'true' : 'false')
      tab.tabIndex = on ? 0 : -1
      tab.style.background = on ? spec.ramp[6] : ''
      tab.style.borderColor = on ? spec.ramp[6] : ''
    })
    if (mounted && !animateTabs) {
      tablist.getBoundingClientRect()
      delete tablist.dataset.motion
    }
    if (notify) opts?.onMetric?.(active)
  }
  specs.forEach((spec, i) => {
    const tab = el('button', 'tri-map-tab')
    tab.setAttribute('type', 'button')
    tab.setAttribute('role', 'tab')
    tab.setAttribute('aria-label', spec.label)
    tab.dataset.shortcut = spec.shortLabel[0].toLowerCase()
    tab.dataset.index = String(i)
    tab.style.setProperty('--tri-map-tab-shortcut-width', `${spec.shortLabel.length}ch`)
    tab.style.setProperty('--tri-map-tab-label-width', `${spec.label.length}ch`)
    const shortcut = el('span', 'tri-map-tab-shortcut', spec.shortLabel)
    const label = el('span', 'tri-map-tab-label', spec.label)
    shortcut.setAttribute('aria-hidden', 'true')
    label.setAttribute('aria-hidden', 'true')
    tab.append(shortcut, label)
    tablist.appendChild(tab)
  })
  const onTabClick = (event: MouseEvent): void => {
    const target = event.target
    if (!(target instanceof Element)) return
    const tab = target.closest<HTMLButtonElement>('.tri-map-tab[data-index]')
    if (!tab || !tablist.contains(tab)) return
    const index = Number(tab.dataset.index)
    if (!Number.isInteger(index) || index < 0 || index >= specs.length) return
    active = index
    draw(event.detail > 0)
  }
  const onTabKeydown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    const tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-map-tab'))
    let next = -1
    if (event.key === 'ArrowLeft') next = (active - 1 + tabs.length) % tabs.length
    else if (event.key === 'ArrowRight') next = (active + 1) % tabs.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = tabs.length - 1
    else {
      const key = event.key.toLowerCase()
      next = tabs.findIndex(tab => tab.dataset.shortcut === key)
    }
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    if (next !== active) {
      active = next
      draw(false)
    }
    tabs[next]?.focus()
  }
  draw(false, false)
  return {
    element: wrap,
    mount: () => {
      mounted = true
      const profile = profileBox.firstElementChild
      if (profile instanceof HTMLElement) mountProfile(profile)
      tablist.addEventListener('click', onTabClick)
      tablist.addEventListener('keydown', onTabKeydown)
      return () => {
        mounted = false
        tablist.removeEventListener('click', onTabClick)
        tablist.removeEventListener('keydown', onTabKeydown)
        analysisController?.dispose()
        analysisController = null
      }
    },
  }
}

export const renderDetail = (
  presentation: TriathlonPresentation,
  source: StravaActivityDetail,
  payload?: DetailPayload | null,
  fillMissingRunPower = false,
  embedded = false,
  dayRouteHref?: string,
  traceSettings?: TriathlonTraceSettings,
): ActivityView => {
  const domF = createDomFactory(presentation)
  const d = powerViewActivity(presentation, source)
  const normalizeBikeMetrics = d.sport === 'bike' && presentation.powerSamples === 'exclude-zero'
  const powerValues = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.w)
    : null
  const detailContext = detailContextFromPayload(payload)
  const wrap = buildActivityNode(
    domF,
    d,
    false,
    detailContext,
    payload?.swimTrend ?? [],
    fillMissingRunPower,
    embedded,
  ) as HTMLElement
  if (d.sport === 'run') {
    const matchedGroup = payload?.matchedRuns?.groups.find(group =>
      group.efforts.some(effort => effort.id === d.id),
    )
    const more = wrap.querySelector<HTMLElement>(':scope > .tri-act-more')
    if (matchedGroup && more && triathlonTraceEnabled(traceSettings, 'matched-runs'))
      more.appendChild(buildMatchedRunGroup(presentation, matchedGroup, d.id, dayRouteHref))
  }
  if (d.sport === 'bike') {
    const matchedGroup = payload?.matchedRides?.groups.find(group =>
      group.efforts.some(effort => effort.id === d.id),
    )
    const more = wrap.querySelector<HTMLElement>(':scope > .tri-act-more')
    if (matchedGroup && more && triathlonTraceEnabled(traceSettings, 'matched-rides'))
      more.insertBefore(
        buildMatchedRideGroup(
          presentation,
          matchedGroup,
          d.id,
          dayRouteHref,
          detailContext.criticalPower,
        ),
        more.querySelector(':scope > .tri-efforts'),
      )
  }
  const surfaces: ScrubSurface[] = []
  for (const trace of wrap.querySelectorAll<HTMLElement>('[data-tri-trace]')) {
    const name = trace.dataset.triTrace
    if (!name || !triathlonTraceEnabled(traceSettings, name)) trace.remove()
  }
  const elev = wrap.querySelector<HTMLElement>(
    '.tri-act-figs .tri-elev-wrap:not(.tri-elev-wrap--unavailable)',
  )
  if (elev) {
    surfaces.push({
      wrap: elev,
      fmt: (p, i) => {
        const g = Math.round(gradeAt(d.route, i) * 10) / 10
        return (
          `${scrubDist(presentation, p.d, d.sport)} · ${formatAltitude(presentation, p.alt)} · ${g >= 0 ? '+' : ''}${g.toFixed(1)}%` +
          (p.hr > 0 ? ` · ${p.hr} bpm` : '')
        )
      },
    })
  }
  const cadenceScale = d.sport === 'run' ? 2 : 1
  const cadenceUnit = d.sport === 'run' ? 'spm' : 'rpm'
  const cadenceValues = normalizeBikeMetrics
    ? interpolatePositiveMetricSeries(d.route, point => point.cad * cadenceScale)
    : null
  const shiftingDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  for (const trace of wrap.querySelectorAll<HTMLElement>('[data-tri-trace]')) {
    if (trace.dataset.triTrace === 'hr')
      surfaces.push({
        wrap: trace,
        fmt: p => `${scrubDist(presentation, p.d, d.sport)} · ${p.hr} bpm`,
      })
    else if (trace.dataset.triTrace === 'power')
      surfaces.push({
        wrap: trace,
        fmt: (p, i) =>
          `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(powerValues?.[i] ?? p.w)} W`,
      })
    else if (trace.dataset.triTrace === 'power-balance')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.rightPowerPct == null || p.w <= 0 ? '—' : powerBalanceText(p.rightPowerPct)}`,
      })
    else if (trace.dataset.triTrace === 'electronic-shifting')
      surfaces.push({
        wrap: trace,
        fmt: p => {
          const shift = gearShiftAtFraction(
            d.gearShifts,
            shiftingDistanceKm,
            p.d / shiftingDistanceKm,
          )
          return `${zoneClock(p.elapsedS)} · ${scrubDist(presentation, p.d, d.sport)}${shift ? ` · ${shift.frontTeeth}×${shift.rearTeeth}` : ''}`
        },
      })
    else if (trace.dataset.triTrace === 'stamina')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${triText(presentation.locale, 'current')} ${p.stamina == null ? '—' : `${Math.round(p.stamina)}%`} · ${triText(presentation.locale, 'potential')} ${p.potentialStamina == null ? '—' : `${Math.round(p.potentialStamina)}%`}`,
      })
    else if (trace.dataset.triTrace === 'cadence')
      surfaces.push({
        wrap: trace,
        fmt: (p, i) =>
          `${scrubDist(presentation, p.d, d.sport)} · ${Math.round(cadenceValues?.[i] ?? p.cad * cadenceScale)} ${cadenceUnit}`,
      })
    else if (
      trace.dataset.triTrace === 'stride-length' ||
      trace.dataset.triTrace === 'estimated-stride-length'
    )
      surfaces.push({
        wrap: trace,
        fmt: p => {
          const stride = runStrideLengthValue(d, p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${stride == null ? '—' : formatStrideLength(presentation, stride)}`
        },
      })
    else if (trace.dataset.triTrace === 'ground-contact-time')
      surfaces.push({
        wrap: trace,
        fmt: p => {
          const groundContact = runGroundContactTimeMs(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${groundContact == null ? '—' : formatGroundContactTime(groundContact)}`
        },
      })
    else if (trace.dataset.triTrace === 'vertical-oscillation')
      surfaces.push({
        wrap: trace,
        fmt: p => {
          const verticalOscillation = runVerticalOscillationCm(p)
          return `${scrubDist(presentation, p.d, d.sport)} · ${verticalOscillation == null ? '—' : formatVerticalOscillation(presentation, verticalOscillation)}`
        },
      })
    else if (trace.dataset.triTrace === 'respiration')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.resp == null ? '—' : formatRespirationRate(p.resp)}`,
      })
    else if (trace.dataset.triTrace === 'temperature')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.tempC == null ? '—' : formatTemperature(presentation, p.tempC)}`,
      })
    else if (trace.dataset.triTrace === 'heat-strain-index')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.heatStrainIndex == null ? '—' : p.heatStrainIndex.toFixed(1)}`,
      })
    else if (trace.dataset.triTrace === 'core-temperature')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.coreTemperatureC == null ? '—' : formatThermalTemperature(presentation, p.coreTemperatureC)}`,
      })
    else if (trace.dataset.triTrace === 'skin-temperature')
      surfaces.push({
        wrap: trace,
        fmt: p =>
          `${scrubDist(presentation, p.d, d.sport)} · ${p.skinTemperatureC == null ? '—' : formatThermalTemperature(presentation, p.skinTemperatureC)}`,
      })
  }
  const interactive =
    (surfaces.length > 0 || wrap.querySelector('[data-tri-analysis]')) && d.route.length >= 2
  return {
    element: wrap,
    mount: () => {
      if (!interactive) return () => {}
      const routeMarker = wrap.querySelector<SVGElement>('.tri-route-cursor')
      const controller = linkScrub(presentation, wrap, routeMarker, surfaces, d.route, d)
      return () => controller?.dispose()
    },
  }
}
