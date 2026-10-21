import type { StravaActivityDetail } from '../../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import {
  activityAnalysisStatAttrs,
  activityEquipmentStatAttrs,
  activitySourceStatAttrs,
  activityStatRows,
  activityTableRows,
  activityTrainingEffectLabel,
  buildActivityIcon,
  buildActivityShareButton,
  buildAnalysisBar,
  buildBestEfforts as buildBestEffortsNode,
  buildCrankTorqueChart,
  buildCyclingPowerChart,
  buildEnvironmentAnalysis as buildEnvironmentAnalysisNode,
  buildPerformanceConditionTrace as buildPerformanceConditionTraceNode,
  buildRunHeartCostChart,
  buildRunStrideCadenceChart,
  buildSaunaHeatTrainingLoad,
  buildStaminaChart as buildStaminaChartNode,
  buildSwimPowerCurve as buildSwimPowerCurveNode,
  buildTrainingEffectDetails as buildTrainingEffectDetailsNode,
  buildWalkPowerTrace as buildWalkPowerTraceNode,
  buildWorkoutAnalysis as buildWorkoutAnalysisNode,
  dominantTrainingEffectGroup,
  hasHeartRateTrace,
  powerViewActivity,
  relativeHumidityStatAttrs,
} from '../../../../util/triathlon-card'
import { buildMyWindsockGraphs } from '../../../../util/triathlon-mywindsock'
import { createDomFactory, el } from '../../runtime/dom'
import { nextMapMetricShortcutIndex } from '../../shell/command-palette'
import { setupPowerCurveTicks } from '../../shell/power-curve-ticks'
import {
  linkScrub,
  type ActivityAnalysisController,
  type ActivityAnalysisRange,
  type ActivityRangeChange,
  type ScrubSurface,
} from '../analysis'
import {
  cyclingPowerWindow,
  setCyclingPowerWindow,
  setupCyclingPowerCharts,
} from '../cycling-power'
import { detailContextFromPayload } from '../data'
import {
  buildHeartRateTrace,
  buildHrZones,
  buildPool,
  buildPowerCurve,
  buildPowerHist,
  buildPowerZones,
  statRow,
  zoneDuo,
} from '../primitives'
import {
  runHeartCostWindow,
  setRunHeartCostWindow,
  setupRunHeartCostCharts,
} from '../run-heart-cost'
import { cyclingChartMode, restoreShiftSettings, setCyclingChartMode } from '../scrub'
import { metricSpecs } from './metric-specs'
import {
  buildHeatLegend,
  buildHeatRoute,
  cyclingPowerScrubSurface,
  physiologyScrubSurfaces,
  RUN_STRIDE_TRACE_SELECTOR,
  runHeartCostScrubSurface,
  strideScrubSurface,
  torqueScrubSurface,
  walkPowerScrubSurface,
  type ActivityView,
  type MapDetailOpts,
} from './scrub-surfaces'

export const renderMapDetail = (
  presentation: TriathlonPresentation,
  source: StravaActivityDetail,
  opts?: MapDetailOpts,
): ActivityView => {
  const domF = createDomFactory(presentation)
  const d = powerViewActivity(presentation, source)
  const wrap = el('section', 'tri-act tri-act--expanded')
  const head = el('div', 'tri-act-head')
  head.appendChild(buildActivityIcon(domF, d))
  const share = buildActivityShareButton(domF, d)
  if (share) {
    const actions = el('div', 'tri-act-head-actions')
    actions.append(share)
    head.append(actions)
  }
  wrap.appendChild(head)

  const stats = el('table', 'tri-act-stats')
  const sbody = document.createElement('tbody')
  const summaryTrainingEffectGroup = dominantTrainingEffectGroup(activityTrainingEffectLabel(d))
  const primaryStatCount = activityStatRows(presentation, d, false).length
  activityTableRows(presentation, d).forEach(([label, value], index) =>
    sbody.appendChild(
      statRow(
        presentation,
        label,
        value,
        label === 'training effect'
          ? { 'data-training-effect-group': summaryTrainingEffectGroup }
          : ((label === 'source' ? activitySourceStatAttrs(d) : undefined) ??
              (label === 'equipment' ? activityEquipmentStatAttrs(d) : undefined) ??
              activityAnalysisStatAttrs(d, label) ??
              (index >= primaryStatCount && label === 'humidity'
                ? relativeHumidityStatAttrs(d)
                : undefined)),
      ),
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
    const analysis = d.sport === 'sauna' ? (buildAnalysisBar(domF, d) as HTMLElement | null) : null
    if (analysis) figs.appendChild(analysis)
    if (figs.childElementCount > 0) wrap.appendChild(figs)
    const more = el('div', 'tri-act-more')
    const workoutAnalysis = buildWorkoutAnalysisNode(
      domF,
      d,
      false,
      (opts?.detailContext ?? detailContextFromPayload()).zones,
      true,
    ) as HTMLElement | null
    const bestEfforts = buildBestEffortsNode(domF, d)
    const heartRate =
      hasHeartRateTrace(d) && workoutAnalysis?.dataset.workoutAnalysisMetric !== 'hr'
        ? buildHeartRateTrace(presentation, d)
        : null
    const environment = buildEnvironmentAnalysisNode(domF, d)
    const trainingEffect = buildTrainingEffectDetailsNode(domF, d) as HTMLElement | null
    for (const z of [
      workoutAnalysis,
      heartRate,
      buildStaminaChartNode(domF, d, null),
      buildPerformanceConditionTraceNode(domF, d, null),
      buildWalkPowerTraceNode(domF, d, null),
      trainingEffect,
      zoneDuo(
        presentation,
        buildHrZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
        buildPowerZones(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
      ),
      buildCrankTorqueChart(domF, d, null),
      buildCyclingPowerChart(domF, d, null),
      environment,
      buildMyWindsockGraphs(domF, d) as HTMLElement | null,
      buildSaunaHeatTrainingLoad(domF, d),
      buildPowerCurve(presentation, d, opts?.detailContext ?? detailContextFromPayload()),
      buildSwimPowerCurveNode(domF, d),
      buildPowerHist(presentation, d),
    ])
      if (z) more.appendChild(z)
    if (bestEfforts) more.appendChild(bestEfforts)
    if (more.childElementCount > 0) wrap.appendChild(more)
    return {
      element: wrap,
      mount: () => {
        const cleanupPowerTicks = setupPowerCurveTicks(wrap)
        const cleanupCyclingPower = setupCyclingPowerCharts(wrap, presentation, d)
        const cleanupRunHeartCost = setupRunHeartCostCharts(wrap, presentation, d)
        const controller = linkScrub(
          presentation,
          wrap,
          null,
          [
            ...physiologyScrubSurfaces(wrap, d, presentation),
            ...Array.from(
              wrap.querySelectorAll<HTMLElement>('[data-tri-trace="walking-power"]'),
              trace => walkPowerScrubSurface(trace, d, presentation),
            ),
            ...Array.from(
              wrap.querySelectorAll<HTMLElement>('[data-tri-trace="crank-torque"]'),
              trace => torqueScrubSurface(trace, d, presentation),
            ),
            ...Array.from(
              wrap.querySelectorAll<HTMLElement>('[data-tri-trace="cycling-power"]'),
              trace => cyclingPowerScrubSurface(trace, d, presentation, opts?.onHover),
            ),
          ],
          d.route,
          d,
          opts?.analysis ?? analysis,
          opts?.onRange,
        )
        return () => {
          cleanupPowerTicks()
          cleanupCyclingPower()
          cleanupRunHeartCost()
          controller?.dispose()
        }
      },
    }
  }

  const tablist = el('div', 'tri-map-tablist')
  tablist.setAttribute('role', 'tablist')
  const figs = el('div', 'tri-act-figs tri-map-figs')
  const profileBox = el('div', 'tri-map-profile')
  const workoutAnalysis = buildWorkoutAnalysisNode(
    domF,
    d,
    false,
    (opts?.detailContext ?? detailContextFromPayload()).zones,
    true,
  ) as HTMLElement | null
  const zoneBox = el('div', 'tri-act-more')
  const bestEfforts = buildBestEffortsNode(domF, d)
  wrap.append(tablist, figs)
  if (workoutAnalysis) wrap.appendChild(workoutAnalysis)
  wrap.appendChild(profileBox)
  wrap.appendChild(zoneBox)

  let active = Math.min(specs.length - 1, Math.max(0, opts?.initialMetric ?? 0))
  const routeSamples = d.route
  let graphDomain: ActivityAnalysisRange | null = null
  let routeMarker: SVGElement | null = null
  let analysisController: ActivityAnalysisController | null = null
  let cleanupPowerTicks: (() => void) | null = null
  let linkedSurfaces: ScrubSurface[] = []
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
    cleanupPowerTicks?.()
    cleanupPowerTicks = null
    const existingCyclingChartMode = cyclingChartMode(wrap)
    const existingPowerWindow = cyclingPowerWindow(
      zoneBox.querySelector<HTMLElement>('.tri-cycling-power:not(.tri-run-heart-cost)'),
    )
    const existingRunCostWindow = runHeartCostWindow(
      zoneBox.querySelector<HTMLElement>('.tri-run-heart-cost'),
    )
    const spec = specs[active]
    const profile = spec.profile(graphDomain)
    const traces = spec.traces?.(graphDomain) ?? []
    profileBox.replaceChildren(profile)
    zoneBox.replaceChildren(...traces.map(trace => trace.wrap))
    const torque = buildCrankTorqueChart(domF, d, null, graphDomain)
    if (torque) zoneBox.appendChild(torque)
    const cyclingPower = buildCyclingPowerChart(domF, d, graphDomain)
    if (cyclingPower) zoneBox.appendChild(cyclingPower)
    const strideChart = buildRunStrideCadenceChart(domF, d, null, graphDomain)
    if (strideChart) zoneBox.appendChild(strideChart)
    const runHeartCost = buildRunHeartCostChart(domF, d, graphDomain)
    if (runHeartCost) zoneBox.appendChild(runHeartCost)
    const environment = buildEnvironmentAnalysisNode(domF, d)
    if (environment) zoneBox.appendChild(environment)
    const windGraphs = buildMyWindsockGraphs(domF, d)
    if (windGraphs) zoneBox.appendChild(windGraphs)
    if (spec.extra) for (const node of spec.extra()) if (node) zoneBox.appendChild(node)
    const swimPower = buildSwimPowerCurveNode(domF, d)
    if (swimPower) zoneBox.appendChild(swimPower)
    if (d.sport === 'swim') {
      const distribution = buildPowerHist(presentation, d)
      if (distribution) zoneBox.appendChild(distribution)
    }
    if (bestEfforts) zoneBox.appendChild(bestEfforts)
    const cyclingChart = zoneBox.querySelector<HTMLElement>('.tri-cycling-mode-chart')
    if (cyclingChart) setCyclingChartMode(cyclingChart, existingCyclingChartMode)
    restoreShiftSettings(wrap)
    linkedSurfaces = [{ wrap: profile, fmt: spec.readout }, ...traces].map(surface =>
      surface.wrap.dataset.triTrace === 'walking-power'
        ? walkPowerScrubSurface(surface.wrap, d, presentation)
        : {
            wrap: surface.wrap,
            samples: routeSamples,
            fmt: index => {
              const point = d.route[index]
              opts?.onHover?.(point, index)
              return surface.fmt(point, index)
            },
          },
    )
    const torqueTrace = torque?.querySelector<HTMLElement>('[data-tri-trace="crank-torque"]')
    if (torqueTrace) linkedSurfaces.push(torqueScrubSurface(torqueTrace, d, presentation))
    if (cyclingPower instanceof HTMLElement) {
      setCyclingPowerWindow(cyclingPower, existingPowerWindow, presentation)
      linkedSurfaces.push(cyclingPowerScrubSurface(cyclingPower, d, presentation, opts?.onHover))
    }
    const strideTrace =
      strideChart instanceof HTMLElement
        ? strideChart.matches(RUN_STRIDE_TRACE_SELECTOR)
          ? strideChart
          : strideChart.querySelector<HTMLElement>(RUN_STRIDE_TRACE_SELECTOR)
        : null
    if (strideTrace) linkedSurfaces.push(strideScrubSurface(strideTrace, d, presentation))
    if (runHeartCost instanceof HTMLElement) {
      setRunHeartCostWindow(runHeartCost, existingRunCostWindow, presentation)
      linkedSurfaces.push(runHeartCostScrubSurface(runHeartCost, d, presentation, opts?.onHover))
    }
    if (mounted) mountProfile()
  }
  const mountProfile = (): void => {
    cleanupPowerTicks = setupPowerCurveTicks(zoneBox)
    analysisController = linkScrub(
      presentation,
      wrap,
      routeMarker,
      [
        ...linkedSurfaces,
        ...(workoutAnalysis ? physiologyScrubSurfaces(workoutAnalysis, d, presentation) : []),
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
    const pool = vals.filter((value, index) =>
      spec.valid && !spec.valid(d.route[index], index) ? false : !spec.zeroGap || value > 0,
    )
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
      const heat = buildHeatRoute(d.route, spec.pick, spec.ramp, spec.zeroGap, spec.valid)
      figs.replaceChildren(heat, buildHeatLegend(lo, hi, spec.fmt, spec.ramp))
      routeMarker = heat.querySelector<SVGElement>('.tri-route-cursor')
    }
    renderProfile()
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
    else
      next = nextMapMetricShortcutIndex(
        tabs.map(tab => tab.dataset.shortcut),
        active,
        event.key,
      )
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
      mountProfile()
      const cleanupCyclingPower = setupCyclingPowerCharts(wrap, presentation, d)
      const cleanupRunHeartCost = setupRunHeartCostCharts(wrap, presentation, d)
      tablist.addEventListener('click', onTabClick)
      tablist.addEventListener('keydown', onTabKeydown)
      return () => {
        mounted = false
        cleanupCyclingPower()
        cleanupRunHeartCost()
        tablist.removeEventListener('click', onTabClick)
        tablist.removeEventListener('keydown', onTabKeydown)
        analysisController?.dispose()
        analysisController = null
        cleanupPowerTicks?.()
        cleanupPowerTicks = null
      }
    },
  }
}
