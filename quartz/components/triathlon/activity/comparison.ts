import type { ActivityHealth, StravaActivityDetail } from '../../../plugins/stores/strava'
import type { ActivityComparisonWindow } from '../../../util/activity-comparison-metrics'
import type { ActivityComparisonMetric } from '../../../util/triathlon-card'
import type { ActivityComparisonStatsScope } from '../../../util/triathlon-card'
import type { DetailCtx } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { ActivityComparisonDragSelection } from './comparison-graph'
import type { ActivityComparisonReading } from './comparison-graph'
import type { ActivityComparisonReadoutMode } from './comparison-graph'
import type { ActivityComparisonScrubContext } from './comparison-graph'
import type { ActivityComparisonScrubState } from './comparison-graph'
import type { ActivityComparisonSelectionRange } from './comparison-graph'
import type { ActivityComparisonMapController } from './comparison-map'
import { activityComparisonMathText } from '../../../util/triathlon-card'
import { activityComparisonDisplayValueAtDistance } from '../../../util/triathlon-card'
import { activityComparisonMetricsForSport } from '../../../util/triathlon-card'
import { buildActivityComparisonStats } from '../../../util/triathlon-card'
import { scrubDist } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { applyI18n } from '../runtime/dom'
import { createDomFactory, setMath } from '../runtime/dom'
import { setupPowerCurveTicks } from '../shell/power-curve-ticks'
import { wireComparisonBinnedCharts } from './comparison-binned'
import { mountComparisonChartTabs } from './comparison-chart-tabs'
import { wireComparisonPowerCurve } from './comparison-curve'
import { activityComparisonMetric } from './comparison-graph'
import { activityComparisonMetricLabel } from './comparison-graph'
import { bindActivityComparisonGraph } from './comparison-graph'
import { positionActivityComparisonCursor } from './comparison-graph'
import { mountActivityComparisonMap } from './comparison-map'
import { buildActivityMapControls } from './map-controls'

export const wireActivityComparison = (
  presentation: TriathlonPresentation,
  comparison: HTMLElement | SVGElement,
  activities: StravaActivityDetail[],
  statsContext: { ctx?: DetailCtx; health?: Readonly<Record<string, ActivityHealth>> | null } = {},
): (() => void) => {
  const text = (key: string): string => triText(presentation.locale, key)
  const setComparisonMath = (node: HTMLElement, value: string): void => {
    const math = activityComparisonMathText(value)
    if (math !== value) {
      node.setAttribute('role', 'img')
      node.setAttribute('aria-label', value)
    } else if (node.getAttribute('role') === 'img') {
      node.removeAttribute('role')
      node.removeAttribute('aria-label')
    }
    setMath(node, math)
  }
  for (const node of comparison.querySelectorAll<HTMLElement>(
    '.tri-cax-yt, .tri-cax-xt:not(button), .tri-curve-cp-k, .tri-critical-power-mechanics, .tri-critical-power-anchor-separator',
  )) {
    const value = node.textContent ?? ''
    if (activityComparisonMathText(value) !== value) setComparisonMath(node, value)
  }
  const cleanups: (() => void)[] = [setupPowerCurveTicks(comparison)]
  const charts = Array.from(
    comparison.querySelectorAll<HTMLElement>('.tri-compare-chart[data-compare-chart]'),
  )
  const distanceCharts = charts.flatMap(chart => {
    const metric = activityComparisonMetric(chart.dataset.compareChart)
    const graph = chart.querySelector<SVGElement>('.tri-compare-graph')
    return metric && graph && Number(chart.dataset.available) > 0 ? [{ chart, graph, metric }] : []
  })
  const map = comparison.querySelector<HTMLElement>('.tri-compare-map')
  const readout = comparison.querySelector<HTMLElement>('[data-compare-readout]')
  const readoutContext = readout?.querySelector<HTMLElement>('[data-compare-readout-context]')
  const maxDistanceKm = Number(
    map?.dataset.domainXMax ?? distanceCharts[0]?.graph.dataset.domainXMax ?? 0,
  )
  const activeSources = new Map<Element, () => void>()
  let activeSource: Element | null = null
  let mapController: ActivityComparisonMapController | null = null
  const hiddenActivities = new Set<string>()

  let statsScope: ActivityComparisonStatsScope = 'whole'
  let statsWindow: ActivityComparisonWindow | null = null
  let statsFrame = 0
  const renderStats = () => {
    const current = comparison.querySelector<HTMLElement>('[data-compare-stats]')
    if (!current) return
    const scroller = current.querySelector<HTMLElement>('.tri-compare-stats-scroll')
    const scrollTop = scroller?.scrollTop ?? 0
    const scrollLeft = scroller?.scrollLeft ?? 0
    const focused =
      document.activeElement instanceof HTMLElement && current.contains(document.activeElement)
        ? document.activeElement
        : null
    const focusSelector = focused?.dataset.compareScope
      ? `[data-compare-scope="${focused.dataset.compareScope}"]`
      : focused?.classList.contains('tri-compare-stats-scroll')
        ? '.tri-compare-stats-scroll'
        : null
    const next = buildActivityComparisonStats(
      createDomFactory(presentation),
      activities,
      statsContext.ctx,
      {
        hiddenIds: hiddenActivities,
        window: statsWindow,
        scope: statsScope,
        health: statsContext.health,
      },
    ) as HTMLElement
    applyI18n(next, presentation)
    current.replaceWith(next)
    const nextScroller = next.querySelector<HTMLElement>('.tri-compare-stats-scroll')
    if (nextScroller) {
      nextScroller.scrollTop = scrollTop
      nextScroller.scrollLeft = scrollLeft
    }
    if (focusSelector)
      next.querySelector<HTMLElement>(focusSelector)?.focus({ preventScroll: true })
  }
  // Drag previews fire per pointer move; one table rebuild per frame is enough.
  const scheduleStats = () => {
    if (statsFrame !== 0) return
    statsFrame = requestAnimationFrame(() => {
      statsFrame = 0
      renderStats()
    })
  }
  const selectionWindow = (
    selection: ActivityComparisonSelectionRange | null,
  ): ActivityComparisonWindow | null =>
    selection && maxDistanceKm > 0 && selection.endFraction > selection.startFraction
      ? {
          startKm: selection.startFraction * maxDistanceKm,
          endKm: selection.endFraction * maxDistanceKm,
        }
      : null
  const scopeStatsToSelection = (selection: ActivityComparisonSelectionRange | null) => {
    statsWindow = selectionWindow(selection)
    statsScope = statsWindow ? 'selection' : 'whole'
    scheduleStats()
  }
  cleanups.push(() => {
    if (statsFrame !== 0) cancelAnimationFrame(statsFrame)
    statsFrame = 0
  })

  const setReadout = (
    mode: ActivityComparisonReadoutMode,
    values: ActivityComparisonReading[],
    context = '',
  ) => {
    if (!readout) return
    const head =
      charts
        .find(chart => chart.dataset.compareChart === mode && !chart.hidden)
        ?.querySelector<HTMLElement>('.tri-compare-chart-head') ??
      map?.closest('.tri-compare-map-panel')?.querySelector<HTMLElement>('.tri-compare-chart-head')
    if (head && readout.parentElement !== head) head.append(readout)
    readout.dataset.visible = 'true'
    readout.dataset.compareReadoutMode = mode
    if (readoutContext) {
      setComparisonMath(readoutContext, context)
      readoutContext.hidden = context.length === 0
    }
    for (const { activity, value, missing } of values) {
      const row = readout.querySelector<HTMLElement>(
        `.tri-compare-readout-row[data-activity-id="${activity.id}"]`,
      )
      const valueNode = row?.querySelector<HTMLElement>('[data-compare-readout-value]')
      if (valueNode) setComparisonMath(valueNode, value)
      row?.classList.toggle('tri-compare-readout-row--missing', missing)
    }
  }

  const showChartCursors = (visible: readonly HTMLElement[]) => {
    for (const chart of charts)
      chart.classList.toggle('tri-compare-chart--hover', visible.includes(chart))
  }
  const hide = () => {
    showChartCursors([])
    if (readout) readout.dataset.visible = 'false'
    mapController?.hideCursors()
  }

  const activate = (source: Element, restore: () => void) => {
    activeSources.set(source, restore)
  }
  const render = (source: Element, restore: () => void) => {
    activeSources.delete(source)
    activeSources.set(source, restore)
    activeSource = source
    restore()
  }
  const deactivate = (source: Element) => {
    activeSources.delete(source)
    if (activeSource !== source) return
    const previous = Array.from(activeSources.entries()).at(-1)
    activeSource = previous?.[0] ?? null
    if (previous) previous[1]()
    else hide()
  }

  const distanceState: ActivityComparisonScrubState = { fraction: 0 }
  const comparisonMetrics = activityComparisonMetricsForSport(activities[0].sport)
  let lockedSelection: ActivityComparisonSelectionRange | null = null
  let previewSelection: ActivityComparisonSelectionRange | null = null
  const showSelection = (
    selection: ActivityComparisonSelectionRange | null,
    selecting: boolean,
  ): void => {
    comparison.classList.toggle('tri-compare--selection', selection != null)
    comparison.classList.toggle('tri-compare--selecting', selection != null && selecting)
    const start = selection?.startFraction ?? 0
    const end = selection?.endFraction ?? 0
    for (const { graph } of distanceCharts) {
      const region = graph.querySelector<SVGRectElement>('.tri-compare-selection-region')
      const clip = graph.querySelector<SVGRectElement>('.tri-compare-selection-clip')
      const x = (start * 100).toFixed(2)
      const width = ((end - start) * 100).toFixed(2)
      region?.setAttribute('x', x)
      region?.setAttribute('width', width)
      clip?.setAttribute('x', x)
      clip?.setAttribute('width', width)
    }
  }
  const distanceSelection: ActivityComparisonDragSelection = {
    preview: (anchorFraction, focusFraction) => {
      previewSelection = {
        startFraction: Math.min(anchorFraction, focusFraction),
        endFraction: Math.max(anchorFraction, focusFraction),
      }
      showSelection(previewSelection, true)
      scopeStatsToSelection(previewSelection)
    },
    commit: () => {
      if (!previewSelection) return
      lockedSelection = previewSelection
      previewSelection = null
      showSelection(lockedSelection, false)
      scopeStatsToSelection(lockedSelection)
    },
    clear: () => {
      lockedSelection = null
      previewSelection = null
      showSelection(null, false)
      scopeStatsToSelection(null)
    },
    restore: () => {
      previewSelection = null
      showSelection(lockedSelection, false)
      scopeStatsToSelection(lockedSelection)
    },
  }
  let activeMetric: ActivityComparisonMetric = comparisonMetrics[0] ?? 'elevation'
  cleanups.push(
    mountComparisonChartTabs(comparison, presentation, charts, chart => {
      const metric = activityComparisonMetric(chart?.dataset.compareChart)
      if (metric) activeMetric = metric
      activeSources.clear()
      activeSource = null
      hide()
    }),
  )
  const showDistance = (fraction: number, metric: ActivityComparisonMetric = activeMetric) => {
    if (!Number.isFinite(maxDistanceKm) || maxDistanceKm <= 0) return
    distanceState.fraction = Math.min(1, Math.max(0, fraction))
    activeMetric = metric
    const distanceKm = distanceState.fraction * maxDistanceKm
    const position = scrubDist(presentation, distanceKm, activities[0].sport)
    showChartCursors(distanceCharts.map(({ chart }) => chart))
    const readings = activities.map(activity => ({
      activity,
      metrics: comparisonMetrics.map(metric => {
        return {
          metric,
          value: activityComparisonDisplayValueAtDistance(
            presentation,
            activity,
            metric,
            distanceKm,
          ),
        }
      }),
    }))
    for (const { graph, metric } of distanceCharts) {
      positionActivityComparisonCursor(graph, distanceState.fraction)
      const values = readings.map(({ activity, metrics }) => ({
        activity,
        value: metrics.find(reading => reading.metric === metric)?.value ?? '–',
      }))
      graph.setAttribute('aria-valuenow', distanceKm.toFixed(3))
      graph.setAttribute(
        'aria-valuetext',
        `${position}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    setReadout(
      activeMetric,
      readings.map(({ activity, metrics }) => {
        const value = metrics.find(reading => reading.metric === activeMetric)?.value ?? '–'
        return { activity, value, missing: value === '–' }
      }),
    )
    map?.setAttribute('aria-valuenow', distanceKm.toFixed(3))
    map?.setAttribute(
      'aria-valuetext',
      `${text(activityComparisonMetricLabel(activeMetric))}; ${position}; ${readings
        .map(({ activity, metrics }) => {
          const value = metrics.find(reading => reading.metric === activeMetric)?.value ?? '–'
          return `${activity.name || text(activity.sport)}: ${value === '–' ? text('no data') : value}`
        })
        .join('; ')}`,
    )
    mapController?.showCursors(distanceKm)
  }

  for (const { graph } of distanceCharts) {
    const help = text('comparison keyboard help')
    graph.setAttribute('aria-description', help)
    graph.setAttribute('title', help)
    graph.setAttribute(
      'aria-keyshortcuts',
      'Shift+ArrowLeft Shift+ArrowRight Shift+Home Shift+End Escape',
    )
    cleanups.push(
      bindActivityComparisonGraph(
        graph,
        distanceState,
        fraction => {
          const metric = activityComparisonMetric(graph.dataset.compareChart)
          if (metric) showDistance(fraction, metric)
        },
        activate,
        render,
        deactivate,
        0.01,
        distanceSelection,
      ),
    )
  }

  if (map && Number(map.dataset.available) > 0 && maxDistanceKm > 0) {
    const restore = () => showDistance(distanceState.fraction)
    const controls = buildActivityMapControls(text)
    map.parentElement?.append(controls.element)
    cleanups.push(() => {
      controls.dispose()
      controls.element.remove()
    })
    mapController = mountActivityComparisonMap(map, activities, {
      unavailableText: text('map unavailable'),
      distance: presentation.distance,
      onScrub: distanceKm => {
        distanceState.fraction = Math.min(1, Math.max(0, distanceKm / maxDistanceKm))
        render(map, restore)
      },
      onLeave: () => deactivate(map),
    })
    cleanups.push(() => {
      mapController?.destroy()
      mapController = null
    })
  }

  const onLegendToggle = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('[data-compare-activity-toggle]')
    const activityId = button?.dataset.compareActivityToggle
    if (!button || !activityId) return
    const visible = hiddenActivities.has(activityId)
    if (visible) hiddenActivities.delete(activityId)
    else hiddenActivities.add(activityId)
    button.setAttribute('aria-pressed', String(visible))
    for (const node of comparison.querySelectorAll<HTMLElement | SVGElement>(
      `[data-activity-id="${activityId}"]`,
    ))
      node.setAttribute('data-compare-hidden', String(!visible))
    mapController?.setVisible(activityId, visible)
    renderStats()
  }
  comparison.addEventListener('click', onLegendToggle)
  cleanups.push(() => comparison.removeEventListener('click', onLegendToggle))

  const onStatsClick = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const scope = event.target.closest<HTMLButtonElement>('[data-compare-scope]')
    if (!scope || scope.disabled) return
    statsScope = scope.dataset.compareScope === 'selection' && statsWindow ? 'selection' : 'whole'
    renderStats()
  }
  comparison.addEventListener('click', onStatsClick)
  cleanups.push(() => comparison.removeEventListener('click', onStatsClick))

  const scrub: ActivityComparisonScrubContext = {
    presentation,
    activities,
    charts,
    text,
    setReadout,
    showChartCursors,
    activate,
    render,
    deactivate,
  }
  cleanups.push(...wireComparisonPowerCurve(scrub), ...wireComparisonBinnedCharts(scrub))

  return () => {
    distanceSelection.clear()
    for (const cleanup of cleanups) cleanup()
  }
}

export const setActivityExpanded = (activity: HTMLElement, expanded: boolean): void => {
  activity.classList.toggle('tri-act--expanded', expanded)
  const toggle = activity.querySelector<HTMLButtonElement>(':scope > .tri-act-toggle')
  if (!toggle) return
  toggle.setAttribute('aria-expanded', String(expanded))
  toggle.textContent = expanded ? '− see less' : '+ see more'
}

export const onCardToggle = (event: Event): void => {
  if (!(event.target instanceof Element)) return
  const toggle = event.target.closest<HTMLButtonElement>('.tri-act-toggle')
  const activity = toggle?.closest<HTMLElement>('.tri-act')
  if (!activity) return
  const expanded = !activity.classList.contains('tri-act--expanded')
  const activities = activity
    .closest('.tri-pop-card')
    ?.querySelectorAll<HTMLElement>(':scope > .tri-act, :scope > .tri-day-track > .tri-act') ?? [
    activity,
  ]
  for (const sibling of activities) setActivityExpanded(sibling, expanded)
}
