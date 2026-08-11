import type { PowerCurvePoint } from '../../../plugins/stores/strava'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { ActivityComparisonMetric } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { ActivityComparisonDragSelection } from './comparison-graph'
import type { ActivityComparisonScrubState } from './comparison-graph'
import type { ActivityComparisonSelectionRange } from './comparison-graph'
import { activityComparisonDisplayValueAtDistance } from '../../../util/triathlon-card'
import { activityComparisonFractionForKey } from '../../../util/triathlon-card'
import { activityComparisonMapPointAtDistance } from '../../../util/triathlon-card'
import { activityComparisonMetricsForSport } from '../../../util/triathlon-card'
import { activityGearRatioDistribution } from '../../../util/triathlon-card'
import { activityPowerDistributionPercentages } from '../../../util/triathlon-card'
import { activityZonePercentages } from '../../../util/triathlon-card'
import { buildActivityComparisonProjection } from '../../../util/triathlon-card'
import { decodePowerCurve } from '../../../util/triathlon-card'
import { dlabel } from '../../../util/triathlon-card'
import { nearestPowerCurveValue } from '../../../util/triathlon-card'
import { normalizePowerCurvePoints } from '../../../util/triathlon-card'
import { powerCurveFraction } from '../../../util/triathlon-card'
import { scrubDist } from '../../../util/triathlon-card'
import { powerCurveReferenceLabel } from '../../../util/triathlon-i18n'
import { triText } from '../../../util/triathlon-i18n'
import { activityComparisonMetric } from './comparison-graph'
import { activityComparisonMetricLabel } from './comparison-graph'
import { bindActivityComparisonGraph } from './comparison-graph'
import { positionActivityComparisonCursor } from './comparison-graph'

export const wireActivityComparison = (
  presentation: TriathlonPresentation,
  comparison: HTMLElement | SVGElement,
  activities: StravaActivityDetail[],
): (() => void) => {
  const text = (key: string): string => triText(presentation.locale, key)
  const cleanups: (() => void)[] = []
  const projection = buildActivityComparisonProjection(activities)
  const charts = Array.from(
    comparison.querySelectorAll<HTMLElement>('.tri-compare-chart[data-compare-chart]'),
  )
  const chartScroller = comparison.querySelector<HTMLElement>('.tri-compare-charts')
  let chartOverflowFrame = 0
  const updateChartOverflow = () => {
    if (!chartScroller) return
    comparison.classList.toggle('tri-compare--top', chartScroller.scrollTop > 4)
    comparison.classList.toggle(
      'tri-compare--more',
      chartScroller.scrollHeight - chartScroller.clientHeight - chartScroller.scrollTop > 4,
    )
  }
  const scheduleChartOverflowUpdate = () => {
    if (chartOverflowFrame !== 0) return
    chartOverflowFrame = window.requestAnimationFrame(() => {
      chartOverflowFrame = 0
      updateChartOverflow()
    })
  }
  const chartResize = new ResizeObserver(scheduleChartOverflowUpdate)
  if (chartScroller) {
    chartScroller.addEventListener('scroll', scheduleChartOverflowUpdate, { passive: true })
    chartResize.observe(chartScroller)
    for (const chart of charts) chartResize.observe(chart)
    scheduleChartOverflowUpdate()
    cleanups.push(() => {
      chartScroller.removeEventListener('scroll', scheduleChartOverflowUpdate)
      chartResize.disconnect()
      if (chartOverflowFrame) window.cancelAnimationFrame(chartOverflowFrame)
      comparison.classList.remove('tri-compare--top', 'tri-compare--more')
    })
  }
  const distanceCharts = charts.flatMap(chart => {
    const metric = activityComparisonMetric(chart.dataset.compareChart)
    const graph = chart.querySelector<SVGElement>('.tri-compare-graph')
    return metric && graph && Number(chart.dataset.available) > 0 ? [{ chart, graph, metric }] : []
  })
  const map = comparison.querySelector<SVGSVGElement>('.tri-compare-map')
  const readout = comparison.querySelector<HTMLElement>('[data-compare-readout]')
  const maxDistanceKm = Number(
    map?.dataset.domainXMax ?? distanceCharts[0]?.graph.dataset.domainXMax ?? 0,
  )
  const activeSources = new Map<SVGElement, () => void>()
  let activeSource: SVGElement | null = null

  const setReadout = (
    mode:
      | ActivityComparisonMetric
      | 'power-curve'
      | 'power-distribution'
      | 'gear-ratio-distribution'
      | 'hr-zones'
      | 'power-zones',
    values: { activity: StravaActivityDetail; value: string; missing: boolean }[],
  ) => {
    if (!readout) return
    readout.dataset.visible = 'true'
    readout.dataset.compareReadoutMode = mode
    for (const { activity, value, missing } of values) {
      const row = readout.querySelector<HTMLElement>(
        `.tri-compare-readout-row[data-activity-id="${activity.id}"]`,
      )
      const valueNode = row?.querySelector<HTMLElement>('[data-compare-readout-value]')
      if (valueNode) valueNode.textContent = value
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
    for (const cursor of comparison.querySelectorAll<SVGElement>(
      '.tri-compare-route-cursor[data-activity-id]',
    ))
      cursor.setAttribute('data-visible', 'false')
  }

  const activate = (source: SVGElement, restore: () => void) => {
    activeSources.set(source, restore)
  }
  const render = (source: SVGElement, restore: () => void) => {
    activeSources.delete(source)
    activeSources.set(source, restore)
    activeSource = source
    restore()
  }
  const deactivate = (source: SVGElement) => {
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
    },
    commit: () => {
      if (!previewSelection) return
      lockedSelection = previewSelection
      previewSelection = null
      showSelection(lockedSelection, false)
    },
    clear: () => {
      lockedSelection = null
      previewSelection = null
      showSelection(null, false)
    },
    restore: () => {
      previewSelection = null
      showSelection(lockedSelection, false)
    },
  }
  let activeMetric: ActivityComparisonMetric = comparisonMetrics[0] ?? 'elevation'
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
        value: metrics.find(reading => reading.metric === metric)?.value ?? '—',
      }))
      graph.setAttribute('aria-valuenow', distanceKm.toFixed(3))
      graph.setAttribute(
        'aria-valuetext',
        `${position}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    setReadout(
      activeMetric,
      readings.map(({ activity, metrics }) => {
        const value = metrics.find(reading => reading.metric === activeMetric)?.value ?? '—'
        return { activity, value, missing: value === '—' }
      }),
    )
    map?.setAttribute('aria-valuenow', distanceKm.toFixed(3))
    map?.setAttribute(
      'aria-valuetext',
      `${text(activityComparisonMetricLabel(activeMetric))}; ${position}; ${readings
        .map(({ activity, metrics }) => {
          const value = metrics.find(reading => reading.metric === activeMetric)?.value ?? '—'
          return `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`
        })
        .join('; ')}`,
    )
    for (const route of projection.routes) {
      const cursor = comparison.querySelector<SVGElement>(
        `.tri-compare-route-cursor[data-activity-id="${route.activityId}"]`,
      )
      const point = activityComparisonMapPointAtDistance(route.pointSegments, distanceKm)
      if (!cursor || !point) {
        cursor?.setAttribute('data-visible', 'false')
        continue
      }
      cursor.setAttribute('cx', point.x.toFixed(2))
      cursor.setAttribute('cy', point.y.toFixed(2))
      cursor.setAttribute('data-visible', 'true')
    }
  }

  for (const { graph } of distanceCharts)
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

  if (map && Number(map.dataset.available) > 0) {
    let frame = 0
    let pending: PointerEvent | null = null
    let pointerActive = false
    let focused = false
    const restore = () => showDistance(distanceState.fraction)
    const release = () => {
      if (pointerActive || focused) return
      pending = null
      if (frame) {
        window.cancelAnimationFrame(frame)
        frame = 0
      }
      deactivate(map)
    }
    const onMapMove = (event: PointerEvent) => {
      if (!pointerActive) {
        pointerActive = true
        activate(map, restore)
      }
      pending = event
      if (frame) return
      frame = window.requestAnimationFrame(() => {
        frame = 0
        if (!pending) return
        const matrix = map.getScreenCTM()
        if (!matrix) return
        const pointer = new DOMPoint(pending.clientX, pending.clientY).matrixTransform(
          matrix.inverse(),
        )
        if (!Number.isFinite(pointer.x) || !Number.isFinite(pointer.y)) return
        let nearestDistanceKm: number | null = null
        let nearestSquared = Infinity
        for (const route of projection.routes)
          for (const segment of route.pointSegments)
            for (const point of segment) {
              const squared = (point.x - pointer.x) ** 2 + (point.y - pointer.y) ** 2
              if (squared >= nearestSquared) continue
              nearestSquared = squared
              nearestDistanceKm = point.distanceKm
            }
        if (nearestDistanceKm != null && maxDistanceKm > 0) {
          distanceState.fraction = nearestDistanceKm / maxDistanceKm
          render(map, restore)
        }
        pending = null
      })
    }
    const onMapLeave = () => {
      pointerActive = false
      release()
    }
    const onMapFocus = () => {
      focused = true
      activate(map, restore)
      render(map, restore)
    }
    const onMapBlur = () => {
      focused = false
      release()
    }
    const onMapKeyDown = (event: KeyboardEvent) => {
      const next = activityComparisonFractionForKey(event.key, distanceState.fraction, 0.01)
      if (next == null) return
      event.preventDefault()
      distanceState.fraction = Math.min(1, Math.max(0, next))
      render(map, restore)
    }
    map.addEventListener('pointermove', onMapMove)
    map.addEventListener('pointerleave', onMapLeave)
    map.addEventListener('pointercancel', onMapLeave)
    map.addEventListener('focus', onMapFocus)
    map.addEventListener('blur', onMapBlur)
    map.addEventListener('keydown', onMapKeyDown)
    cleanups.push(() => {
      map.removeEventListener('pointermove', onMapMove)
      map.removeEventListener('pointerleave', onMapLeave)
      map.removeEventListener('pointercancel', onMapLeave)
      map.removeEventListener('focus', onMapFocus)
      map.removeEventListener('blur', onMapBlur)
      map.removeEventListener('keydown', onMapKeyDown)
      if (frame) window.cancelAnimationFrame(frame)
      deactivate(map)
    })
  }

  const curveChart = charts.find(chart => chart.dataset.compareChart === 'power-curve')
  const curveGraph = curveChart?.querySelector<SVGElement>('.tri-compare-graph')
  const powerCurves = activities.map(activity =>
    normalizePowerCurvePoints(activity.powerCurve ?? []),
  )
  const curves = powerCurves.filter(curve => curve.length >= 2)
  if (curveChart && curveGraph && curves.length > 0) {
    type ComparisonCurveRange = 'six-weeks' | 'year'
    const curveState: ActivityComparisonScrubState = { fraction: 0, selectedFraction: 0 }
    const minDurationS = Math.min(...curves.map(curve => curve[0].s))
    const maxDurationS = Math.max(...curves.map(curve => curve[curve.length - 1].s))
    const curveReferences: Record<ComparisonCurveRange, PowerCurvePoint[]> = {
      'six-weeks': decodePowerCurve(curveGraph.dataset.curveRefSixWeeks),
      year: decodePowerCurve(curveGraph.dataset.curveRefYear),
    }
    let curveRange: ComparisonCurveRange =
      curveGraph.dataset.curveRange === 'year' ? 'year' : 'six-weeks'
    const rawCurveYear = Number(curveGraph.dataset.curveYear)
    const curveYear =
      curveGraph.dataset.curveYear && Number.isInteger(rawCurveYear) ? rawCurveYear : null
    const durationAt = (fraction: number): number =>
      Math.exp(
        Math.log(minDurationS) +
          Math.min(1, Math.max(0, fraction)) * (Math.log(maxDurationS) - Math.log(minDurationS)),
      )
    const showCurve = (fraction: number) => {
      curveState.fraction = Math.min(1, Math.max(0, fraction))
      const durationS = durationAt(curveState.fraction)
      positionActivityComparisonCursor(curveGraph, curveState.fraction)
      const values = activities.map((activity, index) => {
        const value = nearestPowerCurveValue(powerCurves[index], durationS)
        return { activity, value: value == null ? '—' : `${value.toLocaleString()} W` }
      })
      showChartCursors([curveChart])
      setReadout(
        'power-curve',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '—' })),
      )
      const referenceWatts = nearestPowerCurveValue(curveReferences[curveRange], durationS)
      const selectedSeconds = Math.max(1, Math.round(durationAt(curveState.selectedFraction ?? 0)))
      for (const tick of curveChart.querySelectorAll<HTMLButtonElement>('.tri-curve-tick'))
        tick.setAttribute(
          'aria-pressed',
          String(Number(tick.dataset.curveSeconds) === selectedSeconds),
        )
      curveGraph.setAttribute('aria-valuenow', Math.round(durationS).toString())
      curveGraph.setAttribute(
        'aria-valuetext',
        `${dlabel(Math.max(1, Math.round(durationS)))}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`,
          )
          .join('; ')}${
          referenceWatts == null
            ? ''
            : `; ${powerCurveReferenceLabel(presentation.locale, curveRange === 'year' ? curveYear : null)}: ${referenceWatts.toLocaleString()} W`
        }`,
      )
    }
    const selectCurve = (fraction: number) => {
      const selected = Math.min(1, Math.max(0, fraction))
      curveState.fraction = selected
      curveState.selectedFraction = selected
      showCurve(selected)
      curveGraph.focus({ preventScroll: true })
    }
    const selectCurveRange = (range: ComparisonCurveRange) => {
      if (curveReferences[range].length < 2) return
      curveRange = range
      curveGraph.dataset.curveRange = range
      for (const option of curveChart.querySelectorAll<HTMLButtonElement>('.tri-curve-range'))
        option.setAttribute('aria-pressed', String(option.dataset.curveRange === range))
      for (const path of curveGraph.querySelectorAll<SVGElement>(
        '.tri-compare-curve-ref[data-curve-range]',
      ))
        path.toggleAttribute('hidden', path.dataset.curveRange !== range)
      const label = curveChart.querySelector<HTMLElement>('.tri-compare-curve-reference-label')
      if (label) {
        label.removeAttribute('data-i18n')
        label.textContent = powerCurveReferenceLabel(
          presentation.locale,
          range === 'year' ? curveYear : null,
        )
      }
      showCurve(curveState.selectedFraction ?? curveState.fraction)
    }
    const onCurveClick = (event: MouseEvent) => {
      if (!(event.target instanceof Element)) return
      const rangeButton = event.target.closest<HTMLButtonElement>('.tri-curve-range')
      if (rangeButton && !rangeButton.disabled) {
        selectCurveRange(rangeButton.dataset.curveRange === 'year' ? 'year' : 'six-weeks')
        curveGraph.focus({ preventScroll: true })
        return
      }
      const tick = event.target.closest<HTMLButtonElement>('.tri-curve-tick')
      if (tick) {
        const seconds = Number(tick.dataset.curveSeconds)
        if (seconds > 0) selectCurve(powerCurveFraction(seconds, minDurationS, maxDurationS))
        return
      }
      const axis = event.target.closest<HTMLElement>('.tri-cax-xax')
      if (!axis || !curveChart.contains(axis)) return
      const bounds = axis.getBoundingClientRect()
      if (bounds.width > 0) selectCurve((event.clientX - bounds.left) / bounds.width)
    }
    curveChart.addEventListener('click', onCurveClick)
    cleanups.push(
      bindActivityComparisonGraph(curveGraph, curveState, showCurve, activate, render, deactivate),
      () => curveChart.removeEventListener('click', onCurveClick),
    )
  }

  const distributionChart = charts.find(
    chart => chart.dataset.compareChart === 'power-distribution',
  )
  const distributionGraph = distributionChart?.querySelector<SVGElement>(
    '.tri-compare-distribution-graph',
  )
  const distributions = activities.map(activity =>
    activityPowerDistributionPercentages(activity.powerHist),
  )
  const binCount = Math.max(0, ...distributions.map(values => values.length))
  if (distributionChart && distributionGraph && binCount >= 2) {
    const distributionState: ActivityComparisonScrubState = { fraction: 0 }
    const showDistribution = (fraction: number) => {
      distributionState.fraction = Math.min(1, Math.max(0, fraction))
      const index = Math.round(distributionState.fraction * (binCount - 1))
      const selectedFraction = index / (binCount - 1)
      const startWatts = index * 25
      positionActivityComparisonCursor(distributionGraph, selectedFraction)
      const values = activities.map((activity, activityIndex) => {
        const distribution = distributions[activityIndex]
        const value = distribution.length === 0 ? null : (distribution[index] ?? 0)
        return { activity, value: value == null ? '—' : `${value.toFixed(1)}%` }
      })
      showChartCursors([distributionChart])
      setReadout(
        'power-distribution',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '—' })),
      )
      distributionGraph.setAttribute('aria-valuenow', `${startWatts}`)
      distributionGraph.setAttribute(
        'aria-valuetext',
        `${startWatts}–${startWatts + 24} W; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    cleanups.push(
      bindActivityComparisonGraph(
        distributionGraph,
        distributionState,
        showDistribution,
        activate,
        render,
        deactivate,
        1 / (binCount - 1),
      ),
    )
  }

  const gearRatioChart = charts.find(
    chart => chart.dataset.compareChart === 'gear-ratio-distribution',
  )
  const gearRatioGraph = gearRatioChart?.querySelector<SVGElement>(
    '.tri-compare-distribution-graph',
  )
  const gearRatioDistributions = activities.map(activity => activityGearRatioDistribution(activity))
  const gearRatios = [
    ...new Set(gearRatioDistributions.flatMap(points => points.map(point => point.ratio))),
  ].sort((left, right) => left - right)
  if (gearRatioChart && gearRatioGraph && gearRatios.length > 0) {
    const gearRatioState: ActivityComparisonScrubState = { fraction: 0 }
    const showGearRatio = (fraction: number) => {
      const index = Math.round(Math.min(1, Math.max(0, fraction)) * (gearRatios.length - 1))
      const selectedFraction = gearRatios.length <= 1 ? 0.5 : index / (gearRatios.length - 1)
      gearRatioState.fraction = selectedFraction
      const ratio = gearRatios[index]
      positionActivityComparisonCursor(gearRatioGraph, selectedFraction)
      const values = activities.map((activity, activityIndex) => {
        const distribution = gearRatioDistributions[activityIndex]
        const percentage = distribution.find(point => point.ratio === ratio)?.percentage
        const value = distribution.length === 0 ? '—' : `${(percentage ?? 0).toFixed(1)}%`
        return { activity, value }
      })
      showChartCursors([gearRatioChart])
      setReadout(
        'gear-ratio-distribution',
        values.map(({ activity, value }) => ({ activity, value, missing: value === '—' })),
      )
      gearRatioGraph.setAttribute('aria-valuenow', `${index}`)
      gearRatioGraph.setAttribute(
        'aria-valuetext',
        `${ratio.toFixed(2)}×; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    cleanups.push(
      bindActivityComparisonGraph(
        gearRatioGraph,
        gearRatioState,
        showGearRatio,
        activate,
        render,
        deactivate,
        gearRatios.length <= 1 ? 1 : 1 / (gearRatios.length - 1),
      ),
    )
  }

  const zoneKinds: readonly ('hr' | 'power')[] = ['hr', 'power']
  for (const kind of zoneKinds) {
    const chart = charts.find(candidate => candidate.dataset.compareChart === `${kind}-zones`)
    const graph = chart?.querySelector<SVGElement>('.tri-compare-graph')
    const zones = activities.map(activity =>
      activityZonePercentages(kind === 'hr' ? activity.hrZones : activity.powerZones),
    )
    const zoneCount = Math.max(0, ...zones.map(values => values.length))
    if (!chart || !graph || zoneCount === 0) continue
    const zoneState: ActivityComparisonScrubState = { fraction: 0 }
    const showZone = (fraction: number) => {
      zoneState.fraction = Math.min(1, Math.max(0, fraction))
      const index = Math.round(zoneState.fraction * Math.max(0, zoneCount - 1))
      const selectedFraction = zoneCount <= 1 ? 0 : index / (zoneCount - 1)
      positionActivityComparisonCursor(graph, selectedFraction)
      const values = activities.map((activity, activityIndex) => {
        const value = zones[activityIndex][index]
        return { activity, value: value == null ? '—' : `${value.toFixed(1)}%` }
      })
      showChartCursors([chart])
      setReadout(
        `${kind}-zones`,
        values.map(({ activity, value }) => ({ activity, value, missing: value === '—' })),
      )
      graph.setAttribute('aria-valuenow', `${index + 1}`)
      graph.setAttribute(
        'aria-valuetext',
        `Z${index + 1}; ${values
          .map(
            ({ activity, value }) =>
              `${activity.name || text(activity.sport)}: ${value === '—' ? text('no data') : value}`,
          )
          .join('; ')}`,
      )
    }
    const keyboardStep = zoneCount <= 1 ? 1 : 1 / (zoneCount - 1)
    cleanups.push(
      bindActivityComparisonGraph(
        graph,
        zoneState,
        showZone,
        activate,
        render,
        deactivate,
        keyboardStep,
      ),
    )
  }

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
  const toggle = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('.tri-act-toggle')
  const activity = toggle?.closest<HTMLElement>('.tri-act')
  if (activity) setActivityExpanded(activity, !activity.classList.contains('tri-act--expanded'))
}
