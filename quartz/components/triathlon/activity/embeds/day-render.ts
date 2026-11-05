import type { DayCardExtras, SwimTrendMode } from '../../../../util/triathlon-card'
import type { RunHeartCostWindow } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import type { DetailPayload } from '../data'
import { swimChartMetric, type SwimChartMetric } from '../../../../util/swim-metrics'
import { applyI18n } from '../../runtime/dom'
import { analysisFinite } from '../analysis'
import { setActivityExpanded } from '../comparison'
import { runHeartCostWindow, setRunHeartCostWindow } from '../run-heart-cost'
import {
  cyclingChartMode,
  runMetricMode,
  setCyclingChartMode,
  setRunMetricMode,
  type CyclingChartMode,
  type RunMetricMode,
} from '../scrub'
import { buildDayCard } from './card'

export type DayEmbedRenderState = {
  cardCleanup: (() => void) | null
  pendingSwimMode: { index: number; mode: SwimTrendMode } | null
  pendingCyclingChartMode: { activityId: string; trace: string; mode: CyclingChartMode } | null
  pendingRunMetricMode: { activityId: string; group: string; mode: RunMetricMode } | null
  pendingRunCostWindow: { activityId: string; window: RunHeartCostWindow } | null
  pendingAnalysisRange: {
    activityId: string
    kind: string
    id: string
    selected: boolean
    restoreFocus: boolean
  } | null
}

export const renderDayEmbed = (
  context: TriathlonContext,
  embed: HTMLElement,
  date: string,
  data: DetailPayload,
  extras: DayCardExtras,
  renderState: DayEmbedRenderState,
): void => {
  const environmentStates = Array.from(
    embed.querySelectorAll<HTMLElement>('[data-environment-tabs]'),
  ).flatMap(analysis => {
    const activityId = analysis.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
      .activityId
    return activityId
      ? [
          {
            activityId,
            view: analysis.dataset.environmentView,
            mode: analysis.dataset.environmentCumulativeMode,
          },
        ]
      : []
  })
  const swimStates: {
    mode: SwimTrendMode
    focusedMode: SwimTrendMode | null
    charts: { kind: SwimChartMetric; distanceM: number; active: boolean; focused: boolean }[]
  }[] = Array.from(embed.querySelectorAll<HTMLElement>('.tri-swim-trends'), section => {
    const toggle = section.querySelector<HTMLElement>('.tri-swim-mode-toggle')
    const active = document.activeElement
    const charts: {
      kind: SwimChartMetric
      distanceM: number
      active: boolean
      focused: boolean
    }[] = []
    for (const chart of section.querySelectorAll<SVGSVGElement>('.tri-swim-trend-svg')) {
      const distanceM = Number(chart.getAttribute('aria-valuenow'))
      if (!Number.isFinite(distanceM)) continue
      charts.push({
        kind: swimChartMetric(chart.dataset.swimKind),
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
  if (renderState.pendingSwimMode && swimStates[renderState.pendingSwimMode.index])
    swimStates[renderState.pendingSwimMode.index].mode = renderState.pendingSwimMode.mode
  renderState.pendingSwimMode = null
  const cyclingChartStates = Array.from(
    embed.querySelectorAll<HTMLElement>('.tri-cycling-mode-chart[data-tri-trace]'),
  ).flatMap(chart => {
    const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId
    const trace = chart.dataset.triTrace
    return activityId && trace ? [{ activityId, trace, mode: cyclingChartMode(chart) }] : []
  })
  if (renderState.pendingCyclingChartMode) {
    const match = cyclingChartStates.find(
      candidate =>
        candidate.activityId === renderState.pendingCyclingChartMode?.activityId &&
        candidate.trace === renderState.pendingCyclingChartMode.trace,
    )
    if (match) match.mode = renderState.pendingCyclingChartMode.mode
  }
  renderState.pendingCyclingChartMode = null
  const runMetricStates = Array.from(
    embed.querySelectorAll<HTMLElement>('.tri-run-metric-chart[data-run-metric-group]'),
  ).flatMap(chart => {
    const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId
    const group = chart.dataset.runMetricGroup
    const mode = runMetricMode(chart)
    return activityId && group && mode ? [{ activityId, group, mode }] : []
  })
  if (renderState.pendingRunMetricMode) {
    const match = runMetricStates.find(
      candidate =>
        candidate.activityId === renderState.pendingRunMetricMode?.activityId &&
        candidate.group === renderState.pendingRunMetricMode.group,
    )
    if (match) match.mode = renderState.pendingRunMetricMode.mode
  }
  renderState.pendingRunMetricMode = null
  const runCostStates = Array.from(
    embed.querySelectorAll<HTMLElement>('.tri-run-heart-cost'),
  ).flatMap(chart => {
    const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId
    return activityId ? [{ activityId, window: runHeartCostWindow(chart) }] : []
  })
  if (renderState.pendingRunCostWindow) {
    const match = runCostStates.find(
      candidate => candidate.activityId === renderState.pendingRunCostWindow?.activityId,
    )
    if (match) match.window = renderState.pendingRunCostWindow.window
  }
  renderState.pendingRunCostWindow = null
  const analysisStates = Array.from(
    embed.querySelectorAll<HTMLElement>('.tri-act[data-activity-id]'),
    activity => {
      const workspace = activity.querySelector<HTMLElement>('[data-tri-analysis]') ?? activity
      const focused =
        document.activeElement instanceof HTMLButtonElement &&
        workspace.contains(document.activeElement)
          ? document.activeElement.closest<HTMLButtonElement>('[data-analysis-range]')
          : null
      return {
        activityId: activity.dataset.activityId,
        kind: workspace.dataset.selectedKind,
        id: workspace.dataset.selectedId,
        selected: Boolean(workspace.dataset.selectedKind && workspace.dataset.selectedId),
        startDistanceKm: analysisFinite(workspace.dataset.selectionStartDistanceKm),
        endDistanceKm: analysisFinite(workspace.dataset.selectionEndDistanceKm),
        focusedKind: focused?.dataset.rangeKind,
        focusedId: focused?.dataset.rangeId,
      }
    },
  )
  if (renderState.pendingAnalysisRange) {
    const match = analysisStates.find(
      candidate => candidate.activityId === renderState.pendingAnalysisRange?.activityId,
    )
    if (match) {
      match.selected = renderState.pendingAnalysisRange.selected
      match.kind = renderState.pendingAnalysisRange.selected
        ? renderState.pendingAnalysisRange.kind
        : undefined
      match.id = renderState.pendingAnalysisRange.selected
        ? renderState.pendingAnalysisRange.id
        : undefined
      match.focusedKind = renderState.pendingAnalysisRange.restoreFocus
        ? renderState.pendingAnalysisRange.kind
        : undefined
      match.focusedId = renderState.pendingAnalysisRange.restoreFocus
        ? renderState.pendingAnalysisRange.id
        : undefined
    }
  }
  renderState.pendingAnalysisRange = null
  const fresh = buildDayCard(context.presentation, date, data, extras)
  const expanded = Array.from(embed.querySelectorAll('.tri-act'), activity =>
    activity.classList.contains('tri-act--expanded'),
  )
  fresh.element.querySelectorAll('.tri-act').forEach((activity, index) => {
    if (index < expanded.length) setActivityExpanded(activity as HTMLElement, expanded[index])
  })
  renderState.cardCleanup?.()
  embed.replaceChildren(fresh.element)
  renderState.cardCleanup = fresh.mount()
  applyI18n(fresh.element, context.presentation)
  for (const analysis of fresh.element.querySelectorAll<HTMLElement>('[data-environment-tabs]')) {
    const activityId = analysis.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
      .activityId
    const state = environmentStates.find(candidate => candidate.activityId === activityId)
    if (!state) continue
    Array.from(analysis.querySelectorAll<HTMLButtonElement>('[data-environment-tab]'))
      .find(button => button.dataset.environmentTab === state.view)
      ?.click()
    Array.from(analysis.querySelectorAll<HTMLButtonElement>('[data-environment-mode]'))
      .find(button => button.dataset.environmentMode === state.mode)
      ?.click()
  }
  for (const state of analysisStates) {
    const activity = Array.from(
      fresh.element.querySelectorAll<HTMLElement>('.tri-act[data-activity-id]'),
    ).find(candidate => candidate.dataset.activityId === state.activityId)
    if (!activity) continue
    const workspace = activity.querySelector<HTMLElement>('[data-tri-analysis]') ?? activity
    const buttons = Array.from(
      workspace.querySelectorAll<HTMLButtonElement>('[data-analysis-range]'),
    )
    workspace.dispatchEvent(
      new CustomEvent('tri:analysis-restore', {
        detail: {
          selected: state.selected,
          kind: state.kind,
          id: state.id,
          startDistanceKm: state.startDistanceKm,
          endDistanceKm: state.endDistanceKm,
        },
      }),
    )
    buttons
      .find(
        button =>
          button.dataset.rangeKind === state.focusedKind &&
          button.dataset.rangeId === state.focusedId,
      )
      ?.focus({ preventScroll: true })
  }
  fresh.element
    .querySelectorAll<HTMLElement>('.tri-cycling-mode-chart[data-tri-trace]')
    .forEach(chart => {
      const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
        .activityId
      const state = cyclingChartStates.find(
        candidate =>
          candidate.activityId === activityId && candidate.trace === chart.dataset.triTrace,
      )
      if (!state) return
      setCyclingChartMode(chart, state.mode)
    })
  fresh.element
    .querySelectorAll<HTMLElement>('.tri-run-metric-chart[data-run-metric-group]')
    .forEach(chart => {
      const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
        .activityId
      const state = runMetricStates.find(
        candidate =>
          candidate.activityId === activityId && candidate.group === chart.dataset.runMetricGroup,
      )
      if (state) setRunMetricMode(chart, state.mode)
    })
  fresh.element.querySelectorAll<HTMLElement>('.tri-run-heart-cost').forEach(chart => {
    const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId
    const state = runCostStates.find(candidate => candidate.activityId === activityId)
    if (state) setRunHeartCostWindow(chart, state.window, context.presentation)
  })
  fresh.element.querySelectorAll<HTMLElement>('.tri-swim-trends').forEach((section, index) => {
    const state = swimStates[index]
    if (!state) return
    const selected = section.querySelector<HTMLButtonElement>(
      `.tri-swim-mode[data-swim-mode="${state.mode}"]`,
    )
    if (state.mode === '100m') selected?.click()
    if (state.focusedMode)
      section
        .querySelector<HTMLButtonElement>(`.tri-swim-mode[data-swim-mode="${state.focusedMode}"]`)
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
