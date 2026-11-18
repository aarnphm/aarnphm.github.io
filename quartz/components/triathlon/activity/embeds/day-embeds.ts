import type { DayCardExtras } from '../../../../util/triathlon-card'
import type { TriathlonContext } from '../../runtime/context'
import type { DetailPayload } from '../data'
import { dayCardActivitiesExpanded } from '../../../../util/triathlon-card'
import { rootNavSignal } from '../../../scripts/root-lifecycle'
import { TRI_POWER_FILTER_EVENT } from '../../runtime/preferences'
import { setupPowerCurveTicks } from '../../shell/power-curve-ticks'
import { onCardToggle } from '../comparison'
import { mountDaySleepCharts } from '../day-sleep'
import { mountDayTrack } from '../day-track'
import { dayExtrasFromDataset } from '../embed-settings'
import { setRunHeartCostWindow } from '../run-heart-cost'
import { runMetricMode } from '../scrub'
import { buildDayCard } from './card'
import { mountActivitySectionAlignment } from './card'
import { renderDayEmbed, type DayEmbedRenderState } from './day-render'
import { setupActivityTitleTooltip } from './title-tooltip'

export const setupDayEmbeds = (context: TriathlonContext): (() => void) | null => {
  let live = true
  const teardowns: (() => void)[] = []
  const registered = new Map<HTMLElement, () => void>()
  let activityTitleTooltipReady = false
  const upgradeByEmbed = new Map<HTMLElement, () => void>()
  const upgradeObserver = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting || !(entry.target instanceof HTMLElement)) continue
        const upgrade = upgradeByEmbed.get(entry.target)
        if (!upgrade) continue
        upgradeObserver.unobserve(entry.target)
        upgradeByEmbed.delete(entry.target)
        upgrade()
      }
    },
    { rootMargin: '600px 0px' },
  )
  teardowns.push(() => {
    upgradeObserver.disconnect()
    upgradeByEmbed.clear()
  })
  const setupEmbed = (embed: HTMLElement): void => {
    if (registered.has(embed)) return
    const lifecycleSignal = rootNavSignal(embed)
    const embedTeardowns: (() => void)[] = []
    let cleaned = false
    const cleanupEmbed = (): void => {
      if (cleaned) return
      cleaned = true
      lifecycleSignal.removeEventListener('abort', cleanupEmbed)
      upgradeObserver.unobserve(embed)
      upgradeByEmbed.delete(embed)
      registered.delete(embed)
      for (let index = embedTeardowns.length - 1; index >= 0; index -= 1) {
        embedTeardowns[index]()
      }
    }
    registered.set(embed, cleanupEmbed)
    lifecycleSignal.addEventListener('abort', cleanupEmbed, { once: true })
    if (!activityTitleTooltipReady) {
      activityTitleTooltipReady = true
      const cleanup = setupActivityTitleTooltip()
      if (cleanup) teardowns.push(cleanup)
    }
    const date = embed.dataset.triathlonDate!
    const sourceHref = embed
      .closest('.transclude')
      ?.querySelector<HTMLAnchorElement>('.transclude-src')?.href
    const extras: DayCardExtras = {
      ...dayExtrasFromDataset(embed.dataset),
      ...(sourceHref ? { dayRouteHref: sourceHref } : {}),
    }
    const detailPath = embed.dataset.detailPath ?? '/static/strava-detail.json'
    let upgraded = false
    let payload: DetailPayload | null = null
    const state: DayEmbedRenderState = {
      pendingSwimMode: null,
      pendingCyclingChartMode: null,
      pendingRunMetricMode: null,
      pendingRunCostWindow: null,
      pendingAnalysisRange: null,
      cardCleanup: null,
    }
    let analysisPointerActive = false
    let analysisReleaseTimer = 0
    let deferredPayload: DetailPayload | null = null
    embedTeardowns.push(() => {
      state.cardCleanup?.()
      state.cardCleanup = null
    })
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
      state.pendingSwimMode = {
        index,
        mode: button.dataset.swimMode === '100m' ? '100m' : 'lengths',
      }
    }
    const setPendingCyclingChartMode = (target: EventTarget | null): void => {
      if (!(target instanceof Element)) return
      const button = target.closest<HTMLButtonElement>('.tri-cycling-chart-mode')
      const chart = button?.closest<HTMLElement>('.tri-cycling-mode-chart')
      if (!button || !chart) return
      const activityId = chart.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
        .activityId
      const trace = chart.dataset.triTrace
      if (!activityId || !trace) return
      state.pendingCyclingChartMode = {
        activityId,
        trace,
        mode: button.dataset.cyclingChartMode === 'power' ? 'power' : 'distance',
      }
    }
    const setPendingRunMetricMode = (target: EventTarget | null): void => {
      if (!(target instanceof Element)) return
      const button = target.closest<HTMLButtonElement>('.tri-run-metric-mode')
      const chart = button?.closest<HTMLElement>('.tri-run-metric-chart')
      const activityId = chart?.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset
        .activityId
      const group = chart?.dataset.runMetricGroup
      const mode = runMetricMode(button ?? null)
      if (!activityId || !group || !mode) return
      state.pendingRunMetricMode = { activityId, group, mode }
    }
    const setPendingRunCostWindow = (target: EventTarget | null): void => {
      if (!(target instanceof Element)) return
      const button = target.closest<HTMLButtonElement>('.tri-run-cost-window')
      const activityId = button
        ?.closest('.tri-run-heart-cost')
        ?.closest<HTMLElement>('.tri-act[data-activity-id]')?.dataset.activityId
      if (!button || !activityId) return
      state.pendingRunCostWindow = {
        activityId,
        window: button.dataset.runCostWindow === '30' ? '30' : '300',
      }
    }
    const onModePointerDown = (event: PointerEvent): void => {
      setPendingSwimMode(event.target)
      setPendingCyclingChartMode(event.target)
      setPendingRunMetricMode(event.target)
      setPendingRunCostWindow(event.target)
    }
    const onModeKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Enter' && event.key !== ' ') return
      setPendingSwimMode(event.target)
      setPendingCyclingChartMode(event.target)
      setPendingRunMetricMode(event.target)
      setPendingRunCostWindow(event.target)
    }
    const setPendingAnalysisRange = (target: EventTarget | null, restoreFocus = false): boolean => {
      if (payload) return false
      if (!(target instanceof Element)) return false
      const button = target.closest<HTMLButtonElement>('[data-analysis-range]')
      const workspace =
        button?.closest<HTMLElement>('[data-tri-analysis]') ??
        button?.closest<HTMLElement>('.tri-act[data-activity-id]')
      const activityId = workspace?.dataset.activityId
      const kind = button?.dataset.rangeKind
      const id = button?.dataset.rangeId
      if (!activityId || !kind || !id) return false
      state.pendingAnalysisRange = {
        activityId,
        kind,
        id,
        selected: button.getAttribute('aria-pressed') !== 'true',
        restoreFocus,
      }
      return true
    }
    const releaseAnalysisPointer = (): void => {
      window.clearTimeout(analysisReleaseTimer)
      analysisReleaseTimer = 0
      analysisPointerActive = false
      if (!deferredPayload) return
      const data = deferredPayload
      deferredPayload = null
      render(data)
    }
    const onAnalysisPointerDown = (event: PointerEvent): void => {
      if (!setPendingAnalysisRange(event.target)) return
      window.clearTimeout(analysisReleaseTimer)
      analysisReleaseTimer = 0
      analysisPointerActive = true
    }
    const onAnalysisPointerUp = (): void => {
      if (!analysisPointerActive) return
      analysisReleaseTimer = window.setTimeout(releaseAnalysisPointer, 0)
    }
    const onAnalysisClick = (event: MouseEvent): void => {
      if (!state.pendingAnalysisRange) setPendingAnalysisRange(event.target)
      releaseAnalysisPointer()
    }
    const onAnalysisPointerCancel = (): void => {
      state.pendingAnalysisRange = null
      releaseAnalysisPointer()
    }
    embedTeardowns.push(() => {
      window.clearTimeout(analysisReleaseTimer)
      analysisReleaseTimer = 0
      analysisPointerActive = false
      deferredPayload = null
    })
    const onAnalysisKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Enter' || event.key === ' ') setPendingAnalysisRange(event.target, true)
    }
    const clearPendingModes = (): void => {
      state.pendingSwimMode = null
      state.pendingCyclingChartMode = null
      state.pendingRunMetricMode = null
      state.pendingRunCostWindow = null
    }
    embed.addEventListener('pointerdown', onModePointerDown, { passive: true })
    embed.addEventListener('keydown', onModeKeyDown)
    embed.addEventListener('pointerdown', onAnalysisPointerDown, { passive: true })
    embed.addEventListener('pointerup', onAnalysisPointerUp, { passive: true })
    embed.addEventListener('click', onAnalysisClick)
    embed.addEventListener('pointercancel', onAnalysisPointerCancel)
    embed.addEventListener('keydown', onAnalysisKeyDown)
    embed.addEventListener('click', clearPendingModes)
    embed.addEventListener('pointercancel', clearPendingModes)
    embedTeardowns.push(() => {
      embed.removeEventListener('pointerdown', onModePointerDown)
      embed.removeEventListener('keydown', onModeKeyDown)
      embed.removeEventListener('pointerdown', onAnalysisPointerDown)
      embed.removeEventListener('pointerup', onAnalysisPointerUp)
      embed.removeEventListener('click', onAnalysisClick)
      embed.removeEventListener('pointercancel', onAnalysisPointerCancel)
      embed.removeEventListener('keydown', onAnalysisKeyDown)
      embed.removeEventListener('click', clearPendingModes)
      embed.removeEventListener('pointercancel', clearPendingModes)
    })
    const render = (data: DetailPayload): void =>
      renderDayEmbed(context, embed, date, data, extras, state)
    const upgrade = () => {
      if (upgraded) return
      upgraded = true
      upgradeObserver.unobserve(embed)
      upgradeByEmbed.delete(embed)
      void context.resources.detail.load(detailPath).then(result => {
        if (!live || lifecycleSignal.aborted || !embed.isConnected || result.status !== 'ready')
          return
        const data = result.value
        payload = data
        if (analysisPointerActive) {
          deferredPayload = data
          return
        }
        render(data)
      })
    }
    const onUnit = () => (payload ? render(payload) : upgrade())
    window.addEventListener('tri:unit', onUnit)
    window.addEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    embedTeardowns.push(() => {
      window.removeEventListener('tri:unit', onUnit)
      window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    })
    const ssr = embed.querySelector<HTMLElement>(':scope > .tri-pop-card')
    if (ssr) {
      ssr.addEventListener('click', onCardToggle)
      // A cold load can take the click before the detail payload arrives; the SSR chart then holds
      // the chosen window and the upgrade reads it back.
      const onRunCostWindow = (event: MouseEvent): void => {
        if (!(event.target instanceof Element)) return
        const button = event.target.closest<HTMLButtonElement>('.tri-run-cost-window')
        const chart = button?.closest<HTMLElement>('.tri-run-heart-cost')
        if (!button || !chart) return
        setRunHeartCostWindow(
          chart,
          button.dataset.runCostWindow === '30' ? '30' : '300',
          context.presentation,
        )
      }
      ssr.addEventListener('click', onRunCostWindow)
      const cleanupAlignment = mountActivitySectionAlignment(ssr)
      const cleanupTrack = mountDayTrack(ssr, context.presentation)
      const cleanupPowerTicks = setupPowerCurveTicks(ssr)
      const cleanupDaySleepCharts = mountDaySleepCharts(ssr, () => context.presentation.locale)
      state.cardCleanup = () => {
        ssr.removeEventListener('click', onCardToggle)
        ssr.removeEventListener('click', onRunCostWindow)
        cleanupAlignment()
        cleanupTrack()
        cleanupPowerTicks()
        cleanupDaySleepCharts()
      }
      const events = ['pointerdown', 'touchstart'] as const
      for (const ev of events) embed.addEventListener(ev, upgrade, { once: true, passive: true })
      const onKeyboardFocus = (event: FocusEvent): void => {
        if (!(event.target instanceof Element) || !event.target.matches(':focus-visible')) return
        embed.removeEventListener('focusin', onKeyboardFocus)
        upgrade()
      }
      const onChartMove = (event: PointerEvent): void => {
        if (!(event.target instanceof Element) || !event.target.closest('.tri-elev')) return
        embed.removeEventListener('pointermove', onChartMove)
        upgrade()
      }
      embed.addEventListener('focusin', onKeyboardFocus)
      embed.addEventListener('pointermove', onChartMove, { passive: true })
      upgradeByEmbed.set(embed, upgrade)
      upgradeObserver.observe(embed)
      embedTeardowns.push(() => {
        for (const ev of events) embed.removeEventListener(ev, upgrade)
        embed.removeEventListener('focusin', onKeyboardFocus)
        embed.removeEventListener('pointermove', onChartMove)
      })
      if (dayCardActivitiesExpanded(extras)) upgrade()
    } else {
      const initial = buildDayCard(context.presentation, date, null, extras)
      embed.replaceChildren(initial.element)
      state.cardCleanup = initial.mount()
      upgrade()
    }
  }
  const setupWithin = (root: ParentNode): void => {
    if (root instanceof HTMLElement && root.matches('.tri-day-embed[data-triathlon-date]'))
      setupEmbed(root)
    for (const embed of root.querySelectorAll<HTMLElement>('.tri-day-embed[data-triathlon-date]'))
      setupEmbed(embed)
  }
  const onContentMounted = (event: CustomEventMap['contentdecrypted']): void => {
    setupWithin(event.detail.content)
  }
  setupWithin(document)
  document.addEventListener('contentdecrypted', onContentMounted)
  teardowns.push(() => document.removeEventListener('contentdecrypted', onContentMounted))
  return () => {
    live = false
    for (const cleanup of Array.from(registered.values())) cleanup()
    for (const td of teardowns) td()
  }
}
