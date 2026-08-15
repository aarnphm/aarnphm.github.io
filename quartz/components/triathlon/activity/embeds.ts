import type { DayCardExtras } from '../../../util/triathlon-card'
import type { SwimTrendMode } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { TriathlonContext } from '../runtime/context'
import type { DetailPayload } from './data'
import { swimChartMetric, type SwimChartMetric } from '../../../util/swim-metrics'
import { buildActivityComparison } from '../../../util/triathlon-card'
import { buildDayCard as buildDayCardNode } from '../../../util/triathlon-card'
import { dayCardActivitiesExpanded } from '../../../util/triathlon-card'
import { parseExcludedActivityIds } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { decodeActivityComparisonAnchor } from '../../../util/triathlon-comparison'
import { parseTriathlonTraceSettings } from '../../../util/triathlon-trace-settings'
import { applyI18n } from '../runtime/dom'
import { createDomFactory } from '../runtime/dom'
import { el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'
import { analysisFinite } from './analysis'
import { onCardToggle } from './comparison'
import { setActivityExpanded } from './comparison'
import { wireActivityComparison } from './comparison'
import { detailContextFromPayload } from './data'
import { renderDetail } from './render'
import { setupStrengthExerciseOverflow } from './render'

export const buildDayCard = (
  presentation: TriathlonPresentation,
  dateIso: string,
  payload: DetailPayload | null,
  extras: DayCardExtras = {},
): { element: HTMLElement; mount: () => () => void } => {
  const domF = createDomFactory(presentation)
  const activityViews: ReturnType<typeof renderDetail>[] = []
  const card = buildDayCardNode(domF, dateIso, payload, extras, detail => {
    const view = renderDetail(
      presentation,
      detail,
      payload,
      extras.event != null,
      extras.embedded === true,
      extras.dayRouteHref,
      extras.settings,
    )
    activityViews.push(view)
    return view.element
  }) as HTMLElement
  if (dayCardActivitiesExpanded(extras)) {
    card
      .querySelectorAll<HTMLElement>('.tri-act')
      .forEach(activity => setActivityExpanded(activity, true))
  }
  return {
    element: card,
    mount: () => {
      card.addEventListener('click', onCardToggle)
      const cleanups = activityViews.map(view => view.mount())
      return () => {
        card.removeEventListener('click', onCardToggle)
        for (const cleanup of cleanups) cleanup()
      }
    },
  }
}

export const dayExtrasFromDataset = (data: DOMStringMap): DayCardExtras => {
  const excludedActivityIds = parseExcludedActivityIds(data.triathlonFilter)
  const settings = parseTriathlonTraceSettings(data.triathlonSettings)
  return {
    location: data.triathlonLoc,
    event: data.triathlonEvent,
    sport: data.triathlonSport as DayCardExtras['sport'],
    activityId: data.triathlonActivityId,
    ...(excludedActivityIds.length > 0 ? { excludedActivityIds } : {}),
    ...(settings ? { settings } : {}),
    expanded: data.triathlonExpanded === '1',
    embedded: data.triathlonEmbedded === '1',
    dateHref: data.triathlonDateHref,
  }
}

export const setupActivityTitleTooltip = (): (() => void) | null => {
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return null
  document.body.querySelector('.tri-activity-title-tip')?.remove()
  const tip = el('div', 'tri-gloss tri-activity-title-tip', undefined, {
    role: 'tooltip',
    'aria-hidden': 'true',
  })
  document.body.appendChild(tip)
  let current: HTMLElement | null = null
  const hide = (): void => {
    current = null
    tip.classList.remove('tri-gloss--on')
    tip.setAttribute('aria-hidden', 'true')
  }
  const move = (event: PointerEvent): void => {
    const icon =
      event.target instanceof Element
        ? event.target.closest<SVGElement>('.tri-act-head > .tri-ico')
        : null
    const activity = icon?.closest<HTMLElement>('.tri-act[data-activity-title]') ?? null
    const title = activity?.dataset.activityTitle
    if (!activity || !title) {
      hide()
      return
    }
    if (activity !== current) {
      current = activity
      tip.textContent = title
      tip.classList.add('tri-gloss--on')
      tip.setAttribute('aria-hidden', 'false')
    }
    const rect = tip.getBoundingClientRect()
    const offset = 12
    const edge = 8
    const preferredLeft = event.clientX + offset
    const preferredTop = event.clientY + offset
    const left =
      preferredLeft + rect.width <= window.innerWidth - edge
        ? preferredLeft
        : event.clientX - offset - rect.width
    const top =
      preferredTop + rect.height <= window.innerHeight - edge
        ? preferredTop
        : event.clientY - offset - rect.height
    tip.style.left = `${Math.min(Math.max(edge, left), window.innerWidth - edge - rect.width)}px`
    tip.style.top = `${Math.min(Math.max(edge, top), window.innerHeight - edge - rect.height)}px`
  }
  document.body.addEventListener('pointermove', move, { passive: true })
  document.body.addEventListener('pointerleave', hide)
  return () => {
    document.body.removeEventListener('pointermove', move)
    document.body.removeEventListener('pointerleave', hide)
    tip.remove()
  }
}

export const setupDayEmbeds = (context: TriathlonContext): (() => void) | null => {
  let live = true
  const teardowns: (() => void)[] = []
  const registered = new Set<HTMLElement>()
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
    registered.add(embed)
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
    let pendingSwimMode: { index: number; mode: SwimTrendMode } | null = null
    let pendingAnalysisRange: {
      activityId: string
      kind: string
      id: string
      selected: boolean
      restoreFocus: boolean
    } | null = null
    let analysisPointerActive = false
    let analysisReleaseTimer = 0
    let deferredPayload: DetailPayload | null = null
    let cardCleanup: (() => void) | null = null
    teardowns.push(() => {
      cardCleanup?.()
      cardCleanup = null
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
      pendingSwimMode = { index, mode: button.dataset.swimMode === '100m' ? '100m' : 'lengths' }
    }
    const onSwimPointerDown = (event: PointerEvent): void => setPendingSwimMode(event.target)
    const onSwimKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Enter' || event.key === ' ') setPendingSwimMode(event.target)
    }
    const setPendingAnalysisRange = (target: EventTarget | null, restoreFocus = false): boolean => {
      if (payload) return false
      if (!(target instanceof Element)) return false
      const button = target.closest<HTMLButtonElement>('[data-analysis-range]')
      const workspace = button?.closest<HTMLElement>('[data-tri-analysis]')
      const activityId = workspace?.dataset.activityId
      const kind = button?.dataset.rangeKind
      const id = button?.dataset.rangeId
      if (!activityId || !kind || !id) return false
      pendingAnalysisRange = {
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
      if (!pendingAnalysisRange) setPendingAnalysisRange(event.target)
      releaseAnalysisPointer()
    }
    const onAnalysisPointerCancel = (): void => {
      pendingAnalysisRange = null
      releaseAnalysisPointer()
    }
    teardowns.push(() => {
      window.clearTimeout(analysisReleaseTimer)
      analysisReleaseTimer = 0
      analysisPointerActive = false
      deferredPayload = null
    })
    const onAnalysisKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Enter' || event.key === ' ') setPendingAnalysisRange(event.target, true)
    }
    const clearPendingSwimMode = (): void => {
      pendingSwimMode = null
    }
    embed.addEventListener('pointerdown', onSwimPointerDown, { passive: true })
    embed.addEventListener('keydown', onSwimKeyDown)
    embed.addEventListener('pointerdown', onAnalysisPointerDown, { passive: true })
    embed.addEventListener('pointerup', onAnalysisPointerUp, { passive: true })
    embed.addEventListener('click', onAnalysisClick)
    embed.addEventListener('pointercancel', onAnalysisPointerCancel)
    embed.addEventListener('keydown', onAnalysisKeyDown)
    embed.addEventListener('click', clearPendingSwimMode)
    embed.addEventListener('pointercancel', clearPendingSwimMode)
    teardowns.push(() => {
      embed.removeEventListener('pointerdown', onSwimPointerDown)
      embed.removeEventListener('keydown', onSwimKeyDown)
      embed.removeEventListener('pointerdown', onAnalysisPointerDown)
      embed.removeEventListener('pointerup', onAnalysisPointerUp)
      embed.removeEventListener('click', onAnalysisClick)
      embed.removeEventListener('pointercancel', onAnalysisPointerCancel)
      embed.removeEventListener('keydown', onAnalysisKeyDown)
      embed.removeEventListener('click', clearPendingSwimMode)
      embed.removeEventListener('pointercancel', clearPendingSwimMode)
    })
    const render = (data: DetailPayload) => {
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
      if (pendingSwimMode && swimStates[pendingSwimMode.index])
        swimStates[pendingSwimMode.index].mode = pendingSwimMode.mode
      pendingSwimMode = null
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
      if (pendingAnalysisRange) {
        const state = analysisStates.find(
          candidate => candidate.activityId === pendingAnalysisRange?.activityId,
        )
        if (state) {
          state.selected = pendingAnalysisRange.selected
          state.kind = pendingAnalysisRange.selected ? pendingAnalysisRange.kind : undefined
          state.id = pendingAnalysisRange.selected ? pendingAnalysisRange.id : undefined
          state.focusedKind = pendingAnalysisRange.restoreFocus
            ? pendingAnalysisRange.kind
            : undefined
          state.focusedId = pendingAnalysisRange.restoreFocus ? pendingAnalysisRange.id : undefined
        }
      }
      pendingAnalysisRange = null
      const fresh = buildDayCard(context.presentation, date, data, extras)
      const expanded = Array.from(embed.querySelectorAll('.tri-act'), activity =>
        activity.classList.contains('tri-act--expanded'),
      )
      fresh.element.querySelectorAll('.tri-act').forEach((activity, index) => {
        if (index < expanded.length) setActivityExpanded(activity as HTMLElement, expanded[index])
      })
      cardCleanup?.()
      embed.replaceChildren(fresh.element)
      cardCleanup = fresh.mount()
      applyI18n(fresh.element, context.presentation)
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
      fresh.element.querySelectorAll<HTMLElement>('.tri-swim-trends').forEach((section, index) => {
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
      upgradeObserver.unobserve(embed)
      upgradeByEmbed.delete(embed)
      void context.resources.detail.load(detailPath).then(result => {
        if (!live || !embed.isConnected || result.status !== 'ready') return
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
    teardowns.push(() => {
      window.removeEventListener('tri:unit', onUnit)
      window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    })
    const ssr = embed.querySelector<HTMLElement>(':scope > .tri-pop-card')
    if (ssr) {
      ssr.addEventListener('click', onCardToggle)
      const cleanupStrengthOverflow = setupStrengthExerciseOverflow(ssr)
      cardCleanup = () => {
        ssr.removeEventListener('click', onCardToggle)
        cleanupStrengthOverflow()
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
      teardowns.push(() => {
        for (const ev of events) embed.removeEventListener(ev, upgrade)
        embed.removeEventListener('focusin', onKeyboardFocus)
        embed.removeEventListener('pointermove', onChartMove)
      })
      if (extras.expanded) upgrade()
    } else {
      const initial = buildDayCard(context.presentation, date, null, extras)
      embed.replaceChildren(initial.element)
      cardCleanup = initial.mount()
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
    for (const td of teardowns) td()
    registered.clear()
  }
}

export const setupActivityComparisonEmbeds = (context: TriathlonContext): (() => void) | null => {
  let live = true
  const teardowns: (() => void)[] = []
  const registered = new Set<HTMLElement>()
  const upgradeByEmbed = new Map<HTMLElement, () => void>()
  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting || !(entry.target instanceof HTMLElement)) continue
        upgradeByEmbed.get(entry.target)?.()
      }
    },
    { rootMargin: '600px 0px' },
  )
  teardowns.push(() => {
    observer.disconnect()
    upgradeByEmbed.clear()
  })

  const setupEmbed = (embed: HTMLElement): void => {
    if (registered.has(embed)) return
    registered.add(embed)
    const activityIds = decodeActivityComparisonAnchor(embed.dataset.compareAnchor ?? '')
    if (!activityIds) return
    const detailPath = embed.dataset.detailPath ?? '/static/strava-detail.json'
    let upgraded = false
    let payload: DetailPayload | null = null
    let interactionCleanup: (() => void) | null = null
    const render = (data: DetailPayload) => {
      interactionCleanup?.()
      const activities = activityIds.flatMap(activityId => {
        const activity = data.details[activityId]
        return activity ? [powerViewActivity(context.presentation, activity)] : []
      })
      const comparison = buildActivityComparison(
        createDomFactory(context.presentation),
        activities,
        detailContextFromPayload(data),
        { removable: false },
      )
      applyI18n(comparison, context.presentation)
      embed.replaceChildren(comparison)
      interactionCleanup =
        comparison.dataset.compareState === 'ready'
          ? wireActivityComparison(context.presentation, comparison, activities)
          : null
    }
    const upgrade = () => {
      if (upgraded) return
      upgraded = true
      observer.unobserve(embed)
      upgradeByEmbed.delete(embed)
      void context.resources.detail.load(detailPath).then(result => {
        if (!live || !embed.isConnected || result.status !== 'ready') return
        payload = result.value
        render(result.value)
      })
    }
    const onPresentationChange = () => (payload ? render(payload) : upgrade())
    window.addEventListener('tri:unit', onPresentationChange)
    window.addEventListener('tri:locale', onPresentationChange)
    window.addEventListener(TRI_POWER_FILTER_EVENT, onPresentationChange)
    const events = ['pointerdown', 'touchstart'] as const
    for (const event of events)
      embed.addEventListener(event, upgrade, { once: true, passive: true })
    upgradeByEmbed.set(embed, upgrade)
    observer.observe(embed)
    teardowns.push(() => {
      interactionCleanup?.()
      window.removeEventListener('tri:unit', onPresentationChange)
      window.removeEventListener('tri:locale', onPresentationChange)
      window.removeEventListener(TRI_POWER_FILTER_EVENT, onPresentationChange)
      for (const event of events) embed.removeEventListener(event, upgrade)
    })
  }

  const setupWithin = (root: ParentNode): void => {
    if (root instanceof HTMLElement && root.matches('.tri-compare-embed[data-compare-anchor]'))
      setupEmbed(root)
    for (const embed of root.querySelectorAll<HTMLElement>(
      '.tri-compare-embed[data-compare-anchor]',
    ))
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
    for (const teardown of teardowns) teardown()
    registered.clear()
  }
}
