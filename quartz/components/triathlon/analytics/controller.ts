import type { TriathlonContext } from '../runtime/context'
import { start } from '../../../functional'
import { decodeActivityComparisonAnchor } from '../../../util/triathlon-comparison'
import { isRecord } from '../../../util/type-guards'
import { onCardToggle } from '../activity/comparison'
import { el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'
import {
  ANALYTICS_CHAPTERS,
  analyticsChapterFromHash,
  DEFAULT_ANALYTICS_CHAPTER,
  type AnalyticsChapterKey,
} from './catalog'
import { initSession, mountShellChrome, type AnalyticsSession } from './controller/shell'
import { mountActivityViews, mountChapterViews } from './controller/views'
import { initialAnalyticsModel, updateAnalytics } from './model'
import { defaultAnalyticsRange } from './range'

const isChapter = (value: string | undefined): value is AnalyticsChapterKey =>
  ANALYTICS_CHAPTERS.some(chapter => chapter.key === value)

export const setupAnalytics = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-analytics-btn')
  const panel = root.querySelector<HTMLElement>('.tri-analytics')
  const scrim = root.querySelector<HTMLElement>('.tri-analytics-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-ana-close')
  const results = root.querySelector<HTMLElement>('.tri-ana-results')
  const compareToggle = root.querySelector<HTMLButtonElement>('.tri-ana-compare-toggle')
  const compareSubmit = root.querySelector<HTMLButtonElement>('.tri-compare-picker-submit')
  const pageMode = root.dataset.triView === 'analytics'
  if (!panel || (!btn && !pageMode)) return null
  return mountAnalytics(
    root,
    context,
    btn,
    panel,
    scrim,
    closeBtn,
    results,
    compareToggle,
    compareSubmit,
    pageMode,
  )
}

function mountControllerEvents(s: AnalyticsSession): () => void {
  s.onResultsClick = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return
    if (event.target.closest('[data-compare-retry]')) {
      s.enterCompare()
      return
    }
    s.activate(event.target.closest<HTMLElement>('.tri-ana-ritem') ?? undefined)
  }

  s.onPickerKey = (event: KeyboardEvent) => {
    if (!s.inCompareMode()) return
    if (
      event.key === 'Enter' &&
      event.shiftKey &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !event.isComposing &&
      s.inCompareMode()
    ) {
      event.preventDefault()
      s.program.dispatch({ type: 'submit-comparison' })
    } else if (
      event.key === 'ArrowDown' ||
      (event.ctrlKey && (event.key === 'n' || event.key === 'N'))
    ) {
      event.preventDefault()
      s.setSel(s.program.retrieve().selectedResult + 1)
      s.resultItems()[s.program.retrieve().selectedResult]?.focus({ preventScroll: true })
    } else if (
      event.key === 'ArrowUp' ||
      (event.ctrlKey && (event.key === 'p' || event.key === 'P'))
    ) {
      event.preventDefault()
      s.setSel(s.program.retrieve().selectedResult - 1)
      s.resultItems()[s.program.retrieve().selectedResult]?.focus({ preventScroll: true })
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const its = s.resultItems()
      s.activate(
        its.find(item => item === document.activeElement) ??
          its[s.program.retrieve().selectedResult] ??
          its[0],
      )
    }
  }

  s.enterCompare = () => {
    s.closeDetail()
    s.setCompareMode(true)
    const generation = s.detailGeneration
    s.panel.classList.add('tri-analytics--searching')
    if (s.results) {
      s.results.setAttribute('aria-hidden', 'false')
      s.results.replaceChildren(el('div', 'tri-ana-empty', s.context.formatter.text('loading')))
    }
    void s.loadDetails().then(available => {
      if (
        !s.live ||
        !s.inCompareMode() ||
        !s.panel.isConnected ||
        s.panel.getAttribute('aria-hidden') === 'true' ||
        generation !== s.detailGeneration
      )
        return
      if (!available) {
        if (s.results) {
          const error = el('div', 'tri-compare-load-error')
          error.append(
            el('span', 'tri-ana-empty', s.context.formatter.text('activity data unavailable'), {
              'data-i18n': 'activity data unavailable',
            }),
            el('button', 'tri-compare-load-retry', s.context.formatter.text('retry'), {
              type: 'button',
              'data-compare-retry': '',
              'data-i18n': 'retry',
            }),
          )
          s.results.replaceChildren(error)
        }
        return
      }
      s.refreshPicker()
      s.resultItems()[0]?.focus({ preventScroll: true })
    })
  }

  s.onCompareSubmit = () => {
    s.program.dispatch({ type: 'submit-comparison' })
  }

  s.inComparison = (): boolean => s.panel.classList.contains('tri-analytics--comparison')

  s.onCompareToggle = () => {
    if (s.pageMode && s.inComparison()) s.closeDetail()
    else if (s.inCompareMode()) s.toMain()
    else s.enterCompare()
  }

  s.onClose = () => {
    if (s.inComparison()) s.closeDetail()
    else s.close()
  }

  s.open = () => {
    s.toMain()
    s.root.classList.add('tri-analytics-open')
    s.panel.setAttribute('aria-hidden', 'false')
    s.load()
    s.panel.focus({ preventScroll: true })
  }

  s.selectChapter = (next: AnalyticsChapterKey, updateHash: boolean, focus: boolean) => {
    s.chapter = next
    for (const tab of s.chapterTabs) {
      const active = tab.dataset.analyticsChapterTab === next
      tab.setAttribute('aria-selected', String(active))
      tab.tabIndex = active ? 0 : -1
      if (active && focus) tab.focus({ preventScroll: true })
    }
    for (const section of s.chapterPanels)
      section.hidden = section.dataset.analyticsChapter !== next
    s.body?.scrollTo({ top: 0 })
    s.renderChapter()
    if (!updateHash || !s.pageMode) return
    const url = new URL(window.location.href)
    url.hash = next === DEFAULT_ANALYTICS_CHAPTER ? '' : next
    window.history.replaceState(window.history.state, '', url)
  }

  s.onChapterClick = (event: MouseEvent) => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-analytics-chapter-tab]')
        : null
    const next = tab?.dataset.analyticsChapterTab
    if (!isChapter(next)) return
    if (
      s.inCompareMode() ||
      s.panel.classList.contains('tri-analytics--detail') ||
      s.panel.classList.contains('tri-analytics--comparison')
    )
      s.toMain()
    s.selectChapter(next, true, false)
  }

  s.onChapterKey = (event: KeyboardEvent) => {
    if (!(event.target instanceof HTMLButtonElement)) return
    const current = s.chapterTabs.indexOf(event.target)
    if (current < 0) return
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? s.chapterTabs.length - 1
          : event.key === 'ArrowLeft'
            ? (current - 1 + s.chapterTabs.length) % s.chapterTabs.length
            : event.key === 'ArrowRight'
              ? (current + 1) % s.chapterTabs.length
              : -1
    const value = s.chapterTabs[next]?.dataset.analyticsChapterTab
    if (!isChapter(value)) return
    event.preventDefault()
    s.selectChapter(value, true, true)
  }

  s.onHashChange = () => {
    const next = analyticsChapterFromHash(window.location.hash)
    if (next !== s.chapter) s.selectChapter(next, false, false)
  }

  s.onMethodToggle = (event: MouseEvent) => {
    const toggle =
      event.target instanceof Element ? event.target.closest('.tri-ana-method-toggle') : null
    const block = toggle?.closest<HTMLElement>('.tri-ana-block')
    if (!toggle || !block) return
    toggle.setAttribute('aria-expanded', String(block.classList.toggle('tri-ana-block--method')))
  }

  s.onComparisonFill = (event: Event) => {
    if (!(event instanceof CustomEvent) || !isRecord(event.detail)) return
    const anchor = event.detail.anchor
    const activityIds = typeof anchor === 'string' ? decodeActivityComparisonAnchor(anchor) : null
    if (!activityIds) return
    if (s.pageMode) {
      s.toMain()
      s.panel.setAttribute('aria-hidden', 'false')
      s.load()
    } else {
      s.open()
    }
    s.enterCompare()
    const generation = s.detailGeneration
    void s.loadDetails().then(available => {
      if (
        !available ||
        !s.live ||
        !s.inCompareMode() ||
        !s.panel.isConnected ||
        generation !== s.detailGeneration
      )
        return
      const activities = activityIds.flatMap(activityId => {
        const activity = s.detailData?.details[activityId]
        return activity ? [activity] : []
      })
      const sport = activities[0]?.sport
      if (
        activities.length !== activityIds.length ||
        !sport ||
        activities.some(
          activity => activity.sport !== sport || !s.compareActivityEligible(activity),
        )
      )
        return
      s.program.dispatch({ type: 'set-comparison-activities', ids: activityIds })
      s.program.dispatch({ type: 'submit-comparison' })
    })
  }

  s.onKey = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.key !== 'Escape') return
    if (s.panel.classList.contains('tri-analytics--detail')) {
      s.closeDetail()
      return
    }
    if (s.inCompareMode()) {
      s.toMain()
      return
    }
    s.close()
  }

  s.program = start({
    init: () => ({
      model: initialAnalyticsModel(defaultAnalyticsRange(s.root.dataset.triView)),
      effects: [],
    }),
    reduce: updateAnalytics,
    effects: (effect, state) => {
      if (effect.type === 'load-artifact') {
        const path = s.root.dataset.analyticsPath
        if (!path) {
          s.finishPageBoot()
          state.dispatch({ type: 'failed', request: effect.request })
          return
        }
        void s.context.resources.analytics.load(path).then(result => {
          if (!s.live) return
          if (result.status === 'ready') {
            s.sourceData = result.value
            state.dispatch({ type: 'loaded', request: effect.request })
          } else if (result.status === 'error') {
            s.finishPageBoot()
            state.dispatch({ type: 'failed', request: effect.request })
          }
        })
      } else if (effect.type === 'render-panels') {
        if (s.sourceData) s.render(s.sourceData)
      } else if (effect.type === 'render-search') {
        s.renderSearch()
      } else if (effect.type === 'render-activity') {
        s.showActivity(effect.id)
      } else if (effect.type === 'render-comparison') {
        const scrollTop = s.comparisonScrollTop
        s.comparisonScrollTop = 0
        s.showComparison(scrollTop)
      } else if (!s.pageMode) {
        s.btn?.focus({ preventScroll: true })
      }
    },
  })

  s.root.dataset.triAnalyticsRange = s.program.retrieve().range

  s.rangeCleanup = s.context.events.subscribe('analyticsRange', ({ range }) => {
    s.root.dataset.triAnalyticsRange = range
    s.program.dispatch({ type: 'set-range', range })
  })

  s.powerActivityCleanup = s.context.events.subscribe('powerActivity', request => {
    if (!s.panel.contains(request.source)) return
    request.handled = true
    if (s.inCompareMode()) s.setCompareMode(false)
    s.program.dispatch({ type: 'show-activity', id: request.activityId })
  })

  if (s.chapter !== DEFAULT_ANALYTICS_CHAPTER) s.selectChapter(s.chapter, false, false)

  if (s.pageMode) {
    window.addEventListener('hashchange', s.onHashChange)
    s.load()
  } else {
    s.btn?.addEventListener('click', s.open)
    s.closeBtn?.addEventListener('click', s.onClose)
    s.scrim?.addEventListener('click', s.close)
  }

  s.compareToggle?.addEventListener('click', s.onCompareToggle)

  s.compareSubmit?.addEventListener('click', s.onCompareSubmit)

  for (const tab of s.chapterTabs) {
    tab.addEventListener('click', s.onChapterClick)
    tab.addEventListener('keydown', s.onChapterKey)
  }

  s.body?.addEventListener('click', s.onMethodToggle)

  s.results?.addEventListener('click', s.onResultsClick)

  s.results?.addEventListener('keydown', s.onPickerKey)

  s.detail?.addEventListener('click', onCardToggle)

  document.addEventListener('keydown', s.onKey)

  window.addEventListener('tri:comparison-fill', s.onComparisonFill)

  s.onUnitChange = () => {
    const comparisonVisible =
      s.panel.classList.contains('tri-analytics--detail') &&
      s.detail?.querySelector('.tri-compare') != null
    const activityId = s.detail?.querySelector<HTMLElement>('.tri-act[data-activity-id]')?.dataset
      .activityId
    if (s.sourceData) {
      s.render(s.sourceData)
    }
    if (comparisonVisible) s.showComparison()
    else if (activityId) s.showActivity(activityId)
    else if (s.inCompareMode()) s.refreshPicker()
  }

  window.addEventListener('tri:unit', s.onUnitChange)

  window.addEventListener('tri:locale', s.onUnitChange)

  window.addEventListener(TRI_POWER_FILTER_EVENT, s.onUnitChange)

  return () => {
    s.live = false
    s.finishPageBoot()
    s.btn?.removeEventListener('click', s.open)
    s.closeBtn?.removeEventListener('click', s.onClose)
    s.scrim?.removeEventListener('click', s.close)
    s.compareToggle?.removeEventListener('click', s.onCompareToggle)
    s.compareSubmit?.removeEventListener('click', s.onCompareSubmit)
    for (const tab of s.chapterTabs) {
      tab.removeEventListener('click', s.onChapterClick)
      tab.removeEventListener('keydown', s.onChapterKey)
    }
    s.body?.removeEventListener('click', s.onMethodToggle)
    window.removeEventListener('hashchange', s.onHashChange)
    s.results?.removeEventListener('click', s.onResultsClick)
    s.results?.removeEventListener('keydown', s.onPickerKey)
    s.detail?.removeEventListener('click', onCardToggle)
    document.removeEventListener('keydown', s.onKey)
    window.removeEventListener('tri:comparison-fill', s.onComparisonFill)
    window.removeEventListener('tri:unit', s.onUnitChange)
    window.removeEventListener('tri:locale', s.onUnitChange)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, s.onUnitChange)
    s.powerActivityCleanup()
    s.rangeCleanup()
    s.compareCleanup?.()
    s.activityCleanup?.()
    for (const cleanup of s.panelCleanups.values()) cleanup()
    s.panelCleanups.clear()
    s.program.stop()
    if (s.renderFrame !== 0) window.cancelAnimationFrame(s.renderFrame)
  }
}

function mountAnalytics(
  root: HTMLElement,
  context: TriathlonContext,
  btn: HTMLElement | null,
  panel: HTMLElement,
  scrim: HTMLElement | null,
  closeBtn: HTMLElement | null,
  results: HTMLElement | null,
  compareToggle: HTMLButtonElement | null,
  compareSubmit: HTMLButtonElement | null,
  pageMode: boolean,
): () => void {
  const s = {
    root,
    context,
    btn,
    panel,
    scrim,
    closeBtn,
    results,
    compareToggle,
    compareSubmit,
    pageMode,
  } as AnalyticsSession
  initSession(s)
  mountChapterViews(s)
  mountShellChrome(s)
  mountActivityViews(s)
  return mountControllerEvents(s)
}
