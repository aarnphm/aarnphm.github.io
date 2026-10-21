import type { Analytics } from '../../../plugins/stores/analytics'
import type { ActivityKind } from '../../../plugins/stores/strava'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import type { AnalyticsChapterKey, AnalyticsPanelRender } from './catalog'
import { start } from '../../../functional'
import { activityCompareColor } from '../../../util/triathlon-card'
import { buildActivityAnalyzeButton } from '../../../util/triathlon-card'
import { activityComparisonEligible } from '../../../util/triathlon-card'
import { buildActivityComparison } from '../../../util/triathlon-card'
import { dist } from '../../../util/triathlon-card'
import { dur } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { activityComparisonEmbed } from '../../../util/triathlon-comparison'
import { decodeActivityComparisonAnchor } from '../../../util/triathlon-comparison'
import { isRecord } from '../../../util/type-guards'
import { onCardToggle } from '../activity/comparison'
import { setActivityExpanded } from '../activity/comparison'
import { wireActivityComparison } from '../activity/comparison'
import { detailContextFromPayload } from '../activity/data'
import { buildIcon, buildRecovery } from '../activity/primitives'
import { renderDetail } from '../activity/render'
import { applyI18n } from '../runtime/dom'
import { createDomFactory } from '../runtime/dom'
import { el } from '../runtime/dom'
import { svg } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'
import { wireEmbedCopy } from '../shell/timeline'
import { TRI_ANALYTICS_BOOT_CLASS } from './boot'
import { ANALYTICS_CHAPTERS } from './catalog'
import { analyticsChapterFromHash } from './catalog'
import { analyticsPanelDefinition } from './catalog'
import { DEFAULT_ANALYTICS_CHAPTER } from './catalog'
import { initialAnalyticsModel, updateAnalytics } from './model'
import { analyticsForRange, defaultAnalyticsRange } from './range'
import { activityResultItem } from './search'
import { activityResultItems } from './search'
import { detailHead } from './search'
import { setActivityResultSelection } from './search'

const RENDER_SLICE_MS = 8
// Methodology notes stay folded behind a per-panel toggle so the numbers lead.
const METHOD_NOTES = '.tri-accl-method, .tri-ftp-note, .tri-ftp-source'
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
  const title = root.querySelector<HTMLElement>('.tri-ana-title')
  const results = root.querySelector<HTMLElement>('.tri-ana-results')
  const compareToggle = root.querySelector<HTMLButtonElement>('.tri-ana-compare-toggle')
  const pageMode = root.dataset.triView === 'analytics'
  if (!panel || (!btn && !pageMode)) return null

  const body = root.querySelector<HTMLElement>('.tri-ana-body')
  const detail = root.querySelector<HTMLElement>('.tri-ana-detail')
  let live = true
  let sourceData: Analytics | null = null
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailPromise: Promise<boolean> | null = null
  const panelCleanups = new Map<HTMLElement, () => void>()
  let compareCleanup: (() => void) | null = null
  let activityCleanup: (() => void) | null = null
  let detailGeneration = 0
  let comparisonScrollTop = 0
  let renderFrame = 0
  const chapterTabs = Array.from(
    panel.querySelectorAll<HTMLButtonElement>('[data-analytics-chapter-tab]'),
  )
  const chapterPanels = Array.from(panel.querySelectorAll<HTMLElement>('[data-analytics-chapter]'))
  let chapter: AnalyticsChapterKey = pageMode
    ? analyticsChapterFromHash(window.location.hash)
    : DEFAULT_ANALYTICS_CHAPTER
  // Only the open chapter renders; the others catch up to the current version when opened.
  const renderedVersion = new WeakMap<HTMLElement, number>()
  let renderVersion = 0

  const finishPageBoot = (): void => {
    if (pageMode) document.documentElement.classList.remove(TRI_ANALYTICS_BOOT_CLASS)
  }

  const inCompareMode = (): boolean => program.retrieve().mode === 'compare'
  const comparisonIds = (): readonly string[] => program.retrieve().comparisonActivityIds

  const panelView = (
    rendered: AnalyticsPanelRender,
  ): { element: HTMLElement; mount?: () => () => void } =>
    rendered instanceof HTMLElement ? { element: rendered } : rendered

  const renderBlock = (block: HTMLElement, d: Analytics) => {
    const definition = analyticsPanelDefinition(block.dataset.chart ?? '')
    if (!definition) return
    panelCleanups.get(block)?.()
    panelCleanups.delete(block)
    renderedVersion.set(block, renderVersion)
    try {
      const view = panelView(definition.render(d, context))
      block.replaceChildren(view.element)
      if (view.element.querySelector(METHOD_NOTES))
        block.appendChild(
          el('button', 'tri-ana-method-toggle', context.formatter.text('method'), {
            type: 'button',
            'aria-expanded': String(block.classList.contains('tri-ana-block--method')),
            'data-i18n': 'method',
          }),
        )
      const cleanup = view.mount?.()
      if (cleanup) panelCleanups.set(block, cleanup)
      block.dataset.triHydrated = 'true'
    } catch {
      block.dataset.triHydrated = 'failed'
    }
  }
  const renderChapter = () => {
    const d = data
    if (!d) return
    if (renderFrame !== 0) {
      window.cancelAnimationFrame(renderFrame)
      renderFrame = 0
    }
    const scope =
      chapterPanels.find(section => section.dataset.analyticsChapter === chapter) ?? panel
    const pending = Array.from(scope.querySelectorAll<HTMLElement>('.tri-ana-block')).filter(
      block => renderedVersion.get(block) !== renderVersion,
    )
    const step = () => {
      renderFrame = 0
      const deadline = performance.now() + RENDER_SLICE_MS
      while (pending.length > 0) {
        renderBlock(pending.shift() as HTMLElement, d)
        if (performance.now() >= deadline) break
      }
      if (pending.length > 0 && live) {
        renderFrame = window.requestAnimationFrame(step)
        return
      }
      document.dispatchEvent(
        new CustomEvent('contentdecrypted', { detail: { article: panel, content: panel } }),
      )
      finishPageBoot()
      if (panel.classList.contains('tri-analytics--searching')) renderSearch()
    }
    step()
  }
  const render = (source: Analytics) => {
    data = analyticsForRange(source, program.retrieve().range)
    renderVersion += 1
    renderChapter()
  }
  const load = () => {
    const status = program.retrieve().status
    if (status === 'idle' || status === 'failed') program.dispatch({ type: 'load' })
  }
  const closeDetail = () => {
    detailGeneration += 1
    compareCleanup?.()
    compareCleanup = null
    activityCleanup?.()
    activityCleanup = null
    panel.classList.remove('tri-analytics--comparison')
    panel.classList.remove('tri-analytics--detail')
    if (detail) {
      detail.replaceChildren()
      detail.setAttribute('aria-hidden', 'true')
    }
    if (results)
      results.setAttribute(
        'aria-hidden',
        String(!inCompareMode() && !panel.classList.contains('tri-analytics--searching')),
      )
  }
  const setCompareMode = (enabled: boolean) => {
    if (inCompareMode() !== enabled) program.dispatch({ type: 'toggle-compare' })
    panel.classList.toggle('tri-analytics--compare', enabled)
    compareToggle?.setAttribute('aria-pressed', String(enabled))
  }
  const toMain = () => {
    closeDetail()
    setCompareMode(false)
    panel.classList.remove('tri-analytics--searching')
    if (results) {
      results.replaceChildren()
      results.setAttribute('aria-hidden', 'true')
    }
    program.dispatch({ type: 'reset' })
  }
  const close = () => {
    const wasOpen = root.classList.contains('tri-analytics-open')
    detailGeneration += 1
    root.classList.remove('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'true')
    program.dispatch({ type: 'close' })
    if (wasOpen && !pageMode) btn?.focus({ preventScroll: true })
  }
  const loadDetails = (): Promise<boolean> => {
    if (detailData) return Promise.resolve(true)
    if (detailPromise) return detailPromise
    const p = root.dataset.detailPath
    if (!p) return Promise.resolve(false)
    detailPromise = context.resources.detail.load(p).then(result => {
      if (!live || result.status !== 'ready') {
        if (result.status === 'error') detailPromise = null
        return false
      }
      detailData = result.value
      return true
    })
    return detailPromise
  }
  const compareActivityEligible = (activity: StravaActivityDetail | undefined): boolean =>
    activity != null && activityComparisonEligible(activity)
  const selectedCompareActivities = (): StravaActivityDetail[] =>
    comparisonIds().flatMap(id => {
      const activity = detailData?.details[id]
      return compareActivityEligible(activity) && activity ? [activity] : []
    })
  const compareSport = (): ActivityKind | null => selectedCompareActivities()[0]?.sport ?? null
  const showDetail = () => {
    panel.classList.add('tri-analytics--detail')
    detail?.setAttribute('aria-hidden', 'false')
    results?.setAttribute('aria-hidden', 'true')
    body?.scrollTo({ top: 0 })
  }
  const showActivity = (id: string) => {
    if (!detail) return
    const generation = ++detailGeneration
    void loadDetails().then(available => {
      if (
        !available ||
        !live ||
        !panel.isConnected ||
        inCompareMode() ||
        generation !== detailGeneration
      )
        return
      const d = detailData?.details?.[id]
      if (!d) return
      const card = el('div', 'tri-pop-card')
      const { head, back, actions } = detailHead(
        context.formatter.shortDate(d.date),
        d.name || d.sport,
        context.formatter.text('go back'),
      )
      actions.prepend(buildActivityAnalyzeButton(createDomFactory(context.presentation), d))
      card.appendChild(head)
      const activityView = renderDetail(context.presentation, d, detailData)
      activityView.element.querySelector('[data-activity-analyze]')?.remove()
      setActivityExpanded(activityView.element, true)
      card.appendChild(activityView.element)
      const h = detailData?.health?.[d.date]
      if (h) {
        const rec = buildRecovery(context.presentation, h)
        if (rec) card.appendChild(rec)
      }
      detail.replaceChildren(card)
      activityCleanup?.()
      activityCleanup = activityView.mount()
      showDetail()
      back.addEventListener('click', closeDetail, { once: true })
    })
  }
  // Wide comparisons scroll the chart column; narrow ones scroll the whole comparison.
  const comparisonScroller = (comparison: Element) =>
    [...comparison.querySelectorAll<HTMLElement>('.tri-compare-content, .tri-compare-charts')].find(
      node => ['auto', 'scroll'].includes(getComputedStyle(node).overflowY),
    )
  const showComparison = (chartScrollTop = 0) => {
    if (!detail) return
    const activities = selectedCompareActivities().map(activity =>
      powerViewActivity(context.presentation, activity),
    )
    if (
      activities.length < 2 ||
      activities.some(activity => activity.sport !== activities[0].sport)
    )
      return
    detailGeneration += 1
    compareCleanup?.()
    activityCleanup?.()
    activityCleanup = null
    const card = el('div', 'tri-pop-card tri-pop-card--compare')
    const { head, back, actions } = detailHead(
      `${activities.length} ${context.formatter.text('activities')}`,
      undefined,
      context.formatter.text('go back'),
    )
    const copy = el('button', 'tri-compare-copy', undefined, {
      type: 'button',
      'data-site-cursor-action': '',
      'aria-label': context.formatter.text('Copy embed link'),
      title: context.formatter.text('Copy embed link'),
    })
    const copyIcon = svg('svg', {
      class: 'copy-icon',
      width: '16',
      height: '16',
      viewBox: '-4 -4 24 24',
      fill: 'currentColor',
      'aria-hidden': 'true',
      'data-site-cursor-icon': '',
    })
    copyIcon.appendChild(svg('use', { href: '#github-copy' }))
    const checkIcon = svg('svg', {
      class: 'check-icon',
      width: '16',
      height: '16',
      viewBox: '-4 -4 24 24',
      fill: 'currentColor',
      'aria-hidden': 'true',
      'data-site-cursor-icon': '',
    })
    checkIcon.appendChild(svg('use', { href: '#github-check' }))
    copy.append(copyIcon, checkIcon)
    actions.prepend(copy)
    const copyCleanup = wireEmbedCopy(context.formatter, copy, () =>
      activityComparisonEmbed(comparisonIds()),
    )
    const comparisonContext = detailContextFromPayload(detailData)
    const health = detailData?.health ?? null
    const comparison = buildActivityComparison(
      createDomFactory(context.presentation),
      activities,
      comparisonContext,
      { health },
    )
    applyI18n(comparison, context.presentation)
    card.append(head, comparison)
    detail.replaceChildren(card)
    const interactionCleanup = wireActivityComparison(
      context.presentation,
      comparison,
      activities,
      { ctx: comparisonContext, health },
    )
    const onComparisonClick = (event: Event) => {
      if (!(event.target instanceof Element)) return
      const remove = event.target.closest<HTMLButtonElement>('[data-compare-activity-remove]')
      const activityId = remove?.dataset.compareActivityRemove
      if (!activityId || comparisonIds().length <= 2) return
      event.stopPropagation()
      comparisonScrollTop = comparisonScroller(comparison)?.scrollTop ?? 0
      program.dispatch({ type: 'remove-comparison-activity', id: activityId })
    }
    comparison.addEventListener('click', onComparisonClick)
    compareCleanup = () => {
      copyCleanup()
      interactionCleanup()
      comparison.removeEventListener('click', onComparisonClick)
    }
    panel.classList.add('tri-analytics--comparison')
    showDetail()
    const chartScroller = comparisonScroller(comparison)
    if (chartScroller && chartScrollTop > 0) chartScroller.scrollTop = chartScrollTop
    back.addEventListener('click', closeDetail, { once: true })
  }

  const resultItems = (): HTMLElement[] => activityResultItems(results)
  const setSel = (i: number) => {
    program.dispatch({ type: 'select-result', index: setActivityResultSelection(results, i) })
  }
  const comparePicker = (): HTMLElement => {
    const picker = el('div', 'tri-compare-picker')
    const top = el('div', 'tri-compare-picker-top')
    const actions = el('div', 'tri-compare-picker-actions')
    const clear = el('button', 'tri-compare-picker-clear', undefined, {
      type: 'button',
      'data-compare-clear': '',
      'data-site-cursor-close': '',
      'aria-label': context.formatter.text('clear selection'),
      'data-i18n-aria-label': 'clear selection',
      title: context.formatter.text('clear selection'),
    })
    const clearIcon = svg('svg', {
      class: 'tri-compare-picker-icon',
      viewBox: '0 0 24 24',
      fill: 'none',
      'aria-hidden': 'true',
      'data-site-cursor-icon': '',
    })
    clearIcon.appendChild(svg('path', { d: 'M6 6l12 12M18 6 6 18' }))
    clear.appendChild(clearIcon)
    clear.toggleAttribute('disabled', comparisonIds().length === 0)
    const submit = el('button', 'tri-compare-picker-submit', undefined, {
      type: 'button',
      'data-compare-submit': '',
      'data-site-cursor-action': '',
      'aria-label': context.formatter.text('compare selected'),
      'data-i18n-aria-label': 'compare selected',
      'aria-describedby': 'tri-compare-picker-submit-help',
      'aria-keyshortcuts': 'Shift+Enter',
    })
    const submitIcon = svg('svg', {
      class: 'tri-compare-picker-icon',
      viewBox: '0 0 24 24',
      fill: 'none',
      'aria-hidden': 'true',
      'data-site-cursor-icon': '',
    })
    submitIcon.appendChild(svg('path', { d: 'M12 4.5 19.5 12 12 19.5 4.5 12Z' }))
    submit.appendChild(submitIcon)
    submit.toggleAttribute('disabled', comparisonIds().length < 2)
    const submitWrap = el('span', 'tri-compare-picker-submit-wrap')
    const submitHelp = el('span', 'tri-compare-picker-tooltip', undefined, {
      id: 'tri-compare-picker-submit-help',
      role: 'tooltip',
    })
    submitHelp.append(
      el('span', undefined, context.formatter.text('compare activities'), {
        'data-i18n': 'compare activities',
      }),
    )
    submitWrap.append(submit, submitHelp)
    actions.append(submitWrap, clear)
    top.append(
      el(
        'span',
        'tri-compare-picker-instruction',
        context.formatter.text('choose 2 or more activities from one sport'),
        { 'data-i18n': 'choose 2 or more activities from one sport' },
      ),
      actions,
    )
    picker.appendChild(top)
    return picker
  }
  const toggleCompareActivity = (id: string) => {
    const selected = comparisonIds().indexOf(id)
    if (selected >= 0) {
      program.dispatch({ type: 'toggle-comparison-activity', id })
      return
    }
    const activity = detailData?.details[id]
    if (!compareActivityEligible(activity) || !activity) return
    const sport = compareSport()
    if (sport && activity.sport !== sport) return
    program.dispatch({ type: 'toggle-comparison-activity', id })
  }
  const activate = (it: HTMLElement | undefined) => {
    const id = it?.dataset.id
    if (!id) return
    if (inCompareMode()) toggleCompareActivity(id)
    else program.dispatch({ type: 'show-activity', id })
  }
  const renderSearch = () => {
    if (!results) return
    // picking re-renders the list; keep keyboard focus on the row that was picked
    const focusedId =
      document.activeElement instanceof HTMLElement && results.contains(document.activeElement)
        ? document.activeElement.dataset.id
        : undefined
    results.replaceChildren()
    if (!inCompareMode()) {
      panel.classList.remove('tri-analytics--searching')
      results.setAttribute('aria-hidden', 'true')
      return
    }
    panel.classList.add('tri-analytics--searching')
    results.setAttribute('aria-hidden', 'false')
    results.appendChild(comparePicker())
    const resultList = el('div', 'tri-compare-activity-list', undefined, {
      'data-keyboard-scroll': '',
    })
    results.appendChild(resultList)
    const sport = compareSport()
    const acts = (data?.activities ?? []).filter(
      a =>
        compareActivityEligible(detailData?.details[String(a.id)]) && (!sport || a.sport === sport),
    )
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', context.formatter.text('activities')))
      for (const a of acts.slice(0, 50)) {
        const head = el('span', 'tri-ana-ritem-h')
        head.append(buildIcon(context.presentation, a.sport), el('span', '', a.name || a.sport))
        const sub =
          `${context.formatter.longDate(a.date)} · ${dist(context.presentation, a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}` +
          (a.cadence ? (a.sport === 'run' ? ` · ${a.cadence * 2} spm` : ` · ${a.cadence} rpm`) : '')
        const it = activityResultItem(head, sub)
        it.dataset.id = String(a.id)
        const selected = comparisonIds().indexOf(it.dataset.id)
        const chosen = selected >= 0
        it.setAttribute('aria-pressed', String(chosen))
        it.classList.toggle('tri-ana-ritem--chosen', chosen)
        if (chosen) {
          it.style.setProperty('--tri-compare-color', activityCompareColor(selected))
          head.prepend(el('span', 'tri-compare-swatch', undefined, { 'aria-hidden': 'true' }))
        }
        grp.appendChild(it)
      }
      resultList.appendChild(grp)
    } else resultList.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no matches')))
    const index = Math.max(
      0,
      resultItems().findIndex(item => item.dataset.id === focusedId),
    )
    setSel(index)
    if (focusedId) resultItems()[index]?.focus({ preventScroll: true })
  }
  const refreshPicker = (): void => {
    program.dispatch({ type: 'query', value: '' })
  }
  const onResultsClick = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return
    if (event.target.closest('[data-compare-clear]')) {
      program.dispatch({ type: 'set-comparison-activities', ids: [] })
      return
    }
    if (event.target.closest('[data-compare-retry]')) {
      enterCompare()
      return
    }
    if (event.target.closest('[data-compare-submit]')) {
      program.dispatch({ type: 'submit-comparison' })
      return
    }
    activate(event.target.closest<HTMLElement>('.tri-ana-ritem') ?? undefined)
  }
  const onPickerKey = (event: KeyboardEvent) => {
    if (!inCompareMode()) return
    if (
      event.key === 'Enter' &&
      event.shiftKey &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !event.isComposing &&
      inCompareMode()
    ) {
      event.preventDefault()
      program.dispatch({ type: 'submit-comparison' })
    } else if (
      event.key === 'ArrowDown' ||
      (event.ctrlKey && (event.key === 'n' || event.key === 'N'))
    ) {
      event.preventDefault()
      setSel(program.retrieve().selectedResult + 1)
      resultItems()[program.retrieve().selectedResult]?.focus({ preventScroll: true })
    } else if (
      event.key === 'ArrowUp' ||
      (event.ctrlKey && (event.key === 'p' || event.key === 'P'))
    ) {
      event.preventDefault()
      setSel(program.retrieve().selectedResult - 1)
      resultItems()[program.retrieve().selectedResult]?.focus({ preventScroll: true })
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const its = resultItems()
      activate(
        its.find(item => item === document.activeElement) ??
          its[program.retrieve().selectedResult] ??
          its[0],
      )
    }
  }

  const enterCompare = () => {
    closeDetail()
    setCompareMode(true)
    const generation = detailGeneration
    panel.classList.add('tri-analytics--searching')
    if (results) {
      results.setAttribute('aria-hidden', 'false')
      results.replaceChildren(el('div', 'tri-ana-empty', context.formatter.text('loading')))
    }
    void loadDetails().then(available => {
      if (
        !live ||
        !inCompareMode() ||
        !panel.isConnected ||
        panel.getAttribute('aria-hidden') === 'true' ||
        generation !== detailGeneration
      )
        return
      if (!available) {
        if (results) {
          const error = el('div', 'tri-compare-load-error')
          error.append(
            el('span', 'tri-ana-empty', context.formatter.text('activity data unavailable'), {
              'data-i18n': 'activity data unavailable',
            }),
            el('button', 'tri-compare-load-retry', context.formatter.text('retry'), {
              type: 'button',
              'data-compare-retry': '',
              'data-i18n': 'retry',
            }),
          )
          results.replaceChildren(error)
        }
        return
      }
      refreshPicker()
      resultItems()[0]?.focus({ preventScroll: true })
    })
  }
  const onCompareToggle = () => {
    if (inCompareMode()) toMain()
    else enterCompare()
  }
  const open = () => {
    toMain()
    root.classList.add('tri-analytics-open')
    panel.setAttribute('aria-hidden', 'false')
    load()
    panel.focus({ preventScroll: true })
  }
  const selectChapter = (next: AnalyticsChapterKey, updateHash: boolean, focus: boolean) => {
    chapter = next
    for (const tab of chapterTabs) {
      const active = tab.dataset.analyticsChapterTab === next
      tab.setAttribute('aria-selected', String(active))
      tab.tabIndex = active ? 0 : -1
      if (active && focus) tab.focus({ preventScroll: true })
    }
    for (const section of chapterPanels) section.hidden = section.dataset.analyticsChapter !== next
    body?.scrollTo({ top: 0 })
    renderChapter()
    if (!updateHash || !pageMode) return
    const url = new URL(window.location.href)
    url.hash = next === DEFAULT_ANALYTICS_CHAPTER ? '' : next
    window.history.replaceState(window.history.state, '', url)
  }
  const onChapterClick = (event: MouseEvent) => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-analytics-chapter-tab]')
        : null
    const next = tab?.dataset.analyticsChapterTab
    if (!isChapter(next)) return
    if (
      inCompareMode() ||
      panel.classList.contains('tri-analytics--detail') ||
      panel.classList.contains('tri-analytics--comparison')
    )
      toMain()
    selectChapter(next, true, false)
  }
  const onChapterKey = (event: KeyboardEvent) => {
    if (!(event.target instanceof HTMLButtonElement)) return
    const current = chapterTabs.indexOf(event.target)
    if (current < 0) return
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? chapterTabs.length - 1
          : event.key === 'ArrowLeft'
            ? (current - 1 + chapterTabs.length) % chapterTabs.length
            : event.key === 'ArrowRight'
              ? (current + 1) % chapterTabs.length
              : -1
    const value = chapterTabs[next]?.dataset.analyticsChapterTab
    if (!isChapter(value)) return
    event.preventDefault()
    selectChapter(value, true, true)
  }
  const onHashChange = () => {
    const next = analyticsChapterFromHash(window.location.hash)
    if (next !== chapter) selectChapter(next, false, false)
  }
  const onMethodToggle = (event: MouseEvent) => {
    const toggle =
      event.target instanceof Element ? event.target.closest('.tri-ana-method-toggle') : null
    const block = toggle?.closest<HTMLElement>('.tri-ana-block')
    if (!toggle || !block) return
    toggle.setAttribute('aria-expanded', String(block.classList.toggle('tri-ana-block--method')))
  }
  const onComparisonFill = (event: Event) => {
    if (!(event instanceof CustomEvent) || !isRecord(event.detail)) return
    const anchor = event.detail.anchor
    const activityIds = typeof anchor === 'string' ? decodeActivityComparisonAnchor(anchor) : null
    if (!activityIds) return
    if (pageMode) {
      toMain()
      panel.setAttribute('aria-hidden', 'false')
      load()
    } else {
      open()
    }
    enterCompare()
    const generation = detailGeneration
    void loadDetails().then(available => {
      if (
        !available ||
        !live ||
        !inCompareMode() ||
        !panel.isConnected ||
        generation !== detailGeneration
      )
        return
      const activities = activityIds.flatMap(activityId => {
        const activity = detailData?.details[activityId]
        return activity ? [activity] : []
      })
      const sport = activities[0]?.sport
      if (
        activities.length !== activityIds.length ||
        !sport ||
        activities.some(activity => activity.sport !== sport || !compareActivityEligible(activity))
      )
        return
      program.dispatch({ type: 'set-comparison-activities', ids: activityIds })
      program.dispatch({ type: 'submit-comparison' })
    })
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.key !== 'Escape') return
    if (panel.classList.contains('tri-analytics--detail')) {
      closeDetail()
      return
    }
    if (inCompareMode()) {
      toMain()
      return
    }
    close()
  }

  const program = start({
    init: () => ({
      model: initialAnalyticsModel(defaultAnalyticsRange(root.dataset.triView)),
      effects: [],
    }),
    reduce: updateAnalytics,
    effects: (effect, state) => {
      if (effect.type === 'load-artifact') {
        const path = root.dataset.analyticsPath
        if (!path) {
          finishPageBoot()
          state.dispatch({ type: 'failed', request: effect.request })
          return
        }
        void context.resources.analytics.load(path).then(result => {
          if (!live) return
          if (result.status === 'ready') {
            sourceData = result.value
            state.dispatch({ type: 'loaded', request: effect.request })
          } else if (result.status === 'error') {
            finishPageBoot()
            state.dispatch({ type: 'failed', request: effect.request })
          }
        })
      } else if (effect.type === 'render-panels') {
        if (sourceData) render(sourceData)
      } else if (effect.type === 'render-search') {
        renderSearch()
      } else if (effect.type === 'render-activity') {
        showActivity(effect.id)
      } else if (effect.type === 'render-comparison') {
        const scrollTop = comparisonScrollTop
        comparisonScrollTop = 0
        showComparison(scrollTop)
      } else if (!pageMode) {
        btn?.focus({ preventScroll: true })
      }
    },
  })

  root.dataset.triAnalyticsRange = program.retrieve().range
  const rangeCleanup = context.events.subscribe('analyticsRange', ({ range }) => {
    root.dataset.triAnalyticsRange = range
    program.dispatch({ type: 'set-range', range })
  })

  const powerActivityCleanup = context.events.subscribe('powerActivity', request => {
    if (!panel.contains(request.source)) return
    request.handled = true
    if (inCompareMode()) setCompareMode(false)
    program.dispatch({ type: 'show-activity', id: request.activityId })
  })

  if (chapter !== DEFAULT_ANALYTICS_CHAPTER) selectChapter(chapter, false, false)
  if (pageMode) {
    window.addEventListener('hashchange', onHashChange)
    load()
  } else {
    btn?.addEventListener('click', open)
    closeBtn?.addEventListener('click', close)
    title?.addEventListener('click', toMain)
    scrim?.addEventListener('click', close)
  }
  compareToggle?.addEventListener('click', onCompareToggle)
  for (const tab of chapterTabs) {
    tab.addEventListener('click', onChapterClick)
    tab.addEventListener('keydown', onChapterKey)
  }
  body?.addEventListener('click', onMethodToggle)
  results?.addEventListener('click', onResultsClick)
  results?.addEventListener('keydown', onPickerKey)
  detail?.addEventListener('click', onCardToggle)
  document.addEventListener('keydown', onKey)
  window.addEventListener('tri:comparison-fill', onComparisonFill)
  const onUnitChange = () => {
    const comparisonVisible =
      panel.classList.contains('tri-analytics--detail') &&
      detail?.querySelector('.tri-compare') != null
    const activityId = detail?.querySelector<HTMLElement>('.tri-act[data-activity-id]')?.dataset
      .activityId
    if (sourceData) {
      render(sourceData)
    }
    if (comparisonVisible) showComparison()
    else if (activityId) showActivity(activityId)
    else if (inCompareMode()) refreshPicker()
  }
  window.addEventListener('tri:unit', onUnitChange)
  window.addEventListener('tri:locale', onUnitChange)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onUnitChange)

  return () => {
    live = false
    finishPageBoot()
    btn?.removeEventListener('click', open)
    closeBtn?.removeEventListener('click', close)
    title?.removeEventListener('click', toMain)
    scrim?.removeEventListener('click', close)
    compareToggle?.removeEventListener('click', onCompareToggle)
    for (const tab of chapterTabs) {
      tab.removeEventListener('click', onChapterClick)
      tab.removeEventListener('keydown', onChapterKey)
    }
    body?.removeEventListener('click', onMethodToggle)
    window.removeEventListener('hashchange', onHashChange)
    results?.removeEventListener('click', onResultsClick)
    results?.removeEventListener('keydown', onPickerKey)
    detail?.removeEventListener('click', onCardToggle)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('tri:comparison-fill', onComparisonFill)
    window.removeEventListener('tri:unit', onUnitChange)
    window.removeEventListener('tri:locale', onUnitChange)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnitChange)
    powerActivityCleanup()
    rangeCleanup()
    compareCleanup?.()
    activityCleanup?.()
    for (const cleanup of panelCleanups.values()) cleanup()
    panelCleanups.clear()
    program.stop()
    if (renderFrame !== 0) window.cancelAnimationFrame(renderFrame)
  }
}
