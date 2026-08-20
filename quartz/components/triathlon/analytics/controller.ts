import type { Analytics } from '../../../plugins/stores/analytics'
import type { ActivityKind } from '../../../plugins/stores/strava'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import type { AnalyticsPanelRender } from './catalog'
import { start } from '../../../functional'
import { activityCompareColor } from '../../../util/triathlon-card'
import { activityComparisonEligible } from '../../../util/triathlon-card'
import { buildActivityComparison } from '../../../util/triathlon-card'
import { dist } from '../../../util/triathlon-card'
import { dur } from '../../../util/triathlon-card'
import { powerViewActivity } from '../../../util/triathlon-card'
import { activityComparisonEmbed } from '../../../util/triathlon-comparison'
import { decodeActivityComparisonAnchor } from '../../../util/triathlon-comparison'
import { glossFor } from '../../../util/triathlon-i18n'
import { glossKeys } from '../../../util/triathlon-i18n'
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
import { analyticsPanelDefinition } from './catalog'
import { initialAnalyticsModel, updateAnalytics } from './model'
import { activityCommandHints } from './search'
import { activityResultItem } from './search'
import { activityResultItems } from './search'
import { detailHead } from './search'
import { GLOSS_CHART } from './search'
import { matchesActivityTokens } from './search'
import { parseActivityQuery } from './search'
import { SEARCH_SECTIONS } from './search'
import { setActivityResultSelection } from './search'
import { sortActivitiesBy } from './search'

const RENDER_SLICE_MS = 8

export const setupAnalytics = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const btn = root.querySelector<HTMLElement>('.tri-analytics-btn')
  const panel = root.querySelector<HTMLElement>('.tri-analytics')
  const scrim = root.querySelector<HTMLElement>('.tri-analytics-scrim')
  const closeBtn = root.querySelector<HTMLElement>('.tri-ana-close')
  const title = root.querySelector<HTMLElement>('.tri-ana-title')
  const search = root.querySelector<HTMLInputElement>('.tri-ana-search')
  const results = root.querySelector<HTMLElement>('.tri-ana-results')
  const compareToggle = root.querySelector<HTMLButtonElement>('.tri-ana-compare-toggle')
  const pageMode = root.dataset.triView === 'analytics'
  if (!panel || (!btn && !pageMode)) return null

  const body = root.querySelector<HTMLElement>('.tri-ana-body')
  const detail = root.querySelector<HTMLElement>('.tri-ana-detail')
  let live = true
  let data: Analytics | null = null
  let detailData: DetailPayload | null = null
  let detailPromise: Promise<boolean> | null = null
  const panelCleanups = new Map<HTMLElement, () => void>()
  let compareCleanup: (() => void) | null = null
  let activityCleanup: (() => void) | null = null
  let detailGeneration = 0
  let flashTimer: number | null = null
  let comparisonScrollTop = 0
  let renderFrame = 0

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
    try {
      const view = panelView(definition.render(d, context))
      block.replaceChildren(view.element)
      const cleanup = view.mount?.()
      if (cleanup) panelCleanups.set(block, cleanup)
      block.dataset.triHydrated = 'true'
    } catch {
      block.dataset.triHydrated = 'failed'
    }
  }
  const render = (d: Analytics) => {
    data = d
    if (renderFrame !== 0) {
      window.cancelAnimationFrame(renderFrame)
      renderFrame = 0
    }
    const pending = Array.from(panel.querySelectorAll<HTMLElement>('.tri-ana-block'))
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
      if (panel.classList.contains('tri-analytics--searching')) runSearch()
    }
    step()
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
    if (search) search.value = ''
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
      const { head, back } = detailHead(
        context.formatter.shortDate(d.date),
        d.name || d.sport,
        context.formatter.text('go back'),
      )
      card.appendChild(head)
      const activityView = renderDetail(context.presentation, d, detailData)
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
    const comparison = buildActivityComparison(
      createDomFactory(context.presentation),
      activities,
      detailContextFromPayload(detailData),
    )
    applyI18n(comparison, context.presentation)
    card.append(head, comparison)
    detail.replaceChildren(card)
    const interactionCleanup = wireActivityComparison(context.presentation, comparison, activities)
    const onComparisonClick = (event: Event) => {
      if (!(event.target instanceof Element)) return
      const remove = event.target.closest<HTMLButtonElement>('[data-compare-activity-remove]')
      const activityId = remove?.dataset.compareActivityRemove
      if (!activityId || comparisonIds().length <= 2) return
      event.stopPropagation()
      const nextScrollTop =
        comparison.querySelector<HTMLElement>('.tri-compare-charts')?.scrollTop ?? 0
      comparisonScrollTop = nextScrollTop
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
    const chartScroller = comparison.querySelector<HTMLElement>('.tri-compare-charts')
    if (chartScroller && chartScrollTop > 0) chartScroller.scrollTop = chartScrollTop
    back.addEventListener('click', closeDetail, { once: true })
  }

  const scrollToChart = (chart: string) => {
    const block = panel.querySelector<HTMLElement>(`.tri-ana-block[data-chart="${chart}"]`)
    if (search) search.value = ''
    panel.classList.remove('tri-analytics--searching')
    block?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    block?.classList.add('tri-ana-block--flash')
    if (flashTimer != null) window.clearTimeout(flashTimer)
    flashTimer = window.setTimeout(() => {
      flashTimer = null
      block?.classList.remove('tri-ana-block--flash')
    }, 900)
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
    if (!it) return
    if (it.dataset.chart) scrollToChart(it.dataset.chart)
    else if (it.dataset.id) {
      if (inCompareMode()) toggleCompareActivity(it.dataset.id)
      else program.dispatch({ type: 'show-activity', id: it.dataset.id })
    } else if (it.dataset.insert) {
      const tokens = search!.value.trim().split(/\s+/)
      tokens[tokens.length - 1] = it.dataset.insert
      search!.value = tokens.join(' ') + (it.dataset.insert.endsWith(':') ? '' : ' ')
      search!.focus()
      runSearch()
    }
  }
  const renderSearch = () => {
    if (!search || !results) return
    const q = program.retrieve().query.trim().toLowerCase()
    results.replaceChildren()
    if (!q && !inCompareMode()) {
      panel.classList.remove('tri-analytics--searching')
      results.setAttribute('aria-hidden', 'true')
      return
    }
    panel.classList.add('tri-analytics--searching')
    results.setAttribute('aria-hidden', 'false')
    if (inCompareMode()) results.appendChild(comparePicker())
    const resultList = inCompareMode()
      ? el('div', 'tri-compare-activity-list', undefined, { 'data-keyboard-scroll': '' })
      : results
    if (inCompareMode()) results.appendChild(resultList)
    const rawTokens = q ? q.split(/\s+/) : []
    const { filterSport, filterDate, sortKey, tokens } = parseActivityQuery(rawTokens)

    const metrics: HTMLElement[] = []
    const lastToken = rawTokens[rawTokens.length - 1] ?? ''
    const hints = rawTokens.length ? activityCommandHints(lastToken, 'activities') : []

    if (!inCompareMode() && !filterSport && !filterDate && !sortKey) {
      for (const s of SEARCH_SECTIONS)
        if (
          matchesActivityTokens(
            `${s.label} ${context.formatter.text(s.label)} ${s.hay}`.toLowerCase(),
            tokens,
          )
        ) {
          const it = activityResultItem(context.formatter.text(s.label), 'section')
          it.dataset.chart = s.chart
          metrics.push(it)
        }
      for (const key of glossKeys()) {
        const g = glossFor(context.presentation.locale, key)
        if (g && matchesActivityTokens(`${key} ${g.term} ${g.def}`.toLowerCase(), tokens)) {
          const it = activityResultItem(g.term, g.def)
          it.dataset.chart = GLOSS_CHART[key] ?? 'pmc'
          metrics.push(it)
        }
      }
    }

    const acts = sortActivitiesBy(
      (data?.activities ?? []).filter(a => {
        if (filterSport && a.sport !== filterSport) return false
        if (filterDate && (a.date < filterDate.start || a.date > filterDate.end)) return false
        if (inCompareMode()) {
          const activity = detailData?.details[String(a.id)]
          if (!compareActivityEligible(activity)) return false
          const sport = compareSport()
          if (sport && a.sport !== sport) return false
        }
        return (
          tokens.length === 0 ||
          matchesActivityTokens(`${a.name} ${a.sport} ${a.date}`.toLowerCase(), tokens)
        )
      }),
      sortKey,
    )

    if (hints.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
      for (const it of hints) grp.appendChild(it)
      resultList.appendChild(grp)
    }
    if (metrics.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', context.formatter.text('metrics & terms')))
      for (const it of metrics.slice(0, 8)) grp.appendChild(it)
      resultList.appendChild(grp)
    }
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
        if (inCompareMode()) {
          const selected = comparisonIds().indexOf(it.dataset.id)
          const chosen = selected >= 0
          it.setAttribute('aria-pressed', String(chosen))
          it.classList.toggle('tri-ana-ritem--chosen', chosen)
          if (chosen) {
            it.style.setProperty('--tri-compare-color', activityCompareColor(selected))
            head.prepend(el('span', 'tri-compare-swatch', undefined, { 'aria-hidden': 'true' }))
          }
        }
        grp.appendChild(it)
      }
      resultList.appendChild(grp)
    }
    if (!metrics.length && !acts.length && !hints.length)
      resultList.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no matches')))
    setSel(0)
  }
  const runSearch = (): void => {
    program.dispatch({ type: 'query', value: search?.value ?? '' })
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
  const onSearchKey = (event: KeyboardEvent) => {
    if (!panel.classList.contains('tri-analytics--searching')) return
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
    } else if (
      event.key === 'ArrowUp' ||
      (event.ctrlKey && (event.key === 'p' || event.key === 'P'))
    ) {
      event.preventDefault()
      setSel(program.retrieve().selectedResult - 1)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const its = resultItems()
      activate(its[program.retrieve().selectedResult] ?? its[0])
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
      runSearch()
      search?.focus()
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
    if (inCompareMode()) {
      toMain()
      return
    }
    close()
  }

  const program = start({
    init: () => ({ model: initialAnalyticsModel(), effects: [] }),
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
          if (result.status === 'ready') {
            data = result.value
            state.dispatch({ type: 'loaded', request: effect.request })
          } else if (result.status === 'error') {
            finishPageBoot()
            state.dispatch({ type: 'failed', request: effect.request })
          }
        })
      } else if (effect.type === 'render-panels') {
        if (data) render(data)
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

  const powerActivityCleanup = context.events.subscribe('powerActivity', request => {
    if (!panel.contains(request.source)) return
    request.handled = true
    if (inCompareMode()) setCompareMode(false)
    program.dispatch({ type: 'show-activity', id: request.activityId })
  })

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
  compareToggle?.addEventListener('click', onCompareToggle)
  results?.addEventListener('click', onResultsClick)
  detail?.addEventListener('click', onCardToggle)
  document.addEventListener('keydown', onKey)
  window.addEventListener('tri:comparison-fill', onComparisonFill)
  const onUnitChange = () => {
    const comparisonVisible =
      panel.classList.contains('tri-analytics--detail') &&
      detail?.querySelector('.tri-compare') != null
    const activityId = detail?.querySelector<HTMLElement>('.tri-act[data-activity-id]')?.dataset
      .activityId
    if (data) {
      render(data)
    }
    if (comparisonVisible) showComparison()
    else if (activityId) showActivity(activityId)
    else if (inCompareMode()) runSearch()
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
    search?.removeEventListener('input', runSearch)
    search?.removeEventListener('keydown', onSearchKey)
    compareToggle?.removeEventListener('click', onCompareToggle)
    results?.removeEventListener('click', onResultsClick)
    detail?.removeEventListener('click', onCardToggle)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('tri:comparison-fill', onComparisonFill)
    window.removeEventListener('tri:unit', onUnitChange)
    window.removeEventListener('tri:locale', onUnitChange)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnitChange)
    powerActivityCleanup()
    compareCleanup?.()
    activityCleanup?.()
    for (const cleanup of panelCleanups.values()) cleanup()
    panelCleanups.clear()
    program.stop()
    if (renderFrame !== 0) window.cancelAnimationFrame(renderFrame)
    if (flashTimer != null) window.clearTimeout(flashTimer)
  }
}
