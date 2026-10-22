import type { Analytics } from '../../../../plugins/stores/analytics'
import type { AnalyticsSession } from './shell'
import {
  activityCompareColor,
  buildActivityAnalyzeButton,
  buildActivityComparison,
  dist,
  dur,
  powerViewActivity,
} from '../../../../util/triathlon-card'
import { activityComparisonEmbed } from '../../../../util/triathlon-comparison'
import { setActivityExpanded, wireActivityComparison } from '../../activity/comparison'
import { detailContextFromPayload } from '../../activity/data'
import { buildIcon, buildRecovery } from '../../activity/primitives'
import { renderDetail } from '../../activity/render'
import { applyI18n, createDomFactory, el, svg } from '../../runtime/dom'
import { wireEmbedCopy } from '../../shell/timeline'
import { analyticsPanelDefinition, type AnalyticsPanelRender } from '../catalog'
import { analyticsForRange } from '../range'
import {
  activityCommandHints,
  activityQueryTokens,
  activityResultItem,
  activityResultItems,
  detailHead,
  matchesActivityQuery,
  parseActivityQuery,
  setActivityResultSelection,
  sortActivitiesBy,
} from '../search'

const RENDER_SLICE_MS = 8
// Methodology notes stay folded behind a per-panel toggle so the numbers lead.
const METHOD_NOTES = '.tri-accl-method, .tri-ftp-note, .tri-ftp-source'

export function mountChapterViews(s: AnalyticsSession) {
  s.panelView = (
    rendered: AnalyticsPanelRender,
  ): { element: HTMLElement; mount?: () => () => void } =>
    rendered instanceof HTMLElement ? { element: rendered } : rendered

  s.renderBlock = (block: HTMLElement, d: Analytics) => {
    const definition = analyticsPanelDefinition(block.dataset.chart ?? '')
    if (!definition) return
    s.panelCleanups.get(block)?.()
    s.panelCleanups.delete(block)
    s.renderedVersion.set(block, s.renderVersion)
    try {
      const view = s.panelView(definition.render(d, s.context))
      block.replaceChildren(view.element)
      if (view.element.querySelector(METHOD_NOTES))
        block.appendChild(
          el('button', 'tri-ana-method-toggle', s.context.formatter.text('method'), {
            type: 'button',
            'aria-expanded': String(block.classList.contains('tri-ana-block--method')),
            'data-i18n': 'method',
          }),
        )
      const cleanup = view.mount?.()
      if (cleanup) s.panelCleanups.set(block, cleanup)
      block.dataset.triHydrated = 'true'
    } catch {
      block.dataset.triHydrated = 'failed'
    }
  }

  s.renderChapter = () => {
    const d = s.data
    if (!d) return
    if (s.renderFrame !== 0) {
      window.cancelAnimationFrame(s.renderFrame)
      s.renderFrame = 0
    }
    const scope =
      s.chapterPanels.find(section => section.dataset.analyticsChapter === s.chapter) ?? s.panel
    const pending = Array.from(scope.querySelectorAll<HTMLElement>('.tri-ana-block')).filter(
      block => s.renderedVersion.get(block) !== s.renderVersion,
    )
    const step = () => {
      s.renderFrame = 0
      const deadline = performance.now() + RENDER_SLICE_MS
      while (pending.length > 0) {
        s.renderBlock(pending.shift() as HTMLElement, d)
        if (performance.now() >= deadline) break
      }
      if (pending.length > 0 && s.live) {
        s.renderFrame = window.requestAnimationFrame(step)
        return
      }
      document.dispatchEvent(
        new CustomEvent('contentdecrypted', { detail: { article: s.panel, content: s.panel } }),
      )
      s.finishPageBoot()
      if (s.panel.classList.contains('tri-analytics--searching')) s.renderSearch()
    }
    step()
  }

  s.render = (source: Analytics) => {
    s.data = analyticsForRange(source, s.program.retrieve().range)
    s.renderVersion += 1
    s.renderChapter()
  }

  s.load = () => {
    const status = s.program.retrieve().status
    if (status === 'idle' || status === 'failed') s.program.dispatch({ type: 'load' })
  }
}

export function mountActivityViews(s: AnalyticsSession) {
  s.showActivity = (id: string) => {
    if (!s.detail) return
    const generation = ++s.detailGeneration
    void s.loadDetails().then(available => {
      if (
        !available ||
        !s.live ||
        !s.panel.isConnected ||
        s.inCompareMode() ||
        generation !== s.detailGeneration
      )
        return
      const d = s.detailData?.details?.[id]
      if (!d) return
      const card = el('div', 'tri-pop-card')
      const { head, back, actions } = detailHead(
        s.context.formatter.shortDate(d.date),
        d.name || d.sport,
        s.context.formatter.text('go back'),
      )
      actions.prepend(buildActivityAnalyzeButton(createDomFactory(s.context.presentation), d))
      card.appendChild(head)
      const activityView = renderDetail(s.context.presentation, d, s.detailData)
      activityView.element.querySelector('[data-activity-analyze]')?.remove()
      setActivityExpanded(activityView.element, true)
      card.appendChild(activityView.element)
      const h = s.detailData?.health?.[d.date]
      if (h) {
        const rec = buildRecovery(s.context.presentation, h)
        if (rec) card.appendChild(rec)
      }
      s.detail?.replaceChildren(card)
      s.activityCleanup?.()
      s.activityCleanup = activityView.mount()
      s.showDetail()
      back.addEventListener('click', s.closeDetail, { once: true })
    })
  }

  // Wide comparisons scroll the chart column; narrow ones scroll the whole comparison.
  s.comparisonScroller = (comparison: Element) =>
    [...comparison.querySelectorAll<HTMLElement>('.tri-compare-content, .tri-compare-charts')].find(
      node => ['auto', 'scroll'].includes(getComputedStyle(node).overflowY),
    )

  s.showComparison = (chartScrollTop = 0) => {
    if (!s.detail) return
    const activities = s
      .selectedCompareActivities()
      .map(activity => powerViewActivity(s.context.presentation, activity))
    if (
      activities.length < 2 ||
      activities.some(activity => activity.sport !== activities[0].sport)
    )
      return
    s.detailGeneration += 1
    s.compareCleanup?.()
    s.activityCleanup?.()
    s.activityCleanup = null
    const card = el('div', 'tri-pop-card tri-pop-card--compare')
    const copy = el('button', 'tri-compare-copy', undefined, {
      type: 'button',
      'data-site-cursor-action': '',
      'aria-label': s.context.formatter.text('Copy embed link'),
      title: s.context.formatter.text('Copy embed link'),
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
    // The copy sits in the bar beside the compare toggle, which turns into the back arrow.
    if (s.compareToggle) s.compareToggle.before(copy)
    else card.append(copy)
    const copyCleanup = wireEmbedCopy(s.context.formatter, copy, () =>
      activityComparisonEmbed(s.comparisonIds()),
    )
    const comparisonContext = detailContextFromPayload(s.detailData)
    const health = s.detailData?.health ?? null
    const comparison = buildActivityComparison(
      createDomFactory(s.context.presentation),
      activities,
      comparisonContext,
      { health },
    )
    applyI18n(comparison, s.context.presentation)
    card.append(comparison)
    s.detail.replaceChildren(card)
    const interactionCleanup = wireActivityComparison(
      s.context.presentation,
      comparison,
      activities,
      { ctx: comparisonContext, health },
    )
    const onComparisonClick = (event: Event) => {
      if (!(event.target instanceof Element)) return
      const remove = event.target.closest<HTMLButtonElement>('[data-compare-activity-remove]')
      const activityId = remove?.dataset.compareActivityRemove
      if (!activityId || s.comparisonIds().length <= 2) return
      event.stopPropagation()
      s.comparisonScrollTop = s.comparisonScroller(comparison)?.scrollTop ?? 0
      s.program.dispatch({ type: 'remove-comparison-activity', id: activityId })
    }
    comparison.addEventListener('click', onComparisonClick)
    s.compareCleanup = () => {
      copyCleanup()
      copy.remove()
      interactionCleanup()
      comparison.removeEventListener('click', onComparisonClick)
    }
    s.panel.classList.add('tri-analytics--comparison')
    s.setBarBack(true)
    s.showDetail()
    const chartScroller = s.comparisonScroller(comparison)
    if (chartScroller && chartScrollTop > 0) chartScroller.scrollTop = chartScrollTop
  }

  s.resultItems = (): HTMLElement[] => activityResultItems(s.results)

  s.setSel = (i: number) => {
    s.program.dispatch({ type: 'select-result', index: setActivityResultSelection(s.results, i) })
  }

  s.toggleCompareActivity = (id: string) => {
    const selected = s.comparisonIds().indexOf(id)
    if (selected >= 0) {
      s.program.dispatch({ type: 'toggle-comparison-activity', id })
      return
    }
    const activity = s.detailData?.details[id]
    if (!s.compareActivityEligible(activity) || !activity) return
    const sport = s.compareSport()
    if (sport && activity.sport !== sport) return
    s.program.dispatch({ type: 'toggle-comparison-activity', id })
  }

  s.activate = (it: HTMLElement | undefined) => {
    if (it?.dataset.insert && s.search) {
      const tokens = activityQueryTokens(s.search.value)
      tokens[tokens.length - 1] = it.dataset.insert
      s.search.value = tokens.join(' ') + (it.dataset.insert.endsWith(':') ? '' : ' ')
      s.search.focus()
      s.refreshPicker()
      return
    }
    const id = it?.dataset.id
    if (!id) return
    if (s.inCompareMode()) s.toggleCompareActivity(id)
    else s.program.dispatch({ type: 'show-activity', id })
  }

  s.renderSearch = () => {
    if (!s.results) return
    // picking re-renders the list; keep keyboard focus on the row that was picked
    const focusedId =
      document.activeElement instanceof HTMLElement && s.results.contains(document.activeElement)
        ? document.activeElement.dataset.id
        : undefined
    s.results.replaceChildren()
    if (!s.inCompareMode()) {
      s.panel.classList.remove('tri-analytics--searching')
      s.results.setAttribute('aria-hidden', 'true')
      return
    }
    s.panel.classList.add('tri-analytics--searching')
    s.results.setAttribute('aria-hidden', 'false')
    s.compareSubmit?.toggleAttribute('disabled', s.comparisonIds().length < 2)
    const resultList = el('div', 'tri-compare-activity-list', undefined, {
      'data-keyboard-scroll': '',
    })
    s.results.appendChild(resultList)
    const sport = s.compareSport()
    const tokens = activityQueryTokens(s.program.retrieve().query)
    const query = parseActivityQuery(tokens)
    const hints = activityCommandHints(
      tokens.at(-1) ?? '',
      'activities',
      sport ? [sport] : undefined,
    )
    if (hints.length) {
      const group = el('div', 'tri-ana-rgroup')
      group.append(el('div', 'tri-ana-rlabel', 'suggestions'), ...hints)
      resultList.appendChild(group)
    }
    const acts = sortActivitiesBy(
      (s.data?.activities ?? []).filter(
        a =>
          s.compareActivityEligible(s.detailData?.details[String(a.id)]) &&
          (!sport || a.sport === sport) &&
          matchesActivityQuery(a, query),
      ),
      query.sortKey,
    )
    if (acts.length) {
      const grp = el('div', 'tri-ana-rgroup')
      grp.appendChild(el('div', 'tri-ana-rlabel', s.context.formatter.text('activities')))
      for (const a of acts.slice(0, 50)) {
        const head = el('span', 'tri-ana-ritem-h')
        head.append(buildIcon(s.context.presentation, a.sport), el('span', '', a.name || a.sport))
        const sub =
          `${s.context.formatter.longDate(a.date)} · ${dist(s.context.presentation, a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}` +
          (a.cadence ? (a.sport === 'run' ? ` · ${a.cadence * 2} spm` : ` · ${a.cadence} rpm`) : '')
        const it = activityResultItem(head, sub)
        it.dataset.id = String(a.id)
        const selected = s.comparisonIds().indexOf(it.dataset.id)
        const chosen = selected >= 0
        it.setAttribute('aria-pressed', String(chosen))
        it.classList.toggle('tri-ana-ritem--chosen', chosen)
        if (chosen) {
          it.style.setProperty('--tri-compare-color', activityCompareColor(selected))
          head.prepend(
            el('span', 'tri-compare-swatch', undefined, {
              'aria-hidden': 'true',
              'data-activity-index': String(selected),
            }),
          )
        }
        grp.appendChild(it)
      }
      resultList.appendChild(grp)
    } else if (!hints.length)
      resultList.appendChild(el('div', 'tri-ana-empty', s.context.formatter.text('no matches')))
    const index = Math.max(
      0,
      s.resultItems().findIndex(item => item.dataset.id === focusedId),
    )
    s.setSel(index)
    if (focusedId) s.resultItems()[index]?.focus({ preventScroll: true })
  }

  s.refreshPicker = (): void => {
    s.program.dispatch({ type: 'query', value: s.search?.value ?? '' })
  }
}
