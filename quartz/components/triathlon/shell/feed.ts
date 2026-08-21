import type { ActivitySummary } from '../../../plugins/stores/analytics'
import type { TriathlonContext } from '../runtime/context'
import { dist } from '../../../util/triathlon-card'
import { dur } from '../../../util/triathlon-card'
import { rate } from '../../../util/triathlon-card'
import { setActivityExpanded } from '../activity/comparison'
import { buildIcon, buildRecovery } from '../activity/primitives'
import { renderDetail } from '../activity/render'
import { activityCommandHints } from '../analytics/search'
import { activityResultItems } from '../analytics/search'
import { marqueeCtl } from '../analytics/search'
import { matchesActivityTokens } from '../analytics/search'
import { parseActivityQuery } from '../analytics/search'
import { setActivityResultSelection } from '../analytics/search'
import { sortActivitiesBy } from '../analytics/search'
import { el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'

export const setupFeed = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  if (root.dataset.triView !== 'feed') return null
  const feed = root.querySelector<HTMLElement>('.tri-feed')
  const list = root.querySelector<HTMLElement>('.tri-feed-list')
  const search = root.querySelector<HTMLInputElement>('.tri-feed-search')
  const searchWrap = root.querySelector<HTMLElement>('.tri-feed-search-wrap')
  const results = root.querySelector<HTMLElement>('.tri-feed-results')
  const countEl = root.querySelector<HTMLElement>('.tri-feed-count')
  const analyticsPath = root.dataset.analyticsPath
  const detailPath = root.dataset.detailPath
  const datePrefix = root.dataset.feedPrefix ?? ''
  if (!feed || !list || !search || !searchWrap || !results || !analyticsPath) return null

  let live = true
  let acts: ActivitySummary[] = []
  let openId: string | null = null
  let selIndex = -1
  const detailCache = new Map<string, HTMLElement>()
  const detailCleanups = new Map<string, () => void>()

  const buildSub = (a: ActivitySummary): HTMLElement => {
    const sub = el('span', 'tri-feed-sub')
    const cell = (cls: string, val: string): void => {
      sub.appendChild(el('span', `tri-feed-c ${cls}`, val || '-'))
    }
    cell('tri-feed-c--date', a.date)
    cell(
      'tri-feed-c--dist',
      a.distanceKm > 0 ? dist(context.presentation, a.distanceKm, a.sport) : '',
    )
    cell('tri-feed-c--time', a.movingTimeS > 0 ? dur(a.movingTimeS) : '')
    cell(
      'tri-feed-c--pace',
      a.distanceKm > 0 && a.movingTimeS > 0
        ? rate(context.presentation, a.sport, a.distanceKm, a.movingTimeS)
        : '',
    )
    return sub
  }

  const buildDetail = (id: string): HTMLElement => {
    const cached = detailCache.get(id)
    if (cached) return cached
    const wrap = el('div', 'tri-feed-detail')
    wrap.appendChild(el('div', 'tri-ana-empty', context.formatter.text('loading')))
    detailCache.set(id, wrap)
    if (detailPath)
      void context.resources.detail.load(detailPath).then(result => {
        if (!live || !wrap.isConnected) return
        if (result.status !== 'ready') {
          if (result.status === 'error')
            wrap.replaceChildren(el('div', 'tri-ana-empty', context.formatter.text('no detail')))
          return
        }
        const payload = result.value
        const d = payload.details[id]
        if (!d) {
          wrap.replaceChildren(el('div', 'tri-ana-empty', context.formatter.text('no detail')))
          return
        }
        const activityView = renderDetail(context.presentation, d, payload)
        setActivityExpanded(activityView.element, true)
        wrap.replaceChildren(activityView.element)
        detailCleanups.get(id)?.()
        detailCleanups.set(id, activityView.mount())
        const h = payload?.health?.[d.date]
        if (h) {
          const rec = buildRecovery(context.presentation, h)
          if (rec) wrap.appendChild(rec)
        }
      })
    return wrap
  }

  const collapse = () => {
    if (openId == null) return
    const row = list.querySelector<HTMLElement>(`.tri-feed-row[data-id="${openId}"]`)
    row?.querySelector('.tri-feed-detail')?.remove()
    row?.querySelector('.tri-feed-head')?.setAttribute('aria-expanded', 'false')
    row?.classList.remove('tri-feed-row--open')
    openId = null
  }

  const expand = (id: string) => {
    if (openId === id) {
      collapse()
      return
    }
    collapse()
    const row = list.querySelector<HTMLElement>(`.tri-feed-row[data-id="${id}"]`)
    if (!row) return
    openId = id
    row.classList.add('tri-feed-row--open')
    row.querySelector('.tri-feed-head')?.setAttribute('aria-expanded', 'true')
    row.appendChild(buildDetail(id))
  }

  const hideSuggestions = () => {
    feed.classList.remove('tri-feed--searching')
    results.setAttribute('aria-hidden', 'true')
    search.setAttribute('aria-expanded', 'false')
    selIndex = -1
  }

  const resultItems = (): HTMLElement[] => activityResultItems(results)
  const setSel = (index: number) => {
    selIndex = setActivityResultSelection(results, index)
  }

  const renderSuggestions = (rawTokens: string[]) => {
    results.replaceChildren()
    if (rawTokens.length === 0) {
      hideSuggestions()
      return
    }
    const hints = activityCommandHints(rawTokens[rawTokens.length - 1], 'activities')
    if (hints.length === 0) {
      hideSuggestions()
      return
    }
    const group = el('div', 'tri-ana-rgroup')
    group.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
    group.append(...hints)
    results.appendChild(group)
    feed.classList.add('tri-feed--searching')
    results.setAttribute('aria-hidden', 'false')
    search.setAttribute('aria-expanded', 'true')
    setSel(0)
  }

  const renderList = () => {
    const query = search.value.trim().toLowerCase()
    const rawTokens = query ? query.split(/\s+/) : []
    const { filterSport, filterDate, sortKey, tokens } = parseActivityQuery(rawTokens)
    const filtered = sortActivitiesBy(
      acts.filter(activity => {
        if (filterSport && activity.sport !== filterSport) return false
        if (filterDate && (activity.date < filterDate.start || activity.date > filterDate.end))
          return false
        return (
          tokens.length === 0 ||
          matchesActivityTokens(
            `${activity.sport} ${activity.name} ${activity.date}`.toLowerCase(),
            tokens,
          )
        )
      }),
      sortKey ?? 'date',
    )
    list.replaceChildren(
      ...filtered.map(a => {
        const row = el('div', 'tri-feed-row', undefined, { role: 'listitem' })
        row.dataset.id = String(a.id)
        const head = el('button', 'tri-feed-head', undefined, {
          type: 'button',
          'aria-expanded': 'false',
        })
        head.append(
          buildIcon(context.presentation, a.sport),
          el('span', 'tri-feed-name', a.name || a.sport),
          buildSub(a),
        )
        row.appendChild(head)
        return row
      }),
    )
    if (!filtered.length)
      list.appendChild(el('div', 'tri-ana-empty', context.formatter.text('no activities')))
    if (countEl) countEl.textContent = String(filtered.length)
    list.setAttribute('aria-busy', 'false')
    renderSuggestions(rawTokens)
  }

  const activate = (item: HTMLElement | undefined) => {
    const insert = item?.dataset.insert
    if (!insert) return
    const tokens = search.value.trim().split(/\s+/)
    tokens[tokens.length - 1] = insert
    search.value = tokens.join(' ') + (insert.endsWith(':') ? '' : ' ')
    search.focus()
    collapse()
    renderList()
  }

  const onListClick = (e: MouseEvent) => {
    const head = (e.target as HTMLElement).closest<HTMLElement>('.tri-feed-head')
    const id = head?.closest<HTMLElement>('.tri-feed-row')?.dataset.id
    if (id) expand(id)
  }
  const onSearch = () => {
    collapse()
    renderList()
  }
  const onSearchKey = (event: KeyboardEvent) => {
    if (!feed.classList.contains('tri-feed--searching')) return
    if (event.key === 'ArrowDown' || (event.ctrlKey && event.key.toLowerCase() === 'n')) {
      event.preventDefault()
      setSel(selIndex + 1)
    } else if (event.key === 'ArrowUp' || (event.ctrlKey && event.key.toLowerCase() === 'p')) {
      event.preventDefault()
      setSel(selIndex - 1)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const items = resultItems()
      activate(items[selIndex] ?? items[0])
    }
  }
  const onResultsClick = (event: MouseEvent) => {
    activate(
      (event.target as HTMLElement | null)?.closest<HTMLElement>('.tri-ana-ritem') ?? undefined,
    )
  }
  const onSearchFocusOut = (event: FocusEvent) => {
    const next = event.relatedTarget
    if (next instanceof Node && searchWrap.contains(next)) return
    hideSuggestions()
  }
  const onUnit = () => {
    for (const cleanup of detailCleanups.values()) cleanup()
    detailCleanups.clear()
    detailCache.clear()
    const reopen = openId
    openId = null
    renderList()
    if (reopen) expand(reopen)
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== 'Escape') return
    if (search.value) {
      search.value = ''
      onSearch()
      return
    }
    if (openId != null) collapse()
  }

  const marquee = marqueeCtl()
  const onOver = (e: MouseEvent) => {
    const name = (e.target as HTMLElement)
      .closest<HTMLElement>('.tri-feed-head')
      ?.querySelector<HTMLElement>('.tri-feed-name')
    if (name) marquee.run(name)
  }
  const onOut = (e: MouseEvent) => {
    const head = (e.target as HTMLElement).closest<HTMLElement>('.tri-feed-head')
    const to = e.relatedTarget as Node | null
    if (head && to && head.contains(to)) return
    marquee.stop()
  }

  const powerActivityCleanup = context.events.subscribe('powerActivity', request => {
    if (
      !feed.contains(request.source) ||
      !acts.some(activity => String(activity.id) === request.activityId)
    )
      return
    request.handled = true
    collapse()
    if (search.value) {
      search.value = ''
      renderList()
    }
    expand(request.activityId)
    const row = list.querySelector<HTMLElement>(`.tri-feed-row[data-id="${request.activityId}"]`)
    row?.scrollIntoView({ block: 'nearest' })
    row?.querySelector<HTMLButtonElement>('.tri-feed-head')?.focus({ preventScroll: true })
  })

  list.addEventListener('click', onListClick)
  list.addEventListener('mouseover', onOver)
  list.addEventListener('mouseout', onOut)
  search.addEventListener('input', onSearch)
  search.addEventListener('keydown', onSearchKey)
  searchWrap.addEventListener('focusout', onSearchFocusOut)
  results.addEventListener('click', onResultsClick)
  window.addEventListener('tri:unit', onUnit)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onUnit)
  document.addEventListener('keydown', onKey)

  void context.resources.analytics.load(analyticsPath).then(result => {
    if (!live) return
    if (result.status === 'ready') {
      acts = (result.value.activities ?? []).filter(activity => activity.date.startsWith(datePrefix))
      renderList()
    } else if (result.status === 'error') {
      list.setAttribute('aria-busy', 'false')
      list.replaceChildren(el('div', 'tri-ana-empty', context.formatter.text('no data')))
    }
  })

  return () => {
    live = false
    marquee.stop()
    list.removeEventListener('click', onListClick)
    list.removeEventListener('mouseover', onOver)
    list.removeEventListener('mouseout', onOut)
    search.removeEventListener('input', onSearch)
    search.removeEventListener('keydown', onSearchKey)
    searchWrap.removeEventListener('focusout', onSearchFocusOut)
    results.removeEventListener('click', onResultsClick)
    window.removeEventListener('tri:unit', onUnit)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    document.removeEventListener('keydown', onKey)
    powerActivityCleanup()
    for (const cleanup of detailCleanups.values()) cleanup()
    detailCleanups.clear()
  }
}
