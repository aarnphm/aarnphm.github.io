import { createNoteRanker } from '../../util/note-pagerank'
import {
  filterNodes,
  pageRankModel,
  type PageRankModel,
  type PageRankNode,
  type PageRankQuery,
  type PageRankSort,
} from '../../util/pagerank-view'
import { buildSelect } from '../controls/select'
import { currentNavSignal } from './nav-lifecycle'
import { rankSizeChart } from './pagerank/chart'
import { renderInspector } from './pagerank/inspector'
import { formatShare, formatTimes, markSelected, pageSize, renderRows } from './pagerank/table'

const sorts: readonly PageRankSort[] = ['position', 'inbound', 'outbound', 'delta']
const number = (value: number) => value.toLocaleString('en')

interface PageRankState {
  query: PageRankQuery
  page: number
  selected: number
}

function readState(model: PageRankModel): PageRankState {
  const params = new URLSearchParams(location.search)
  const sort = sorts.find(key => key === params.get('sort')) ?? 'position'
  const folder = params.get('folder') ?? ''
  const note = params.get('note')
  const selected = model.nodes.find(node => node.slug === note) ?? model.byPosition[0]
  return {
    query: {
      q: params.get('q') ?? '',
      folder: model.folders.some(entry => entry.folder === folder) ? folder : '',
      hideCourse: params.get('courses') !== '1',
      linked: params.get('linked') === '1',
      sort,
      desc: params.has('dir') ? params.get('dir') === 'desc' : sort !== 'position',
    },
    page: Math.max(1, Number.parseInt(params.get('page') ?? '1', 10) || 1),
    selected: selected?.id ?? -1,
  }
}

function writeState(model: PageRankModel, state: PageRankState) {
  const { query } = state
  const params = new URLSearchParams()
  if (query.q) params.set('q', query.q)
  if (query.folder) params.set('folder', query.folder)
  if (!query.hideCourse) params.set('courses', '1')
  if (query.linked) params.set('linked', '1')
  if (query.sort !== 'position') params.set('sort', query.sort)
  if (query.desc !== (query.sort !== 'position')) params.set('dir', query.desc ? 'desc' : 'asc')
  if (state.page > 1) params.set('page', String(state.page))
  const note = model.nodes[state.selected]
  if (note && note !== model.byPosition[0]) params.set('note', note.slug)
  const search = params.toString()
  history.replaceState(history.state, '', `${location.pathname}${search ? `?${search}` : ''}`)
}

function fillSummary(root: HTMLElement, model: PageRankModel) {
  const s = model.summary
  const percent = (count: number) => `${Math.round((count / (s.notes || 1)) * 100)}%`
  const values: Record<string, string> = {
    notes: number(s.notes),
    links: number(s.links),
    median: `${formatTimes(s.median)}×`,
    max: `${formatTimes(s.max)}×`,
    gini: s.gini.toFixed(2),
    top10: formatShare(s.top10),
    noInbound: `${percent(s.noInbound)} · ${number(s.noInbound)}`,
    noOutbound: `${percent(s.noOutbound)} · ${number(s.noOutbound)}`,
  }
  for (const dd of root.querySelectorAll<HTMLElement>('[data-pr-stat]'))
    dd.textContent = values[dd.dataset.prStat!] ?? '–'
  root.querySelector<HTMLElement>('[data-pr-meta]')!.textContent =
    `d = ${s.damping} · N = ${number(s.notes)} · ${s.iterations} iterations to a change below 1e-10 · ` +
    `fitted α = ${s.alpha.toFixed(2)} · every note starts from ${formatTimes(s.base)}× · ` +
    `protected notes, attachments and excluded paths are not ranked`
}

document.addEventListener('nav', async () => {
  const root = document.querySelector<HTMLElement>('[data-pagerank]')
  if (!root) return
  const signal = currentNavSignal()
  const data = await fetchData.catch(() => undefined)
  if (signal.aborted || !root.isConnected) return
  const body = root.querySelector<HTMLTableSectionElement>('[data-pr-rows]')!
  if (!data) {
    body.querySelector('td')!.textContent = 'the link graph did not load. reload to try again.'
    return
  }
  const model = pageRankModel(createNoteRanker(data).graph(), slug => data[slug]?.title ?? '')
  const state = readState(model)
  fillSummary(root, model)

  const chartHost = root.querySelector<HTMLElement>('[data-pr-chart]')!
  const chart = rankSizeChart(chartHost, model)
  const chartReadout = root.querySelector<HTMLElement>('[data-pr-chart-readout]')!
  const resize = new ResizeObserver(([entry]) => chart.render(Math.floor(entry.contentRect.width)))
  resize.observe(chartHost)
  window.addCleanup(() => resize.disconnect())

  const scroll = root.querySelector<HTMLElement>('.pagerank-scroll')!
  const search = root.querySelector<HTMLInputElement>('[data-pr-query]')!
  const pageLabel = root.querySelector<HTMLElement>('[data-pr-page-label]')!
  const filtered = root.querySelector<HTMLElement>('[data-pr-filtered]')!
  const [previous, next] = root.querySelectorAll<HTMLButtonElement>('[data-pr-page]')
  let rows: PageRankNode[] = []
  let pointer: PageRankNode | null = null

  const describe = (label: string, node: PageRankNode | null) =>
    node
      ? `${node.title} ${label}, rank ${node.position} at ${formatTimes(node.times)}× the mean.`
      : ''
  const mark = () => {
    const selected = model.nodes[state.selected] ?? null
    chart.mark(selected, pointer)
    chartReadout.textContent = [
      describe('is selected', selected),
      pointer === selected ? '' : describe('is under the pointer', pointer),
    ]
      .filter(Boolean)
      .join(' ')
  }

  const renderList = (resetScroll: boolean) => {
    rows = filterNodes(model, state.query)
    const pages = Math.max(1, Math.ceil(rows.length / pageSize))
    state.page = Math.min(state.page, pages)
    renderRows(body, rows.slice((state.page - 1) * pageSize, state.page * pageSize), state.selected)
    pageLabel.textContent = `${state.page} / ${pages}`
    previous.disabled = state.page <= 1
    next.disabled = state.page >= pages
    const mass = rows.reduce((sum, node) => sum + node.times, 0) / (model.summary.notes || 1)
    const mean = (pick: (node: PageRankNode) => number) =>
      rows.length ? (rows.reduce((sum, node) => sum + pick(node), 0) / rows.length).toFixed(1) : '–'
    filtered.textContent =
      `${number(rows.length)} of ${number(model.summary.notes)} notes · ${formatShare(mass)} of mass` +
      ` · mean in ${mean(node => node.inbound)} · mean out ${mean(node => node.outbound)}`
    for (const button of root.querySelectorAll<HTMLButtonElement>('[data-pr-sort]')) {
      const active = button.dataset.prSort === state.query.sort
      button
        .closest('th')!
        .setAttribute(
          'aria-sort',
          active ? (state.query.desc ? 'descending' : 'ascending') : 'none',
        )
    }
    if (resetScroll) scroll.scrollTop = 0
    writeState(model, state)
  }

  const select = (id: number, reveal: boolean) => {
    const node = model.nodes[id]
    if (!node) return
    state.selected = id
    renderInspector(root, model, node)
    const index = rows.indexOf(node)
    const page = Math.floor(index / pageSize) + 1
    if (reveal && index >= 0 && page !== state.page) {
      state.page = page
      renderList(false)
    } else {
      markSelected(body, id)
      writeState(model, state)
    }
    const tr = body.querySelector<HTMLElement>(`tr[data-id="${id}"]`)
    const headerHeight = scroll.querySelector('thead')?.getBoundingClientRect().height ?? 0
    if (
      reveal &&
      tr &&
      (tr.offsetTop < scroll.scrollTop + headerHeight ||
        tr.offsetTop > scroll.scrollTop + scroll.clientHeight - tr.offsetHeight)
    )
      scroll.scrollTop = tr.offsetTop - scroll.clientHeight / 2
    mark()
  }

  const update = (change: Partial<PageRankQuery>) => {
    state.query = { ...state.query, ...change }
    state.page = 1
    renderList(true)
  }

  search.value = state.query.q
  for (const button of root.querySelectorAll<HTMLButtonElement>('[data-pr-toggle]')) {
    // The courses button shows course indexes, which are hidden by default.
    const courses = button.dataset.prToggle === 'courses'
    button.setAttribute(
      'aria-pressed',
      String(courses ? !state.query.hideCourse : state.query.linked),
    )
    button.addEventListener(
      'click',
      () => {
        const on = button.getAttribute('aria-pressed') !== 'true'
        button.setAttribute('aria-pressed', String(on))
        update(courses ? { hideCourse: !on } : { linked: on })
      },
      { signal },
    )
  }
  search.addEventListener('input', () => update({ q: search.value }), { signal })

  const folders = buildSelect<string>({
    id: 'pagerank-folder',
    label: 'folder',
    groups: [
      {
        options: [
          { value: '', label: `all folders · ${number(model.summary.notes)}` },
          ...model.folders.map(({ folder, count }) => ({
            value: folder,
            label: `${folder} · ${number(count)}`,
          })),
        ],
      },
    ],
    selected: state.query.folder,
    onSelect: folder => update({ folder }),
  })
  root.querySelector<HTMLElement>('[data-pr-folder]')!.replaceChildren(folders.element)
  window.addCleanup(folders.mount())

  // Column headings explain themselves in the line above the table on hover or focus.
  const explainLine = root.querySelector<HTMLElement>('[data-pr-explain-line]')!
  const explainDefault = explainLine.textContent ?? ''
  const thead = root.querySelector<HTMLElement>('.pagerank-table thead')!
  const explainFrom = (event: Event) => {
    const th = (event.target as Element).closest<HTMLElement>('th[data-pr-explain]')
    explainLine.textContent = th?.dataset.prExplain ?? explainDefault
  }
  const explainReset = () => (explainLine.textContent = explainDefault)
  thead.addEventListener('pointerover', explainFrom, { signal })
  thead.addEventListener('focusin', explainFrom, { signal })
  thead.addEventListener('pointerleave', explainReset, { signal })
  thead.addEventListener('focusout', explainReset, { signal })

  for (const button of root.querySelectorAll<HTMLButtonElement>('[data-pr-sort]'))
    button.addEventListener(
      'click',
      () => {
        const sort = button.dataset.prSort as PageRankSort
        const desc = sort === state.query.sort ? !state.query.desc : sort !== 'position'
        update({ sort, desc })
      },
      { signal },
    )
  previous.addEventListener('click', () => ((state.page -= 1), renderList(true)), { signal })
  next.addEventListener('click', () => ((state.page += 1), renderList(true)), { signal })

  document.addEventListener(
    'keydown',
    event => {
      const key = event.key.toLowerCase()
      if (
        event.defaultPrevented ||
        !event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey ||
        (key !== 'n' && key !== 'p')
      )
        return
      const target = event.target
      if (!(target instanceof HTMLElement)) return
      if (target !== document.body && !root.contains(target)) return
      if (
        target.closest(
          'input, textarea, select, [contenteditable], [role="dialog"], [role="listbox"]',
        )
      )
        return
      if (!rows.length) return
      event.preventDefault()
      const start = (state.page - 1) * pageSize
      const end = Math.min(start + pageSize, rows.length)
      const index = rows.findIndex(node => node.id === state.selected)
      const nextIndex =
        index < start || index >= end
          ? key === 'n'
            ? start
            : end - 1
          : Math.max(0, Math.min(rows.length - 1, index + (key === 'n' ? 1 : -1)))
      const node = rows[nextIndex]
      select(node.id, true)
      body
        .querySelector<HTMLButtonElement>(`tr[data-id="${node.id}"] .pagerank-select`)
        ?.focus({ preventScroll: true })
    },
    { signal, capture: true },
  )

  root.addEventListener(
    'click',
    event => {
      const target = event.target as Element
      if (target.closest('a')) return
      const button = target.closest<HTMLElement>('[data-pr-select]')
      const tr = target.closest<HTMLElement>('tbody[data-pr-rows] tr[data-id]')
      const id = button?.dataset.prSelect ?? tr?.dataset.id
      if (id !== undefined) select(Number(id), !!button && !tr)
    },
    { signal },
  )
  body.addEventListener(
    'keydown',
    event => {
      const button = (event.target as Element).closest<HTMLButtonElement>('.pagerank-select')
      if (!button || (event.key !== 'ArrowDown' && event.key !== 'ArrowUp')) return
      const tr = button.closest('tr')!
      const sibling = event.key === 'ArrowDown' ? tr.nextElementSibling : tr.previousElementSibling
      const target = sibling?.querySelector<HTMLButtonElement>('.pagerank-select')
      if (!target) return
      event.preventDefault()
      target.focus()
    },
    { signal },
  )
  const point = (event: Event) => {
    const tr = (event.target as Element).closest<HTMLElement>('tr[data-id]')
    const node = tr ? model.nodes[Number(tr.dataset.id)] : null
    if (node === pointer) return
    pointer = node
    mark()
  }
  const leave = () => {
    pointer = null
    mark()
  }
  body.addEventListener('pointerover', point, { signal })
  body.addEventListener('focusin', point, { signal })
  body.addEventListener('pointerleave', leave, { signal })
  body.addEventListener('focusout', leave, { signal })

  renderList(false)
  select(state.selected, false)
})
