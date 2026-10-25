import { h, render } from 'preact'
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
import { pagerankField } from './pagerank/field'
import { renderInspector } from './pagerank/inspector'
import {
  describeNode,
  metaLine,
  number,
  pagerLine,
  readState,
  summaryValues,
  type PageRankState,
} from './pagerank/state'
import { formatTimes, pageSize, RankRows, RankSizePlot } from './pagerank/views'

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
  // The first visible row is the default selection, so it needs no parameter.
  if (note && note !== filterNodes(model, query)[0]) params.set('note', note.slug)
  const search = params.toString()
  history.replaceState(history.state, '', `${location.pathname}${search ? `?${search}` : ''}`)
}

function fillSummary(root: HTMLElement, model: PageRankModel) {
  const values = summaryValues(model)
  for (const dd of root.querySelectorAll<HTMLElement>('[data-pr-stat]'))
    dd.textContent = values[dd.dataset.prStat!] ?? '–'
  root.querySelector<HTMLElement>('[data-pr-meta]')!.textContent = metaLine(model)
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
  const state = readState(model, new URLSearchParams(location.search))
  fillSummary(root, model)

  const chartHost = root.querySelector<HTMLElement>('[data-pr-chart]')!
  const chartReadout = root.querySelector<HTMLElement>('[data-pr-chart-readout]')!
  let chartWidth: number | undefined
  let pointer: PageRankNode | null = null
  const drawChart = () =>
    render(
      h(RankSizePlot, {
        model,
        width: chartWidth,
        selected: model.nodes[state.selected] ?? null,
        pointer,
      }),
      chartHost,
    )
  const fieldHost = root.querySelector<HTMLElement>('[data-pr-field]')!
  const fieldReadout = root.querySelector<HTMLElement>('[data-pr-field-readout]')!
  const fieldHover = root.querySelector<HTMLElement>('[data-pr-field-hover]')!
  const field = pagerankField(
    fieldHost,
    model,
    {
      point: node => {
        pointer = node
        mark()
      },
      select: node => select(node.id, true),
    },
    signal,
  )
  const resize = new ResizeObserver(entries => {
    for (const entry of entries) {
      const width = Math.floor(entry.contentRect.width)
      if (entry.target === fieldHost) {
        field.render(width)
        fieldHost.dataset.layoutMs = field.layoutMs.toFixed(1)
      } else if (width > 0 && width !== chartWidth) {
        chartWidth = width
        drawChart()
      }
    }
  })
  resize.observe(chartHost)
  resize.observe(fieldHost)
  window.addCleanup(() => resize.disconnect())
  document.addEventListener('themechange', () => field.paint(), { signal })

  const scroll = root.querySelector<HTMLElement>('.pagerank-scroll')!
  const search = root.querySelector<HTMLInputElement>('[data-pr-query]')!
  const pageLabel = root.querySelector<HTMLElement>('[data-pr-page-label]')!
  const filtered = root.querySelector<HTMLElement>('[data-pr-filtered]')!
  const [previous, next] = root.querySelectorAll<HTMLButtonElement>('[data-pr-page]')
  let rows: PageRankNode[] = []

  const line = (node: PageRankNode) =>
    `${node.title} · #${node.position} · ${formatTimes(node.times)}×`
  const mark = () => {
    const selected = model.nodes[state.selected] ?? null
    drawChart()
    field.mark(selected, pointer)
    chartReadout.textContent = [
      describeNode('is selected', selected),
      pointer === selected ? '' : describeNode('is under the pointer', pointer),
    ]
      .filter(Boolean)
      .join(' ')
    // Hover reads inline at the top of the plot, in place of the #1 note; the line under the
    // canvas keeps the selection.
    const counts = (node: PageRankNode) =>
      `${line(node)} · in ${node.inbound} · out ${node.outbound}`
    if (pointer) fieldHover.textContent = counts(pointer)
    fieldHost.toggleAttribute('data-pointer', !!pointer)
    fieldReadout.textContent = selected ? `selected: ${counts(selected)}` : '–'
    fieldHost.setAttribute(
      'aria-label',
      `${number(model.summary.notes)} notes as dots on a log score axis, ` +
        `${number(model.summary.noInbound)} with no inbound link in a block to its left. ` +
        (selected ? `selected: ${line(selected)}. ` : '') +
        'left and right arrows move the selection down and up the ranking.',
    )
  }

  const renderList = (resetScroll: boolean) => {
    rows = filterNodes(model, state.query)
    const pages = Math.max(1, Math.ceil(rows.length / pageSize))
    state.page = Math.min(state.page, pages)
    const visible = rows.slice((state.page - 1) * pageSize, state.page * pageSize)
    render(h(RankRows, { rows: visible, selected: state.selected }), body)
    pageLabel.textContent = `${state.page} / ${pages}`
    previous.disabled = state.page <= 1
    next.disabled = state.page >= pages
    filtered.textContent = pagerLine(model, rows)
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
      render(
        h(RankRows, {
          rows: rows.slice((state.page - 1) * pageSize, state.page * pageSize),
          selected: id,
        }),
        body,
      )
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

  // A filter that hides the selected note moves the selection to the first visible row.
  const keepSelectionVisible = () => {
    if (rows.length > 0 && !rows.some(node => node.id === state.selected)) select(rows[0].id, false)
  }

  const update = (change: Partial<PageRankQuery>) => {
    state.query = { ...state.query, ...change }
    state.page = 1
    renderList(true)
    keepSelectionVisible()
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
