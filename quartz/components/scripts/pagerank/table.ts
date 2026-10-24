import type { PageRankNode } from '../../../util/pagerank-view'
import { resolveRelative, type FullSlug } from '../../../util/path'

export const pageSize = 100
const oneSource = 0.6

const cell = (className: string, text?: string) => {
  const td = document.createElement('td')
  td.className = className
  if (text !== undefined) td.textContent = text
  return td
}

export const formatTimes = (times: number) => times.toFixed(2)
export const formatShare = (share: number) =>
  share >= 0.1 || share === 0 ? `${Math.round(share * 100)}%` : `${(share * 100).toFixed(1)}%`
export const formatDelta = (delta: number) =>
  delta > 0 ? `+${delta}` : delta < 0 ? `−${-delta}` : '0'

export const noteHref = (slug: string) => resolveRelative('pagerank' as FullSlug, slug as FullSlug)

const chip = (text: string, title: string) => {
  const span = document.createElement('span')
  span.className = 'pagerank-chip'
  span.textContent = text
  span.title = title
  return span
}

function row(node: PageRankNode, selected: boolean): HTMLTableRowElement {
  const tr = document.createElement('tr')
  tr.dataset.id = String(node.id)
  if (selected) tr.setAttribute('aria-current', 'true')

  const select = document.createElement('button')
  select.type = 'button'
  select.className = 'pagerank-select'
  select.dataset.prSelect = String(node.id)
  select.setAttribute('aria-pressed', String(selected))
  select.setAttribute('aria-label', `inspect ${node.title}, rank ${node.position}`)
  select.textContent = String(node.position)
  const position = cell('pagerank-col-position')
  position.append(select)

  const note = cell('pagerank-col-note')
  const line = document.createElement('span')
  line.className = 'pagerank-note-line'
  const link = document.createElement('a')
  link.className = 'internal'
  link.href = noteHref(node.slug)
  link.dataset.slug = node.slug
  link.textContent = node.title
  line.append(link)
  if (node.courseIndex) line.append(chip('course index', 'course navigation index'))
  if (node.inbound > 0 && node.topShare > oneSource)
    line.append(chip('one source', `one note gives ${formatShare(node.topShare)} of this score`))
  const slug = document.createElement('span')
  slug.className = 'pagerank-slug'
  slug.textContent = node.slug
  note.append(line, slug)

  const axis = cell('pagerank-col-axis')
  const mark = document.createElement('span')
  mark.className = 'pagerank-axis-mark'
  mark.style.setProperty('--px', node.axis.toFixed(4))
  axis.append(mark)

  const delta = cell('pagerank-col-delta', formatDelta(node.delta))
  if (node.delta > 0) delta.classList.add('pagerank-rise')

  tr.append(
    position,
    note,
    cell('pagerank-col-times', formatTimes(node.times)),
    axis,
    cell('pagerank-col-inbound', String(node.inbound)),
    cell('pagerank-col-outbound', String(node.outbound)),
    delta,
  )
  return tr
}

export function renderRows(
  body: HTMLTableSectionElement,
  rows: readonly PageRankNode[],
  selected: number,
) {
  const fragment = document.createDocumentFragment()
  if (rows.length === 0) {
    const tr = document.createElement('tr')
    tr.className = 'pagerank-status'
    const td = cell(
      '',
      'no notes match. global positions are unchanged; clear a filter to see them.',
    )
    td.colSpan = 7
    tr.append(td)
    fragment.append(tr)
  }
  for (const node of rows) fragment.append(row(node, node.id === selected))
  body.replaceChildren(fragment)
}

export function markSelected(body: HTMLTableSectionElement, selected: number) {
  for (const tr of body.querySelectorAll<HTMLTableRowElement>('tr[data-id]')) {
    const on = Number(tr.dataset.id) === selected
    if (on) tr.setAttribute('aria-current', 'true')
    else tr.removeAttribute('aria-current')
    tr.querySelector('.pagerank-select')?.setAttribute('aria-pressed', String(on))
  }
}
