import {
  pageRankLinks,
  type PageRankLink,
  type PageRankModel,
  type PageRankNode,
} from '../../../util/pagerank-view'
import { formatDelta, formatShare, formatTimes, noteHref } from './table'

const SVG = 'http://www.w3.org/2000/svg'
/** Row slots per side; fixed so the panel keeps one height for every note. */
const slots = 8

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text?: string) => {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const square = (node: PageRankNode) => {
  const mark = el('span', 'pagerank-square')
  mark.style.setProperty('--px', node.axis.toFixed(3))
  return mark
}

const noteLink = (node: PageRankNode) => {
  const link = el('a', 'internal', node.title)
  link.href = noteHref(node.slug)
  link.dataset.slug = node.slug
  return link
}

function side(
  kind: 'inbound' | 'outbound',
  links: readonly PageRankLink[],
  other: ReadonlySet<number>,
) {
  const list = el('ol', `pagerank-hood-side pagerank-hood-${kind}`)
  const drawn = links.length > slots ? links.slice(0, slots - 1) : links
  if (links.length === 0)
    list.append(
      el(
        'li',
        'pagerank-hood-empty',
        kind === 'inbound' ? 'no inbound links' : 'no outbound links',
      ),
    )
  for (const link of drawn) {
    const item = el('li')
    const button = el('button', 'pagerank-hood-node')
    button.type = 'button'
    button.dataset.prSelect = String(link.node.id)
    const both = other.has(link.node.id) ? ', links both ways' : ''
    button.setAttribute(
      'aria-label',
      `inspect ${link.node.title}, rank ${link.node.position}, carries ${formatShare(link.share)} of ${kind === 'inbound' ? 'this' : 'its'} score${both}`,
    )
    button.title = `${link.node.title} · ${formatShare(link.share)}`
    button.append(
      square(link.node),
      el('span', 'pagerank-hood-rank', `${link.node.position}`),
      el('span', 'pagerank-hood-title', link.node.title),
    )
    if (both) button.append(el('span', 'pagerank-hood-both', '↔'))
    item.append(button)
    list.append(item)
  }
  if (drawn.length < links.length)
    list.append(
      el('li', 'pagerank-hood-more', `+${links.length - drawn.length} more in the ledger`),
    )

  const edges = document.createElementNS(SVG, 'svg')
  edges.setAttribute('class', 'pagerank-hood-edges')
  edges.setAttribute('viewBox', `0 0 40 ${slots * 10}`)
  edges.setAttribute('preserveAspectRatio', 'none')
  edges.setAttribute('aria-hidden', 'true')
  const middle = slots * 5
  for (const [index, link] of drawn.entries()) {
    const y = index * 10 + 5
    const [from, to] = kind === 'inbound' ? [y, middle] : [middle, y]
    const path = document.createElementNS(SVG, 'path')
    path.setAttribute('d', `M0 ${from}C20 ${from} 20 ${to} 40 ${to}`)
    path.setAttribute('stroke-width', (0.75 + 3.25 * Math.min(1, link.share)).toFixed(2))
    path.setAttribute('vector-effect', 'non-scaling-stroke')
    edges.append(path)
  }
  return { list, edges, drawn: drawn.length }
}

const fact = (label: string, value: string) => {
  const box = el('div')
  box.append(el('dt', '', label), el('dd', '', value))
  return box
}

const ledgerRow = (link: PageRankLink) => {
  const tr = el('tr')
  const name = el('td')
  name.append(el('span', 'pagerank-hood-rank', String(link.node.position)), noteLink(link.node))
  const share = el('td', 'pagerank-share')
  const bar = el('span', 'pagerank-bar')
  bar.style.setProperty('--share', Math.min(1, link.share).toFixed(4))
  share.append(bar, el('span', '', formatShare(link.share)))
  tr.append(name, el('td', '', formatTimes(link.gives)), share)
  return tr
}

export function renderInspector(root: HTMLElement, model: PageRankModel, node: PageRankNode) {
  const { inbound, outbound } = pageRankLinks(model, node.id)
  const n = model.summary.notes
  const query = <T extends Element>(selector: string) => root.querySelector<T>(selector)!

  const title = query<HTMLElement>('[data-pr-title]')
  title.replaceChildren(noteLink(node))
  query<HTMLElement>('[data-pr-slug]').textContent = node.slug
  const top = model.nodes[node.topSource]
  query<HTMLElement>('[data-pr-facts]').replaceChildren(
    fact('rank', `${node.position} of ${n.toLocaleString('en')}`),
    fact('×mean', formatTimes(node.times)),
    fact('in · out', `${node.inbound} · ${node.outbound}`),
    fact('Δ in', formatDelta(node.delta)),
    fact('top source', top ? `${formatShare(node.topShare)} from ${top.title}` : '–'),
  )

  const inIds = new Set(inbound.map(link => link.node.id))
  const outIds = new Set(outbound.map(link => link.node.id))
  const left = side('inbound', inbound, outIds)
  const right = side('outbound', outbound, inIds)
  const center = el('div', 'pagerank-hood-center')
  center.append(
    square(node),
    el('span', 'pagerank-hood-rank', String(node.position)),
    el('span', '', `${formatTimes(node.times)}×`),
  )
  const graph = el('div', 'pagerank-hood-grid')
  graph.append(left.list, left.edges, center, right.edges, right.list)
  query<HTMLElement>('[data-pr-hood]').replaceChildren(graph)
  const count = (drawn: number, total: number) =>
    total === 0 ? 'none' : drawn === total ? `all ${total}` : `${drawn} of ${total}`
  query<HTMLElement>('[data-pr-hood-caption]').textContent =
    `inbound ${count(left.drawn, inbound.length)}, outbound ${count(right.drawn, outbound.length)}, largest share first. ` +
    `line weight is the share of the receiving note's score that the link carries; square size is the score on the log axis; the number is the global rank.`

  const base = model.summary.base
  const sum = inbound.reduce((total, link) => total + link.gives, base)
  query<HTMLElement>('[data-pr-ledger-in]').replaceChildren(
    ...(inbound.length ? inbound.map(ledgerRow) : [emptyRow('no inbound links')]),
  )
  query<HTMLElement>('[data-pr-ledger-out]').replaceChildren(
    ...(outbound.length
      ? outbound.map(ledgerRow)
      : [emptyRow('no outbound links. its walk restarts at any note.')]),
  )
  query<HTMLElement>('[data-pr-ledger-base]').textContent = formatTimes(base)
  query<HTMLElement>('[data-pr-ledger-base-share]').textContent = formatShare(base / node.times)
  query<HTMLElement>('[data-pr-ledger-total]').textContent = formatTimes(sum)
}

const emptyRow = (text: string) => {
  const tr = el('tr', 'pagerank-status')
  const td = el('td', '', text)
  td.colSpan = 3
  tr.append(td)
  return tr
}
