import {
  pageRankLinks,
  type PageRankLink,
  type PageRankModel,
  type PageRankNode,
} from '../../../util/pagerank-view'
import { formatDelta, formatShare, formatTimes, noteHref } from './table'

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text?: string) => {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const noteLink = (node: PageRankNode) => {
  const link = el('a', 'internal', node.title)
  link.href = noteHref(node.slug)
  link.dataset.slug = node.slug
  return link
}

const ledgerRow = (link: PageRankLink) => {
  const tr = el('tr')
  const name = el('td')
  // The rank selects the note here; the title opens it.
  const rank = el('button', 'pagerank-link-rank', String(link.node.position))
  rank.type = 'button'
  rank.dataset.prSelect = String(link.node.id)
  rank.setAttribute('aria-label', `inspect ${link.node.title}, rank ${link.node.position}`)
  name.append(rank, noteLink(link.node))
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
  const facts: Record<string, string> = {
    rank: `${node.position} of ${n.toLocaleString('en')}`,
    times: formatTimes(node.times),
    links: `${node.inbound} · ${node.outbound}`,
    delta: formatDelta(node.delta),
    source: top ? `${formatShare(node.topShare)} from ${top.title}` : '–',
  }
  for (const dd of root.querySelectorAll<HTMLElement>('[data-pr-fact]'))
    dd.textContent = facts[dd.dataset.prFact!] ?? '–'

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
