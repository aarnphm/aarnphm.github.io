import { h, render } from 'preact'
import type { PageRankModel, PageRankNode } from '../../../util/pagerank-view'
import { emptyInbound, emptyOutbound, inspectorData } from './state'
import { LinkRows, NoteLink } from './views'

/** Fills the side panel's slots; the emitter renders the same slots for the first note. */
export function renderInspector(root: HTMLElement, model: PageRankModel, node: PageRankNode) {
  const query = <T extends Element>(selector: string) => root.querySelector<T>(selector)!
  const data = inspectorData(model, node)
  render(h(NoteLink, { node }), query('[data-pr-title]'))
  query<HTMLElement>('[data-pr-slug]').textContent = node.slug
  for (const dd of root.querySelectorAll<HTMLElement>('[data-pr-fact]'))
    dd.textContent = data.facts[dd.dataset.prFact!] ?? '–'
  render(h(LinkRows, { links: data.inbound, empty: emptyInbound }), query('[data-pr-ledger-in]'))
  render(h(LinkRows, { links: data.outbound, empty: emptyOutbound }), query('[data-pr-ledger-out]'))
  query<HTMLElement>('[data-pr-ledger-base]').textContent = data.base
  query<HTMLElement>('[data-pr-ledger-base-share]').textContent = data.baseShare
  query<HTMLElement>('[data-pr-ledger-total]').textContent = data.total
}
