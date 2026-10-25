// Page state and readout text, pure so the emitter renders the same first view the script shows.
import {
  filterNodes,
  pageRankLinks,
  type PageRankModel,
  type PageRankNode,
  type PageRankQuery,
  type PageRankSort,
} from '../../../util/pagerank-view'
import { formatDelta, formatShare, formatTimes } from './views'

const sorts: readonly PageRankSort[] = ['position', 'inbound', 'outbound', 'delta']
export const number = (value: number) => value.toLocaleString('en')

export interface PageRankState {
  query: PageRankQuery
  page: number
  selected: number
}

/** Reads the URL; an empty one gives the static page's view. */
export function readState(model: PageRankModel, params: URLSearchParams): PageRankState {
  const sort = sorts.find(key => key === params.get('sort')) ?? 'position'
  const folder = params.get('folder') ?? ''
  const query: PageRankQuery = {
    q: params.get('q') ?? '',
    folder: model.folders.some(entry => entry.folder === folder) ? folder : '',
    hideCourse: params.get('courses') !== '1',
    linked: params.get('linked') === '1',
    sort,
    desc: params.has('dir') ? params.get('dir') === 'desc' : sort !== 'position',
  }
  // A linked note stays selected even when a filter hides it; otherwise the first visible row.
  const note = params.get('note')
  const selected =
    model.nodes.find(node => node.slug === note) ??
    filterNodes(model, query)[0] ??
    model.byPosition[0]
  return {
    query,
    page: Math.max(1, Number.parseInt(params.get('page') ?? '1', 10) || 1),
    selected: selected?.id ?? -1,
  }
}

export function summaryValues(model: PageRankModel): Record<string, string> {
  const s = model.summary
  const percent = (count: number) => `${Math.round((count / (s.notes || 1)) * 100)}%`
  return {
    notes: number(s.notes),
    links: number(s.links),
    median: `${formatTimes(s.median)}×`,
    max: `${formatTimes(s.max)}×`,
    gini: s.gini.toFixed(2),
    top10: formatShare(s.top10),
    noInbound: `${percent(s.noInbound)} · ${number(s.noInbound)}`,
    noOutbound: `${percent(s.noOutbound)} · ${number(s.noOutbound)}`,
  }
}

export function metaLine(model: PageRankModel): string {
  const s = model.summary
  return `d = ${s.damping} · N = ${number(s.notes)} · fitted α = ${s.alpha.toFixed(2)} · baseline ${formatTimes(s.base)}×`
}

export function pagerLine(model: PageRankModel, rows: readonly PageRankNode[]): string {
  const mass = rows.reduce((sum, node) => sum + node.times, 0) / (model.summary.notes || 1)
  const mean = (pick: (node: PageRankNode) => number) =>
    rows.length ? (rows.reduce((sum, node) => sum + pick(node), 0) / rows.length).toFixed(1) : '–'
  return (
    `${number(rows.length)} of ${number(model.summary.notes)} notes · ${formatShare(mass)} of mass` +
    ` · mean in ${mean(node => node.inbound)} · mean out ${mean(node => node.outbound)}`
  )
}

export const describeNode = (label: string, node: PageRankNode | null) =>
  node
    ? `${node.title} ${label}, rank ${node.position} at ${formatTimes(node.times)}× the mean.`
    : ''

/** Everything the side panel shows for one note. */
export function inspectorData(model: PageRankModel, node: PageRankNode) {
  const { inbound, outbound } = pageRankLinks(model, node.id)
  const top = model.nodes[node.topSource]
  const base = model.summary.base
  return {
    inbound,
    outbound,
    facts: {
      rank: `${node.position} of ${number(model.summary.notes)}`,
      times: formatTimes(node.times),
      links: `${node.inbound} · ${node.outbound}`,
      delta: formatDelta(node.delta),
      source: top ? top.title : '–',
    } as Record<string, string>,
    base: formatTimes(base),
    baseShare: formatShare(base / node.times),
    total: formatTimes(inbound.reduce((total, link) => total + link.gives, base)),
  }
}

export const emptyInbound = 'no inbound links'
export const emptyOutbound = 'no outbound links. its walk restarts at any note.'
