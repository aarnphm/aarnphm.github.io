import type { NoteGraph } from './note-pagerank'

/** Scores on this page are read as score·N, so 1.00× is the mean score 1/N. */
export interface PageRankNode {
  readonly id: number
  readonly slug: string
  readonly title: string
  readonly folder: string
  readonly position: number
  readonly times: number
  /** Place on the one log axis shared by every note, 0 at the lowest score and 1 at the highest. */
  readonly axis: number
  readonly inbound: number
  readonly outbound: number
  readonly inRank: number
  /** Rank by inbound count minus rank by PageRank; positive means the sources lift the note. */
  readonly delta: number
  readonly topSource: number
  readonly topShare: number
  readonly courseIndex: boolean
  readonly search: string
}

export interface PageRankSummary {
  readonly notes: number
  readonly links: number
  readonly median: number
  readonly max: number
  readonly gini: number
  readonly top10: number
  readonly noInbound: number
  readonly noOutbound: number
  readonly iterations: number
  readonly damping: number
  /** Teleport plus spread dangling mass that every note receives, in ×mean. */
  readonly base: number
  /** Least squares fit of log score on log position over notes with an inbound link. */
  readonly alpha: number
  readonly intercept: number
}

export interface PageRankModel {
  readonly graph: NoteGraph
  readonly nodes: readonly PageRankNode[]
  readonly byPosition: readonly PageRankNode[]
  readonly folders: readonly { folder: string; count: number }[]
  readonly summary: PageRankSummary
}

export interface PageRankLink {
  readonly node: PageRankNode
  /** ×mean carried over this link: d·x(q)/L(q). */
  readonly gives: number
  /** Share of the receiving note's score that this link carries. */
  readonly share: number
}

export type PageRankSort = 'position' | 'inbound' | 'outbound' | 'delta'

export interface PageRankQuery {
  readonly q: string
  readonly folder: string
  readonly hideCourse: boolean
  readonly linked: boolean
  readonly sort: PageRankSort
  readonly desc: boolean
}

export const fold = (text: string) => text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase()

export const folderOf = (slug: string) => (slug.includes('/') ? slug.split('/', 1)[0] : 'root')

export function pageRankModel(graph: NoteGraph, title: (slug: string) => string): PageRankModel {
  const { slugs, scores, outgoing, incoming, damping } = graph
  const n = slugs.length
  const times = Array.from(scores, score => score * n)
  const order = slugs.map((_, id) => id).sort((a, b) => scores[b] - scores[a] || a - b)
  const position = new Int32Array(n)
  for (const [index, id] of order.entries()) position[id] = index + 1

  const byIn = slugs.map((_, id) => id).sort((a, b) => incoming[b].length - incoming[a].length)
  const inRank = new Int32Array(n)
  for (const [index, id] of byIn.entries()) {
    const previous = byIn[index - 1]
    inRank[id] =
      index > 0 && incoming[previous].length === incoming[id].length ? inRank[previous] : index + 1
  }

  let dangling = 0
  for (let id = 0; id < n; id++) if (outgoing[id].length === 0) dangling += scores[id]
  const base = 1 - damping + damping * dangling

  const logs = times.map(value => Math.log(Math.max(value, Number.MIN_VALUE)))
  const low = Math.min(...logs)
  const span = Math.max(...logs) - low || 1

  const nodes = slugs.map((slug, id): PageRankNode => {
    let topSource = -1
    let topGives = 0
    for (const source of incoming[id]) {
      const gives = (damping * times[source]) / outgoing[source].length
      if (gives > topGives) [topSource, topGives] = [source, gives]
    }
    const name = title(slug) || slug
    return {
      id,
      slug,
      title: name,
      folder: folderOf(slug),
      position: position[id],
      times: times[id],
      axis: (logs[id] - low) / span,
      inbound: incoming[id].length,
      outbound: outgoing[id].length,
      inRank: inRank[id],
      delta: inRank[id] - position[id],
      topSource,
      topShare: times[id] > 0 ? topGives / times[id] : 0,
      courseIndex: graph.courseIndexes.has(id),
      search: fold(`${name} ${slug}`),
    }
  })
  const byPosition = order.map(id => nodes[id])

  const counts = new Map<string, number>()
  for (const node of nodes) counts.set(node.folder, (counts.get(node.folder) ?? 0) + 1)
  const folders = [...counts]
    .map(([folder, count]) => ({ folder, count }))
    .sort((a, b) => b.count - a.count || a.folder.localeCompare(b.folder))

  return { graph, nodes, byPosition, folders, summary: summarize(graph, byPosition, base) }
}

function summarize(
  graph: NoteGraph,
  byPosition: readonly PageRankNode[],
  base: number,
): PageRankSummary {
  const n = byPosition.length
  const ascending = byPosition.map(node => node.times).reverse()
  const total = ascending.reduce((sum, value) => sum + value, 0) || 1
  let weighted = 0
  for (const [index, value] of ascending.entries()) weighted += (index + 1) * value
  const top = byPosition.slice(0, Math.ceil(n / 10)).reduce((sum, node) => sum + node.times, 0)
  const middle = Math.floor(n / 2)
  const median =
    n === 0 ? 0 : n % 2 ? ascending[middle] : (ascending[middle - 1] + ascending[middle]) / 2

  const fit = byPosition.filter(node => node.inbound > 0)
  const xs = fit.map(node => Math.log10(node.position))
  const ys = fit.map(node => Math.log10(node.times))
  const meanX = xs.reduce((sum, value) => sum + value, 0) / (xs.length || 1)
  const meanY = ys.reduce((sum, value) => sum + value, 0) / (ys.length || 1)
  let covariance = 0
  let variance = 0
  for (const [index, x] of xs.entries()) {
    covariance += (x - meanX) * (ys[index] - meanY)
    variance += (x - meanX) ** 2
  }
  const slope = variance > 0 ? covariance / variance : 0

  return {
    notes: n,
    links: graph.outgoing.reduce((sum, targets) => sum + targets.length, 0),
    median,
    max: byPosition[0]?.times ?? 0,
    gini: n > 0 ? (2 * weighted) / (n * total) - (n + 1) / n : 0,
    top10: top / total,
    noInbound: byPosition.filter(node => node.inbound === 0).length,
    noOutbound: byPosition.filter(node => node.outbound === 0).length,
    iterations: graph.iterations,
    damping: graph.damping,
    base,
    alpha: -slope,
    intercept: meanY - slope * meanX,
  }
}

/** Every authored link into and out of a note, weighted by what it carries. */
export function pageRankLinks(model: PageRankModel, id: number) {
  const { graph, nodes } = model
  const node = nodes[id]
  const carried = (source: number) =>
    (graph.damping * nodes[source].times) / graph.outgoing[source].length
  const inbound = graph.incoming[id]
    .map(source => {
      const gives = carried(source)
      return { node: nodes[source], gives, share: gives / node.times }
    })
    .sort((a, b) => b.gives - a.gives || a.node.position - b.node.position)
  const outbound = graph.outgoing[id]
    .map(target => {
      const gives = carried(id)
      return { node: nodes[target], gives, share: gives / nodes[target].times }
    })
    .sort((a, b) => b.share - a.share || a.node.position - b.node.position)
  return { inbound: inbound as PageRankLink[], outbound: outbound as PageRankLink[] }
}

export function filterNodes(model: PageRankModel, query: PageRankQuery): PageRankNode[] {
  const needle = fold(query.q.trim())
  const rows = model.byPosition.filter(
    node =>
      (!needle || node.search.includes(needle)) &&
      (!query.folder || node.folder === query.folder) &&
      !(query.hideCourse && node.courseIndex) &&
      !(query.linked && node.inbound === 0),
  )
  if (query.sort === 'position' && !query.desc) return rows
  const key = (node: PageRankNode) =>
    query.sort === 'inbound'
      ? node.inbound
      : query.sort === 'outbound'
        ? node.outbound
        : query.sort === 'delta'
          ? node.delta
          : -node.position
  const sign = query.desc ? -1 : 1
  return rows.sort((a, b) => sign * (key(a) - key(b)) || a.position - b.position)
}

/** Round decade ticks with 2× and 5× steps when a decade alone gives too few. */
export function logTicks(min: number, max: number): number[] {
  const steps = [1, 2, 5]
  const ticks: number[] = []
  for (
    let exponent = Math.floor(Math.log10(min));
    exponent <= Math.ceil(Math.log10(max));
    exponent++
  )
    for (const step of steps) {
      const value = step * 10 ** exponent
      if (value >= min && value <= max) ticks.push(value)
    }
  return ticks.length > 6 ? ticks.filter(value => Number.isInteger(Math.log10(value))) : ticks
}
