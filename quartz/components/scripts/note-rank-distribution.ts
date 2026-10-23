import type { NoteGraph, NoteRank, NoteRanker } from '../../util/note-pagerank'

const namespace = 'http://www.w3.org/2000/svg'
const width = 80
const barTop = 14
const barHeight = 7
const axisY = 25
// Sources narrower than this merge into one remainder segment.
const minSegment = 2

interface Scale {
  ids: Map<string, number>
  min: number
  span: number
  median: number
}

const scales = new WeakMap<NoteGraph, Scale>()

function scaleOf(graph: NoteGraph): Scale {
  let scale = scales.get(graph)
  if (scale) return scale
  const logs = Array.from(graph.scores, score => Math.log(Math.max(score, Number.MIN_VALUE)))
  const sorted = [...logs].sort((a, b) => a - b)
  const min = sorted[0] ?? 0
  scale = {
    ids: new Map(graph.slugs.map((slug, id) => [slug, id])),
    min,
    span: (sorted.at(-1) ?? 0) - min || 1,
    median: sorted[Math.floor(sorted.length / 2)] ?? 0,
  }
  scales.set(graph, scale)
  return scale
}

const node = (name: string, attributes: Record<string, string | number>) => {
  const element = document.createElementNS(namespace, name)
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value))
  return element
}

/**
 * One note's score as a bar on the corpus log scale, split into what makes it up: the share every
 * note gets, then each inbound link's gift d·x_q/L(q), largest first. Length places the note among all
 * notes; the segments say whether one source or many carry it.
 */
export function noteRankDistribution(ranker: NoteRanker, rank: NoteRank): SVGSVGElement {
  const graph = ranker.graph()
  const scale = scaleOf(graph)
  const n = graph.slugs.length
  const id = scale.ids.get(rank.slug)
  const times = rank.score * n
  const gifts =
    id === undefined
      ? []
      : graph.incoming[id]
          .map(q => (graph.damping * graph.scores[q] * n) / (graph.outgoing[q].length || 1))
          .sort((a, b) => b - a)
  const received = gifts.reduce((sum, gift) => sum + gift, 0)
  const base = Math.max(0, times - received)
  const x = (score: number) =>
    ((Math.log(Math.max(score, Number.MIN_VALUE)) - scale.min) / scale.span) * width
  const length = Math.max(2, x(rank.score))

  const top = gifts[0] ?? 0
  const share = (value: number) => `${Math.round((value / (times || 1)) * 100)}%`
  const label =
    `PageRank ${rank.position} of ${n} notes, ${times.toFixed(2)} times the mean. ` +
    (gifts.length
      ? `${gifts.length} inbound ${gifts.length === 1 ? 'link gives' : 'links give'} ${share(received)}; the largest gives ${share(top)}. `
      : 'No inbound links; the whole score is the share every note gets. ') +
    'Bar length is the score on a log scale across all notes; the tick is the median note.'

  const svg = node('svg', {
    viewBox: `0 0 ${width} 28`,
    role: 'img',
    'aria-label': label,
  }) as SVGSVGElement
  svg.classList.add('suggestion-rank')
  const title = node('title', {})
  title.textContent = label
  const position = node('text', { x: 0, y: 9, class: 'rank-position' })
  position.textContent = `#${rank.position}`
  const inbound = node('text', { x: width, y: 9, 'text-anchor': 'end', class: 'rank-inbound' })
  inbound.textContent = `${gifts.length} in`
  svg.append(
    title,
    position,
    inbound,
    node('line', { x1: 0, x2: width, y1: axisY, y2: axisY, class: 'rank-axis' }),
  )

  // Base, then sources largest first. The largest always draws;
  // later sources under minSegment fold into one remainder.
  const segments: { value: number; kind: string }[] = [{ value: base, kind: 'rank-base' }]
  let rest = 0
  for (const [index, gift] of gifts.entries()) {
    if (index > 0 && (gift / (times || 1)) * length < minSegment) rest += gift
    else segments.push({ value: gift, kind: index === 0 ? 'rank-top' : 'rank-source' })
  }
  if (rest > 0) segments.push({ value: rest, kind: 'rank-rest' })
  let cursor = 0
  for (const { value, kind } of segments) {
    const span = (value / (times || 1)) * length
    if (span <= 0) continue
    svg.append(
      node('rect', {
        x: cursor.toFixed(2),
        y: barTop,
        width: Math.max(0.5, span - (cursor + span < length ? 0.75 : 0)).toFixed(2),
        height: barHeight,
        class: kind,
      }),
    )
    cursor += span
  }
  const median = (((scale.median - scale.min) / scale.span) * width).toFixed(2)
  svg.append(node('line', { x1: median, x2: median, y1: barTop - 2, y2: 28, class: 'rank-median' }))
  return svg
}
