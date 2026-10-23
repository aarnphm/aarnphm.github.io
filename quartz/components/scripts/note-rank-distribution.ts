import type { NoteRank } from '../../util/note-pagerank'

const namespace = 'http://www.w3.org/2000/svg'
const bins = 24
const cache = new WeakMap<readonly NoteRank[], { counts: number[]; min: number; span: number }>()

/** Logarithmic score bins keep the long PageRank tail visible in a small chart. */
export function noteRankDistribution(rows: readonly NoteRank[], rank: NoteRank): SVGSVGElement {
  let distribution = cache.get(rows)
  if (!distribution) {
    const scores = rows.map(row => Math.log(Math.max(row.score, Number.MIN_VALUE)))
    const min = Math.min(...scores)
    const span = Math.max(...scores) - min || 1
    const counts = Array.from({ length: bins }, () => 0)
    for (const score of scores)
      counts[Math.min(bins - 1, Math.floor(((score - min) / span) * bins))]++
    distribution = { counts, min, span }
    cache.set(rows, distribution)
  }
  const { counts, min, span } = distribution
  const svg = document.createElementNS(namespace, 'svg')
  svg.classList.add('suggestion-rank')
  svg.setAttribute('viewBox', '0 0 96 32')
  svg.setAttribute('role', 'img')
  const label = `PageRank ${rank.position} of ${rows.length} notes, ${(rank.score * rows.length).toFixed(2)} times the mean. Score distribution, low to high on a logarithmic scale; marker shows this note.`
  svg.setAttribute('aria-label', label)
  const title = document.createElementNS(namespace, 'title')
  title.textContent = label
  svg.append(title)
  const peak = Math.max(...counts)
  for (const [index, count] of counts.entries()) {
    if (!count) continue
    const bar = document.createElementNS(namespace, 'rect')
    const height = (count / peak) * 23
    bar.setAttribute('x', String(index * 4))
    bar.setAttribute('y', String(31 - height))
    bar.setAttribute('width', '3')
    bar.setAttribute('height', String(height))
    bar.setAttribute('fill', 'currentColor')
    svg.append(bar)
  }
  const x = Math.max(
    2,
    Math.min(94, ((Math.log(Math.max(rank.score, Number.MIN_VALUE)) - min) / span) * 96),
  )
  const marker = document.createElementNS(namespace, 'path')
  marker.classList.add('rank-marker')
  marker.setAttribute('d', `M${x - 2} 0h4l-2 5z M${x - 0.5} 6h1v26h-1z`)
  svg.append(marker)
  return svg
}
