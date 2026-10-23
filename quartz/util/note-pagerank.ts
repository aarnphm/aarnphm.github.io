import { isFullSlug, simplifySlug, type FullSlug } from './path'
import { isSearchExcluded } from './search-exclude'
import { searchScopeIncludes } from './search-scope'

export interface NoteRankMetadata {
  readonly links: readonly string[]
  readonly filePath: string
  readonly fileName: string
  readonly protected?: boolean
}

export interface NoteRanker {
  related(currentSlug: string, limit: number): readonly FullSlug[]
  global(): readonly NoteRank[]
  position(slug: string): NoteRank | undefined
}

export interface NoteRank {
  readonly slug: FullSlug
  readonly position: number
  readonly score: number
}

const damping = 0.85
const compareSlug = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

function rank(neighbors: readonly (readonly number[])[], seed?: number): Float64Array {
  const size = neighbors.length
  let scores = new Float64Array(size)
  if (size === 0) return scores
  if (seed === undefined) scores.fill(1 / size)
  else scores[seed] = 1
  let next = new Float64Array(size)
  for (let step = 0; step < 160; step++) {
    let dangling = 0
    for (let i = 0; i < size; i++) if (neighbors[i].length === 0) dangling += scores[i]
    const restart = 1 - damping + damping * dangling
    next.fill(seed === undefined ? restart / size : 0)
    if (seed !== undefined) next[seed] = restart
    for (let i = 0; i < size; i++) {
      const contribution = (damping * scores[i]) / neighbors[i].length
      for (const target of neighbors[i]) next[target] += contribution
    }
    let difference = 0
    for (let i = 0; i < size; i++) difference += Math.abs(next[i] - scores[i])
    ;[scores, next] = [next, scores]
    if (difference <= 1e-10) break
  }
  return scores
}

/** Snapshots the public note graph; its caller only supplies the navigation context. */
export function createNoteRanker(data: Readonly<Record<string, NoteRankMetadata>>): NoteRanker {
  const slugs = Object.keys(data)
    .filter(isFullSlug)
    .filter(slug => {
      const note = data[slug]
      return (
        !note.protected &&
        searchScopeIncludes('notes', { ...note, slug }) &&
        !isSearchExcluded(slug) &&
        !isSearchExcluded(note.fileName)
      )
    })
    .sort(compareSlug)
  const ids = new Map<string, number>(slugs.map((slug, id) => [slug, id]))
  const simplified = new Map<string, number | null>()
  for (const [id, slug] of slugs.entries()) {
    const key = simplifySlug(slug).replace(/\/$/, '') || '/'
    simplified.set(key, simplified.has(key) ? null : id)
  }
  const resolve = (value: string): number | undefined => {
    const path = value.split('#', 1)[0]
    const exact = ids.get(path)
    if (exact !== undefined) return exact
    const normalized = path.replace(/\/$/, '') || '/'
    return simplified.get(normalized) ?? undefined
  }
  const directed = slugs.map(() => new Set<number>())
  const undirected = slugs.map(() => new Set<number>())
  for (const [source, slug] of slugs.entries()) {
    for (const link of data[slug].links) {
      const target = resolve(link)
      if (target === undefined || source === target) continue
      directed[source].add(target)
      undirected[source].add(target)
      undirected[target].add(source)
    }
  }
  const adjacency = (sets: Set<number>[]) => sets.map(set => [...set].sort((a, b) => a - b))
  const outgoing = adjacency(directed)
  const nearby = adjacency(undirected)
  // Course routing reserves these indexes for course homes and imported navigation.
  // Keep them in the graph so course-local walks and authored note scores stay intact.
  const courseIndexes = new Set(
    slugs.flatMap((slug, id) =>
      /^courses\/[^/]+\/(?:index|(?:pages|resources)\/(?:.+\/)?index)$/.test(slug) ? [id] : [],
    ),
  )
  const order = (scores: Float64Array) =>
    slugs.map((_, id) => id).sort((a, b) => scores[b] - scores[a] || a - b)
  let globalOrder: number[] | undefined
  let globalRows: readonly NoteRank[] | undefined
  const positions = new Map<number, NoteRank>()
  function global() {
    if (globalRows) return globalRows
    const scores = rank(outgoing)
    globalOrder = order(scores)
    globalRows = Object.freeze(
      globalOrder.map((id, index) => {
        const row = Object.freeze({ slug: slugs[id], position: index + 1, score: scores[id] })
        positions.set(id, row)
        return row
      }),
    )
    return globalRows
  }
  const cache = new Map<string, readonly FullSlug[]>()
  return {
    global,
    position(slug) {
      global()
      const id = resolve(slug)
      return id === undefined ? undefined : positions.get(id)
    },
    related(currentSlug, limit) {
      if (!Number.isFinite(limit) || limit < 1 || slugs.length === 0) return Object.freeze([])
      const count = Math.min(slugs.length, Math.floor(limit))
      const current = resolve(currentSlug)
      const path = currentSlug.split('#', 1)[0]
      const homepage = path === 'index' || path === '/' || path === ''
      const key = `${homepage ? 'home' : (current ?? 'missing')}:${count}`
      const cached = cache.get(key)
      if (cached) {
        cache.delete(key)
        cache.set(key, cached)
        return cached
      }
      const selected = new Set<number>()
      if (current !== undefined && slugs[current] !== 'index' && nearby[current].length > 0) {
        const scores = rank(nearby, current)
        for (const id of order(scores)) {
          if (id !== current && scores[id] > 0) selected.add(id)
          if (selected.size === count) break
        }
      }
      if (selected.size < count) {
        global()
        for (const id of globalOrder ?? []) {
          if (homepage && courseIndexes.has(id)) continue
          if (id !== current) selected.add(id)
          if (selected.size === count) break
        }
      }
      const result = Object.freeze([...selected].map(id => slugs[id]))
      cache.set(key, result)
      if (cache.size > 16) {
        const oldest = cache.keys().next().value
        if (oldest !== undefined) cache.delete(oldest)
      }
      return result
    },
  }
}
