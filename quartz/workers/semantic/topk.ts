import type { SemanticHit } from '../../util/semantic-search'

/**
 * Dot products of one unit query against every row. Scores land in `out`, one per row. A
 * Float64Array keeps the float64 sums, so near-tied rows keep their exact order.
 */
export function scanScores(
  query: Float32Array,
  vectors: Float32Array,
  rows: number,
  dims: number,
  out: Float32Array | Float64Array,
): void {
  for (let row = 0, base = 0; row < rows; row++, base += dims) {
    let score = 0
    for (let d = 0; d < dims; d++) score += query[d] * vectors[base + d]
    out[row] = score
  }
}

/**
 * Best row per group. Chunks of one document, or passages of one PDF, supply a single piece
 * of evidence: counting them would reward length.
 */
export function reduceGroups(
  scores: Float32Array,
  groups: Int32Array,
  groupCount: number,
): { best: Float32Array; row: Int32Array } {
  const best = new Float32Array(groupCount).fill(-Infinity)
  const row = new Int32Array(groupCount).fill(-1)
  for (let i = 0; i < groups.length; i++) {
    const score = scores[i]
    const group = groups[i]
    if (score > best[group]) {
      best[group] = score
      row[group] = i
    }
  }
  return { best, row }
}

/**
 * The `k` highest finite scores, best first. A bounded min-heap keeps this at O(n log k)
 * where callers ask for every document, instead of the quadratic sorted insert it replaces.
 */
export function topK(
  scores: Float32Array | Float64Array,
  k: number,
  rowOf?: Int32Array,
): SemanticHit[] {
  const size = Math.max(0, Math.min(k, scores.length))
  if (size === 0) return []
  const heapScore = new Float64Array(size)
  const heapRow = new Int32Array(size)
  let count = 0
  const siftDown = (start: number) => {
    let parent = start
    while (true) {
      const left = parent * 2 + 1
      const right = left + 1
      let smallest = parent
      if (left < count && heapScore[left] < heapScore[smallest]) smallest = left
      if (right < count && heapScore[right] < heapScore[smallest]) smallest = right
      if (smallest === parent) return
      const score = heapScore[parent]
      const row = heapRow[parent]
      heapScore[parent] = heapScore[smallest]
      heapRow[parent] = heapRow[smallest]
      heapScore[smallest] = score
      heapRow[smallest] = row
      parent = smallest
    }
  }
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i]
    if (!Number.isFinite(score)) continue
    const row = rowOf ? rowOf[i] : i
    if (row < 0) continue
    if (count < size) {
      let child = count++
      heapScore[child] = score
      heapRow[child] = row
      while (child > 0) {
        const parent = (child - 1) >> 1
        if (heapScore[parent] <= heapScore[child]) break
        heapScore[child] = heapScore[parent]
        heapRow[child] = heapRow[parent]
        heapScore[parent] = score
        heapRow[parent] = row
        child = parent
      }
    } else if (score > heapScore[0]) {
      heapScore[0] = score
      heapRow[0] = row
      siftDown(0)
    }
  }
  const hits: SemanticHit[] = []
  for (let i = 0; i < count; i++) hits.push({ id: heapRow[i], score: heapScore[i] })
  return hits.sort((a, b) => b.score - a.score || a.id - b.id)
}
