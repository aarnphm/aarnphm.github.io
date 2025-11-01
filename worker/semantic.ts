type VectorShardMeta = {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  sha256?: string
  byteStride: number
}

type LevelSection = {
  level: number
  indptr: { offset: number; elements: number; byteLength: number }
  indices: { offset: number; elements: number; byteLength: number }
}

type ChunkMetadata = {
  parentSlug: string
  chunkId: number
}

type Manifest = {
  version: number
  dims: number
  dtype: string
  normalized: boolean
  rows: number
  shardSizeRows: number
  vectors: {
    dtype: string
    rows: number
    dims: number
    shards: VectorShardMeta[]
  }
  ids: string[]
  titles?: string[]
  chunkMetadata?: Record<string, ChunkMetadata>
  hnsw: {
    M: number
    efConstruction: number
    entryPoint: number
    maxLevel: number
    graph: {
      path: string
      sha256?: string
      levels: LevelSection[]
    }
  }
}

type SearchHit = { id: number; score: number }

class SemanticSearchEngine {
  private manifest: Manifest | null = null
  private vectorsView: Float32Array | null = null
  private dims = 0
  private rows = 0
  private entryPoint = -1
  private maxLevel = 0
  private efDefault = 128
  private levelGraph: { indptr: Uint32Array; indices: Uint32Array }[] = []
  private manifestIds: string[] = []
  private chunkMetadata: Record<string, ChunkMetadata> = {}

  private async fetchBinary(path: string): Promise<ArrayBuffer> {
    const res = await fetch(path)
    if (!res.ok) {
      throw new Error(`failed to fetch ${path}: ${res.status} ${res.statusText}`)
    }
    return await res.arrayBuffer()
  }

  async initialize() {
    const res = await fetch("/embeddings/manifest.json")
    if (!res.ok) {
      throw new Error(`failed to fetch manifest: ${res.status}`)
    }
    this.manifest = (await res.json()) as Manifest

    if (this.manifest.vectors.dtype !== "fp32") {
      throw new Error(
        `unsupported embedding dtype '${this.manifest.vectors.dtype}', expected fp32`,
      )
    }

    this.dims = this.manifest.dims
    this.rows = this.manifest.rows
    this.manifestIds = this.manifest.ids || []
    this.chunkMetadata = this.manifest.chunkMetadata || {}

    const totalBytes = this.rows * this.dims * Float32Array.BYTES_PER_ELEMENT
    const buffer = new Float32Array(totalBytes)

    for (const shard of this.manifest.vectors.shards) {
      const payload = await this.fetchBinary(shard.path)
      const view = new Float32Array(payload)
      if (view.length !== shard.rows * this.dims) {
        throw new Error(
          `shard ${shard.path} has mismatched length (expected ${shard.rows * this.dims}, got ${view.length})`,
        )
      }
      buffer.set(view, shard.rowOffset * this.dims)
    }

    this.vectorsView = buffer

    const graphBuffer = await this.fetchBinary(this.manifest.hnsw.graph.path)

    this.entryPoint = this.manifest.hnsw.entryPoint
    this.maxLevel = this.manifest.hnsw.maxLevel
    this.efDefault = Math.max(64, this.manifest.hnsw.M * 4)

    this.levelGraph = this.manifest.hnsw.graph.levels.map((level) => {
      const indptr = new Uint32Array(graphBuffer, level.indptr.offset, level.indptr.elements)
      const indices = new Uint32Array(graphBuffer, level.indices.offset, level.indices.elements)
      return { indptr, indices }
    })
  }

  private vectorSlice(id: number): Float32Array {
    if (!this.vectorsView) {
      throw new Error("vector buffer not configured")
    }
    const start = id * this.dims
    const end = start + this.dims
    return this.vectorsView.subarray(start, end)
  }

  private dot(a: Float32Array, b: Float32Array): number {
    let s = 0
    for (let i = 0; i < this.dims; i++) {
      s += a[i] * b[i]
    }
    return s
  }

  private neighborsFor(level: number, node: number): Uint32Array {
    const meta = this.levelGraph[level]
    if (!meta) return new Uint32Array()
    const { indptr, indices } = meta
    if (node < 0 || node + 1 >= indptr.length) return new Uint32Array()
    const start = indptr[node]
    const end = indptr[node + 1]
    return indices.subarray(start, end)
  }

  private insertSortedDescending(arr: SearchHit[], item: SearchHit) {
    let idx = arr.length
    while (idx > 0 && arr[idx - 1].score < item.score) {
      idx -= 1
    }
    arr.splice(idx, 0, item)
  }

  private hnswSearch(query: Float32Array, k: number): SearchHit[] {
    if (!this.manifest || !this.vectorsView || this.entryPoint < 0 || this.levelGraph.length === 0) {
      throw new Error("semantic graph not initialised")
    }
    const ef = Math.max(this.efDefault, k * 10)
    let ep = this.entryPoint
    let epScore = this.dot(query, this.vectorSlice(ep))

    for (let level = this.maxLevel; level > 0; level--) {
      let changed = true
      while (changed) {
        changed = false
        const neigh = this.neighborsFor(level, ep)
        for (let i = 0; i < neigh.length; i++) {
          const candidate = neigh[i]
          if (candidate >= this.rows) continue
          const score = this.dot(query, this.vectorSlice(candidate))
          if (score > epScore) {
            epScore = score
            ep = candidate
            changed = true
          }
        }
      }
    }

    const visited = new Set<number>()
    const candidateQueue: SearchHit[] = []
    const best: SearchHit[] = []
    this.insertSortedDescending(candidateQueue, { id: ep, score: epScore })
    this.insertSortedDescending(best, { id: ep, score: epScore })
    visited.add(ep)

    while (candidateQueue.length > 0) {
      const current = candidateQueue.shift()!
      const worstBest = best.length >= ef ? best[best.length - 1].score : -Infinity
      if (current.score < worstBest && best.length >= ef) {
        break
      }
      const neigh = this.neighborsFor(0, current.id)
      for (let i = 0; i < neigh.length; i++) {
        const candidate = neigh[i]
        if (candidate >= this.rows || visited.has(candidate)) continue
        visited.add(candidate)
        const score = this.dot(query, this.vectorSlice(candidate))
        const hit = { id: candidate, score }
        this.insertSortedDescending(candidateQueue, hit)
        if (best.length < ef || score > best[best.length - 1].score) {
          this.insertSortedDescending(best, hit)
          if (best.length > ef) {
            best.pop()
          }
        }
      }
    }

    best.sort((a, b) => b.score - a.score)
    return best.slice(0, k)
  }

  private getParentSlug(slug: string): string {
    const meta = this.chunkMetadata[slug]
    return meta ? meta.parentSlug : slug
  }

  private aggregateChunkResults(results: SearchHit[]): Map<string, { rrfScore: number; maxScore: number }> {
    const docChunks = new Map<string, Array<{ score: number }>>()

    results.forEach(({ id, score }) => {
      const chunkSlug = this.manifestIds[id]
      if (!chunkSlug) return

      const parentSlug = this.getParentSlug(chunkSlug)

      if (!docChunks.has(parentSlug)) {
        docChunks.set(parentSlug, [])
      }

      docChunks.get(parentSlug)!.push({ score })
    })

    const aggregated = new Map<string, { rrfScore: number; maxScore: number }>()
    const RRF_K = 60

    for (const [parentSlug, chunks] of Array.from(docChunks)) {
      chunks.sort((a, b) => b.score - a.score)

      const rrfScore = chunks.reduce((sum, _, rank) => sum + 1.0 / (RRF_K + rank), 0)
      const maxScore = chunks[0].score

      aggregated.set(parentSlug, { rrfScore, maxScore })
    }

    return aggregated
  }

  async search(queryEmbedding: number[], k: number): Promise<Array<{ slug: string; score: number }>> {
    if (!this.vectorsView) {
      throw new Error("search engine not initialized")
    }

    const queryVec = new Float32Array(queryEmbedding)
    const chunkResults = this.hnswSearch(queryVec, k * 3)

    const aggregated = this.aggregateChunkResults(chunkResults)

    const results = Array.from(aggregated.entries())
      .map(([slug, { rrfScore, maxScore }]) => ({ slug, score: maxScore }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)

    return results
  }
}

let searchEngine: SemanticSearchEngine | null = null

export async function getSearchEngine(): Promise<SemanticSearchEngine> {
  if (!searchEngine) {
    searchEngine = new SemanticSearchEngine()
    await searchEngine.initialize()
  }
  return searchEngine
}

export async function semanticSearch(
  env: { AI: any },
  query: string,
  limit: number = 8,
): Promise<Array<{ slug: string; score: number }>> {
  const prefixedQuery = `task: search result | query: ${query}`
  const embedding = await env.AI.run("@cf/google/embeddinggemma-300m", {
    text: [prefixedQuery],
  })

  const embeddingData = embedding.data[0] as number[]

  const engine = await getSearchEngine()
  return await engine.search(embeddingData, limit)
}

