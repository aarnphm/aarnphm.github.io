/**
 * The vector math of `layout: latent`, shared by the browser's vectors worker
 * (workers/latent.worker.ts) and the build (latent embeds, util/latent/embed/). One int8 matrix
 * per url (document-level or chunk-level, 256-d MRL): projections onto three unit directions,
 * top-k queries, and per-row channels for steering, which ranks rows by a weighted sum of
 * channels (docs/latent-steering.md: renormalising a steered vector never reorders a dot product,
 * so steering is score fusion and a slider drag is one pass over cached arrays). Keeping one copy
 * means an embed's build-time results are the explorer's results.
 */
export type LatentWorkerRequest =
  | { type: 'load'; seq: number; url: string; rows: number; dims: number }
  | {
      type: 'project'
      seq: number
      url: string
      dirs: Float32Array
      mean: Float32Array
      scale: number
    }
  /**
   * Top-k rows of the matrix at `url` by dot product with a unit query vector. `only` holds one
   * byte per row; rows with a zero never rank (the search box's `filter:`).
   */
  | {
      type: 'query'
      seq: number
      url: string
      vector: Float32Array
      scale: number
      k: number
      only?: Uint8Array
    }
  /** One row of the matrix as a unit float vector (the features tab steers from it). */
  | { type: 'row'; seq: number; url: string; row: number; scale: number }
  /**
   * A per-row channel kept under `key`: (row · vector + offset) × gain, clamped at zero when
   * `relu`. The features tab's base cosine, a feature's detector and a tag or axis projection are
   * all channels; steering fuses them.
   */
  | {
      type: 'channel'
      seq: number
      url: string
      key: string
      vector: Float32Array
      offset: number
      gain: number
      relu: boolean
      scale: number
    }
  /**
   * Top-k rows by Σ weight × channel (score fusion). `report` names the channel whose value is
   * returned per hit (the base cosine, so a bar still reads as similarity); `skip` rows never rank.
   */
  | {
      type: 'fuse'
      seq: number
      url: string
      terms: { key: string; weight: number }[]
      report: string
      skip: number[]
      k: number
      /** one byte per row, as for `query` */
      only?: Uint8Array
    }

export type LatentWorkerResponse =
  | { type: 'loaded'; seq: number; rows: number }
  | { type: 'projected'; seq: number; coords: Float32Array; extent: number }
  | { type: 'queried'; seq: number; hits: Uint32Array; scores: Float32Array }
  | { type: 'row'; seq: number; vector: Float32Array }
  | { type: 'channel'; seq: number; key: string }
  | { type: 'fused'; seq: number; hits: Uint32Array; scores: Float32Array; fused: Float32Array }
  | { type: 'error'; seq: number; message: string }

interface Matrix {
  matrix: Int8Array
  rows: number
  dims: number
}

/** channels per matrix url, least recently used first; each is rows × 4 bytes (228 KB at 57k) */
const CHANNELS = 48

export class VectorEngine {
  private matrices = new Map<string, Matrix>()
  private pending = new Map<string, Promise<number>>()
  private channels = new Map<string, Float32Array>()

  constructor(private readonly fetchBytes: (url: string) => Promise<ArrayBuffer>) {}

  load(url: string, expectRows: number, expectDims: number): Promise<number> {
    const known = this.matrices.get(url)
    if (known) return Promise.resolve(known.rows)
    let pending = this.pending.get(url)
    if (!pending) {
      pending = this.fetchBytes(url).then(buffer => {
        const bytes = new Int8Array(buffer)
        if (bytes.length !== expectRows * expectDims)
          throw new Error('vector sidecar has the wrong size')
        this.matrices.set(url, { matrix: bytes, rows: expectRows, dims: expectDims })
        return expectRows
      })
      pending.finally(() => this.pending.delete(url)).catch(() => {})
      this.pending.set(url, pending)
    }
    return pending
  }

  private matrixAt(url: string): Matrix {
    const m = this.matrices.get(url)
    if (!m) throw new Error(`vectors not loaded: ${url}`)
    return m
  }

  private channelKey(url: string, key: string): string {
    return `${url}\u0000${key}`
  }

  makeChannel(
    url: string,
    key: string,
    vector: Float32Array,
    offset: number,
    gain: number,
    relu: boolean,
    scale: number,
  ): void {
    const { matrix, rows, dims } = this.matrixAt(url)
    const out = new Float32Array(rows)
    const inv = scale / 127
    for (let r = 0; r < rows; r++) {
      const base = r * dims
      let s = 0
      for (let d = 0; d < dims; d++) s += matrix[base + d] * vector[d]
      const v = (s * inv + offset) * gain
      out[r] = relu && v < 0 ? 0 : v
    }
    const id = this.channelKey(url, key)
    this.channels.delete(id)
    this.channels.set(id, out)
    while (this.channels.size > CHANNELS) this.channels.delete(this.channels.keys().next().value!)
  }

  private channelAt(url: string, key: string): Float32Array {
    const id = this.channelKey(url, key)
    const c = this.channels.get(id)
    if (!c) throw new Error(`channel not computed: ${key}`)
    this.channels.delete(id)
    this.channels.set(id, c)
    return c
  }

  fuse(
    url: string,
    terms: { key: string; weight: number }[],
    report: string,
    skip: number[],
    k: number,
    only?: Uint8Array,
  ): { hits: Uint32Array; scores: Float32Array; fused: Float32Array } {
    const { rows } = this.matrixAt(url)
    const live = terms
      .filter(t => t.weight !== 0)
      .map(t => ({ c: this.channelAt(url, t.key), w: t.weight }))
    const shown = this.channelAt(url, report)
    const total = new Float32Array(rows)
    for (const { c, w } of live) for (let r = 0; r < rows; r++) total[r] += w * c[r]
    for (const r of skip) if (r >= 0 && r < rows) total[r] = -Infinity
    if (only) for (let r = 0; r < rows; r++) if (!only[r]) total[r] = -Infinity
    const n = Math.min(k, rows)
    let hits = new Uint32Array(n)
    let best = new Float32Array(n).fill(-Infinity)
    for (let r = 0; r < rows; r++) {
      const v = total[r]
      if (v <= best[n - 1]) continue
      let i = n - 1
      while (i > 0 && best[i - 1] < v) {
        best[i] = best[i - 1]
        hits[i] = hits[i - 1]
        i--
      }
      best[i] = v
      hits[i] = r
    }
    // a mask that keeps fewer rows than k leaves empty slots at the end
    const filled = best.findIndex(v => v === -Infinity)
    if (filled !== -1) {
      hits = hits.slice(0, filled)
      best = best.slice(0, filled)
    }
    const scores = new Float32Array(hits.length)
    for (let i = 0; i < hits.length; i++) scores[i] = shown[hits[i]]
    return { hits, scores, fused: best }
  }

  /** Rows are unit vectors before quantisation, so dot · scale / 127 is the cosine. */
  query(
    url: string,
    vector: Float32Array,
    scale: number,
    k: number,
    only?: Uint8Array,
  ): { hits: Uint32Array; scores: Float32Array } {
    const { matrix, rows, dims } = this.matrixAt(url)
    if (vector.length < dims) throw new Error('query vector is shorter than the matrix')
    const inv = scale / 127
    const n = Math.min(k, rows)
    const hits = new Uint32Array(n)
    const best = new Float32Array(n).fill(-Infinity)
    let filled = 0
    // partial selection: n is small (16), rows are many; masked rows skip the dot product
    for (let r = 0; r < rows; r++) {
      if (only && !only[r]) continue
      const base = r * dims
      let s = 0
      for (let d = 0; d < dims; d++) s += matrix[base + d] * vector[d]
      const v = s * inv
      if (filled === n && v <= best[n - 1]) continue
      let i = Math.min(filled, n - 1)
      while (i > 0 && best[i - 1] < v) {
        best[i] = best[i - 1]
        hits[i] = hits[i - 1]
        i--
      }
      best[i] = v
      hits[i] = r
      if (filled < n) filled++
    }
    return filled === n
      ? { hits, scores: best }
      : { hits: hits.slice(0, filled), scores: best.slice(0, filled) }
  }

  project(
    url: string,
    dirs: Float32Array,
    mean: Float32Array,
    scale: number,
  ): { coords: Float32Array; extent: number } {
    return projectMatrix(this.matrixAt(url), dirs, mean, scale)
  }

  row(url: string, row: number, scale: number): Float32Array {
    const { matrix, rows, dims } = this.matrixAt(url)
    if (row < 0 || row >= rows) throw new Error(`row ${row} out of range`)
    const vector = new Float32Array(dims)
    let length = 0
    for (let d = 0; d < dims; d++) {
      vector[d] = (matrix[row * dims + d] * scale) / 127
      length += vector[d] * vector[d]
    }
    length = Math.sqrt(length) || 1
    for (let d = 0; d < dims; d++) vector[d] /= length
    return vector
  }

  /** One request, answered as the worker answers it (with the buffers to transfer). */
  async handle(
    request: LatentWorkerRequest,
  ): Promise<{ response: LatentWorkerResponse; transfer: Transferable[] }> {
    const seq = request.seq
    try {
      switch (request.type) {
        case 'load': {
          const rows = await this.load(request.url, request.rows, request.dims)
          return { response: { type: 'loaded', seq, rows }, transfer: [] }
        }
        case 'project': {
          const { coords, extent } = this.project(
            request.url,
            request.dirs,
            request.mean,
            request.scale,
          )
          return { response: { type: 'projected', seq, coords, extent }, transfer: [coords.buffer] }
        }
        case 'row': {
          const vector = this.row(request.url, request.row, request.scale)
          return { response: { type: 'row', seq, vector }, transfer: [vector.buffer] }
        }
        case 'channel': {
          const { url, key, vector, offset, gain, relu, scale } = request
          this.makeChannel(url, key, vector, offset, gain, relu, scale)
          return { response: { type: 'channel', seq, key }, transfer: [] }
        }
        case 'fuse': {
          const { hits, scores, fused } = this.fuse(
            request.url,
            request.terms,
            request.report,
            request.skip,
            request.k,
            request.only,
          )
          return {
            response: { type: 'fused', seq, hits, scores, fused },
            transfer: [hits.buffer, scores.buffer, fused.buffer],
          }
        }
        case 'query': {
          const { hits, scores } = this.query(
            request.url,
            request.vector,
            request.scale,
            request.k,
            request.only,
          )
          return {
            response: { type: 'queried', seq, hits, scores },
            transfer: [hits.buffer, scores.buffer],
          }
        }
      }
    } catch (error) {
      return {
        response: {
          type: 'error',
          seq,
          message: error instanceof Error ? error.message : String(error),
        },
        transfer: [],
      }
    }
  }
}

/** Centred projection of every row onto three unit directions: (v − mean) · d per axis. */
export function projectMatrix(
  { matrix, rows, dims }: Matrix,
  dirs: Float32Array,
  mean: Float32Array,
  scale: number,
): { coords: Float32Array; extent: number } {
  const coords = new Float32Array(rows * 3)
  // centre: (v - mean) · d = v · d - mean · d
  const offsets = [0, 1, 2].map(a => {
    let s = 0
    for (let d = 0; d < dims; d++) s += mean[d] * dirs[a * dims + d]
    return s
  })
  const inv = scale / 127
  let extent = 0
  for (let r = 0; r < rows; r++) {
    const base = r * dims
    for (let a = 0; a < 3; a++) {
      let s = 0
      const off = a * dims
      for (let d = 0; d < dims; d++) s += matrix[base + d] * dirs[off + d]
      const v = s * inv - offsets[a]
      coords[r * 3 + a] = v
      const m = Math.abs(v)
      if (m > extent) extent = m
    }
  }
  return { coords, extent }
}
