/**
 * Projections for the 3-d bench of `layout: latent`. Holds one int8 matrix (document-level or
 * chunk-level, 256-d MRL) and projects every row onto three unit directions on request. It also
 * keeps per-row channels for the features tab's steering, which ranks rows by a weighted sum of
 * channels (docs/latent-steering.md: renormalising a steered vector never reorders a dot product,
 * so steering is score fusion and a slider drag is one pass over cached arrays).
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
   * A per-row channel kept in the worker under `key`: (row · vector + offset) × gain, clamped at
   * zero when `relu`. The features tab's base cosine, a feature's detector and a tag or axis
   * projection are all channels; steering fuses them.
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

const matrices = new Map<string, Matrix>()

async function load(url: string, expectRows: number, expectDims: number): Promise<number> {
  const known = matrices.get(url)
  if (known) return known.rows
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  const bytes = new Int8Array(await response.arrayBuffer())
  if (bytes.length !== expectRows * expectDims) throw new Error('vector sidecar has the wrong size')
  matrices.set(url, { matrix: bytes, rows: expectRows, dims: expectDims })
  return expectRows
}

/** channels per matrix url, least recently used first; each is rows × 4 bytes (228 KB at 57k) */
const channels = new Map<string, Float32Array>()
const CHANNELS = 48

function channelKey(url: string, key: string): string {
  return `${url}\u0000${key}`
}

function makeChannel(
  url: string,
  key: string,
  vector: Float32Array,
  offset: number,
  gain: number,
  relu: boolean,
  scale: number,
): void {
  const { matrix, rows, dims } = matrixAt(url)
  const out = new Float32Array(rows)
  const inv = scale / 127
  for (let r = 0; r < rows; r++) {
    const base = r * dims
    let s = 0
    for (let d = 0; d < dims; d++) s += matrix[base + d] * vector[d]
    const v = (s * inv + offset) * gain
    out[r] = relu && v < 0 ? 0 : v
  }
  const id = channelKey(url, key)
  channels.delete(id)
  channels.set(id, out)
  while (channels.size > CHANNELS) channels.delete(channels.keys().next().value!)
}

function channelAt(url: string, key: string): Float32Array {
  const id = channelKey(url, key)
  const c = channels.get(id)
  if (!c) throw new Error(`channel not computed: ${key}`)
  channels.delete(id)
  channels.set(id, c)
  return c
}

function fuse(
  url: string,
  terms: { key: string; weight: number }[],
  report: string,
  skip: number[],
  k: number,
  only?: Uint8Array,
): { hits: Uint32Array; scores: Float32Array; fused: Float32Array } {
  const { rows } = matrixAt(url)
  const live = terms
    .filter(t => t.weight !== 0)
    .map(t => ({ c: channelAt(url, t.key), w: t.weight }))
  const shown = channelAt(url, report)
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

function matrixAt(url: string): Matrix {
  const m = matrices.get(url)
  if (!m) throw new Error(`vectors not loaded: ${url}`)
  return m
}

/** Rows are unit vectors before quantisation, so dot · scale / 127 is the cosine. */
function query(
  url: string,
  vector: Float32Array,
  scale: number,
  k: number,
  only?: Uint8Array,
): { hits: Uint32Array; scores: Float32Array } {
  const { matrix, rows, dims } = matrixAt(url)
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

function project(
  url: string,
  dirs: Float32Array,
  mean: Float32Array,
  scale: number,
): { coords: Float32Array; extent: number } {
  const { matrix, rows, dims } = matrixAt(url)
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

self.onmessage = async (event: MessageEvent<LatentWorkerRequest>) => {
  const request = event.data
  try {
    if (request.type === 'load') {
      const n = await load(request.url, request.rows, request.dims)
      self.postMessage({ type: 'loaded', seq: request.seq, rows: n } satisfies LatentWorkerResponse)
    } else if (request.type === 'project') {
      const { coords, extent } = project(request.url, request.dirs, request.mean, request.scale)
      ;(self as unknown as Worker).postMessage(
        { type: 'projected', seq: request.seq, coords, extent } satisfies LatentWorkerResponse,
        [coords.buffer],
      )
    } else if (request.type === 'row') {
      const { matrix, rows, dims } = matrixAt(request.url)
      if (request.row < 0 || request.row >= rows) throw new Error(`row ${request.row} out of range`)
      const vector = new Float32Array(dims)
      let length = 0
      for (let d = 0; d < dims; d++) {
        vector[d] = (matrix[request.row * dims + d] * request.scale) / 127
        length += vector[d] * vector[d]
      }
      length = Math.sqrt(length) || 1
      for (let d = 0; d < dims; d++) vector[d] /= length
      ;(self as unknown as Worker).postMessage(
        { type: 'row', seq: request.seq, vector } satisfies LatentWorkerResponse,
        [vector.buffer],
      )
    } else if (request.type === 'channel') {
      const { url, key, vector, offset, gain, relu, scale } = request
      makeChannel(url, key, vector, offset, gain, relu, scale)
      self.postMessage({ type: 'channel', seq: request.seq, key } satisfies LatentWorkerResponse)
    } else if (request.type === 'fuse') {
      const { hits, scores, fused } = fuse(
        request.url,
        request.terms,
        request.report,
        request.skip,
        request.k,
        request.only,
      )
      ;(self as unknown as Worker).postMessage(
        { type: 'fused', seq: request.seq, hits, scores, fused } satisfies LatentWorkerResponse,
        [hits.buffer, scores.buffer, fused.buffer],
      )
    } else if (request.type === 'query') {
      const { hits, scores } = query(
        request.url,
        request.vector,
        request.scale,
        request.k,
        request.only,
      )
      ;(self as unknown as Worker).postMessage(
        { type: 'queried', seq: request.seq, hits, scores } satisfies LatentWorkerResponse,
        [hits.buffer, scores.buffer],
      )
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      seq: request.seq,
      message: error instanceof Error ? error.message : String(error),
    } satisfies LatentWorkerResponse)
  }
}
