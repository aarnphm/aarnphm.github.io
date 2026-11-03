/**
 * Projections for the 3-d bench of `layout: latent`. Holds one int8 matrix (document-level or
 * chunk-level, 256-d MRL) and projects every row onto three unit directions on request.
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
  /** Top-k rows of the matrix at `url` by dot product with a unit query vector. */
  | { type: 'query'; seq: number; url: string; vector: Float32Array; scale: number; k: number }

export type LatentWorkerResponse =
  | { type: 'loaded'; seq: number; rows: number }
  | { type: 'projected'; seq: number; coords: Float32Array; extent: number }
  | { type: 'queried'; seq: number; hits: Uint32Array; scores: Float32Array }
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
): { hits: Uint32Array; scores: Float32Array } {
  const { matrix, rows, dims } = matrixAt(url)
  if (vector.length < dims) throw new Error('query vector is shorter than the matrix')
  const scores = new Float32Array(rows)
  const inv = scale / 127
  for (let r = 0; r < rows; r++) {
    const base = r * dims
    let s = 0
    for (let d = 0; d < dims; d++) s += matrix[base + d] * vector[d]
    scores[r] = s * inv
  }
  const n = Math.min(k, rows)
  const hits = new Uint32Array(n)
  const best = new Float32Array(n)
  // partial selection: n is small (16), rows are many
  for (let r = 0; r < rows; r++) {
    const v = scores[r]
    if (r >= n && v <= best[n - 1]) continue
    let i = Math.min(r, n - 1)
    while (i > 0 && best[i - 1] < v) {
      best[i] = best[i - 1]
      hits[i] = hits[i - 1]
      i--
    }
    best[i] = v
    hits[i] = r
  }
  return { hits, scores: best }
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
    } else if (request.type === 'query') {
      const { hits, scores } = query(request.url, request.vector, request.scale, request.k)
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
