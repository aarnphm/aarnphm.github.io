/**
 * Projections for the 3-d bench of `layout: latent`. Holds one int8 matrix (document-level or
 * chunk-level, 256-d MRL) and projects every row onto three unit directions on request.
 */
export type LatentWorkerRequest =
  | { type: 'load'; seq: number; url: string; rows: number; dims: number }
  | { type: 'project'; seq: number; dirs: Float32Array; mean: Float32Array; scale: number }

export type LatentWorkerResponse =
  | { type: 'loaded'; seq: number; rows: number }
  | { type: 'projected'; seq: number; coords: Float32Array; extent: number }
  | { type: 'error'; seq: number; message: string }

let matrix: Int8Array | null = null
let rows = 0
let dims = 0

async function load(url: string, expectRows: number, expectDims: number): Promise<number> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  const bytes = new Int8Array(await response.arrayBuffer())
  if (bytes.length !== expectRows * expectDims) throw new Error('vector sidecar has the wrong size')
  matrix = bytes
  rows = expectRows
  dims = expectDims
  return rows
}

function project(
  dirs: Float32Array,
  mean: Float32Array,
  scale: number,
): { coords: Float32Array; extent: number } {
  if (!matrix) throw new Error('vectors not loaded')
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
      const { coords, extent } = project(request.dirs, request.mean, request.scale)
      ;(self as unknown as Worker).postMessage(
        { type: 'projected', seq: request.seq, coords, extent } satisfies LatentWorkerResponse,
        [coords.buffer],
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
