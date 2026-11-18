import { fetchBytes, fetchJson, type Runtime } from '../runtime'

interface BakeMeta {
  layers: number
  width: number
  top: number
  attnLayers: number[]
  attnTop: number
  rows: Record<string, [number, boolean]>
}

export interface BakeRow {
  T: number
  L: number
  K: number
  A: number
  KA: number
  spans: Uint32Array
  ids: Uint16Array
  acts: Float32Array
  src: Uint16Array
  w: Float32Array
}

export interface Label {
  label: string
  score?: number
}

export interface Bake {
  base: string
  meta: BakeMeta
  labels: Promise<Record<string, Record<string, Label | string>>>
  rows: Map<number, Promise<BakeRow>>
  index: Map<
    number,
    Promise<{ starts: Uint32Array; r: Uint32Array; pos: Uint16Array; a: Float32Array }>
  >
  gloss: Map<number, Promise<Record<string, string>>>
}

export const PORT = 8093

// ---- f16 ------------------------------------------------------------------------------------

let F16: Float32Array | null = null

function f16(bits: Uint16Array): Float32Array {
  if (!F16) {
    F16 = new Float32Array(65536)
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1
      const e = (h >> 10) & 0x1f
      const m = h & 0x3ff
      F16[h] =
        e === 0
          ? s * m * 2 ** -24
          : e === 31
            ? m
              ? NaN
              : s * Infinity
            : s * (1 + m / 1024) * 2 ** (e - 15)
    }
  }
  const out = new Float32Array(bits.length)
  for (let i = 0; i < bits.length; i++) out[i] = F16[bits[i]]
  return out
}

function decodeRow(buf: ArrayBuffer): BakeRow {
  const dv = new DataView(buf)
  if (dv.getUint32(0, false) !== 0x504c5442) throw new Error('not a bake row')
  const [T, L, K, A, KA] = [8, 12, 16, 20, 24].map(o => dv.getUint32(o, true))
  let o = 28
  const take = <A extends Uint16Array | Uint32Array>(
    Ctor: { new (b: ArrayBuffer, o: number, n: number): A; BYTES_PER_ELEMENT: number },
    n: number,
  ): A => {
    const view = new Ctor(buf.slice(o, o + n * Ctor.BYTES_PER_ELEMENT), 0, n)
    o += n * Ctor.BYTES_PER_ELEMENT
    return view
  }
  const spans = take(Uint32Array, T * 2)
  const ids = take(Uint16Array, T * L * K)
  const acts = f16(take(Uint16Array, T * L * K))
  const src = take(Uint16Array, T * A * KA)
  const w = f16(take(Uint16Array, T * A * KA))
  return { T, L, K, A, KA, spans, ids, acts, src, w }
}

// ---- bake -----------------------------------------------------------------------------------

const BAKES = new WeakMap<Runtime, Promise<Bake | null>>()

/** the page's bake, opened once and shared by the selection and circuit tabs */
export function bakeOf(rt: Runtime): Promise<Bake | null> {
  let got = BAKES.get(rt)
  if (!got) {
    got = openBake(rt)
    BAKES.set(rt, got)
  }
  return got
}

async function openBake(rt: Runtime): Promise<Bake | null> {
  // the bake belongs to the /embeddings page; an embed carries no hash, and its build-time
  // render has no root at all
  if (rt.mode === 'embed') return null
  const h8 = (rt.root.dataset.latentHash ?? '').slice(0, 8)
  if (!h8) return null
  const local = ['localhost', '127.0.0.1'].includes(window.location.hostname)
  const bases = [
    ...(local ? [`http://localhost:${PORT}/bake/${h8}`] : []),
    `/models/latent-plt/bake/${h8}`,
  ]
  for (const base of bases) {
    try {
      const meta = await fetchJson<BakeMeta>(rt, `${base}/meta.json`)
      const labels = fetchJson<Record<string, Record<string, Label | string>>>(
        rt,
        `${base}/labels.json`,
      ).catch(() => ({}))
      return { base, meta, labels, rows: new Map(), index: new Map(), gloss: new Map() }
    } catch {
      // next base
    }
  }
  return null
}

export function rowOf(rt: Runtime, bake: Bake, row: number): Promise<BakeRow> {
  let got = bake.rows.get(row)
  if (!got) {
    got = fetchBytes(rt, `${bake.base}/rows/${row}.bin`).then(decodeRow)
    got.catch(() => bake.rows.delete(row))
    bake.rows.set(row, got)
  }
  return got
}

export function indexOf(rt: Runtime, bake: Bake, layer: number) {
  let got = bake.index.get(layer)
  if (!got) {
    const M = bake.meta.width
    got = fetchBytes(rt, `${bake.base}/index/L${String(layer).padStart(2, '0')}.bin`).then(buf => {
      const counts = new Uint32Array(buf, 0, M)
      const starts = new Uint32Array(M + 1)
      for (let f = 0; f < M; f++) starts[f + 1] = starts[f] + counts[f]
      const n = starts[M]
      let o = 4 * M
      const r = new Uint32Array(buf.slice(o, o + 4 * n))
      o += 4 * n
      const pos = new Uint16Array(buf.slice(o, o + 2 * n))
      o += 2 * n
      const a = f16(new Uint16Array(buf.slice(o, o + 2 * n)))
      return { starts, r, pos, a }
    })
    got.catch(() => bake.index.delete(layer))
    bake.index.set(layer, got)
  }
  return got
}

/** a feature's strongest distinct tokens over the bake, for features with no label */
export function glossOf(rt: Runtime, bake: Bake, layer: number): Promise<Record<string, string>> {
  let got = bake.gloss.get(layer)
  if (!got) {
    got = fetchJson<Record<string, string>>(
      rt,
      `${bake.base}/gloss/L${String(layer).padStart(2, '0')}.json`,
    ).catch(() => ({}))
    bake.gloss.set(layer, got)
  }
  return got
}

export function labelOf(
  labels: Record<string, Record<string, Label | string>>,
  l: number,
  f: number,
): Label | null {
  const v = labels[`L${String(l).padStart(2, '0')}`]?.[String(f)]
  if (!v) return null
  return typeof v === 'string' ? { label: v } : v
}
