import { type LatentKind } from '../../../util/latent/types'
import {
  type FeatureState,
  type FeatureDict,
  type FeatureDictMeta,
  type FeatureRow,
  type CodeBlock,
  type TagProbes,
  type SteerTables,
  type WordTable,
} from './features-types'
import { type Runtime, fetchJson, fetchBytes } from './runtime'

/** units when feat/steer.json is missing: the medians of the first calibration */
export const DEFAULT_UNIT = { feature: 0.45, tag: 0.6, axis: 1 }

/** the slider spans ±RANGE units, or the term's own strength when ± stepped past it (up to MAX) */
export const RANGE = 4

export const MAX = 12

export const KIND_ORDER: LatentKind[] = ['note', 'pdf', 'image', 'base']

export function loadDict(rt: Runtime, st: FeatureState, name: string): Promise<FeatureDict> {
  let dict = st.dicts.get(name)
  if (!dict) {
    const base = `${rt.base}/feat/${name}`
    dict = Promise.all([
      fetchJson<FeatureDictMeta>(rt, `${base}/meta.json`),
      fetchJson<FeatureRow[]>(rt, `${base}/features.json`),
    ]).then(([meta, table]) => {
      const children = new Map<number, number[]>()
      for (const row of table) {
        if (row.parent === undefined || row.parent < 0) continue
        const list = children.get(row.parent) ?? []
        list.push(row.f)
        children.set(row.parent, list)
      }
      for (const list of children.values()) list.sort((a, b) => table[b].n - table[a].n)
      return { meta, table, children, blocks: new Map(), enc: null, dec: null }
    })
    dict.catch(() => st.dicts.delete(name))
    st.dicts.set(name, dict)
  }
  return dict
}

export async function weights(
  rt: Runtime,
  dict: FeatureDict,
  which: 'enc' | 'dec',
): Promise<Int8Array> {
  const have = dict[which]
  if (have) return have
  const bytes = new Int8Array(
    await fetchBytes(rt, `${rt.base}/feat/${dict.meta.name}/${which}-int8.bin`),
  )
  dict[which] = bytes
  return bytes
}

/** codes/NNNNN.bin: a block of rows, ids (u16) then activations (u8), fetched on first use. */
export function codeBlock(rt: Runtime, dict: FeatureDict, b: number): Promise<CodeBlock> {
  let block = dict.blocks.get(b)
  if (!block) {
    const { k, rows, codeBlock: size } = dict.meta
    const n = Math.min(size, rows - b * size) * k
    block = fetchBytes(
      rt,
      `${rt.base}/feat/${dict.meta.name}/codes/${String(b).padStart(5, '0')}.bin`,
    ).then(bytes => ({ ids: new Uint16Array(bytes, 0, n), acts: new Uint8Array(bytes, n * 2, n) }))
    block.catch(() => dict.blocks.delete(b))
    dict.blocks.set(b, block)
  }
  return block
}

/** The features the row fires in one dictionary, strongest first: [feature, activation 0..1]. */
export async function codesOf(
  rt: Runtime,
  dict: FeatureDict,
  row: number,
): Promise<[number, number][]> {
  const { k, codeBlock: size } = dict.meta
  const block = await codeBlock(rt, dict, Math.floor(row / size))
  const at = (row % size) * k
  const out: [number, number][] = []
  for (let j = 0; j < k; j++) {
    const a = block.acts[at + j]
    if (a > 0) out.push([block.ids[at + j], a / 255])
  }
  return out
}

export function tagsOf(
  rt: Runtime,
  st: FeatureState,
): Promise<Map<number, [string, number][]> | null> {
  st.tags ??= fetchJson<TagProbes>(rt, `${rt.base}/feat/tags.json`)
    .then(
      probes =>
        new Map(
          probes.pred.map(([d, got]) => [
            d,
            got.map(([j, p]) => [probes.tags[j].t.toLowerCase(), p] as [string, number]),
          ]),
        ),
    )
    .catch(() => null)
  return st.tags
}

export function steerTables(rt: Runtime, st: FeatureState): Promise<SteerTables | null> {
  st.tables ??= fetchJson<SteerTables>(rt, `${rt.base}/feat/steer.json`)
    .then(t => (st.tablesNow = t))
    .catch(() => null)
  return st.tables
}

export function wordsOf(rt: Runtime, st: FeatureState): Promise<WordTable | null> {
  st.words ??= fetchJson<WordTable>(rt, `${rt.base}/feat/text/words.json`).catch(() => null)
  return st.words
}

/** Encode a unit 256-d vector into the dictionary's features, as the build did: top-k of z. */
export async function encodeVector(
  rt: Runtime,
  dict: FeatureDict,
  vector: Float32Array,
): Promise<[number, number][]> {
  const enc = await weights(rt, dict, 'enc')
  const { features: M, dims: D, k, mean, norm, bias, encScale } = dict.meta
  const x = new Float32Array(D)
  for (let d = 0; d < D; d++) x[d] = (vector[d] - mean[d]) / norm
  const z: [number, number][] = []
  for (let f = 0; f < M; f++) {
    let s = 0
    const off = f * D
    for (let d = 0; d < D; d++) s += enc[off + d] * x[d]
    const v = s * encScale[f] + bias[f]
    if (v > 0) z.push([f, v])
  }
  z.sort((a, b) => b[1] - a[1])
  return z.slice(0, k).map(([f, v]) => [f, Math.min(1, v / (dict.table[f].max || 1))])
}
