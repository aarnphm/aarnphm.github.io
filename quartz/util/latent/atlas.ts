import type { FeatureDict } from '../../components/scripts/latent-client/features-types'

/**
 * The bench's features level: every SAE feature with a firing row is one point, where its
 * decoder direction lands on the chosen x, y and z: the vector a single firing at its maximum
 * reconstructs, mean + norm · max · dec_f, centred and projected as the worker projects the
 * rows. Pure, so the explorer (latent-client/bench-feats.ts) and the build (latent embeds'
 * posters, util/latent/embed/) build the same atlas.
 */
export interface Atlas {
  dicts: FeatureDict[]
  /** per point: index into dicts, feature id, Matryoshka level, source kind index */
  dict: Uint8Array
  f: Uint16Array
  lvl: Uint8Array
  kind: Uint8Array
  /** point index by `${dict name}:${feature}` */
  at: Map<string, number>
  /** per point, the reconstruction minus the bench mean: n × dims */
  vec: Float32Array
  dims: number
}

const KIND_INDEX: Record<string, number> = { note: 0, pdf: 1, image: 2, base: 3 }
/** point radius by Matryoshka level, before the depth term */
export const LEVEL_RADIUS = [3.4, 2.2, 1.4]

function levelOf(dict: FeatureDict, f: number): number {
  const level = dict.table[f].level
  if (level !== undefined) return Math.min(2, level)
  const prefixes = dict.meta.prefixes ?? []
  const i = prefixes.findIndex(p => f < p)
  return Math.min(2, i < 0 ? prefixes.length : i)
}

/** The source a feature belongs to: its dictionary's one kind, or for text notes unless under a
 * tenth of its firing rows are notes (salmon stays pdf-only). */
function kindOf(dict: FeatureDict, f: number): number {
  const kinds = dict.meta.kinds
  if (kinds.length === 1) return KIND_INDEX[kinds[0]] ?? 0
  const counts = dict.table[f].kinds
  const own = kinds.map(kind => counts[KIND_INDEX[kind]] ?? 0)
  const total = own.reduce((a, b) => a + b, 0)
  const note = kinds.indexOf('note')
  if (note >= 0 && own[note] >= total * 0.1) return 0
  let best = 0
  for (let i = 1; i < own.length; i++) if (own[i] > own[best]) best = i
  return KIND_INDEX[kinds[best]] ?? 0
}

/** The atlas of `dicts` (same dims as the bench) with their int8 decoders, around `mean`. */
export function atlasFrom(dicts: FeatureDict[], decs: Int8Array[], mean0: number[]): Atlas {
  const D = mean0.length
  let n = 0
  for (const dict of dicts) for (const row of dict.table) if (row.n > 0) n++
  const atlas: Atlas = {
    dicts,
    dict: new Uint8Array(n),
    f: new Uint16Array(n),
    lvl: new Uint8Array(n),
    kind: new Uint8Array(n),
    at: new Map(),
    vec: new Float32Array(n * D),
    dims: D,
  }
  let i = 0
  dicts.forEach((dict, di) => {
    const dec = decs[di]
    const { mean, norm, decScale } = dict.meta
    for (const row of dict.table) {
      if (!(row.n > 0)) continue
      const f = row.f
      atlas.dict[i] = di
      atlas.f[i] = f
      atlas.lvl[i] = levelOf(dict, f)
      atlas.kind[i] = kindOf(dict, f)
      atlas.at.set(`${dict.meta.name}:${f}`, i)
      const gain = norm * (row.max || 1) * decScale[f]
      const off = i * D
      for (let d = 0; d < D; d++) atlas.vec[off + d] = mean[d] - mean0[d] + gain * dec[f * D + d]
      i++
    }
  })
  return atlas
}

/** The atlas on the bench's three axes: the same centred projection the worker gives the rows. */
export function projectAtlas(
  atlas: Atlas,
  dirs: Float32Array,
): { coords: Float32Array; extent: number } {
  const D = atlas.dims
  const n = atlas.f.length
  const coords = new Float32Array(n * 3)
  const reach = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const off = i * D
    for (let a = 0; a < 3; a++) {
      let s = 0
      const axis = a * D
      for (let d = 0; d < D; d++) s += atlas.vec[off + d] * dirs[axis + d]
      coords[i * 3 + a] = s
      if (Math.abs(s) > reach[i]) reach[i] = Math.abs(s)
    }
  }
  // a few features with large decoder norms sit far out; framing on them crushes the rest into
  // the middle, so the frame takes the 95th percentile and lets those few run off the edge
  const sorted = reach.slice().sort()
  const extent = n ? sorted[Math.min(n - 1, Math.floor(n * 0.95))] * 1.1 : 1
  return { coords, extent }
}
