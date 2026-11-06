import type { Bench, Runtime } from './runtime'
import { atlasFrom, type Atlas } from '../../../util/latent/atlas'
import { benchFeatures, type FeatureDict } from './features'

export { LEVEL_RADIUS, projectAtlas, type Atlas } from '../../../util/latent/atlas'

/**
 * The bench's features level and a feature's heat over the other two. At the features level
 * every SAE feature with a firing row is one point, where its decoder direction lands on the
 * chosen x, y and z: the vector a single firing at its maximum reconstructs, mean + norm · max ·
 * dec_f, centred and projected as the worker projects the rows. Coarse Matryoshka features draw
 * large, fine ones small, so the level reads through size. A picked feature carries over to
 * documents and chunks as heat: the rows it fires lit by activation, the rest a faint ink floor.
 * Nothing here loads until the features level is opened; the codes load on the first pick, once
 * per dictionary, into an inverted index.
 */

/** a dictionary's codes turned around: the rows of feature f are rows[start[f]..start[f + 1]] */
type Csr = { start: Int32Array; rows: Uint32Array; acts: Uint8Array }

const building = new WeakMap<Bench, Promise<Atlas | null>>()

export function loadAtlas(rt: Runtime, bench: Bench): Promise<Atlas | null> {
  if (bench.atlas) return Promise.resolve(bench.atlas)
  let pending = building.get(bench)
  if (pending) return pending
  pending = buildAtlas(rt, bench).then(atlas => (bench.atlas = atlas))
  pending.catch(() => building.delete(bench))
  building.set(bench, pending)
  return pending
}

async function buildAtlas(rt: Runtime, bench: Bench): Promise<Atlas | null> {
  const api = benchFeatures(rt)
  const meta = bench.meta
  if (!api || !meta) return null
  const D = meta.mean.length
  const dicts = (await Promise.all(api.names.map(name => api.dict(name)))).filter(
    dict => dict.meta.dims === D,
  )
  const decs = await Promise.all(dicts.map(dict => api.dec(dict)))
  return atlasFrom(dicts, decs, meta.mean)
}

/** The atlas point of the bench's feature, or -1. */
export function atlasPoint(bench: Bench): number {
  if (!bench.atlas || !bench.feature) return -1
  return bench.atlas.at.get(`${bench.feature.dict}:${bench.feature.f}`) ?? -1
}

function dictOf(atlas: Atlas, name: string): FeatureDict | undefined {
  return atlas.dicts.find(dict => dict.meta.name === name)
}

/** The bench's feature as its table row, when it still names one (a rebuilt store can drop it). */
export function featureRow(bench: Bench) {
  const feature = bench.feature
  const dict = feature && bench.atlas ? dictOf(bench.atlas, feature.dict) : undefined
  return dict && feature && feature.f < dict.table.length
    ? { dict, row: dict.table[feature.f] }
    : null
}

/** The status line for an atlas point. */
export function atlasName(rt: Runtime, bench: Bench, i: number): string {
  const atlas = bench.atlas
  const api = benchFeatures(rt)
  if (!atlas || !api || i < 0 || i >= atlas.f.length) return ''
  const dict = atlas.dicts[atlas.dict[i]]
  const row = dict.table[atlas.f[i]]
  const finer = dict.children.get(row.f)?.length ?? 0
  return [
    dict.meta.name,
    api.name(row),
    `L${atlas.lvl[i]}`,
    `${row.n.toLocaleString()} rows`,
    row.near.length ? `${row.near.length} near` : null,
    finer ? `${finer} finer` : null,
  ]
    .filter(Boolean)
    .join(' · ')
}

type Screen = { xs: Float32Array; ys: Float32Array; depth: Float32Array }

/**
 * An atlas point's edges, drawn over the points like a row's: its nearest features (by decoder
 * cosine) in grey, the coarser feature it sits in solid ink, its finer children dashed.
 */
export function drawAtlasEdges(
  ctx: CanvasRenderingContext2D,
  bench: Bench,
  at: Screen,
  shown: (i: number) => boolean,
  point: number,
  colours: { ink: string; near: string },
): void {
  const atlas = bench.atlas
  if (!atlas || point < 0 || point >= atlas.f.length || !shown(point)) return
  const dict = atlas.dicts[atlas.dict[point]]
  const name = dict.meta.name
  const row = dict.table[atlas.f[point]]
  const index = (g: number) => atlas.at.get(`${name}:${g}`) ?? -1
  const deep = (b: number) => (at.depth[point] + at.depth[b] > 0 ? 1 : 0)
  const stroke = (targets: number[], colour: string, width: number, dash: number[]) => {
    const paths = [new Path2D(), new Path2D()]
    for (const b of targets) {
      if (b < 0 || !shown(b)) continue
      paths[deep(b)].moveTo(at.xs[point], at.ys[point])
      paths[deep(b)].lineTo(at.xs[b], at.ys[b])
    }
    ctx.strokeStyle = colour
    ctx.lineWidth = width
    ctx.setLineDash(dash)
    ctx.globalAlpha = 0.4
    ctx.stroke(paths[0])
    ctx.globalAlpha = 0.85
    ctx.stroke(paths[1])
  }
  ctx.save()
  stroke(
    row.near.map(([g]) => index(g)),
    colours.near,
    0.75,
    [],
  )
  if (row.parent !== undefined && row.parent >= 0) stroke([index(row.parent)], colours.ink, 1, [])
  stroke((dict.children.get(row.f) ?? []).slice(0, 8).map(index), colours.ink, 1, [3, 2])
  ctx.restore()
}

const indexes = new WeakMap<Bench, Map<string, Promise<Csr>>>()

/** Every code block of the dictionary, eight requests at a time, turned around per feature. */
function csrOf(rt: Runtime, bench: Bench, dict: FeatureDict): Promise<Csr> {
  let map = indexes.get(bench)
  if (!map) indexes.set(bench, (map = new Map()))
  let csr = map.get(dict.meta.name)
  if (csr) return csr
  const api = benchFeatures(rt)!
  csr = (async () => {
    const { rows, codeBlock: size, k, features: M } = dict.meta
    const count = Math.ceil(rows / size)
    const blocks: { ids: Uint16Array; acts: Uint8Array }[] = Array.from({ length: count })
    for (let b = 0; b < count; b += 8)
      await Promise.all(
        Array.from({ length: Math.min(8, count - b) }, async (_, j) => {
          blocks[b + j] = await api.block(dict, b + j)
        }),
      )
    const start = new Int32Array(M + 1)
    for (const { ids, acts } of blocks)
      for (let j = 0; j < acts.length; j++) if (acts[j] > 0 && ids[j] < M) start[ids[j] + 1]++
    for (let f = 0; f < M; f++) start[f + 1] += start[f]
    const out = new Uint32Array(start[M])
    const act = new Uint8Array(start[M])
    const fill = start.slice(0, M)
    blocks.forEach(({ ids, acts }, b) => {
      for (let j = 0; j < acts.length; j++) {
        if (!(acts[j] > 0) || ids[j] >= M) continue
        const slot = fill[ids[j]]++
        out[slot] = b * size + Math.floor(j / k)
        act[slot] = acts[j]
      }
    })
    return { start, rows: out, acts: act }
  })()
  csr.catch(() => map.delete(dict.meta.name))
  map.set(dict.meta.name, csr)
  return csr
}

/**
 * Bring `bench.heat` in line with the bench's feature and level: one activation byte per point
 * (a document takes its strongest row), null at the features level, with no feature, or while
 * the codes load. `say` gets the status line once the heat lands, or why it did not.
 */
export async function syncHeat(
  rt: Runtime,
  bench: Bench,
  redraw: () => void,
  say: (text: string) => void,
): Promise<void> {
  const picked = bench.level === 'feats' ? null : featureRow(bench)
  const key = picked ? `${bench.level}:${picked.dict.meta.name}:${picked.row.f}` : ''
  if (key === bench.heatKey) return
  bench.heatKey = key
  bench.heat = null
  redraw()
  if (!picked) return
  const { dict, row } = picked
  const api = benchFeatures(rt)!
  say(`loading ${dict.meta.name} codes…`)
  let csr: Csr
  try {
    csr = await csrOf(rt, bench, dict)
  } catch (error) {
    if (bench.heatKey === key) say('codes unavailable')
    console.warn('[latent] bench codes', error)
    return
  }
  if (bench.heatKey !== key) return
  const docs = bench.level === 'docs'
  const heat = new Uint8Array(docs ? rt.docs.length : (bench.meta?.rows ?? 0))
  for (let s = csr.start[row.f]; s < csr.start[row.f + 1]; s++) {
    const r = csr.rows[s]
    const p = docs ? rt.docByRow[r]?.d : r
    if (p === undefined || p >= heat.length) continue
    if (csr.acts[s] > heat[p]) heat[p] = csr.acts[s]
  }
  bench.heat = heat
  redraw()
  say(`# ${api.name(row)} · ${row.n.toLocaleString()} rows · ${row.docs.toLocaleString()} docs`)
}
