import { bakeOf, glossOf, labelOf, ll } from './gemma'
import { fetchJson, type Runtime } from './runtime'

// Bridges between the two dictionaries (quartz/scripts/latent/plt_bridges.py): per passage-SAE
// text feature, the Gemma transcoder features whose activity in a passage predicts it, from a
// sparse logistic probe scored on held-out documents.

export interface Bridges {
  rows: number
  docs: number
  /** text feature -> fit rows, held-out auc, [layer, gemma id, weight, phi][] strongest first */
  features: Record<string, { n: number; auc: number | null; b: [number, number, number, number][] }>
  /** "L<ll>" -> gemma id -> [text feature, weight][] */
  gemma: Record<string, Record<string, [number, number][]>>
}

/** held-out auc under this, or none (under 2 held-out rows), and the probe says nothing */
export const MIN_AUC = 0.7

const LOADED = new WeakMap<Runtime, Promise<Bridges | null>>()

export function bridgesOf(rt: Runtime): Promise<Bridges | null> {
  let got = LOADED.get(rt)
  if (!got) {
    got = bakeOf(rt).then(bake =>
      bake ? fetchJson<Bridges>(rt, `${bake.base}/bridges.json`).catch(() => null) : null,
    )
    LOADED.set(rt, got)
  }
  return got
}

/** the bridge of one text feature, or null when it has none or its probe is weak */
export function bridgeOf(br: Bridges | null, f: number) {
  const got = br?.features[String(f)]
  if (!got?.b.length || got.auc === null || got.auc < MIN_AUC) return null
  return got
}

/** the text features a Gemma feature predicts, through bridges whose probe holds up */
export function predicts(br: Bridges | null, l: number, f: number): [number, number][] {
  return (br?.gemma[ll(l)]?.[String(f)] ?? []).filter(([j]) => bridgeOf(br, j))
}

/** each Gemma feature's label, else its gloss (strongest tokens), else its id */
export async function gemmaNames(rt: Runtime, pairs: [number, number][]): Promise<string[]> {
  const bake = await bakeOf(rt)
  if (!bake) return pairs.map(([, f]) => `f${f}`)
  const labels = await bake.labels
  return Promise.all(
    pairs.map(async ([l, f]) => {
      const lab = labelOf(labels, l, f)
      if (lab) return lab.label
      const gloss = (await glossOf(rt, bake, l))[String(f)]
      return gloss || `f${f}`
    }),
  )
}
