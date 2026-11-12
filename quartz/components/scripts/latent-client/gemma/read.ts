import type { BakeRow } from './bake'

// ---- reading the selection --------------------------------------------------------------------

export function tokensIn(data: BakeRow, s: number, e: number): number[] {
  const out: number[] = []
  for (let t = 0; t < data.T; t++) {
    const a = data.spans[2 * t]
    const b = data.spans[2 * t + 1]
    if (b > s && a < e) out.push(t)
  }
  return out
}

/** per layer, the strongest features over the picked tokens (max-pooled), and each layer's max */
export function pool(data: BakeRow, picked: number[]) {
  const { L, K, T } = data
  const layerMax = new Float32Array(L)
  for (let t = 0; t < T; t++)
    for (let l = 0; l < L; l++) layerMax[l] = Math.max(layerMax[l], data.acts[(t * L + l) * K])
  const layers: { f: number; a: number }[][] = []
  for (let l = 0; l < L; l++) {
    const best = new Map<number, number>()
    for (const t of picked)
      for (let k = 0; k < K; k++) {
        const i = (t * L + l) * K + k
        const a = data.acts[i]
        if (a <= 0) continue
        const f = data.ids[i]
        if (a > (best.get(f) ?? 0)) best.set(f, a)
      }
    layers.push([...best].map(([f, a]) => ({ f, a })).sort((x, y) => y.a - x.a))
  }
  return { layers, layerMax }
}

export function tokensOfFeature(data: BakeRow, l: number, f: number): { t: number; a: number }[] {
  const out: { t: number; a: number }[] = []
  for (let t = 0; t < data.T; t++)
    for (let k = 0; k < data.K; k++) {
      const i = (t * data.L + l) * data.K + k
      if (data.ids[i] === f && data.acts[i] > 0) out.push({ t, a: data.acts[i] })
    }
  return out
}

/** what the picked tokens read at one global layer: sources outside the selection, by weight */
export function reads(data: BakeRow, picked: number[], a: number): Map<number, number> {
  const inside = new Set(picked)
  const got = new Map<number, number>()
  for (const t of picked)
    for (let k = 0; k < data.KA; k++) {
      const i = (t * data.A + a) * data.KA + k
      const s = data.src[i]
      const w = data.w[i]
      if (w <= 0 || inside.has(s)) continue
      got.set(s, (got.get(s) ?? 0) + w / picked.length)
    }
  return got
}
