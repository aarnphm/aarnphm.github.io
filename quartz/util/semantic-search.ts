/**
 * ONNX weight precision for the in-browser query encoder. The fp16 variants are
 * left out on purpose: embeddinggemma-2 returns NaN or silently degraded vectors in float16.
 */
export type SemanticQueryDType = 'q4' | 'q8' | 'fp32'
/** Browser-side ONNX repo for each indexed model id. The manifest keeps the canonical Hub id. */
export const SEMANTIC_ONNX_MODELS: Readonly<Record<string, string>> = {
  'intfloat/multilingual-e5-large': 'Xenova/multilingual-e5-large',
  'google/embeddinggemma-300m': 'onnx-community/embeddinggemma-300m-ONNX',
  'google/embeddinggemma-2': 'onnx-community/embeddinggemma-2-ONNX',
  'Qwen/Qwen3-Embedding-0.6B': 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
}
export type SemanticHit = { id: number; score: number }
export type SemanticDocument = { slug: string; score: number }
export type SemanticIndex = {
  ids: readonly string[]
  chunkMetadata?: Readonly<Record<string, { parentSlug: string; chunkId: number }>>
}

export function aggregateSemanticResults(
  hits: readonly SemanticHit[],
  index: SemanticIndex,
): SemanticDocument[] {
  const scores = new Map<string, number>()
  for (const { id, score } of hits) {
    const chunkSlug = index.ids[id]
    if (!chunkSlug || !Number.isFinite(score)) continue
    const slug = index.chunkMetadata?.[chunkSlug]?.parentSlug ?? chunkSlug
    const previous = scores.get(slug)
    // Overlapping passages supply one document's evidence; counting them rewards length.
    if (previous === undefined || score > previous) scores.set(slug, score)
  }
  return Array.from(scores, ([slug, score]) => ({ slug, score })).sort(
    (a, b) => b.score - a.score || a.slug.localeCompare(b.slug),
  )
}

export function fuseSearchRanks<Id extends string | number>(
  rankings: readonly { ids: readonly Id[]; weight: number; boosts?: ReadonlyMap<Id, number> }[],
  rankConstant = 60,
): { id: Id; score: number }[] {
  const scores = new Map<Id, number>()
  for (const { ids, weight, boosts } of rankings) {
    if (weight <= 0) continue
    let rank = 0
    for (const id of new Set(ids)) {
      const contribution = (weight * (boosts?.get(id) ?? 1)) / (rankConstant + ++rank)
      scores.set(id, (scores.get(id) ?? 0) + contribution)
    }
  }
  return Array.from(scores, ([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score)
}
