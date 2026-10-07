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
