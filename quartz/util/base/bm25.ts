import type { BaseGeoFields } from './geo-types'

const FIELDS = ['title', 'type', 'description', 'address'] as const
type Field = (typeof FIELDS)[number]
const WEIGHT: Record<Field, number> = { title: 4, type: 3, description: 2, address: 1 }
const K1 = 1.2
const B = 0.75

function tokens(value: string): string[] {
  return value
    .normalize('NFD')
    .replace(/\u0300-\u036f/g, '')
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(token => token.length > 0)
}

function termFrequency(list: readonly string[], token: string): number {
  let count = 0
  for (const item of list) if (item === token) count += 1
  return count
}

export function bm25Rank(
  members: { slug: string; fields: BaseGeoFields }[],
  query: string,
): { slug: string; score: number }[] {
  const queryTokens = tokens(query)
  if (queryTokens.length === 0 || members.length === 0) return []
  const docs = members.map(member => ({
    slug: member.slug,
    fields: {
      title: tokens(member.fields.title),
      type: tokens(member.fields.type),
      description: tokens(member.fields.description),
      address: tokens(member.fields.address),
    },
  }))
  const needed = new Set(queryTokens)
  const hits = docs.filter(doc =>
    queryTokens.every(token => FIELDS.some(field => doc.fields[field].includes(token))),
  )
  if (hits.length === 0) return []
  const total = docs.length
  const average: Record<Field, number> = { title: 0, type: 0, description: 0, address: 0 }
  const documentFrequency: Record<Field, Map<string, number>> = {
    title: new Map(),
    type: new Map(),
    description: new Map(),
    address: new Map(),
  }
  for (const doc of docs) {
    for (const field of FIELDS) {
      average[field] += doc.fields[field].length
      const seen = new Set<string>()
      for (const token of doc.fields[field]) {
        if (!needed.has(token) || seen.has(token)) continue
        seen.add(token)
        documentFrequency[field].set(token, (documentFrequency[field].get(token) ?? 0) + 1)
      }
    }
  }
  for (const field of FIELDS) average[field] /= total
  const ranked = hits.map(doc => {
    let score = 0
    for (const field of FIELDS) {
      const avg = average[field]
      if (avg === 0) continue
      const length = doc.fields[field].length
      let fieldScore = 0
      for (const token of queryTokens) {
        const frequency = termFrequency(doc.fields[field], token)
        if (frequency === 0) continue
        const df = documentFrequency[field].get(token) ?? 0
        const idf = Math.log(1 + (total - df + 0.5) / (df + 0.5))
        const denominator = frequency + K1 * (1 - B + (B * length) / avg)
        fieldScore += (idf * (frequency * (K1 + 1))) / denominator
      }
      score += WEIGHT[field] * fieldScore
    }
    return { slug: doc.slug, score }
  })
  ranked.sort(
    (left, right) =>
      right.score - left.score || (left.slug < right.slug ? -1 : left.slug > right.slug ? 1 : 0),
  )
  return ranked
}
