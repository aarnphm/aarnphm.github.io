import {
  aggregateSemanticResults,
  type SemanticDocument,
  type SemanticIndex,
} from '../../util/semantic-search'
import { isRecord } from '../../util/type-guards'
import { SemanticClient } from './semantic.inline'

export interface SiteSemantic {
  client: SemanticClient
  index: SemanticIndex
  cached(query: string, limit: number): SemanticDocument[] | undefined
  search(query: string, limit: number): Promise<SemanticDocument[]>
}

async function readIndex(): Promise<SemanticIndex> {
  const response = await fetch('/embeddings/manifest.json')
  if (!response.ok) throw new Error(`failed to load semantic metadata: ${response.status}`)
  const data: unknown = await response.json()
  if (!isRecord(data) || !Array.isArray(data.ids) || !data.ids.every(id => typeof id === 'string'))
    throw new Error('The semantic index has no valid note inventory.')
  const chunkMetadata: Record<string, { parentSlug: string; chunkId: number }> = {}
  if (data.chunkMetadata !== undefined) {
    if (!isRecord(data.chunkMetadata)) throw new Error('Invalid semantic passage inventory.')
    for (const [slug, chunk] of Object.entries(data.chunkMetadata)) {
      if (
        !isRecord(chunk) ||
        typeof chunk.parentSlug !== 'string' ||
        typeof chunk.chunkId !== 'number'
      )
        throw new Error('Invalid semantic passage metadata.')
      chunkMetadata[slug] = { parentSlug: chunk.parentSlug, chunkId: chunk.chunkId }
    }
  }
  return { ids: data.ids, chunkMetadata }
}

export function siteSemantic(): Promise<SiteSemantic> {
  if (window.quartzSiteSemantic) return window.quartzSiteSemantic
  const client = new SemanticClient(semanticCfg)
  const ready = Promise.all([client.ensureReady(), readIndex()])
    .then(async ([, index]) => {
      await client.search('search', 1)
      const cache = new Map<
        string,
        { promise: Promise<SemanticDocument[]>; documents?: SemanticDocument[] }
      >()
      const cacheKey = (query: string, limit: number) => `${limit}\u0000${query.trim()}`
      let queue: Promise<unknown> = Promise.resolve()
      return {
        client,
        index,
        cached(query: string, limit: number) {
          return cache.get(cacheKey(query, limit))?.documents
        },
        search(query: string, limit: number) {
          const text = query.trim()
          const key = cacheKey(text, limit)
          const previous = cache.get(key)
          if (previous) return previous.promise
          const result = queue.then(async () => {
            const { semantic } = await client.search(text, limit)
            const documents = aggregateSemanticResults(semantic, index)
            entry.documents = documents
            return documents
          })
          const entry: { promise: Promise<SemanticDocument[]>; documents?: SemanticDocument[] } = {
            promise: result,
          }
          cache.set(key, entry)
          if (cache.size > 32) {
            const first = cache.keys().next().value
            if (first !== undefined) cache.delete(first)
          }
          queue = result.catch(() => {})
          void result.catch(() => {
            if (cache.get(key) === entry) cache.delete(key)
          })
          return result
        },
      }
    })
    .catch(error => {
      client.dispose()
      if (window.quartzSiteSemantic === ready) window.quartzSiteSemantic = undefined
      throw error
    })
  window.quartzSiteSemantic = ready
  return ready
}

export function resetSiteSemantic(client: SemanticClient) {
  void window.quartzSiteSemantic?.then(
    service => {
      if (service.client !== client) return
      client.dispose()
      window.quartzSiteSemantic = undefined
    },
    () => {},
  )
}
