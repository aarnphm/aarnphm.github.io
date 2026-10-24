import type { SemanticDocument } from '../../util/semantic-search'
import { IMAGE_MANIFEST_URL, imageSearchResults } from '../../util/image-search'
import { isSemanticCancellation, SemanticClient } from './semantic.inline'

export interface ImageSemantic {
  cached(query: string, limit: number): SemanticDocument[] | undefined
  search(query: string, limit: number): Promise<SemanticDocument[]>
  cancel(): void
  dispose(): void
}

/** The worker resolves the root pointer to its immutable generation and returns slugs. */
export async function createImageSemantic(): Promise<ImageSemantic> {
  let client = new SemanticClient({ ...semanticCfg, manifestUrl: IMAGE_MANIFEST_URL })
  try {
    await client.ensureReady()
  } catch (error) {
    client.dispose()
    throw error
  }
  type Entry = { promise: Promise<SemanticDocument[]>; documents?: SemanticDocument[] }
  const cache = new Map<string, Entry>()
  const cacheKey = (query: string, limit: number) => `${limit}\u0000${query.trim()}`
  let disposed = false
  let generation = 0
  const cancel = () => {
    generation++
    client.cancel()
    for (const [key, entry] of cache) {
      if (!entry.documents) cache.delete(key)
    }
  }
  async function search(text: string, limit: number): Promise<SemanticDocument[]> {
    const queryGeneration = generation
    const assertCurrent = () => {
      if (disposed || queryGeneration !== generation)
        throw new DOMException('Image query superseded', 'AbortError')
    }
    try {
      const { semantic, ids } = await client.search(text, limit)
      assertCurrent()
      return imageSearchResults(semantic, ids ?? [])
    } catch (error) {
      assertCurrent()
      if (isSemanticCancellation(error)) throw error
      // A generation may have been replaced under the worker; a fresh client resolves the new one.
      client.dispose()
      client = new SemanticClient({ ...semanticCfg, manifestUrl: IMAGE_MANIFEST_URL })
      cache.clear()
      const { semantic, ids } = await client.search(text, limit)
      assertCurrent()
      return imageSearchResults(semantic, ids ?? [])
    }
  }
  return {
    cached(query, limit) {
      return cache.get(cacheKey(query, limit))?.documents
    },
    search(query, limit) {
      if (disposed) return Promise.reject(new Error('image search has been disposed'))
      const text = query.trim()
      if (!text || limit <= 0) return Promise.resolve([])
      const key = cacheKey(text, limit)
      const previous = cache.get(key)
      if (previous) return previous.promise
      const entry: Entry = {
        promise: search(text, limit).then(documents => {
          entry.documents = documents
          return documents
        }),
      }
      cache.set(key, entry)
      if (cache.size > 32) {
        const oldest = cache.keys().next().value
        if (oldest !== undefined) cache.delete(oldest)
      }
      void entry.promise.catch(() => {
        if (cache.get(key) === entry) cache.delete(key)
      })
      return entry.promise
    },
    cancel,
    dispose() {
      if (disposed) return
      disposed = true
      cancel()
      cache.clear()
      client.dispose()
    },
  }
}
