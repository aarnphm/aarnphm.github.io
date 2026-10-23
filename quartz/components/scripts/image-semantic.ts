import type { SemanticDocument } from '../../util/semantic-search'
import {
  IMAGE_MANIFEST_URL,
  imageManifestUrl,
  imageSearchResults,
  readImageManifest,
} from '../../util/image-search'
import { SemanticClient } from './semantic.inline'

export interface ImageSemantic {
  cached(query: string, limit: number): SemanticDocument[] | undefined
  search(query: string, limit: number): Promise<SemanticDocument[]>
  cancel(): void
  dispose(): void
}

export async function createImageSemantic(): Promise<ImageSemantic> {
  const response = await fetch(IMAGE_MANIFEST_URL, { cache: 'no-cache' })
  if (!response.ok) throw new Error(`failed to load image search metadata: ${response.status}`)
  const index = readImageManifest(await response.json())
  // Immutable metadata and vector URLs keep row IDs bound even while a new generation publishes.
  const client = new SemanticClient({ ...semanticCfg, manifestUrl: imageManifestUrl(index) })
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
  const cancel = () => {
    client.cancel()
    for (const [key, entry] of cache) {
      if (!entry.documents) cache.delete(key)
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
        promise: client.search(text, limit).then(({ semantic }) => {
          const documents = imageSearchResults(semantic, index)
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
