import type { SemanticDocument } from '../../util/semantic-search'
import { IMAGE_MANIFEST_URL, imageSearchResults } from '../../util/image-search'
import { isSemanticCancellation, SemanticClient } from './semantic.inline'

export interface ImageSimilarity {
  ids(): Promise<ReadonlySet<string>>
  search(assetId: string, limit: number, signal?: AbortSignal): Promise<SemanticDocument[]>
  cancel(): void
  dispose(): void
}

/**
 * The worker holds the image index and scans its stored vectors. Similarity never loads the
 * query encoder, and the client opens on first use so a page load starts no worker.
 */
export function createImageSimilarity(manifestUrl = IMAGE_MANIFEST_URL): ImageSimilarity {
  let client: SemanticClient | undefined
  let indexed: Promise<ReadonlySet<string>> | undefined
  let disposed = false
  let generation = 0
  const open = () => {
    if (disposed) throw new Error('Image similarity has been disposed.')
    return (client ??= new SemanticClient({ ...semanticCfg, manifestUrl }))
  }
  const cancel = () => {
    generation++
    client?.cancel()
  }

  async function similar(assetId: string, limit: number): Promise<SemanticDocument[]> {
    const current = generation
    const assertCurrent = () => {
      if (disposed || current !== generation)
        throw new DOMException('Image similarity superseded.', 'AbortError')
    }
    try {
      const { semantic, ids } = await open().similar(assetId, limit)
      assertCurrent()
      return imageSearchResults(semantic, ids ?? [])
    } catch (error) {
      assertCurrent()
      if (isSemanticCancellation(error)) throw error
      // A generation may have been replaced under the worker; a fresh client resolves the new one.
      client?.dispose()
      client = undefined
      indexed = undefined
      const { semantic, ids } = await open().similar(assetId, limit)
      assertCurrent()
      return imageSearchResults(semantic, ids ?? [])
    }
  }

  return {
    ids() {
      if (!indexed) {
        const pending = (async () => new Set(await open().ids()))()
        indexed = pending
        void pending.catch(() => {
          if (indexed === pending) indexed = undefined
        })
      }
      return indexed
    },
    async search(assetId, limit, signal) {
      if (disposed) throw new Error('Image similarity has been disposed.')
      signal?.throwIfAborted()
      if (!Number.isSafeInteger(limit) || limit <= 0) return []
      cancel()
      const current = generation
      const abort = () => {
        if (generation === current) cancel()
      }
      signal?.addEventListener('abort', abort, { once: true })
      try {
        return await similar(assetId, limit)
      } finally {
        signal?.removeEventListener('abort', abort)
      }
    },
    cancel,
    dispose() {
      if (disposed) return
      disposed = true
      cancel()
      indexed = undefined
      client?.dispose()
      client = undefined
    },
  }
}
