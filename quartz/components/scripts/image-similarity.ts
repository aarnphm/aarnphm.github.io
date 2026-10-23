import type { SemanticDocument } from '../../util/semantic-search'
import {
  IMAGE_MANIFEST_URL,
  imageSearchResults,
  readImageManifest,
  validateImageVectorShard,
  type ImageSearchManifest,
} from '../../util/image-search'

export interface ImageSimilarity {
  ids(): Promise<ReadonlySet<string>>
  search(assetId: string, limit: number, signal?: AbortSignal): Promise<SemanticDocument[]>
  cancel(): void
  dispose(): void
}

type ImageVectors = { index: ImageSearchManifest; vectors: Float32Array }

/** Stored-image queries only load the static index; they never initialize an encoder. */
export function createImageSimilarity(manifestUrl = IMAGE_MANIFEST_URL): ImageSimilarity {
  let disposed = false
  let generation = 0
  let request: AbortController | undefined
  let loading: Promise<ImageVectors> | undefined
  let loaded: ImageVectors | undefined
  let metadata: { index: ImageSearchManifest; url: string } | undefined
  let metadataLoading: Promise<{ index: ImageSearchManifest; url: string }> | undefined
  let metadataRequest: AbortController | undefined

  async function readIndex() {
    if (disposed) throw new Error('Image similarity has been disposed.')
    if (metadata) return metadata
    if (metadataLoading) return metadataLoading
    const controller = new AbortController()
    metadataRequest = controller
    const pending = (async () => {
      const response = await fetch(manifestUrl, { cache: 'no-cache', signal: controller.signal })
      if (!response.ok) throw new Error(`Image similarity index unavailable: ${response.status}`)
      const index = readImageManifest(await response.json())
      controller.signal.throwIfAborted()
      metadata = { index, url: response.url }
      return metadata
    })()
    metadataLoading = pending
    try {
      return await pending
    } finally {
      if (metadataLoading === pending) metadataLoading = undefined
      if (metadataRequest === controller) metadataRequest = undefined
    }
  }

  async function load(): Promise<ImageVectors> {
    if (loaded) return loaded
    if (loading) return loading
    const controller = new AbortController()
    request = controller
    const pending = (async () => {
      const { index, url: manifestResponseUrl } = await readIndex()
      controller.signal.throwIfAborted()
      const vectors = new Float32Array(index.rows * index.dims)
      for (const shard of index.vectors.shards) {
        const url = new URL(shard.path, manifestResponseUrl)
        if (url.origin !== new URL(manifestResponseUrl).origin)
          throw new Error('Image vectors must have the same origin as their manifest.')
        const result = await fetch(url, { signal: controller.signal })
        if (!result.ok) throw new Error(`Image vectors unavailable: ${result.status}`)
        const view = await validateImageVectorShard(await result.arrayBuffer(), shard, index.dims)
        controller.signal.throwIfAborted()
        vectors.set(view, shard.rowOffset * index.dims)
      }
      controller.signal.throwIfAborted()
      loaded = { index, vectors }
      return loaded
    })()
    loading = pending
    try {
      return await pending
    } catch (error) {
      if (loading === pending) metadata = undefined
      throw error
    } finally {
      if (loading === pending) loading = undefined
      if (request === controller) request = undefined
    }
  }

  function cancel() {
    generation++
    request?.abort()
    metadataRequest?.abort()
    metadataLoading = undefined
    metadataRequest = undefined
    request = undefined
    loading = undefined
  }

  return {
    async ids() {
      return new Set((await readIndex()).index.ids)
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
        const { index, vectors } = await load()
        signal?.throwIfAborted()
        if (disposed || current !== generation)
          throw new DOMException('Image similarity superseded.', 'AbortError')
        const seed = index.ids.indexOf(assetId)
        if (seed < 0) throw new Error('This image is not indexed for visual similarity.')
        const offset = seed * index.dims
        const hits: { id: number; score: number }[] = []
        for (let row = 0; row < index.rows; row++) {
          if (row === seed) continue
          let score = 0
          const base = row * index.dims
          for (let dimension = 0; dimension < index.dims; dimension++)
            score += vectors[base + dimension] * vectors[offset + dimension]
          hits.push({ id: row, score })
        }
        return imageSearchResults(hits, index).slice(0, limit)
      } finally {
        signal?.removeEventListener('abort', abort)
      }
    },
    cancel,
    dispose() {
      disposed = true
      cancel()
      loaded = undefined
      metadata = undefined
    },
  }
}
