import {
  paintSceneryPart,
  sceneryOrder,
  type SceneryMessage,
  type SceneryPart,
  type SceneryParts,
  type SceneryRequest,
} from './scenery-parts'

declare const TRAIL_SCENERY_WORKER_URL: string

// Paints the terraces, hinterland, terrain and backdrop off the main thread. Each part arrives as
// soon as it is ready. A blocked or failed worker paints the remaining parts on demand.
export function prepareScenery(request: SceneryRequest, signal: AbortSignal) {
  const results = new Map<SceneryPart, SceneryParts[SceneryPart]>()
  const waiting = new Map<SceneryPart, ((value: unknown) => void)[]>()
  let worker: Worker | null = null
  let local = false
  const stop = () => {
    worker?.terminate()
    worker = null
    signal.removeEventListener('abort', cancel)
  }
  const cancel = () => {
    stop()
    for (const resolvers of waiting.values()) for (const resolve of resolvers) resolve(null)
    waiting.clear()
  }
  const fallback = () => {
    stop()
    local = true
    for (const [part, resolvers] of waiting) {
      const value = paintSceneryPart(part, request)
      results.set(part, value)
      for (const resolve of resolvers) resolve(value)
    }
    waiting.clear()
  }
  if (!signal.aborted) {
    signal.addEventListener('abort', cancel, { once: true })
    try {
      worker = new Worker(TRAIL_SCENERY_WORKER_URL, { name: 'trail-scenery' })
      worker.addEventListener('message', (event: MessageEvent<SceneryMessage>) => {
        const { part, value } = event.data
        results.set(part, value)
        for (const resolve of waiting.get(part) ?? []) resolve(value)
        waiting.delete(part)
        if (results.size === sceneryOrder.length) stop()
      })
      worker.addEventListener('error', event => {
        event.preventDefault()
        fallback()
      })
      worker.addEventListener('messageerror', fallback)
      worker.postMessage(request)
    } catch {
      fallback()
    }
  }
  return {
    // Resolves to null once the scene stops waiting for this render.
    get<K extends SceneryPart>(part: K): Promise<SceneryParts[K] | null> {
      if (signal.aborted) return Promise.resolve(null)
      if (results.has(part)) return Promise.resolve(results.get(part) as SceneryParts[K])
      if (local) {
        const value = paintSceneryPart(part, request)
        results.set(part, value)
        return Promise.resolve(value)
      }
      return new Promise(resolve => {
        const resolvers = waiting.get(part) ?? []
        resolvers.push(resolve as (value: unknown) => void)
        waiting.set(part, resolvers)
      })
    },
  }
}
