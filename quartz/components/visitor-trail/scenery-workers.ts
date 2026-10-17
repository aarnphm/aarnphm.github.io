import { isRecord } from '../../util/type-guards'
import {
  paintSceneryPart,
  sceneryOrder,
  type SceneryMessage,
  type SceneryPart,
  type SceneryParts,
  type SceneryRequest,
  type SceneryWorkerRequest,
} from './scenery-parts'

declare const TRAIL_SCENERY_WORKER_SOURCE: string
declare const TRAIL_SCENERY_VERSION: string

const cacheName = 'garden-trail-scenery-v1'
const maximumBytes = 1024 * 1024
const maximumScenes = 3

function sceneryParts(value: unknown): value is SceneryParts {
  return (
    isRecord(value) &&
    typeof value.terraces === 'string' &&
    typeof value.hinterland === 'string' &&
    typeof value.terrain === 'string' &&
    typeof value.backdrop === 'string'
  )
}

async function cacheTarget(request: SceneryRequest) {
  try {
    const bytes = new TextEncoder().encode(
      JSON.stringify([TRAIL_SCENERY_VERSION, { ...request, backdropClip: undefined }]),
    )
    const [digest, cache] = await Promise.all([
      crypto.subtle.digest('SHA-256', bytes),
      caches.open(cacheName),
    ])
    const key = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join(
      '',
    )
    return { cache, url: new URL(`/.trail-scenery/${key}`, parent.location.href).href }
  } catch {
    return null
  }
}

// Paints the terraces, hinterland, terrain and backdrop off the main thread. Each part arrives as
// soon as it is ready. A blocked or failed worker paints the remaining parts on demand.
export function prepareScenery(request: SceneryRequest, signal: AbortSignal) {
  const results = new Map<SceneryPart, SceneryParts[SceneryPart]>()
  const waiting = new Map<SceneryPart, ((value: string | null) => void)[]>()
  const workers: Worker[] = []
  let workerUrl: string | null = null
  let local = false
  let complete = false
  let cacheHit = false
  const lookupStart = performance.now()
  const target = signal.aborted ? Promise.resolve(null) : cacheTarget(request)
  const stop = () => {
    for (const worker of workers) worker.terminate()
    workers.length = 0
    if (workerUrl) URL.revokeObjectURL(workerUrl)
    workerUrl = null
    signal.removeEventListener('abort', cancel)
  }
  const cancel = () => {
    complete = true
    stop()
    for (const resolvers of waiting.values()) for (const resolve of resolvers) resolve(null)
    waiting.clear()
  }
  const persist = async () => {
    try {
      const destination = await target
      if (!destination || cacheHit) return
      const parts = Object.fromEntries(results)
      if (!sceneryParts(parts)) return
      const body = JSON.stringify({ clip: request.backdropClip, parts })
      if (new Blob([body]).size > maximumBytes) return
      await destination.cache.put(destination.url, new Response(body))
      const keys = await destination.cache.keys()
      await Promise.all(keys.slice(0, -maximumScenes).map(key => destination.cache.delete(key)))
    } catch {
      // Storage is optional; the generated scene is already complete.
    }
  }
  const receive = (part: SceneryPart, value: string) => {
    if (complete || signal.aborted) return
    results.set(part, value)
    for (const resolve of waiting.get(part) ?? []) resolve(value)
    waiting.delete(part)
    if (results.size === sceneryOrder.length) {
      complete = true
      stop()
      void persist()
    }
  }
  const fallback = () => {
    stop()
    local = true
    for (const part of waiting.keys()) receive(part, paintSceneryPart(part, request))
    waiting.clear()
  }
  if (!signal.aborted) {
    signal.addEventListener('abort', cancel, { once: true })
    try {
      workerUrl = URL.createObjectURL(
        new Blob([TRAIL_SCENERY_WORKER_SOURCE], { type: 'text/javascript' }),
      )
      // Terrain and mountains are independent; neither waits for the other's pixel sampling.
      const batches: SceneryPart[][] = [['terraces', 'hinterland', 'terrain'], ['backdrop']]
      for (const parts of batches) {
        const worker = new Worker(workerUrl, { name: `trail-${parts.at(-1)}` })
        workers.push(worker)
        worker.addEventListener('message', (event: MessageEvent<SceneryMessage>) => {
          const { part, value } = event.data
          if (parent.sessionStorage.getItem('trail:profile') === 'true')
            parent.performance.mark('trail:scenery-worker-response', { detail: { part } })
          receive(part, value)
        })
        worker.addEventListener('error', event => {
          event.preventDefault()
          fallback()
        })
        worker.addEventListener('messageerror', fallback)
        const message: SceneryWorkerRequest = { ...request, parts }
        if (parent.sessionStorage.getItem('trail:profile') === 'true')
          parent.performance.mark('trail:scenery-worker-request', { detail: { parts } })
        worker.postMessage(message)
      }
    } catch {
      fallback()
    }
  }
  void (async () => {
    try {
      const destination = await target
      const response = await destination?.cache.match(destination.url)
      if (complete || signal.aborted) return
      const body = await response?.blob()
      const value: unknown =
        body && body.size <= maximumBytes ? JSON.parse(await body.text()) : null
      if (complete || signal.aborted) return
      if (
        isRecord(value) &&
        typeof value.clip === 'string' &&
        sceneryParts(value.parts) &&
        value.parts.backdrop.includes(`clip-path="url(#${value.clip})"`)
      ) {
        cacheHit = true
        for (const part of sceneryOrder)
          receive(
            part,
            value.parts[part].replaceAll(`url(#${value.clip})`, `url(#${request.backdropClip})`),
          )
        parent.performance.mark('trail:scenery-cache-hit', {
          detail: { lookupMs: performance.now() - lookupStart },
        })
        return
      }
    } catch {
      // A blocked or malformed cache leaves the normal worker running.
    }
    if (!complete && !signal.aborted)
      parent.performance.mark('trail:scenery-cache-miss', {
        detail: { lookupMs: performance.now() - lookupStart },
      })
  })()
  return {
    // Resolves to null once the scene stops waiting for this render.
    get(part: SceneryPart): Promise<string | null> {
      if (signal.aborted) return Promise.resolve(null)
      const value = results.get(part)
      if (value !== undefined) return Promise.resolve(value)
      if (local) {
        const value = paintSceneryPart(part, request)
        receive(part, value)
        return Promise.resolve(value)
      }
      return new Promise(resolve => {
        const resolvers = waiting.get(part) ?? []
        resolvers.push(resolve)
        waiting.set(part, resolvers)
      })
    },
  }
}
