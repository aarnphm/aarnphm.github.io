import type { Landscape } from './landscape'
import type { ForestSettings, TreePlacement } from './scenery'
import type { TreeWorkerRequest, TreeWorkerResponse } from './tree-worker-types'

declare const TRAIL_TREE_WORKER_SOURCE: string

export function prepareTreeSprites(
  landscape: Landscape,
  trees: TreePlacement[],
  settings: ForestSettings,
  signal: AbortSignal,
): Promise<Map<string, string> | null> {
  if (signal.aborted) return Promise.resolve(null)
  if (!trees.length) return Promise.resolve(new Map())
  const count = Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1), trees.length)
  const source = URL.createObjectURL(
    new Blob([TRAIL_TREE_WORKER_SOURCE], { type: 'text/javascript' }),
  )
  return new Promise(resolve => {
    const workers: Worker[] = []
    const sprites = new Map<string, string>()
    let remaining = count
    let finished = false
    const finish = (result: Map<string, string> | null) => {
      if (finished) return
      finished = true
      for (const worker of workers) worker.terminate()
      URL.revokeObjectURL(source)
      signal.removeEventListener('abort', cancel)
      resolve(result)
    }
    const cancel = () => finish(null)
    signal.addEventListener('abort', cancel, { once: true })
    try {
      for (let index = 0; index < count; index++) {
        const worker = new Worker(source, { name: `trail-trees-${index}` })
        workers.push(worker)
        worker.addEventListener('message', (event: MessageEvent<TreeWorkerResponse>) => {
          for (const { id, markup } of event.data) sprites.set(id, markup)
          if (--remaining === 0) finish(sprites)
        })
        worker.addEventListener('error', event => {
          event.preventDefault()
          finish(null)
        })
        worker.addEventListener('messageerror', cancel)
        const request: TreeWorkerRequest = {
          landscape,
          settings,
          trees: trees.filter((_, tree) => tree % count === index),
        }
        worker.postMessage(request)
      }
    } catch {
      // A blocked worker keeps the synchronous painter available for this opening.
      finish(null)
    }
  })
}
