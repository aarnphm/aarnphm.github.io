import type { Landscape } from './landscape'
import type { ForestSettings, TreePlacement } from './scenery'
import type { TreeWorkerRequest, TreeWorkerResponse } from './tree-worker-types'

declare const TRAIL_TREE_WORKER_SOURCE: string

export function prepareTreeSprites(
  landscape: Landscape,
  trees: TreePlacement[],
  settings: ForestSettings,
  signal: AbortSignal,
): Map<string, Promise<string | null>> {
  const sprites = new Map<string, Promise<string | null>>()
  if (signal.aborted || !trees.length) return sprites
  const pending = new Map<string, (markup: string | null) => void>()
  for (const { id } of trees) sprites.set(id, new Promise(resolve => pending.set(id, resolve)))
  const count = Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1), trees.length)
  const source = URL.createObjectURL(
    new Blob([TRAIL_TREE_WORKER_SOURCE], { type: 'text/javascript' }),
  )
  const workers: Worker[] = []
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    for (const worker of workers) worker.terminate()
    URL.revokeObjectURL(source)
    signal.removeEventListener('abort', finish)
    for (const resolve of pending.values()) resolve(null)
    pending.clear()
  }
  signal.addEventListener('abort', finish, { once: true })
  try {
    for (let index = 0; index < count; index++) {
      const worker = new Worker(source, { name: `trail-trees-${index}` })
      workers.push(worker)
      worker.addEventListener('message', (event: MessageEvent<TreeWorkerResponse>) => {
        for (const { id, markup } of event.data) {
          pending.get(id)?.(markup)
          pending.delete(id)
        }
        if (!pending.size) finish()
      })
      worker.addEventListener('error', event => {
        event.preventDefault()
        finish()
      })
      worker.addEventListener('messageerror', finish)
      const request: TreeWorkerRequest = {
        landscape,
        settings,
        trees: trees.filter((_, tree) => tree % count === index),
      }
      worker.postMessage(request)
    }
  } catch {
    // A blocked worker keeps the synchronous painter available for this opening.
    finish()
  }
  return sprites
}
