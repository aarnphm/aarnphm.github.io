import type { ForestSprite } from './forest-sprites'
import type { Landscape } from './landscape'
import type { ForestSettings } from './scene-settings'
import type { TreePlacement } from './scenery'
import type { ForegroundTreeInput, TreeWorkerRequest, TreeWorkerMessage } from './tree-worker-types'
import { repoussoirSpriteSteps } from './repoussoir'

declare const TRAIL_TREE_WORKER_URL: string

function foregroundFallback(
  landscape: Landscape,
  input: ForegroundTreeInput,
  signal: AbortSignal,
): Promise<ForestSprite[]> {
  if (signal.aborted) return Promise.resolve([])
  return new Promise(resolve => {
    const search = repoussoirSpriteSteps(landscape, input.seed, input.clearings)
    let timer = 0
    const finish = (sprites: ForestSprite[]) => {
      clearTimeout(timer)
      signal.removeEventListener('abort', cancel)
      resolve(sprites)
    }
    const cancel = () => finish([])
    const tick = () => {
      if (signal.aborted) return finish([])
      const deadline = performance.now() + 6
      let step = search.next()
      while (!step.done && performance.now() < deadline) step = search.next()
      if (step.done) finish(step.value)
      else timer = window.setTimeout(tick, 0)
    }
    signal.addEventListener('abort', cancel, { once: true })
    timer = window.setTimeout(tick, 0)
  })
}

export function prepareSceneTrees(
  landscape: Landscape,
  trees: TreePlacement[],
  settings: ForestSettings,
  signal: AbortSignal,
  foreground?: ForegroundTreeInput,
): { trees: Map<string, Promise<string | null>>; foreground: Promise<ForestSprite[]> } {
  const sprites = new Map<string, Promise<string | null>>()
  let resolveForeground: ((sprites: ForestSprite[]) => void) | null = null
  const foregroundReady =
    foreground && !signal.aborted
      ? new Promise<ForestSprite[]>(resolve => {
          resolveForeground = resolve
        })
      : Promise.resolve([])
  const prepared = { trees: sprites, foreground: foregroundReady }
  if (signal.aborted || (!trees.length && !foreground)) return prepared
  const pending = new Map<string, (markup: string | null) => void>()
  for (const { id } of trees) sprites.set(id, new Promise(resolve => pending.set(id, resolve)))
  const count = Math.min(
    4,
    Math.max(1, (navigator.hardwareConcurrency || 2) - 1),
    Math.max(1, trees.length),
  )
  const workers: Worker[] = []
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    for (const worker of workers) worker.terminate()
    signal.removeEventListener('abort', finish)
    for (const resolve of pending.values()) resolve(null)
    pending.clear()
    if (resolveForeground) {
      const resolve = resolveForeground
      resolveForeground = null
      if (foreground && !signal.aborted)
        void foregroundFallback(landscape, foreground, signal).then(resolve)
      else resolve([])
    }
  }
  signal.addEventListener('abort', finish, { once: true })
  try {
    for (let index = 0; index < count; index++) {
      const worker = new Worker(TRAIL_TREE_WORKER_URL, { name: `trail-trees-${index}` })
      workers.push(worker)
      worker.addEventListener('message', (event: MessageEvent<TreeWorkerMessage>) => {
        if (finished || signal.aborted) return
        if ('kind' in event.data) {
          const resolve = resolveForeground
          resolveForeground = null
          resolve?.(
            event.data.sprites.map(({ markup, ...sprite }) => ({
              ...sprite,
              render: () => markup,
            })),
          )
        } else {
          for (const { id, markup } of event.data) {
            pending.get(id)?.(markup)
            pending.delete(id)
          }
        }
        if (!pending.size && !resolveForeground) finish()
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
        foreground: index === 0 ? foreground : undefined,
      }
      worker.postMessage(request)
    }
  } catch {
    finish()
  }
  return prepared
}

export function prepareTreeSprites(
  landscape: Landscape,
  trees: TreePlacement[],
  settings: ForestSettings,
  signal: AbortSignal,
): Map<string, Promise<string | null>> {
  return prepareSceneTrees(landscape, trees, settings, signal).trees
}
