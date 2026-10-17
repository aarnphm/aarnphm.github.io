import type { ForestSprite } from './forest-sprites'
import type { Landscape } from './landscape'
import type { ForestSettings } from './scene-settings'
import type { TreePlacement } from './scenery'
import type { ForegroundTreeInput, TreeWorkerRequest, TreeWorkerMessage } from './tree-worker-types'
import { repoussoirSpriteSteps } from './repoussoir'
import { treeGrowthDepth } from './scenery'
import {
  readSceneTrees,
  sceneTreeCacheKey,
  writeSceneTrees,
  type CachedSceneTrees,
} from './tree-cache'

declare const TRAIL_TREE_WORKER_SOURCE: string
declare const TRAIL_TREE_VERSION: string

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
    foreground ? 5 : 4,
    Math.max(1, (navigator.hardwareConcurrency || 2) - 1),
    Math.max(1, trees.length + Number(Boolean(foreground))),
  )
  const separateForeground = Boolean(foreground) && count > 1
  const treeWorkers = count - Number(separateForeground)
  const workers: Worker[] = []
  const completed: CachedSceneTrees = { trees: [], foreground: [] }
  let cacheKey: string | null = null
  let workerUrl: string | null = null
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    for (const worker of workers) worker.terminate()
    if (workerUrl) URL.revokeObjectURL(workerUrl)
    workerUrl = null
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
  const startWorkers = () => {
    try {
      workerUrl = URL.createObjectURL(
        new Blob([TRAIL_TREE_WORKER_SOURCE], { type: 'text/javascript' }),
      )
      for (let index = 0; index < count; index++) {
        const worker = new Worker(workerUrl, { name: `trail-trees-${index}` })
        workers.push(worker)
        worker.addEventListener('message', (event: MessageEvent<TreeWorkerMessage>) => {
          if (parent.sessionStorage.getItem('trail:profile') === 'true')
            parent.performance.mark('trail:tree-worker-response', {
              detail: { index, kind: 'kind' in event.data ? event.data.kind : 'trees' },
            })
          if (finished || signal.aborted) return
          if ('kind' in event.data) {
            completed.foreground = event.data.sprites
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
              completed.trees.push({ id, markup })
              pending.get(id)?.(markup)
              pending.delete(id)
            }
          }
          if (!pending.size && !resolveForeground) {
            finish()
            if (cacheKey) void writeSceneTrees(cacheKey, completed)
          }
        })
        worker.addEventListener('error', event => {
          event.preventDefault()
          finish()
        })
        worker.addEventListener('messageerror', finish)
        const request: TreeWorkerRequest = {
          landscape,
          settings,
          trees: trees.filter(
            (_, tree) => tree % treeWorkers === index - Number(separateForeground),
          ),
          foreground: index === 0 ? foreground : undefined,
        }
        if (parent.sessionStorage.getItem('trail:profile') === 'true')
          parent.performance.mark('trail:tree-worker-request', { detail: { index, request } })
        worker.postMessage(request)
      }
    } catch {
      finish()
    }
  }
  const restore = async () => {
    if (!foreground) return startWorkers()
    let timer = 0
    const lookupStart = performance.now()
    try {
      // Key only the growth values represented in the SVG; fractional forest time keeps moving.
      cacheKey = await sceneTreeCacheKey([
        'scene-trees-v2',
        TRAIL_TREE_VERSION,
        landscape,
        settings,
        foreground,
        trees.map(tree => ({
          ...tree,
          growth: tree.growth.toFixed(2),
          grown: treeGrowthDepth(tree.growth),
          age: tree.age === undefined ? null : Math.floor(tree.age),
          succeeds: undefined,
          successor: undefined,
          log: undefined,
        })),
      ])
      const cached = await Promise.race([
        readSceneTrees(cacheKey),
        new Promise<null>(resolve => {
          timer = window.setTimeout(() => resolve(null), 250)
        }),
      ])
      if (finished || signal.aborted) return
      if (
        cached &&
        cached.trees.length === trees.length &&
        new Set(cached.trees.map(tree => tree.id)).size === trees.length &&
        cached.trees.every(tree => pending.has(tree.id))
      ) {
        for (const tree of cached.trees) pending.get(tree.id)?.(tree.markup)
        pending.clear()
        resolveForeground?.(
          cached.foreground.map(({ markup, ...sprite }) => ({ ...sprite, render: () => markup })),
        )
        resolveForeground = null
        finish()
        parent.performance.mark('trail:tree-cache-hit', {
          detail: {
            trees: cached.trees.length,
            key: cacheKey,
            lookupMs: performance.now() - lookupStart,
          },
        })
        return
      }
      parent.performance.mark('trail:tree-cache-miss', {
        detail: { lookupMs: performance.now() - lookupStart, key: cacheKey },
      })
    } catch {
      cacheKey = null
    } finally {
      window.clearTimeout(timer)
    }
    if (!finished && !signal.aborted) startWorkers()
  }
  void restore()
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
