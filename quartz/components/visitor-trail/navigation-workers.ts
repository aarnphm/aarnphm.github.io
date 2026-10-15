import type { CollisionActor } from './actor-spacing'
import type { JourneyRoute } from './hiker-journey'
import type { Landscape } from './landscape'
import type { NavigationWorkerRequest, NavigationWorkerResponse } from './navigation-worker-types'
import {
  detourTrailSteps,
  plannedTrail,
  trailNavigationInput,
  type NavigableTrail,
  type PlannedTrail,
} from './obstacle-route'

declare const TRAIL_NAVIGATION_WORKER_SOURCE: string

export async function prepareNavigableTrails(
  routes: JourneyRoute[],
  landscape: Landscape,
  obstacles: readonly CollisionActor[],
  signal: AbortSignal,
): Promise<NavigableTrail[]> {
  if (signal.aborted || !routes.length) return []
  const requests = routes.map(route => ({
    id: route.id,
    ...trailNavigationInput(route, landscape, obstacles),
  }))
  const prepared = await new Promise<Map<string, PlannedTrail | null> | null>(resolve => {
    const count = Math.min(Math.max(1, navigator.hardwareConcurrency - 1), requests.length)
    const source = URL.createObjectURL(
      new Blob([TRAIL_NAVIGATION_WORKER_SOURCE], { type: 'text/javascript' }),
    )
    const workers: Worker[] = []
    const results = new Map<string, PlannedTrail | null>()
    let finished = false
    const finish = (result: Map<string, PlannedTrail | null> | null) => {
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
        const worker = new Worker(source, { name: `trail-navigation-${index}` })
        workers.push(worker)
        worker.addEventListener('message', (event: MessageEvent<NavigationWorkerResponse>) => {
          results.set(event.data.id, event.data.planned)
          if (results.size === routes.length) finish(results)
        })
        worker.addEventListener('error', event => {
          event.preventDefault()
          finish(null)
        })
        worker.addEventListener('messageerror', cancel)
        const request: NavigationWorkerRequest = requests.filter(
          (_, route) => route % count === index,
        )
        worker.postMessage(request)
      }
    } catch {
      finish(null)
    }
  })
  if (signal.aborted) return []
  const result: NavigableTrail[] = []
  for (const [index, route] of routes.entries()) {
    let planned = prepared?.get(route.id) ?? null
    if (!prepared) {
      // A blocked worker uses the same search in short, cancellable main-thread slices.
      const input = requests[index]
      const search = detourTrailSteps(input.road, input.radius, input.obstacles, input.bounds)
      let step = search.next()
      while (!step.done) {
        const deadline = performance.now() + 8
        do {
          step = search.next()
        } while (!step.done && performance.now() < deadline)
        if (!step.done) await new Promise<void>(resolve => setTimeout(resolve, 0))
        if (signal.aborted) return []
      }
      planned = step.value
    }
    const next = plannedTrail(route, planned)
    if (next) result.push(next)
  }
  return result
}
