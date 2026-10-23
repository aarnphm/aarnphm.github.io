import type { JourneyRoute } from './hiker-journey'
import type { Landscape, Point } from './landscape'
import type {
  NavigationObstacle,
  NavigationQuery,
  NavigationRoute,
  NavigationWorkerRequest,
  NavigationWorkerResponse,
} from './navigation-worker-types'
import { canMove, type CollisionActor, type SpacingBounds } from './actor-spacing'
import { navigationSearch } from './navigation-search'
import {
  plannedTrail,
  trailNavigationInput,
  type NavigableTrail,
  type PlannedTrail,
} from './obstacle-route'

declare const TRAIL_NAVIGATION_WORKER_URL: string

const snapshot = (obstacles: readonly CollisionActor[]): NavigationObstacle[] =>
  obstacles.map(({ element, ...body }) => ({
    ...body,
    yieldingTo: element?.dataset.yieldingTo ?? body.yieldingTo,
    yieldingThrough: element?.dataset.yieldingThrough ?? body.yieldingThrough,
  }))
const routesSnapshot = (routes: JourneyRoute[]): NavigationRoute[] =>
  routes.map(route => ({
    id: route.id,
    readyAfter: route.readyAfter,
    points:
      route.points ??
      Array.from({ length: Math.max(1, Math.ceil(route.length / 4)) + 1 }, (_, index) =>
        route.point((route.length * index) / Math.max(1, Math.ceil(route.length / 4))),
      ),
  }))

type Job = {
  id: number
  query: NavigationQuery
  signal?: AbortSignal
  resolve: (value: PlannedTrail | null) => void
  cancel: () => void
  done: boolean
}
type Slot = { worker: Worker; job: Job | null }

export function createTrailNavigation(lifetime: AbortSignal) {
  let sequence = 0
  let unavailable = false
  const queue: Job[] = []
  const slots: Slot[] = []
  const jobs = new Set<Job>()
  const localQueue: { job: Job; search: ReturnType<typeof navigationSearch> }[] = []
  let localTimer = 0
  const limit = Math.min(3, Math.max(1, (navigator.hardwareConcurrency || 2) - 1))
  const finish = (job: Job, value: PlannedTrail | null) => {
    if (job.done) return
    job.done = true
    job.signal?.removeEventListener('abort', job.cancel)
    jobs.delete(job)
    if (!jobs.size) {
      clearTimeout(localTimer)
      localTimer = 0
      localQueue.length = 0
    }
    job.resolve(value)
  }
  const advanceLocal = () => {
    localTimer = 0
    const deadline = performance.now() + 6
    while (localQueue.length && performance.now() < deadline) {
      const entry = localQueue.shift()
      if (!entry || entry.job.done) continue
      if (lifetime.aborted || entry.job.signal?.aborted) {
        finish(entry.job, null)
        continue
      }
      const slice = Math.min(deadline, performance.now() + 1)
      let step = entry.search.next()
      while (!step.done && performance.now() < slice) step = entry.search.next()
      if (step.done) finish(entry.job, step.value)
      else localQueue.push(entry)
    }
    if (localQueue.length) localTimer = window.setTimeout(advanceLocal, 0)
  }
  const locally = (job: Job) => {
    if (localQueue.some(entry => entry.job === job)) return
    localQueue.push({ job, search: navigationSearch(job.query) })
    if (!localTimer) localTimer = window.setTimeout(advanceLocal, 0)
  }
  const failWorkers = () => {
    if (unavailable) return
    unavailable = true
    for (const slot of slots) slot.worker.terminate()
    slots.length = 0
    queue.length = 0
    for (const job of jobs) if (!job.done) locally(job)
  }
  const spawn = (): Slot | null => {
    try {
      const worker = new Worker(TRAIL_NAVIGATION_WORKER_URL, {
        name: `trail-navigation-${slots.length}`,
      })
      const slot: Slot = { worker, job: null }
      worker.addEventListener('message', (event: MessageEvent<NavigationWorkerResponse>) => {
        const job = slot.job
        if (!slots.includes(slot) || !job || event.data.id !== job.id) return
        slot.job = null
        finish(job, event.data.planned)
        pump()
      })
      worker.addEventListener('error', event => {
        event.preventDefault()
        if (slots.includes(slot)) failWorkers()
      })
      worker.addEventListener('messageerror', () => {
        if (slots.includes(slot)) failWorkers()
      })
      slots.push(slot)
      return slot
    } catch {
      failWorkers()
      return null
    }
  }
  const pump = () => {
    if (lifetime.aborted || unavailable) return
    while (queue.length) {
      const slot = slots.find(item => !item.job) ?? (slots.length < limit ? spawn() : null)
      if (!slot) return
      const job = queue.shift()
      if (!job || job.done) continue
      slot.job = job
      const request: NavigationWorkerRequest = { id: job.id, query: job.query }
      try {
        slot.worker.postMessage(request)
      } catch {
        failWorkers()
        return
      }
    }
  }
  const run = (
    query: NavigationQuery,
    signal?: AbortSignal,
    execution: 'worker' | 'local' = 'worker',
  ): Promise<PlannedTrail | null> => {
    if (lifetime.aborted || signal?.aborted) return Promise.resolve(null)
    return new Promise(resolve => {
      const job: Job = {
        id: ++sequence,
        query,
        signal,
        resolve,
        done: false,
        cancel: () => {
          const index = slots.findIndex(slot => slot.job === job)
          if (index >= 0) {
            slots[index].worker.terminate()
            slots.splice(index, 1)
          }
          const queued = queue.indexOf(job)
          if (queued >= 0) queue.splice(queued, 1)
          finish(job, null)
          queueMicrotask(pump)
        },
      }
      jobs.add(job)
      signal?.addEventListener('abort', job.cancel, { once: true })
      if (unavailable || execution === 'local') locally(job)
      else {
        queue.push(job)
        pump()
      }
    })
  }
  lifetime.addEventListener(
    'abort',
    () => {
      for (const slot of slots) slot.worker.terminate()
      slots.length = queue.length = 0
      for (const job of jobs) finish(job, null)
    },
    { once: true },
  )
  return {
    plan: (
      road: readonly Point[],
      radius: number,
      obstacles: readonly CollisionActor[],
      bounds: SpacingBounds,
      signal?: AbortSignal,
    ) =>
      run(
        { kind: 'trail', road: [...road], radius, obstacles: snapshot(obstacles), bounds },
        signal,
      ),
    async repair(
      actor: string,
      from: Point,
      waypoints: readonly Point[],
      radius: number,
      obstacles: readonly CollisionActor[],
      bounds: SpacingBounds,
      waitingForAnimal?: string | null,
      signal?: AbortSignal,
    ) {
      const result = await run(
        {
          kind: 'repair',
          actor,
          from: { ...from },
          waypoints: [...waypoints],
          radius,
          obstacles: snapshot(obstacles),
          bounds,
          waitingForAnimal,
        },
        signal,
      )
      return result?.points ?? null
    },
    async patrol(
      from: Point,
      to: Point,
      routes: JourneyRoute[],
      landscape: Landscape,
      obstacles: readonly CollisionActor[],
      signal?: AbortSignal,
    ) {
      const result = await run(
        {
          kind: 'patrol',
          from: { ...from },
          to: { ...to },
          routes: routesSnapshot(routes),
          landscape,
          obstacles: snapshot(obstacles),
        },
        signal,
      )
      return result?.points ?? null
    },
    async departure(
      from: Point,
      routes: JourneyRoute[],
      landscape: Landscape,
      obstacles: readonly CollisionActor[],
      signal?: AbortSignal,
    ) {
      const result = await run(
        {
          kind: 'departure',
          from: { ...from },
          routes: routesSnapshot(routes),
          landscape,
          obstacles: snapshot(obstacles),
        },
        signal,
      )
      return result?.points ?? null
    },
    async prepare(
      routes: JourneyRoute[],
      landscape: Landscape,
      obstacles: readonly CollisionActor[],
      signal?: AbortSignal,
      phase: 'opening' | 'active' = 'opening',
    ): Promise<NavigableTrail[]> {
      const results = await Promise.all(
        routes.map(async route => {
          const input = trailNavigationInput(route, landscape, obstacles)
          const result = await run(
            { kind: 'trail', ...input, obstacles: snapshot(obstacles) },
            signal,
            phase === 'opening' ? 'local' : 'worker',
          )
          return plannedTrail(route, result)
        }),
      )
      if (lifetime.aborted || signal?.aborted) return []
      return results.filter((route): route is NavigableTrail => route !== null)
    },
  }
}
export type TrailNavigation = ReturnType<typeof createTrailNavigation>

export function createNavigationSlot(lifetime: AbortSignal, element?: SVGElement) {
  let controller: AbortController | null = null
  let disposed = false
  let retryAt = 0
  const cancel = () => {
    controller?.abort()
    controller = null
    if (element) element.dataset.routePlanning = 'idle'
  }
  lifetime.addEventListener('abort', cancel, { once: true })
  return {
    cancel,
    dispose() {
      disposed = true
      cancel()
      lifetime.removeEventListener('abort', cancel)
    },
    get pending() {
      return controller !== null
    },
    run<T>(request: (signal: AbortSignal) => Promise<T>, apply: (result: T) => void) {
      cancel()
      if (lifetime.aborted || disposed) return
      const current = new AbortController()
      controller = current
      if (element) element.dataset.routePlanning = 'searching'
      void request(current.signal).then(result => {
        if (lifetime.aborted || current.signal.aborted || controller !== current) return
        controller = null
        if (element) element.dataset.routePlanning = 'idle'
        retryAt = performance.now() + 250
        apply(result)
      })
    },
    repair(
      navigation: TrailNavigation,
      id: string,
      from: Point,
      waypoints: readonly Point[],
      radius: number,
      obstacles: readonly CollisionActor[],
      bounds: SpacingBounds,
      waitingForAnimal: string | null,
      apply: (points: Point[]) => void,
      current: () => Point,
    ) {
      const goal = waypoints[0]
      if (controller || performance.now() < retryAt || !goal) return
      const bodies = obstacles.filter(body => body.id !== id)
      if (canMove(from, goal, radius, bodies)) return
      const origin = { ...from }
      this.run(
        signal =>
          navigation.repair(
            id,
            origin,
            waypoints,
            radius,
            bodies,
            bounds,
            waitingForAnimal,
            signal,
          ),
        points => {
          if (
            points?.length &&
            Math.hypot(current().x - origin.x, current().y - origin.y) <= radius * 2
          )
            apply(points)
        },
      )
    },
  }
}
