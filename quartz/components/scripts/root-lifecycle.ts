import { cleanupEscapeHandlers } from './escape-handler'
import { currentNavSignal } from './nav-lifecycle'

interface RootLifecycle {
  navSignal: AbortSignal
  controller: AbortController
  dispose: () => void
}

export interface RootLifecycleRegistry {
  lifecycles: Map<Node, RootLifecycle>
}

function getRootRegistry(): RootLifecycleRegistry {
  const registry = window.quartzRootLifecycles ?? { lifecycles: new Map<Node, RootLifecycle>() }
  window.quartzRootLifecycles = registry
  return registry
}

if (typeof window !== 'undefined') {
  getRootRegistry()
}

export function rootNavSignal(root: Node): AbortSignal {
  const registry = getRootRegistry()
  const navSignal = currentNavSignal()
  const existing = registry.lifecycles.get(root)
  if (existing?.navSignal === navSignal && !existing.controller.signal.aborted) {
    return existing.controller.signal
  }
  existing?.dispose()

  const controller = new AbortController()
  let disposed = false
  let lifecycle: RootLifecycle
  const dispose = (): void => {
    if (disposed) return
    disposed = true
    controller.abort()
    navSignal.removeEventListener('abort', dispose)
    if (registry.lifecycles.get(root) === lifecycle) registry.lifecycles.delete(root)
  }
  lifecycle = { navSignal, controller, dispose }
  registry.lifecycles.set(root, lifecycle)
  navSignal.addEventListener('abort', dispose, { once: true })
  return controller.signal
}

export function cleanupRootLifecycles(root?: Node): void {
  const registry = getRootRegistry()
  for (const [owner, lifecycle] of registry.lifecycles) {
    if (root && owner !== root && !root.contains(owner)) continue
    lifecycle.dispose()
  }
}

export function cleanupHydratedRoot(root: ParentNode & Node): void {
  cleanupEscapeHandlers(root)
  cleanupRootLifecycles(root)
  window.quartzCanvas?.cleanup(root)
  window.quartzPdfEmbeds?.cleanup(root)
}
