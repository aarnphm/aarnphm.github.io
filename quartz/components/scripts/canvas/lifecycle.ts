export interface CanvasLifecycle {
  controller: AbortController
  cleanups: Set<() => void>
}

const canvasLifecycles = new Map<HTMLElement, CanvasLifecycle>()

export function registerCanvas(container: HTMLElement): CanvasLifecycle | undefined {
  if (canvasLifecycles.has(container)) return
  const lifecycle = { controller: new AbortController(), cleanups: new Set<() => void>() }
  canvasLifecycles.set(container, lifecycle)
  return lifecycle
}

export function addCanvasCleanup(lifecycle: CanvasLifecycle, cleanup: () => void): void {
  if (lifecycle.controller.signal.aborted) {
    cleanup()
    return
  }
  lifecycle.cleanups.add(cleanup)
}

export function disposeCanvas(container: HTMLElement, lifecycle: CanvasLifecycle): void {
  if (canvasLifecycles.get(container) !== lifecycle) return

  // Release this generation before abort handlers or cleanup callbacks can reenter disposal.
  canvasLifecycles.delete(container)
  lifecycle.controller.abort()
  lifecycle.cleanups.forEach(cleanup => cleanup())
  lifecycle.cleanups.clear()
}

export function isCanvasActive(container: HTMLElement, lifecycle: CanvasLifecycle): boolean {
  return (
    container.isConnected &&
    !lifecycle.controller.signal.aborted &&
    canvasLifecycles.get(container) === lifecycle
  )
}

export function cleanupCanvases(root?: ParentNode): void {
  for (const [container, lifecycle] of canvasLifecycles) {
    if (root instanceof Node && !root.contains(container)) continue
    disposeCanvas(container, lifecycle)
  }
}

export function cleanupDetachedCanvases(): void {
  for (const [container, lifecycle] of canvasLifecycles) {
    if (!container.isConnected) disposeCanvas(container, lifecycle)
  }
}
