export interface CanvasLifecycle {
  controller: AbortController
  cleanups: Set<() => void>
}

export const canvasLifecycles = new WeakMap<HTMLElement, CanvasLifecycle>()
export const activeCanvasContainers = new Set<HTMLElement>()

export function createCanvasLifecycle(): CanvasLifecycle {
  return { controller: new AbortController(), cleanups: new Set() }
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

  lifecycle.controller.abort()
  lifecycle.cleanups.forEach(cleanup => cleanup())
  lifecycle.cleanups.clear()
  canvasLifecycles.delete(container)
  activeCanvasContainers.delete(container)
}

export function isCanvasActive(container: HTMLElement, lifecycle: CanvasLifecycle): boolean {
  return (
    container.isConnected &&
    !lifecycle.controller.signal.aborted &&
    canvasLifecycles.get(container) === lifecycle
  )
}

export function cleanupCanvases(root?: ParentNode): void {
  for (const container of activeCanvasContainers) {
    if (root instanceof Node && !root.contains(container)) continue
    const lifecycle = canvasLifecycles.get(container)
    if (lifecycle) disposeCanvas(container, lifecycle)
  }
}
