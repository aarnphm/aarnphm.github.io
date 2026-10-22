import {
  activeCanvasContainers,
  canvasLifecycles,
  cleanupCanvases,
  createCanvasLifecycle,
  disposeCanvas,
} from './canvas/lifecycle'
import { renderCanvas } from './canvas/viewer'

window.quartzCanvas = { cleanup: cleanupCanvases }

function initializeCanvases(root: ParentNode) {
  root
    .querySelectorAll<HTMLElement>(['.canvas-container', '.canvas-embed-container'].join(','))
    .forEach(container => {
      if (canvasLifecycles.has(container)) return

      const lifecycle = createCanvasLifecycle()
      canvasLifecycles.set(container, lifecycle)
      activeCanvasContainers.add(container)
      void renderCanvas(container, lifecycle)
    })
}

document.addEventListener('nav', () => {
  window.addCleanup(cleanupCanvases)
  activeCanvasContainers.forEach(container => {
    if (container.isConnected) return
    const lifecycle = canvasLifecycles.get(container)
    if (lifecycle) disposeCanvas(container, lifecycle)
  })
  initializeCanvases(document)
  const onContentMounted = (event: CustomEventMap['contentdecrypted']) => {
    initializeCanvases(event.detail.content)
  }
  document.addEventListener('contentdecrypted', onContentMounted)
  window.addCleanup(() => document.removeEventListener('contentdecrypted', onContentMounted))
})
