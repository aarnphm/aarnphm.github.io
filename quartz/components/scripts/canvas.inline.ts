import { cleanupCanvases, cleanupDetachedCanvases, registerCanvas } from './canvas/lifecycle'
import { renderCanvas } from './canvas/viewer'

window.quartzCanvas = { cleanup: cleanupCanvases }

function initializeCanvases(root: ParentNode) {
  root
    .querySelectorAll<HTMLElement>(['.canvas-container', '.canvas-embed-container'].join(','))
    .forEach(container => {
      const lifecycle = registerCanvas(container)
      if (lifecycle) void renderCanvas(container, lifecycle)
    })
}

document.addEventListener('nav', () => {
  window.addCleanup(cleanupCanvases)
  cleanupDetachedCanvases()
  initializeCanvases(document)
  const onContentMounted = (event: CustomEventMap['contentdecrypted']) => {
    initializeCanvases(event.detail.content)
  }
  document.addEventListener('contentdecrypted', onContentMounted)
  window.addCleanup(() => document.removeEventListener('contentdecrypted', onContentMounted))
})
