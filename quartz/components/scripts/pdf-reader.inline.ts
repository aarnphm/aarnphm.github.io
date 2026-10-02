import { currentNavSignal } from './nav-lifecycle'

type PdfReaderModule = typeof import('../pdf-reader/app')

// Every page loads this chunk, so the reader ships as the LazyScripts bundle `pdf-reader.js`
// (quartz.config.ts), fetched only where a reader mounts.
document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('[data-pdf-reader-mount]')
  if (!root) return
  const signal = currentNavSignal()
  let unmount: (() => void) | undefined
  window.addCleanup(() => unmount?.())
  void (import(new URL('pdf-reader.js', import.meta.url).href) as Promise<PdfReaderModule>)
    .then(({ mountPdfReader }) => {
      if (signal.aborted || !root.isConnected) return
      unmount = mountPdfReader(root, signal)
    })
    .catch(error => {
      console.error(error)
      root
        .querySelector('[role="status"]')
        ?.replaceChildren('The reader failed to load. Reload to retry.')
    })
})
