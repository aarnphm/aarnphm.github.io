import { currentNavSignal } from './nav-lifecycle'

// Every page loads this chunk, so the reader itself stays behind a lazy import.
document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('[data-pdf-reader-mount]')
  if (!root) return
  const signal = currentNavSignal()
  let unmount: (() => void) | undefined
  window.addCleanup(() => unmount?.())
  void import('../pdf-reader/app')
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
