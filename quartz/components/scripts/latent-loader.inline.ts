// The latent instrument is server-rendered; this loads the runtime only on a latent page, as a
// lazy script under static/scripts/, so other pages never pay for it. Transformer resources are
// bundled as iife, where import.meta.url is empty, and LazyScripts leaves the name unhashed, so
// the path is spelled out.
type LatentModule = typeof import('./latent.client')
let modulePromise: Promise<LatentModule> | undefined
let failedLoads = 0

function loadLatent(): Promise<LatentModule> {
  if (!modulePromise) {
    const source = new URL('/static/scripts/latent.js', document.baseURI)
    if (failedLoads) source.searchParams.set('retry', String(failedLoads))
    modulePromise = import(source.href).catch(cause => {
      modulePromise = undefined
      failedLoads++
      throw cause
    })
  }
  return modulePromise
}

/** Put the server-rendered focus chunk under the reading line before the runtime arrives. */
function alignFocus(root: HTMLElement): void {
  if (new URL(window.location.href).searchParams.has('focus')) return
  const chunk = root.querySelector<HTMLElement>('.lat-chunk[data-focus]')
  if (!chunk) return
  const top = chunk.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.38 + 8
  if (top > 0) window.scrollTo({ top, behavior: 'auto' })
}

document.addEventListener('nav', () => {
  if (document.body.dataset.layout !== 'latent') return
  const root = document.querySelector<HTMLElement>('.latent[data-latent-base]')
  if (!root) return
  alignFocus(root)
  void loadLatent().then(runtime => {
    if (!root.isConnected) return
    runtime.mountLatent(root)
  })
})
