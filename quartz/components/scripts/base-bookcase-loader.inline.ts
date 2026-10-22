import { type Cover, retainCovers } from './base-bookcase-artwork'
import { currentNavSignal } from './nav-lifecycle'
import { rootNavSignal } from './root-lifecycle'

type BookcaseModule = typeof import('./base-bookcase.inline')
let modulePromise: Promise<BookcaseModule> | undefined
let failedLoads = 0

function loadBookcase(): Promise<BookcaseModule> {
  if (!modulePromise) {
    const source = new URL('base-bookcase.js', import.meta.url)
    // Browsers cache failed module imports for the document, including network failures.
    if (failedLoads) source.searchParams.set('retry', String(failedLoads))
    modulePromise = import(source.href).catch(cause => {
      modulePromise = undefined
      failedLoads++
      throw cause
    })
  }
  return modulePromise
}

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  const observed = new Set<HTMLElement>()
  const entering = new Set<HTMLElement>()
  const covers = new Map<string, Cover>()
  let loading = false

  const activate = async (): Promise<void> => {
    if (loading || signal.aborted || entering.size === 0) return
    loading = true
    try {
      const runtime = await loadBookcase()
      // Navigation can finish while the module is in flight. Its static imports must not mount.
      if (signal.aborted) return
      for (const root of entering) {
        if (!root.isConnected || root.closest('[hidden]')) continue
        runtime.initializeBookcase(root, covers)
        observer.unobserve(root)
        entering.delete(root)
      }
    } catch (cause) {
      if (!signal.aborted) console.warn('Bookcase could not load. Enter the scene to retry.', cause)
    } finally {
      loading = false
    }
  }

  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!(entry.target instanceof HTMLElement)) continue
      if (entry.isIntersecting && entry.intersectionRect.width && entry.intersectionRect.height)
        entering.add(entry.target)
      else entering.delete(entry.target)
    }
    void activate()
  })

  const discover = (): void => {
    for (const root of observed) {
      if (root.isConnected) continue
      observer.unobserve(root)
      entering.delete(root)
      observed.delete(root)
    }
    for (const root of document.querySelectorAll<HTMLElement>('.bookcase')) {
      // Keep emitted edition bindings before a client query can replace an unopened scene.
      retainCovers(root, covers)
      if (observed.has(root)) continue
      observed.add(root)
      observer.observe(root)
      const rootSignal = rootNavSignal(root)
      rootSignal.addEventListener(
        'abort',
        () => {
          observer.unobserve(root)
          observed.delete(root)
          entering.delete(root)
        },
        { once: true },
      )
      const enter = () => {
        if (root.dataset.bookcaseReady !== undefined || root.closest('[hidden]')) return
        entering.add(root)
        void activate()
      }
      root.addEventListener('pointerenter', enter, { signal: rootSignal })
      root.addEventListener('pointerdown', enter, { signal: rootSignal })
      root.addEventListener('focusin', enter, { signal: rootSignal })
      root.addEventListener('keydown', enter, { signal: rootSignal })
    }
  }

  discover()
  document.addEventListener('base-content-changed', discover, { signal })
  document.addEventListener('contentdecrypted', discover, { signal })
  window.addCleanup(() => {
    observer.disconnect()
    observed.clear()
    entering.clear()
    covers.clear()
  })
})

export default ''
