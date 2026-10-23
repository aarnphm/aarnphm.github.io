import { currentNavSignal } from './nav-lifecycle'

type Setup = (root: HTMLElement) => void

// Slide fragments, decrypted notes and canvas cards mount without a page navigation.
export function registerMdxContent(selector: string, setup: Setup): void {
  document.addEventListener('nav', () => {
    const mounted = new WeakSet<HTMLElement>()
    const signal = currentNavSignal()
    const discover = (scope: Document | HTMLElement) => {
      const roots = Array.from(scope.querySelectorAll<HTMLElement>(selector))
      if (scope instanceof HTMLElement && scope.matches(selector)) roots.unshift(scope)
      for (const root of roots) {
        const slide = root.closest('.slide')
        if (mounted.has(root) || (slide && !slide.classList.contains('active'))) continue
        mounted.add(root)
        setup(root)
      }
    }
    document.addEventListener('contentdecrypted', event => discover(event.detail.content), {
      signal,
    })
    document.addEventListener('slidechange', () => discover(document), { signal })
    discover(document)
  })
}
