export {}

const FDF_ROOT_SELECTOR = '[data-flash-data-flow]'

const setupFlashDataFlow = (root: HTMLElement): (() => void) | null => {
  const tabs = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-fdf-tab]'))
  if (!tabs.length) return null

  const bound = tabs.map(btn => {
    const handler = () => {
      const mode = btn.getAttribute('data-fdf-tab')
      if (!mode || root.getAttribute('data-fdf-mode') === mode) return
      root.setAttribute('data-fdf-mode', mode)
      for (const other of tabs) {
        const active = other === btn
        other.setAttribute('aria-selected', active ? 'true' : 'false')
        other.setAttribute('tabindex', active ? '0' : '-1')
      }
    }
    btn.addEventListener('click', handler)
    return { btn, handler }
  })

  return () => {
    for (const { btn, handler } of bound) btn.removeEventListener('click', handler)
  }
}

document.addEventListener('nav', () => {
  const roots = document.querySelectorAll<HTMLElement>(FDF_ROOT_SELECTOR)
  for (const root of roots) {
    const cleanup = setupFlashDataFlow(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
