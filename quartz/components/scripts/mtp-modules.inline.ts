export {}

const ROOT = '[data-mtp-modules]'

const setup = (root: HTMLElement): (() => void) | null => {
  const out = (key: string) => root.querySelector<HTMLElement>(`[data-mtp-out="${key}"]`)

  const render = (depth: number) => {
    root.setAttribute('data-mtp-depth', String(depth))
    for (const col of root.querySelectorAll<HTMLElement>('[data-mtp-col]')) {
      const id = col.getAttribute('data-mtp-col')
      if (id === 'main') continue
      if (Number(id) > depth) col.setAttribute('data-mtp-off', 'true')
      else col.removeAttribute('data-mtp-off')
    }
    const chain = out('chain')
    if (chain) {
      const parts = ['main', ...Array.from({ length: depth }, (_, i) => `MTP ${i + 1}`)]
      chain.textContent = parts.join(' → ')
    }
    const count = out('count')
    if (count) count.textContent = String(depth)
  }

  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement | null
    if (!input?.matches('[data-mtp-depth-input]')) return
    render(Number(input.value))
  }

  root.addEventListener('change', onChange)
  render(Number(root.getAttribute('data-mtp-depth') ?? '2'))
  return () => root.removeEventListener('change', onChange)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ROOT)) {
    const cleanup = setup(root)
    if (cleanup) window.addCleanup(cleanup)
  }
})
