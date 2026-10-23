import { registerMdxContent } from './mdx-content'

const ROOT = '[data-param-decomposition]'

type Config = { columns: number[][]; nonzero: number }

const setup = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-apd-state')
  if (!raw) return null
  const configs = JSON.parse(raw) as Record<string, Config>
  const out = (key: string) => root.querySelector<HTMLElement>(`[data-apd-out="${key}"]`)

  const render = (n: string) => {
    root.setAttribute('data-apd-n', n)
    for (const set of root.querySelectorAll<HTMLElement>('[data-apd-colset]')) {
      if (set.getAttribute('data-apd-colset') === n) set.removeAttribute('data-apd-off')
      else set.setAttribute('data-apd-off', 'true')
    }
    const cfg = configs[n]
    const nOut = out('n')
    if (nOut) nOut.textContent = n
    const fill = out('fill')
    if (fill && cfg) fill.textContent = `${Math.round(cfg.nonzero * 100)}%`
  }

  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement | null
    if (!input?.matches('[data-apd-n-input]')) return
    render(input.value)
  }

  root.addEventListener('change', onChange)
  render(root.getAttribute('data-apd-n') ?? '6')
  return () => root.removeEventListener('change', onChange)
}

registerMdxContent(ROOT, root => {
  const cleanup = setup(root)
  if (cleanup) window.addCleanup(cleanup)
})
