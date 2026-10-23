import { registerMdxContent } from './mdx-content'

type ClientState = { nodes: string[]; mask: string; out: Record<string, string> }

const SDT_ROOT_SELECTOR = '[data-speculative-draft-tree]'

const setupSpeculativeDraftTree = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-sdt-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, ClientState>
  const checked = (selector: string) =>
    root.querySelector<HTMLInputElement>(`${selector}:checked`)?.value

  const apply = () => {
    const method = checked('[data-sdt-method-input]')
    const step = checked('[data-sdt-step-input]')
    const key = `${method}:${step}`
    const state = states[key]
    if (!method || !step || !state) return
    root.setAttribute('data-sdt-method', method)
    root.setAttribute('data-sdt-step', step)
    for (const el of root.querySelectorAll<HTMLElement>('[data-sdt-key]'))
      el.hidden = el.getAttribute('data-sdt-key') !== key
    for (const el of root.querySelectorAll<HTMLElement>('[data-sdt-m]'))
      el.hidden = el.getAttribute('data-sdt-m') !== method
    const panel = root.querySelector(`.sdt-panels[data-sdt-m="${method}"]`)
    if (!panel) return
    for (const el of panel.querySelectorAll('[data-sdt-node]'))
      el.setAttribute('data-state', state.nodes[Number(el.getAttribute('data-sdt-node'))])
    for (const el of panel.querySelectorAll('[data-sdt-cell]'))
      el.setAttribute('data-state', state.mask[Number(el.getAttribute('data-sdt-cell'))])
    for (const el of root.querySelectorAll<HTMLElement>('[data-sdt-out]'))
      el.textContent = state.out[el.getAttribute('data-sdt-out') ?? ''] ?? ''
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(SDT_ROOT_SELECTOR, root => {
  const cleanup = setupSpeculativeDraftTree(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
