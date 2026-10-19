export {}

const ROOT_SELECTOR = '[data-rel-alg-tree]'

// Every node's relation is rendered on the server; the script only marks the chosen node, its
// subtree and its panel. The wide tree and the narrow outline each hold a radio group for the
// same nodes, so a choice in one is copied to the other.
const setupRelAlgTree = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-rat-subtrees')
  if (!raw) return null
  const subtrees = JSON.parse(raw) as Record<string, string[]>
  const nodes = Array.from(root.querySelectorAll<SVGGElement>('[data-rat-node]'))
  const edges = Array.from(root.querySelectorAll<SVGLineElement>('[data-rat-edge]'))
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[data-rat-panel]'))
  const inputs = Array.from(root.querySelectorAll<HTMLInputElement>('[data-rat-input]'))

  const apply = (event?: Event) => {
    const target = event?.target
    if (target instanceof HTMLInputElement && target.hasAttribute('data-rat-input'))
      for (const input of inputs) input.checked = input.value === target.value
    const sel = root.querySelector<HTMLInputElement>('[data-rat-input]:checked')?.value
    const sub = sel ? subtrees[sel] : undefined
    if (!sel || !sub) return
    root.setAttribute('data-rat-sel', sel)
    for (const n of nodes) {
      const id = n.getAttribute('data-rat-node') ?? ''
      n.setAttribute('data-state', id === sel ? 'lit' : sub.includes(id) ? 'done' : 'idle')
    }
    for (const e of edges) {
      const child = e.getAttribute('data-rat-edge') ?? ''
      e.setAttribute('data-state', child !== sel && sub.includes(child) ? 'on' : 'off')
    }
    for (const p of panels) p.hidden = p.getAttribute('data-rat-panel') !== sel
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ROOT_SELECTOR)) {
    const cleanup = setupRelAlgTree(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
