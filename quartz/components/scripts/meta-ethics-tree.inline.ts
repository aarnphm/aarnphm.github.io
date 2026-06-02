export {}

const mepSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-meta-ethics]')
  for (const root of roots) {
    if (root.dataset.mepBound === 'true') continue
    root.dataset.mepBound = 'true'

    const nodes = Array.from(root.querySelectorAll<SVGGElement>('[data-mep-node]'))
    const edges = Array.from(root.querySelectorAll<SVGGElement>('[data-mep-edge]'))
    const details = Array.from(root.querySelectorAll<HTMLElement>('[data-mep-detail]'))
    const fallback = root.getAttribute('data-mep-default') ?? ''

    const select = (id: string) => {
      const target = root.querySelector<SVGGElement>(`[data-mep-node="${id}"]`)
      if (!target) return
      const pathNodes = new Set(
        (target.getAttribute('data-mep-path') ?? '').split(',').filter(Boolean),
      )
      const pathEdges = new Set(
        (target.getAttribute('data-mep-edges') ?? '').split(',').filter(Boolean),
      )

      for (const node of nodes) {
        const nid = node.getAttribute('data-mep-node') ?? ''
        node.classList.toggle('is-active', nid === id)
        node.classList.toggle('is-path', nid !== id && pathNodes.has(nid))
      }
      for (const edge of edges) {
        edge.classList.toggle('is-path', pathEdges.has(edge.getAttribute('data-mep-edge') ?? ''))
      }
      for (const detail of details) {
        detail.classList.toggle('is-active', detail.getAttribute('data-mep-detail') === id)
      }
    }

    const handlers: Array<() => void> = []

    for (const node of nodes) {
      if (node.getAttribute('data-mep-selectable') !== 'true') continue
      const id = node.getAttribute('data-mep-node') ?? ''
      const onClick = () => select(id)
      const onKey = (ev: KeyboardEvent) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault()
          select(id)
        }
      }
      node.addEventListener('click', onClick)
      node.addEventListener('keydown', onKey)
      handlers.push(() => {
        node.removeEventListener('click', onClick)
        node.removeEventListener('keydown', onKey)
      })
    }

    const resetBtn = root.querySelector<HTMLButtonElement>('[data-mep-reset]')
    if (resetBtn) {
      const onReset = () => select(fallback)
      resetBtn.addEventListener('click', onReset)
      handlers.push(() => resetBtn.removeEventListener('click', onReset))
    }

    select(fallback)

    window.addCleanup(() => {
      for (const off of handlers) off()
      delete root.dataset.mepBound
    })
  }
}

document.addEventListener('nav', mepSetup)
