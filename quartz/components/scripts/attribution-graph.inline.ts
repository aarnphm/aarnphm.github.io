import {
  AG_BAR_SPAN,
  AG_EDGES,
  AG_NODES,
  AG_REST,
  type AgState,
  edgeKey,
  evaluate,
  fixed,
  pruned,
  strokeFor,
} from '../../util/attribution-graph'

const byId = new Map(AG_NODES.map(n => [n.id, n]))

const setupAttributionGraph = () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-attribution-graph]')) {
    if (root.dataset.agBound === 'true') continue
    root.dataset.agBound = 'true'

    const prune = root.querySelector<HTMLInputElement>('[data-ag-prune]')
    const pruneValue = root.querySelector<HTMLElement>('[data-ag-prune-value]')
    const restore = root.querySelector<HTMLButtonElement>('[data-ag-restore]')
    const readNode = root.querySelector<HTMLElement>('[data-ag-read-node]')
    const readAct = root.querySelector<HTMLElement>('[data-ag-read-act]')
    const rows = root.querySelector<HTMLElement>('[data-ag-rows]')
    const nodes = Array.from(root.querySelectorAll<SVGGElement>('[data-ag-node]'))
    const edges = Array.from(root.querySelectorAll<SVGPathElement>('[data-ag-edge]'))
    if (!prune || !rows) continue

    const base = evaluate(new Set())
    const ablated = new Set<string>()
    let state: AgState = base
    let selected = root.dataset.agSelected ?? 'logit'

    const paint = () => {
      const threshold = Number(prune.value)
      for (const path of edges) {
        const a = state.attr.get(path.dataset.agEdge ?? '') ?? 0
        path.setAttribute('stroke-width', String(strokeFor(a)))
        path.dataset.agPruned = String(a < threshold || a === 0)
        path.dataset.agIn = String(path.dataset.agTo === selected)
      }
      for (const g of nodes) {
        const id = g.dataset.agNode ?? ''
        const a = state.act.get(id) ?? 0
        const n = byId.get(id)
        g.dataset.agAblated = String(ablated.has(id))
        g.dataset.agSelected = String(id === selected)
        if (g.getAttribute('aria-pressed') != null)
          g.setAttribute('aria-pressed', String(ablated.has(id)))
        if (n) g.setAttribute('aria-label', `${n.label}, ${n.note}, activation ${fixed(a)}`)
        const bar = g.querySelector<SVGRectElement>('[data-ag-bar]')
        if (bar) bar.setAttribute('width', String(AG_BAR_SPAN * Math.min(1, a)))
      }
      const stats = pruned(state, threshold)
      prune.setAttribute('aria-valuetext', `threshold ${fixed(threshold)}`)
      if (pruneValue)
        pruneValue.textContent = `${fixed(threshold)} · ${stats.count}/${AG_EDGES.length} edges · ${Math.round(stats.share * 100)}% kept`
      if (restore) restore.hidden = ablated.size === 0
      root.dataset.agSelected = selected
      readout()
    }

    const readout = () => {
      const n = byId.get(selected)
      if (!n) return
      if (readNode) readNode.textContent = `${n.label} ${n.note}`
      const a = state.act.get(n.id) ?? 0
      const a0 = base.act.get(n.id) ?? 0
      if (readAct) readAct.textContent = a === a0 ? fixed(a) : `${fixed(a0)} → ${fixed(a)}`
      const list: { label: string; value: number }[] = []
      for (const e of AG_EDGES)
        if (e.to === n.id)
          list.push({
            label: byId.get(e.from)?.label ?? e.from,
            value: state.attr.get(edgeKey(e)) ?? 0,
          })
      const rest = AG_REST[n.id]
      if (rest != null)
        list.push({ label: n.id === 'logit' ? 'other paths' : 'error node', value: rest })
      if (n.kind === 'token') list.push({ label: 'embedding', value: ablated.has(n.id) ? 0 : 1 })
      rows.replaceChildren(
        ...list.map(row => {
          const div = document.createElement('div')
          const dt = document.createElement('dt')
          dt.textContent = row.label
          const dd = document.createElement('dd')
          dd.textContent = fixed(row.value)
          div.append(dt, dd)
          return div
        }),
      )
    }

    const select = (id: string) => {
      if (id === selected) return
      selected = id
      paint()
    }

    const toggle = (g: SVGGElement) => {
      const id = g.dataset.agNode ?? ''
      if (!id || g.getAttribute('aria-disabled') === 'true') return
      if (ablated.has(id)) ablated.delete(id)
      else ablated.add(id)
      state = evaluate(ablated)
      selected = id
      paint()
    }

    const onOver = (event: Event) => {
      const g = (event.target as Element).closest<SVGGElement>('[data-ag-node]')
      if (g?.dataset.agNode) select(g.dataset.agNode)
    }
    const onClick = (event: Event) => {
      const g = (event.target as Element).closest<SVGGElement>('[data-ag-node]')
      if (g) toggle(g)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Enter' && event.key !== ' ') return
      const g = (event.target as Element).closest<SVGGElement>('[data-ag-node]')
      if (!g) return
      event.preventDefault()
      toggle(g)
    }
    const onRestore = () => {
      ablated.clear()
      state = base
      paint()
    }

    root.addEventListener('pointerover', onOver)
    root.addEventListener('focusin', onOver)
    root.addEventListener('click', onClick)
    root.addEventListener('keydown', onKey)
    prune.addEventListener('input', paint)
    restore?.addEventListener('click', onRestore)
    paint()

    window.addCleanup(() => {
      root.removeEventListener('pointerover', onOver)
      root.removeEventListener('focusin', onOver)
      root.removeEventListener('click', onClick)
      root.removeEventListener('keydown', onKey)
      prune.removeEventListener('input', paint)
      restore?.removeEventListener('click', onRestore)
      delete root.dataset.agBound
    })
  }
}

document.addEventListener('nav', setupAttributionGraph)
