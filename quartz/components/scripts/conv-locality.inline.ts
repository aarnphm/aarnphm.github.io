type PoolState = { cells: string; max: string; rows: string; cols: string }
type PoolMeta = { n: number; skipped: number; w2: string }
type ConnState = { ids: string[]; params: string; reads: string; range: string; share: string }

const setupConvLocality = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-cvl-states')
  if (!raw) return null
  const view = root.getAttribute('data-cvl-view')
  const inputs = Array.from(root.querySelectorAll<HTMLElement>('[data-cvl-in]'))
  const checked = (selector: string, scope: ParentNode = root) =>
    scope.querySelector<HTMLInputElement>(`${selector}:checked`)?.value
  const out = (key: string, text: string) => {
    const el = root.querySelector(`[data-cvl-out="${key}"]`)
    if (el) el.textContent = text
  }

  // The server computes every state; the script only swaps attributes and text.
  const applyPooling = () => {
    const states = JSON.parse(raw) as Record<string, PoolState>
    const meta = JSON.parse(root.getAttribute('data-cvl-meta') ?? '{}') as Record<string, PoolMeta>
    return () => {
      const config = `${checked('[data-cvl-size]')}:${checked('[data-cvl-stride]')}`
      let active: HTMLElement | null = null
      for (const set of root.querySelectorAll<HTMLElement>('[data-cvl-config]')) {
        const on = set.getAttribute('data-cvl-config') === config
        set.hidden = !on
        if (on) active = set
      }
      const win = active ? checked('[data-cvl-window]', active) : undefined
      const s = states[`${config}:${win}`]
      const m = meta[config]
      if (!s || !m) return
      inputs.forEach((cell, k) => cell.setAttribute('data-cvl-state', s.cells[k]))
      out('w2', m.w2)
      out('span', `${s.rows}, ${s.cols}`)
      out('max', s.max)
      out('skipped', String(m.skipped))
    }
  }

  const applyConnectivity = () => {
    const states = JSON.parse(raw) as Record<string, ConnState>
    return () => {
      const mode = checked('[data-cvl-mode]')
      const s = states[`${mode}:${checked('[data-cvl-unit]')}`]
      if (!mode || !s) return
      root.setAttribute('data-cvl-mode', mode)
      inputs.forEach((cell, k) => {
        cell.textContent = s.ids[k]
        cell.setAttribute('data-cvl-state', s.ids[k] === '' ? '0' : 'w')
      })
      out('params', s.params)
      out('reads', s.reads)
      out('range', s.range)
      out('share', s.share)
    }
  }

  const apply = view === 'connectivity' ? applyConnectivity() : applyPooling()
  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-conv-locality]')) {
    const cleanup = setupConvLocality(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
