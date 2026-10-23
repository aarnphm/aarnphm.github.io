import { registerMdxContent } from './mdx-content'

type ClientState = {
  mask: string
  rows: number[]
  rowMass: string[]
  kept: string
  layers: string
  mean: string
  worst: string
}

const KVB_ROOT_SELECTOR = '[data-kv-budget-allocation]'
const OUTPUTS = ['kept', 'layers', 'mean', 'worst'] as const

const setupKVBudgetAllocation = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-kvb-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, ClientState>
  const cells = Array.from(root.querySelectorAll<SVGRectElement>('[data-kvb-cell]'))
  const rowKept = Array.from(root.querySelectorAll<SVGTextElement>('[data-kvb-row]'))
  const rowMass = Array.from(root.querySelectorAll<SVGTextElement>('[data-kvb-row-mass]'))

  const checked = (selector: string) =>
    root.querySelector<HTMLInputElement>(`${selector}:checked`)?.value

  const apply = () => {
    const policy = checked('[data-kvb-policy-input]')
    const budget = checked('[data-kvb-budget-input]')
    const state = policy && budget ? states[`${policy}:${budget}`] : undefined
    if (!policy || !state) return
    root.setAttribute('data-kvb-policy', policy)
    for (const cell of cells) {
      const i = Number(cell.getAttribute('data-kvb-cell'))
      cell.setAttribute('data-kvb-kept', state.mask[i] === '1' ? 'true' : 'false')
    }
    rowKept.forEach((el, r) => (el.textContent = String(state.rows[r])))
    rowMass.forEach((el, r) => (el.textContent = state.rowMass[r]))
    for (const key of OUTPUTS) {
      const out = root.querySelector(`[data-kvb-out="${key}"]`)
      if (out) out.textContent = state[key]
    }
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(KVB_ROOT_SELECTOR, root => {
  const cleanup = setupKVBudgetAllocation(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
