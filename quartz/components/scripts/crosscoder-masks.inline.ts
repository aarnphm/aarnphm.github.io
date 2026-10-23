import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

type ClientState = {
  cells: string
  used: string
  lit: number[]
  groups: string
  pairs: string
  reads: string
  writes: string
}

const CCM_ROOT_SELECTOR = '[data-crosscoder-masks]'
const OUTPUTS = ['groups', 'pairs', 'reads', 'writes'] as const

const setupCrosscoderMasks = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-ccm-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, ClientState>
  const dims = root.getAttribute('data-ccm-dims')
  const cells = Array.from(root.querySelectorAll<SVGRectElement>('[data-ccm-cell]'))
  const stations = Array.from(root.querySelectorAll<SVGCircleElement>('[data-ccm-station]'))

  const checked = (name: string) =>
    root.querySelector<HTMLInputElement>(`[data-ccm-input="${name}"]:checked`)?.value

  const apply = () => {
    const variant =
      dims === 'coder' ? checked('coder') : `${checked('causality')}-${checked('locality')}`
    const read = checked('read')
    const state = states[`${variant}:${read}`]
    if (!variant || !state) return
    root.setAttribute('data-ccm-variant', variant)
    for (const el of root.querySelectorAll('[data-ccm-rule]'))
      el.setAttribute('data-on', String(el.getAttribute('data-ccm-rule') === variant))
    for (const group of root.querySelectorAll('[data-ccm-variant-group]')) {
      const on = group.getAttribute('data-ccm-variant-group') === variant
      group.setAttribute('data-on', String(on))
      for (const el of group.querySelectorAll('[data-ccm-box], [data-ccm-edges]')) {
        const k = Number(el.getAttribute('data-ccm-box') ?? el.getAttribute('data-ccm-edges'))
        el.setAttribute('data-lit', String(on && state.lit.includes(k)))
      }
    }
    for (const cell of cells)
      cell.setAttribute('data-state', state.cells[Number(cell.getAttribute('data-ccm-cell'))])
    stations.forEach((el, l) => el.setAttribute('data-used', state.used[l]))
    for (const key of OUTPUTS) {
      const out = root.querySelector(`[data-ccm-out="${key}"]`)
      if (out) out.textContent = state[key]
    }
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(CCM_ROOT_SELECTOR, root => {
  const cleanup = setupCrosscoderMasks(root)
  if (cleanup) window.addCleanup?.(cleanup)
  window.addCleanup?.(observeSvgScale(root.querySelectorAll<SVGSVGElement>('.ccm-svg'), '--ccm-u'))
})
