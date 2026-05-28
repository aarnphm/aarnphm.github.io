export {}

type AcirCircuit = 'qk' | 'ov' | 'none'

type AcirState = {
  root: HTMLElement
  pills: HTMLButtonElement[]
  graphs: SVGElement[]
  rankToggle: HTMLInputElement | null
  ovSplit: SVGGElement | null
  ovRank: SVGGElement | null
  current: AcirCircuit
  rankView: boolean
}

const ACIR_ROOT_SELECTOR = '[data-attention-circuits]'

function acirSetCircuit(state: AcirState, next: AcirCircuit) {
  state.current = next
  for (const pill of state.pills) {
    const pressed =
      pill.dataset.acirPill === next || (next === 'none' && pill.dataset.acirPill === 'none')
    pill.setAttribute('aria-pressed', pressed ? 'true' : 'false')
  }
  for (const graph of state.graphs) {
    graph.classList.remove('is-highlight-qk', 'is-highlight-ov')
    if (next === 'qk') graph.classList.add('is-highlight-qk')
    else if (next === 'ov') graph.classList.add('is-highlight-ov')
  }
  state.root.setAttribute('data-acir-highlight', next)
}

function acirSetRankView(state: AcirState, on: boolean) {
  state.rankView = on
  if (state.rankToggle) state.rankToggle.checked = on
  if (state.ovSplit) {
    if (on) state.ovSplit.setAttribute('hidden', '')
    else state.ovSplit.removeAttribute('hidden')
  }
  if (state.ovRank) {
    if (on) state.ovRank.removeAttribute('hidden')
    else state.ovRank.setAttribute('hidden', '')
  }
  state.root.setAttribute('data-acir-rank', on ? 'true' : 'false')
}

function acirSetup(root: HTMLElement): (() => void) | null {
  if (root.dataset.acirBound === 'true') return null
  const pills = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-acir-pill]'))
  if (pills.length === 0) return null
  const graphs = Array.from(root.querySelectorAll<SVGElement>('.acir-graph'))
  const rankToggle = root.querySelector<HTMLInputElement>('[data-acir-rank-toggle]')
  const ovSplit = root.querySelector<SVGGElement>('[data-acir-ov-split]')
  const ovRank = root.querySelector<SVGGElement>('[data-acir-ov-rank]')

  const state: AcirState = {
    root,
    pills,
    graphs,
    rankToggle,
    ovSplit,
    ovRank,
    current: 'none',
    rankView: false,
  }

  acirSetCircuit(state, 'none')
  acirSetRankView(state, false)
  root.dataset.acirBound = 'true'

  const pillHandlers = pills.map(pill => {
    const handler = () => {
      const next = (pill.dataset.acirPill as AcirCircuit | undefined) ?? 'none'
      if (next === state.current) {
        acirSetCircuit(state, 'none')
      } else {
        acirSetCircuit(state, next)
      }
    }
    pill.addEventListener('click', handler)
    return { pill, handler }
  })

  const hoverEnter = (circuit: AcirCircuit) => () => {
    if (state.current !== 'none') return
    for (const graph of state.graphs) {
      graph.classList.remove('is-highlight-qk', 'is-highlight-ov')
      if (circuit === 'qk') graph.classList.add('is-highlight-qk')
      else if (circuit === 'ov') graph.classList.add('is-highlight-ov')
    }
  }
  const hoverLeave = () => {
    if (state.current !== 'none') return
    for (const graph of state.graphs) {
      graph.classList.remove('is-highlight-qk', 'is-highlight-ov')
    }
  }

  const qkPill = pills.find(p => p.dataset.acirPill === 'qk')
  const ovPill = pills.find(p => p.dataset.acirPill === 'ov')
  const qkEnter = hoverEnter('qk')
  const ovEnter = hoverEnter('ov')

  qkPill?.addEventListener('mouseenter', qkEnter)
  qkPill?.addEventListener('focus', qkEnter)
  qkPill?.addEventListener('mouseleave', hoverLeave)
  qkPill?.addEventListener('blur', hoverLeave)
  ovPill?.addEventListener('mouseenter', ovEnter)
  ovPill?.addEventListener('focus', ovEnter)
  ovPill?.addEventListener('mouseleave', hoverLeave)
  ovPill?.addEventListener('blur', hoverLeave)

  const rankHandler = () => {
    if (!rankToggle) return
    acirSetRankView(state, rankToggle.checked)
  }
  rankToggle?.addEventListener('change', rankHandler)

  return () => {
    for (const { pill, handler } of pillHandlers) {
      pill.removeEventListener('click', handler)
    }
    qkPill?.removeEventListener('mouseenter', qkEnter)
    qkPill?.removeEventListener('focus', qkEnter)
    qkPill?.removeEventListener('mouseleave', hoverLeave)
    qkPill?.removeEventListener('blur', hoverLeave)
    ovPill?.removeEventListener('mouseenter', ovEnter)
    ovPill?.removeEventListener('focus', ovEnter)
    ovPill?.removeEventListener('mouseleave', hoverLeave)
    ovPill?.removeEventListener('blur', hoverLeave)
    rankToggle?.removeEventListener('change', rankHandler)
    delete root.dataset.acirBound
  }
}

document.addEventListener('nav', () => {
  const roots = document.querySelectorAll<HTMLElement>(ACIR_ROOT_SELECTOR)
  for (const root of roots) {
    const cleanup = acirSetup(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
