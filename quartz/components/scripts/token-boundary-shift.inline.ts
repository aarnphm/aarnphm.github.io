export {}

type ClientState = {
  forced: string
  mass: string
  paul: string
  kl: string
  tv: string
  tokens: string[]
}

const OUTPUTS = ['forced', 'mass', 'paul', 'kl', 'tv'] as const

const setupTokenBoundaryShift = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-tbs-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, ClientState>
  const groups = Array.from(root.querySelectorAll<SVGGElement>('[data-tbs-boundary]'))
  const tokens = Array.from(root.querySelectorAll<HTMLElement>('[data-tbs-token]'))

  const apply = () => {
    const id = root.querySelector<HTMLInputElement>('[data-tbs-input]:checked')?.value
    const state = id ? states[id] : undefined
    if (!id || !state) return
    for (const g of groups)
      g.setAttribute('data-on', g.getAttribute('data-tbs-boundary') === id ? 'true' : 'false')
    tokens.forEach((td, i) => (td.textContent = state.tokens[i]))
    for (const key of OUTPUTS) {
      const out = root.querySelector(`[data-tbs-out="${key}"]`)
      if (out) out.textContent = state[key]
    }
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-token-boundary-shift]')) {
    const cleanup = setupTokenBoundaryShift(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
