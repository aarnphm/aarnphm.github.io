export {}

type Step = {
  phase: string
  labels: string[]
  states: string[]
  senders: number[]
  sends: string
  sent: string
}
type Config = { steps: Step[]; chunk: string; total: string; count: number }

const ARS_ROOT_SELECTOR = '[data-all-reduce-steps]'

const setupAllReduceSteps = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-ars-states')
  if (!raw) return null
  const configs = JSON.parse(raw) as Record<string, Config>
  const out = (key: string) => root.querySelector<HTMLElement>(`[data-ars-out="${key}"]`)
  const nav = (key: string) => root.querySelector<HTMLButtonElement>(`[data-ars-nav="${key}"]`)

  let n = root.getAttribute('data-ars-n') ?? '4'
  let step = 0

  const render = () => {
    const config = configs[n]
    if (!config) return
    const state = config.steps[step]
    root.setAttribute('data-ars-n', n)
    root.setAttribute('data-ars-step', String(step))
    for (const svg of root.querySelectorAll<SVGSVGElement>('[data-ars-grid]'))
      svg.toggleAttribute('hidden', svg.getAttribute('data-ars-grid') !== n)
    const grid = root.querySelector(`[data-ars-grid="${n}"]`)
    const senders = new Set(state.senders)
    for (const cell of grid?.querySelectorAll<SVGGElement>('[data-ars-cell]') ?? []) {
      const k = Number(cell.getAttribute('data-ars-cell'))
      cell.setAttribute('data-ars-state', state.states[k])
      cell.setAttribute('data-ars-sender', senders.has(k) ? 'true' : 'false')
      const text = cell.querySelector('text')
      if (text) text.textContent = state.labels[k]
    }
    const stepOut = out('step')
    if (stepOut) stepOut.textContent = `step ${step} of ${config.count}`
    const values: Record<string, string> = {
      phase: state.phase,
      sends: state.sends,
      chunk: config.chunk,
      sent: state.sent,
      total: config.total,
    }
    for (const [key, value] of Object.entries(values)) {
      const el = out(key)
      if (el) el.textContent = value
    }
    const prev = nav('-1')
    const next = nav('1')
    const reset = nav('reset')
    if (prev) prev.disabled = step === 0
    if (reset) reset.disabled = step === 0
    if (next) next.disabled = step === config.count
  }

  const onClick = (event: MouseEvent) => {
    const button = (event.target as Element | null)?.closest<HTMLButtonElement>('[data-ars-nav]')
    if (!button || !root.contains(button) || button.disabled) return
    const action = button.getAttribute('data-ars-nav')
    const count = configs[n]?.count ?? 0
    step = action === 'reset' ? 0 : Math.min(count, Math.max(0, step + Number(action)))
    render()
    // A disabled button drops focus; keep keyboard users inside the stepper.
    if (button.disabled) nav(action === '1' ? '-1' : '1')?.focus()
  }

  const onChange = (event: Event) => {
    const input = event.target as HTMLInputElement | null
    if (!input?.matches('[data-ars-n-input]')) return
    n = input.value
    step = 0
    render()
  }

  root.addEventListener('click', onClick)
  root.addEventListener('change', onChange)
  render()
  return () => {
    root.removeEventListener('click', onClick)
    root.removeEventListener('change', onChange)
  }
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ARS_ROOT_SELECTOR)) {
    const cleanup = setupAllReduceSteps(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
