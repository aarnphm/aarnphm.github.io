import { registerMdxContent } from './mdx-content'

type StepState = {
  step: string
  who: string
  code: string
  src: string
  does: string
  between: string
}

const OUTPUTS = ['step', 'who', 'code', 'src', 'does', 'between'] as const

const setupLogitsProcessorTrace = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-lpt-states')
  const range = root.querySelector<HTMLInputElement>('[data-lpt-step]')
  if (!raw || !range) return null
  const states = JSON.parse(raw) as Record<string, StepState[]>
  const sets = Array.from(root.querySelectorAll<SVGGElement>('[data-lpt-set]'))

  const apply = () => {
    const guided = root.querySelector<HTMLInputElement>('[data-lpt-guided]:checked')?.value
    const steps = guided ? states[guided] : undefined
    if (!guided || !steps) return
    range.max = String(steps.length)
    const step = Math.min(Math.max(1, Number(range.value)), steps.length)
    range.value = String(step)
    const state = steps[step - 1]
    range.setAttribute('aria-valuetext', `call ${state.step}: ${state.code}`)
    for (const set of sets) {
      const active = set.getAttribute('data-lpt-set') === guided
      set.setAttribute('data-on', active ? 'true' : 'false')
      for (const call of set.querySelectorAll('[data-lpt-call]'))
        call.setAttribute(
          'data-on',
          active && call.getAttribute('data-lpt-call') === String(step - 1) ? 'true' : 'false',
        )
    }
    for (const key of OUTPUTS) {
      const out = root.querySelector(`[data-lpt-out="${key}"]`)
      if (out) out.textContent = state[key]
    }
  }

  root.addEventListener('change', apply)
  range.addEventListener('input', apply)
  apply()
  return () => {
    root.removeEventListener('change', apply)
    range.removeEventListener('input', apply)
  }
}

registerMdxContent('[data-logits-processor-trace]', root => {
  const cleanup = setupLogitsProcessorTrace(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
