import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

type ClientFrame = {
  n: string
  e: string
  m: string
  t: string[]
  shown: number
  lit: number
  out: Record<string, string>
}

const FSM_ROOT_SELECTOR = '[data-fsm-decoding]'

const setupFSMDecoding = (root: HTMLElement): (() => void) => {
  const raw = root.getAttribute('data-fsm-frames')
  const frames = raw ? (JSON.parse(raw) as Record<string, ClientFrame[]>) : null
  const back = root.querySelector<HTMLButtonElement>('[data-fsm-back]')
  const step = root.querySelector<HTMLButtonElement>('[data-fsm-step]')
  const reset = root.querySelector<HTMLButtonElement>('[data-fsm-reset]')
  let mode = root.getAttribute('data-fsm-mode') ?? 'char'
  let k = 0

  const setSt = (els: Iterable<Element>, code: (i: number) => string, attr: string) => {
    for (const el of els) el.setAttribute('data-st', code(Number(el.getAttribute(attr))))
  }

  const apply = () => {
    root.setAttribute('data-fsm-mode', mode)
    if (!frames) return
    const list = frames[mode]
    const f = list?.[k]
    if (!f) return
    const graph = root.querySelectorAll(`[data-fsm-graph="${mode}"]`)
    for (const g of graph) {
      setSt(g.querySelectorAll('[data-fsm-node]'), i => f.n[i], 'data-fsm-node')
      setSt(g.querySelectorAll('[data-fsm-edge]'), i => f.e[i], 'data-fsm-edge')
    }
    setSt(root.querySelectorAll('[data-fsm-tok]'), i => f.m[i], 'data-fsm-tok')
    for (const el of root.querySelectorAll('[data-fsm-tok-to]'))
      el.textContent = f.t[Number(el.getAttribute('data-fsm-tok-to'))]
    const tape = root.querySelector(`[data-fsm-tape-mode="${mode}"]`)
    if (tape)
      setSt(
        tape.querySelectorAll('[data-fsm-tape]'),
        i => (i < f.shown - f.lit ? 's' : i < f.shown ? 'l' : 'h'),
        'data-fsm-tape',
      )
    for (const p of root.querySelectorAll<HTMLElement>('[data-fsm-act]'))
      p.hidden = p.getAttribute('data-fsm-act') !== `${mode}:${k}`
    for (const dd of root.querySelectorAll('[data-fsm-out]')) {
      const key = dd.getAttribute('data-fsm-out')
      if (key && key in f.out) dd.textContent = f.out[key]
    }
    if (back) back.disabled = k === 0
    if (reset) reset.disabled = k === 0
    if (step) step.disabled = k === list.length - 1
  }

  const onChange = (ev: Event) => {
    const input = ev.target as HTMLInputElement
    if (!input.matches('[data-fsm-mode-input]')) return
    mode = input.value
    k = 0
    apply()
  }
  const move = (to: number) => (ev: Event) => {
    const n = frames?.[mode]?.length ?? 1
    k = Math.max(0, Math.min(n - 1, to === 0 ? 0 : k + to))
    apply()
    // A button that becomes disabled drops focus; keep focus on the stepper.
    const own = ev.currentTarget as HTMLButtonElement
    if (own.disabled) (own === step ? back : step)?.focus()
  }
  const onBack = move(-1)
  const onStep = move(1)
  const onReset = move(0)

  root.addEventListener('change', onChange)
  back?.addEventListener('click', onBack)
  step?.addEventListener('click', onStep)
  reset?.addEventListener('click', onReset)
  apply()
  return () => {
    root.removeEventListener('change', onChange)
    back?.removeEventListener('click', onBack)
    step?.removeEventListener('click', onStep)
    reset?.removeEventListener('click', onReset)
  }
}

registerMdxContent(FSM_ROOT_SELECTOR, root => {
  window.addCleanup?.(setupFSMDecoding(root))
  window.addCleanup?.(observeSvgScale(root.querySelectorAll<SVGSVGElement>('.fsm-svg'), '--fsm-u'))
})
