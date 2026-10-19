export {}

type State = {
  orig: number[][]
  deq: number[][]
  err: number[][]
  ticks: string
  max: number
  errLabel: string
  valLabel: string
  out: Record<string, string>
}

const place = (el: Element | null, [y, h]: number[]) => {
  el?.setAttribute('y', String(y))
  el?.setAttribute('height', String(h))
}

const setupMXBlockScale = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-mxb-states')
  if (!raw) return null
  const states = JSON.parse(raw) as Record<string, State>
  const cols = Array.from(root.querySelectorAll<SVGGElement>('[data-mxb-col]'))
  const ticks = root.querySelector('[data-mxb-ticks]')
  const errLabel = root.querySelector('[data-mxb-errmax]')
  const valLabel = root.querySelector('[data-mxb-valmax]')
  const outputs = Array.from(root.querySelectorAll<HTMLElement>('[data-mxb-out]'))

  const checked = (selector: string) =>
    root.querySelector<HTMLInputElement>(`${selector}:checked`)?.value

  const apply = () => {
    const state =
      states[`${checked('[data-mxb-block-input]')}:${checked('[data-mxb-format-input]')}`]
    if (!state) return
    cols.forEach((col, i) => {
      col.setAttribute('data-mxb-max', i === state.max ? 'true' : 'false')
      place(col.querySelector('.mxb-orig'), state.orig[i])
      place(col.querySelector('.mxb-deq'), state.deq[i])
      place(col.querySelector('.mxb-err'), state.err[i])
    })
    ticks?.setAttribute('d', state.ticks)
    if (errLabel) errLabel.textContent = state.errLabel
    if (valLabel) valLabel.textContent = state.valLabel
    // Readout strings are KaTeX markup rendered at build time from the figure's own model.
    for (const out of outputs)
      out.innerHTML = state.out[out.getAttribute('data-mxb-out') ?? ''] ?? ''
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-mx-block-scale]')) {
    const cleanup = setupMXBlockScale(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
