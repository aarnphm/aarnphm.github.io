import { registerMdxContent } from './mdx-content'

type ClientRun = {
  steps: [nodes: string, edges: string, store: string, say: string][]
  rows: [label: string, values: string[], revealAt: number[]][]
  reached: string
}

const ROOT_SELECTOR = '[data-graph-traversal]'

const setupGraphTraversal = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-gtr-runs')
  if (!raw) return null
  const runs = JSON.parse(raw) as Record<string, ClientRun>
  const range = root.querySelector<HTMLInputElement>('[data-gtr-range]')
  if (!range) return null
  const nodes = Array.from(root.querySelectorAll<SVGGElement>('[data-gtr-node]'))
  const edges = Array.from(root.querySelectorAll<SVGLineElement>('[data-gtr-edge]'))
  const slots = Array.from(root.querySelectorAll<HTMLElement>('[data-gtr-slot]'))
  const rowLabels = Array.from(root.querySelectorAll<HTMLElement>('[data-gtr-row-label]'))
  const cells = Array.from(root.querySelectorAll<HTMLElement>('[data-gtr-cell]'))
  const say = root.querySelector('[data-gtr-say]')
  const count = root.querySelector('[data-gtr-count]')
  const reached = root.querySelector('[data-gtr-out="reached"]')

  const checked = (selector: string) =>
    root.querySelector<HTMLInputElement>(`${selector}:checked`)?.value
  const current = () => {
    const mode = checked('[data-gtr-mode-input]')
    const start = checked('[data-gtr-start-input]')
    return mode && start ? { mode, run: runs[`${mode}:${start}`] } : null
  }

  const show = (step: number) => {
    const sel = current()
    if (!sel?.run) return
    const { mode, run } = sel
    const last = run.steps.length - 1
    const k = Math.min(Math.max(step, 0), last)
    const [nodeCodes, edgeCodes, store, text] = run.steps[k]
    const stored = store ? store.split(',') : []
    root.setAttribute('data-gtr-mode', mode)
    range.max = String(last)
    range.value = String(k)
    range.setAttribute('aria-valuetext', `step ${k} of ${last}`)
    nodes.forEach((el, i) => el.setAttribute('data-state', nodeCodes[i]))
    edges.forEach((el, i) => el.setAttribute('data-state', edgeCodes[i]))
    slots.forEach((el, i) => {
      el.textContent = stored[i] === undefined ? '' : `n${stored[i]}`
      el.toggleAttribute('data-filled', stored[i] !== undefined)
    })
    rowLabels.forEach((el, r) => (el.textContent = run.rows[r][0]))
    for (const cell of cells) {
      const [r, i] = (cell.getAttribute('data-gtr-cell') ?? '0:0').split(':').map(Number)
      const [, values, at] = run.rows[r]
      cell.textContent = at[i] >= 0 && at[i] <= k ? values[i] || '–' : '–'
    }
    if (say) say.textContent = text
    if (count) count.textContent = `${k} / ${last}`
    if (reached) reached.textContent = run.reached
  }

  const onChange = (event: Event) => {
    const target = event.target as HTMLElement
    if (target === range) return
    if (target.matches('[data-gtr-mode-input], [data-gtr-start-input]')) show(0)
  }
  const onInput = () => show(Number(range.value))
  const onClick = (event: Event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-gtr-go]')
    if (!button) return
    const k = Number(range.value)
    const go = button.getAttribute('data-gtr-go')
    show(go === 'first' ? 0 : go === 'prev' ? k - 1 : go === 'next' ? k + 1 : Infinity)
  }

  root.addEventListener('change', onChange)
  range.addEventListener('input', onInput)
  root.addEventListener('click', onClick)
  show(Number(range.value))
  return () => {
    root.removeEventListener('change', onChange)
    range.removeEventListener('input', onInput)
    root.removeEventListener('click', onClick)
  }
}

registerMdxContent(ROOT_SELECTOR, root => {
  const cleanup = setupGraphTraversal(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
