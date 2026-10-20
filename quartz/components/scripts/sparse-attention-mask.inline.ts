export {}

type QueryState = {
  t: number
  cmp: number
  blocks: number
  slcH: number[]
  slcY: number[]
  chosen: number[]
  tags: string[]
  frameY: number
  frameW: number
  winX: number
  winW: number
  unionW: number[]
  dsaY: number[]
  dsaH: number[]
  top: number[]
  out: Record<string, string>
}

const SAM_ROOT_SELECTOR = '[data-sparse-attention-mask]'
// SparseAttentionMask.tsx precomputes every coordinate; this script only writes them.

const setupSparseAttentionMask = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-sam-states')
  const slider = root.querySelector<HTMLInputElement>('[data-sam-query-input]')
  if (!raw || !slider) return null
  const states = JSON.parse(raw) as QueryState[]
  const rows = Array.from(root.querySelectorAll<SVGGElement>('[data-sam-row]'))
  const frame = root.querySelector<SVGRectElement>('[data-sam-row-frame]')
  const value = root.querySelector<HTMLElement>('[data-sam-query-value]')
  const head = root.querySelector<HTMLElement>('[data-sam-head]')
  const spans = Array.from(root.querySelectorAll<SVGRectElement>('[data-sam-cmp-span]'))
  const blocks = Array.from(root.querySelectorAll<SVGGElement>('[data-sam-block]'))
  const unionBlocks = Array.from(root.querySelectorAll<SVGRectElement>('[data-sam-union-block]'))
  const wins = Array.from(root.querySelectorAll<SVGRectElement>('[data-sam-win]'))
  const bars = Array.from(root.querySelectorAll<SVGRectElement>('[data-sam-bar]'))
  const outs = Array.from(root.querySelectorAll<HTMLElement>('[data-sam-out]'))

  const applyQuery = () => {
    const s = states[Number(slider.value)]
    if (!s) return
    const t = s.t
    rows.forEach((row, i) => row.setAttribute('data-lit', i === t ? 'true' : 'false'))
    frame?.setAttribute('y', String(s.frameY))
    frame?.setAttribute('width', String(s.frameW))
    if (value) value.textContent = String(t)
    if (head) head.textContent = `t=${t}`
    slider.setAttribute('aria-valuetext', `query ${t}`)

    for (const span of spans) {
      const i = Number(span.getAttribute('data-sam-cmp-span'))
      span.setAttribute('data-on', i < s.cmp ? 'true' : 'false')
    }
    for (const block of blocks) {
      const j = Number(block.getAttribute('data-sam-block'))
      block.setAttribute('data-shown', j < s.blocks ? 'true' : 'false')
      block.setAttribute('data-on', s.chosen.includes(j) ? 'true' : 'false')
      const score = block.querySelector<SVGRectElement>('[data-sam-score]')
      score?.setAttribute('y', String(s.slcY[j] ?? 0))
      score?.setAttribute('height', String(s.slcH[j] ?? 0))
      const tag = block.querySelector<SVGTextElement>('[data-sam-tag]')
      if (tag) tag.textContent = s.tags[j] ?? ''
    }
    for (const block of unionBlocks) {
      const j = Number(block.getAttribute('data-sam-union-block'))
      block.setAttribute('data-on', s.chosen.includes(j) ? 'true' : 'false')
      block.setAttribute('width', String(s.unionW[j] ?? 0))
    }
    for (const win of wins) {
      win.setAttribute('x', String(s.winX))
      win.setAttribute('width', String(s.winW))
    }
    bars.forEach((bar, i) => {
      bar.setAttribute('data-shown', i <= t ? 'true' : 'false')
      bar.setAttribute('data-on', s.top.includes(i) ? 'true' : 'false')
      bar.setAttribute('y', String(s.dsaY[i] ?? 0))
      bar.setAttribute('height', String(s.dsaH[i] ?? 0))
    })
    for (const out of outs) out.textContent = s.out[out.getAttribute('data-sam-out') ?? ''] ?? ''
  }

  const onChange = (event: Event) => {
    const target = event.target as HTMLInputElement | null
    if (target?.matches('[data-sam-mode-input]') && target.checked)
      root.setAttribute('data-sam-mode', target.value)
  }

  root.addEventListener('change', onChange)
  slider.addEventListener('input', applyQuery)
  const checked = root.querySelector<HTMLInputElement>('[data-sam-mode-input]:checked')
  if (checked) root.setAttribute('data-sam-mode', checked.value)
  applyQuery()
  return () => {
    root.removeEventListener('change', onChange)
    slider.removeEventListener('input', applyQuery)
  }
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(SAM_ROOT_SELECTOR)) {
    const cleanup = setupSparseAttentionMask(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
