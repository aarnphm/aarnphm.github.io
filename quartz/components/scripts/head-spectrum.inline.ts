import { registerMdxContent } from './mdx-content'

type LayoutState = { xS: number[]; yS: number[]; points: string[] }
type HeadState = {
  sigma: string[]
  energy: string[]
  layouts: Record<string, LayoutState>
  attention: string[][][]
}

const HS_ROOT_SELECTOR = '[data-head-spectrum]'
const PROMPT_IDS = ['plain', 'comma'] as const

const setupHeadSpectrum = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-hs-states')
  const slider = root.querySelector<HTMLInputElement>('[data-hs-k-input]')
  if (!raw || !slider) return null
  const states = JSON.parse(raw) as Record<string, HeadState>
  const q = <T extends Element>(sel: string) => root.querySelector<T>(sel)
  const plots = Array.from(root.querySelectorAll<SVGSVGElement>('[data-hs-layout]'))
  const out = (key: string) => q<HTMLElement>(`[data-hs-out="${key}"]`)

  // One active cell per prompt panel, kept in the lower triangle.
  const active = PROMPT_IDS.map(pid => {
    const panel = q<HTMLElement>(`[data-hs-prompt-panel="${pid}"]`)
    const n = Number(panel?.querySelector('[data-hs-cells]')?.getAttribute('data-hs-n') ?? 0)
    return { panel, n, row: n - 1, col: 0 }
  })

  const current = () => {
    const head = root.querySelector<HTMLInputElement>('[data-hs-head-input]:checked')?.value
    const prompt = root.querySelector<HTMLInputElement>('[data-hs-prompt-input]:checked')?.value
    return { head: head ?? '', prompt: prompt ?? 'comma', state: head ? states[head] : undefined }
  }

  const tokenText = (panel: HTMLElement, attr: string, i: number) =>
    panel.querySelector(`[${attr}="${i}"]`)?.textContent ?? ''

  const applyCell = (pi: number, state: HeadState) => {
    const a = active[pi]
    if (!a.panel) return
    const cell = a.panel.querySelector<SVGRectElement>(`[data-hs-cell="${a.row}-${a.col}"]`)
    const mark = a.panel.querySelector<SVGRectElement>('[data-hs-active]')
    // SSR places the mark on the bottom-left cell; move it by that cell's offset.
    const origin = a.panel.querySelector<SVGRectElement>(`[data-hs-cell="${a.n - 1}-0"]`)
    if (cell && mark && origin) {
      const dx = cell.x.baseVal.value - origin.x.baseVal.value
      const dy = cell.y.baseVal.value - origin.y.baseVal.value
      mark.setAttribute('transform', `translate(${dx},${dy})`)
    }
    for (const el of a.panel.querySelectorAll('[data-hs-row],[data-hs-col]')) {
      const on =
        el.getAttribute('data-hs-row') === String(a.row) ||
        el.getAttribute('data-hs-col') === String(a.col)
      el.setAttribute('data-hs-on', on ? 'true' : 'false')
    }
    const readout = a.panel.querySelector('[data-hs-cell-readout]')
    if (readout)
      readout.textContent = `${tokenText(a.panel, 'data-hs-row', a.row)} → ${tokenText(a.panel, 'data-hs-col', a.col)}: ${state.attention[pi][a.row][a.col]}`
  }

  const applyPlot = (svg: SVGSVGElement, geo: LayoutState, i: number) => {
    const at = <T extends Element>(sel: string) => svg.querySelector<T>(sel)
    const area = at<SVGPolygonElement>('[data-hs-kept-area]')
    const base = area?.getAttribute('data-hs-base') ?? '0'
    const [xk, yk] = geo.points[i].split(',')
    const [x1] = geo.points[0].split(',')
    at('[data-hs-kept-band]')?.setAttribute('width', String(geo.xS[i] - geo.xS[0]))
    area?.setAttribute(
      'points',
      `${x1},${base} ${geo.points.slice(0, i + 1).join(' ')} ${xk},${base}`,
    )
    for (const [sel, x] of [
      ['[data-hs-marker-s]', String(geo.xS[i])],
      ['[data-hs-marker-e]', xk],
    ]) {
      at(sel)?.setAttribute('x1', x)
      at(sel)?.setAttribute('x2', x)
    }
    at('[data-hs-dot-s]')?.setAttribute('cx', String(geo.xS[i]))
    at('[data-hs-dot-s]')?.setAttribute('cy', String(geo.yS[i]))
    at('[data-hs-dot-e]')?.setAttribute('cx', xk)
    at('[data-hs-dot-e]')?.setAttribute('cy', yk)
  }

  const applyK = (state: HeadState) => {
    const k = Number(slider.value)
    const i = k - 1
    for (const svg of plots) {
      const geo = state.layouts[svg.getAttribute('data-hs-layout') ?? '']
      if (geo) applyPlot(svg, geo, i)
    }
    const kOut = out('k')
    const sOut = out('sigmaK')
    const eOut = out('energyK')
    if (kOut) kOut.textContent = String(k)
    if (sOut) sOut.textContent = state.sigma[i]
    if (eOut) eOut.textContent = state.energy[i]
    root.setAttribute('data-hs-k', String(k))
    slider.setAttribute('aria-valuetext', `k ${k}, E_k ${state.energy[i]}`)
  }

  const apply = () => {
    const { head, prompt, state } = current()
    if (!state) return
    root.setAttribute('data-hs-head', head)
    root.setAttribute('data-hs-prompt', prompt)
    for (const el of root.querySelectorAll('[data-hs-for]'))
      el.setAttribute('data-active', el.getAttribute('data-hs-for') === head ? 'true' : 'false')
    active.forEach(({ panel }, pi) => {
      if (!panel) return
      for (const cell of panel.querySelectorAll<SVGRectElement>('[data-hs-cell]')) {
        const [r, c] = (cell.getAttribute('data-hs-cell') ?? '0-0').split('-').map(Number)
        cell.style.setProperty('--hs-a', state.attention[pi][r][c])
      }
      applyCell(pi, state)
    })
    applyK(state)
  }

  const onInput = () => {
    const { state } = current()
    if (state) applyK(state)
  }

  const moveTo = (pi: number, row: number, col: number) => {
    const a = active[pi]
    a.row = Math.min(Math.max(row, 0), a.n - 1)
    a.col = Math.min(Math.max(col, 0), a.row)
    const { state } = current()
    if (state) applyCell(pi, state)
  }

  const cellIndex = (target: EventTarget | null) => {
    const el = target instanceof Element ? target.closest('[data-hs-cell]') : null
    const panel = el?.closest('[data-hs-prompt-panel]')
    if (!el || !panel) return null
    const pi = PROMPT_IDS.indexOf(
      panel.getAttribute('data-hs-prompt-panel') as (typeof PROMPT_IDS)[number],
    )
    const [row, col] = (el.getAttribute('data-hs-cell') ?? '0-0').split('-').map(Number)
    return { pi, row, col }
  }

  const onPointer = (e: PointerEvent) => {
    const hit = cellIndex(e.target)
    if (hit) moveTo(hit.pi, hit.row, hit.col)
  }

  const onKey = (e: KeyboardEvent) => {
    const group = e.target instanceof Element ? e.target.closest('[data-hs-cells]') : null
    const panel = group?.closest('[data-hs-prompt-panel]')
    if (!group || !panel) return
    const pi = PROMPT_IDS.indexOf(
      panel.getAttribute('data-hs-prompt-panel') as (typeof PROMPT_IDS)[number],
    )
    const a = active[pi]
    const step: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    }
    if (e.key in step) {
      const [dr, dc] = step[e.key]
      moveTo(pi, a.row + dr, a.col + dc)
    } else if (e.key === 'Home') moveTo(pi, a.row, 0)
    else if (e.key === 'End') moveTo(pi, a.row, a.row)
    else return
    e.preventDefault()
  }

  root.addEventListener('change', apply)
  slider.addEventListener('input', onInput)
  root.addEventListener('pointerover', onPointer)
  root.addEventListener('keydown', onKey)
  apply()
  return () => {
    root.removeEventListener('change', apply)
    slider.removeEventListener('input', onInput)
    root.removeEventListener('pointerover', onPointer)
    root.removeEventListener('keydown', onKey)
  }
}

registerMdxContent(HS_ROOT_SELECTOR, root => {
  const cleanup = setupHeadSpectrum(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
