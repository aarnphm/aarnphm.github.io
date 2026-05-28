type Mode = 'window' | 'global' | 'masked'

const DILATIONS = [1, 2, 4] as const

const classify = (i: number, j: number, w: number, d: number, g: number): Mode => {
  if (i < g || j < g) return 'global'
  const delta = j - i
  if (Math.abs(delta) <= w && (d === 1 || delta % d === 0)) return 'window'
  return 'masked'
}

const modeClass = (mode: Mode): string => {
  if (mode === 'window') return 'swm-cell swm-cell--window'
  if (mode === 'global') return 'swm-cell swm-cell--global'
  return 'swm-cell swm-cell--masked'
}

const formatPercent = (numerator: number, denominator: number): string => {
  if (denominator === 0) return '0%'
  const ratio = (numerator / denominator) * 100
  return ratio >= 10 ? `${ratio.toFixed(1)}%` : `${ratio.toFixed(2)}%`
}

const renderState = (root: HTMLElement) => {
  const L = Number(root.dataset.swmLength ?? '0')
  const w = Number(root.dataset.swmW ?? '0')
  const d = Number(root.dataset.swmD ?? '1')
  const g = Number(root.dataset.swmG ?? '0')
  if (!Number.isFinite(L) || L <= 0) return

  let active = 0
  const cells = root.querySelectorAll<SVGRectElement>('[data-swm-cell]')
  for (const cell of cells) {
    const i = Number(cell.dataset.swmI ?? '0')
    const j = Number(cell.dataset.swmJ ?? '0')
    const mode = classify(i, j, w, d, g)
    if (mode !== 'masked') active += 1
    const cls = modeClass(mode)
    if (cell.getAttribute('class') !== cls) cell.setAttribute('class', cls)
  }

  const total = L * L
  const activeEl = root.querySelector('[data-swm-active]')
  if (activeEl) activeEl.textContent = String(active)
  const ratioEl = root.querySelector('[data-swm-ratio]')
  if (ratioEl) ratioEl.textContent = formatPercent(active, total)

  const complexityEl = root.querySelector('[data-swm-complexity]')
  if (complexityEl) {
    const wTerm = d === 1 ? `${2 * w + 1}` : `~${Math.floor((2 * w) / d) + 1}`
    complexityEl.textContent = ` = O(L*${wTerm}) + O(L*${g})`
  }

  const receptiveEl = root.querySelector('[data-swm-receptive]')
  if (receptiveEl) receptiveEl.textContent = String(Math.min(8 * w, L))

  const wInput = root.querySelector<HTMLInputElement>('[data-swm-w-input]')
  if (wInput) {
    wInput.value = String(w)
    wInput.setAttribute('aria-valuenow', String(w))
    wInput.setAttribute('aria-valuetext', `window radius ${w}`)
  }
  const wValue = root.querySelector('[data-swm-w-value]')
  if (wValue) wValue.textContent = String(w)

  const gInput = root.querySelector<HTMLInputElement>('[data-swm-g-input]')
  if (gInput) {
    gInput.value = String(g)
    gInput.setAttribute('aria-valuenow', String(g))
    gInput.setAttribute('aria-valuetext', `${g} global tokens`)
  }
  const gValue = root.querySelector('[data-swm-g-value]')
  if (gValue) gValue.textContent = String(g)

  for (const btn of root.querySelectorAll<HTMLButtonElement>('[data-swm-d-btn]')) {
    const value = Number(btn.dataset.swmDBtn ?? '1')
    const isActive = value === d
    btn.classList.toggle('is-active', isActive)
    btn.setAttribute('aria-checked', isActive ? 'true' : 'false')
  }
}

const clampParam = (raw: number, min: number, max: number): number => {
  if (!Number.isFinite(raw)) return min
  return Math.min(Math.max(Math.round(raw), min), max)
}

const setupSlidingWindowMask = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-sliding-window-mask]')
  for (const root of roots) {
    if (root.dataset.swmBound === 'true') continue
    root.dataset.swmBound = 'true'

    const L = Number(root.dataset.swmLength ?? '0')
    const wMax = Math.max(1, Math.floor(L / 2))

    renderState(root)

    const wInput = root.querySelector<HTMLInputElement>('[data-swm-w-input]')
    const gInput = root.querySelector<HTMLInputElement>('[data-swm-g-input]')
    const dButtons = root.querySelectorAll<HTMLButtonElement>('[data-swm-d-btn]')

    const handleW = () => {
      if (!wInput) return
      const next = clampParam(Number(wInput.value), 1, wMax)
      root.dataset.swmW = String(next)
      renderState(root)
    }
    const handleG = () => {
      if (!gInput) return
      const next = clampParam(Number(gInput.value), 0, 3)
      root.dataset.swmG = String(next)
      renderState(root)
    }
    const dHandlers: Array<() => void> = []
    for (const btn of dButtons) {
      const handler = () => {
        const value = Number(btn.dataset.swmDBtn ?? '1')
        if (!DILATIONS.includes(value as (typeof DILATIONS)[number])) return
        root.dataset.swmD = String(value)
        renderState(root)
      }
      btn.addEventListener('click', handler)
      dHandlers.push(() => btn.removeEventListener('click', handler))
    }

    wInput?.addEventListener('input', handleW)
    gInput?.addEventListener('input', handleG)

    window.addCleanup(() => {
      wInput?.removeEventListener('input', handleW)
      gInput?.removeEventListener('input', handleG)
      for (const off of dHandlers) off()
      delete root.dataset.swmBound
    })
  }
}

document.addEventListener('nav', setupSlidingWindowMask)
