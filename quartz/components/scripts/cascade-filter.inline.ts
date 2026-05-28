import katex from 'katex'

export {}

type Preset = 'uniform' | 'spiky' | 'longtail'

type CascadeState = {
  root: HTMLElement
  slider: HTMLInputElement
  tauReadout: HTMLElement
  thresholdLine: SVGLineElement
  presetBtns: HTMLButtonElement[]
  coarseGroups: SVGGElement[]
  coarseBars: SVGRectElement[]
  fineGroups: SVGGElement[]
  statKept: HTMLElement
  statSpeedup: HTMLElement
  statRecall: HTMLElement
  tauFo: SVGForeignObjectElement | null
  n: number
  barTop: number
  barH: number
  scores: Record<Preset, number[]>
  preset: Preset
  threshold: number
}

const CF_ROOT_SELECTOR = '[data-cascade-filter]'
const CF_BAR_TOP = 18
const CF_BAR_BOTTOM_PAD = 28
const CF_COARSE_H = 138

const parseScoreString = (raw: string): number[] =>
  raw.split(',').map(part => {
    const v = Number.parseFloat(part)
    return Number.isFinite(v) ? v : 0
  })

const readScores = (root: HTMLElement): Record<Preset, number[]> | null => {
  const raw = root.getAttribute('data-scores')
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Record<Preset, string>
    return {
      spiky: parseScoreString(parsed.spiky ?? ''),
      uniform: parseScoreString(parsed.uniform ?? ''),
      longtail: parseScoreString(parsed.longtail ?? ''),
    }
  } catch {
    return null
  }
}

const cfRenderMath = (tex: string): string =>
  katex.renderToString(tex, {
    displayMode: false,
    output: 'html',
    strict: false,
    throwOnError: false,
  })

const cfRecallTex = (r: number): string => `${Math.round(r * 100)}\\%`
const cfSpeedupTex = (s: number): string =>
  Number.isFinite(s) ? `${s.toFixed(1)}\\times` : '\\infty'

const renderCascade = (state: CascadeState) => {
  const { threshold, preset, scores, n, barH } = state
  const active = scores[preset]
  let kept = 0
  let keptMass = 0
  let totalMass = 0
  for (let i = 0; i < n; i++) {
    const s = active[i] ?? 0
    totalMass += s
    const survives = s >= threshold
    if (survives) {
      kept++
      keptMass += s
    }
    const group = state.coarseGroups[i]
    const bar = state.coarseBars[i]
    if (group && bar) {
      group.classList.toggle('is-keep', survives)
      group.classList.toggle('is-drop', !survives)
      const h = s * barH
      bar.setAttribute('y', String(CF_BAR_TOP + (barH - h)))
      bar.setAttribute('height', String(h))
    }
    const fine = state.fineGroups[i]
    if (fine) {
      fine.classList.toggle('is-keep', survives)
      fine.classList.toggle('is-drop', !survives)
    }
  }
  const lineY = CF_BAR_TOP + barH * (1 - threshold)
  state.thresholdLine.setAttribute('y1', String(lineY))
  state.thresholdLine.setAttribute('y2', String(lineY))
  if (state.tauFo) {
    state.tauFo.setAttribute('y', String(lineY - 22))
  }
  state.tauReadout.innerHTML = cfRenderMath(threshold.toFixed(2))
  state.slider.setAttribute('aria-valuenow', threshold.toFixed(2))
  state.slider.setAttribute('aria-valuetext', `tau equals ${threshold.toFixed(2)}`)
  state.root.style.setProperty('--cf-tau', threshold.toFixed(3))
  state.statKept.innerHTML = cfRenderMath(String(kept))
  state.statSpeedup.innerHTML = cfRenderMath(kept > 0 ? cfSpeedupTex(n / kept) : '\\infty')
  state.statRecall.innerHTML = cfRenderMath(
    totalMass > 0 ? cfRecallTex(keptMass / totalMass) : '0\\%',
  )
}

const setupCascade = (root: HTMLElement): (() => void) | null => {
  const slider = root.querySelector<HTMLInputElement>('[data-cf-slider]')
  const tauReadout = root.querySelector<HTMLElement>('[data-cf-tau]')
  const thresholdLine = root.querySelector<SVGLineElement>('[data-cf-threshold-line]')
  const statKept = root.querySelector<HTMLElement>('[data-cf-stat="kept"]')
  const statSpeedup = root.querySelector<HTMLElement>('[data-cf-stat="speedup"]')
  const statRecall = root.querySelector<HTMLElement>('[data-cf-stat="recall"]')
  if (!slider || !tauReadout || !thresholdLine || !statKept || !statSpeedup || !statRecall) {
    return null
  }
  const scores = readScores(root)
  if (!scores) return null
  const n = Number.parseInt(root.getAttribute('data-tiles') ?? '0', 10)
  if (!Number.isFinite(n) || n <= 0) return null
  const initialPreset = (root.getAttribute('data-preset') as Preset | null) ?? 'spiky'
  const initialThreshold = Number.parseFloat(root.getAttribute('data-threshold') ?? '0.5')

  const coarseGroups = Array.from(
    root.querySelectorAll<SVGGElement>('.cf-svg--coarse [data-tile-idx]'),
  )
  const coarseBars = coarseGroups.map(g => g.querySelector<SVGRectElement>('[data-tile-bar]')!)
  const fineGroups = Array.from(root.querySelectorAll<SVGGElement>('.cf-svg--fine [data-fine-idx]'))
  if (coarseGroups.length !== n || fineGroups.length !== n) return null

  const presetBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-cf-preset]'))
  const tauFo = root.querySelector<SVGForeignObjectElement>('[data-cf-tau-label]')
  const barH = CF_COARSE_H - CF_BAR_TOP - CF_BAR_BOTTOM_PAD

  const state: CascadeState = {
    root,
    slider,
    tauReadout,
    thresholdLine,
    presetBtns,
    coarseGroups,
    coarseBars,
    fineGroups,
    statKept,
    statSpeedup,
    statRecall,
    tauFo,
    n,
    barTop: CF_BAR_TOP,
    barH,
    scores,
    preset: initialPreset,
    threshold: Number.isFinite(initialThreshold) ? initialThreshold : 0.5,
  }

  renderCascade(state)

  const handleSlider = () => {
    const next = Number.parseFloat(slider.value)
    if (!Number.isFinite(next)) return
    state.threshold = Math.max(0, Math.min(1, next))
    renderCascade(state)
  }

  const handlePreset = (btn: HTMLButtonElement) => () => {
    const next = btn.getAttribute('data-cf-preset') as Preset | null
    if (!next || next === state.preset) return
    state.preset = next
    for (const b of presetBtns) {
      const active = b === btn
      b.setAttribute('aria-selected', active ? 'true' : 'false')
      b.setAttribute('tabindex', active ? '0' : '-1')
    }
    root.setAttribute('data-preset', next)
    renderCascade(state)
  }

  const presetHandlers = presetBtns.map(btn => {
    const handler = handlePreset(btn)
    btn.addEventListener('click', handler)
    return { btn, handler }
  })

  slider.addEventListener('input', handleSlider)

  return () => {
    slider.removeEventListener('input', handleSlider)
    for (const { btn, handler } of presetHandlers) {
      btn.removeEventListener('click', handler)
    }
  }
}

document.addEventListener('nav', () => {
  const roots = document.querySelectorAll<HTMLElement>(CF_ROOT_SELECTOR)
  for (const root of roots) {
    const cleanup = setupCascade(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
