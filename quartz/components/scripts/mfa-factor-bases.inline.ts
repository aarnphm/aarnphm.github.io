import {
  mfaApprox,
  mfaDense,
  mfaFactorMatrix,
  mfaL,
  mfaMaxM,
  mfaMaxR,
} from '../../util/mfa-factor-model'

function mfaRange(values: number[]): { lo: number; hi: number } {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
    return { lo: -1, hi: 1 }
  }
  return { lo, hi }
}

function mfaSignedFill(v: number, lo: number, hi: number, palette: string[], idx: number): string {
  const span = Math.max(Math.abs(lo), Math.abs(hi), 1e-6)
  const t = Math.max(-1, Math.min(1, v / span))
  const mag = Math.abs(t)
  const color = palette[idx % palette.length] ?? '#fdb2a2'
  if (t >= 0) {
    const alpha = 0.08 + 0.82 * mag
    return `color-mix(in srgb, ${color} ${(alpha * 100).toFixed(1)}%, var(--mfa-cell-bg))`
  }
  const alpha = 0.08 + 0.62 * mag
  return `color-mix(in srgb, var(--mfa-neg) ${(alpha * 100).toFixed(1)}%, var(--mfa-cell-bg))`
}

function mfaPaintGrid(
  root: HTMLElement,
  attr: string,
  data: number[][],
  palette: string[],
  factorIdx: number,
) {
  const flat: number[] = []
  for (let i = 0; i < mfaL; i++) for (let j = 0; j < mfaL; j++) flat.push(data[i][j])
  const { lo, hi } = mfaRange(flat)
  const cells = root.querySelectorAll<SVGRectElement>(`[${attr}]`)
  for (const cell of cells) {
    const key = cell.getAttribute(attr) ?? ''
    const [iStr, jStr] = key.split(',')
    const i = Number(iStr)
    const j = Number(jStr)
    cell.style.fill = mfaSignedFill(data[i][j], lo, hi, palette, factorIdx)
  }
}

function mfaPaintFactorRow(
  root: HTMLElement,
  m: number,
  r: number,
  palette: string[],
  gateOn: boolean,
) {
  const groups = root.querySelectorAll<SVGGElement>('[data-mfa-factor]')
  for (const g of groups) {
    const idx = Number(g.dataset.mfaFactor)
    const active = idx < m
    g.dataset.mfaActive = active ? 'true' : 'false'
    if (!active) {
      const cells = g.querySelectorAll<SVGRectElement>('[data-mfa-factor-cell]')
      for (const c of cells) c.style.fill = 'var(--mfa-cell-bg)'
      continue
    }
    const F = mfaFactorMatrix(idx, r)
    const flat: number[] = []
    for (let i = 0; i < mfaL; i++) for (let j = 0; j < mfaL; j++) flat.push(F[i][j])
    const { lo, hi } = mfaRange(flat)
    const cells = g.querySelectorAll<SVGRectElement>('[data-mfa-factor-cell]')
    for (const cell of cells) {
      const key = cell.getAttribute('data-mfa-factor-cell') ?? ''
      const parts = key.split(',')
      const i = Number(parts[1])
      const j = Number(parts[2])
      cell.style.fill = mfaSignedFill(F[i][j], lo, hi, palette, idx)
    }
    const tag = g.querySelector<HTMLElement>('[data-mfa-factor-tag]')
    if (tag) {
      tag.style.color = palette[idx % palette.length]
      const gate = tag.querySelector<HTMLElement>('[data-mfa-factor-gate]')
      if (gate) gate.dataset.mfaActive = gateOn ? 'true' : 'false'
    }
  }
}

function mfaStateKey(m: number, r: number, gateOn: boolean): string {
  return `${m}-${r}-${gateOn ? 1 : 0}`
}

function mfaUpdateReadout(root: HTMLElement, m: number, r: number, gateOn: boolean) {
  const key = mfaStateKey(m, r, gateOn)
  const states = root.querySelectorAll<HTMLElement>('[data-mfa-readout-state]')
  for (const state of states) {
    state.dataset.mfaActive = state.dataset.mfaReadoutState === key ? 'true' : 'false'
  }
}

function mfaUpdateAria(root: HTMLElement, m: number, r: number) {
  const canvas = root.querySelector<SVGElement>('[data-mfa-canvas]')
  if (canvas) {
    canvas.setAttribute(
      'aria-label',
      `Dense L by L attention matrix on the left versus a low-rank sum of ${m} factor outer product${m === 1 ? '' : 's'} at rank ${r} on the right.`,
    )
  }
  const mSlider = root.querySelector<HTMLInputElement>('[data-mfa-m]')
  if (mSlider) mSlider.setAttribute('aria-valuenow', String(m))
  const rSlider = root.querySelector<HTMLInputElement>('[data-mfa-r]')
  if (rSlider) rSlider.setAttribute('aria-valuenow', String(r))
}

function mfaSetup(root: HTMLElement) {
  if (root.dataset.mfaBound === 'true') return
  root.dataset.mfaBound = 'true'

  const palette = (root.dataset.mfaPalette ?? '#fdb2a2,#cdd597').split(',')
  const mSlider = root.querySelector<HTMLInputElement>('[data-mfa-m]')
  const rSlider = root.querySelector<HTMLInputElement>('[data-mfa-r]')
  const gateInput = root.querySelector<HTMLInputElement>('[data-mfa-gate]')
  if (!mSlider || !rSlider || !gateInput) return

  const apply = () => {
    const m = Math.max(1, Math.min(mfaMaxM, Number(mSlider.value)))
    const r = Math.max(1, Math.min(mfaMaxR, Number(rSlider.value)))
    const gateOn = gateInput.checked
    const approx = mfaApprox(m, r, gateOn)
    mfaPaintGrid(root, 'data-mfa-dense', mfaDense, palette, 0)
    mfaPaintGrid(root, 'data-mfa-approx', approx, palette, 1)
    mfaPaintFactorRow(root, m, r, palette, gateOn)
    mfaUpdateReadout(root, m, r, gateOn)
    mfaUpdateAria(root, m, r)
  }

  mSlider.addEventListener('input', apply)
  rSlider.addEventListener('input', apply)
  gateInput.addEventListener('change', apply)
  apply()

  window.addCleanup(() => {
    mSlider.removeEventListener('input', apply)
    rSlider.removeEventListener('input', apply)
    gateInput.removeEventListener('change', apply)
    delete root.dataset.mfaBound
  })
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-mfa-factor-bases]')) {
    mfaSetup(root)
  }
})

export {}
