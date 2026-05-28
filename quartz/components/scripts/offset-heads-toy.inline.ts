export {}

type OhtMatrix = number[][]

const OHT_ROOT = '[data-offset-heads-toy]'
const OHT_HEAD_DIM = 4

const ohtMulberry32 = (seed: number) => {
  let oht_t = seed >>> 0
  return () => {
    oht_t = (oht_t + 0x6d2b79f5) >>> 0
    let r = oht_t
    r = Math.imul(r ^ (r >>> 15), r | 1)
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

const ohtNormal = (rand: () => number): number => {
  let u = 0
  let v = 0
  while (u === 0) u = rand()
  while (v === 0) v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

const ohtZeroMatrix = (rows: number, cols: number): OhtMatrix =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0))

const ohtFillRandom = (rand: () => number, rows: number, cols: number): OhtMatrix =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => ohtNormal(rand)))

const ohtRowSoftmax = (scores: OhtMatrix): OhtMatrix =>
  scores.map(row => {
    let max = -Infinity
    for (const v of row) if (v > max) max = v
    let sum = 0
    const exps = row.map(v => {
      const e = Math.exp(v - max)
      sum += e
      return e
    })
    const denom = sum === 0 ? 1 : sum
    return exps.map(e => e / denom)
  })

const ohtMatmul = (a: OhtMatrix, b: OhtMatrix): OhtMatrix => {
  const m = a.length
  const k = b.length
  const n = b[0].length
  const out: OhtMatrix = Array.from({ length: m }, () => Array.from({ length: n }, () => 0))
  for (let i = 0; i < m; i++) {
    const ai = a[i]
    const oi = out[i]
    for (let p = 0; p < k; p++) {
      const aip = ai[p]
      if (aip === 0) continue
      const bp = b[p]
      for (let j = 0; j < n; j++) oi[j] += aip * bp[j]
    }
  }
  return out
}

const ohtAddMatrix = (a: OhtMatrix, b: OhtMatrix): OhtMatrix =>
  a.map((row, i) =>
    row.map((aij, j) => {
      const aa = aij < -1e8 ? -1e9 : aij
      const bb = b[i][j] < -1e8 ? -1e9 : b[i][j]
      return aa + bb
    }),
  )

const ohtFrobeniusDiff = (a: OhtMatrix, b: OhtMatrix): number => {
  let total = 0
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]
    const bi = b[i]
    for (let j = 0; j < ai.length; j++) {
      const d = ai[j] - bi[j]
      total += d * d
    }
  }
  return Math.sqrt(total)
}

const ohtFrobenius = (a: OhtMatrix): number => {
  let total = 0
  for (const row of a) for (const v of row) total += v * v
  return Math.sqrt(total)
}

const ohtBuildOffsetScores = (L: number, offset: number, beta: number): OhtMatrix =>
  Array.from({ length: L }, (_, i) =>
    Array.from({ length: L }, (_, j) => (j === i + offset ? beta : -1e9)),
  )

const ohtTranspose = (a: OhtMatrix): OhtMatrix => {
  const r = a.length
  const c = a[0].length
  return Array.from({ length: c }, (_, j) => Array.from({ length: r }, (_, i) => a[i][j]))
}

const ohtInvertSmall = (m: OhtMatrix): OhtMatrix | null => {
  const n = m.length
  const a: OhtMatrix = m.map((row, i) => {
    const aug = Array.from({ length: 2 * n }, () => 0)
    for (let j = 0; j < n; j++) aug[j] = row[j]
    aug[n + i] = 1
    return aug
  })
  for (let i = 0; i < n; i++) {
    let pivotRow = i
    let pivotVal = Math.abs(a[i][i])
    for (let k = i + 1; k < n; k++) {
      const v = Math.abs(a[k][i])
      if (v > pivotVal) {
        pivotVal = v
        pivotRow = k
      }
    }
    if (pivotVal < 1e-12) return null
    if (pivotRow !== i) {
      const tmp = a[i]
      a[i] = a[pivotRow]
      a[pivotRow] = tmp
    }
    const pivot = a[i][i]
    for (let j = 0; j < 2 * n; j++) a[i][j] /= pivot
    for (let k = 0; k < n; k++) {
      if (k === i) continue
      const factor = a[k][i]
      if (factor === 0) continue
      for (let j = 0; j < 2 * n; j++) a[k][j] -= factor * a[i][j]
    }
  }
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => a[i][n + j]))
}

const ohtForcedMatch = (
  P1: OhtMatrix,
  P2: OhtMatrix,
  Ps: OhtMatrix,
  V: OhtMatrix,
  WO1: OhtMatrix,
  WO2: OhtMatrix,
): OhtMatrix => {
  const Y1 = ohtMatmul(ohtMatmul(P1, V), WO1)
  const Y2 = ohtMatmul(ohtMatmul(P2, V), WO2)
  const target: OhtMatrix = Y1.map((row, i) => row.map((y1ij, j) => y1ij + Y2[i][j]))
  const PsV = ohtMatmul(Ps, V)
  const PsV_T = ohtTranspose(PsV)
  const gram = ohtMatmul(PsV_T, PsV)
  for (let i = 0; i < gram.length; i++) gram[i][i] += 1e-3
  const invGram = ohtInvertSmall(gram)
  if (!invGram) return ohtZeroMatrix(PsV.length, PsV[0].length)
  const PsVtY = ohtMatmul(PsV_T, target)
  const WOtilde = ohtMatmul(invGram, PsVtY)
  return ohtMatmul(PsV, WOtilde)
}

type OhtState = {
  L: number
  beta: number
  seed: number
  mode: 'independent' | 'forced'
  P1: OhtMatrix
  P2: OhtMatrix
  Ps: OhtMatrix
  norm: number
  ref: number
}

const ohtCompute = (
  L: number,
  beta: number,
  seed: number,
  mode: 'independent' | 'forced',
): OhtState => {
  const rand = ohtMulberry32(seed)
  const V = ohtFillRandom(rand, L, OHT_HEAD_DIM)
  const WO1 = ohtFillRandom(rand, OHT_HEAD_DIM, OHT_HEAD_DIM)
  const WO2 = ohtFillRandom(rand, OHT_HEAD_DIM, OHT_HEAD_DIM)
  const WOtilde = ohtFillRandom(rand, OHT_HEAD_DIM, OHT_HEAD_DIM)

  const S1 = ohtBuildOffsetScores(L, 1, beta)
  const Sm1 = ohtBuildOffsetScores(L, -1, beta)
  const Ssum = ohtAddMatrix(S1, Sm1)

  const P1 = ohtRowSoftmax(S1)
  const P2 = ohtRowSoftmax(Sm1)
  const Ps = ohtRowSoftmax(Ssum)

  const Y_mha_part1 = ohtMatmul(ohtMatmul(P1, V), WO1)
  const Y_mha_part2 = ohtMatmul(ohtMatmul(P2, V), WO2)
  const Y_mha: OhtMatrix = Y_mha_part1.map((row, i) => row.map((v, j) => v + Y_mha_part2[i][j]))

  const Y_sh =
    mode === 'forced'
      ? ohtForcedMatch(P1, P2, Ps, V, WO1, WO2)
      : ohtMatmul(ohtMatmul(Ps, V), WOtilde)

  return {
    L,
    beta,
    seed,
    mode,
    P1,
    P2,
    Ps,
    norm: ohtFrobeniusDiff(Y_mha, Y_sh),
    ref: ohtFrobenius(Y_mha),
  }
}

const ohtFmt = (v: number, digits = 3): string => {
  if (!Number.isFinite(v)) return 'inf'
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2)
  return v.toFixed(digits)
}

const ohtPaintPanel = (root: HTMLElement, key: string, matrix: OhtMatrix, L: number) => {
  let max = 0
  for (const row of matrix) for (const v of row) if (v > max) max = v
  const norm = max > 1e-9 ? max : 1
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      const cell = root.querySelector<SVGRectElement>(`[data-oht-cell="${key}-${i}-${j}"]`)
      if (!cell) continue
      const value = matrix[i][j]
      const intensity = Math.min(1, Math.max(0, value / norm))
      const display = 0.06 + intensity * 0.94
      cell.setAttribute('fill-opacity', display.toFixed(3))
      const tooltip = cell.querySelector<SVGTitleElement>(`[data-oht-tooltip="${key}-${i}-${j}"]`)
      if (tooltip) tooltip.textContent = ohtFmt(value)
    }
  }
}

const ohtRender = (root: HTMLElement, state: OhtState) => {
  ohtPaintPanel(root, 'p1', state.P1, state.L)
  ohtPaintPanel(root, 'm1', state.P2, state.L)
  ohtPaintPanel(root, 'sh', state.Ps, state.L)

  const normEl = root.querySelector('[data-oht-stat="norm"]')
  if (normEl) normEl.textContent = ohtFmt(state.norm)
  const refEl = root.querySelector('[data-oht-stat="ref"]')
  if (refEl) refEl.textContent = ohtFmt(state.ref)
  const relEl = root.querySelector('[data-oht-stat="rel"]')
  if (relEl) {
    const ratio = state.ref > 1e-9 ? (state.norm / state.ref) * 100 : 0
    relEl.textContent = `${ratio.toFixed(1)}%`
  }
  const seedEl = root.querySelector('[data-oht-stat="seed"]')
  if (seedEl) seedEl.textContent = String(state.seed)

  const betaValue = root.querySelector('[data-oht-beta-value]')
  if (betaValue) betaValue.textContent = state.beta.toFixed(1)
  const betaInput = root.querySelector<HTMLInputElement>('[data-oht-beta-input]')
  if (betaInput) {
    betaInput.setAttribute('aria-valuenow', String(state.beta))
    betaInput.setAttribute('aria-valuetext', `beta ${state.beta.toFixed(1)}`)
  }

  for (const btn of root.querySelectorAll<HTMLButtonElement>('[data-oht-mode-btn]')) {
    const isActive = btn.dataset.ohtModeBtn === state.mode
    btn.classList.toggle('is-active', isActive)
    btn.setAttribute('aria-checked', isActive ? 'true' : 'false')
  }

  const sr = root.querySelector('[data-oht-sr]')
  if (sr) {
    sr.textContent = `Norm gap ${ohtFmt(state.norm)} at beta ${state.beta.toFixed(1)}, seed ${state.seed}, mode ${state.mode}.`
  }
}

const ohtSetupRoot = (root: HTMLElement) => {
  if (root.dataset.ohtBound === 'true') return
  root.dataset.ohtBound = 'true'

  const L = Number(root.dataset.ohtLength ?? '6')
  let beta = Number(root.dataset.ohtBeta ?? '6')
  let seed = Number(root.dataset.ohtSeed ?? '1')
  let mode: 'independent' | 'forced' = root.dataset.ohtMode === 'forced' ? 'forced' : 'independent'

  const rerender = () => {
    const state = ohtCompute(L, beta, seed, mode)
    ohtRender(root, state)
  }
  rerender()

  const betaInput = root.querySelector<HTMLInputElement>('[data-oht-beta-input]')
  const reseedBtn = root.querySelector<HTMLButtonElement>('[data-oht-reseed]')
  const modeBtns = root.querySelectorAll<HTMLButtonElement>('[data-oht-mode-btn]')

  const handleBeta = () => {
    if (!betaInput) return
    const next = Number(betaInput.value)
    if (!Number.isFinite(next)) return
    beta = next
    root.dataset.ohtBeta = String(beta)
    rerender()
  }
  const handleReseed = () => {
    seed = (seed + 1) >>> 0 || 1
    root.dataset.ohtSeed = String(seed)
    rerender()
  }
  const modeHandlers: Array<() => void> = []
  for (const btn of modeBtns) {
    const handler = () => {
      const next = btn.dataset.ohtModeBtn
      if (next !== 'forced' && next !== 'independent') return
      mode = next
      root.dataset.ohtMode = mode
      rerender()
    }
    btn.addEventListener('click', handler)
    modeHandlers.push(() => btn.removeEventListener('click', handler))
  }

  betaInput?.addEventListener('input', handleBeta)
  reseedBtn?.addEventListener('click', handleReseed)

  window.addCleanup(() => {
    betaInput?.removeEventListener('input', handleBeta)
    reseedBtn?.removeEventListener('click', handleReseed)
    for (const off of modeHandlers) off()
    delete root.dataset.ohtBound
  })
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(OHT_ROOT)) {
    ohtSetupRoot(root)
  }
})
