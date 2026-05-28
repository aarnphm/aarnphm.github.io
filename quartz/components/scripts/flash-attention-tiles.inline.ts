type Vec = [number, number]

const SCORES: number[][] = [
  [1.2, 0.4, -0.8, 0.1],
  [0.3, 1.5, 0.2, -0.5],
  [-0.4, 0.8, 1.3, 0.6],
  [0.7, -0.2, 0.5, 1.1],
]

const VALUES: Vec[] = [
  [1.0, 0.2],
  [0.4, 0.9],
  [-0.3, 0.6],
  [0.7, -0.1],
]

const B_M = 2
const B_N = 2
const M = SCORES.length / B_M
const N = SCORES[0].length / B_N
const SEQ_LEN = SCORES.length
const HEAD_DIM = VALUES[0].length

type StepState = {
  step: number
  outerJ: number
  innerI: number
  qRows: [number, number]
  kCols: [number, number]
  m: [number, number]
  l: [number, number]
  O: [Vec, Vec]
  bytesHbm: number
}

function computeAllStates(): StepState[] {
  const states: StepState[] = []
  const m: number[] = Array.from({ length: SEQ_LEN }, () => -Infinity)
  const l: number[] = Array.from({ length: SEQ_LEN }, () => 0)
  const O: Vec[] = Array.from({ length: SEQ_LEN }, () => [0, 0] as Vec)

  let step = 0
  let bytesHbm = 0
  const bytesPerEl = 4

  for (let j = 0; j < N; j++) {
    bytesHbm += B_N * HEAD_DIM * bytesPerEl * 2
    for (let i = 0; i < M; i++) {
      step += 1
      bytesHbm += B_M * HEAD_DIM * bytesPerEl
      const qRows: [number, number] = [i * B_M, i * B_M + B_M - 1]
      const kCols: [number, number] = [j * B_N, j * B_N + B_N - 1]

      for (let ii = qRows[0]; ii <= qRows[1]; ii++) {
        let tileMax = -Infinity
        for (let jj = kCols[0]; jj <= kCols[1]; jj++) {
          if (SCORES[ii][jj] > tileMax) tileMax = SCORES[ii][jj]
        }
        const mNew = Math.max(m[ii], tileMax)
        const alpha = m[ii] === -Infinity ? 0 : Math.exp(m[ii] - mNew)
        let tileSum = 0
        const tileWeightedV: Vec = [0, 0]
        for (let jj = kCols[0]; jj <= kCols[1]; jj++) {
          const p = Math.exp(SCORES[ii][jj] - mNew)
          tileSum += p
          tileWeightedV[0] += p * VALUES[jj][0]
          tileWeightedV[1] += p * VALUES[jj][1]
        }
        l[ii] = alpha * l[ii] + tileSum
        O[ii] = [alpha * O[ii][0] + tileWeightedV[0], alpha * O[ii][1] + tileWeightedV[1]]
        m[ii] = mNew
      }
      bytesHbm += B_M * HEAD_DIM * bytesPerEl
      states.push({
        step,
        outerJ: j,
        innerI: i,
        qRows,
        kCols,
        m: [m[qRows[0]], m[qRows[1]]],
        l: [l[qRows[0]], l[qRows[1]]],
        O: [
          [O[qRows[0]][0], O[qRows[0]][1]],
          [O[qRows[1]][0], O[qRows[1]][1]],
        ],
        bytesHbm,
      })
    }
  }
  return states
}

const ALL_STATES = computeAllStates()
const TOTAL_STEPS = ALL_STATES.length
const NAIVE_BYTES = SEQ_LEN * SEQ_LEN * 4

function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return '-inf'
  return v.toFixed(digits)
}

function fmtVec(v: Vec): string {
  return `[${fmt(v[0])}, ${fmt(v[1])}]`
}

function setCellState(el: Element, state: 'pending' | 'active' | 'done' | 'faded') {
  el.classList.remove('is-active', 'is-done', 'is-faded')
  if (state === 'active') el.classList.add('is-active')
  else if (state === 'done') el.classList.add('is-done')
  else if (state === 'faded') el.classList.add('is-faded')
}

function setArrowState(el: Element, active: boolean) {
  el.classList.toggle('is-active', active)
}

function renderStep(root: HTMLElement, idx: number) {
  if (idx < 0 || idx >= TOTAL_STEPS) return
  const state = ALL_STATES[idx]

  for (let i = 0; i < SEQ_LEN; i++) {
    const qCell = root.querySelector(`[data-fat-q-row="${i}"]`)
    const oCell = root.querySelector(`[data-fat-o-row="${i}"]`)
    if (qCell) {
      if (i >= state.qRows[0] && i <= state.qRows[1]) setCellState(qCell, 'active')
      else setCellState(qCell, 'faded')
    }
    if (oCell) {
      if (i >= state.qRows[0] && i <= state.qRows[1]) setCellState(oCell, 'active')
      else if (idx > 0 && hasBeenWritten(idx, i)) setCellState(oCell, 'done')
      else setCellState(oCell, 'faded')
    }
  }
  for (let j = 0; j < SEQ_LEN; j++) {
    const kCell = root.querySelector(`[data-fat-k-row="${j}"]`)
    const vCell = root.querySelector(`[data-fat-v-row="${j}"]`)
    const inTile = j >= state.kCols[0] && j <= state.kCols[1]
    const beenLoaded = j < state.kCols[0]
    const tileState = inTile ? 'active' : beenLoaded ? 'done' : 'faded'
    if (kCell) setCellState(kCell, tileState)
    if (vCell) setCellState(vCell, tileState)
  }

  for (const arrow of root.querySelectorAll('[data-fat-arrow]')) {
    setArrowState(arrow, true)
  }

  const stepReadout = root.querySelector('[data-fat-step-readout]')
  if (stepReadout) {
    stepReadout.innerHTML = `Step <strong>${state.step}</strong> / ${TOTAL_STEPS}; outer j=<strong>${state.outerJ + 1}</strong>, inner i=<strong>${state.innerI + 1}</strong>`
  }

  const sramQ = root.querySelector('[data-fat-sram-q]')
  if (sramQ)
    sramQ.innerHTML = `<i>Q</i><sub>${state.innerI + 1}</sub> <small>(rows ${state.qRows[0] + 1}-${state.qRows[1] + 1})</small>`
  const sramK = root.querySelector('[data-fat-sram-k]')
  if (sramK)
    sramK.innerHTML = `<i>K</i><sub>${state.outerJ + 1}</sub> <small>(cols ${state.kCols[0] + 1}-${state.kCols[1] + 1})</small>`
  const sramV = root.querySelector('[data-fat-sram-v]')
  if (sramV) sramV.innerHTML = `<i>V</i><sub>${state.outerJ + 1}</sub>`

  const stats: Record<string, string> = {
    'm-0': fmt(state.m[0]),
    'm-1': fmt(state.m[1]),
    'l-0': fmt(state.l[0]),
    'l-1': fmt(state.l[1]),
    'o-0': fmtVec(state.O[0]),
    'o-1': fmtVec(state.O[1]),
  }
  for (const [key, val] of Object.entries(stats)) {
    const el = root.querySelector(`[data-fat-stat="${key}"]`)
    if (el) el.textContent = val
  }

  const ratio = root.querySelector('[data-fat-ratio]')
  if (ratio) {
    const saved = (NAIVE_BYTES / state.bytesHbm).toFixed(2)
    ratio.innerHTML = `HBM bytes touched: <strong>${state.bytesHbm}</strong>; naive baseline: <strong>${NAIVE_BYTES}</strong>; ratio <strong>${saved}x</strong>`
  }

  const prev = root.querySelector<HTMLButtonElement>('[data-fat-prev]')
  const next = root.querySelector<HTMLButtonElement>('[data-fat-next]')
  if (prev) prev.disabled = idx <= 0
  if (next) next.disabled = idx >= TOTAL_STEPS - 1
}

function hasBeenWritten(stepIdx: number, row: number): boolean {
  for (let s = 0; s < stepIdx; s++) {
    const prev = ALL_STATES[s]
    if (row >= prev.qRows[0] && row <= prev.qRows[1]) return true
  }
  return false
}

function setupFlashAttentionTiles() {
  const roots = document.querySelectorAll<HTMLElement>('[data-flash-attention-tiles]')
  for (const root of roots) {
    if (root.dataset.fatBound) continue
    root.dataset.fatBound = 'true'

    let idx = 0
    renderStep(root, idx)

    const next = root.querySelector<HTMLButtonElement>('[data-fat-next]')
    const prev = root.querySelector<HTMLButtonElement>('[data-fat-prev]')
    const reset = root.querySelector<HTMLButtonElement>('[data-fat-reset]')

    const handleNext = () => {
      if (idx < TOTAL_STEPS - 1) {
        idx += 1
        renderStep(root, idx)
      }
    }
    const handlePrev = () => {
      if (idx > 0) {
        idx -= 1
        renderStep(root, idx)
      }
    }
    const handleReset = () => {
      idx = 0
      renderStep(root, idx)
    }

    next?.addEventListener('click', handleNext)
    prev?.addEventListener('click', handlePrev)
    reset?.addEventListener('click', handleReset)

    window.addCleanup(() => {
      next?.removeEventListener('click', handleNext)
      prev?.removeEventListener('click', handlePrev)
      reset?.removeEventListener('click', handleReset)
      delete root.dataset.fatBound
    })
  }
}

document.addEventListener('nav', setupFlashAttentionTiles)

export {}
