type RingState = { p: number; step: number; playing: boolean }

const VIEW = 360
const CENTER = VIEW / 2
const RING_R = 118
const CHIP_R = 162
const TOKEN_R = 16
const STEP_DURATION_MS = 720
const PULSE_DURATION_MS = 360
const PLAY_GAP_MS = 220
const MIN_DEVICES = 2
const MAX_DEVICES = 8

const RR_ROOT = '[data-ring-rotation]'

const polar = (angleDeg: number, radius: number) => {
  const rad = (angleDeg * Math.PI) / 180
  return { x: CENTER + radius * Math.cos(rad), y: CENTER + radius * Math.sin(rad) }
}

const deviceAngle = (i: number, p: number) => -90 + (360 * i) / p

const mod = (n: number, m: number) => ((n % m) + m) % m

const rrDeviceHtml = (i: number): string => `<span class="rr-dyn-math">d<sub>${i}</sub></span>`
const rrSliceHtml = (i: number): string =>
  `<span class="rr-dyn-math">k<sub>${i}</sub>,v<sub>${i}</sub></span>`

const ringArcPath = (p: number) => {
  const start = polar(deviceAngle(0, p), RING_R)
  const segs: string[] = [`M ${start.x.toFixed(2)} ${start.y.toFixed(2)}`]
  for (let i = 1; i <= p; i++) {
    const pt = polar(deviceAngle(i % p, p), RING_R)
    segs.push(`A ${RING_R} ${RING_R} 0 0 1 ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
  }
  return segs.join(' ')
}

const clampInt = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3)

const renderNodes = (svg: SVGSVGElement, p: number) => {
  const flow = svg.querySelector<SVGPathElement>('.rr-ring-flow')
  if (flow) flow.setAttribute('d', ringArcPath(p))

  const nodesGroup = svg.querySelector<SVGGElement>('[data-rr-nodes]')
  if (!nodesGroup) return

  const SVG_NS = 'http://www.w3.org/2000/svg'
  while (nodesGroup.firstChild) nodesGroup.removeChild(nodesGroup.firstChild)

  for (let i = 0; i < p; i++) {
    const { x, y } = polar(deviceAngle(i, p), RING_R)
    const chip = polar(deviceAngle(i, p), CHIP_R)

    const g = document.createElementNS(SVG_NS, 'g')
    g.setAttribute('class', 'rr-node')
    g.setAttribute('data-rr-node', String(i))

    const disc = document.createElementNS(SVG_NS, 'circle')
    disc.setAttribute('class', 'rr-node-disc')
    disc.setAttribute('cx', String(x))
    disc.setAttribute('cy', String(y))
    disc.setAttribute('r', '22')
    g.appendChild(disc)

    const labelFo = document.createElementNS(SVG_NS, 'foreignObject')
    labelFo.setAttribute('x', String(x - 18))
    labelFo.setAttribute('y', String(y - 12))
    labelFo.setAttribute('width', '36')
    labelFo.setAttribute('height', '24')

    const label = document.createElement('div')
    label.setAttribute('class', 'rr-fo rr-fo--node')
    label.innerHTML = rrDeviceHtml(i)
    labelFo.appendChild(label)
    g.appendChild(labelFo)

    const chipRect = document.createElementNS(SVG_NS, 'rect')
    chipRect.setAttribute('class', 'rr-slice-chip')
    chipRect.setAttribute('x', String(chip.x - 26))
    chipRect.setAttribute('y', String(chip.y - 13))
    chipRect.setAttribute('width', '52')
    chipRect.setAttribute('height', '26')
    chipRect.setAttribute('rx', '4')
    chipRect.setAttribute('data-rr-chip', String(i))
    g.appendChild(chipRect)

    const fo = document.createElementNS(SVG_NS, 'foreignObject')
    fo.setAttribute('x', String(chip.x - 26))
    fo.setAttribute('y', String(chip.y - 13))
    fo.setAttribute('width', '52')
    fo.setAttribute('height', '26')

    const inner = document.createElement('div')
    inner.setAttribute('class', 'rr-fo rr-fo--chip')
    inner.innerHTML = rrSliceHtml(i)
    fo.appendChild(inner)
    g.appendChild(fo)

    nodesGroup.appendChild(g)
  }
}

const renderMatrix = (root: HTMLElement, p: number) => {
  const grid = root.querySelector<HTMLElement>('[data-rr-matrix]')
  if (!grid) return
  grid.style.setProperty('--rr-p', String(p))
  while (grid.firstChild) grid.removeChild(grid.firstChild)
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      const cell = document.createElement('span')
      cell.className = 'rr-cell'
      cell.setAttribute('data-rr-cell', `${i}-${j}`)
      cell.setAttribute('data-rr-row', String(i))
      cell.setAttribute('data-rr-col', String(j))
      cell.setAttribute('role', 'gridcell')
      cell.setAttribute('aria-label', `device ${i} has slice ${j}: pending`)
      grid.appendChild(cell)
    }
  }
}

const updateCell = (
  root: HTMLElement,
  i: number,
  j: number,
  state: 'empty' | 'filled' | 'pulse' | 'diag',
) => {
  const cell = root.querySelector<HTMLElement>(`[data-rr-cell="${i}-${j}"]`)
  if (!cell) return
  if (state === 'empty') {
    cell.removeAttribute('data-rr-state')
    cell.setAttribute('aria-label', `device ${i} has slice ${j}: pending`)
  } else {
    cell.setAttribute('data-rr-state', state)
    cell.setAttribute(
      'aria-label',
      `device ${i} has slice ${j}: ${state === 'diag' ? 'local' : 'received'}`,
    )
  }
}

const paintMatrixToStep = (root: HTMLElement, state: RingState) => {
  for (let i = 0; i < state.p; i++) {
    for (let j = 0; j < state.p; j++) {
      if (i === j) {
        updateCell(root, i, j, 'diag')
      } else {
        const stepLit = mod(i - j, state.p)
        if (stepLit > 0 && stepLit <= state.step) updateCell(root, i, j, 'filled')
        else updateCell(root, i, j, 'empty')
      }
    }
  }
}

const updateReadout = (root: HTMLElement, state: RingState) => {
  const mem = root.querySelector<HTMLElement>('[data-rr-mem]')
  const comm = root.querySelector<HTMLElement>('[data-rr-comm]')
  const rounds = root.querySelector<HTMLElement>('[data-rr-rounds]')
  const stepEl = root.querySelector<HTMLElement>('[data-rr-step]')
  const stepTotal = root.querySelector<HTMLElement>('[data-rr-step-total]')
  const sliderVal = root.querySelector<HTMLElement>('[data-rr-devices-value]')

  if (mem) mem.textContent = `L/${state.p} d`
  if (comm) comm.textContent = `${state.p - 1} L/${state.p} d`
  if (rounds) rounds.textContent = String(state.p - 1)
  if (stepEl) stepEl.textContent = String(state.step)
  if (stepTotal) stepTotal.textContent = String(state.p - 1)
  if (sliderVal) sliderVal.textContent = String(state.p)
}

const updateButtons = (root: HTMLElement, state: RingState) => {
  const stepBtn = root.querySelector<HTMLButtonElement>('[data-rr-step-btn]')
  const playBtn = root.querySelector<HTMLButtonElement>('[data-rr-play]')
  const playLabel = root.querySelector<HTMLElement>('[data-rr-play-label]')
  const resetBtn = root.querySelector<HTMLButtonElement>('[data-rr-reset]')

  const atEnd = state.step >= state.p - 1
  if (stepBtn) stepBtn.disabled = state.playing || atEnd
  if (resetBtn) resetBtn.disabled = state.playing || state.step === 0
  if (playBtn) {
    playBtn.disabled = atEnd && !state.playing
    playBtn.setAttribute('aria-pressed', state.playing ? 'true' : 'false')
  }
  if (playLabel) playLabel.textContent = state.playing ? 'pause' : atEnd ? 'done' : 'play'
}

const setActiveSenders = (svg: SVGSVGElement, sender: number | null) => {
  const nodes = svg.querySelectorAll<SVGGElement>('[data-rr-node]')
  for (const node of nodes) {
    if (sender !== null && Number(node.getAttribute('data-rr-node')) === sender) {
      node.setAttribute('data-rr-active', 'true')
      const chip = node.querySelector<SVGRectElement>('[data-rr-chip]')
      chip?.setAttribute('data-rr-chip-traveling', 'true')
    } else {
      node.removeAttribute('data-rr-active')
      const chip = node.querySelector<SVGRectElement>('[data-rr-chip]')
      chip?.removeAttribute('data-rr-chip-traveling')
    }
  }
}

const animateToken = (
  svg: SVGSVGElement,
  fromIdx: number,
  toIdx: number,
  sliceIdx: number,
  p: number,
  signal: AbortSignal,
): Promise<void> => {
  const token = svg.querySelector<SVGCircleElement>('[data-rr-token]')
  const tokenText = svg.querySelector<SVGForeignObjectElement>('[data-rr-token-text]')
  if (!token || !tokenText) return Promise.resolve()

  const fromAngle = deviceAngle(fromIdx, p)
  const toAngle = deviceAngle(toIdx, p)
  let delta = toAngle - fromAngle
  if (delta <= -180) delta += 360
  if (delta > 180) delta -= 360

  const tokenTextLabel = tokenText.querySelector<HTMLElement>('.rr-fo--token')
  if (tokenTextLabel) tokenTextLabel.innerHTML = rrSliceHtml(sliceIdx)
  else tokenText.textContent = `k${sliceIdx},v${sliceIdx}`
  token.setAttribute('data-rr-token-visible', 'true')
  tokenText.setAttribute('data-rr-token-visible', 'true')

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (reduced) {
    const end = polar(toAngle, RING_R)
    token.setAttribute('cx', String(end.x))
    token.setAttribute('cy', String(end.y))
    tokenText.setAttribute('x', String(end.x - TOKEN_R))
    tokenText.setAttribute('y', String(end.y - TOKEN_R / 2))
    token.setAttribute('r', String(TOKEN_R))
    return new Promise(resolve => {
      const timer = window.setTimeout(() => resolve(), 60)
      signal.addEventListener('abort', () => {
        window.clearTimeout(timer)
        resolve()
      })
    })
  }

  return new Promise(resolve => {
    const start = performance.now()
    let raf = 0

    const tick = (now: number) => {
      if (signal.aborted) {
        token.removeAttribute('data-rr-token-visible')
        tokenText.removeAttribute('data-rr-token-visible')
        resolve()
        return
      }
      const raw = (now - start) / STEP_DURATION_MS
      const t = raw < 0 ? 0 : raw > 1 ? 1 : raw
      const eased = easeOutCubic(t)
      const angle = fromAngle + delta * eased
      const { x, y } = polar(angle, RING_R)
      token.setAttribute('cx', String(x))
      token.setAttribute('cy', String(y))
      tokenText.setAttribute('x', String(x - TOKEN_R))
      tokenText.setAttribute('y', String(y - TOKEN_R / 2))
      if (t < 1) {
        raf = requestAnimationFrame(tick)
      } else {
        resolve()
      }
    }

    raf = requestAnimationFrame(tick)
    signal.addEventListener('abort', () => {
      cancelAnimationFrame(raf)
      token.removeAttribute('data-rr-token-visible')
      tokenText.removeAttribute('data-rr-token-visible')
      resolve()
    })
  })
}

const pulseReceivers = (root: HTMLElement, state: RingState, signal: AbortSignal) => {
  const receivers: Array<{ row: number; col: number }> = []
  for (let i = 0; i < state.p; i++) {
    const j = mod(i - state.step, state.p)
    if (j === i) continue
    receivers.push({ row: i, col: j })
  }
  for (const { row, col } of receivers) updateCell(root, row, col, 'pulse')
  return new Promise<void>(resolve => {
    const timer = window.setTimeout(() => {
      if (!signal.aborted) {
        for (const { row, col } of receivers) updateCell(root, row, col, 'filled')
      }
      resolve()
    }, PULSE_DURATION_MS)
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer)
      resolve()
    })
  })
}

const advanceOnce = async (
  root: HTMLElement,
  svg: SVGSVGElement,
  state: RingState,
  signal: AbortSignal,
) => {
  if (state.step >= state.p - 1) return
  const sourceStep = state.step
  for (let sender = 0; sender < state.p; sender++) {
    const receiver = mod(sender + 1, state.p)
    const sliceOriginator = mod(sender - sourceStep, state.p)
    setActiveSenders(svg, sender)
    await animateToken(svg, sender, receiver, sliceOriginator, state.p, signal)
    if (signal.aborted) return
  }
  setActiveSenders(svg, null)
  state.step = sourceStep + 1
  await pulseReceivers(root, state, signal)
  updateReadout(root, state)
  updateButtons(root, state)
}

const resetState = (
  root: HTMLElement,
  svg: SVGSVGElement,
  state: RingState,
  controller: AbortController,
) => {
  controller.abort()
  state.step = 0
  state.playing = false
  setActiveSenders(svg, null)
  const token = svg.querySelector<SVGCircleElement>('[data-rr-token]')
  const tokenText = svg.querySelector<SVGForeignObjectElement>('[data-rr-token-text]')
  token?.removeAttribute('data-rr-token-visible')
  tokenText?.removeAttribute('data-rr-token-visible')
  paintMatrixToStep(root, state)
  updateReadout(root, state)
  updateButtons(root, state)
}

type Bound = { controllers: Set<AbortController>; state: RingState }

const wireOne = (root: HTMLElement): (() => void) => {
  const svg = root.querySelector<SVGSVGElement>('[data-rr-ring]')
  if (!svg) return () => {}

  const initial = clampInt(
    Number(root.dataset.devicesInitial ?? '4') || 4,
    MIN_DEVICES,
    MAX_DEVICES,
  )

  const state: RingState = { p: initial, step: 0, playing: false }
  const bound: Bound = { controllers: new Set(), state }

  const slider = root.querySelector<HTMLInputElement>('[data-rr-devices]')
  const stepBtn = root.querySelector<HTMLButtonElement>('[data-rr-step-btn]')
  const playBtn = root.querySelector<HTMLButtonElement>('[data-rr-play]')
  const resetBtn = root.querySelector<HTMLButtonElement>('[data-rr-reset]')

  const newController = (): AbortController => {
    const c = new AbortController()
    bound.controllers.add(c)
    return c
  }
  const cancelAll = () => {
    for (const c of bound.controllers) c.abort()
    bound.controllers.clear()
  }

  const refresh = () => {
    renderNodes(svg, state.p)
    renderMatrix(root, state.p)
    paintMatrixToStep(root, state)
    updateReadout(root, state)
    updateButtons(root, state)
  }

  refresh()

  const handleStep = async () => {
    if (state.playing) return
    const controller = newController()
    await advanceOnce(root, svg, state, controller.signal)
    bound.controllers.delete(controller)
  }

  const handlePlay = async () => {
    if (state.playing) {
      state.playing = false
      cancelAll()
      updateButtons(root, state)
      return
    }
    if (state.step >= state.p - 1) return
    state.playing = true
    updateButtons(root, state)
    const controller = newController()
    while (state.playing && state.step < state.p - 1) {
      await advanceOnce(root, svg, state, controller.signal)
      if (controller.signal.aborted) break
      if (state.step < state.p - 1) {
        await new Promise<void>(resolve => {
          const timer = window.setTimeout(resolve, PLAY_GAP_MS)
          controller.signal.addEventListener('abort', () => {
            window.clearTimeout(timer)
            resolve()
          })
        })
      }
    }
    state.playing = false
    bound.controllers.delete(controller)
    updateButtons(root, state)
  }

  const handleReset = () => {
    const controller = newController()
    resetState(root, svg, state, controller)
    bound.controllers.delete(controller)
  }

  const handleSlider = (event: Event) => {
    const value = clampInt(
      Number((event.target as HTMLInputElement).value),
      MIN_DEVICES,
      MAX_DEVICES,
    )
    if (value === state.p) return
    cancelAll()
    state.p = value
    state.step = 0
    state.playing = false
    if (slider) slider.setAttribute('aria-valuenow', String(value))
    refresh()
  }

  stepBtn?.addEventListener('click', handleStep)
  playBtn?.addEventListener('click', handlePlay)
  resetBtn?.addEventListener('click', handleReset)
  slider?.addEventListener('input', handleSlider)

  return () => {
    cancelAll()
    stepBtn?.removeEventListener('click', handleStep)
    playBtn?.removeEventListener('click', handlePlay)
    resetBtn?.removeEventListener('click', handleReset)
    slider?.removeEventListener('input', handleSlider)
  }
}

const setup = () => {
  const roots = document.querySelectorAll<HTMLElement>(RR_ROOT)
  const teardowns: Array<() => void> = []
  for (const root of roots) teardowns.push(wireOne(root))
  window.addCleanup(() => {
    for (const t of teardowns) t()
  })
}

document.addEventListener('nav', setup)
