export {}

const SVG_NS = 'http://www.w3.org/2000/svg'
const PEC_LENGTHS = [8, 16, 32, 64] as const
const PEC_DEFAULT_DIM = 16
const PEC_ABS_DIM = 16
const PEC_ROPE_PAIRS = 4
const PEC_ROPE_POSITIONS = 8

const pecAbsValue = (pos: number, dim: number, totalDim: number): number => {
  const halfIndex = Math.floor(dim / 2)
  const denom = Math.pow(10000, (2 * halfIndex) / totalDim)
  const phase = pos / denom
  return dim % 2 === 0 ? Math.sin(phase) : Math.cos(phase)
}

const pecRelBias = (delta: number, length: number): number => {
  if (length <= 1) return 0
  const abs = Math.abs(delta)
  const norm = Math.sqrt(abs) / Math.sqrt(length - 1)
  return Math.max(0, 1 - norm)
}

const pecAlibiSlope = (length: number): number => 4 / Math.max(1, length - 1)

const pecRopeAngle = (pos: number, pairIndex: number, totalDim: number): number => {
  const denom = Math.pow(10000, (2 * pairIndex) / totalDim)
  return pos / denom
}

const pecLerpColor = (value: number, root: HTMLElement): string => {
  const styles = getComputedStyle(root)
  const pos = styles.getPropertyValue('--pec-pos').trim() || '#fdb2a2'
  const neutral = styles.getPropertyValue('--pec-neutral').trim() || '#cdd597'
  const neg = styles.getPropertyValue('--pec-neg').trim() || '#808080'
  const v = Math.max(-1, Math.min(1, value))
  if (v >= 0) {
    const mix = Math.round(v * 78)
    return `color-mix(in srgb, ${pos} ${mix}%, ${neutral})`
  }
  const mix = Math.round(-v * 65)
  return `color-mix(in srgb, ${neg} ${mix}%, ${neutral})`
}

const pecMagFill = (mag: number, root: HTMLElement): string => {
  const styles = getComputedStyle(root)
  const pos = styles.getPropertyValue('--pec-pos').trim() || '#fdb2a2'
  const neutral = styles.getPropertyValue('--pec-neutral').trim() || '#cdd597'
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 82)
  return `color-mix(in srgb, ${pos} ${mix}%, ${neutral})`
}

const pecPenaltyFill = (mag: number, root: HTMLElement): string => {
  const styles = getComputedStyle(root)
  const neg = styles.getPropertyValue('--pec-neg').trim() || '#808080'
  const neutral = styles.getPropertyValue('--pec-neutral').trim() || '#cdd597'
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 78)
  return `color-mix(in srgb, ${neg} ${mix}%, ${neutral})`
}

const pecBuildAbsolute = (host: SVGSVGElement, length: number, root: HTMLElement) => {
  while (host.firstChild) host.removeChild(host.firstChild)
  host.setAttribute('viewBox', `0 0 ${PEC_ABS_DIM} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  for (let p = 0; p < length; p++) {
    for (let d = 0; d < PEC_ABS_DIM; d++) {
      const cell = document.createElementNS(SVG_NS, 'rect')
      cell.setAttribute('class', 'pec-cell')
      cell.setAttribute('x', String(d))
      cell.setAttribute('y', String(p))
      cell.setAttribute('width', '1')
      cell.setAttribute('height', '1')
      cell.setAttribute('fill', pecLerpColor(pecAbsValue(p, d, PEC_ABS_DIM), root))
      host.appendChild(cell)
    }
  }
}

const pecBuildRelative = (host: SVGSVGElement, length: number, root: HTMLElement) => {
  while (host.firstChild) host.removeChild(host.firstChild)
  host.setAttribute('viewBox', `0 0 ${length} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  for (let i = 0; i < length; i++) {
    for (let j = 0; j < length; j++) {
      const cell = document.createElementNS(SVG_NS, 'rect')
      cell.setAttribute('class', 'pec-cell')
      cell.setAttribute('x', String(j))
      cell.setAttribute('y', String(i))
      cell.setAttribute('width', '1')
      cell.setAttribute('height', '1')
      cell.setAttribute('fill', pecMagFill(pecRelBias(j - i, length), root))
      host.appendChild(cell)
    }
  }
}

const pecBuildAlibi = (host: SVGSVGElement, length: number, root: HTMLElement) => {
  while (host.firstChild) host.removeChild(host.firstChild)
  host.setAttribute('viewBox', `0 0 ${length} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  const m = pecAlibiSlope(length)
  for (let i = 0; i < length; i++) {
    for (let j = 0; j < length; j++) {
      const cell = document.createElementNS(SVG_NS, 'rect')
      cell.setAttribute('class', 'pec-cell')
      cell.setAttribute('x', String(j))
      cell.setAttribute('y', String(i))
      cell.setAttribute('width', '1')
      cell.setAttribute('height', '1')
      const penalty = m * Math.abs(j - i)
      cell.setAttribute('fill', pecPenaltyFill(Math.min(1, penalty), root))
      host.appendChild(cell)
    }
  }
}

const pecBuildRope = (host: SVGSVGElement, length: number, _root: HTMLElement) => {
  while (host.firstChild) host.removeChild(host.firstChild)
  const cols = PEC_ROPE_PAIRS
  const rows = Math.min(PEC_ROPE_POSITIONS, length)
  host.setAttribute('viewBox', `0 0 ${cols} ${rows}`)
  host.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  host.removeAttribute('shape-rendering')
  const step = Math.max(1, Math.floor(length / rows))
  const totalDim = PEC_DEFAULT_DIM
  for (let r = 0; r < rows; r++) {
    const p = r * step
    for (let c = 0; c < cols; c++) {
      const angle = pecRopeAngle(p, c, totalDim)
      const cx = c + 0.5
      const cy = r + 0.5
      const radius = 0.36
      const handLen = radius * 0.6
      const bg = document.createElementNS(SVG_NS, 'rect')
      bg.setAttribute('class', 'pec-cell')
      bg.setAttribute('x', String(c))
      bg.setAttribute('y', String(r))
      bg.setAttribute('width', '1')
      bg.setAttribute('height', '1')
      bg.setAttribute('fill', 'transparent')
      host.appendChild(bg)
      const dial = document.createElementNS(SVG_NS, 'circle')
      dial.setAttribute('class', 'pec-clock-bg')
      dial.setAttribute('cx', String(cx))
      dial.setAttribute('cy', String(cy))
      dial.setAttribute('r', String(radius))
      host.appendChild(dial)
      const ring = document.createElementNS(SVG_NS, 'circle')
      ring.setAttribute('class', 'pec-clock')
      ring.setAttribute('cx', String(cx))
      ring.setAttribute('cy', String(cy))
      ring.setAttribute('r', String(radius))
      host.appendChild(ring)
      const hand = document.createElementNS(SVG_NS, 'line')
      hand.setAttribute('class', 'pec-clock-hand')
      hand.setAttribute('x1', String(cx))
      hand.setAttribute('y1', String(cy))
      hand.setAttribute('x2', String(cx + handLen * Math.cos(angle)))
      hand.setAttribute('y2', String(cy + handLen * Math.sin(angle)))
      host.appendChild(hand)
    }
  }
}

const pecBuildPanel = (root: HTMLElement, kind: string, length: number) => {
  const host = root.querySelector<SVGSVGElement>(`[data-pec-vis="${kind}"]`)
  if (!host) return
  if (kind === 'absolute') pecBuildAbsolute(host, length, root)
  else if (kind === 'relative') pecBuildRelative(host, length, root)
  else if (kind === 'rope') pecBuildRope(host, length, root)
  else if (kind === 'alibi') pecBuildAlibi(host, length, root)
}

const pecRenderAll = (root: HTMLElement, length: number) => {
  pecBuildPanel(root, 'absolute', length)
  pecBuildPanel(root, 'relative', length)
  pecBuildPanel(root, 'rope', length)
  pecBuildPanel(root, 'alibi', length)
  root.dataset.pecLength = String(length)
}

const pecUpdateLengthButtons = (root: HTMLElement, length: number) => {
  for (const btn of root.querySelectorAll<HTMLButtonElement>('[data-pec-length-btn]')) {
    const value = Number(btn.dataset.pecLengthBtn ?? '0')
    const active = value === length
    btn.classList.toggle('is-active', active)
    btn.setAttribute('aria-selected', active ? 'true' : 'false')
    btn.tabIndex = active ? 0 : -1
  }
}

const pecSetupRoot = (root: HTMLElement) => {
  if (root.dataset.pecBound === 'true') return
  root.dataset.pecBound = 'true'

  const initial = Number(root.dataset.pecLength ?? '16')
  const length = (PEC_LENGTHS as readonly number[]).includes(initial) ? initial : 16
  root.dataset.pecLength = String(length)
  pecUpdateLengthButtons(root, length)

  const handlers: Array<() => void> = []
  const buttons = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-pec-length-btn]'))
  const selectAt = (index: number, focus: boolean) => {
    const btn = buttons[index]
    if (!btn) return
    const next = Number(btn.dataset.pecLengthBtn ?? '16')
    if (!(PEC_LENGTHS as readonly number[]).includes(next)) return
    pecRenderAll(root, next)
    pecUpdateLengthButtons(root, next)
    if (focus) btn.focus()
  }
  buttons.forEach((btn, index) => {
    const onClick = () => selectAt(index, false)
    const onKey = (event: KeyboardEvent) => {
      const last = buttons.length - 1
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
        event.preventDefault()
        selectAt(index === last ? 0 : index + 1, true)
      } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
        event.preventDefault()
        selectAt(index === 0 ? last : index - 1, true)
      } else if (event.key === 'Home') {
        event.preventDefault()
        selectAt(0, true)
      } else if (event.key === 'End') {
        event.preventDefault()
        selectAt(last, true)
      }
    }
    btn.addEventListener('click', onClick)
    btn.addEventListener('keydown', onKey)
    handlers.push(() => btn.removeEventListener('click', onClick))
    handlers.push(() => btn.removeEventListener('keydown', onKey))
  })

  const switchInput = root.querySelector<HTMLInputElement>('[data-pec-logit-toggle]')
  const onSwitch = () => {
    root.dataset.pecLogit = switchInput?.checked ? 'true' : 'false'
  }
  switchInput?.addEventListener('change', onSwitch)

  window.addCleanup(() => {
    for (const off of handlers) off()
    switchInput?.removeEventListener('change', onSwitch)
    delete root.dataset.pecBound
  })
}

const pecSetup = () => {
  for (const root of document.querySelectorAll<HTMLElement>(
    '[data-positional-encoding-comparison]',
  )) {
    pecSetupRoot(root)
  }
}

document.addEventListener('nav', pecSetup)
