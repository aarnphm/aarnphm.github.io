export {}

const SVG_NS = 'http://www.w3.org/2000/svg'
const PEC_LENGTHS = [8, 16, 32, 64] as const
const PEC_DEFAULT_DIM = 16
const PEC_ABS_DIM = 16
const PEC_ROPE_PAIRS = 4
const PEC_ROPE_POSITIONS = 8

type PecPalette = { pos: string; neutral: string; neg: string }

const pecResolvePalette = (root: HTMLElement): PecPalette => {
  const styles = getComputedStyle(root)
  return {
    pos: styles.getPropertyValue('--pec-pos').trim() || '#fdb2a2',
    neutral: styles.getPropertyValue('--pec-neutral').trim() || '#cdd597',
    neg: styles.getPropertyValue('--pec-neg').trim() || '#808080',
  }
}

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

const pecLerpColor = (value: number, pal: PecPalette): string => {
  const v = Math.max(-1, Math.min(1, value))
  if (v >= 0) {
    const mix = Math.round(v * 78)
    return `color-mix(in srgb, ${pal.pos} ${mix}%, ${pal.neutral})`
  }
  const mix = Math.round(-v * 65)
  return `color-mix(in srgb, ${pal.neg} ${mix}%, ${pal.neutral})`
}

const pecMagFill = (mag: number, pal: PecPalette): string => {
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 82)
  return `color-mix(in srgb, ${pal.pos} ${mix}%, ${pal.neutral})`
}

const pecPenaltyFill = (mag: number, pal: PecPalette): string => {
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 78)
  return `color-mix(in srgb, ${pal.neg} ${mix}%, ${pal.neutral})`
}

const pecCell = (x: number, y: number, fill: string): SVGRectElement => {
  const cell = document.createElementNS(SVG_NS, 'rect')
  cell.setAttribute('class', 'pec-cell')
  cell.setAttribute('x', String(x))
  cell.setAttribute('y', String(y))
  cell.setAttribute('width', '1')
  cell.setAttribute('height', '1')
  cell.setAttribute('fill', fill)
  return cell
}

const pecClear = (host: SVGSVGElement) => {
  while (host.firstChild) host.removeChild(host.firstChild)
}

const pecBuildAbsolute = (host: SVGSVGElement, length: number, pal: PecPalette) => {
  pecClear(host)
  host.setAttribute('viewBox', `0 0 ${PEC_ABS_DIM} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  const frag = document.createDocumentFragment()
  for (let p = 0; p < length; p++) {
    for (let d = 0; d < PEC_ABS_DIM; d++) {
      frag.appendChild(pecCell(d, p, pecLerpColor(pecAbsValue(p, d, PEC_ABS_DIM), pal)))
    }
  }
  host.appendChild(frag)
}

const pecBuildRelative = (host: SVGSVGElement, length: number, pal: PecPalette) => {
  pecClear(host)
  host.setAttribute('viewBox', `0 0 ${length} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  const frag = document.createDocumentFragment()
  for (let i = 0; i < length; i++) {
    for (let j = 0; j < length; j++) {
      frag.appendChild(pecCell(j, i, pecMagFill(pecRelBias(j - i, length), pal)))
    }
  }
  host.appendChild(frag)
}

const pecBuildAlibi = (host: SVGSVGElement, length: number, pal: PecPalette) => {
  pecClear(host)
  host.setAttribute('viewBox', `0 0 ${length} ${length}`)
  host.setAttribute('shape-rendering', 'crispEdges')
  host.setAttribute('preserveAspectRatio', 'none')
  const m = pecAlibiSlope(length)
  const frag = document.createDocumentFragment()
  for (let i = 0; i < length; i++) {
    for (let j = 0; j < length; j++) {
      frag.appendChild(pecCell(j, i, pecPenaltyFill(Math.min(1, m * Math.abs(j - i)), pal)))
    }
  }
  host.appendChild(frag)
}

const pecBuildRope = (host: SVGSVGElement, length: number) => {
  pecClear(host)
  const cols = PEC_ROPE_PAIRS
  const rows = Math.min(PEC_ROPE_POSITIONS, length)
  host.setAttribute('viewBox', `0 0 ${cols} ${rows}`)
  host.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  host.removeAttribute('shape-rendering')
  const step = Math.max(1, Math.floor(length / rows))
  const totalDim = PEC_DEFAULT_DIM
  const radius = 0.36
  const handLen = radius * 0.6
  const frag = document.createDocumentFragment()
  for (let r = 0; r < rows; r++) {
    const p = r * step
    for (let c = 0; c < cols; c++) {
      const angle = pecRopeAngle(p, c, totalDim)
      const cx = c + 0.5
      const cy = r + 0.5
      frag.appendChild(pecCell(c, r, 'transparent'))
      const dial = document.createElementNS(SVG_NS, 'circle')
      dial.setAttribute('class', 'pec-clock-bg')
      dial.setAttribute('cx', String(cx))
      dial.setAttribute('cy', String(cy))
      dial.setAttribute('r', String(radius))
      frag.appendChild(dial)
      const ring = document.createElementNS(SVG_NS, 'circle')
      ring.setAttribute('class', 'pec-clock')
      ring.setAttribute('cx', String(cx))
      ring.setAttribute('cy', String(cy))
      ring.setAttribute('r', String(radius))
      frag.appendChild(ring)
      const hand = document.createElementNS(SVG_NS, 'line')
      hand.setAttribute('class', 'pec-clock-hand')
      hand.setAttribute('x1', String(cx))
      hand.setAttribute('y1', String(cy))
      hand.setAttribute('x2', String(cx + handLen * Math.cos(angle)))
      hand.setAttribute('y2', String(cy + handLen * Math.sin(angle)))
      frag.appendChild(hand)
    }
  }
  host.appendChild(frag)
}

const pecBuildPanel = (root: HTMLElement, kind: string, length: number, pal: PecPalette) => {
  const host = root.querySelector<SVGSVGElement>(`[data-pec-vis="${kind}"]`)
  if (!host) return
  if (kind === 'absolute') pecBuildAbsolute(host, length, pal)
  else if (kind === 'relative') pecBuildRelative(host, length, pal)
  else if (kind === 'rope') pecBuildRope(host, length)
  else if (kind === 'alibi') pecBuildAlibi(host, length, pal)
}

const pecRenderAll = (root: HTMLElement, length: number) => {
  const pal = pecResolvePalette(root)
  pecBuildPanel(root, 'absolute', length, pal)
  pecBuildPanel(root, 'relative', length, pal)
  pecBuildPanel(root, 'rope', length, pal)
  pecBuildPanel(root, 'alibi', length, pal)
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
