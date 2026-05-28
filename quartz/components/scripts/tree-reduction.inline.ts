type Triple = { m: number; z: number; y: number }

type Pulse = { stage: number; nodes: number[] }

const ROOT_SELECTOR = '[data-tree-reduction]'
const STEP_SELECTOR = '[data-tree-step]'
const RESET_SELECTOR = '[data-tree-reset]'
const STAGE_LABEL_SELECTOR = '[data-tree-stage]'
const NODE_SELECTOR = '[data-tree-node]'
const VALUE_SELECTOR = '[data-tree-value]'
const EDGE_SELECTOR = '[data-tree-edge]'
const PULSE_CLASS = 'tr-pulse'
const ACTIVE_CLASS = 'tr-active'
const SETTLED_CLASS = 'tr-settled'

function fmt(value: number): string {
  if (!Number.isFinite(value)) return '-'
  const abs = Math.abs(value)
  if (abs >= 100) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

function tripleLabel(t: Triple): string {
  return `(${fmt(t.m)}, ${fmt(t.z)}, ${fmt(t.y)})`
}

function mergeTriple(a: Triple, b: Triple): Triple {
  const m = Math.max(a.m, b.m)
  const wa = Math.exp(a.m - m)
  const wb = Math.exp(b.m - m)
  return { m, z: a.z * wa + b.z * wb, y: a.y * wa + b.y * wb }
}

function parseLeaves(root: HTMLElement): Triple[] {
  const raw = root.getAttribute('data-leaves-json')
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map((entry: unknown) => {
      if (entry && typeof entry === 'object') {
        const e = entry as Record<string, unknown>
        return { m: Number(e.m ?? 0), z: Number(e.z ?? 0), y: Number(e.y ?? 0) }
      }
      return { m: 0, z: 0, y: 0 }
    })
  } catch {
    return []
  }
}

function computeLevels(leaves: Triple[]): Triple[][] {
  const levels: Triple[][] = [leaves.slice()]
  while (levels[levels.length - 1].length > 1) {
    const prev = levels[levels.length - 1]
    const next: Triple[] = []
    for (let i = 0; i < prev.length; i += 2) {
      const a = prev[i]
      const b = prev[i + 1] ?? a
      next.push(mergeTriple(a, b))
    }
    levels.push(next)
  }
  return levels
}

function setNodeValue(root: HTMLElement, level: number, index: number, label: string) {
  const node = root.querySelector<HTMLElement>(
    `${NODE_SELECTOR}[data-level="${level}"][data-index="${index}"]`,
  )
  if (!node) return
  const valueHolder = node.querySelector<HTMLElement>(VALUE_SELECTOR)
  if (valueHolder) valueHolder.textContent = label
}

function setNodeClasses(
  root: HTMLElement,
  level: number,
  index: number,
  classes: { pulse?: boolean; active?: boolean; settled?: boolean },
) {
  const node = root.querySelector<HTMLElement>(
    `${NODE_SELECTOR}[data-level="${level}"][data-index="${index}"]`,
  )
  if (!node) return
  node.classList.toggle(PULSE_CLASS, classes.pulse ?? false)
  node.classList.toggle(ACTIVE_CLASS, classes.active ?? false)
  node.classList.toggle(SETTLED_CLASS, classes.settled ?? false)
}

function setEdgeActive(root: HTMLElement, level: number, index: number, active: boolean) {
  const edge = root.querySelector<HTMLElement>(
    `${EDGE_SELECTOR}[data-level="${level}"][data-index="${index}"]`,
  )
  if (!edge) return
  edge.classList.toggle(ACTIVE_CLASS, active)
}

function clearPulses(root: HTMLElement) {
  for (const node of root.querySelectorAll<HTMLElement>(NODE_SELECTOR)) {
    node.classList.remove(PULSE_CLASS)
  }
}

function render(root: HTMLElement, levels: Triple[][], stage: number, pulse: Pulse | null) {
  for (let lvl = 0; lvl < levels.length; lvl++) {
    const settled = lvl <= stage
    for (let idx = 0; idx < levels[lvl].length; idx++) {
      const visible = settled
      const label = visible ? tripleLabel(levels[lvl][idx]) : '-'
      setNodeValue(root, lvl, idx, label)
      const isPulse = pulse?.stage === lvl && pulse.nodes.includes(idx)
      setNodeClasses(root, lvl, idx, { pulse: isPulse, active: visible, settled })
    }
    if (lvl > 0) {
      const activeEdges = lvl <= stage
      for (let idx = 0; idx < levels[lvl].length; idx++) {
        setEdgeActive(root, lvl, idx * 2, activeEdges)
        setEdgeActive(root, lvl, idx * 2 + 1, activeEdges)
      }
    }
  }

  const rounds = levels.length - 1
  const stageLabel = root.querySelector<HTMLElement>(STAGE_LABEL_SELECTOR)
  if (stageLabel) stageLabel.textContent = `stage ${stage} of ${rounds}`

  const stepBtn = root.querySelector<HTMLButtonElement>(STEP_SELECTOR)
  if (stepBtn) {
    const isDone = stage >= rounds
    stepBtn.disabled = isDone
    stepBtn.setAttribute('aria-disabled', String(isDone))
  }
}

function setupReductionRoot(root: HTMLElement) {
  if (root.dataset.treeBound === 'true') return
  root.dataset.treeBound = 'true'

  const leaves = parseLeaves(root)
  if (leaves.length < 2) return
  const levels = computeLevels(leaves)
  let stage = 0

  const stepBtn = root.querySelector<HTMLButtonElement>(STEP_SELECTOR)
  const resetBtn = root.querySelector<HTMLButtonElement>(RESET_SELECTOR)

  render(root, levels, stage, null)

  const onStep = () => {
    const rounds = levels.length - 1
    if (stage >= rounds) return
    stage += 1
    const nodes = levels[stage].map((_, idx) => idx)
    render(root, levels, stage, { stage, nodes })
    window.setTimeout(() => {
      clearPulses(root)
    }, 720)
  }

  const onReset = () => {
    stage = 0
    clearPulses(root)
    render(root, levels, stage, null)
  }

  stepBtn?.addEventListener('click', onStep)
  resetBtn?.addEventListener('click', onReset)

  window.addCleanup?.(() => {
    stepBtn?.removeEventListener('click', onStep)
    resetBtn?.removeEventListener('click', onReset)
    root.dataset.treeBound = 'false'
  })
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ROOT_SELECTOR)) {
    setupReductionRoot(root)
  }
})

export {}
