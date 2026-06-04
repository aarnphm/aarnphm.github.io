import katex from 'katex'

type RzrPolicy = 'razor' | 'lru' | 'fifo'

type RzrSlotData = { token: string; score: number; insertedAt: number; lastAt: number }

type RzrHistoryEntry = {
  step: number
  token: string
  score: number
  fate: 'kept' | 'evicted'
  evictedToken?: string
}

type RzrState = {
  root: HTMLElement
  cap: number
  policy: RzrPolicy
  step: number
  evictions: number
  slots: (RzrSlotData | null)[]
  history: RzrHistoryEntry[]
  busy: boolean
  slotGroups: SVGGElement[]
  slotBars: SVGRectElement[]
  slotLabels: HTMLElement[]
  slotScores: HTMLElement[]
  historyList: HTMLOListElement
  statResidents: HTMLElement
  statEvictions: HTMLElement
  statAvg: HTMLElement
  statMass: HTMLElement
  policyBtns: HTMLButtonElement[]
  ruleBlocks: HTMLElement[]
  nextBtn: HTMLButtonElement
  resetBtn: HTMLButtonElement
}

const RZR_ROOT_SELECTOR = '[data-razor-evictor]'
const RZR_SLOT_H = 88
const RZR_HISTORY_MAX = 16
const RZR_EVICT_DELAY = 360

const rzrTokens = [
  'the',
  'razor',
  'cache',
  'token',
  'attn',
  'softmax',
  'score',
  'evict',
  'slot',
  'gpu',
  'mem',
  'predict',
  'shave',
  'prompt',
  'decode',
  'fold',
  'mask',
  'head',
  'rope',
  'kv',
  'q',
  'flash',
  'page',
  'ring',
  'tree',
  'cascade',
  'fifo',
  'lru',
  'hit',
  'miss',
]

function rzrMulberry(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function rzrSampleStep(step: number): { token: string; score: number } {
  const rng = rzrMulberry(0x9e3779b1 ^ (step * 0x85ebca6b))
  const token = rzrTokens[Math.floor(rng() * rzrTokens.length)]
  const u = rng()
  const v = rng()
  const score = Math.max(0.04, Math.min(0.99, 0.18 + 0.78 * Math.sqrt(u * v)))
  return { token, score }
}

function rzrPickEvictIdx(state: RzrState): number {
  const { slots, policy } = state
  let best = -1
  let bestKey = Number.POSITIVE_INFINITY
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i]
    if (!slot) continue
    let key: number
    if (policy === 'razor') key = slot.score
    else if (policy === 'lru') key = slot.lastAt
    else key = slot.insertedAt
    if (key < bestKey) {
      bestKey = key
      best = i
    }
  }
  return best
}

function rzrFmt(v: number): string {
  return v.toFixed(2)
}

function rzrTex(tex: string): string {
  try {
    return katex.renderToString(tex, {
      displayMode: false,
      output: 'html',
      throwOnError: false,
      strict: false,
    })
  } catch {
    return tex
  }
}

function rzrRenderSlot(state: RzrState, i: number, marker: 'stable' | 'new' | 'evict' | 'empty') {
  const group = state.slotGroups[i]
  const bar = state.slotBars[i]
  const label = state.slotLabels[i]
  const score = state.slotScores[i]
  if (!group || !bar || !label || !score) return
  group.classList.remove('is-stable', 'is-new', 'is-evict', 'is-empty')
  group.classList.add(`is-${marker}`)
  const slot = state.slots[i]
  if (slot && marker !== 'empty') {
    const h = Math.max(2, slot.score * (RZR_SLOT_H - 8))
    bar.setAttribute('y', String(RZR_SLOT_H - 4 - h))
    bar.setAttribute('height', String(h))
    label.textContent = slot.token
    score.textContent = rzrFmt(slot.score)
  } else {
    bar.setAttribute('y', String(RZR_SLOT_H - 4))
    bar.setAttribute('height', '0')
    label.textContent = '-'
    score.textContent = ''
  }
}

function rzrRenderMetrics(state: RzrState) {
  const residents = state.slots.filter(s => s !== null).length
  let sum = 0
  for (const slot of state.slots) if (slot) sum += slot.score
  const avg = residents > 0 ? sum / residents : 0
  let totalSeen = 0
  for (const entry of state.history) totalSeen += entry.score
  const mass = totalSeen > 0 ? sum / totalSeen : 0
  state.statResidents.innerHTML = rzrTex(String(residents))
  state.statEvictions.innerHTML = rzrTex(String(state.evictions))
  state.statAvg.innerHTML = rzrTex(rzrFmt(avg))
  state.statMass.innerHTML = rzrTex(`${Math.round(mass * 100)}\\%`)
}

function rzrRenderHistory(state: RzrState) {
  const list = state.historyList
  list.innerHTML = ''
  const recent = state.history.slice(-RZR_HISTORY_MAX)
  for (const entry of recent) {
    const li = document.createElement('li')
    li.className = `rzr-history-item is-${entry.fate}`
    li.setAttribute('data-rzr-fate', entry.fate)
    const tok = document.createElement('span')
    tok.className = 'rzr-history-token'
    tok.textContent = entry.token
    const sc = document.createElement('span')
    sc.className = 'rzr-history-score'
    sc.textContent = rzrFmt(entry.score)
    li.appendChild(tok)
    li.appendChild(sc)
    const tag = document.createElement('span')
    if (entry.fate === 'evicted') {
      tag.className = 'rzr-history-evicted'
      tag.textContent = entry.evictedToken ? `evict ${entry.evictedToken}` : 'evicted'
    } else {
      tag.className = 'rzr-history-kept'
      tag.textContent = 'accepted'
    }
    li.appendChild(tag)
    list.appendChild(li)
  }
}

function rzrRenderAll(state: RzrState) {
  for (let i = 0; i < state.cap; i++) {
    rzrRenderSlot(state, i, state.slots[i] ? 'stable' : 'empty')
  }
  rzrRenderMetrics(state)
  rzrRenderHistory(state)
}

function rzrCommitInsert(state: RzrState, idx: number, sample: { token: string; score: number }) {
  state.step += 1
  state.slots[idx] = {
    token: sample.token,
    score: sample.score,
    insertedAt: state.step,
    lastAt: state.step,
  }
  rzrRenderSlot(state, idx, 'new')
  rzrRenderMetrics(state)
  window.setTimeout(() => {
    if (state.slots[idx]?.insertedAt === state.step) {
      rzrRenderSlot(state, idx, 'stable')
    }
  }, RZR_EVICT_DELAY + 220)
}

function rzrHandleNext(state: RzrState) {
  if (state.busy) return
  const sample = rzrSampleStep(state.step + 1)
  const emptyIdx = state.slots.findIndex(s => s === null)
  if (emptyIdx >= 0) {
    rzrCommitInsert(state, emptyIdx, sample)
    state.history.push({ step: state.step, token: sample.token, score: sample.score, fate: 'kept' })
    rzrRenderHistory(state)
    return
  }
  const evictIdx = rzrPickEvictIdx(state)
  if (evictIdx < 0) return
  const evicted = state.slots[evictIdx]
  if (!evicted) return
  state.busy = true
  rzrRenderSlot(state, evictIdx, 'evict')
  window.setTimeout(() => {
    state.slots[evictIdx] = null
    state.evictions += 1
    rzrCommitInsert(state, evictIdx, sample)
    state.history.push({
      step: state.step,
      token: sample.token,
      score: sample.score,
      fate: 'evicted',
      evictedToken: evicted.token,
    })
    rzrRenderHistory(state)
    state.busy = false
  }, RZR_EVICT_DELAY)
}

function rzrHandleReset(state: RzrState) {
  state.slots = Array.from({ length: state.cap }, () => null)
  state.history = []
  state.step = 0
  state.evictions = 0
  state.busy = false
  rzrRenderAll(state)
}

function rzrSetPolicy(state: RzrState, policy: RzrPolicy) {
  state.policy = policy
  for (const btn of state.policyBtns) {
    btn.setAttribute('aria-checked', btn.dataset.rzrPolicy === policy ? 'true' : 'false')
  }
  for (const block of state.ruleBlocks) {
    block.hidden = block.dataset.rzrRule !== policy
  }
  state.root.setAttribute('data-policy', policy)
}

function rzrSetup(root: HTMLElement): (() => void) | null {
  if (root.dataset.rzrBound === 'true') return null
  const cap = Number.parseInt(root.getAttribute('data-capacity') ?? '0', 10)
  if (!Number.isFinite(cap) || cap <= 0) return null

  const nextBtn = root.querySelector<HTMLButtonElement>('[data-rzr-next]')
  const resetBtn = root.querySelector<HTMLButtonElement>('[data-rzr-reset]')
  const historyList = root.querySelector<HTMLOListElement>('[data-rzr-history]')
  const statResidents = root.querySelector<HTMLElement>('[data-rzr-stat="residents"]')
  const statEvictions = root.querySelector<HTMLElement>('[data-rzr-stat="evictions"]')
  const statAvg = root.querySelector<HTMLElement>('[data-rzr-stat="avg"]')
  const statMass = root.querySelector<HTMLElement>('[data-rzr-stat="mass"]')
  if (
    !nextBtn ||
    !resetBtn ||
    !historyList ||
    !statResidents ||
    !statEvictions ||
    !statAvg ||
    !statMass
  ) {
    return null
  }

  const slotGroups = Array.from(root.querySelectorAll<SVGGElement>('[data-rzr-slot]'))
  if (slotGroups.length !== cap) return null
  const slotBars = slotGroups.map(g => g.querySelector<SVGRectElement>('[data-rzr-slot-bar]')!)
  const slotLabels = slotGroups.map(g => g.querySelector<HTMLElement>('[data-rzr-slot-label]')!)
  const slotScores = slotGroups.map(g => g.querySelector<HTMLElement>('[data-rzr-slot-score]')!)

  const policyBtns = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-rzr-policy]'))
  const ruleBlocks = Array.from(root.querySelectorAll<HTMLElement>('[data-rzr-rule]'))

  const state: RzrState = {
    root,
    cap,
    policy: 'razor',
    step: 0,
    evictions: 0,
    slots: Array.from({ length: cap }, () => null),
    history: [],
    busy: false,
    slotGroups,
    slotBars,
    slotLabels,
    slotScores,
    historyList,
    statResidents,
    statEvictions,
    statAvg,
    statMass,
    policyBtns,
    ruleBlocks,
    nextBtn,
    resetBtn,
  }

  rzrSetPolicy(state, 'razor')
  rzrRenderAll(state)
  root.dataset.rzrBound = 'true'

  const handleNext = () => rzrHandleNext(state)
  const handleReset = () => rzrHandleReset(state)
  const policyHandlers = policyBtns.map(btn => {
    const handler = () => {
      const p = btn.dataset.rzrPolicy as RzrPolicy | undefined
      if (p && p !== state.policy) rzrSetPolicy(state, p)
    }
    btn.addEventListener('click', handler)
    return { btn, handler }
  })

  nextBtn.addEventListener('click', handleNext)
  resetBtn.addEventListener('click', handleReset)

  return () => {
    nextBtn.removeEventListener('click', handleNext)
    resetBtn.removeEventListener('click', handleReset)
    for (const { btn, handler } of policyHandlers) btn.removeEventListener('click', handler)
    delete root.dataset.rzrBound
  }
}

document.addEventListener('nav', () => {
  const roots = document.querySelectorAll<HTMLElement>(RZR_ROOT_SELECTOR)
  for (const root of roots) {
    const cleanup = rzrSetup(root)
    if (cleanup) window.addCleanup?.(cleanup)
  }
})
