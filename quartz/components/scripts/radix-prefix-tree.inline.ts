import katex from 'katex'

type RptBranch = 'chat_a' | 'chat_b' | 'few_shot' | 'new' | 'root'

type RptPromptSegment = { id: string; label: string; tokens: number; x: number; y: number }

type RptPrompt = {
  id: string
  label: string
  labelTex: string
  branch: RptBranch
  matchPath: string[]
  newSegments: RptPromptSegment[]
  newTokens: number
  evictTarget?: 'chat_a' | 'chat_b' | 'few_shot'
}

type RptResidentMap = Record<'chat_a' | 'chat_b' | 'few_shot', boolean>

const RPT_BRANCHES: Array<'chat_a' | 'chat_b' | 'few_shot'> = ['chat_a', 'chat_b', 'few_shot']
const RPT_BRANCH_NODES: Record<'chat_a' | 'chat_b' | 'few_shot', string[]> = {
  chat_a: ['a', 'a1', 'a2'],
  chat_b: ['b', 'b1'],
  few_shot: ['f', 'f1'],
}

const RPT_STEP_MS = 90

const rptIsBranchKey = (key: string): key is 'chat_a' | 'chat_b' | 'few_shot' =>
  key === 'chat_a' || key === 'chat_b' || key === 'few_shot'

const rptParsePrompts = (root: HTMLElement): RptPrompt[] => {
  const raw = root.getAttribute('data-rpt-prompts')
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as RptPrompt[]
  } catch {
    return []
  }
}

const rptEdgeKey = (from: string, to: string) => `${from}->${to}`

const rptPathEdges = (path: string[]): string[] => {
  const edges: string[] = []
  for (let i = 1; i < path.length; i++) edges.push(rptEdgeKey(path[i - 1], path[i]))
  return edges
}

const rptResetVisualState = (root: HTMLElement) => {
  for (const node of root.querySelectorAll<SVGGElement>('[data-rpt-node]')) {
    node.classList.remove('is-match', 'is-pinned', 'is-evicted')
    node.style.removeProperty('--rpt-delay')
  }
  for (const edge of root.querySelectorAll<SVGPathElement>('[data-rpt-edge]')) {
    edge.classList.remove('is-match', 'is-evicted')
    edge.style.removeProperty('--rpt-delay')
  }
  for (const group of root.querySelectorAll<SVGGElement>('[data-rpt-new-group]')) {
    group.classList.remove('is-visible')
    group.style.removeProperty('--rpt-delay')
    const path = group.querySelector<SVGPathElement>('.rpt-edge--new')
    const newNode = group.querySelector<SVGGElement>('.rpt-node--new')
    const label = group.querySelector<SVGForeignObjectElement>('.rpt-edge-label--new')
    path?.classList.remove('is-visible')
    newNode?.classList.remove('is-visible')
    label?.classList.remove('is-visible')
  }
  for (const btn of root.querySelectorAll<HTMLButtonElement>('[data-rpt-prompt]')) {
    btn.classList.remove('is-active')
  }
}

const rptSetBranchEvicted = (root: HTMLElement, branch: 'chat_a' | 'chat_b' | 'few_shot') => {
  const ids = RPT_BRANCH_NODES[branch]
  for (const id of ids) {
    const node = root.querySelector<SVGGElement>(`[data-rpt-node="${id}"]`)
    node?.classList.add('is-evicted')
    node?.classList.remove('is-match', 'is-pinned')
  }
  for (const edge of root.querySelectorAll<SVGPathElement>('[data-rpt-edge]')) {
    const key = edge.getAttribute('data-rpt-edge') ?? ''
    const [from, to] = key.split('->')
    if (ids.includes(to) || ids.includes(from)) edge.classList.add('is-evicted')
  }
}

const rptApplyPrompt = (
  root: HTMLElement,
  prompt: RptPrompt,
  resident: RptResidentMap,
  lru: Array<'chat_a' | 'chat_b' | 'few_shot'>,
) => {
  rptResetVisualState(root)

  for (const branch of RPT_BRANCHES) {
    if (!resident[branch]) rptSetBranchEvicted(root, branch)
  }

  prompt.matchPath.forEach((nodeId, i) => {
    const node = root.querySelector<SVGGElement>(`[data-rpt-node="${nodeId}"]`)
    if (!node) return
    node.style.setProperty('--rpt-delay', `${i * RPT_STEP_MS}ms`)
    node.classList.add('is-match')
  })
  const lastMatchedId = prompt.matchPath[prompt.matchPath.length - 1]
  const lastNode = root.querySelector<SVGGElement>(`[data-rpt-node="${lastMatchedId}"]`)
  lastNode?.classList.add('is-pinned')

  rptPathEdges(prompt.matchPath).forEach((edgeKey, i) => {
    const edge = root.querySelector<SVGPathElement>(`[data-rpt-edge="${edgeKey}"]`)
    if (!edge) return
    edge.style.setProperty('--rpt-delay', `${(i + 1) * RPT_STEP_MS}ms`)
    edge.classList.add('is-match')
  })

  const group = root.querySelector<SVGGElement>(`[data-rpt-new-group="${prompt.id}"]`)
  if (group) {
    group.style.setProperty('--rpt-delay', `${prompt.matchPath.length * RPT_STEP_MS}ms`)
    group.classList.add('is-visible')
    group.querySelector<SVGPathElement>('.rpt-edge--new')?.classList.add('is-visible')
    group.querySelector<SVGGElement>('.rpt-node--new')?.classList.add('is-visible')
    group
      .querySelector<SVGForeignObjectElement>('.rpt-edge-label--new')
      ?.classList.add('is-visible')
  }

  const btn = root.querySelector<HTMLButtonElement>(`[data-rpt-prompt="${prompt.id}"]`)
  btn?.classList.add('is-active')

  if (prompt.branch !== 'new' && rptIsBranchKey(prompt.branch)) {
    const idx = lru.indexOf(prompt.branch)
    if (idx >= 0) lru.splice(idx, 1)
    lru.push(prompt.branch)
  }
}

const rptCountResident = (resident: RptResidentMap): { resident: number; evicted: number } => {
  const counts = RPT_BRANCHES.reduce(
    (acc, b) => {
      const size = RPT_BRANCH_NODES[b].length
      if (resident[b]) acc.resident += size
      else acc.evicted += size
      return acc
    },
    { resident: 1, evicted: 0 },
  )
  return counts
}

const rptCachedTokens = (
  prompt: RptPrompt,
  nodeTokens: Record<string, number>,
  resident: RptResidentMap,
): number => {
  let total = 0
  for (const id of prompt.matchPath) {
    if (id === 'root') continue
    const branch = rptBranchOfNode(id)
    if (!branch) continue
    if (!resident[branch]) continue
    total += nodeTokens[id] ?? 0
  }
  return total
}

const rptBranchOfNode = (id: string): 'chat_a' | 'chat_b' | 'few_shot' | null => {
  if (id.startsWith('a')) return 'chat_a'
  if (id.startsWith('b')) return 'chat_b'
  if (id.startsWith('f')) return 'few_shot'
  return null
}

const rptBuildTokenMap = (root: HTMLElement): Record<string, number> => {
  const map: Record<string, number> = {}
  for (const label of root.querySelectorAll<HTMLElement>('[data-rpt-edge-label]')) {
    const key = label.getAttribute('data-rpt-edge-label') ?? ''
    const to = key.split('->')[1]
    if (!to) continue
    const txt = label.textContent ?? ''
    const num = Number.parseInt(txt.replace(/[^0-9]/g, ''), 10)
    if (Number.isFinite(num)) map[to] = num
  }
  return map
}

const rptFormatPct = (num: number, denom: number): string => {
  if (denom === 0) return '0%'
  const pct = (num / denom) * 100
  return pct >= 10 ? `${pct.toFixed(0)}%` : `${pct.toFixed(1)}%`
}

const rptRenderMath = (tex: string): string => {
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

const rptStatTex = (value: string): string => {
  if (value === '-') return '\\text{-}'
  if (value.includes('%')) return value.replace('%', '\\%')
  if (value.includes('/')) return value.replace(/\s*\/\s*/, '\\,/\\,')
  return value
}

const rptSetStat = (root: HTMLElement, key: string, value: string) => {
  const el = root.querySelector<HTMLElement>(`[data-rpt-stat="${key}"]`)
  if (!el) return
  el.innerHTML = rptRenderMath(key === 'last' ? value : rptStatTex(value))
}

const rptUpdateStats = (
  root: HTMLElement,
  prompt: RptPrompt,
  cached: number,
  resident: RptResidentMap,
  cumCached: number,
  cumTotal: number,
) => {
  const total = cached + prompt.newTokens
  rptSetStat(root, 'last', prompt.labelTex)
  rptSetStat(root, 'cached', String(cached))
  rptSetStat(root, 'new', String(prompt.newTokens))
  rptSetStat(root, 'hit', rptFormatPct(cached, total))
  rptSetStat(root, 'cum', rptFormatPct(cumCached, cumTotal))
  const { resident: r, evicted: e } = rptCountResident(resident)
  rptSetStat(root, 'resident', `${r} / ${e}`)
}

const rptResetStats = (root: HTMLElement) => {
  const reset: Record<string, string> = {
    last: '\\text{-}',
    cached: '0',
    new: '0',
    hit: '-',
    cum: '-',
    resident: '8 / 0',
  }
  for (const [key, value] of Object.entries(reset)) rptSetStat(root, key, value)
}

const rptSetup = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-radix-prefix-tree]')
  for (const root of roots) {
    if (root.dataset.rptBound === 'true') continue
    root.dataset.rptBound = 'true'

    const prompts = rptParsePrompts(root)
    if (!prompts.length) continue
    const nodeTokens = rptBuildTokenMap(root)

    rptResetStats(root)

    const resident: RptResidentMap = { chat_a: true, chat_b: true, few_shot: true }
    let lru: Array<'chat_a' | 'chat_b' | 'few_shot'> = ['few_shot', 'chat_b', 'chat_a']
    let cumCached = 0
    let cumTotal = 0

    const handlers: Array<() => void> = []

    const handleClick = (prompt: RptPrompt) => () => {
      if (prompt.evictTarget) resident[prompt.evictTarget] = false
      const cached = rptCachedTokens(prompt, nodeTokens, resident)
      cumCached += cached
      cumTotal += cached + prompt.newTokens
      rptApplyPrompt(root, prompt, resident, lru)
      rptUpdateStats(root, prompt, cached, resident, cumCached, cumTotal)
    }

    for (const prompt of prompts) {
      const btn = root.querySelector<HTMLButtonElement>(`[data-rpt-prompt="${prompt.id}"]`)
      if (!btn) continue
      const fn = handleClick(prompt)
      btn.addEventListener('click', fn)
      handlers.push(() => btn.removeEventListener('click', fn))
    }

    const resetBtn = root.querySelector<HTMLButtonElement>('[data-rpt-reset]')
    const handleReset = () => {
      resident.chat_a = true
      resident.chat_b = true
      resident.few_shot = true
      lru = ['few_shot', 'chat_b', 'chat_a']
      cumCached = 0
      cumTotal = 0
      rptResetVisualState(root)
      rptResetStats(root)
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', handleReset)
      handlers.push(() => resetBtn.removeEventListener('click', handleReset))
    }

    window.addCleanup(() => {
      for (const off of handlers) off()
      delete root.dataset.rptBound
    })
  }
}

document.addEventListener('nav', rptSetup)
