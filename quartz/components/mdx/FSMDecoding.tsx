import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/fsm-decoding.inline'
import style from '../styles/fsmDecoding.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type PresetId = 'json' | 'number'
type ModeId = 'char' | 'compressed'
type Props = { caption?: string; preset?: PresetId; mode?: ModeId; view?: 'stepper' | 'graph' }

const EOS = 'EOS'

type Preset = {
  regex: string
  // Token spellings in vocabulary order; EOS is the last entry.
  vocab: readonly string[]
  // The tokens the model samples in ordinary FSM decoding, ending with EOS.
  path: readonly string[]
  // Row breaks for the character graph: a new row starts after each listed depth.
  wide: readonly number[]
  narrow: readonly number[] | null
  compressedNarrow: readonly number[] | null
  // Largest drawn scale; a two-state graph would otherwise render as a thumbnail.
  scale: number
}

const PRESETS: Record<PresetId, Preset> = {
  // The Paul/John language of the note's string-labelled index.
  json: {
    regex: '\\{"name":"(Paul|John)","age":(20|30)\\}',
    vocab: [
      '{',
      '{"',
      '"',
      'name',
      'nam',
      'e',
      '":"',
      '":',
      'P',
      'Paul',
      'aul',
      'John',
      '","',
      'age',
      'Age',
      '2',
      '3',
      '30',
      '0}',
      '}',
      EOS,
    ],
    path: ['{"', 'name', '":"', 'Paul', '","', 'age', '":', '3', '0}', EOS],
    wide: [13],
    narrow: [6, 13, 20],
    compressedNarrow: [3],
    scale: 1.3,
  },
  // The regex and five token spellings of the Willard and Louf example.
  number: {
    regex: '([0-9]*)?\\.?[0-9]*',
    vocab: ['A', '.', '42', '.2', '1', EOS],
    path: ['1', '.2', '42', EOS],
    wide: [],
    narrow: null,
    compressedNarrow: null,
    scale: 2,
  },
}

// ---------- regex to minimal DFA ----------

type Ast =
  | { kind: 'chars'; set: string[] }
  | { kind: 'cat'; parts: Ast[] }
  | { kind: 'alt'; opts: Ast[] }
  | { kind: 'star' | 'plus' | 'opt'; body: Ast }

const parseRegex = (src: string): Ast => {
  let i = 0
  const atom = (): Ast => {
    const c = src[i++]
    if (c === '(') {
      const inner = alt()
      if (src[i++] !== ')') throw new Error(`FSMDecoding: unclosed group in ${src}`)
      return inner
    }
    if (c === '[') {
      const set: string[] = []
      while (src[i] !== ']') {
        const a = src[i++]
        if (src[i] === '-' && src[i + 1] !== ']') {
          const b = src[i + 1]
          i += 2
          for (let k = a.charCodeAt(0); k <= b.charCodeAt(0); k++) set.push(String.fromCharCode(k))
        } else set.push(a)
      }
      i++
      return { kind: 'chars', set }
    }
    if (c === '\\') return { kind: 'chars', set: [src[i++]] }
    return { kind: 'chars', set: [c] }
  }
  const seq = (): Ast => {
    const parts: Ast[] = []
    while (i < src.length && src[i] !== '|' && src[i] !== ')') {
      let a = atom()
      const q = src[i]
      if (q === '*' || q === '+' || q === '?') {
        i++
        a = { kind: q === '*' ? 'star' : q === '+' ? 'plus' : 'opt', body: a }
      }
      parts.push(a)
    }
    return { kind: 'cat', parts }
  }
  const alt = (): Ast => {
    const opts = [seq()]
    while (src[i] === '|') {
      i++
      opts.push(seq())
    }
    return opts.length === 1 ? opts[0] : { kind: 'alt', opts }
  }
  const root = alt()
  if (i !== src.length) throw new Error(`FSMDecoding: trailing input in ${src}`)
  return root
}

type Dfa = { n: number; accept: boolean[]; next: Map<string, number>[]; alphabet: string[] }

// Thompson NFA, subset construction, Moore minimisation, then BFS numbering from the start.
const compile = (src: string): Dfa => {
  const eps: number[][] = []
  const edges: [string[], number][][] = []
  const alphabet: string[] = []
  const add = () => {
    edges.push([])
    return eps.push([]) - 1
  }
  const frag = (n: Ast): [number, number] => {
    if (n.kind === 'chars') {
      const s = add()
      const e = add()
      edges[s].push([n.set, e])
      for (const c of n.set) if (!alphabet.includes(c)) alphabet.push(c)
      return [s, e]
    }
    if (n.kind === 'cat') {
      const s = add()
      let cur = s
      for (const p of n.parts) {
        const [a, b] = frag(p)
        eps[cur].push(a)
        cur = b
      }
      return [s, cur]
    }
    const s = add()
    const e = add()
    if (n.kind === 'alt') {
      for (const o of n.opts) {
        const [a, b] = frag(o)
        eps[s].push(a)
        eps[b].push(e)
      }
      return [s, e]
    }
    const [a, b] = frag(n.body)
    eps[s].push(a)
    eps[b].push(e)
    if (n.kind !== 'plus') eps[s].push(e)
    if (n.kind !== 'opt') eps[b].push(a)
    return [s, e]
  }
  const [nStart, nEnd] = frag(parseRegex(src))

  const closure = (seed: number[]) => {
    const seen = new Set(seed)
    const stack = [...seed]
    while (stack.length)
      for (const t of eps[stack.pop()!])
        if (!seen.has(t)) {
          seen.add(t)
          stack.push(t)
        }
    return [...seen].sort((a, b) => a - b)
  }
  const subsets: number[][] = [closure([nStart])]
  const keys = new Map([[subsets[0].join(','), 0]])
  const raw: Map<string, number>[] = []
  for (let d = 0; d < subsets.length; d++) {
    raw.push(new Map())
    for (const c of alphabet) {
      const moved = subsets[d].flatMap(s =>
        edges[s].filter(([set]) => set.includes(c)).map(e => e[1]),
      )
      if (!moved.length) continue
      const target = closure(moved)
      const key = target.join(',')
      if (!keys.has(key)) keys.set(key, subsets.push(target) - 1)
      raw[d].set(c, keys.get(key)!)
    }
  }
  const rawAccept = subsets.map(s => s.includes(nEnd))

  let cls: number[] = rawAccept.map(a => (a ? 1 : 0))
  for (;;) {
    const sigs = new Map<string, number>()
    const nextCls = cls.map((c, d) => {
      const sig = [c, ...alphabet.map(a => (raw[d].has(a) ? cls[raw[d].get(a)!] : -1))].join(',')
      if (!sigs.has(sig)) sigs.set(sig, sigs.size)
      return sigs.get(sig)!
    })
    const stable = new Set(nextCls).size === new Set(cls).size
    cls = nextCls
    if (stable) break
  }

  const order = [cls[0]]
  const id = new Map([[cls[0], 0]])
  const next: Map<string, number>[] = []
  const accept: boolean[] = []
  for (let k = 0; k < order.length; k++) {
    const rep = cls.indexOf(order[k])
    accept.push(rawAccept[rep])
    const row = new Map<string, number>()
    for (const [c, t] of raw[rep]) {
      if (!id.has(cls[t])) id.set(cls[t], order.push(cls[t]) - 1)
      row.set(c, id.get(cls[t])!)
    }
    next.push(row)
  }
  return { n: order.length, accept, next, alphabet }
}

const deltaStar = (dfa: Dfa, q: number, s: string): number[] | null => {
  const trace = [q]
  for (const c of s) {
    const t = dfa.next[trace[trace.length - 1]].get(c)
    if (t === undefined) return null
    trace.push(t)
  }
  return trace
}

// ---------- graphs ----------

type GEdge = { from: number; to: number; label: string; path: number[] }
type Graph = { nodes: number[]; edges: GEdge[] }

const classLabel = (chars: string[]) => {
  if (chars.length === 1) return chars[0]
  const codes = chars.map(c => c.charCodeAt(0)).sort((a, b) => a - b)
  const run = codes.every((c, i) => i === 0 || c === codes[i - 1] + 1)
  return run && codes.length > 2
    ? `[${String.fromCharCode(codes[0])}-${String.fromCharCode(codes[codes.length - 1])}]`
    : `[${codes.map(c => String.fromCharCode(c)).join('')}]`
}

const charGraph = (dfa: Dfa): Graph => {
  const edges: GEdge[] = []
  for (let q = 0; q < dfa.n; q++) {
    const byTarget = new Map<number, string[]>()
    for (const [c, t] of dfa.next[q]) byTarget.set(t, [...(byTarget.get(t) ?? []), c])
    for (const [t, chars] of byTarget)
      edges.push({ from: q, to: t, label: classLabel(chars), path: [q, t] })
  }
  return { nodes: Array.from({ length: dfa.n }, (_, q) => q), edges }
}

const outChars = (dfa: Dfa, q: number) => dfa.next[q].size

// A compressed chain stops at the start, an accepting state, a choice, a merge, or the head of a branch.
const compressedGraph = (dfa: Dfa, g: Graph): Graph => {
  const preds = (q: number) => g.edges.filter(e => e.to === q && e.from !== q).map(e => e.from)
  const kept = (q: number) =>
    q === 0 ||
    dfa.accept[q] ||
    outChars(dfa, q) !== 1 ||
    preds(q).length !== 1 ||
    preds(q).some(p => outChars(dfa, p) > 1)
  const nodes = g.nodes.filter(kept)
  const edges: GEdge[] = []
  for (const e of g.edges) {
    if (!kept(e.from)) continue
    let label = e.label
    const path = [e.from, e.to]
    let cur = e.to
    while (!kept(cur) && cur !== e.from) {
      const [[c, t]] = [...dfa.next[cur]]
      label += c
      path.push(t)
      cur = t
    }
    edges.push({ from: e.from, to: cur, label, path })
  }
  return { nodes, edges }
}

// ---------- layout ----------

const NODE = 16
const H = NODE / 2
const LANE = 24
const ROW_H = 104
const TOP = 42
const MARGIN = 14
const CH = 6.1
const labelW = (s: string) => s.length * CH

type Pos = { x: number; y: number; row: number; lane: number; depth: number }
type EdgeGeo = {
  kind: 'line' | 'elbow' | 'loop'
  points: [number, number][]
  lx: number
  ly: number
  label: string
  anchor?: 'start'
}
type Layout = { w: number; h: number; pos: Map<number, Pos>; geo: EdgeGeo[] }

const layout = (g: Graph, breaks: readonly number[]): Layout => {
  const depth = new Map([[g.nodes[0], 0]])
  const queue = [g.nodes[0]]
  while (queue.length) {
    const q = queue.shift()!
    for (const e of g.edges)
      if (e.from === q && !depth.has(e.to)) {
        depth.set(e.to, depth.get(q)! + 1)
        queue.push(e.to)
      }
  }
  const maxDepth = Math.max(...depth.values())
  const rowOf = (d: number) => breaks.filter(b => b < d).length
  const lanes = new Map<number, number>()
  for (let d = 0; d <= maxDepth; d++) {
    const at = g.nodes.filter(q => depth.get(q) === d)
    if (at.length > 2) throw new Error('FSMDecoding: layout holds at most two lanes')
    at.forEach((q, i) => lanes.set(q, at.length === 1 ? 0 : i === 0 ? -1 : 1))
  }

  // The gap before depth d fits the widest label entering it, and loop labels on either side.
  const loopW = (d: number) =>
    Math.max(
      0,
      ...g.edges.filter(e => e.from === e.to && depth.get(e.from) === d).map(e => labelW(e.label)),
    )
  const gap = (d: number) =>
    Math.max(
      18,
      ...g.edges
        .filter(e => e.from !== e.to && depth.get(e.to) === d && rowOf(d) === rowOf(d - 1))
        .map(e => labelW(e.label) + 12),
      (loopW(d) + loopW(d - 1)) / 2 + 10,
    )
  const xAt = new Map<number, number>()
  for (let d = 0; d <= maxDepth; d++)
    xAt.set(
      d,
      d === 0 || rowOf(d) !== rowOf(d - 1)
        ? MARGIN + H + Math.max(0, loopW(d) / 2 - H)
        : xAt.get(d - 1)! + NODE + gap(d),
    )
  const pos = new Map<number, Pos>()
  for (const q of g.nodes) {
    const d = depth.get(q)!
    const row = rowOf(d)
    pos.set(q, {
      x: xAt.get(d)!,
      y: TOP + row * ROW_H + lanes.get(q)! * LANE,
      row,
      lane: lanes.get(q)!,
      depth: d,
    })
  }

  const geo = g.edges.map((e): EdgeGeo => {
    const a = pos.get(e.from)!
    const b = pos.get(e.to)!
    if (e.from === e.to) {
      const top = a.y - H - 14
      return {
        kind: 'loop',
        points: [
          [a.x - 7, a.y - H],
          [a.x - 7, top],
          [a.x + 7, top],
          [a.x + 7, a.y - H],
        ],
        lx: a.x,
        ly: top - 4,
        label: e.label,
      }
    }
    if (b.depth !== a.depth + 1) throw new Error('FSMDecoding: edges must advance one depth')
    if (b.row !== a.row) {
      const mid = TOP + a.row * ROW_H + ROW_H / 2
      return {
        kind: 'elbow',
        points: [
          [a.x, a.y + H],
          [a.x, mid],
          [b.x, mid],
          [b.x, b.y - H],
        ],
        // The label sits on the arriving leg, next to the state it enters.
        lx: b.x + 5,
        ly: (mid + b.y - H) / 2 + 3,
        anchor: 'start',
        label: e.label,
      }
    }
    const below = a.lane > 0 || b.lane > 0
    return {
      kind: 'line',
      points: [
        [a.x + H, a.y],
        [b.x - H, b.y],
      ],
      lx: (a.x + b.x) / 2,
      ly: (a.y + b.y) / 2 + (below ? 13 : -5),
      label: e.label,
    }
  })

  const right = Math.max(...[...pos.values()].map(p => p.x)) + H + MARGIN
  const loopRight = Math.max(
    0,
    ...geo.filter(e => e.kind === 'loop').map(e => e.lx + labelW(e.label) / 2 + MARGIN),
  )
  const rows = rowOf(maxDepth) + 1
  const hasLower = [...pos.values()].some(p => p.lane > 0 && p.row === rows - 1)
  return {
    w: Math.ceil(Math.max(right, loopRight)),
    h: TOP + (rows - 1) * ROW_H + (hasLower ? LANE + 22 : 16),
    pos,
    geo,
  }
}

// ---------- decoding frames ----------

type Part = string | { code: string }
type Frame = {
  state: number
  kind: 'sample' | 'forced' | 'done'
  tokens: number[]
  litPath: number[]
  seenPath: number[]
  passes: number
  forced: number
  text: Part[]
}

type Model = {
  preset: Preset
  dfa: Dfa
  graphs: Record<ModeId, Graph>
  frames: Record<ModeId, Frame[]>
  hasForced: boolean
  mask: (q: number) => { allowed: boolean; target: number | null }[]
}

const buildModel = (preset: Preset): Model => {
  const dfa = compile(preset.regex)
  const live = dfa.accept.slice()
  for (let changed = true; changed; ) {
    changed = false
    for (let q = 0; q < dfa.n; q++)
      if (!live[q] && [...dfa.next[q].values()].some(t => live[t])) live[q] = changed = true
  }
  const mask = (q: number) =>
    preset.vocab.map(tok => {
      if (tok === EOS) return { allowed: dfa.accept[q], target: null }
      const trace = deltaStar(dfa, q, tok)
      const end = trace ? trace[trace.length - 1] : -1
      return end >= 0 && live[end]
        ? { allowed: true, target: end }
        : { allowed: false, target: null }
    })
  const tokenId = (t: string) => {
    const v = preset.vocab.indexOf(t)
    if (v < 0) throw new Error(`FSMDecoding: ${t} is not in the vocabulary`)
    return v
  }
  const V = preset.vocab.length

  // Ordinary FSM decoding: one forward pass samples each token, forced or not.
  const charFrames: Frame[] = []
  let q = 0
  let seen: number[] = [0]
  preset.path.forEach((tok, i) => {
    const v = tokenId(tok)
    const m = mask(q)
    if (!m[v].allowed) throw new Error(`FSMDecoding: ${tok} is masked at state ${q}`)
    const allowed = m.filter(x => x.allowed).length
    const trace = tok === EOS ? [q] : deltaStar(dfa, q, tok)!
    const to = trace[trace.length - 1]
    charFrames.push({
      state: q,
      kind: 'sample',
      tokens: [v],
      litPath: trace,
      seenPath: seen,
      passes: i + 1,
      forced: 0,
      text:
        tok === EOS
          ? [
              `Pass ${i + 1} at state ${q}: only EOS is allowed, so the pass samples `,
              { code: EOS },
              ' and the output ends.',
            ]
          : [
              `Pass ${i + 1} at state ${q}: the mask allows ${allowed} of ${V} tokens. The model samples `,
              { code: tok },
              `, which moves the FSM to state ${to}.`,
            ],
    })
    seen = [...seen, ...trace.slice(1)]
    q = to
  })
  if (!dfa.accept[q])
    throw new Error('FSMDecoding: the sampled path does not end in an accepting state')
  const charPasses = preset.path.length
  charFrames.push({
    state: q,
    kind: 'done',
    tokens: [],
    litPath: [],
    seenPath: seen,
    passes: charPasses,
    forced: 0,
    text: [`Done. ${charPasses} forward passes produced ${charPasses} tokens; state ${q} accepts.`],
  })

  // Jump-forward: follow a single-character path to the next choice and append it unsampled.
  const forcedFrom = (from: number) => {
    let s = ''
    let cur = from
    while (!dfa.accept[cur] && outChars(dfa, cur) === 1) {
      const [[c, t]] = [...dfa.next[cur]]
      s += c
      cur = t
    }
    return { s, to: cur }
  }
  const ordinary = preset.vocab.filter(t => t !== EOS)
  const greedy = (s: string) => {
    const out: string[] = []
    for (let i = 0; i < s.length; ) {
      const t = ordinary.filter(t => s.startsWith(t, i)).sort((a, b) => b.length - a.length)[0]
      if (!t) throw new Error(`FSMDecoding: cannot tokenize forced text ${s}`)
      out.push(t)
      i += t.length
    }
    return out
  }
  const offsets: number[] = []
  preset.path.reduce((at, t) => (offsets.push(at), at + (t === EOS ? 0 : t.length)), 0)
  const jfFrames: Frame[] = []
  q = 0
  seen = [0]
  let text = ''
  let passes = 0
  let forcedTotal = 0
  for (let guard = 0; guard < 64; guard++) {
    const { s, to } = forcedFrom(q)
    const end = dfa.accept[to] && outChars(dfa, to) === 0
    if (s || (end && dfa.accept[q] && outChars(dfa, q) === 0)) {
      const toks = greedy(s).map(tokenId)
      if (end) toks.push(tokenId(EOS))
      forcedTotal += toks.length
      const trace = deltaStar(dfa, q, s)!
      const shown = toks.map(v => ({ code: preset.vocab[v] }))
      const list: Part[] = shown.flatMap((p, i) => (i ? [', ', p] : [p]))
      jfFrames.push({
        state: q,
        kind: 'forced',
        tokens: toks,
        litPath: trace,
        seenPath: seen,
        passes,
        forced: forcedTotal,
        text: [
          `State ${q} has one way forward. Jump-forward appends `,
          ...(s ? [{ code: s }] : []),
          end
            ? `${s ? ' and ' : ''}EOS without sampling (${toks.length} tokens: `
            : ` without sampling (${toks.length} tokens: `,
          ...list,
          end
            ? '). The output is complete, so no further pass runs.'
            : `). Pass ${passes + 1} processes them as one chunk.`,
        ],
      })
      seen = [...seen, ...trace.slice(1)]
      text += s
      q = to
      if (end) break
      continue
    }
    const k = offsets.findIndex((o, i) => o === text.length && preset.path[i] !== EOS)
    const tok = k >= 0 ? preset.path[k] : EOS
    const v = tokenId(tok)
    const m = mask(q)
    if (!m[v].allowed) throw new Error(`FSMDecoding: ${tok} is masked at state ${q}`)
    passes += 1
    const trace = tok === EOS ? [q] : deltaStar(dfa, q, tok)!
    const to2 = trace[trace.length - 1]
    jfFrames.push({
      state: q,
      kind: 'sample',
      tokens: [v],
      litPath: trace,
      seenPath: seen,
      passes,
      forced: forcedTotal,
      text: [
        `Pass ${passes} at state ${q}: the mask allows ${m.filter(x => x.allowed).length} of ${V} tokens. The model samples `,
        { code: tok },
        tok === EOS ? '.' : `, which moves the FSM to state ${to2}.`,
      ],
    })
    seen = [...seen, ...trace.slice(1)]
    if (tok === EOS) break
    text += tok
    q = to2
  }
  jfFrames.push({
    state: q,
    kind: 'done',
    tokens: [],
    litPath: [],
    seenPath: seen,
    passes,
    forced: forcedTotal,
    text: [
      `Done. ${passes} forward passes sampled ${passes} tokens; jump-forward appended ${forcedTotal} more. State ${q} accepts.`,
    ],
  })

  const char = charGraph(dfa)
  const compressed = compressedGraph(dfa, char)
  for (const f of jfFrames)
    if (!compressed.nodes.includes(f.state))
      throw new Error(`FSMDecoding: state ${f.state} is not a node of the compressed graph`)
  return {
    preset,
    dfa,
    graphs: { char, compressed },
    frames: { char: charFrames, compressed: jfFrames },
    hasForced:
      compressed.edges.some(e => e.path.length > 2) || compressed.nodes.length < char.nodes.length,
    mask,
  }
}

const MODELS: Record<PresetId, Model> = {
  json: buildModel(PRESETS.json),
  number: buildModel(PRESETS.number),
}

// An edge is lit or seen when its whole state path lies on the traversed path, in order.
const onPath = (edge: GEdge, path: number[]) => {
  for (let i = 0; i + edge.path.length <= path.length; i++)
    if (edge.path.every((s, k) => path[i + k] === s)) return true
  return false
}

const graphState = (g: Graph, f: Frame) => ({
  n: g.nodes.map(q => (q === f.state ? 'c' : f.seenPath.includes(q) ? 's' : 'n')).join(''),
  e: g.edges.map(e => (onPath(e, f.litPath) ? 'l' : onPath(e, f.seenPath) ? 's' : 'n')).join(''),
})

type ClientFrame = {
  n: string
  e: string
  m: string
  t: string[]
  shown: number
  lit: number
  out: Record<'step' | 'state' | 'allowed' | 'passes' | 'forced', string>
}

const clientFrames = (model: Model, mode: ModeId): ClientFrame[] => {
  const frames = model.frames[mode]
  const V = model.preset.vocab.length
  let shown = 0
  return frames.map((f, k) => {
    const m = model.mask(f.state)
    const pick = f.kind === 'sample' ? f.tokens[0] : -1
    const frame: ClientFrame = {
      ...graphState(model.graphs[mode], f),
      m: m.map((x, v) => (v === pick ? 'p' : x.allowed ? 'a' : 'x')).join(''),
      t: m.map((x, v) =>
        x.allowed ? (model.preset.vocab[v] === EOS ? 'end' : `→ ${x.target}`) : '',
      ),
      shown: shown + f.tokens.length,
      lit: f.tokens.length,
      out: {
        step: f.kind === 'done' ? 'done' : `${k + 1} / ${frames.length - 1}`,
        state: String(f.state),
        allowed: `${m.filter(x => x.allowed).length} / ${V}`,
        passes: String(f.passes),
        forced: String(f.forced),
      },
    }
    shown += f.tokens.length
    return frame
  })
}

const tape = (model: Model, mode: ModeId) => {
  const cells: { tok: string; tag: string }[] = []
  for (const f of model.frames[mode])
    for (const v of f.tokens)
      cells.push({ tok: model.preset.vocab[v], tag: f.kind === 'forced' ? 'ff' : `p${f.passes}` })
  return cells
}

const CLIENT: Record<PresetId, Record<ModeId, ClientFrame[]>> = {
  json: {
    char: clientFrames(MODELS.json, 'char'),
    compressed: clientFrames(MODELS.json, 'compressed'),
  },
  number: {
    char: clientFrames(MODELS.number, 'char'),
    compressed: clientFrames(MODELS.number, 'compressed'),
  },
}

const MODES: readonly { id: ModeId; label: string }[] = [
  { id: 'char', label: 'Character FSM' },
  { id: 'compressed', label: 'Compressed, jump-forward' },
]

let instance = 0

const GraphSvg = ({
  id,
  g,
  lay,
  frame,
  dfa,
  variant,
  label,
  scale,
}: {
  id: string
  g: Graph
  lay: Layout
  frame: ClientFrame | null
  dfa: Dfa
  variant: string
  label: string
  scale: number
}) => {
  const marker = `${id}-${variant}-arrow`
  return (
    <svg
      class={`fsm-svg fsm-svg--${variant}`}
      viewBox={`0 0 ${lay.w} ${lay.h}`}
      style={{ maxWidth: `${Math.round(lay.w * scale)}px` }}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={label}
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 6 6"
          markerWidth="6"
          markerHeight="6"
          refX="6"
          refY="3"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path class="fsm-arrow" d="M0,0 L6,3 L0,6 z" />
        </marker>
      </defs>
      {lay.geo.map((e, j) => {
        const pts = e.points.map(p => p.join(',')).join(' ')
        return (
          <g class="fsm-edge" data-fsm-edge={String(j)} data-st={frame ? frame.e[j] : 'n'}>
            <polyline class="fsm-edge-band" points={pts} />
            <polyline class="fsm-edge-line" points={pts} marker-end={`url(#${marker})`} />
            <text class="fsm-edge-label" x={e.lx} y={e.ly} text-anchor={e.anchor ?? 'middle'}>
              {e.label}
            </text>
          </g>
        )
      })}
      {g.nodes.map((q, i) => {
        const p = lay.pos.get(q)!
        return (
          <g class="fsm-node" data-fsm-node={String(i)} data-st={frame ? frame.n[i] : 'n'}>
            <rect class="fsm-node-box" x={p.x - H} y={p.y - H} width={NODE} height={NODE} />
            {dfa.accept[q] ? (
              <rect
                class="fsm-node-accept"
                x={p.x - H + 2.5}
                y={p.y - H + 2.5}
                width={NODE - 5}
                height={NODE - 5}
              />
            ) : null}
            <text class="fsm-node-label" x={p.x} y={p.y + 2.7} text-anchor="middle">
              {String(q)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

const PartsText = ({ parts }: { parts: Part[] }) => (
  <>{parts.map(p => (typeof p === 'string' ? p : <code>{p.code}</code>))}</>
)

const FSMDecodingImpl: QuartzMdxComponent<Props> = ({
  caption,
  preset = 'json',
  mode = 'char',
  view = 'stepper',
}) => {
  const stepper = view !== 'graph'
  const id = `fsm-${++instance}`
  const p0: PresetId = preset in MODELS ? preset : 'json'
  const model = MODELS[p0]
  const toggle = model.hasForced
  const m0: ModeId = toggle && mode === 'compressed' ? 'compressed' : 'char'
  const frames = CLIENT[p0]
  const V = model.preset.vocab.length
  const g = model.graphs
  const lays = {
    char: layout(g.char, model.preset.wide),
    charNarrow: model.preset.narrow ? layout(g.char, model.preset.narrow) : null,
    compressed: layout(g.compressed, []),
    compressedNarrow: model.preset.compressedNarrow
      ? layout(g.compressed, model.preset.compressedNarrow)
      : null,
  }
  const passes = (m: ModeId) => model.frames[m][model.frames[m].length - 1].passes
  const charLabel = `Character FSM of ${model.preset.regex}: ${g.char.nodes.length} states, ${g.char.edges.length} edges.`
  const compLabel = `Compressed FSM: ${g.compressed.nodes.length} states, ${g.compressed.edges.length} string-labelled edges.`
  const f0 = (m: ModeId) => (stepper ? frames[m][0] : null)

  return (
    <figure
      class="fsm-decoding"
      data-fsm-decoding
      data-fsm-mode={m0}
      data-fsm-stepper={stepper ? 'true' : 'false'}
      data-fsm-frames={stepper ? JSON.stringify(frames) : undefined}
    >
      <div class="fsm-controls">
        {toggle ? (
          <fieldset class="fsm-segments">
            <legend class="fsm-legend">Automaton</legend>
            {MODES.map(m => (
              <label class="fsm-segment">
                <input
                  type="radio"
                  name={`${id}-mode`}
                  value={m.id}
                  checked={m.id === m0}
                  data-fsm-mode-input
                />
                <span>{m.label}</span>
              </label>
            ))}
          </fieldset>
        ) : null}
        {stepper ? (
          <div class="fsm-stepper">
            <button type="button" class="fsm-btn" data-fsm-back disabled>
              back
            </button>
            <button type="button" class="fsm-btn fsm-btn--primary" data-fsm-step>
              step
            </button>
            <button type="button" class="fsm-btn" data-fsm-reset disabled>
              reset
            </button>
          </div>
        ) : null}
      </div>

      <p class="fsm-regex">
        <span class="fsm-regex-label">Regex</span> <code>{model.preset.regex}</code>
      </p>

      <div class="fsm-graph" data-fsm-graph="char">
        <GraphSvg
          scale={model.preset.scale}
          id={id}
          g={g.char}
          lay={lays.char}
          frame={f0('char')}
          dfa={model.dfa}
          variant="wide"
          label={charLabel}
        />
        {lays.charNarrow ? (
          <GraphSvg
            scale={model.preset.scale}
            id={id}
            g={g.char}
            lay={lays.charNarrow}
            frame={f0('char')}
            dfa={model.dfa}
            variant="narrow"
            label={charLabel}
          />
        ) : null}
      </div>
      {toggle ? (
        <div class="fsm-graph" data-fsm-graph="compressed">
          <GraphSvg
            scale={model.preset.scale}
            id={id}
            g={g.compressed}
            lay={lays.compressed}
            frame={f0('compressed')}
            dfa={model.dfa}
            variant="wide"
            label={compLabel}
          />
          {lays.compressedNarrow ? (
            <GraphSvg
              scale={model.preset.scale}
              id={id}
              g={g.compressed}
              lay={lays.compressedNarrow}
              frame={f0('compressed')}
              dfa={model.dfa}
              variant="narrow"
              label={compLabel}
            />
          ) : null}
        </div>
      ) : null}

      {stepper ? (
        <>
          <div class="fsm-actions" aria-live="polite">
            {(toggle ? MODES : MODES.slice(0, 1)).map(m =>
              model.frames[m.id].map((f, k) => (
                <p
                  class="fsm-action"
                  data-fsm-act={`${m.id}:${k}`}
                  hidden={!(m.id === m0 && k === 0)}
                >
                  <PartsText parts={f.text} />
                </p>
              )),
            )}
          </div>

          <div class="fsm-panel">
            <div class="fsm-mask-wrap">
              <h4 class="fsm-heading">{`Next-token mask over the ${V}-token vocabulary`}</h4>
              <div class="fsm-mask" style={{ '--fsm-cols': String(V % 7 === 0 ? 7 : V) }}>
                {model.preset.vocab.map((tok, v) => {
                  const f = frames[m0][0]
                  return (
                    <div class="fsm-tok" data-fsm-tok={String(v)} data-st={f.m[v]}>
                      <code class="fsm-tok-text">{tok}</code>
                      <span class="fsm-tok-to" data-fsm-tok-to={String(v)}>
                        {f.t[v]}
                      </span>
                    </div>
                  )
                })}
              </div>
              <h4 class="fsm-heading fsm-heading--tape">
                Output tokens, tagged by the pass that sampled them
              </h4>
              {(toggle ? MODES : MODES.slice(0, 1)).map(m => (
                <ol
                  class="fsm-tape"
                  data-fsm-tape-mode={m.id}
                  style={{
                    '--fsm-tape-cols': String(tape(model, m.id).length),
                    '--fsm-tape-narrow': String(
                      tape(model, m.id).length % 5 === 0 ? 5 : tape(model, m.id).length,
                    ),
                  }}
                >
                  {tape(model, m.id).map((c, k) => {
                    const f = frames[m.id][0]
                    const st = k < f.shown - f.lit ? 's' : k < f.shown ? 'l' : 'h'
                    return (
                      <li class="fsm-cell" data-fsm-tape={String(k)} data-st={st}>
                        <code>{c.tok}</code>
                        <span>{c.tag}</span>
                      </li>
                    )
                  })}
                </ol>
              ))}
            </div>

            <dl class="fsm-readout fsm-readout--pairs">
              <div>
                <dt>Step</dt>
                <dd data-fsm-out="step">{frames[m0][0].out.step}</dd>
              </div>
              <div>
                <dt>Tokens the mask allows</dt>
                <dd data-fsm-out="allowed">{frames[m0][0].out.allowed}</dd>
              </div>
              <div>
                <dt>Forward passes so far</dt>
                <dd data-fsm-out="passes">{frames[m0][0].out.passes}</dd>
              </div>
              {toggle ? (
                <>
                  <div>
                    <dt>Tokens appended without sampling</dt>
                    <dd data-fsm-out="forced">{frames[m0][0].out.forced}</dd>
                  </div>
                  <div>
                    <dt>Passes for the output, character FSM</dt>
                    <dd>{String(passes('char'))}</dd>
                  </div>
                  <div>
                    <dt>Passes for the output, jump-forward</dt>
                    <dd>{String(passes('compressed'))}</dd>
                  </div>
                </>
              ) : (
                <div>
                  <dt>Current state</dt>
                  <dd data-fsm-out="state">{frames[m0][0].out.state}</dd>
                </div>
              )}
            </dl>
          </div>
        </>
      ) : (
        <div class="fsm-panel fsm-panel--static">
          <div class="fsm-edges">
            <h4 class="fsm-heading">String-labelled edges of the compressed FSM</h4>
            <div class="fsm-edge-table" role="table">
              <div role="row" class="fsm-edge-row fsm-edge-row--head">
                <span role="columnheader">from</span>
                <span role="columnheader">edge label</span>
                <span role="columnheader">to</span>
              </div>
              {g.compressed.edges.map(e => (
                <div role="row" class="fsm-edge-row">
                  <span role="cell">{String(e.from)}</span>
                  <code role="cell">{e.label}</code>
                  <span role="cell">{String(e.to)}</span>
                </div>
              ))}
            </div>
          </div>
          <dl class="fsm-readout">
            <div>
              <dt>Character FSM states</dt>
              <dd>{String(g.char.nodes.length)}</dd>
            </div>
            <div>
              <dt>Character FSM edges</dt>
              <dd>{String(g.char.edges.length)}</dd>
            </div>
            <div>
              <dt>Compressed FSM states</dt>
              <dd>{String(g.compressed.nodes.length)}</dd>
            </div>
            <div>
              <dt>Compressed FSM edges</dt>
              <dd>{String(g.compressed.edges.length)}</dd>
            </div>
            <div>
              <dt>Choice states</dt>
              <dd>
                {g.char.nodes
                  .filter(q => outChars(model.dfa, q) > 1)
                  .map(String)
                  .join(' · ')}
              </dd>
            </div>
          </dl>
        </div>
      )}

      {caption ? (
        <figcaption class="fsm-caption">
          <MathText text={caption} mathClass="fsm-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const FSMDecodingComponent = FSMDecodingImpl as QuartzMdxComponent<Props>
FSMDecodingComponent.css = style
FSMDecodingComponent.afterDOMLoaded = script

export const FSMDecoding = registerMdxComponent('FSMDecoding', FSMDecodingComponent)

export default (() => FSMDecoding) satisfies (opts: undefined) => QuartzMdxComponent<Props>
