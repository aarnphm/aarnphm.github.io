import {
  bakeOf,
  clearHighlights,
  esc,
  labelOf,
  ll,
  markPassage,
  probeTab,
  type Label,
} from './gemma'
import { fetchJson, type Runtime } from './runtime'

// The probe's circuit tab: the attribution graph (quartz/scripts/latent/attribute.py) of the
// focused passage's baked target token, laid out as circuit-tracer does. x is the token, y the
// layer (embeddings at the bottom, the logit on top); features are dots sized by influence, error
// nodes gray squares, so the share the transcoders leave unexplained stays visible.

interface GraphNode {
  id: string
  kind: 'feature' | 'error' | 'embedding' | 'logit' | 'bias'
  layer: number
  pos: number
  feature?: number
  act?: number
  influence?: number
  token?: string
  prob?: number
  label?: string
}

interface Graph {
  prompt: string
  tokens: string[]
  logits: { token: string; prob: number }[]
  nodes: GraphNode[]
  links: { source: string; target: string; weight: number }[]
  /** transcoder fraction of variance unexplained per layer, from training */
  fvu?: Record<string, number | null>
  /** plt_graphs.py: where the graph sits in its passage, UTF-16 offsets into the chunk text */
  passage?: {
    row: number
    start: number
    end: number
    target: [number, number]
    spans: [number, number][]
    token: string
    p: number
  }
}

interface CircuitState {
  n: number
  graph: Graph | null
}

const W = 340
const H = 470
const PAD = { l: 26, r: 8, t: 14, b: 58 }
const EDGES = 160
const SHOWN_TOKENS = 18

// Gemma 4 E4B's text decoder as the graph uses it (mlx-community/gemma-4-e4b-it-4bit; acts.py)
const GLOBAL = new Set([5, 11, 17, 23, 29, 35, 41])
const KV_SHARED_FROM = 24 // layers 24..41 reuse the keys and values of L22 (sliding) or L23 (global)
const MODEL_TIP = [
  'Gemma 4 E4B · 4-bit',
  '42 layers · d 2560 · MLP 10240 · PLE 256',
  'global attn: L05 L11 L17 L23 L29 L35 L41',
  'sliding attn: the rest',
  'KV shared: L24–L41 ← L22 / L23',
  'transcoder per MLP: 8192 × top 64',
  '● feature  ■ error  ○ embedding',
].join('\n')

function layerTip(g: Graph, l: number): string {
  const here = g.nodes.filter(n => n.layer === l && (n.kind === 'feature' || n.kind === 'error'))
  const sum = (k: string) =>
    here.filter(n => n.kind === k).reduce((a, n) => a + (n.influence ?? 0), 0)
  const fvu = g.fvu?.[String(l)]
  const kv = l >= KV_SHARED_FROM ? ` · KV ← ${GLOBAL.has(l) ? 'L23' : 'L22'}` : ''
  return [
    `${ll(l)} · ${GLOBAL.has(l) ? 'global' : 'sliding'} attn${kv}`,
    fvu != null ? `transcoder FVU ${fvu.toFixed(2)}` : '',
    `${here.filter(n => n.kind === 'feature').length} features ${sum('feature').toFixed(3)} · error ${sum('error').toFixed(3)}`,
  ]
    .filter(Boolean)
    .join('\n')
}

const tok = (t: string | undefined) => `“${(t ?? '').replace(/\n/g, '↵')}”`

/**
 * x is the token, but a token with no feature or error node in the graph gets a quarter column:
 * most of the influence sits on the last few tokens, and equal columns crowded them into the
 * right edge.
 */
function layout(g: Graph) {
  const first = Math.max(1, g.tokens.length - SHOWN_TOKENS)
  const busy = new Set(
    g.nodes.filter(n => n.kind === 'feature' || n.kind === 'error').map(n => n.pos),
  )
  const widths: number[] = g.tokens.map((_, pos) => (pos < first ? 0 : busy.has(pos) ? 1 : 0.25))
  const total = widths.reduce((s, w) => s + w, 0) || 1
  const left: number[] = []
  let acc = 0
  for (const w of widths) {
    left.push(acc)
    acc += w
  }
  const span = W - PAD.l - PAD.r
  const x = (pos: number) => {
    const p = Math.min(Math.max(pos, first), g.tokens.length - 1)
    return PAD.l + ((left[p] + widths[p] / 2) / total) * span
  }
  const y = (layer: number) => PAD.t + (1 - (layer + 1) / 44) * (H - PAD.t - PAD.b)
  return { first, x, y }
}

function svg(g: Graph, labels: Record<string, Record<string, Label | string>>): string {
  const { first, x, y } = layout(g)
  const nodes = g.nodes.filter(n => n.kind !== 'bias' && (n.pos >= first || n.kind === 'logit'))
  const byId = new Map(nodes.map(n => [n.id, n]))
  const max = Math.max(...nodes.filter(n => n.kind !== 'logit').map(n => n.influence ?? 0), 1e-9)
  const at = (n: GraphNode) => [
    x(n.pos),
    y(n.kind === 'embedding' ? -1 : n.kind === 'logit' ? 42 : n.layer),
  ]
  const links = g.links
    .filter(l => byId.has(l.source) && byId.has(l.target))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, EDGES)
  const wmax = Math.max(...links.map(l => Math.abs(l.weight)), 1e-9)
  const edgeSvg = links
    .map(l => {
      const [x1, y1] = at(byId.get(l.source)!)
      const [x2, y2] = at(byId.get(l.target)!)
      const o = (0.12 + (0.6 * Math.abs(l.weight)) / wmax).toFixed(2)
      return `<line class="lat-cx-e" data-s="${esc(l.source)}" data-t="${esc(l.target)}"${l.weight < 0 ? ' data-neg' : ''} x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke-opacity="${o}"/>`
    })
    .join('')
  const nodeSvg = nodes
    .map(n => {
      const [cx, cy] = at(n)
      const r = 1.2 + 4.5 * Math.sqrt((n.influence ?? 0) / max)
      if (n.kind === 'error')
        return `<rect class="lat-cx-n" data-kind="error" data-id="${esc(n.id)}" data-tip="${esc(`error ${ll(n.layer)} · ${tok(g.tokens[n.pos])}\ntranscoder residual · infl ${(n.influence ?? 0).toFixed(3)}`)}" x="${(cx - r).toFixed(1)}" y="${(cy - r).toFixed(1)}" width="${(2 * r).toFixed(1)}" height="${(2 * r).toFixed(1)}"/>`
      if (n.kind === 'logit')
        return `<circle class="lat-cx-n" data-kind="logit" data-id="${esc(n.id)}" data-tip="${esc(`logit ${tok(n.token)} · p ${(n.prob ?? 0).toFixed(3)}`)}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4"/>`
      const lab = n.kind === 'feature' ? labelOf(labels, n.layer, n.feature ?? -1) : null
      const infl = `infl ${(n.influence ?? 0).toFixed(3)}`
      const tip =
        n.kind === 'feature'
          ? [
              `${ll(n.layer)} f${n.feature} · ${tok(g.tokens[n.pos])}`,
              lab
                ? `${lab.label}${lab.score !== undefined ? ` · det ${lab.score.toFixed(2)}` : ''}`
                : 'unlabelled',
              `act ${(n.act ?? 0).toFixed(2)} · ${infl}`,
            ].join('\n')
          : `emb ${tok(g.tokens[n.pos])} · ${infl}`
      return `<circle class="lat-cx-n" data-kind="${n.kind}"${(lab?.score ?? 0) >= 0.75 ? ' data-strong' : ''} data-id="${esc(n.id)}" data-tip="${esc(tip)}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}"/>`
    })
    .join('')
  // the grid the graph lives on: a hairline per layer, one per token column
  const x0 = x(first)
  const x1 = x(g.tokens.length - 1)
  const grid = [
    ...Array.from(
      { length: 42 },
      (_, l) =>
        `<line class="lat-cx-g"${l % 10 === 0 || l === 41 ? ' data-major' : ''} x1="${x0.toFixed(1)}" x2="${x1.toFixed(1)}" y1="${y(l).toFixed(1)}" y2="${y(l).toFixed(1)}"/>`,
    ),
    ...g.tokens.slice(first).map((_, i) => {
      const px = x(first + i).toFixed(1)
      return `<line class="lat-cx-g" x1="${px}" x2="${px}" y1="${y(41).toFixed(1)}" y2="${y(-1).toFixed(1)}"/>`
    }),
  ].join('')
  const step = y(0) - y(1)
  const rows = Array.from(
    { length: 42 },
    (_, l) =>
      `<rect class="lat-cx-row" data-tip="${esc(layerTip(g, l))}" x="0" width="${W}" y="${(y(l) - step / 2).toFixed(1)}" height="${step.toFixed(1)}"/>`,
  ).join('')
  const ticks = [0, 10, 20, 30, 41]
    .map(l => `<text class="lat-cx-t" x="2" y="${(y(l) + 3).toFixed(1)}">${ll(l)}</text>`)
    .join('')
  const toks = g.tokens
    .slice(first)
    .map((t, i) => {
      const px = x(first + i)
      const py = H - PAD.b + 8
      return `<text class="lat-cx-tok" transform="translate(${px.toFixed(1)},${py}) rotate(-60)">${esc(t.replace(/\n/g, '↵').slice(0, 14))}</text>`
    })
    .join('')
  return `<svg class="lat-cx" viewBox="0 0 ${W} ${H}" role="img" aria-label="attribution graph">${rows}${grid}${ticks}<g>${edgeSvg}</g><g>${nodeSvg}</g>${toks}</svg>`
}

function shares(g: Graph): string {
  const by: Record<string, number> = {}
  for (const n of g.nodes)
    if (n.kind !== 'logit') by[n.kind] = (by[n.kind] ?? 0) + (n.influence ?? 0)
  return ['feature', 'error', 'embedding']
    .map(
      k =>
        `${k === 'feature' ? 'features' : k === 'error' ? 'errors' : 'emb'} ${(by[k] ?? 0).toFixed(2)}`,
    )
    .join(' · ')
}

function topFeatures(g: Graph, labels: Record<string, Record<string, Label | string>>): string {
  return g.nodes
    .filter(n => n.kind === 'feature')
    .sort((a, b) => (b.influence ?? 0) - (a.influence ?? 0))
    .slice(0, 10)
    .map(n => {
      const lab = labelOf(labels, n.layer, n.feature ?? -1)
      const strong = (lab?.score ?? 0) >= 0.75
      return `<li class="lat-feat"><span class="lat-feat-meta">${(n.influence ?? 0).toFixed(3)}</span><span class="lat-feat-name"${strong ? '' : ' data-weak'} data-cx="${esc(n.id)}">${esc(lab?.label ?? n.label ?? `f${n.feature}`)}</span><span class="lat-feat-meta">${ll(n.layer)} ${esc(g.tokens[n.pos] ?? '')}</span><span class="lat-feat-meta">${lab?.score !== undefined ? lab.score.toFixed(2) : ''}</span></li>`
    })
    .join('')
}

async function render(rt: Runtime, row: number, seq: CircuitState): Promise<void> {
  const n = ++seq.n
  seq.graph = null
  const bake = await bakeOf(rt)
  if (!bake || n !== seq.n) return
  const g = await fetchJson<Graph>(rt, `${bake.base}/graphs/${row}.json`).catch(() => null)
  if (n !== seq.n) return
  const have = rt.probe.querySelector<HTMLElement>('.lat-panel[data-panel="circuit"]')
  if (!g) {
    if (have)
      have.innerHTML = `<section class="lat-feat-group"><p class="lat-gm-note">no circuit for this passage</p></section>`
    return
  }
  const labels = await bake.labels
  if (n !== seq.n) return
  seq.graph = g
  const panel = have ?? probeTab(rt, 'circuit', 'lat-features lat-gemma lat-circuit')
  const target = g.logits[0]
  panel.innerHTML = [
    `<section class="lat-feat-group lat-gm-head"><header><span class="lat-feat-head">→ “${esc(target?.token ?? '')}”</span><span class="lat-count">p ${(target?.prob ?? 0).toFixed(2)}</span></header>`,
    `<p class="lat-gm-note"><span data-tip="${esc(MODEL_TIP)}" data-tip-wide>Gemma 4 E4B</span> · influence ${shares(g)}</p></section>`,
    `<section class="lat-feat-group">${svg(g, labels)}</section>`,
    `<section class="lat-feat-group"><header><span class="lat-feat-head">most influence</span><span class="lat-count">detection</span></header><ul class="lat-feat-list">${topFeatures(g, labels)}</ul></section>`,
  ].join('')
}

/** the graph's window in the reader: context faint, the traced token, the hovered node's token strong */
function markWindow(rt: Runtime, g: Graph, id: string | null): void {
  const at = g.passage
  if (!at) return
  const node = id ? g.nodes.find(n => n.id === id) : undefined
  const spans: [number, number, number][] = [
    [at.start, at.end, 0],
    [at.target[0], at.target[1], 1],
  ]
  const own = node && node.kind !== 'logit' && node.pos >= 1 ? at.spans[node.pos - 1] : undefined
  if (own) spans.push([own[0], own[1], 2])
  void markPassage(rt, at.row, spans)
}

/** dim every edge not touching the hovered node */
function hover(panel: HTMLElement, id: string | null): void {
  const svgEl = panel.querySelector('svg.lat-cx')
  if (!svgEl) return
  if (!id) {
    delete (svgEl as SVGElement).dataset.hover
    for (const e of svgEl.querySelectorAll('.lat-cx-e[data-on]')) e.removeAttribute('data-on')
    return
  }
  ;(svgEl as SVGElement).dataset.hover = ''
  for (const e of svgEl.querySelectorAll<SVGElement>('.lat-cx-e'))
    if (e.dataset.s === id || e.dataset.t === id) e.setAttribute('data-on', '')
    else e.removeAttribute('data-on')
}

export function bindCircuit(rt: Runtime): void {
  if (rt.mode === 'embed') return
  const seq: CircuitState = { n: 0, graph: null }
  const show = () => void render(rt, rt.focus, seq).catch(() => {})
  show()
  const observer = new MutationObserver(show)
  observer.observe(rt.root, { attributes: true, attributeFilter: ['data-latent-focus'] })
  rt.signal.addEventListener('abort', () => observer.disconnect())
  rt.probe.addEventListener(
    'pointerover',
    event => {
      const panel = (event.target as HTMLElement).closest<HTMLElement>('.lat-circuit')
      if (!panel) return
      const el = (event.target as Element).closest<HTMLElement | SVGElement>('[data-id], [data-cx]')
      const id = el
        ? ((el as HTMLElement).dataset.id ?? (el as HTMLElement).dataset.cx ?? null)
        : null
      hover(panel, id)
      if (seq.graph) markWindow(rt, seq.graph, id)
    },
    { signal: rt.signal },
  )
  rt.probe.addEventListener(
    'pointerout',
    event => {
      const panel = (event.target as HTMLElement).closest<HTMLElement>('.lat-circuit')
      const to = (event as PointerEvent).relatedTarget as Node | null
      if (panel && !(to && panel.contains(to))) clearHighlights()
    },
    { signal: rt.signal },
  )
}
