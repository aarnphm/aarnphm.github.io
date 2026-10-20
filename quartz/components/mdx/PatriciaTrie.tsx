import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/patricia-trie.inline'
import style from '../styles/patriciaTrie.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; words?: string[]; step?: number }

// The keys of the Wikimedia Patricia trie figure, in its numbering.
const DEFAULT_WORDS = ['romane', 'romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'rubicundus']

type TNode = { id: number; key?: number; edges: TEdge[] }
type TEdge = { label: string; to: TNode }
type Insert = {
  root: TNode
  word: string
  matched: string[]
  split?: [string, string, string]
  added?: string
  // Child id of each edge on the matched path, and of the new edge.
  path: number[]
  fresh?: number
}

const clone = (n: TNode): TNode => ({
  id: n.id,
  key: n.key,
  edges: n.edges.map(e => ({ label: e.label, to: clone(e.to) })),
})

const common = (a: string, b: string) => {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  return i
}

// Pure insert: returns a new tree. A divergence inside an edge splits it at the common prefix.
const insert = (prev: TNode, word: string, key: number, nextId: () => number): Insert => {
  const root = clone(prev)
  const out: Insert = { root, word, matched: [], path: [] }
  let v = root
  let i = 0
  while (i < word.length) {
    const rest = word.slice(i)
    const e = v.edges.find(x => x.label[0] === rest[0])
    if (!e) {
      const leaf: TNode = { id: nextId(), key, edges: [] }
      v.edges.push({ label: rest, to: leaf })
      v.edges.sort((a, b) => a.label.localeCompare(b.label))
      out.added = rest
      out.fresh = leaf.id
      return out
    }
    const p = common(e.label, rest)
    if (p < e.label.length) {
      const mid: TNode = { id: nextId(), edges: [{ label: e.label.slice(p), to: e.to }] }
      out.split = [e.label, e.label.slice(0, p), e.label.slice(p)]
      e.label = e.label.slice(0, p)
      e.to = mid
    }
    out.matched.push(e.label)
    out.path.push(e.to.id)
    v = e.to
    i += p
  }
  v.key = key
  return out
}

type Placed = {
  nodes: { id: number; x: number; y: number; key?: number; depth: number; spell: string }[]
  edges: { id: number; x1: number; y1: number; x2: number; y2: number; label: string }[]
}

const VIEW_W = 520
const TOP = 22
const LEVEL = 62

const place = (root: TNode): Placed => {
  const leaves: TNode[] = []
  const collect = (n: TNode) =>
    n.edges.length ? n.edges.forEach(e => collect(e.to)) : leaves.push(n)
  collect(root)
  const slot = (VIEW_W - 40) / Math.max(leaves.length, 1)
  const xs = new Map<number, number>()
  leaves.forEach((l, i) => xs.set(l.id, 20 + slot * (i + 0.5)))
  const xOf = (n: TNode): number => {
    if (xs.has(n.id)) return xs.get(n.id)!
    const cx = n.edges.map(e => xOf(e.to))
    const x = (cx[0] + cx[cx.length - 1]) / 2
    xs.set(n.id, x)
    return x
  }
  const placed: Placed = { nodes: [], edges: [] }
  const walk = (n: TNode, depth: number, spell: string) => {
    const x = xOf(n)
    const y = TOP + depth * LEVEL
    placed.nodes.push({ id: n.id, x, y, key: n.key, depth, spell })
    for (const e of n.edges) {
      placed.edges.push({ id: e.to.id, x1: x, y1: y, x2: xOf(e.to), y2: y + LEVEL, label: e.label })
      walk(e.to, depth + 1, spell + e.label)
    }
  }
  walk(root, 0, '')
  return placed
}

const count = (n: TNode): number => 1 + n.edges.reduce((s, e) => s + count(e.to), 0)
const symbols = (n: TNode): number =>
  n.edges.reduce((s, e) => s + e.label.length + symbols(e.to), 0)
const trieNodes = (words: string[]) => {
  const prefixes = new Set([''])
  for (const w of words) for (let i = 1; i <= w.length; i++) prefixes.add(w.slice(0, i))
  return prefixes.size
}

type State = { tree: Placed; lit?: number; path: number[]; say: string; rows: [string, string][] }

const build = (words: string[]): State[] => {
  let id = 0
  const nextId = () => ++id
  let root: TNode = { id: 0, edges: [] }
  const states: State[] = []
  const rows = (k: number): [string, string][] => [
    ['Keys stored', String(k)],
    ['Radix tree nodes', String(count(root))],
    ['Uncompressed trie nodes', String(trieNodes(words.slice(0, k)))],
    ['Symbols on edges, either tree', String(symbols(root))],
  ]
  states.push({
    tree: place(root),
    path: [],
    say: 'The root exists for the empty dictionary. Pick a key to insert the keys up to it.',
    rows: rows(0),
  })
  words.forEach((w, k) => {
    const r = insert(root, w, k + 1, nextId)
    root = r.root
    const parts = [
      r.matched.length ? `match ${r.matched.map(m => `“${m}”`).join(' + ')}` : 'no edge matches',
      r.split ? `split “${r.split[0]}” into “${r.split[1]}” + “${r.split[2]}”` : '',
      r.added ? `add edge “${r.added}”` : 'mark the node terminal',
    ].filter(Boolean)
    states.push({
      tree: place(root),
      lit: r.fresh,
      path: r.path,
      say: `insert ${k + 1} “${w}”: ${parts.join('; ')}.`,
      rows: rows(k + 1),
    })
  })
  return states
}

const LEAF = 16
const INNER = 8
const CH = 7.2

let instance = 0

const PatriciaTrieImpl: QuartzMdxComponent<Props> = ({ caption, words, step }) => {
  const id = `ptr-${++instance}`
  const keys = words?.length ? words : DEFAULT_WORDS
  const states = build(keys)
  const s0 = step !== undefined && step >= 0 && step < states.length ? step : states.length - 1
  const depth = Math.max(...states.flatMap(s => s.tree.nodes.map(n => n.depth)))
  const viewH = TOP + depth * LEVEL + LEAF + 6

  return (
    <figure class="patricia-trie" data-patricia-trie>
      <fieldset class="ptr-segments">
        <legend class="ptr-legend">Insert keys up to</legend>
        {states.map((_, i) => (
          <label class="ptr-segment">
            <input
              type="radio"
              name={`${id}-step`}
              value={String(i)}
              checked={i === s0}
              data-ptr-step-input
            />
            <span>{i === 0 ? '0 empty' : `${i} ${keys[i - 1]}`}</span>
          </label>
        ))}
      </fieldset>

      {states.map((s, i) => (
        <div class="ptr-state" data-ptr-state={String(i)} hidden={i !== s0}>
          <svg
            class="ptr-svg"
            viewBox={`0 0 ${VIEW_W} ${viewH}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Radix tree after ${i} insertions: ${s.tree.nodes.length} nodes, edges ${s.tree.edges.map(e => e.label).join(', ') || 'none'}.`}
          >
            {s.tree.edges.map(e => (
              <line
                class="ptr-edge"
                data-state={e.id === s.lit ? 'lit' : s.path.includes(e.id) ? 'path' : 'idle'}
                x1={e.x1}
                y1={e.y1}
                x2={e.x2}
                y2={e.y2}
              />
            ))}
            {s.tree.nodes.map(n =>
              n.key === undefined ? (
                <rect
                  class={n.depth === 0 ? 'ptr-node ptr-node--root' : 'ptr-node'}
                  x={n.x - INNER / 2}
                  y={n.y - INNER / 2}
                  width={INNER}
                  height={INNER}
                />
              ) : (
                <g
                  class="ptr-key"
                  data-state={n.id === s.lit ? 'lit' : 'idle'}
                  data-ptr-key={String(n.key)}
                  data-ptr-spell={n.spell}
                >
                  <rect x={n.x - LEAF / 2} y={n.y - LEAF / 2} width={LEAF} height={LEAF} />
                  <text x={n.x} y={n.y + 3.5} text-anchor="middle">
                    {n.key}
                  </text>
                </g>
              ),
            )}
            {s.tree.edges.map(e => {
              const w = e.label.length * CH + 8
              const cx = (e.x1 + e.x2) / 2
              const cy = (e.y1 + e.y2) / 2
              return (
                <g
                  class="ptr-label"
                  data-ptr-edge={e.label}
                  data-state={e.id === s.lit ? 'lit' : s.path.includes(e.id) ? 'path' : 'idle'}
                >
                  <rect x={cx - w / 2} y={cy - 8} width={w} height={16} />
                  <text x={cx} y={cy + 3.5} text-anchor="middle">
                    {e.label}
                  </text>
                </g>
              )
            })}
            <text class="ptr-root-label" x={s.tree.nodes[0].x + 10} y={TOP + 3}>
              root
            </text>
          </svg>
          <p class="ptr-say" data-ptr-say>
            {s.say}
          </p>
          <dl class="ptr-readout">
            {s.rows.map(([label, value]) => (
              <div>
                <dt>{label}</dt>
                <dd data-ptr-out={label}>{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}

      {caption ? (
        <figcaption class="ptr-caption">
          <MathText text={caption} mathClass="ptr-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

PatriciaTrieImpl.css = style
PatriciaTrieImpl.afterDOMLoaded = script

export const PatriciaTrie = registerMdxComponent('PatriciaTrie', PatriciaTrieImpl)

export default (() => PatriciaTrie) satisfies (opts: undefined) => QuartzMdxComponent<Props>
