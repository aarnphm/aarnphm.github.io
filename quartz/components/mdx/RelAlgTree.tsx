import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/rel-alg-tree.inline'
import style from '../styles/relAlgTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; select?: string }

type Relation = { attrs: string[]; rows: string[][] }
type Pred = { attr: string; op: '=' | '<'; value: string }
type Expr =
  | { k: 'rel'; id: string; name: string }
  | { k: 'select'; id: string; preds: Pred[]; child: Expr }
  | { k: 'project'; id: string; attrs: string[]; child: Expr }
  | { k: 'rename'; id: string; to: string; attrs: string[]; child: Expr }
  | { k: 'union'; id: string; left: Expr; right: Expr }

// Transcribed from the Sells and Bars tables of the Relational Algebra note. The note renames
// Bars.name to Bars.bar for its natural-join example; the expression tree uses Bars(name, addr).
const DB: Record<string, Relation> = {
  Sells: {
    attrs: ['bar', 'beer', 'price'],
    rows: [
      ['Joe’s', 'Bud', '2.50'],
      ['Joe’s', 'Miller', '2.75'],
      ['Sue’s', 'Bud', '2.50'],
      ['Sue’s', 'Coors', '3.00'],
    ],
  },
  Bars: {
    attrs: ['name', 'addr'],
    rows: [
      ['Joe’s', 'Maple St.'],
      ['Sue’s', 'River Rd.'],
    ],
  },
}

// Bars on Maple St., or bars that sell Bud for less than $3.
const QUERY: Expr = {
  k: 'union',
  id: 'union',
  left: {
    k: 'project',
    id: 'pi-name',
    attrs: ['name'],
    child: {
      k: 'select',
      id: 'sigma-addr',
      preds: [{ attr: 'addr', op: '=', value: 'Maple St.' }],
      child: { k: 'rel', id: 'bars', name: 'Bars' },
    },
  },
  right: {
    k: 'rename',
    id: 'rho',
    to: 'R',
    attrs: ['name'],
    child: {
      k: 'project',
      id: 'pi-bar',
      attrs: ['bar'],
      child: {
        k: 'select',
        id: 'sigma-bud',
        preds: [
          { attr: 'price', op: '<', value: '3' },
          { attr: 'beer', op: '=', value: 'Bud' },
        ],
        child: { k: 'rel', id: 'sells', name: 'Sells' },
      },
    },
  },
}

// ---- evaluator: set semantics ----

const isNum = (v: string) => /^-?\d+(\.\d+)?$/.test(v)
const test = (p: Pred, v: string) => (p.op === '<' ? Number(v) < Number(p.value) : v === p.value)
const distinct = (rows: string[][]) => {
  const seen = new Set<string>()
  return rows.filter(r => {
    const key = JSON.stringify(r)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

const children = (e: Expr): Expr[] =>
  e.k === 'rel' ? [] : e.k === 'union' ? [e.left, e.right] : [e.child]

const evaluate = (e: Expr): Relation => {
  if (e.k === 'rel') return DB[e.name]
  if (e.k === 'union') {
    const a = evaluate(e.left)
    const b = evaluate(e.right)
    if (a.attrs.join() !== b.attrs.join()) throw new Error('union inputs are not union compatible')
    return { attrs: a.attrs, rows: distinct([...a.rows, ...b.rows]) }
  }
  const r = evaluate(e.child)
  if (e.k === 'select')
    return {
      attrs: r.attrs,
      rows: r.rows.filter(row => e.preds.every(p => test(p, row[r.attrs.indexOf(p.attr)]))),
    }
  if (e.k === 'project')
    return {
      attrs: e.attrs,
      rows: distinct(r.rows.map(row => e.attrs.map(a => row[r.attrs.indexOf(a)]))),
    }
  return { attrs: e.attrs, rows: r.rows }
}

// ---- TeX ----

const word = (t: string) => `\\text{${t}}`
const predTex = (p: Pred) =>
  `${word(p.attr)} ${p.op} ${isNum(p.value) ? p.value : word(`“${p.value}”`)}`
const opTex = (e: Expr): string => {
  if (e.k === 'rel') return word(e.name)
  if (e.k === 'union') return '\\cup'
  if (e.k === 'select') return `\\sigma_{${e.preds.map(predTex).join('\\ \\text{AND}\\ ')}}`
  if (e.k === 'project') return `\\pi_{${e.attrs.map(word).join(',')}}`
  return `\\rho_{${e.to}(${e.attrs.map(word).join(',')})}`
}
const exprTex = (e: Expr): string =>
  e.k === 'rel'
    ? opTex(e)
    : e.k === 'union'
      ? `${exprTex(e.left)} \\cup ${exprTex(e.right)}`
      : `${opTex(e)}\\left(${exprTex(e.child)}\\right)`

const plainName = (e: Expr): string =>
  e.k === 'rel'
    ? e.name
    : e.k === 'union'
      ? 'union'
      : e.k === 'select'
        ? `select ${e.preds.map(p => `${p.attr} ${p.op} ${p.value}`).join(' and ')}`
        : e.k === 'project'
          ? `project ${e.attrs.join(', ')}`
          : `rename to ${e.to}(${e.attrs.join(', ')})`

// ---- one state per node ----

type Table = {
  title: string
  attrs: string[]
  tests?: string[]
  rows: string[][]
  kept?: boolean[]
}
type NodeState = {
  id: string
  name: string
  label: string
  expr: string
  text: string
  subtree: string[]
  inputs: Table[]
  result: Table
}

const subtreeOf = (e: Expr): string[] => [e.id, ...children(e).flatMap(subtreeOf)]
const tuples = (n: number) => `${n} tuple${n === 1 ? '' : 's'}`

const describe = (e: Expr): NodeState => {
  const out = evaluate(e)
  const base = {
    id: e.id,
    name: plainName(e),
    label: opTex(e),
    expr: exprTex(e),
    subtree: subtreeOf(e),
  }
  const result: Table = { title: 'Result', attrs: out.attrs, rows: out.rows }
  if (e.k === 'rel')
    return {
      ...base,
      text: `The stored relation ${e.name}(${out.attrs.join(', ')}) holds ${tuples(out.rows.length)}.`,
      inputs: [],
      result: { ...result, title: `Stored relation ${e.name}` },
    }
  if (e.k === 'union') {
    const a = evaluate(e.left)
    const b = evaluate(e.right)
    const both = a.rows.filter(r => b.rows.some(s => s.join() === r.join())).map(r => r.join(', '))
    return {
      ...base,
      text: `Union takes ${a.rows.length} + ${b.rows.length} tuples over the same attributes, ${out.attrs.join(', ')}. ${both.length ? `${both.join(' and ')} ${both.length === 1 ? 'is' : 'are'} in both inputs, so the` : 'The'} set union holds ${tuples(out.rows.length)}.`,
      inputs: [
        { title: 'Left input', ...a },
        { title: 'Right input', ...b },
      ],
      result,
    }
  }
  const r = evaluate(e.child)
  const input: Table = { title: 'Input', ...r }
  if (e.k === 'select') {
    const results = r.rows.map(row => e.preds.map(p => test(p, row[r.attrs.indexOf(p.attr)])))
    const kept = results.map(rs => rs.every(Boolean))
    return {
      ...base,
      text: `Selection tests each of the ${tuples(r.rows.length)} against ${e.preds.map(p => `$${predTex(p)}$`).join(' and ')} and keeps the ${kept.filter(Boolean).length} for which ${e.preds.length > 1 ? 'all hold' : 'it holds'}. The attributes stay the same.`,
      inputs: [
        {
          ...input,
          tests: e.preds.map(predTex),
          rows: r.rows.map((row, i) => [...row, ...results[i].map(String)]),
          kept,
        },
      ],
      result,
    }
  }
  if (e.k === 'project')
    return {
      ...base,
      text: `Projection keeps the ${e.attrs.join(', ')} column of ${tuples(r.rows.length)}. Under set semantics, equal tuples collapse, which leaves ${out.rows.length}.`,
      inputs: [input],
      result,
    }
  const renamed = r.attrs.flatMap((a, i) =>
    a === e.attrs[i] ? [] : [`${a} becomes ${e.attrs[i]}`],
  )
  return {
    ...base,
    text: `Rename calls the result ${e.to} and ${renamed.join(', ')}. The tuples do not change.`,
    inputs: [input],
    result: { ...result, title: `Result ${e.to}` },
  }
}

const walk = (e: Expr): Expr[] => [...children(e).flatMap(walk), e]

// ---- layout ----

const NODE_H = 34
// Wide: the course figure's tree, with the left branch one row short because it has no rename.
const ROW_Y = [6, 66, 126, 186, 246]
const LEFT_X = 140
const RIGHT_X = 408
const PLACE: Record<string, [number, number]> = {
  union: [0, (LEFT_X + RIGHT_X) / 2],
  'pi-name': [2, LEFT_X],
  'sigma-addr': [3, LEFT_X],
  bars: [4, LEFT_X],
  rho: [1, RIGHT_X],
  'pi-bar': [2, RIGHT_X],
  'sigma-bud': [3, RIGHT_X],
  sells: [4, RIGHT_X],
}
// Narrow: the same tree as an indented outline, one node per row, so labels keep their size.
const INDENT = 20
const ROW_H = 42

// Rough rendered width of a node label in figure units: subscripts are smaller.
const labelW = (e: Expr): number => {
  const sub =
    e.k === 'select'
      ? e.preds.map(p => `${p.attr} ${p.op} “${p.value}”`).join(' AND ')
      : e.k === 'project' || e.k === 'rename'
        ? `${e.k === 'rename' ? e.to : ''}(${e.attrs.join(',')})`
        : ''
  const main = e.k === 'rel' ? e.name.length * 8.4 : 13
  return Math.round(main + sub.length * 6 + 22)
}

type Placed = { e: Expr; x: number; y: number; w: number }
type Layout = { nodes: Placed[]; edges: { child: string; d: string }[]; w: number; h: number }

const wideLayout = (order: Expr[]): Layout => {
  const at = new Map(
    order.map(e => {
      const [row, x] = PLACE[e.id]
      const w = labelW(e)
      return [e.id, { e, x: x - w / 2, y: ROW_Y[row], w }]
    }),
  )
  const edges = order.flatMap(e =>
    children(e).map(c => {
      const p = at.get(e.id)!
      const q = at.get(c.id)!
      return { child: c.id, d: `M${p.x + p.w / 2} ${p.y + NODE_H}L${q.x + q.w / 2} ${q.y}` }
    }),
  )
  return { nodes: [...at.values()], edges, w: 560, h: ROW_Y[4] + NODE_H + 6 }
}

const narrowLayout = (root: Expr): Layout => {
  const nodes: Placed[] = []
  const edges: Layout['edges'] = []
  const place = (e: Expr, depth: number, parent?: Placed) => {
    const p = { e, x: 4 + depth * INDENT, y: 4 + nodes.length * ROW_H, w: labelW(e) }
    nodes.push(p)
    if (parent)
      edges.push({
        child: e.id,
        d: `M${parent.x + INDENT / 2} ${parent.y + NODE_H}V${p.y + NODE_H / 2}H${p.x}`,
      })
    for (const c of children(e)) place(c, depth + 1, p)
  }
  place(root, 0)
  return {
    // Rows read top-down, but the radios keep the evaluation order of the wide tree.
    nodes: walk(root).map(e => nodes.find(n => n.e === e)!),
    edges,
    w: Math.max(...nodes.map(n => n.x + n.w)) + 4,
    h: nodes.length * ROW_H + 4,
  }
}

const tex = (t: string, display = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })
const labelHtml = (t: string) => renderFigureLabel(t, m => tex(m))

const TableView: FunctionalComponent<{ t: Table; node: string; role: string }> = ({
  t,
  node,
  role,
}) => (
  <div class="rat-card">
    <p class="rat-card-title">{t.title}</p>
    <div class="rat-scroll">
      <table class="rat-table" data-rat-table={`${node}:${role}`}>
        <thead>
          <tr>
            {t.attrs.map(a => (
              <th scope="col">{a}</th>
            ))}
            {(t.tests ?? []).map(p => (
              <th scope="col" class="rat-test" dangerouslySetInnerHTML={{ __html: tex(p) }} />
            ))}
          </tr>
        </thead>
        <tbody>
          {t.rows.map((row, i) => (
            <tr data-kept={t.kept ? String(t.kept[i]) : undefined}>
              {row.map(v => (
                <td>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    {t.rows.length === 0 ? <p class="rat-empty">no tuples</p> : null}
  </div>
)

let instance = 0

const RelAlgTreeImpl: QuartzMdxComponent<Props> = ({ caption, select = 'union' }) => {
  const uid = `rat-${++instance}`
  const order = walk(QUERY)
  const states = order.map(describe)
  const sel = states.some(s => s.id === select) ? select : 'union'
  const subtrees = Object.fromEntries(states.map(s => [s.id, s.subtree]))
  const stateOf = (id: string) =>
    id === sel ? 'lit' : subtrees[sel].includes(id) ? 'done' : 'idle'
  const layouts: [string, Layout][] = [
    ['wide', wideLayout(order)],
    ['narrow', narrowLayout(QUERY)],
  ]

  return (
    <figure
      class="rel-alg-tree"
      data-rel-alg-tree
      data-rat-sel={sel}
      data-rat-subtrees={JSON.stringify(subtrees)}
    >
      <fieldset class="rat-tree">
        <legend class="rat-legend">Expression tree: select a node</legend>
        {layouts.map(([kind, l]) => (
          <svg
            class={`rat-svg rat-svg--${kind}`}
            viewBox={`0 0 ${l.w} ${l.h}`}
            style={{ maxWidth: `${Math.round(l.w * 1.2)}px` }}
            role="group"
            aria-label="Expression tree: the union of the Maple St. bars and the renamed bars that sell Bud for less than $3."
          >
            {l.edges.map(edge => (
              <path
                class="rat-edge"
                data-rat-edge={edge.child}
                data-state={stateOf(edge.child) === 'done' ? 'on' : 'off'}
                d={edge.d}
              />
            ))}
            {l.nodes.map(p => {
              const s = states.find(x => x.id === p.e.id)!
              return (
                <g class="rat-node" data-rat-node={p.e.id} data-state={stateOf(p.e.id)}>
                  <rect x={p.x} y={p.y} width={p.w} height={NODE_H} />
                  <foreignObject x={p.x} y={p.y} width={p.w} height={NODE_H}>
                    <label class="rat-fo">
                      <input
                        type="radio"
                        name={`${uid}-${kind}`}
                        value={p.e.id}
                        checked={p.e.id === sel}
                        data-rat-input
                        aria-label={`${s.name}, ${tuples(s.result.rows.length)}`}
                      />
                      <span
                        class="rat-label"
                        dangerouslySetInnerHTML={{ __html: labelHtml(s.label) }}
                      />
                    </label>
                  </foreignObject>
                </g>
              )
            })}
          </svg>
        ))}
      </fieldset>

      {states.map(s => (
        <section class="rat-panel" data-rat-panel={s.id} hidden={s.id !== sel}>
          <div class="rat-expr" dangerouslySetInnerHTML={{ __html: tex(s.expr, true) }} />
          <p class="rat-text">
            <MathText text={s.text} mathClass="rat-math" />
          </p>
          <div class="rat-tables">
            {s.inputs.map((t, i) => (
              <TableView t={t} node={s.id} role={`in${i}`} />
            ))}
            <TableView t={s.result} node={s.id} role="out" />
          </div>
        </section>
      ))}

      {caption ? (
        <figcaption class="rat-caption">
          <MathText text={caption} mathClass="rat-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

RelAlgTreeImpl.css = style
RelAlgTreeImpl.afterDOMLoaded = script

export const RelAlgTree = registerMdxComponent('RelAlgTree', RelAlgTreeImpl)

export default (() => RelAlgTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
