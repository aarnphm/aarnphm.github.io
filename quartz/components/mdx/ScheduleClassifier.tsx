import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/schedule-classifier.inline'
import style from '../styles/scheduleClassifier.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Kind = 'R' | 'W' | 'C' | 'A'
type Op = { tx: number; kind: Kind; item?: string }

type PresetId =
  | 'serial'
  | 'strict'
  | 'lost-update'
  | 'dirty-write'
  | 'dirty-write-cycle'
  | 'serialisable'
  | 'dirty-read-cycle'
  | 'unrecoverable'
  | 'unrecoverable-cycle'

// view="single" draws one preset without the picker, for the slide tables in the note.
type Props = { caption?: string; preset?: PresetId; view?: 'explore' | 'single' }

// Commits are explicit because recoverability, ACA and strictness are defined by them. The
// slide tables omit commits; the serial and serialisable presets add them at each end.
const PRESETS: readonly { id: PresetId; label: string; source: string; ops: string }[] = [
  {
    id: 'serial',
    label: 'serial',
    source: 'The serial slide table. Each transaction commits as it finishes.',
    ops: 'R1(A) W1(A) R1(B) W1(B) C1 R2(A) W2(A) R2(B) W2(B) C2',
  },
  {
    id: 'strict',
    label: 'strict',
    source: '$T_2$ starts early on $B$ and touches $A$ only after $C_1$.',
    ops: 'R1(A) W1(A) R2(B) C1 R2(A) W2(A) C2',
  },
  {
    id: 'lost-update',
    label: 'lost update',
    source: 'The lost-update example from the conflict section.',
    ops: 'R1(A) R2(A) W1(A) C1 W2(A) C2',
  },
  {
    id: 'dirty-write',
    label: 'dirty write',
    source: 'Two blind writes. $T_2$ overwrites $A$ before $T_1$ ends.',
    ops: 'W1(A) W2(A) C1 C2',
  },
  {
    id: 'dirty-write-cycle',
    label: 'dirty write, cycle',
    source: 'The lost update again, with $W_2(A)$ moved before $C_1$.',
    ops: 'R1(A) R2(A) W1(A) W2(A) C1 C2',
  },
  {
    id: 'serialisable',
    label: 'serialisable',
    source:
      'The serialisable slide table: the same read/write order on each item as $T_1$ followed by $T_2$. Both commit at the end.',
    ops: 'R1(A) W1(A) R2(A) W2(A) R1(B) W1(B) R2(B) W2(B) C1 C2',
  },
  {
    id: 'dirty-read-cycle',
    label: 'dirty read, cycle',
    source: '$T_2$ reads $B$ before $T_1$ writes it and reads $A$ after.',
    ops: 'R2(B) W1(A) R2(A) W1(B) C1 C2',
  },
  {
    id: 'unrecoverable',
    label: 'unrecoverable',
    source: 'The abort slide table: $T_2$ commits after reading $W_1(A)$, then $T_1$ aborts.',
    ops: 'R1(A) W1(A) R2(A) W2(A) R2(B) W2(B) C2 A1',
  },
  {
    id: 'unrecoverable-cycle',
    label: 'unrecoverable, cycle',
    source: 'The dirty-read cycle with $C_2$ moved before $C_1$.',
    ops: 'R2(B) W1(A) R2(A) W1(B) C2 C1',
  },
]

const parse = (src: string): Op[] =>
  src.split(/\s+/).map(tok => {
    const m = /^([RWCA])(\d)(?:\((\w)\))?$/.exec(tok)
    if (!m) throw new Error(`bad op ${tok}`)
    return { tx: Number(m[2]), kind: m[1] as Kind, item: m[3] }
  })

const sub = (n: number) => '₀₁₂₃₄₅₆₇₈₉'[n]
const opText = (op: Op) =>
  op.item ? `${op.kind}${sub(op.tx)}(${op.item})` : `${op.kind}${sub(op.tx)}`
const opTex = (op: Op) => (op.item ? `${op.kind}_${op.tx}(${op.item})` : `${op.kind}_${op.tx}`)
const cellText = (op: Op) =>
  op.kind === 'C' ? 'commit' : op.kind === 'A' ? 'abort' : `${op.kind}(${op.item})`

type Level = 'all' | 'rec' | 'aca' | 'strict'
type Region = 'serial' | `${Level}-${'csr' | 'ncsr'}`

type Edge = { from: number; to: number; first: [number, number] }

type Result = {
  ops: Op[]
  txs: number[]
  serial: boolean
  csr: boolean
  order: number[]
  edges: Edge[]
  recoverable: boolean
  aca: boolean
  strict: boolean
  region: Region
  reason: string
  witness: number[]
}

const isData = (op: Op) => op.kind === 'R' || op.kind === 'W'
const isEnd = (op: Op) => op.kind === 'C' || op.kind === 'A'

const classify = (ops: Op[]): Result => {
  const txs = [...new Set(ops.map(o => o.tx))].sort((a, b) => a - b)
  const endAt = (tx: number) => ops.findIndex(o => o.tx === tx && isEnd(o))
  const commitAt = (tx: number) => ops.findIndex(o => o.tx === tx && o.kind === 'C')

  // Serial: each transaction's operations, its commit or abort included, are contiguous.
  const runs = ops.map(o => o.tx).filter((tx, i, a) => i === 0 || a[i - 1] !== tx)
  const serial = new Set(runs).size === runs.length

  // Precedence graph over every transaction, aborted or not, as the note's test builds it.
  const edges: Edge[] = []
  ops.forEach((a, i) => {
    if (!isData(a)) return
    for (let j = i + 1; j < ops.length; j++) {
      const b = ops[j]
      const conflict =
        isData(b) && b.tx !== a.tx && b.item === a.item && (a.kind === 'W' || b.kind === 'W')
      if (conflict && !edges.some(e => e.from === a.tx && e.to === b.tx))
        edges.push({ from: a.tx, to: b.tx, first: [i, j] })
    }
  })
  const order: number[] = []
  const remaining = new Set(txs)
  while (remaining.size) {
    const next = [...remaining].find(
      t => !edges.some(e => e.to === t && remaining.has(e.from) && e.from !== t),
    )
    if (next === undefined) break
    order.push(next)
    remaining.delete(next)
  }
  const csr = remaining.size === 0

  // A read sees the latest preceding write on its item unless that write's transaction aborted first.
  const readsFrom = ops.flatMap((op, j) => {
    if (op.kind !== 'R') return []
    for (let i = j - 1; i >= 0; i--) {
      const w = ops[i]
      if (w.kind !== 'W' || w.item !== op.item) continue
      const abort = ops.findIndex(o => o.tx === w.tx && o.kind === 'A')
      if (abort !== -1 && abort < j) continue
      return w.tx === op.tx ? [] : [{ w: i, r: j }]
    }
    return []
  })

  const unrecoverable = readsFrom.find(({ w, r }) => {
    const readerCommit = commitAt(ops[r].tx)
    if (readerCommit === -1) return false
    const writerCommit = commitAt(ops[w].tx)
    return writerCommit === -1 || writerCommit > readerCommit
  })
  const dirtyRead = readsFrom.find(({ w, r }) => {
    const writerCommit = commitAt(ops[w].tx)
    return writerCommit === -1 || writerCommit > r
  })
  let unstrict: [number, number] | undefined
  ops.forEach((w, i) => {
    if (unstrict || w.kind !== 'W') return
    const end = endAt(w.tx)
    for (let j = i + 1; j < ops.length && (end === -1 || j < end); j++) {
      const o = ops[j]
      if (isData(o) && o.tx !== w.tx && o.item === w.item) {
        unstrict = [i, j]
        return
      }
    }
  })

  const recoverable = !unrecoverable
  const aca = !dirtyRead
  const strict = !unstrict
  const level: Level = strict ? 'strict' : aca ? 'aca' : recoverable ? 'rec' : 'all'
  const region: Region = serial ? 'serial' : `${level}-${csr ? 'csr' : 'ncsr'}`

  let reason: string
  let witness: number[]
  if (unrecoverable) {
    const { w, r } = unrecoverable
    const writer = ops[w].tx
    const readerCommit = commitAt(ops[r].tx)
    const end = endAt(writer)
    witness = [w, r, readerCommit, ...(end === -1 ? [] : [end])]
    reason = `Not recoverable: $${opTex(ops[r])}$ reads $${opTex(ops[w])}$, and $${opTex(ops[readerCommit])}$ comes before $T_${writer}$ commits${end === -1 ? '' : `; $${opTex(ops[end])}$ follows`}.`
  } else if (dirtyRead) {
    const { w, r } = dirtyRead
    witness = [w, r]
    reason = `Recoverable, not ACA: $${opTex(ops[r])}$ reads the uncommitted $${opTex(ops[w])}$. $T_${ops[r].tx}$ commits only after $T_${ops[w].tx}$, but an abort of $T_${ops[w].tx}$ would cascade.`
  } else if (unstrict) {
    const [i, j] = unstrict
    witness = [i, j]
    reason = `ACA, not strict: $${opTex(ops[j])}$ overwrites the uncommitted $${opTex(ops[i])}$. No read is dirty.`
  } else if (!serial) {
    const first = ops[0].tx
    const end = endAt(first)
    const j = ops.findIndex((o, k) => o.tx !== first && k < end)
    witness = [j]
    reason = `Strict, not serial: no item written by an active transaction is read or overwritten, but $${opTex(ops[j])}$ runs before $T_${first}$ ends.`
  } else {
    witness = []
    reason = `Serial: each transaction runs to completion before the next starts.`
  }

  return { ops, txs, serial, csr, order, edges, recoverable, aca, strict, region, reason, witness }
}

const RESULTS = Object.fromEntries(PRESETS.map(p => [p.id, classify(parse(p.ops))])) as Record<
  PresetId,
  Result
>

const REGION_WORDS: Record<Region, string> = {
  serial: 'serial',
  'strict-csr': 'strict, conflict serialisable, not serial',
  'strict-ncsr': 'strict, not conflict serialisable',
  'aca-csr': 'ACA, not strict, conflict serialisable',
  'aca-ncsr': 'ACA, not strict, not conflict serialisable',
  'rec-csr': 'recoverable, not ACA, conflict serialisable',
  'rec-ncsr': 'recoverable, not ACA, not conflict serialisable',
  'all-csr': 'not recoverable, conflict serialisable',
  'all-ncsr': 'not recoverable, not conflict serialisable',
}

// Venn geometry follows the slide: four nested recovery classes, serial at the centre, and the
// conflict-serialisable band crossing all of them.
type Box = { x0: number; y0: number; x1: number; y1: number }
const VENN_W = 340
const VENN_H = 230
const BOX: Record<'all' | 'rec' | 'aca' | 'strict' | 'serial' | 'csr', Box> = {
  all: { x0: 2, y0: 2, x1: 338, y1: 228 },
  rec: { x0: 14, y0: 46, x1: 326, y1: 186 },
  aca: { x0: 30, y0: 66, x1: 310, y1: 166 },
  strict: { x0: 46, y0: 86, x1: 294, y1: 146 },
  serial: { x0: 146, y0: 102, x1: 194, y1: 130 },
  csr: { x0: 118, y0: 14, x1: 222, y1: 206 },
}
const inside = (b: Box, x: number, y: number) => x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1
const regionAt = (x: number, y: number): Region => {
  if (inside(BOX.serial, x, y)) return 'serial'
  const level: Level = inside(BOX.strict, x, y)
    ? 'strict'
    : inside(BOX.aca, x, y)
      ? 'aca'
      : inside(BOX.rec, x, y)
        ? 'rec'
        : 'all'
  return `${level}-${inside(BOX.csr, x, y) ? 'csr' : 'ncsr'}`
}
const cuts = (axis: 'x' | 'y') =>
  [...new Set(Object.values(BOX).flatMap(b => (axis === 'x' ? [b.x0, b.x1] : [b.y0, b.y1])))].sort(
    (a, b) => a - b,
  )
const XS = cuts('x')
const YS = cuts('y')
const VENN_CELLS = XS.slice(0, -1).flatMap((x, i) =>
  YS.slice(0, -1).map((y, j) => ({
    x,
    y,
    w: XS[i + 1] - x,
    h: YS[j + 1] - y,
    region: regionAt((x + XS[i + 1]) / 2, (y + YS[j + 1]) / 2),
  })),
)

const VennDiagram = ({ id }: { id: string }) => (
  <svg
    class="scl-venn"
    viewBox={`0 0 ${VENN_W} ${VENN_H}`}
    role="img"
    aria-labelledby={`${id}-venn-title`}
  >
    <title id={`${id}-venn-title`}>
      Schedule classes: recoverable contains ACA, which contains strict, which contains serial. The
      conflict-serialisable band crosses every class and contains serial.
    </title>
    {VENN_CELLS.map(c => (
      <rect class="scl-cell" data-region={c.region} x={c.x} y={c.y} width={c.w} height={c.h} />
    ))}
    {(['all', 'rec', 'aca', 'strict', 'csr', 'serial'] as const).map(k => (
      <rect
        class={`scl-class scl-class--${k}`}
        x={BOX[k].x0}
        y={BOX[k].y0}
        width={BOX[k].x1 - BOX[k].x0}
        height={BOX[k].y1 - BOX[k].y0}
      />
    ))}
    <text class="scl-venn-label" x={BOX.all.x1 - 6} y={BOX.all.y0 + 12} text-anchor="end">
      all schedules
    </text>
    <text class="scl-venn-label" x={BOX.rec.x0 + 5} y={BOX.rec.y0 + 12}>
      recoverable
    </text>
    <text class="scl-venn-label" x={BOX.aca.x0 + 5} y={BOX.aca.y0 + 12}>
      ACA
    </text>
    <text class="scl-venn-label" x={BOX.strict.x0 + 5} y={BOX.strict.y0 + 12}>
      strict
    </text>
    <text
      class="scl-venn-label"
      x={(BOX.csr.x0 + BOX.csr.x1) / 2}
      y={BOX.csr.y0 + 12}
      text-anchor="middle"
    >
      conflict
    </text>
    <text
      class="scl-venn-label"
      x={(BOX.csr.x0 + BOX.csr.x1) / 2}
      y={BOX.csr.y0 + 24}
      text-anchor="middle"
    >
      serialisable
    </text>
    <text
      class="scl-venn-label"
      x={(BOX.serial.x0 + BOX.serial.x1) / 2}
      y={BOX.serial.y0 + 17}
      text-anchor="middle"
    >
      serial
    </text>
  </svg>
)

const NODE = 36
const GRAPH_W = 220
const GRAPH_H = 72
const NODE_Y = (GRAPH_H - NODE) / 2
const NODE_X: Record<number, number> = { 1: 6, 2: GRAPH_W - 6 - NODE }

const PrecedenceGraph = ({ r }: { r: Result }) => {
  const cycle = !r.csr
  return (
    <svg
      class="scl-graph"
      viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`}
      role="img"
      aria-label={graphLabel(r)}
    >
      {r.edges.map(e => {
        const forward = NODE_X[e.from] < NODE_X[e.to]
        const y = forward ? NODE_Y + 11 : NODE_Y + NODE - 11
        const x0 = forward ? NODE_X[e.from] + NODE : NODE_X[e.from]
        const x1 = forward ? NODE_X[e.to] : NODE_X[e.to] + NODE
        const head = forward ? -7 : 7
        const [a, b] = e.first
        return (
          <g class={`scl-edge${cycle ? ' scl-edge--cycle' : ''}`}>
            <line x1={x0} y1={y} x2={x1 + head} y2={y} />
            <polygon points={`${x1},${y} ${x1 + head},${y - 3.5} ${x1 + head},${y + 3.5}`} />
            <text x={GRAPH_W / 2} y={forward ? y - 5 : y + 12} text-anchor="middle">
              {`${opText(r.ops[a])} before ${opText(r.ops[b])}`}
            </text>
          </g>
        )
      })}
      {r.txs.map(t => (
        <g class="scl-node">
          <rect x={NODE_X[t]} y={NODE_Y} width={NODE} height={NODE} />
          <text x={NODE_X[t] + NODE / 2} y={NODE_Y + NODE / 2 + 4} text-anchor="middle">
            {`T${sub(t)}`}
          </text>
        </g>
      ))}
    </svg>
  )
}

const graphLabel = (r: Result) =>
  r.edges.length === 0
    ? 'Precedence graph with no edges.'
    : `Precedence graph: ${r.edges.map(e => `T${e.from} to T${e.to}`).join(', ')}${r.csr ? '' : ', a cycle'}.`

const yesNo = (b: boolean) => (b ? 'yes' : 'no')
const csrText = (r: Result) =>
  r.csr ? `yes, order $${r.order.map(t => `T_${t}`).join(',')}$` : `no, cycle $T_1\\to T_2\\to T_1$`

let instance = 0

const ScheduleClassifierImpl: QuartzMdxComponent<Props> = ({
  caption,
  preset = 'serialisable',
  view = 'explore',
}) => {
  const controls = view !== 'single'
  const id = `scl-${++instance}`
  const p0 = PRESETS.some(p => p.id === preset) ? preset : 'serialisable'
  const shown = controls ? PRESETS : PRESETS.filter(p => p.id === p0)
  const regions = Object.fromEntries(PRESETS.map(p => [p.id, RESULTS[p.id].region]))

  return (
    <figure
      class="schedule-classifier"
      data-schedule-classifier
      data-scl-preset={p0}
      data-scl-region={RESULTS[p0].region}
      data-scl-regions={JSON.stringify(regions)}
    >
      {controls ? (
        <fieldset class="scl-segments">
          <legend class="scl-legend">Schedule</legend>
          {PRESETS.map(p => (
            <label class="scl-segment">
              <input
                type="radio"
                name={`${id}-preset`}
                value={p.id}
                checked={p.id === p0}
                data-scl-input
              />
              <span>{p.label}</span>
            </label>
          ))}
        </fieldset>
      ) : null}

      {shown.map(p => (
        <p class="scl-source" data-scl-panel={p.id} hidden={p.id !== p0}>
          {/* One math run per operation, so the schedule wraps at commas on narrow screens. */}
          <span class="scl-schedule">
            <MathText
              text={`$S$: ${RESULTS[p.id].ops.map(o => `$${opTex(o)}$`).join(', ')}`}
              mathClass="scl-math"
            />
          </span>
          <span class="scl-source-note">
            <MathText text={p.source} mathClass="scl-math" />
          </span>
        </p>
      ))}

      <div class="scl-body">
        <div class="scl-left">
          {shown.map(p => {
            const r = RESULTS[p.id]
            const lit = new Set(r.witness)
            return (
              <table class="scl-timeline" data-scl-panel={p.id} hidden={p.id !== p0}>
                <thead>
                  <tr>
                    <th scope="col">step</th>
                    {r.txs.map(t => (
                      <th scope="col">
                        <MathText text={`$T_${t}$`} mathClass="scl-math" />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {r.ops.map((op, i) => (
                    <tr>
                      <td class="scl-step">{i + 1}</td>
                      {r.txs.map(t =>
                        t === op.tx ? (
                          <td
                            class={`scl-op scl-op--${op.kind}`}
                            data-scl-witness={lit.has(i) ? 'true' : undefined}
                          >
                            {cellText(op)}
                          </td>
                        ) : (
                          <td class="scl-op scl-op--empty" />
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          })}
        </div>
        <div class="scl-right">
          <VennDiagram id={id} />
          <p class="scl-heading">Precedence graph</p>
          {shown.map(p => (
            <div class="scl-graph-panel" data-scl-panel={p.id} hidden={p.id !== p0}>
              <PrecedenceGraph r={RESULTS[p.id]} />
            </div>
          ))}
        </div>
      </div>

      {shown.map(p => {
        const r = RESULTS[p.id]
        return (
          <div class="scl-readout-panel" data-scl-panel={p.id} hidden={p.id !== p0}>
            <dl class="scl-readout">
              <div>
                <dt>Conflict serialisable</dt>
                <dd data-scl-out="csr">
                  <MathText text={csrText(r)} mathClass="scl-math" />
                </dd>
              </div>
              <div>
                <dt>Recoverable</dt>
                <dd data-scl-out="recoverable">{yesNo(r.recoverable)}</dd>
              </div>
              <div>
                <dt>ACA (cascadeless)</dt>
                <dd data-scl-out="aca">{yesNo(r.aca)}</dd>
              </div>
              <div>
                <dt>Strict</dt>
                <dd data-scl-out="strict">{yesNo(r.strict)}</dd>
              </div>
              <div>
                <dt>Serial</dt>
                <dd data-scl-out="serial">{yesNo(r.serial)}</dd>
              </div>
              <div>
                <dt>Region</dt>
                <dd data-scl-out="region" data-scl-region-value={r.region}>
                  {REGION_WORDS[r.region]}
                </dd>
              </div>
            </dl>
            <p class="scl-reason">
              <MathText text={r.reason} mathClass="scl-math" />
            </p>
          </div>
        )
      })}

      {caption ? (
        <figcaption class="scl-caption">
          <MathText text={caption} mathClass="scl-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ScheduleClassifierComponent = ScheduleClassifierImpl as QuartzMdxComponent<Props>
ScheduleClassifierComponent.css = style
ScheduleClassifierComponent.afterDOMLoaded = script

export const ScheduleClassifier = registerMdxComponent(
  'ScheduleClassifier',
  ScheduleClassifierComponent,
)

export default (() => ScheduleClassifier) satisfies (opts: undefined) => QuartzMdxComponent<Props>
