import { type FunctionalComponent } from 'preact'
import { MathText, renderInlineMath } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/constrained-decoding-fsm.inline'
import style from '../styles/constrainedDecodingFsm.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

// One decode step: the FSM state before sampling, the candidate tokens the mask checks there,
// and the token the sampler picks. `to` is the state an allowed token leads to.
type Candidate = { tok: string; to?: number }
type Step = { state: number; taken: string[]; cands: Candidate[]; sampled?: string }

// Character FSM for `"age":[0-9]+,` as LMSys draws it (state 0 is after the opening quote).
const EDGES = [
  { id: 'in', from: -1, to: 0, label: '"' },
  { id: '0-1', from: 0, to: 1, label: 'a' },
  { id: '1-2', from: 1, to: 2, label: 'g' },
  { id: '2-3', from: 2, to: 3, label: 'e' },
  { id: '3-4', from: 3, to: 4, label: '"' },
  { id: '4-5', from: 4, to: 5, label: ':' },
  { id: '5-6', from: 5, to: 6, label: '[0-9]' },
  { id: '6-6', from: 6, to: 6, label: '[0-9]' },
  { id: '6-7', from: 6, to: 7, label: ',' },
] as const

const STATES = [0, 1, 2, 3, 4, 5, 6, 7]
const ACCEPT = 7

const PREFIX =
  'Please fill in the following information about Harry Potter.\n{\n  "name": "Harry",\n  "'

const STEPS: Step[] = [
  {
    state: 0,
    taken: ['in'],
    cands: [{ tok: 'age', to: 3 }, { tok: 'Age' }, { tok: 'hou' }],
    sampled: 'age',
  },
  {
    state: 3,
    taken: ['0-1', '1-2', '2-3'],
    cands: [{ tok: '":', to: 5 }, { tok: '"', to: 4 }, { tok: 's' }],
    sampled: '":',
  },
  {
    state: 5,
    taken: ['3-4', '4-5'],
    cands: [{ tok: '0', to: 6 }, { tok: '1', to: 6 }, { tok: 'fif' }],
    sampled: '1',
  },
  {
    state: 6,
    taken: ['5-6'],
    cands: [{ tok: '7', to: 6 }, { tok: ',', to: 7 }, { tok: '}' }],
    sampled: '7',
  },
  {
    state: 6,
    taken: ['6-6'],
    cands: [{ tok: ',', to: 7 }, { tok: '0', to: 6 }, { tok: '"' }],
    sampled: ',',
  },
  { state: 7, taken: ['6-7'], cands: [] },
]

const LAST = STEPS.length - 1

const allowedCount = (s: Step) => s.cands.filter(c => c.to !== undefined).length

function statusText(i: number): string {
  const s = STEPS[i]
  if (!s.sampled)
    return `step ${i} of ${LAST}: state ${s.state} accepts, the age clause is complete`
  return `step ${i} of ${LAST}: state ${s.state}, ${allowedCount(s)} of ${s.cands.length} candidates allowed, sampled ${s.sampled}`
}
const SAMPLED = STEPS.flatMap(s => (s.sampled ? [s.sampled] : []))

// Cumulative states and edges reached before each step; the script and the SSR state read these.
const VIEWS = STEPS.map((s, i) => {
  const before = STEPS.slice(0, i + 1)
  const edges = before.flatMap(b => b.taken)
  const states = new Set<number>([0])
  for (const id of edges) {
    const e = EDGES.find(x => x.id === id)
    if (e && e.to >= 0) states.add(e.to)
  }
  return {
    state: s.state,
    visited: [...states],
    seen: edges.filter(e => !s.taken.includes(e)),
    taken: s.taken,
    status: statusText(i),
  }
})

const R = 17
type Pt = { x: number; y: number }
type Layout = { w: number; h: number; pos: Pt[]; entry: Pt; loopBelow: boolean }

const WIDE: Layout = {
  w: 690,
  h: 120,
  // The 5 to 6 edge carries the widest label, [0-9], so it gets extra room.
  pos: STATES.map(i => ({ x: 64 + i * 80 + (i >= 6 ? 18 : 0), y: 84 })),
  entry: { x: 12, y: 84 },
  loopBelow: false,
}

// Two rows, the second running right to left, so the labels stay legible at phone width.
const NARROW: Layout = {
  w: 340,
  h: 250,
  pos: [50, 130, 210, 290, 290, 210, 116, 36].map((x, i) => ({ x, y: i < 4 ? 64 : 168 })),
  entry: { x: 4, y: 64 },
  loopBelow: true,
}

type EdgeGeometry = { d: string; lx: number; ly: number; anchor?: 'start' | 'middle' }

const edgeGeometry = (layout: Layout, e: (typeof EDGES)[number]): EdgeGeometry => {
  const { pos } = layout
  if (e.from === -1) {
    const b = pos[0]
    return {
      d: `M ${layout.entry.x} ${b.y} L ${b.x - R} ${b.y}`,
      lx: (layout.entry.x + b.x - R) / 2,
      ly: b.y - 9,
    }
  }
  if (e.from === e.to) {
    const p = pos[e.from]
    const s = layout.loopBelow ? 1 : -1
    return {
      d: `M ${p.x + 8} ${p.y + s * (R - 2)} C ${p.x + 30} ${p.y + s * 62}, ${p.x - 30} ${p.y + s * 62}, ${p.x - 8} ${p.y + s * (R - 2)}`,
      lx: p.x,
      ly: p.y + s * 50 + (layout.loopBelow ? 18 : -10),
    }
  }
  const a = pos[e.from]
  const b = pos[e.to]
  if (a.y === b.y) {
    const dir = Math.sign(b.x - a.x)
    return {
      d: `M ${a.x + dir * R} ${a.y} L ${b.x - dir * R} ${b.y}`,
      lx: (a.x + b.x) / 2,
      ly: a.y - 9,
    }
  }
  return {
    d: `M ${a.x} ${a.y + R} L ${b.x} ${b.y - R}`,
    lx: a.x + 14,
    ly: (a.y + b.y) / 2 + 4,
    anchor: 'start',
  }
}

const Fsm: FunctionalComponent<{ layout: Layout; variant: string }> = ({ layout, variant }) => {
  const v0 = VIEWS[0]
  return (
    <svg
      class={`cdfsm-svg cdfsm-svg--${variant}`}
      viewBox={`0 0 ${layout.w} ${layout.h}`}
      role="img"
      aria-label='Finite-state machine for the "age" clause: states 0 to 7 read the characters quote, a, g, e, quote, colon, then one digit to state 6, which loops on further digits, and a comma to the accepting state 7.'
    >
      {EDGES.map(e => {
        const g = edgeGeometry(layout, e)
        const cls = v0.taken.includes(e.id) ? ' is-taken' : ''
        return (
          <g key={e.id} class={`cdfsm-edge${cls}`} data-cdfsm-edge={e.id}>
            <path d={g.d} />
            <text x={g.lx} y={g.ly} text-anchor={g.anchor ?? 'middle'}>
              {e.label}
            </text>
          </g>
        )
      })}
      {STATES.map(i => {
        const p = layout.pos[i]
        const cls = [
          i === v0.state ? 'is-current' : '',
          v0.visited.includes(i) ? 'is-visited' : '',
          i === ACCEPT ? 'is-accept' : '',
        ]
          .filter(Boolean)
          .join(' ')
        return (
          <g key={i} class={`cdfsm-state ${cls}`} data-cdfsm-state={i}>
            <circle cx={p.x} cy={p.y} r={R} />
            {i === ACCEPT ? <circle class="cdfsm-accept" cx={p.x} cy={p.y} r={R - 4} /> : null}
            <text x={p.x} y={p.y}>
              {i}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

const REGEX = [
  { text: '"""\\{', age: false },
  { text: '    "name": "[\\w\\d\\s]+",', age: false },
  { text: '    "age": [0-9]+,', age: true },
  { text: '    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",', age: false },
  { text: '\\}', age: false },
  { text: '"""', age: false },
]

const CandidatePanel: FunctionalComponent<{ step: Step; index: number }> = ({ step, index }) => (
  <div class="cdfsm-panel" data-cdfsm-panel={index} hidden={index !== 0}>
    <p class="cdfsm-card-title">
      mask at state{' '}
      <span dangerouslySetInnerHTML={{ __html: renderInlineMath(`q_{${step.state}}`) }} />
    </p>
    {step.cands.length ? (
      <>
        <div class="cdfsm-mask" role="table" aria-label="Candidate tokens and the mask">
          <div class="cdfsm-mask-row" role="row">
            <span role="columnheader">token</span>
            <span role="columnheader">mask</span>
            <span role="columnheader">next</span>
          </div>
          {step.cands.map(c => {
            const ok = c.to !== undefined
            const cls = [
              ok ? 'is-allowed' : 'is-masked',
              c.tok === step.sampled ? 'is-sampled' : '',
            ]
              .filter(Boolean)
              .join(' ')
            return (
              <div key={c.tok} class={`cdfsm-mask-row ${cls}`} role="row">
                <span role="cell">
                  <code>{c.tok}</code>
                </span>
                <span role="cell">
                  <span class="cdfsm-mark" aria-hidden="true">
                    {ok ? '✓' : '✗'}
                  </span>
                  {ok ? 'allowed' : 'masked'}
                </span>
                <span role="cell">
                  {ok ? (
                    <span dangerouslySetInnerHTML={{ __html: renderInlineMath(`q_{${c.to}}`) }} />
                  ) : (
                    '–'
                  )}
                </span>
              </div>
            )
          })}
        </div>
        <p class="cdfsm-sampled">
          sampled <code>{step.sampled}</code>, {allowedCount(step)} of {step.cands.length} allowed
        </p>
      </>
    ) : (
      <p class="cdfsm-done">
        state 7 accepts. The mask moves on to the <code>"house"</code> clause.
      </p>
    )}
  </div>
)

const ConstrainedDecodingFSMImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="constrained-decoding-fsm" data-cdfsm data-cdfsm-views={JSON.stringify(VIEWS)}>
    <svg class="cdfsm-defs" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        <marker
          id="cdfsm-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path class="cdfsm-arrowhead" d="M0,0 L10,5 L0,10 z" />
        </marker>
        <marker
          id="cdfsm-arrow-taken"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path class="cdfsm-arrowhead cdfsm-arrowhead--taken" d="M0,0 L10,5 L0,10 z" />
        </marker>
      </defs>
    </svg>

    <div class="cdfsm-top">
      <div class="cdfsm-card cdfsm-regex">
        <p class="cdfsm-card-title">regular expression</p>
        <div class="cdfsm-code" aria-label="Regular expression for the Harry Potter JSON schema">
          {REGEX.map(l => (
            <span key={l.text} class={l.age ? 'cdfsm-regex-line is-age' : 'cdfsm-regex-line'}>
              {l.text}
            </span>
          ))}
        </div>
      </div>

      <div class="cdfsm-card cdfsm-fsm">
        <p class="cdfsm-card-title">
          finite-state machine for <code>"age": [0-9]+,</code>
        </p>
        <Fsm layout={WIDE} variant="wide" />
        <Fsm layout={NARROW} variant="narrow" />
      </div>
    </div>

    <div class="cdfsm-decode">
      <div class="cdfsm-card cdfsm-buffer">
        <p class="cdfsm-card-title">decoding buffer</p>
        <div class="cdfsm-code">
          {PREFIX}
          {SAMPLED.map((t, i) => (
            <span key={i} class="cdfsm-tok" data-cdfsm-tok={i} hidden>
              {t}
            </span>
          ))}
          <span class="cdfsm-caret" aria-hidden="true" />
        </div>
      </div>
      <div class="cdfsm-card cdfsm-candidates">
        {STEPS.map((s, i) => (
          <CandidatePanel key={i} step={s} index={i} />
        ))}
      </div>
    </div>

    <div class="cdfsm-controls" role="group" aria-label="Decode steps">
      <button type="button" class="cdfsm-btn" data-cdfsm-action="back" aria-disabled="true">
        back
      </button>
      <button type="button" class="cdfsm-btn" data-cdfsm-action="step">
        step
      </button>
      <button type="button" class="cdfsm-btn" data-cdfsm-action="reset">
        reset
      </button>
      <p class="cdfsm-status" data-cdfsm-status aria-live="polite">
        {VIEWS[0].status}
      </p>
    </div>

    {caption ? (
      <figcaption class="cdfsm-caption">
        <MathText text={caption} mathClass="cdfsm-math" />
      </figcaption>
    ) : null}
  </figure>
)

const ConstrainedDecodingFSMComponent = ConstrainedDecodingFSMImpl as QuartzMdxComponent<Props>
ConstrainedDecodingFSMComponent.css = style
ConstrainedDecodingFSMComponent.afterDOMLoaded = script

export const ConstrainedDecodingFSM = registerMdxComponent(
  'ConstrainedDecodingFSM',
  ConstrainedDecodingFSMComponent,
)

// The overview that sits above the FSM figure: schema-side automaton and model-side sampling meet
// at the constrained sampler.
type FlowNode = {
  id: string
  label: string[]
  x: number
  y: number
  w: number
  tone?: 'lit' | 'done'
}
type FlowEdge = { d: string; label: string[]; lx: number; ly: number; anchor?: 'start' | 'middle' }
type FlowGroup = { label: string; x: number; y: number; w: number; h: number }
type FlowLayout = {
  w: number
  h: number
  nodes: FlowNode[]
  edges: FlowEdge[]
  groups: FlowGroup[]
}

const NH = 36

const FLOW_WIDE: FlowLayout = {
  w: 700,
  h: 226,
  groups: [
    { label: 'structured automaton', x: 8, y: 8, w: 456, h: 88 },
    { label: 'decoding', x: 8, y: 130, w: 456, h: 88 },
  ],
  nodes: [
    { id: 'schema', label: ['schema'], x: 24, y: 46, w: 130 },
    { id: 'bias', label: ['logit bias'], x: 300, y: 46, w: 150 },
    { id: 'lm', label: ['language model'], x: 24, y: 168, w: 130 },
    { id: 'cs', label: ['constrained sampling'], x: 300, y: 168, w: 150, tone: 'lit' },
    { id: 'out', label: ['structured output'], x: 538, y: 168, w: 150, tone: 'done' },
  ],
  edges: [
    { d: 'M 154 64 L 300 64', label: ['defines'], lx: 227, ly: 56 },
    { d: 'M 375 82 L 375 168', label: ['mask'], lx: 383, ly: 130, anchor: 'start' },
    { d: 'M 154 186 L 300 186', label: ['token distribution'], lx: 227, ly: 178 },
    { d: 'M 450 186 L 538 186', label: [], lx: 0, ly: 0 },
  ],
}

const FLOW_NARROW: FlowLayout = {
  w: 340,
  h: 300,
  groups: [
    { label: 'structured automaton', x: 4, y: 4, w: 332, h: 96 },
    { label: 'decoding', x: 4, y: 134, w: 332, h: 96 },
  ],
  nodes: [
    { id: 'schema', label: ['schema'], x: 16, y: 46, w: 120 },
    { id: 'bias', label: ['logit bias'], x: 204, y: 46, w: 120 },
    { id: 'lm', label: ['language', 'model'], x: 16, y: 176, w: 120 },
    { id: 'cs', label: ['constrained', 'sampling'], x: 204, y: 176, w: 120, tone: 'lit' },
    { id: 'out', label: ['structured', 'output'], x: 204, y: 258, w: 120, tone: 'done' },
  ],
  edges: [
    { d: 'M 136 64 L 204 64', label: ['defines'], lx: 170, ly: 57 },
    { d: 'M 264 82 L 264 176', label: ['mask'], lx: 272, ly: 132, anchor: 'start' },
    { d: 'M 136 194 L 204 194', label: ['token', 'distribution'], lx: 170, ly: 170 },
    { d: 'M 264 212 L 264 258', label: [], lx: 0, ly: 0 },
  ],
}

const Flow: FunctionalComponent<{ layout: FlowLayout; variant: string }> = ({
  layout,
  variant,
}) => (
  <svg
    class={`sdflow-svg sdflow-svg--${variant}`}
    viewBox={`0 0 ${layout.w} ${layout.h}`}
    role="img"
    aria-label="The schema defines a logit bias. The language model gives a token distribution. Constrained sampling applies the bias as a mask to that distribution and emits structured output."
  >
    {layout.groups.map(g => (
      <g key={g.label} class="sdflow-group">
        <rect x={g.x} y={g.y} width={g.w} height={g.h} />
        <text x={g.x + 10} y={g.y + 18}>
          {g.label}
        </text>
      </g>
    ))}
    {layout.edges.map(e => (
      <g key={e.d} class="sdflow-edge">
        <path d={e.d} />
        {e.label.map((l, i) => (
          <text
            key={l}
            x={e.lx}
            y={e.ly - (e.label.length - 1 - i) * 13}
            text-anchor={e.anchor ?? 'middle'}
          >
            {l}
          </text>
        ))}
      </g>
    ))}
    {layout.nodes.map(n => (
      <g key={n.id} class={`sdflow-node${n.tone ? ` is-${n.tone}` : ''}`}>
        <rect x={n.x} y={n.y} width={n.w} height={NH} />
        {n.label.map((l, i) => (
          <text key={l} x={n.x + n.w / 2} y={n.y + NH / 2 + (i - (n.label.length - 1) / 2) * 13}>
            {l}
          </text>
        ))}
      </g>
    ))}
  </svg>
)

const StructuredDecodingFlowImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="structured-decoding-flow">
    <svg class="cdfsm-defs" width="0" height="0" aria-hidden="true" focusable="false">
      <defs>
        <marker
          id="sdflow-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path class="sdflow-arrowhead" d="M0,0 L10,5 L0,10 z" />
        </marker>
      </defs>
    </svg>
    <Flow layout={FLOW_WIDE} variant="wide" />
    <Flow layout={FLOW_NARROW} variant="narrow" />
    {caption ? (
      <figcaption class="cdfsm-caption">
        <MathText text={caption} mathClass="cdfsm-math" />
      </figcaption>
    ) : null}
  </figure>
)

const StructuredDecodingFlowComponent = StructuredDecodingFlowImpl as QuartzMdxComponent<Props>
StructuredDecodingFlowComponent.css = style

export const StructuredDecodingFlow = registerMdxComponent(
  'StructuredDecodingFlow',
  StructuredDecodingFlowComponent,
)

export default (() => ConstrainedDecodingFSM) satisfies (
  opts: undefined,
) => QuartzMdxComponent<Props>
