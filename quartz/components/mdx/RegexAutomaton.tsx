import { MathText } from '../../util/math-text'
import style from '../styles/regexAutomaton.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Machine = 'dfa' | 'nfa' | 'A0' | 'A13'
type Float = 'none' | 'end'
type Props = { caption?: string; machine?: Machine; float?: Float }

type State = { id: string; x: number; y: number; start?: boolean; accept?: boolean; lit?: boolean }
type Edge = { from: string; to: string; label: string; bend?: number }

type Spec = { w: number; h: number; states: State[]; edges: Edge[]; label: string }

const MACHINES: Record<Machine, Spec> = {
  dfa: {
    w: 420,
    h: 180,
    label: 'Deterministic finite automaton',
    states: [
      { id: '0', x: 50, y: 90, start: true, accept: true, lit: true },
      { id: '12', x: 160, y: 90 },
      { id: '3', x: 270, y: 90 },
      { id: '4', x: 370, y: 90, accept: true },
    ],
    edges: [
      { from: '0', to: '12', label: 'a' },
      { from: '12', to: '3', label: 'b' },
      { from: '3', to: '4', label: 'c' },
      { from: '4', to: '12', label: 'a', bend: -40 },
    ],
  },
  nfa: {
    w: 420,
    h: 200,
    label: 'Nondeterministic finite automaton',
    states: [
      { id: '0', x: 50, y: 100, start: true, lit: true },
      { id: '1', x: 150, y: 50 },
      { id: '2', x: 150, y: 150 },
      { id: '3', x: 260, y: 100 },
      { id: '4', x: 370, y: 100, accept: true },
    ],
    edges: [
      { from: '0', to: '1', label: 'a' },
      { from: '0', to: '2', label: 'a' },
      { from: '1', to: '3', label: 'b' },
      { from: '2', to: '3', label: 'a' },
      { from: '3', to: '4', label: 'c' },
      { from: '4', to: '1', label: 'a', bend: -36 },
    ],
  },
  A0: {
    w: 360,
    h: 180,
    label: 'Automaton A₀',
    states: [
      { id: 'q₀', x: 70, y: 90, start: true, lit: true },
      { id: 'q₁', x: 200, y: 50, accept: true },
      { id: 'q₂', x: 200, y: 140 },
    ],
    edges: [
      { from: 'q₀', to: 'q₀', label: 'ε', bend: -48 },
      { from: 'q₀', to: 'q₁', label: 'a' },
      { from: 'q₀', to: 'q₂', label: 'c' },
      { from: 'q₁', to: 'q₁', label: 'b', bend: -40 },
      { from: 'q₁', to: 'q₂', label: 'c' },
    ],
  },
  A13: {
    w: 360,
    h: 180,
    label: 'Automaton A₁₃',
    states: [
      { id: '0', x: 60, y: 90, start: true, lit: true },
      { id: '1', x: 160, y: 50 },
      { id: '2', x: 160, y: 140 },
      { id: '3', x: 290, y: 90, accept: true },
    ],
    edges: [
      { from: '0', to: '1', label: 'a' },
      { from: '0', to: '2', label: 'b' },
      { from: '1', to: '3', label: 'c' },
      { from: '2', to: '3', label: 'a' },
      { from: '3', to: '1', label: 'a', bend: -34 },
    ],
  },
}

let instance = 0

function edgePath(from: State, to: State, bend = 0): string {
  if (from.id === to.id) {
    const x = from.x
    const y = from.y - 28
    return `M ${from.x - 14} ${from.y - 18} C ${x - 28} ${y - 28}, ${x + 28} ${y - 28}, ${from.x + 14} ${from.y - 18}`
  }
  if (!bend) {
    return `M ${from.x} ${from.y} L ${to.x} ${to.y}`
  }
  const mx = (from.x + to.x) / 2
  const my = (from.y + to.y) / 2 + bend
  return `M ${from.x} ${from.y} Q ${mx} ${my} ${to.x} ${to.y}`
}

function labelPoint(from: State, to: State, bend = 0): { x: number; y: number } {
  if (from.id === to.id) return { x: from.x, y: from.y - 58 }
  if (!bend) return { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 - 10 }
  return { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 + bend / 2 - 8 }
}

const RegexAutomatonImpl: QuartzMdxComponent<Props> = ({
  caption,
  machine = 'dfa',
  float = 'none',
}) => {
  const spec = MACHINES[machine] ?? MACHINES.dfa
  const byId = new Map(spec.states.map(s => [s.id, s]))
  const marker = `ra-arrow-${++instance}`
  const floatClass = float === 'end' ? ' regex-automaton--float-end' : ''

  return (
    <figure class={`regex-automaton${floatClass}`} data-ra-machine={machine}>
      <svg
        class="ra-svg"
        viewBox={`0 0 ${spec.w} ${spec.h}`}
        role="img"
        aria-label={spec.label}
      >
        <defs>
          <marker
            id={marker}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path class="ra-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>
        {spec.edges.map((e, i) => {
          const from = byId.get(e.from)
          const to = byId.get(e.to)
          if (!from || !to) return null
          const lp = labelPoint(from, to, e.bend)
          return (
            <g key={`${e.from}-${e.to}-${i}`}>
              <path
                class="ra-edge"
                d={edgePath(from, to, e.bend)}
                marker-end={`url(#${marker})`}
              />
              <text class="ra-edge-label" x={lp.x} y={lp.y} text-anchor="middle">
                {e.label}
              </text>
            </g>
          )
        })}
        {spec.states.map(s => (
          <g key={s.id}>
            {s.start ? (
              <line class="ra-start" x1={s.x - 34} y1={s.y} x2={s.x - 22} y2={s.y} />
            ) : null}
            <circle
              class={s.lit ? 'ra-state ra-state--lit' : 'ra-state'}
              cx={s.x}
              cy={s.y}
              r={18}
            />
            {s.accept ? <circle class="ra-accept" cx={s.x} cy={s.y} r={14} /> : null}
            <text class="ra-state-label" x={s.x} y={s.y + 4} text-anchor="middle">
              {s.id}
            </text>
          </g>
        ))}
      </svg>
      {caption ? (
        <figcaption class="ra-caption">
          <MathText text={caption} mathClass="ra-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const RegexAutomatonComponent = RegexAutomatonImpl as QuartzMdxComponent<Props>
RegexAutomatonComponent.css = style

export const RegexAutomaton = registerMdxComponent('RegexAutomaton', RegexAutomatonComponent)

export default (() => RegexAutomaton) satisfies (opts: undefined) => QuartzMdxComponent<Props>
