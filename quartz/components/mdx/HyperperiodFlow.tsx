import { MathText } from '../../util/math-text'
import style from '../styles/hyperperiodFlow.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const NODES = {
  source: { x: 60, y: 120, label: 'Source' },
  Ji: { x: 200, y: 60, label: 'Jᵢ' },
  Jk: { x: 200, y: 180, label: 'Jₖ' },
  x: { x: 360, y: 40, label: 'x' },
  y: { x: 360, y: 120, label: 'y' },
  z: { x: 360, y: 200, label: 'z' },
  sink: { x: 520, y: 120, label: 'Sink' },
} as const

const EDGES: { from: keyof typeof NODES; to: keyof typeof NODES; label: string }[] = [
  { from: 'source', to: 'Ji', label: 'Cᵢ' },
  { from: 'source', to: 'Jk', label: 'Cₖ' },
  { from: 'Ji', to: 'x', label: 'f' },
  { from: 'Ji', to: 'y', label: 'f' },
  { from: 'Jk', to: 'y', label: 'f' },
  { from: 'Jk', to: 'z', label: 'f' },
  { from: 'x', to: 'sink', label: 'f' },
  { from: 'y', to: 'sink', label: 'f' },
  { from: 'z', to: 'sink', label: 'f' },
]

let instance = 0

const HyperperiodFlowImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `hpf-arrow-${++instance}`

  return (
    <figure class="hyperperiod-flow">
      <svg
        class="hpf-svg"
        viewBox="0 0 580 260"
        role="img"
        aria-label="Flow network from source through jobs and frames to sink for hyperperiod scheduling."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path class="hpf-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <text class="hpf-col" x={200} y={24} text-anchor="middle">
          Jobs
        </text>
        <text class="hpf-col" x={360} y={24} text-anchor="middle">
          Frames
        </text>

        {EDGES.map(e => {
          const a = NODES[e.from]
          const b = NODES[e.to]
          const mx = (a.x + b.x) / 2
          const my = (a.y + b.y) / 2 - 8
          return (
            <g key={`${e.from}-${e.to}`}>
              <line
                class="hpf-edge"
                x1={a.x + 14}
                y1={a.y}
                x2={b.x - 14}
                y2={b.y}
                marker-end={`url(#${arrow})`}
              />
              <text class="hpf-edge-label" x={mx} y={my} text-anchor="middle">
                {e.label}
              </text>
            </g>
          )
        })}

        {Object.entries(NODES).map(([key, n]) => (
          <g key={key}>
            <circle
              class={key === 'source' || key === 'sink' ? 'hpf-node hpf-node--end' : 'hpf-node'}
              cx={n.x}
              cy={n.y}
              r={14}
            />
            <text class="hpf-node-label" x={n.x} y={n.y - 22} text-anchor="middle">
              {n.label}
            </text>
          </g>
        ))}
      </svg>
      {caption ? (
        <figcaption class="hpf-caption">
          <MathText text={caption} mathClass="hpf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const HyperperiodFlowComponent = HyperperiodFlowImpl as QuartzMdxComponent<Props>
HyperperiodFlowComponent.css = style

export const HyperperiodFlow = registerMdxComponent('HyperperiodFlow', HyperperiodFlowComponent)

export default (() => HyperperiodFlow) satisfies (opts: undefined) => QuartzMdxComponent<Props>
