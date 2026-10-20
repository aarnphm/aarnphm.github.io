import { MathText } from '../../util/math-text'
import style from '../styles/cfgDiagram.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Pair = 'branch' | 'loop'
type Props = { caption?: string; pair?: Pair }

type Kind = 'if-then' | 'if-else' | 'while' | 'seq'

const PAIRS: Record<Pair, Kind[]> = { branch: ['if-then', 'if-else'], loop: ['while', 'seq'] }

const CODE: Record<Kind, string[]> = {
  'if-then': ['if COND then S1;'],
  'if-else': ['if COND then', '  S1;', 'else', '  S2;'],
  while: ['while COND loop', '  S1;', 'end loop;'],
  seq: ['S1;', 'S2;'],
}

const Flow = ({ kind }: { kind: Kind }) => {
  if (kind === 'seq') {
    return (
      <svg class="cfg-svg" viewBox="0 0 120 160" role="img" aria-label="Sequence CFG G1 then G2">
        <circle class="cfg-end" cx={60} cy={12} r={4} />
        <line class="cfg-edge" x1={60} y1={16} x2={60} y2={36} />
        <rect class="cfg-block cfg-block--lit" x={30} y={36} width={60} height={28} />
        <text class="cfg-label" x={60} y={54} text-anchor="middle">
          G1
        </text>
        <line class="cfg-edge" x1={60} y1={64} x2={60} y2={84} />
        <rect class="cfg-block" x={30} y={84} width={60} height={28} />
        <text class="cfg-label" x={60} y={102} text-anchor="middle">
          G2
        </text>
        <line class="cfg-edge" x1={60} y1={112} x2={60} y2={140} />
        <circle class="cfg-end" cx={60} cy={144} r={4} />
      </svg>
    )
  }

  if (kind === 'while') {
    return (
      <svg class="cfg-svg" viewBox="0 0 160 180" role="img" aria-label="While-loop CFG">
        <circle class="cfg-end" cx={80} cy={12} r={4} />
        <line class="cfg-edge" x1={80} y1={16} x2={80} y2={40} />
        <polygon class="cfg-diamond cfg-diamond--lit" points="80,40 110,70 80,100 50,70" />
        <text class="cfg-label" x={80} y={74} text-anchor="middle">
          COND
        </text>
        <line class="cfg-edge" x1={80} y1={100} x2={80} y2={118} />
        <rect class="cfg-block" x={50} y={118} width={60} height={28} />
        <text class="cfg-label" x={80} y={136} text-anchor="middle">
          G1
        </text>
        <path class="cfg-edge" d="M 50 132 H 28 V 70 H 50" />
        <path class="cfg-edge" d="M 110 70 H 140 V 165 H 80" />
        <circle class="cfg-end" cx={80} cy={168} r={4} />
      </svg>
    )
  }

  if (kind === 'if-then') {
    return (
      <svg class="cfg-svg" viewBox="0 0 160 170" role="img" aria-label="If-then CFG">
        <circle class="cfg-end" cx={80} cy={12} r={4} />
        <line class="cfg-edge" x1={80} y1={16} x2={80} y2={40} />
        <polygon class="cfg-diamond cfg-diamond--lit" points="80,40 110,70 80,100 50,70" />
        <text class="cfg-label" x={80} y={74} text-anchor="middle">
          COND
        </text>
        <path class="cfg-edge" d="M 50 70 H 28 V 118 H 50" />
        <rect class="cfg-block" x={50} y={118} width={60} height={28} />
        <text class="cfg-label" x={80} y={136} text-anchor="middle">
          G1
        </text>
        <path class="cfg-edge" d="M 80 146 V 152" />
        <path class="cfg-edge" d="M 110 70 H 140 V 152 H 80" />
        <circle class="cfg-end" cx={80} cy={156} r={4} />
      </svg>
    )
  }

  return (
    <svg class="cfg-svg" viewBox="0 0 180 180" role="img" aria-label="If-then-else CFG">
      <circle class="cfg-end" cx={90} cy={12} r={4} />
      <line class="cfg-edge" x1={90} y1={16} x2={90} y2={40} />
      <polygon class="cfg-diamond cfg-diamond--lit" points="90,40 120,70 90,100 60,70" />
      <text class="cfg-label" x={90} y={74} text-anchor="middle">
        COND
      </text>
      <path class="cfg-edge" d="M 60 70 H 40 V 118" />
      <rect class="cfg-block" x={10} y={118} width={60} height={28} />
      <text class="cfg-label" x={40} y={136} text-anchor="middle">
        G1
      </text>
      <path class="cfg-edge" d="M 120 70 H 140 V 118" />
      <rect class="cfg-block" x={110} y={118} width={60} height={28} />
      <text class="cfg-label" x={140} y={136} text-anchor="middle">
        G2
      </text>
      <path class="cfg-edge" d="M 40 146 V 160 H 140 V 146" />
      <line class="cfg-edge" x1={90} y1={160} x2={90} y2={168} />
      <circle class="cfg-end" cx={90} cy={172} r={4} />
    </svg>
  )
}

const CfgDiagramImpl: QuartzMdxComponent<Props> = ({ caption, pair = 'branch' }) => {
  const kinds = PAIRS[pair] ?? PAIRS.branch
  return (
    <figure class="cfg-diagram" data-cfg-pair={pair}>
      <div class="cfg-row">
        {kinds.map(kind => (
          <div class="cfg-col" key={kind}>
            <pre class="cfg-code">
              {CODE[kind].map(line => (
                <code key={line}>{line}</code>
              ))}
            </pre>
            <Flow kind={kind} />
          </div>
        ))}
      </div>
      {caption ? (
        <figcaption class="cfg-caption">
          <MathText text={caption} mathClass="cfg-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CfgDiagramComponent = CfgDiagramImpl as QuartzMdxComponent<Props>
CfgDiagramComponent.css = style

export const CfgDiagram = registerMdxComponent('CfgDiagram', CfgDiagramComponent)

export default (() => CfgDiagram) satisfies (opts: undefined) => QuartzMdxComponent<Props>
