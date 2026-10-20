import { MathText } from '../../util/math-text'
import style from '../styles/weakEntity.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

let instance = 0

const WeakEntityImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `we-arrow-${++instance}`

  return (
    <figure class="weak-entity">
      <svg
        class="we-svg"
        viewBox="0 0 640 260"
        role="img"
        aria-label="Weak entity Players with identifying relationship Plays-on to strong entity Teams."
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
            <path class="we-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <ellipse class="we-attr" cx={90} cy={48} rx={42} ry={20} />
        <text class="we-label" x={90} y={52} text-anchor="middle">
          name
        </text>
        <ellipse class="we-attr we-attr--key" cx={190} cy={48} rx={48} ry={20} />
        <text class="we-label we-label--key" x={190} y={52} text-anchor="middle">
          number
        </text>
        <line class="we-edge" x1={120} y1={120} x2={90} y2={68} />
        <line class="we-edge" x1={160} y1={120} x2={190} y2={68} />

        <rect class="we-entity we-entity--weak-outer" x={70} y={110} width={140} height={48} />
        <rect class="we-entity we-entity--weak" x={76} y={116} width={128} height={36} />
        <text class="we-label" x={140} y={140} text-anchor="middle">
          Players
        </text>

        <line class="we-edge" x1={210} y1={134} x2={280} y2={134} />
        <polygon
          class="we-rel we-rel--outer"
          points="340,100 390,134 340,168 290,134"
        />
        <polygon class="we-rel" points="340,108 382,134 340,160 298,134" />
        <text class="we-label" x={340} y={138} text-anchor="middle">
          Plays-on
        </text>
        <line
          class="we-edge"
          x1={390}
          y1={134}
          x2={470}
          y2={134}
          marker-end={`url(#${arrow})`}
        />

        <ellipse class="we-attr we-attr--key" cx={540} cy={48} rx={42} ry={20} />
        <text class="we-label we-label--key" x={540} y={52} text-anchor="middle">
          name
        </text>
        <line class="we-edge" x1={540} y1={120} x2={540} y2={68} />
        <rect class="we-entity" x={470} y={116} width={140} height={36} />
        <text class="we-label" x={540} y={140} text-anchor="middle">
          Teams
        </text>

        <text class="we-note" x={400} y={210}>
          each player needs a team for the key
        </text>
        <line class="we-edge we-edge--soft" x1={430} y1={196} x2={450} y2={140} />
      </svg>
      {caption ? (
        <figcaption class="we-caption">
          <MathText text={caption} mathClass="we-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const WeakEntityComponent = WeakEntityImpl as QuartzMdxComponent<Props>
WeakEntityComponent.css = style

export const WeakEntity = registerMdxComponent('WeakEntity', WeakEntityComponent)

export default (() => WeakEntity) satisfies (opts: undefined) => QuartzMdxComponent<Props>
