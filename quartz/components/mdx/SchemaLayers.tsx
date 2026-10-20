import { MathText } from '../../util/math-text'
import style from '../styles/schemaLayers.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

let instance = 0

const SchemaLayersImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `scl-arrow-${++instance}`
  const views = [
    { x: 70, label: 'View 1' },
    { x: 250, label: 'View 2' },
    { x: 430, label: 'View 3' },
  ]

  return (
    <figure class="schema-layers">
      <svg
        class="scl-svg"
        viewBox="0 0 560 320"
        role="img"
        aria-label="Three-schema architecture: external views over a conceptual schema, physical schema, and storage."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="5"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path class="scl-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        {views.map(v => (
          <g key={v.label}>
            <rect class="scl-box scl-box--view" x={v.x} y={24} width={100} height={40} />
            <text class="scl-label" x={v.x + 50} y={48} text-anchor="middle">
              {v.label}
            </text>
            <line
              class="scl-edge"
              x1={v.x + 50}
              y1={64}
              x2={280}
              y2={100}
              marker-end={`url(#${arrow})`}
            />
          </g>
        ))}

        <rect class="scl-box scl-box--concept" x={160} y={100} width={240} height={44} />
        <text class="scl-label" x={280} y={127} text-anchor="middle">
          Conceptual Schema
        </text>

        <line
          class="scl-edge"
          x1={280}
          y1={144}
          x2={280}
          y2={176}
          marker-end={`url(#${arrow})`}
        />

        <rect class="scl-box scl-box--phys" x={180} y={176} width={200} height={44} />
        <text class="scl-label" x={280} y={203} text-anchor="middle">
          Physical Schema
        </text>

        <line
          class="scl-edge"
          x1={280}
          y1={220}
          x2={280}
          y2={248}
          marker-end={`url(#${arrow})`}
        />

        <ellipse class="scl-disk" cx={280} cy={268} rx={54} ry={14} />
        <rect class="scl-disk-body" x={226} y={268} width={108} height={28} />
        <ellipse class="scl-disk" cx={280} cy={296} rx={54} ry={14} />
        <text class="scl-sub" x={280} y={290} text-anchor="middle">
          storage
        </text>
      </svg>
      {caption ? (
        <figcaption class="scl-caption">
          <MathText text={caption} mathClass="scl-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const SchemaLayersComponent = SchemaLayersImpl as QuartzMdxComponent<Props>
SchemaLayersComponent.css = style

export const SchemaLayers = registerMdxComponent('SchemaLayers', SchemaLayersComponent)

export default (() => SchemaLayers) satisfies (opts: undefined) => QuartzMdxComponent<Props>
