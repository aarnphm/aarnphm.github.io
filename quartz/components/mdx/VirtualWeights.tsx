import { type FunctionalComponent } from 'preact'
import style from '../styles/virtualWeights.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Emphasis = 'normal' | 'highlight' | 'dots-only'

type LayerProps = { k: number; topY: number; arrowId: string; emphasis: Emphasis }

const LAYER_HEIGHT = 80
const LAYER_TOPS = [40, 240, 440]
const CORNER = 5

const SuperSubW: FunctionalComponent<{ sup: string | number; sub: string }> = ({ sup, sub }) => (
  <>
    <tspan class="vw-italic">W</tspan>
    <tspan baseline-shift="super" font-size="7" dx="-1">
      {sup}
    </tspan>
    <tspan baseline-shift="sub" font-size="7" dx="-3">
      {sub}
    </tspan>
  </>
)

const Layer: FunctionalComponent<LayerProps> = ({ k, topY, arrowId, emphasis }) => {
  const midY = topY + LAYER_HEIGHT / 2
  const botY = topY + LAYER_HEIGHT

  const frameClass = emphasis === 'dots-only' ? 'vw-frame vw-frame--faded' : 'vw-frame'
  const dotsHighlight = emphasis === 'highlight' || emphasis === 'dots-only'
  const dotsClass = `vw-box vw-box-dots${dotsHighlight ? ' vw-box-dots--highlight' : ''}`

  const bottomPath = `M 95 ${botY} L ${130 - CORNER} ${botY} Q 130 ${botY} 130 ${botY - CORNER} L 130 ${midY + 12}`
  const topPath = `M 130 ${midY - 12} L 130 ${topY + CORNER} Q 130 ${topY} ${130 - CORNER} ${topY} L 95 ${topY}`

  return (
    <g class="vw-layer" data-layer={k}>
      <g class={frameClass}>
        <line class="vw-line" x1="30" y1={botY} x2="55" y2={botY} marker-end={`url(#${arrowId})`} />
        <path class="vw-line" d={bottomPath} marker-end={`url(#${arrowId})`} />
        <path class="vw-line" d={topPath} marker-end={`url(#${arrowId})`} />
        <line class="vw-line" x1="55" y1={topY} x2="37" y2={topY} marker-end={`url(#${arrowId})`} />

        <rect class="vw-box vw-box-weight" x="55" y={topY - 12} width="40" height="24" rx="3" />
        <text class="vw-box-text" x="75" y={topY}>
          <SuperSubW sup={k} sub="O" />
        </text>

        <rect class="vw-box vw-box-weight" x="55" y={botY - 12} width="40" height="24" rx="3" />
        <text class="vw-box-text" x="75" y={botY}>
          <SuperSubW sup={k} sub="I" />
        </text>

        <circle class="vw-node" cx="30" cy={topY} r="7" />
        <text class="vw-node-text" x="30" y={topY}>
          +
        </text>
      </g>

      <rect class={dotsClass} x="130" y={midY - 12} width="40" height="24" rx="3" />
      <text class="vw-box-text vw-box-text--sm" x="150" y={midY}>
        …
      </text>
    </g>
  )
}

const VirtualLabel: FunctionalComponent<{
  x: number
  y: number
  sup1: string | number
  sub1: string
  sup2: string | number
  sub2: string
}> = ({ x, y, sup1, sub1, sup2, sub2 }) => (
  <g transform={`translate(${x}, ${y})`}>
    <rect class="vw-vw-bg" x="-30" y="-13" width="60" height="26" rx="3" />
    <text class="vw-vw-text" x="0" y="0">
      <tspan class="vw-italic">W</tspan>
      <tspan baseline-shift="super" font-size="8" dx="-1">
        {sup1}
      </tspan>
      <tspan baseline-shift="sub" font-size="8" dx="-3">
        {sub1}
      </tspan>
      <tspan class="vw-italic" dx="2">
        W
      </tspan>
      <tspan baseline-shift="super" font-size="8" dx="-1">
        {sup2}
      </tspan>
      <tspan baseline-shift="sub" font-size="8" dx="-3">
        {sub2}
      </tspan>
    </text>
  </g>
)

type PanelKind = 'reading' | 'virtual'

const Panel: FunctionalComponent<{ kind: PanelKind; arrowId: string }> = ({ kind, arrowId }) => {
  const isVirtual = kind === 'virtual'
  const layers = LAYER_TOPS.map((topY, idx) => ({
    k: 3 - idx,
    topY,
    emphasis: isVirtual ? ('dots-only' as Emphasis) : idx === 1 ? 'highlight' : 'normal',
  }))

  return (
    <svg
      class={`vw-graph vw-graph--${kind}`}
      viewBox="0 0 320 580"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={
        kind === 'reading'
          ? 'Three transformer layers reading from and writing to a shared residual stream'
          : 'Faded layers with dashed virtual-weight curves spanning across layers'
      }
    >
      <defs>
        <marker
          id={arrowId}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path class="vw-arrowhead" d="M0,0 L10,5 L0,10 z" />
        </marker>
        {isVirtual ? (
          <marker
            id={`${arrowId}-curve`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path class="vw-curve-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        ) : null}
      </defs>

      <line class="vw-rs-line" x1="30" y1="540" x2="30" y2="20" />
      <line class="vw-rs-line" x1="30" y1="32" x2="30" y2="20" marker-end={`url(#${arrowId})`} />
      <line class="vw-rs-line" x1="30" y1="560" x2="30" y2="548" marker-end={`url(#${arrowId})`} />

      {layers.map(({ k, topY, emphasis }) => (
        <Layer k={k} topY={topY} arrowId={arrowId} emphasis={emphasis} />
      ))}

      {isVirtual ? (
        <g class="vw-virtual">
          <path
            class="vw-curve"
            d="M 170 480 Q 290 280 175 90"
            marker-end={`url(#${arrowId}-curve)`}
          />
          <path
            class="vw-curve"
            d="M 170 280 Q 230 180 175 90"
            marker-end={`url(#${arrowId}-curve)`}
          />
          <VirtualLabel x={205} y={170} sup1={3} sub1="I" sup2={2} sub2="O" />
          <VirtualLabel x={265} y={285} sup1={3} sub1="I" sup2={1} sub2="O" />
        </g>
      ) : null}
    </svg>
  )
}

const VirtualWeightsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="virtual-weights" data-virtual-weights>
    <div class="vw-panels">
      <section class="vw-panel">
        <p class="vw-panel-title">
          The residual stream is modified by a sequence of MLP and attention layers “reading from”
          and “writing to” it with linear operations.
        </p>
        <div class="vw-stage">
          <Panel kind="reading" arrowId="vw-arrow-r" />
          <div class="vw-callouts vw-callouts--reading">
            <p class="vw-callout">
              Each layer <strong>“writes”</strong> to the residual stream by adding a linear
              projection of its results.
            </p>
            <p class="vw-callout">
              Each layer <strong>“reads”</strong> from the residual stream with a linear projection.
            </p>
          </div>
        </div>
      </section>

      <section class="vw-panel">
        <p class="vw-panel-title">
          Because all these operations are linear, we can “multiply through” the residual stream.
        </p>
        <div class="vw-stage">
          <Panel kind="virtual" arrowId="vw-arrow-v" />
          <div class="vw-callouts vw-callouts--virtual">
            <p class="vw-callout">
              Multiplying out the weights reveals “virtual weights” implicitly connecting each pair
              of layers.
            </p>
            <p class="vw-callout">
              By using different subspaces of the residual stream, a layer can send different
              information to different layers, or even not interact with other layers.
            </p>
          </div>
        </div>
      </section>
    </div>
    {caption ? <figcaption class="vw-caption">{caption}</figcaption> : null}
  </figure>
)

const VirtualWeightsComponent = VirtualWeightsImpl as QuartzMdxComponent<Props>
VirtualWeightsComponent.css = style

export const VirtualWeights = registerMdxComponent('VirtualWeights', VirtualWeightsComponent)

export default (() => VirtualWeights) satisfies (opts: undefined) => QuartzMdxComponent<Props>
