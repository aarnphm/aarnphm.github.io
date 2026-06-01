import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
import style from '../styles/residualStream.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; markerId?: string }

function renderMath(tex: string, display: boolean): string {
  return katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })
}

function MathLabel({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = renderMath(tex, display)
  const cls = `rs-math ${display ? 'rs-math--display' : 'rs-math--inline'}`
  const Tag = display ? 'div' : 'span'
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: html }} />
}

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  tex: string
  cls?: string
  align?: 'center' | 'start'
}> = ({ x, y, w, h, tex, cls, align = 'center' }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`rs-fo rs-fo--${align} ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex, false) }}
    />
  </foreignObject>
)

const ResidualStreamImpl: QuartzMdxComponent<Props> = ({ caption, markerId }) => {
  const arrowId = markerId ?? 'rs-arrow'

  return (
    <figure class="residual-stream" data-residual-stream>
      <div class="rs-frame">
        <svg
          class="rs-graph"
          viewBox="0 0 240 700"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Transformer residual stream: tokens to embed, attention heads added at a sum node, MLP added at a sum node, then unembed to logits"
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
              <path class="rs-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <rect class="rs-box rs-box-neutral" x="145" y="20" width="80" height="32" rx="3" />
          <MathFO x={145} y={20} w={80} h={32} tex="\text{logits}" />
          <line class="rs-line" x1="185" y1="84" x2="185" y2="54" marker-end={`url(#${arrowId})`} />
          <rect class="rs-box rs-box-accent" x="145" y="84" width="80" height="32" rx="3" />
          <MathFO x={145} y={84} w={80} h={32} tex="\text{unembed}" />
          <line
            class="rs-line rs-line--dotted"
            x1="185"
            y1="188"
            x2="185"
            y2="118"
            marker-end={`url(#${arrowId})`}
          />
          <MathFO x={200} y={138} w={40} h={20} tex="x_{-1}" align="start" cls="rs-fo--axis" />

          <line class="rs-line" x1="185" y1="207" x2="185" y2="370" />
          <path
            class="rs-line"
            d="M 185 305 L 115 305 Q 110 305 110 300 L 110 205 Q 110 200 115 200 L 178 200"
            marker-end={`url(#${arrowId})`}
          />
          <rect class="rs-box rs-box-accent" x="50" y="255" width="120" height="30" rx="3" />
          <MathFO x={50} y={255} w={120} h={30} tex="\text{MLP}\, m" cls="rs-fo--mlp" />

          <circle class="rs-node" cx="185" cy="200" r="7" />
          <line class="rs-node-glyph" x1="180" y1="200" x2="190" y2="200" />
          <line class="rs-node-glyph" x1="185" y1="195" x2="185" y2="205" />
          <MathFO x={200} y={190} w={40} h={20} tex="x_{i+2}" align="start" cls="rs-fo--axis" />

          <line class="rs-line" x1="185" y1="387" x2="185" y2="540" />
          <path class="rs-line" d="M 185 510 L 115 510 Q 110 510 110 505 L 110 478" />
          <path
            class="rs-line"
            d="M 110 448 L 110 385 Q 110 380 115 380 L 178 380"
            marker-end={`url(#${arrowId})`}
          />
          <path
            class="rs-line"
            d="M 78 474 L 78 476 Q 78 478 82 478 L 138 478 Q 142 478 142 476 L 142 474"
          />
          <path
            class="rs-line"
            d="M 78 452 L 78 450 Q 78 448 82 448 L 138 448 Q 142 448 142 450 L 142 452"
          />
          <line class="rs-line" x1="110" y1="448" x2="110" y2="452" />
          <line class="rs-line" x1="110" y1="478" x2="110" y2="474" />
          <rect class="rs-box rs-box-accent" x="64" y="452" width="28" height="22" rx="3" />
          <MathFO x={64} y={452} w={28} h={22} tex="h_0" cls="rs-fo--sm" />
          <rect class="rs-box rs-box-accent" x="96" y="452" width="28" height="22" rx="3" />
          <MathFO x={96} y={452} w={28} h={22} tex="h_1" cls="rs-fo--sm" />
          <rect class="rs-box rs-box-accent" x="128" y="452" width="28" height="22" rx="3" />
          <MathFO x={128} y={452} w={28} h={22} tex="\cdots" cls="rs-fo--sm" />

          <circle class="rs-node" cx="185" cy="380" r="7" />
          <line class="rs-node-glyph" x1="180" y1="380" x2="190" y2="380" />
          <line class="rs-node-glyph" x1="185" y1="375" x2="185" y2="385" />
          <MathFO x={200} y={370} w={40} h={20} tex="x_{i+1}" align="start" cls="rs-fo--axis" />

          <MathFO x={200} y={530} w={40} h={20} tex="x_i" align="start" cls="rs-fo--axis" />
          <line
            class="rs-line rs-line--dotted"
            x1="185"
            y1="595"
            x2="185"
            y2="541"
            marker-end={`url(#${arrowId})`}
          />
          <MathFO x={200} y={580} w={40} h={20} tex="x_0" align="start" cls="rs-fo--axis" />
          <rect class="rs-box rs-box-accent" x="145" y="600" width="80" height="32" rx="3" />
          <MathFO x={145} y={600} w={80} h={32} tex="\text{embed}" />
          <line
            class="rs-line"
            x1="185"
            y1="668"
            x2="185"
            y2="634"
            marker-end={`url(#${arrowId})`}
          />
          <rect class="rs-box rs-box-neutral" x="145" y="668" width="80" height="32" rx="3" />
          <MathFO x={145} y={668} w={80} h={32} tex="\text{tokens}" />
        </svg>

        <div class="rs-divider" aria-hidden="true" />

        <div class="rs-note rs-note--unembed">
          <p>The final logits are produced by applying the unembedding.</p>
          <MathLabel tex="T(t) = W_U x_{-1}" display />
        </div>

        <div class="rs-note rs-note--mlp">
          <p>
            An MLP layer, <MathLabel tex="m" />, is run and added to the residual stream.
          </p>
          <MathLabel tex="x_{i+2} = x_{i+1} + m(x_{i+1})" display />
        </div>

        <div class="rs-note rs-note--attn">
          <p>
            Each attention head, <MathLabel tex="h" />, is run and added to the residual stream.
          </p>
          <MathLabel tex="x_{i+1} = x_i + \sum_{h \in H_i} h(x_i)" display />
        </div>

        <div class="rs-note rs-note--embed">
          <p>Token embedding.</p>
          <MathLabel tex="x_0 = W_E t" display />
        </div>

        <aside class="rs-bracket" aria-label="One residual block">
          <span class="rs-bracket-line" aria-hidden="true" />
          <span class="rs-bracket-label">
            One
            <br />
            residual
            <br />
            block
          </span>
        </aside>
      </div>
      {caption ? (
        <figcaption class="rs-caption">
          <MathText text={caption} mathClass="rs-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ResidualStreamComponent = ResidualStreamImpl as QuartzMdxComponent<Props>
ResidualStreamComponent.css = style

export const ResidualStream = registerMdxComponent('ResidualStream', ResidualStreamComponent)

export default (() => ResidualStream) satisfies (opts: undefined) => QuartzMdxComponent<Props>
