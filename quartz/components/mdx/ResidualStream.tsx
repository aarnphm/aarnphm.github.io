import katex from 'katex'
import { customMacros, katexOptions } from '../../cfg'
import style from '../styles/residualStream.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; markerId?: string }

function renderMath(tex: string, display: boolean): string {
  return katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'htmlAndMathml',
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
          <text class="rs-box-text" x="185" y="36">
            logits
          </text>
          <line class="rs-line" x1="185" y1="84" x2="185" y2="54" marker-end={`url(#${arrowId})`} />
          <rect class="rs-box rs-box-accent" x="145" y="84" width="80" height="32" rx="3" />
          <text class="rs-box-text" x="185" y="100">
            unembed
          </text>
          <line
            class="rs-line rs-line--dotted"
            x1="185"
            y1="188"
            x2="185"
            y2="118"
            marker-end={`url(#${arrowId})`}
          />
          <text class="rs-axis-label" x="200" y="148">
            x
            <tspan baseline-shift="sub" font-size="8">
              −1
            </tspan>
          </text>

          <line class="rs-line" x1="185" y1="207" x2="185" y2="370" />
          <path
            class="rs-line"
            d="M 185 305 L 115 305 Q 110 305 110 300 L 110 215 Q 110 210 115 210 L 178 210"
            marker-end={`url(#${arrowId})`}
          />
          <rect class="rs-box rs-box-accent" x="50" y="255" width="120" height="30" rx="3" />
          <text class="rs-box-text" x="110" y="270">
            <tspan>MLP</tspan>
            <tspan class="rs-italic" dx="6">
              m
            </tspan>
          </text>

          <circle class="rs-node" cx="185" cy="200" r="7" />
          <line class="rs-node-glyph" x1="180" y1="200" x2="190" y2="200" />
          <line class="rs-node-glyph" x1="185" y1="195" x2="185" y2="205" />
          <text class="rs-axis-label" x="200" y="200">
            x
            <tspan baseline-shift="sub" font-size="8">
              i+2
            </tspan>
          </text>

          <line class="rs-line" x1="185" y1="387" x2="185" y2="540" />
          <path class="rs-line" d="M 185 510 L 115 510 Q 110 510 110 505 L 110 478" />
          <path
            class="rs-line"
            d="M 110 448 L 110 395 Q 110 390 115 390 L 178 390"
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
          <text class="rs-box-text rs-box-text--sm" x="78" y="463">
            <tspan class="rs-italic">h</tspan>
            <tspan baseline-shift="sub" font-size="7">
              0
            </tspan>
          </text>
          <rect class="rs-box rs-box-accent" x="96" y="452" width="28" height="22" rx="3" />
          <text class="rs-box-text rs-box-text--sm" x="110" y="463">
            <tspan class="rs-italic">h</tspan>
            <tspan baseline-shift="sub" font-size="7">
              1
            </tspan>
          </text>
          <rect class="rs-box rs-box-accent" x="128" y="452" width="28" height="22" rx="3" />
          <text class="rs-box-text rs-box-text--sm" x="142" y="463">
            …
          </text>

          <circle class="rs-node" cx="185" cy="380" r="7" />
          <line class="rs-node-glyph" x1="180" y1="380" x2="190" y2="380" />
          <line class="rs-node-glyph" x1="185" y1="375" x2="185" y2="385" />
          <text class="rs-axis-label" x="200" y="380">
            x
            <tspan baseline-shift="sub" font-size="8">
              i+1
            </tspan>
          </text>

          <text class="rs-axis-label" x="200" y="540">
            x
            <tspan baseline-shift="sub" font-size="8">
              i
            </tspan>
          </text>
          <line
            class="rs-line rs-line--dotted"
            x1="185"
            y1="595"
            x2="185"
            y2="555"
            marker-end={`url(#${arrowId})`}
          />
          <text class="rs-axis-label" x="200" y="590">
            x
            <tspan baseline-shift="sub" font-size="8">
              0
            </tspan>
          </text>
          <rect class="rs-box rs-box-accent" x="145" y="600" width="80" height="32" rx="3" />
          <text class="rs-box-text" x="185" y="616">
            embed
          </text>
          <line
            class="rs-line"
            x1="185"
            y1="668"
            x2="185"
            y2="634"
            marker-end={`url(#${arrowId})`}
          />
          <rect class="rs-box rs-box-neutral" x="145" y="668" width="80" height="32" rx="3" />
          <text class="rs-box-text" x="185" y="684">
            tokens
          </text>
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
      {caption ? <figcaption class="rs-caption">{caption}</figcaption> : null}
    </figure>
  )
}

const ResidualStreamComponent = ResidualStreamImpl as QuartzMdxComponent<Props>
ResidualStreamComponent.css = style

export const ResidualStream = registerMdxComponent('ResidualStream', ResidualStreamComponent)

export default (() => ResidualStream) satisfies (opts: undefined) => QuartzMdxComponent<Props>
