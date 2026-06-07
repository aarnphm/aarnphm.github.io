import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/multi-head-attention.inline'
import style from '../styles/multiHeadAttention.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; heads?: number }

const HEAD_OPTIONS = [1, 2, 4, 8] as const
const DEFAULT_HEADS = 4
const DEFAULT_DM = 512
const SEQ_LEN = 4

const VIEW_W = 760
const VIEW_H = 560
const PAD_L = 18
const PAD_R = 18

const COL_X = { input: 60, proj: 200, score: 332, head: 472, concat: 612, wo: 700 } as const
const HEAD_BAND_TOP = 36
const HEAD_BAND_BOT = VIEW_H - 36
const HEAD_BAND_H = HEAD_BAND_BOT - HEAD_BAND_TOP

const PROJ_W = 52
const PROJ_H = 14
const PROJ_GAP = 4
const SCORE_BOX = 52
const HEAD_OUT_W = 56
const HEAD_OUT_H = 22

const SCORE_PATTERNS: number[][][] = [
  [
    [0.85, 0.06, 0.04, 0.05],
    [0.05, 0.78, 0.12, 0.05],
    [0.04, 0.1, 0.82, 0.04],
    [0.05, 0.04, 0.06, 0.85],
  ],
  [
    [0.05, 0.85, 0.05, 0.05],
    [0.04, 0.04, 0.88, 0.04],
    [0.04, 0.05, 0.04, 0.87],
    [0.83, 0.06, 0.06, 0.05],
  ],
  [
    [0.62, 0.18, 0.12, 0.08],
    [0.16, 0.5, 0.22, 0.12],
    [0.1, 0.2, 0.46, 0.24],
    [0.06, 0.12, 0.28, 0.54],
  ],
  [
    [0.25, 0.25, 0.25, 0.25],
    [0.2, 0.3, 0.3, 0.2],
    [0.18, 0.32, 0.32, 0.18],
    [0.3, 0.2, 0.2, 0.3],
  ],
  [
    [0.7, 0.2, 0.06, 0.04],
    [0.5, 0.3, 0.15, 0.05],
    [0.3, 0.35, 0.25, 0.1],
    [0.1, 0.2, 0.35, 0.35],
  ],
  [
    [0.1, 0.2, 0.3, 0.4],
    [0.15, 0.25, 0.35, 0.25],
    [0.25, 0.35, 0.25, 0.15],
    [0.4, 0.3, 0.2, 0.1],
  ],
  [
    [0.88, 0.04, 0.04, 0.04],
    [0.06, 0.05, 0.84, 0.05],
    [0.78, 0.08, 0.08, 0.06],
    [0.06, 0.82, 0.06, 0.06],
  ],
  [
    [0.4, 0.1, 0.4, 0.1],
    [0.1, 0.4, 0.1, 0.4],
    [0.42, 0.08, 0.42, 0.08],
    [0.08, 0.42, 0.08, 0.42],
  ],
]

const renderMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  tex: string
  cls?: string
}> = ({ x, y, w, h, tex, cls }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`mha-fo ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({ tex, display }) => (
  <span
    class={`mha-math${display ? ' mha-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }}
  />
)

const headBandY = (i: number, h: number): { top: number; mid: number; bot: number } => {
  const slot = HEAD_BAND_H / h
  const top = HEAD_BAND_TOP + slot * i
  return { top, mid: top + slot / 2, bot: top + slot }
}

const fmtBytes = (n: number): string => {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${n} B`
}

const fmtCompact = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

const MultiHeadAttentionImpl: QuartzMdxComponent<Props> = ({ caption, heads = DEFAULT_HEADS }) => {
  const initialH = HEAD_OPTIONS.includes(heads as 1 | 2 | 4 | 8) ? (heads as number) : DEFAULT_HEADS
  const headIndices = Array.from({ length: 8 }, (_, i) => i)
  const initialDh = DEFAULT_DM / initialH
  const initialParams = 4 * DEFAULT_DM * DEFAULT_DM
  const initialCachePerToken = 2 * DEFAULT_DM * 2
  const initialFlops = 4 * SEQ_LEN * SEQ_LEN * DEFAULT_DM

  return (
    <figure
      class="multi-head-attention"
      data-multi-head-attention
      data-mha-heads={String(initialH)}
      data-mha-dm={String(DEFAULT_DM)}
      data-mha-show-pattern="true"
      data-mha-options={HEAD_OPTIONS.join(',')}
      data-mha-seq={String(SEQ_LEN)}
    >
      <div class="mha-stage">
        <svg
          class="mha-graph"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Multi-head attention with ${initialH} heads: input x splits into per-head Q, K, V projections, each head runs an independent softmax over a per-head attention pattern, outputs are concatenated and projected by W_O, then added back to the residual stream.`}
          data-mha-canvas
        >
          <defs>
            <marker
              id="mha-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path class="mha-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
            <marker
              id="mha-arrow-residual"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path class="mha-arrowhead mha-arrowhead--residual" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <rect
            class="mha-residual"
            x={PAD_L}
            y={12}
            width={VIEW_W - PAD_L - PAD_R}
            height={20}
            rx={4}
          />
          <MathFO
            x={PAD_L + 8}
            y={13}
            w={120}
            h={18}
            tex="\text{residual stream}"
            cls="mha-fo--residual-label"
          />

          <g class="mha-input">
            <rect
              class="mha-box mha-box--input"
              x={COL_X.input - 36}
              y={VIEW_H / 2 - 22}
              width={72}
              height={44}
              rx={5}
            />
            <MathFO
              x={COL_X.input - 36}
              y={VIEW_H / 2 - 22}
              w={72}
              h={44}
              tex="x"
              cls="mha-fo--input"
            />
            <MathFO
              x={COL_X.input - 20}
              y={VIEW_H / 2 + 28}
              w={40}
              h={16}
              tex="d_m"
              cls="mha-fo--axis"
            />
          </g>

          <g data-mha-fanout>
            {headIndices.map(i => {
              const visible = i < initialH
              const { mid } = headBandY(i, initialH)
              return (
                <path
                  class="mha-fanout"
                  data-mha-fanout-line={String(i)}
                  data-mha-hidden={visible ? 'false' : 'true'}
                  d={`M ${COL_X.input + 36} ${VIEW_H / 2} C ${COL_X.input + 70} ${VIEW_H / 2}, ${COL_X.proj - 60} ${mid}, ${COL_X.proj - PROJ_W / 2 - 4} ${mid}`}
                />
              )
            })}
          </g>

          <g data-mha-heads>
            {headIndices.map(i => {
              const visible = i < initialH
              const { mid } = headBandY(i, initialH)
              const projTopY = mid - (PROJ_H * 3 + PROJ_GAP * 2) / 2
              const scoreY = mid - SCORE_BOX / 2
              const headOutY = mid - HEAD_OUT_H / 2
              const accent = i === 0
              return (
                <g
                  class={`mha-head${accent ? ' mha-head--accent' : ''}`}
                  data-mha-head={String(i)}
                  data-mha-hidden={visible ? 'false' : 'true'}
                >
                  <g data-mha-projs>
                    {['Q', 'K', 'V'].map((kind, k) => {
                      const py = projTopY + k * (PROJ_H + PROJ_GAP)
                      return (
                        <g>
                          <rect
                            class={`mha-proj mha-proj--${kind.toLowerCase()}`}
                            x={COL_X.proj - PROJ_W / 2}
                            y={py}
                            width={PROJ_W}
                            height={PROJ_H}
                            rx={2}
                          />
                          <MathFO
                            x={COL_X.proj - PROJ_W / 2}
                            y={py}
                            w={PROJ_W}
                            h={PROJ_H}
                            tex={`W_{${kind},${i + 1}}`}
                            cls="mha-fo--proj"
                          />
                        </g>
                      )
                    })}
                  </g>

                  <line
                    class="mha-link mha-link--proj-score"
                    x1={COL_X.proj + PROJ_W / 2}
                    y1={mid}
                    x2={COL_X.score - SCORE_BOX / 2}
                    y2={mid}
                    marker-end="url(#mha-arrow)"
                  />

                  <g
                    class="mha-score"
                    data-mha-score={String(i)}
                    data-mha-score-visible={visible ? 'true' : 'false'}
                  >
                    <rect
                      class="mha-score-frame"
                      x={COL_X.score - SCORE_BOX / 2}
                      y={scoreY}
                      width={SCORE_BOX}
                      height={SCORE_BOX}
                      rx={3}
                    />
                    {SCORE_PATTERNS[i].map((row, r) =>
                      row.map((v, c) => {
                        const cell = SCORE_BOX / SEQ_LEN
                        return (
                          <rect
                            class="mha-score-cell"
                            x={COL_X.score - SCORE_BOX / 2 + c * cell}
                            y={scoreY + r * cell}
                            width={cell}
                            height={cell}
                            style={`opacity:${0.12 + v * 0.78}`}
                          />
                        )
                      }),
                    )}
                  </g>

                  <g
                    class="mha-score-collapsed"
                    data-mha-score-collapsed={String(i)}
                    data-mha-collapsed-visible="false"
                  >
                    <rect
                      class="mha-box mha-box--collapsed"
                      x={COL_X.score - SCORE_BOX / 2}
                      y={mid - 11}
                      width={SCORE_BOX}
                      height={22}
                      rx={3}
                    />
                    <MathFO
                      x={COL_X.score - SCORE_BOX / 2}
                      y={mid - 11}
                      w={SCORE_BOX}
                      h={22}
                      tex={`\\text{head}_{${i + 1}}\\,\\text{attn}`}
                      cls="mha-fo--collapsed"
                    />
                  </g>

                  <line
                    class="mha-link mha-link--score-out"
                    x1={COL_X.score + SCORE_BOX / 2}
                    y1={mid}
                    x2={COL_X.head - HEAD_OUT_W / 2}
                    y2={mid}
                    marker-end="url(#mha-arrow)"
                  />

                  <rect
                    class={`mha-box mha-box--head${accent ? ' mha-box--accent' : ''}`}
                    data-mha-head-rect
                    x={COL_X.head - HEAD_OUT_W / 2}
                    y={headOutY}
                    width={HEAD_OUT_W}
                    height={HEAD_OUT_H}
                    rx={3}
                  />
                  <foreignObject
                    x={COL_X.head - HEAD_OUT_W / 2}
                    y={headOutY}
                    width={HEAD_OUT_W}
                    height={HEAD_OUT_H}
                    data-mha-head-label
                  >
                    <div
                      class="mha-fo mha-fo--head"
                      dangerouslySetInnerHTML={{ __html: renderMath(`\\text{head}_{${i + 1}}`) }}
                    />
                  </foreignObject>
                </g>
              )
            })}
          </g>

          <g data-mha-concat-lines>
            {headIndices.map(i => {
              const visible = i < initialH
              const { mid } = headBandY(i, initialH)
              return (
                <path
                  class="mha-concat-line"
                  data-mha-concat-line={String(i)}
                  data-mha-hidden={visible ? 'false' : 'true'}
                  d={`M ${COL_X.head + HEAD_OUT_W / 2} ${mid} C ${COL_X.head + 60} ${mid}, ${COL_X.concat - 30} ${VIEW_H / 2}, ${COL_X.concat - 22} ${VIEW_H / 2}`}
                />
              )
            })}
          </g>

          <g class="mha-concat">
            <rect
              class="mha-box mha-box--concat"
              x={COL_X.concat - 22}
              y={VIEW_H / 2 - 28}
              width={44}
              height={56}
              rx={4}
            />
            <MathFO
              x={COL_X.concat - 22}
              y={VIEW_H / 2 - 28}
              w={44}
              h={56}
              tex="\operatorname{concat}"
              cls="mha-fo--concat"
            />
          </g>

          <line
            class="mha-link mha-link--wo"
            x1={COL_X.concat + 22}
            y1={VIEW_H / 2}
            x2={COL_X.wo - 18}
            y2={VIEW_H / 2}
            marker-end="url(#mha-arrow)"
          />
          <g class="mha-wo">
            <rect
              class="mha-box mha-box--wo"
              x={COL_X.wo - 18}
              y={VIEW_H / 2 - 22}
              width={42}
              height={44}
              rx={5}
            />
            <MathFO
              x={COL_X.wo - 18}
              y={VIEW_H / 2 - 22}
              w={42}
              h={44}
              tex="W_O"
              cls="mha-fo--wo"
            />
          </g>

          <path
            class="mha-residual-arc"
            d={`M ${COL_X.wo + 3} ${VIEW_H / 2 - 22} C ${COL_X.wo + 40} ${VIEW_H / 2 - 60}, ${COL_X.wo + 3} 90, ${COL_X.wo + 3} 34`}
            marker-end="url(#mha-arrow-residual)"
          />
          <foreignObject x={COL_X.wo - 14} y={VIEW_H / 2 + 36} width={120} height={18}>
            <div
              class="mha-fo mha-fo--residual-label"
              dangerouslySetInnerHTML={{ __html: renderMath('+\\ x') }}
            />
          </foreignObject>
        </svg>

        <aside class="mha-side" aria-label="MHA dimensional readout">
          <dl class="mha-readout" data-mha-readout>
            <div class="mha-readout-row">
              <dt>per-head dim</dt>
              <dd>
                <MathLabel tex="d_h = d_m / h" />
                <span class="mha-readout-eq">=</span>
                <span data-mha-dh class="mha-readout-val mha-readout-val--accent">
                  {initialDh}
                </span>
              </dd>
            </div>
            <div class="mha-readout-row">
              <dt>params / layer</dt>
              <dd>
                <MathLabel tex="4 d_m^2" />
                <span class="mha-readout-eq">=</span>
                <span data-mha-params class="mha-readout-val">
                  {fmtCompact(initialParams)}
                </span>
              </dd>
            </div>
            <div class="mha-readout-row">
              <dt>KV cache / tok / layer</dt>
              <dd>
                <MathLabel tex="2 d_m \cdot \text{fp16}" />
                <span class="mha-readout-eq">=</span>
                <span data-mha-cache class="mha-readout-val">
                  {fmtBytes(initialCachePerToken)}
                </span>
              </dd>
            </div>
            <div class="mha-readout-row">
              <dt>
                FLOPs / layer{' '}
                <span class="mha-readout-eq">
                  <MathLabel tex="L=4" />
                </span>
              </dt>
              <dd>
                <MathLabel tex="\Theta(L^2 d_m)" />
                <span class="mha-readout-eq">=</span>
                <span data-mha-flops class="mha-readout-val">
                  {fmtCompact(initialFlops)}
                </span>
              </dd>
            </div>
          </dl>

          <div class="mha-invariant" role="status" data-mha-invariant>
            <span class="mha-invariant-bullet" aria-hidden="true" />
            <span>
              cache sum and param budget are <strong>flat in h</strong>; the slider only{' '}
              <em>rearranges</em> the <MathLabel tex="d_m" /> budget.
            </span>
          </div>

          <div class="mha-equation">
            <MathLabel
              tex="\text{MHA}(x) = \operatorname{concat}(\text{head}_1,\dots,\text{head}_{h})\,W_O"
              display
            />
            <MathLabel
              tex="\text{head}_i = \operatorname{softmax}\!\left(\tfrac{Q_i K_i^{\top}}{\sqrt{d_h}}\right) V_i"
              display
            />
          </div>
        </aside>
      </div>

      <div class="mha-controls" role="group" aria-label="MHA configuration">
        <div class="mha-control mha-control--slider">
          <label class="mha-label" for="mha-heads-slider">
            heads <MathLabel tex="h" />
          </label>
          <input
            id="mha-heads-slider"
            class="mha-slider"
            type="range"
            min="0"
            max={HEAD_OPTIONS.length - 1}
            step="1"
            value={HEAD_OPTIONS.indexOf(initialH as 1 | 2 | 4 | 8)}
            data-mha-heads-input
            aria-valuemin={HEAD_OPTIONS[0]}
            aria-valuemax={HEAD_OPTIONS[HEAD_OPTIONS.length - 1]}
            aria-valuenow={initialH}
            aria-valuetext={`${initialH} heads`}
          />
          <span
            class="mha-slider-value"
            data-mha-heads-value
            dangerouslySetInnerHTML={{ __html: renderMath(`h = ${initialH}`) }}
          />
          <div class="mha-tick-strip" data-mha-tick-strip>
            {HEAD_OPTIONS.map(h => (
              <span
                class="mha-tick"
                data-mha-tick={String(h)}
                data-mha-active={h === initialH ? 'true' : 'false'}
              >
                {h}
              </span>
            ))}
          </div>
        </div>

        <div class="mha-control mha-control--dm">
          <label class="mha-label" for="mha-dm-input">
            model dim <MathLabel tex="d_m" />
          </label>
          <input
            id="mha-dm-input"
            class="mha-num"
            type="number"
            min="64"
            max="16384"
            step="64"
            value={DEFAULT_DM}
            data-mha-dm-input
            inputmode="numeric"
          />
        </div>

        <div class="mha-control mha-control--toggle">
          <button
            type="button"
            class="mha-toggle is-active"
            data-mha-pattern-toggle
            aria-pressed="true"
          >
            <span class="mha-toggle-dot" aria-hidden="true" />
            <span class="mha-toggle-label">show per-head attention pattern</span>
          </button>
        </div>
      </div>

      <p class="mha-intuition">
        Each head's softmax is an <strong>independent normaliser</strong>. <MathLabel tex="h" />{' '}
        heads are strictly more expressive than one wider head with the same{' '}
        <MathLabel tex="h \cdot d_h" /> dimension because{' '}
        <MathLabel tex="\operatorname{softmax}(A+B) \neq \operatorname{softmax}(A) + \operatorname{softmax}(B)" />
        . The slider rearranges the <MathLabel tex="d_m" /> budget across heads; it does not grow
        the cache.
      </p>

      {caption ? (
        <figcaption class="mha-caption">
          <MathText text={caption} mathClass="mha-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MultiHeadAttentionComponent = MultiHeadAttentionImpl as QuartzMdxComponent<Props>
MultiHeadAttentionComponent.css = style
MultiHeadAttentionComponent.afterDOMLoaded = script

export const MultiHeadAttention = registerMdxComponent(
  'MultiHeadAttention',
  MultiHeadAttentionComponent,
)

export default (() => MultiHeadAttention) satisfies (opts: undefined) => QuartzMdxComponent<Props>
