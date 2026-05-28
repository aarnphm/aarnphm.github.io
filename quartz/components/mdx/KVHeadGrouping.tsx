import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/kv-head-grouping.inline'
import style from '../styles/kvHeadGrouping.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; heads?: number }

const MIN_HEADS = 4
const MAX_HEADS = 16
const DEFAULT_HEADS = 8

const VIEW_W = 520
const VIEW_H = 260
const ROW_Q_Y = 56
const ROW_KV_Y = 184
const BOX_W = 36
const BOX_H = 32
const ROW_PAD = 28

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
      class={`kvg-fo ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({ tex, display }) => (
  <span
    class={`kvg-math${display ? ' kvg-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }}
  />
)

const clamp = (value: number, lo: number, hi: number) => Math.min(Math.max(value, lo), hi)

const nearestPowerOfTwo = (n: number) => {
  if (n <= 1) return 1
  const exp = Math.round(Math.log2(n))
  return 2 ** exp
}

const headCenterX = (i: number, h: number) => {
  const span = VIEW_W - 2 * ROW_PAD
  return ROW_PAD + (span / h) * (i + 0.5)
}

const kvCenterX = (g: number, nk: number) => {
  const span = VIEW_W - 2 * ROW_PAD
  return ROW_PAD + (span / nk) * (g + 0.5)
}

const ratioName = (r: number, h: number) => {
  if (r === 1) return 'MHA'
  if (r === h) return 'MQA'
  return `GQA r=${r}`
}

const ratioTex = (r: number, h: number) => {
  if (r === 1) return '\\mathrm{MHA}'
  if (r === h) return '\\mathrm{MQA}'
  return `\\mathrm{GQA}\\ r=${r}`
}

const KVHeadGroupingImpl: QuartzMdxComponent<Props> = ({ caption, heads = DEFAULT_HEADS }) => {
  const h = clamp(nearestPowerOfTwo(heads), MIN_HEADS, MAX_HEADS)
  const initialR = 2
  const initialNk = h / initialR
  const headIndices = Array.from({ length: h }, (_, i) => i)
  const groupIndices = Array.from({ length: h }, (_, i) => i)
  const ratios: number[] = []
  for (let r = 1; r <= h; r *= 2) ratios.push(r)

  return (
    <figure
      class="kv-head-grouping"
      data-kv-head-grouping
      data-heads={String(h)}
      data-ratios={ratios.join(',')}
    >
      <div class="kvg-stage">
        <svg
          class="kvg-graph"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Query heads on top connecting to ${initialNk} shared KV box${initialNk === 1 ? '' : 'es'} below; current grouping ${ratioName(initialR, h)}.`}
          data-kvg-canvas
        >
          <defs>
            <marker
              id="kvg-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path class="kvg-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <MathFO x={ROW_PAD - 42} y={ROW_Q_Y} w={28} h={BOX_H} tex="Q" cls="kvg-fo--row-label" />
          <MathFO
            x={ROW_PAD - 42}
            y={ROW_KV_Y}
            w={28}
            h={BOX_H}
            tex="K,V"
            cls="kvg-fo--row-label"
          />

          <g data-kvg-lines>
            {headIndices.map(i => {
              const g = Math.floor(i / initialR)
              const x1 = headCenterX(i, h)
              const x2 = kvCenterX(g, initialNk)
              return (
                <line
                  class="kvg-link"
                  data-kvg-link={String(i)}
                  data-kvg-group={String(g)}
                  x1={x1}
                  y1={ROW_Q_Y + BOX_H}
                  x2={x2}
                  y2={ROW_KV_Y}
                />
              )
            })}
          </g>

          <g data-kvg-queries>
            {headIndices.map(i => {
              const x = headCenterX(i, h) - BOX_W / 2
              const g = Math.floor(i / initialR)
              return (
                <g class="kvg-query" data-kvg-query={String(i)} data-kvg-group={String(g)}>
                  <rect
                    class="kvg-box kvg-box--query"
                    x={x}
                    y={ROW_Q_Y}
                    width={BOX_W}
                    height={BOX_H}
                    rx="4"
                  />
                  <MathFO
                    x={x}
                    y={ROW_Q_Y}
                    w={BOX_W}
                    h={BOX_H}
                    tex={`Q_{${i}}`}
                    cls="kvg-fo--box"
                  />
                </g>
              )
            })}
          </g>

          <g data-kvg-kvs>
            {groupIndices.map(g => {
              const visible = g < initialNk
              const cx = kvCenterX(Math.min(g, initialNk - 1), initialNk)
              const span = (VIEW_W - 2 * ROW_PAD) / initialNk
              const w = Math.max(BOX_W, span - 12)
              const x = cx - w / 2
              return (
                <g
                  class="kvg-kv"
                  data-kvg-kv={String(g)}
                  data-kvg-hidden={visible ? 'false' : 'true'}
                >
                  <rect
                    class="kvg-box kvg-box--kv"
                    x={x}
                    y={ROW_KV_Y}
                    width={w}
                    height={BOX_H}
                    rx="4"
                  />
                  <MathFO
                    x={x}
                    y={ROW_KV_Y}
                    w={w}
                    h={BOX_H}
                    tex={`K_{${g}},V_{${g}}`}
                    cls="kvg-fo--kv"
                  />
                </g>
              )
            })}
          </g>
        </svg>

        <aside class="kvg-side" aria-label="Cache bandwidth readout">
          <dl class="kvg-readout" data-kvg-readout>
            <div class="kvg-readout-row">
              <dt>
                <MathLabel tex="\text{regime}" />
              </dt>
              <dd>
                {ratios.map(r => (
                  <span
                    data-kvg-regime-r={String(r)}
                    data-kvg-active={r === initialR ? 'true' : 'false'}
                    class="kvg-readout-val kvg-readout-val--big"
                    dangerouslySetInnerHTML={{ __html: renderMath(ratioTex(r, h)) }}
                  />
                ))}
              </dd>
            </div>
            <div class="kvg-readout-row">
              <dt>
                <MathLabel tex="\mathrm{KV}\ \text{heads}" />
              </dt>
              <dd>
                {ratios.map(r => {
                  const nk = h / r
                  return (
                    <span
                      data-kvg-nk-r={String(r)}
                      data-kvg-active={nk === initialNk ? 'true' : 'false'}
                      class="kvg-readout-val"
                      dangerouslySetInnerHTML={{ __html: renderMath(`n_k = h/r = ${nk}`) }}
                    />
                  )
                })}
              </dd>
            </div>
            <div class="kvg-readout-row">
              <dt>
                <MathLabel tex="\text{decode bandwidth}" />
              </dt>
              <dd>
                {ratios.map(r => (
                  <span
                    data-kvg-bw-r={String(r)}
                    data-kvg-active={r === initialR ? 'true' : 'false'}
                    class="kvg-readout-val"
                    dangerouslySetInnerHTML={{
                      __html: renderMath(r === 1 ? '1\\times' : `\\frac{1}{${r}}\\times`),
                    }}
                  />
                ))}
                <MathLabel tex="\mathrm{MHA}" />
              </dd>
            </div>
            <div class="kvg-readout-row">
              <dt>
                <MathLabel tex="\text{cache bytes}/\text{token}" />
              </dt>
              <dd>
                <MathLabel tex="2 n_k d_h" />
                <MathLabel tex="\to" />
                {ratios.map(r => {
                  const nk = h / r
                  return (
                    <span
                      data-kvg-cache-nk={String(nk)}
                      data-kvg-active={nk === initialNk ? 'true' : 'false'}
                      class="kvg-readout-val"
                      dangerouslySetInnerHTML={{ __html: renderMath(`2 \\cdot ${nk} \\cdot d_h`) }}
                    />
                  )
                })}
              </dd>
            </div>
          </dl>
          <div class="kvg-equation">
            <MathLabel
              tex="\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_{g(i)}^{\top}}{\sqrt{d_h}}\right) V_{g(i)}"
              display
            />
          </div>
        </aside>
      </div>

      <div class="kvg-controls" role="group" aria-label="Group ratio controls">
        <div class="kvg-ratio-strip" data-kvg-ratio-strip>
          {ratios.map((r, i) => (
            <span
              class="kvg-ratio-tick"
              data-kvg-ratio-label={String(r)}
              data-kvg-active={r === initialR ? 'true' : 'false'}
              style={{ '--kvg-stop': ratios.length === 1 ? '0' : String(i / (ratios.length - 1)) }}
            >
              <MathLabel tex={ratioTex(r, h)} />
            </span>
          ))}
        </div>
        <div class="kvg-control kvg-control--slider">
          <label class="kvg-label" for="kvg-ratio-slider">
            <MathLabel tex="\text{group ratio } r" />
          </label>
          <input
            id="kvg-ratio-slider"
            class="kvg-slider"
            type="range"
            min="0"
            max={ratios.length - 1}
            step="1"
            value={ratios.indexOf(initialR)}
            data-kvg-ratio
            aria-valuemin={1}
            aria-valuemax={h}
            aria-valuenow={initialR}
            aria-valuetext={ratioName(initialR, h)}
          />
        </div>
      </div>

      <p class="kvg-intuition">
        Per-head queries stay cheap to compute, but per-head <MathLabel tex="K,V" /> are{' '}
        <strong>expensive at decode</strong> because every token reads the full cache. Sharing{' '}
        <MathLabel tex="K,V" /> across <MathLabel tex="r" /> query heads preserves query diversity
        and slashes cache reads by <MathLabel tex="1/r" />.
      </p>

      {caption ? (
        <figcaption class="kvg-caption">
          <MathText text={caption} mathClass="kvg-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const KVHeadGroupingComponent = KVHeadGroupingImpl as QuartzMdxComponent<Props>
KVHeadGroupingComponent.css = style
KVHeadGroupingComponent.afterDOMLoaded = script

export const KVHeadGrouping = registerMdxComponent('KVHeadGrouping', KVHeadGroupingComponent)

export default (() => KVHeadGrouping) satisfies (opts: undefined) => QuartzMdxComponent<Props>
