import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/sliding-window-mask.inline'
import style from '../styles/slidingWindowMask.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; length?: number }

const MIN_LENGTH = 8
const MAX_LENGTH = 64
const DEFAULT_LENGTH = 24
const DEFAULT_W = 2
const DEFAULT_D = 1
const DEFAULT_G = 1
const DILATIONS = [1, 2, 4] as const

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const renderMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({
  tex,
  display = false,
}) => {
  const Tag = display ? 'div' : 'span'
  const cls = `swm-math ${display ? 'swm-math--display' : 'swm-math--inline'}`
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }} />
}

const cellClass = (i: number, j: number, w: number, d: number, g: number): string => {
  const inGlobal = i < g || j < g
  if (inGlobal) return 'swm-cell swm-cell--global'
  const delta = j - i
  const inBand = Math.abs(delta) <= w
  const dilationOk = d === 1 || delta % d === 0
  if (inBand && dilationOk) return 'swm-cell swm-cell--window'
  return 'swm-cell swm-cell--masked'
}

const SlidingWindowMaskImpl: QuartzMdxComponent<Props> = ({ caption, length = DEFAULT_LENGTH }) => {
  const L = clamp(Math.round(length), MIN_LENGTH, MAX_LENGTH)
  const initialW = clamp(DEFAULT_W, 1, Math.floor(L / 2))
  const rows = Array.from({ length: L }, (_, i) => i)
  const ariaLabel = `Sliding window attention mask: ${L} by ${L} grid where rows are query indices and columns are key indices. Salmon cells lie inside the window, sage cells are global tokens, gray cells are masked.`

  return (
    <figure
      class="sliding-window-mask"
      data-sliding-window-mask
      data-swm-length={String(L)}
      data-swm-w={String(initialW)}
      data-swm-d={String(DEFAULT_D)}
      data-swm-g={String(DEFAULT_G)}
    >
      <div class="swm-stage">
        <div class="swm-grid-wrap">
          <svg
            class="swm-graph"
            viewBox={`0 0 ${L} ${L}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={ariaLabel}
            data-swm-svg
            shape-rendering="crispEdges"
          >
            <g data-swm-cells>
              {rows.map(i =>
                rows.map(j => (
                  <rect
                    class={cellClass(i, j, initialW, DEFAULT_D, DEFAULT_G)}
                    x={j}
                    y={i}
                    width="1"
                    height="1"
                    data-swm-cell={`${i}-${j}`}
                    data-swm-i={String(i)}
                    data-swm-j={String(j)}
                  />
                )),
              )}
            </g>
          </svg>
          <div class="swm-axis swm-axis--x" aria-hidden="true">
            <MathLabel tex="\text{key index}\ j" />
          </div>
          <div class="swm-axis swm-axis--y" aria-hidden="true">
            <MathLabel tex="\text{query index}\ i" />
          </div>
        </div>

        <div class="swm-side">
          <div class="swm-legend" aria-label="Mask legend">
            <span class="swm-legend-item">
              <span class="swm-swatch swm-swatch--window" aria-hidden="true" />
              window <MathLabel tex="|i-j|\le w" />
            </span>
            <span class="swm-legend-item">
              <span class="swm-swatch swm-swatch--global" aria-hidden="true" />
              global <MathLabel tex="i\in G\ \text{or}\ j\in G" />
            </span>
            <span class="swm-legend-item">
              <span class="swm-swatch swm-swatch--masked" aria-hidden="true" />
              masked <MathLabel tex="M_{ij}=-\infty" />
            </span>
          </div>

          <div class="swm-formula">
            <MathLabel
              display
              tex="M_{ij}=\begin{cases}0 & |i-j|\le w\ \text{or}\ j\in G\\ -\infty & \text{otherwise}\end{cases}"
            />
          </div>

          <dl class="swm-readout" data-swm-readout>
            <div class="swm-readout-row">
              <dt>active cells</dt>
              <dd>
                <span data-swm-active class="swm-readout-val swm-readout-val--big">
                  0
                </span>
                <span class="swm-readout-eq">/</span>
                <span class="swm-readout-val swm-readout-val--muted">{L * L}</span>
              </dd>
            </div>
            <div class="swm-readout-row">
              <dt>cost ratio</dt>
              <dd>
                <MathLabel tex="\dfrac{\#\text{active}}{L^2}" />
                <span class="swm-readout-eq">=</span>
                <span data-swm-ratio class="swm-readout-val">
                  0%
                </span>
              </dd>
            </div>
            <div class="swm-readout-row">
              <dt>complexity</dt>
              <dd class="swm-readout-val swm-readout-val--mono">
                <MathLabel tex="O(Lw) + O(Lg)" />
                <span data-swm-complexity />
              </dd>
            </div>
            <div class="swm-readout-row">
              <dt>receptive field after k layers</dt>
              <dd>
                <MathLabel tex="\approx k\cdot w" />
                <span class="swm-readout-eq">at</span>
                <MathLabel tex="k=8" />
                <span class="swm-readout-eq">:</span>
                <span data-swm-receptive class="swm-readout-val">
                  16
                </span>
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div class="swm-controls" role="group" aria-label="Sliding window mask controls">
        <div class="swm-control swm-control--slider">
          <label class="swm-label" for="swm-w-slider">
            window <MathLabel tex="w" />
          </label>
          <input
            id="swm-w-slider"
            class="swm-slider"
            type="range"
            min="1"
            max={Math.floor(L / 2)}
            value={initialW}
            step="1"
            data-swm-w-input
            aria-valuemin={1}
            aria-valuemax={Math.floor(L / 2)}
            aria-valuenow={initialW}
            aria-valuetext={`window radius ${initialW}`}
          />
          <span class="swm-slider-value" data-swm-w-value>
            {initialW}
          </span>
        </div>

        <div class="swm-control swm-control--toggle">
          <span class="swm-label">
            dilation <MathLabel tex="d" />
          </span>
          <div class="swm-toggle" role="radiogroup" aria-label="Dilation factor">
            {DILATIONS.map(d => (
              <button
                type="button"
                class={`swm-toggle-btn${d === DEFAULT_D ? ' is-active' : ''}`}
                data-swm-d-btn={String(d)}
                role="radio"
                aria-checked={d === DEFAULT_D ? 'true' : 'false'}
                aria-label={`dilation ${d}`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        <div class="swm-control swm-control--slider">
          <label class="swm-label" for="swm-g-slider">
            global <MathLabel tex="g" />
          </label>
          <input
            id="swm-g-slider"
            class="swm-slider"
            type="range"
            min="0"
            max="3"
            value={DEFAULT_G}
            step="1"
            data-swm-g-input
            aria-valuemin={0}
            aria-valuemax={3}
            aria-valuenow={DEFAULT_G}
            aria-valuetext={`${DEFAULT_G} global tokens`}
          />
          <span class="swm-slider-value" data-swm-g-value>
            {DEFAULT_G}
          </span>
        </div>
      </div>

      <p class="swm-intuition">
        Information clusters locally. Cap each token's view to a neighbourhood of radius{' '}
        <MathLabel tex="w" /> and pay a linear price; a handful of global tokens preserve long-range
        channels. As <MathLabel tex="w" /> grows the band thickens; at <MathLabel tex="w=L" /> you
        recover full attention.
      </p>

      {caption ? (
        <figcaption class="swm-caption">
          <MathText text={caption} mathClass="swm-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const SlidingWindowMaskComponent = SlidingWindowMaskImpl as QuartzMdxComponent<Props>
SlidingWindowMaskComponent.css = style
SlidingWindowMaskComponent.afterDOMLoaded = script

export const SlidingWindowMask = registerMdxComponent(
  'SlidingWindowMask',
  SlidingWindowMaskComponent,
)

export default (() => SlidingWindowMask) satisfies (opts: undefined) => QuartzMdxComponent<Props>
