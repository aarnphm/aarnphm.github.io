import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/offset-heads-toy.inline'
import style from '../styles/offsetHeadsToy.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; length?: number }

const MIN_LENGTH = 4
const MAX_LENGTH = 10
const DEFAULT_LENGTH = 6
const DEFAULT_BETA = 6
const MIN_BETA = 1
const MAX_BETA = 20
const BETA_STEP = 0.5
const HEAD_DIM = 4

const PANELS = [
  {
    key: 'p1',
    label: 'P^{(+1)}',
    title: 'next-token head',
    swatch: 'oht-swatch--head1',
    panel: 'oht-panel--head1',
    cell: 'oht-cell--head1',
  },
  {
    key: 'm1',
    label: 'P^{(-1)}',
    title: 'prev-token head',
    swatch: 'oht-swatch--head2',
    panel: 'oht-panel--head2',
    cell: 'oht-cell--head2',
  },
  {
    key: 'sh',
    label: 'P = \\operatorname{softmax}(S^{(+1)} + S^{(-1)})',
    title: 'single-head surrogate',
    swatch: 'oht-swatch--surrogate',
    panel: 'oht-panel--surrogate',
    cell: 'oht-cell--surrogate',
  },
] as const

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
  const cls = `oht-math ${display ? 'oht-math--display' : 'oht-math--inline'}`
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }} />
}

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  tex: string
  align?: 'center' | 'start' | 'end'
}> = ({ x, y, w, h, tex, align = 'center' }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`oht-fo oht-fo--${align}`}
      dangerouslySetInnerHTML={{ __html: renderMath(tex, false) }}
    />
  </foreignObject>
)

const OffsetHeadsToyImpl: QuartzMdxComponent<Props> = ({ caption, length = DEFAULT_LENGTH }) => {
  const L = clamp(Math.round(length), MIN_LENGTH, MAX_LENGTH)
  const rows = Array.from({ length: L }, (_, i) => i)
  const viewSize = L + 2
  const cellPad = 1
  const ariaSummary = `Two-offset-heads toy: three ${L} by ${L} probability matrices side by side, next-token head, prev-token head, and a single-head surrogate that adds the two score matrices before one softmax.`

  return (
    <figure
      class="offset-heads-toy"
      data-offset-heads-toy
      data-oht-length={String(L)}
      data-oht-beta={String(DEFAULT_BETA)}
      data-oht-seed="1"
      data-oht-mode="independent"
      data-oht-head-dim={String(HEAD_DIM)}
    >
      <div class="oht-panels" role="group" aria-label="Attention probability matrices">
        {PANELS.map(panel => (
          <div class={`oht-panel ${panel.panel}`} key={panel.key} data-oht-panel={panel.key}>
            <header class="oht-panel-head">
              <span class={`oht-swatch ${panel.swatch}`} aria-hidden="true" />
              <span class="oht-panel-title">{panel.title}</span>
            </header>
            <svg
              class="oht-grid"
              viewBox={`0 0 ${viewSize} ${viewSize}`}
              preserveAspectRatio="xMidYMid meet"
              role="img"
              aria-label={`${panel.title} probability matrix, ${L} rows by ${L} columns; brighter cells carry more attention mass.`}
              shape-rendering="crispEdges"
            >
              <g data-oht-axis>
                {rows.map(i => (
                  <MathFO
                    x={cellPad - 0.62}
                    y={cellPad + i + 0.18}
                    w={0.42}
                    h={0.64}
                    tex={String(i)}
                  />
                ))}
                {rows.map(j => (
                  <MathFO
                    x={cellPad + j + 0.22}
                    y={cellPad - 0.62}
                    w={0.56}
                    h={0.52}
                    tex={String(j)}
                  />
                ))}
              </g>
              <g data-oht-cells={panel.key}>
                {rows.map(i =>
                  rows.map(j => (
                    <rect
                      class={`oht-cell ${panel.cell}`}
                      x={cellPad + j}
                      y={cellPad + i}
                      width="1"
                      height="1"
                      data-oht-cell={`${panel.key}-${i}-${j}`}
                      data-oht-i={String(i)}
                      data-oht-j={String(j)}
                      fill-opacity={0}
                    >
                      <title data-oht-tooltip={`${panel.key}-${i}-${j}`}>0.000</title>
                    </rect>
                  )),
                )}
              </g>
              <MathFO
                x={cellPad + L / 2 - 1.6}
                y={cellPad + L + 0.05}
                w={3.2}
                h={0.85}
                tex="\text{key } j"
              />
              <foreignObject x={-0.1} y={cellPad + L / 2 - 1.6} width={0.85} height={3.2}>
                <div class="oht-fo oht-fo--center oht-fo--vert">
                  <div
                    dangerouslySetInnerHTML={{ __html: renderMath('\\text{query } i', false) }}
                  />
                </div>
              </foreignObject>
            </svg>
            <div class="oht-panel-formula">
              <MathLabel tex={panel.label} />
            </div>
          </div>
        ))}
      </div>

      <div class="oht-readout" data-oht-readout>
        <div class="oht-formula">
          <MathLabel
            display
            tex="\operatorname{softmax}(A+B) \;\neq\; \operatorname{softmax}(A) + \operatorname{softmax}(B)"
          />
        </div>
        <dl class="oht-stats">
          <div class="oht-stat-row">
            <dt>
              <MathLabel tex="\| Y_{\text{MHA}} - Y_{\text{SH}} \|_F" />
            </dt>
            <dd>
              <span class="oht-stat-val oht-stat-val--big" data-oht-stat="norm" aria-live="polite">
                0.000
              </span>
            </dd>
          </div>
          <div class="oht-stat-row">
            <dt>
              <MathLabel tex="\| Y_{\text{MHA}} \|_F" />
            </dt>
            <dd>
              <span class="oht-stat-val" data-oht-stat="ref">
                0.000
              </span>
            </dd>
          </div>
          <div class="oht-stat-row">
            <dt>relative gap</dt>
            <dd>
              <span class="oht-stat-val" data-oht-stat="rel">
                0%
              </span>
            </dd>
          </div>
          <div class="oht-stat-row">
            <dt>seed</dt>
            <dd>
              <span class="oht-stat-val oht-stat-val--mono" data-oht-stat="seed">
                1
              </span>
            </dd>
          </div>
        </dl>
        <p class="oht-note">
          Two independent normalisers supply two distributions you can recombine with{' '}
          <MathLabel tex="W_O^{(1)}, W_O^{(2)}" />. One normaliser splits the same mass across both
          diagonals; no single <MathLabel tex="\tilde W_O" /> closes the gap.
        </p>
      </div>

      <div class="oht-controls" role="group" aria-label="Toy controls">
        <div class="oht-control oht-control--slider">
          <label class="oht-label" for="oht-beta-slider">
            sharpness <MathLabel tex="\beta" />
          </label>
          <input
            id="oht-beta-slider"
            class="oht-slider"
            type="range"
            min={MIN_BETA}
            max={MAX_BETA}
            step={BETA_STEP}
            value={DEFAULT_BETA}
            data-oht-beta-input
            aria-valuemin={MIN_BETA}
            aria-valuemax={MAX_BETA}
            aria-valuenow={DEFAULT_BETA}
            aria-valuetext={`beta ${DEFAULT_BETA}`}
          />
          <span class="oht-slider-value" data-oht-beta-value>
            {DEFAULT_BETA.toFixed(1)}
          </span>
        </div>

        <div class="oht-control oht-control--toggle">
          <span class="oht-label">surrogate</span>
          <div class="oht-toggle" role="radiogroup" aria-label="Surrogate projection mode">
            <button
              type="button"
              class="oht-toggle-btn is-active"
              data-oht-mode-btn="independent"
              role="radio"
              aria-checked="true"
              aria-label="Independent random projection"
            >
              independent <MathLabel tex="\tilde W_O" />
            </button>
            <button
              type="button"
              class="oht-toggle-btn"
              data-oht-mode-btn="forced"
              role="radio"
              aria-checked="false"
              aria-label="Forced match attempt"
            >
              forced match
            </button>
          </div>
        </div>

        <div class="oht-control oht-control--actions">
          <button
            type="button"
            class="oht-btn oht-btn--primary"
            data-oht-reseed
            aria-label="Reseed value and projection matrices"
          >
            reseed
          </button>
        </div>
      </div>

      <div class="oht-sr" aria-live="polite" data-oht-sr>
        {ariaSummary}
      </div>

      {caption ? (
        <figcaption class="oht-caption">
          <MathText text={caption} mathClass="oht-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const OffsetHeadsToyComponent = OffsetHeadsToyImpl as QuartzMdxComponent<Props>
OffsetHeadsToyComponent.css = style
OffsetHeadsToyComponent.afterDOMLoaded = script

export const OffsetHeadsToy = registerMdxComponent('OffsetHeadsToy', OffsetHeadsToyComponent)

export default (() => OffsetHeadsToy) satisfies (opts: undefined) => QuartzMdxComponent<Props>
