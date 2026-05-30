import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/cascade-filter.inline'
import style from '../styles/cascadeFilter.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; tiles?: number }
type Preset = 'uniform' | 'spiky' | 'longtail'

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

function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function computeScores(preset: Preset, n: number): number[] {
  const rng = mulberry32(0xc45c4de + n)
  const base: number[] = []
  if (preset === 'uniform') {
    for (let i = 0; i < n; i++) base.push(0.42 + rng() * 0.18)
  } else if (preset === 'spiky') {
    const spikeA = Math.floor(n * 0.22)
    const spikeB = Math.floor(n * 0.71)
    for (let i = 0; i < n; i++) {
      const noise = rng() * 0.08
      if (i === spikeA) base.push(0.95 + noise * 0.05)
      else if (i === spikeB) base.push(0.82 + noise * 0.08)
      else base.push(0.06 + noise)
    }
  } else {
    for (let i = 0; i < n; i++) base.push(Math.exp(-i / (n * 0.28)) * (0.85 + rng() * 0.15))
    for (let i = 0; i < n; i++) {
      const j = Math.floor(rng() * (i + 1))
      ;[base[i], base[j]] = [base[j], base[i]]
    }
  }
  return base
}

const encodeScores = (scores: number[]): string => scores.map(s => s.toFixed(4)).join(',')

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
      class={`cf-fo ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex, false) }}
    />
  </foreignObject>
)

function MathLabel({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = renderMath(tex, display)
  const cls = `cf-math ${display ? 'cf-math--display' : 'cf-math--inline'}`
  const Tag = display ? 'div' : 'span'
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: html }} />
}

const TILE_W = 36
const TILE_GAP = 8
const PAD_X = 28
const COARSE_H = 138
const FINE_H = 116
const BAR_TOP = 18
const BAR_BOTTOM_PAD = 28

const CascadeFilterImpl: QuartzMdxComponent<Props> = ({ caption, tiles = 16 }) => {
  const n = Math.max(4, Math.min(48, Math.floor(tiles)))
  const presets: Preset[] = ['spiky', 'uniform', 'longtail']
  const scoresByPreset: Record<Preset, number[]> = {
    spiky: computeScores('spiky', n),
    uniform: computeScores('uniform', n),
    longtail: computeScores('longtail', n),
  }
  const preset: Preset = 'spiky'
  const tau = 0.5
  const width = PAD_X * 2 + n * TILE_W + (n - 1) * TILE_GAP
  const barH = COARSE_H - BAR_TOP - BAR_BOTTOM_PAD
  const lineY = BAR_TOP + barH * (1 - tau)
  const scoresData = JSON.stringify({
    spiky: encodeScores(scoresByPreset.spiky),
    uniform: encodeScores(scoresByPreset.uniform),
    longtail: encodeScores(scoresByPreset.longtail),
  })
  const scores = scoresByPreset[preset]

  return (
    <figure
      class="cascade-filter"
      data-cascade-filter
      data-tiles={n}
      data-preset={preset}
      data-threshold={tau.toFixed(3)}
      data-scores={scoresData}
    >
      <header class="cf-header">
        <div
          class="cf-tablist"
          role="tablist"
          aria-label="attention score distribution"
          aria-orientation="horizontal"
        >
          {presets.map(p => (
            <button
              type="button"
              class="cf-tab"
              data-cf-preset={p}
              role="tab"
              aria-selected={p === preset ? 'true' : 'false'}
              tabindex={p === preset ? 0 : -1}
            >
              {p === 'longtail' ? 'long tail' : p}
            </button>
          ))}
        </div>
        <div class="cf-slider-wrap">
          <label class="cf-slider-label" for="cf-threshold">
            <MathLabel tex="\text{threshold}\;\tau" />
          </label>
          <input
            id="cf-threshold"
            class="cf-slider"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={tau}
            data-cf-slider
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={tau}
            aria-valuetext={`tau equals ${tau.toFixed(2)}`}
          />
          <span
            class="cf-slider-value"
            data-cf-tau
            dangerouslySetInnerHTML={{ __html: renderMath(tau.toFixed(2), false) }}
          />
        </div>
      </header>

      <div class="cf-stage">
        <section class="cf-panel cf-panel--coarse" aria-label="Coarse scorer">
          <div
            class="cf-panel-title"
            dangerouslySetInnerHTML={{ __html: renderMath('\\text{coarse scorer}', false) }}
          />
          <svg
            class="cf-svg cf-svg--coarse"
            viewBox={`0 0 ${width} ${COARSE_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Coarse importance scores per KV tile with threshold line"
          >
            {scores.map((s, i) => {
              const x = PAD_X + i * (TILE_W + TILE_GAP)
              const h = s * barH
              const keep = s >= tau
              return (
                <g
                  class={`cf-tile-group ${keep ? 'is-keep' : 'is-drop'}`}
                  data-tile-idx={i}
                  transform={`translate(${x}, 0)`}
                >
                  <rect
                    class="cf-tile-frame"
                    x={0}
                    y={BAR_TOP}
                    width={TILE_W}
                    height={barH}
                    rx={3}
                  />
                  <rect
                    class="cf-tile-bar"
                    data-tile-bar
                    x={3}
                    y={BAR_TOP + (barH - h)}
                    width={TILE_W - 6}
                    height={h}
                    rx={2}
                  />
                  <MathFO
                    x={0}
                    y={COARSE_H - 22}
                    w={TILE_W}
                    h={16}
                    tex={`b_{${i}}`}
                    cls="cf-fo--tile-idx"
                  />
                </g>
              )
            })}
            <line
              class="cf-threshold-line"
              data-cf-threshold-line
              x1={PAD_X - 6}
              x2={width - PAD_X + 6}
              y1={lineY}
              y2={lineY}
            />
            <foreignObject
              data-cf-tau-label
              x={width - PAD_X - 30}
              y={lineY - 22}
              width={50}
              height={18}
            >
              <div
                class="cf-fo cf-fo--tau"
                dangerouslySetInnerHTML={{ __html: renderMath('\\tau', false) }}
              />
            </foreignObject>
          </svg>
        </section>

        <section class="cf-panel cf-panel--fine" aria-label="Fine attention">
          <div
            class="cf-panel-title"
            dangerouslySetInnerHTML={{
              __html: renderMath('\\text{fine attention (survivors)}', false),
            }}
          />
          <svg
            class="cf-svg cf-svg--fine"
            viewBox={`0 0 ${width} ${FINE_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Surviving KV tiles passed to fine softmax attention"
          >
            <MathFO
              x={PAD_X - 8}
              y={18}
              w={240}
              h={26}
              tex="\sum_{j \in \mathcal{S}} \exp(q\,k_j^{\top})\, v_j"
              cls="cf-fo--formula"
            />
            {scores.map((s, i) => {
              const x = PAD_X + i * (TILE_W + TILE_GAP)
              const keep = s >= tau
              return (
                <g
                  class={`cf-fine-group ${keep ? 'is-keep' : 'is-drop'}`}
                  data-fine-idx={i}
                  transform={`translate(${x}, 0)`}
                >
                  <rect class="cf-fine-tile" x={0} y={56} width={TILE_W} height={40} rx={3} />
                  <MathFO x={0} y={66} w={TILE_W} h={20} tex={`b_{${i}}`} cls="cf-fo--fine-label" />
                </g>
              )
            })}
          </svg>
        </section>
      </div>

      <aside class="cf-sidebar">
        <div class="cf-card">
          <div
            class="cf-card-title"
            dangerouslySetInnerHTML={{ __html: renderMath('\\text{live metrics}', false) }}
          />
          <dl class="cf-stats">
            <dt>survivors</dt>
            <dd>
              <strong
                data-cf-stat="kept"
                dangerouslySetInnerHTML={{ __html: renderMath('0', false) }}
              />
              <span
                class="cf-stat-of"
                dangerouslySetInnerHTML={{ __html: renderMath(`/\\,${n}`, false) }}
              />
            </dd>
            <dt>speedup</dt>
            <dd>
              <strong
                data-cf-stat="speedup"
                dangerouslySetInnerHTML={{ __html: renderMath('0.0\\times', false) }}
              />
            </dd>
            <dt>recall</dt>
            <dd>
              <strong
                data-cf-stat="recall"
                dangerouslySetInnerHTML={{ __html: renderMath('0\\%', false) }}
              />
            </dd>
          </dl>
        </div>
      </aside>

      <section class="cf-decomp" aria-label="Decomposition">
        <div
          class="cf-card-title"
          dangerouslySetInnerHTML={{ __html: renderMath('\\text{decomposition}', false) }}
        />
        <div class="cf-decomp-body">
          <MathLabel tex="A(Q,K,V) \approx A(Q,\, K_{\mathcal{S}},\, V_{\mathcal{S}})" display />
          <p class="cf-card-note">
            <MathLabel tex="\mathcal{S} = \{ j : s_j \ge \tau \}" />, speedup{' '}
            <MathLabel tex="= n/k" />, recall{' '}
            <MathLabel tex="= \tfrac{\sum_{j\in\mathcal{S}} s_j}{\sum_j s_j}" />.
          </p>
        </div>
      </section>

      {caption ? (
        <figcaption class="cf-caption">
          <MathText text={caption} mathClass="cf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CascadeFilterComponent = CascadeFilterImpl as QuartzMdxComponent<Props>
CascadeFilterComponent.css = style
CascadeFilterComponent.afterDOMLoaded = script

export const CascadeFilter = registerMdxComponent('CascadeFilter', CascadeFilterComponent)

export default (() => CascadeFilter) satisfies (opts: undefined) => QuartzMdxComponent<Props>
