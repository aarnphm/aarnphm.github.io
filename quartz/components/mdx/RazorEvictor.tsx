import katex from 'katex'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/razor-evictor.inline'
import style from '../styles/razorEvictor.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; capacity?: number }
type Policy = 'razor' | 'lru' | 'fifo'

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
  const cls = `rzr-math ${display ? 'rzr-math--display' : 'rzr-math--inline'}`
  const Tag = display ? 'div' : 'span'
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: html }} />
}

function MathFO({
  x,
  y,
  w,
  h,
  tex,
  cls,
}: {
  x: number
  y: number
  w: number
  h: number
  tex: string
  cls?: string
}) {
  return (
    <foreignObject x={x} y={y} width={w} height={h}>
      <div
        class={`rzr-fo ${cls ?? ''}`.trim()}
        dangerouslySetInnerHTML={{ __html: renderMath(tex, false) }}
      />
    </foreignObject>
  )
}

const SLOT_W = 56
const SLOT_GAP = 10
const SLOT_H = 88
const PAD_X = 24
const PAD_Y = 28

const RazorEvictorImpl: QuartzMdxComponent<Props> = ({ caption, capacity = 8 }) => {
  const cap = Math.max(4, Math.min(16, Math.floor(capacity)))
  const policies: Policy[] = ['razor', 'lru', 'fifo']
  const width = PAD_X * 2 + cap * SLOT_W + (cap - 1) * SLOT_GAP
  const height = PAD_Y * 2 + SLOT_H

  return (
    <figure class="razor-evictor" data-razor-evictor data-capacity={cap}>
      <header class="rzr-header">
        <div class="rzr-controls">
          <button
            type="button"
            class="rzr-btn rzr-btn--primary"
            data-rzr-next
            aria-label="Insert next token into cache"
          >
            next token
          </button>
          <button
            type="button"
            class="rzr-btn rzr-btn--ghost"
            data-rzr-reset
            aria-label="Reset cache state"
          >
            reset
          </button>
        </div>
        <div class="rzr-policies" role="radiogroup" aria-label="Eviction policy">
          {policies.map(p => (
            <button
              type="button"
              class="rzr-policy"
              data-rzr-policy={p}
              role="radio"
              aria-checked={p === 'razor' ? 'true' : 'false'}
              aria-label={`${p} eviction policy`}
            >
              {p}
            </button>
          ))}
        </div>
      </header>

      <div class="rzr-stage">
        <section class="rzr-panel" aria-label="KV cache slots">
          <h4 class="rzr-panel-title">
            cache slots (<MathLabel tex={`|C| = ${cap}`} />)
          </h4>
          <svg
            class="rzr-svg"
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Fixed-size KV cache with ${cap} slots, each showing a token label and importance bar`}
          >
            <MathFO x={4} y={PAD_Y - 8} w={36} h={16} tex="s_i" cls="rzr-fo--axis" />
            {Array.from({ length: cap }).map((_, i) => {
              const x = PAD_X + i * (SLOT_W + SLOT_GAP)
              return (
                <g
                  class="rzr-slot is-empty"
                  data-rzr-slot={i}
                  transform={`translate(${x}, ${PAD_Y})`}
                >
                  <rect class="rzr-slot-frame" x={0} y={0} width={SLOT_W} height={SLOT_H} rx={4} />
                  <rect
                    class="rzr-slot-bar"
                    data-rzr-slot-bar
                    x={4}
                    y={SLOT_H - 4}
                    width={SLOT_W - 8}
                    height={0}
                    rx={2}
                  />
                  <foreignObject x={0} y={SLOT_H / 2 - 22} width={SLOT_W} height={22}>
                    <div class="rzr-slot-label" data-rzr-slot-label>
                      -
                    </div>
                  </foreignObject>
                  <foreignObject x={0} y={SLOT_H / 2 + 2} width={SLOT_W} height={18}>
                    <div class="rzr-slot-score" data-rzr-slot-score />
                  </foreignObject>
                  <foreignObject x={0} y={SLOT_H + 4} width={SLOT_W} height={16}>
                    <div
                      class="rzr-fo rzr-fo--slot-idx"
                      dangerouslySetInnerHTML={{ __html: renderMath(`s_{${i}}`, false) }}
                    />
                  </foreignObject>
                </g>
              )
            })}
          </svg>
        </section>

        <section class="rzr-panel rzr-panel--history" aria-label="Token history">
          <h4 class="rzr-panel-title">history (last 16)</h4>
          <ol class="rzr-history" data-rzr-history aria-live="polite" />
        </section>
      </div>

      <aside class="rzr-sidebar">
        <div class="rzr-card">
          <h4>live metrics</h4>
          <dl class="rzr-stats">
            <dt>residents</dt>
            <dd>
              <strong data-rzr-stat="residents">0</strong>
              <span class="rzr-stat-of">/ {cap}</span>
            </dd>
            <dt>evictions</dt>
            <dd>
              <strong data-rzr-stat="evictions">0</strong>
            </dd>
            <dt>avg score</dt>
            <dd>
              <strong data-rzr-stat="avg">0.00</strong>
            </dd>
            <dt>mass retained</dt>
            <dd>
              <strong data-rzr-stat="mass">0%</strong>
            </dd>
          </dl>
        </div>
        <div class="rzr-card">
          <h4>eviction rule</h4>
          <div class="rzr-rule rzr-rule--razor" data-rzr-rule="razor">
            <MathLabel tex="i_{\text{evict}} = \argmin_{i \in C}\, s_i" display />
          </div>
          <div class="rzr-rule rzr-rule--lru" data-rzr-rule="lru" hidden>
            <MathLabel tex="i_{\text{evict}} = \argmin_{i \in C}\, t^{\text{last}}_i" display />
          </div>
          <div class="rzr-rule rzr-rule--fifo" data-rzr-rule="fifo" hidden>
            <MathLabel tex="i_{\text{evict}} = \argmin_{i \in C}\, t^{\text{insert}}_i" display />
          </div>
          <p class="rzr-card-note">
            razor shaves by relevance; LRU by recency; FIFO by insertion order. Expected retained
            mass <MathLabel tex="\mathbb{E}\Big[\sum_{i \in C(t)} s_i\Big]" /> differs by policy.
          </p>
        </div>
        <div class="rzr-card rzr-card--legend">
          <h4>legend</h4>
          <ul class="rzr-legend">
            <li>
              <span class="rzr-swatch rzr-swatch--stable" aria-hidden="true" /> resident
            </li>
            <li>
              <span class="rzr-swatch rzr-swatch--new" aria-hidden="true" /> just inserted
            </li>
            <li>
              <span class="rzr-swatch rzr-swatch--evict" aria-hidden="true" /> being evicted
            </li>
          </ul>
        </div>
      </aside>

      {caption ? (
        <figcaption class="rzr-caption">
          <MathText text={caption} />
        </figcaption>
      ) : null}
    </figure>
  )
}

const RazorEvictorComponent = RazorEvictorImpl as QuartzMdxComponent<Props>
RazorEvictorComponent.css = style
RazorEvictorComponent.afterDOMLoaded = script

export const RazorEvictor = registerMdxComponent('RazorEvictor', RazorEvictorComponent)

export default (() => RazorEvictor) satisfies (opts: undefined) => QuartzMdxComponent<Props>
