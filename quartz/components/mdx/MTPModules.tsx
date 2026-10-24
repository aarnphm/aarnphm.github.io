import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/mtp-modules.inline'
import style from '../styles/mtpModules.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; depth?: number }

const DEPTHS = [1, 2, 3] as const
const TOKENS = [1, 2, 3, 4, 5, 6, 7] as const

const shift = (start: number, n: number) =>
  Array.from({ length: n }, (_, i) => TOKENS[start + i - 1] ?? start + i)

const Arrow = ({ side }: { side?: 'right' }) => (
  <div class="mtp-varrow" data-side={side} aria-hidden="true">
    <span class="mtp-varrow-line" />
  </div>
)

let instance = 0

const MTPModulesImpl: QuartzMdxComponent<Props> = ({ caption, depth = 2 }) => {
  const id = `mtp-${++instance}`
  const d0 = (DEPTHS as readonly number[]).includes(depth) ? depth : 2

  return (
    <figure class="mtp-modules" data-mtp-modules data-mtp-depth={String(d0)}>
      <div class="mtp-controls">
        <fieldset class="mtp-segments">
          <legend class="mtp-legend">
            <MathText text="MTP depth $D$" mathClass="mtp-math" />
          </legend>
          {DEPTHS.map(d => (
            <label class="mtp-segment">
              <input
                type="radio"
                name={`${id}-d`}
                value={String(d)}
                checked={d === d0}
                data-mtp-depth-input
              />
              <span>{String(d)}</span>
            </label>
          ))}
        </fieldset>
        <ul class="mtp-legend-keys" aria-label="token kinds">
          <li>
            <span class="mtp-tok mtp-swatch" data-kind="input" />
            input
          </li>
          <li>
            <span class="mtp-tok mtp-swatch" data-kind="target" />
            target
          </li>
        </ul>
      </div>

      <div class="mtp-stage" data-mtp-stage>
        <svg class="mtp-links" data-mtp-links aria-hidden="true" />
        <div class="mtp-rail" data-mtp-rail>
          <section class="mtp-col" data-mtp-col="main">
            <header class="mtp-head">
              <h3 class="mtp-title">main</h3>
              <p class="mtp-sub">NTP</p>
            </header>
            <div class="mtp-targets" data-mtp-node="targets">
              {shift(2, 4).map(t => (
                <span class="mtp-tok" data-kind="target">
                  t{t}
                </span>
              ))}
            </div>
            <Arrow />
            <div class="mtp-block mtp-block--op" data-mtp-node="loss">
              cross-entropy
              <span class="mtp-loss-tag">
                <MathText text="$\mathcal{L}_{\mathrm{Main}}$" mathClass="mtp-math" />
              </span>
            </div>
            <Arrow />
            <div class="mtp-block mtp-block--shared" data-mtp-shared="head" data-mtp-node="head">
              output head
            </div>
            <Arrow />
            <div class="mtp-block mtp-block--stack" data-mtp-out-h data-mtp-node="transformer">
              transformer × L
            </div>
            <Arrow />
            <div class="mtp-block mtp-block--shared" data-mtp-shared="emb" data-mtp-node="emb">
              embedding
            </div>
            <Arrow />
            <div class="mtp-inputs" data-mtp-node="inputs">
              {shift(1, 4).map(t => (
                <span class="mtp-tok" data-kind="input">
                  t{t}
                </span>
              ))}
            </div>
          </section>

          {DEPTHS.map(d => (
            <section
              class="mtp-col"
              data-mtp-col={String(d)}
              data-mtp-off={d > d0 ? 'true' : undefined}
              key={d}
            >
              <header class="mtp-head">
                <h3 class="mtp-title">MTP {d}</h3>
                <p class="mtp-sub">
                  next<sup>{d + 1}</sup> token
                </p>
              </header>
              <div class="mtp-targets" data-mtp-node="targets">
                {shift(2 + d, 4).map(t => (
                  <span class="mtp-tok" data-kind="target">
                    t{t}
                  </span>
                ))}
              </div>
              <Arrow />
              <div class="mtp-block mtp-block--op" data-mtp-node="loss">
                cross-entropy
                <span class="mtp-loss-tag">
                  <MathText text={`$\\mathcal{L}_{\\mathrm{MTP}}^{${d}}$`} mathClass="mtp-math" />
                </span>
              </div>
              <Arrow />
              <div class="mtp-block mtp-block--shared" data-mtp-shared="head" data-mtp-node="head">
                output head
              </div>
              <Arrow />
              <div class="mtp-block" data-mtp-out-h data-mtp-node="transformer">
                transformer block
              </div>
              <Arrow />
              <div class="mtp-block" data-mtp-node="proj">
                linear projection
              </div>
              <Arrow />
              <div class="mtp-block mtp-block--concat" data-mtp-node="concat">
                concatenation
              </div>
              <div class="mtp-merge" aria-hidden="true">
                <span class="mtp-merge-stem mtp-merge-stem--l" />
                <span class="mtp-merge-stem mtp-merge-stem--r" />
                <span class="mtp-merge-bar" />
                <span class="mtp-varrow-line mtp-merge-up" />
              </div>
              <div class="mtp-row" data-mtp-node="norms">
                <div class="mtp-block mtp-block--small" data-mtp-in-h data-mtp-norm="prev">
                  RMSNorm
                </div>
                <div class="mtp-block mtp-block--small" data-mtp-norm="emb">
                  RMSNorm
                </div>
              </div>
              <Arrow side="right" />
              <div class="mtp-block mtp-block--shared" data-mtp-shared="emb" data-mtp-node="emb">
                embedding
              </div>
              <Arrow />
              <div class="mtp-inputs" data-mtp-node="inputs">
                {shift(1 + d, 4).map(t => (
                  <span class="mtp-tok" data-kind="input">
                    t{t}
                  </span>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>

      <dl class="mtp-readout">
        <div>
          <dt>causal chain</dt>
          <dd data-mtp-out="chain">main → MTP 1 → … → MTP {d0}</dd>
        </div>
        <div>
          <dt>shared</dt>
          <dd>embedding + output head</dd>
        </div>
        <div>
          <dt>modules</dt>
          <dd data-mtp-out="count">{d0}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="mtp-caption">
          <MathText text={caption} mathClass="mtp-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MTPModulesComponent = MTPModulesImpl as QuartzMdxComponent<Props>
MTPModulesComponent.css = style
MTPModulesComponent.afterDOMLoaded = script

export const MTPModules = registerMdxComponent('MTPModules', MTPModulesComponent)

export default (() => MTPModules) satisfies (opts: undefined) => QuartzMdxComponent<Props>
