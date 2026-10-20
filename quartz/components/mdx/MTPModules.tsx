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

let instance = 0

const MTPModulesImpl: QuartzMdxComponent<Props> = ({ caption, depth = 2 }) => {
  const id = `mtp-${++instance}`
  const d0 = (DEPTHS as readonly number[]).includes(depth) ? depth : 2

  return (
    <figure
      class="mtp-modules"
      data-mtp-modules
      data-mtp-depth={String(d0)}
    >
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
      </div>

      <div class="mtp-rail" data-mtp-rail>
        <section class="mtp-col" data-mtp-col="main">
          <header class="mtp-head">
            <h3 class="mtp-title">main model</h3>
            <p class="mtp-sub">next-token prediction</p>
          </header>
          <div class="mtp-targets">
            <span class="mtp-k">targets</span>
            {shift(2, 4).map(t => (
              <span class="mtp-tok">t{t}</span>
            ))}
          </div>
          <div class="mtp-block mtp-block--shared" data-mtp-shared="head">
            output head
          </div>
          <div class="mtp-block mtp-block--stack">transformer × L</div>
          <div class="mtp-block mtp-block--shared" data-mtp-shared="emb">
            embedding
          </div>
          <div class="mtp-inputs">
            <span class="mtp-k">inputs</span>
            {shift(1, 4).map(t => (
              <span class="mtp-tok">t{t}</span>
            ))}
          </div>
          <div class="mtp-loss">
            <MathText text="$\mathcal{L}_{\mathrm{Main}}$" mathClass="mtp-math" />
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
            <div class="mtp-targets">
              <span class="mtp-k">targets</span>
              {shift(2 + d, 4).map(t => (
                <span class="mtp-tok">t{t}</span>
              ))}
            </div>
            <div class="mtp-block mtp-block--shared" data-mtp-shared="head">
              output head
              <span class="mtp-share">shared</span>
            </div>
            <div class="mtp-block">transformer block</div>
            <div class="mtp-block">linear projection</div>
            <div class="mtp-row">
              <div class="mtp-block mtp-block--small">RMSNorm</div>
              <div class="mtp-block mtp-block--small">RMSNorm</div>
            </div>
            <div class="mtp-concat">concat ← previous hidden</div>
            <div class="mtp-block mtp-block--shared" data-mtp-shared="emb">
              embedding
              <span class="mtp-share">shared</span>
            </div>
            <div class="mtp-inputs">
              <span class="mtp-k">inputs</span>
              {shift(1 + d, 4).map(t => (
                <span class="mtp-tok">t{t}</span>
              ))}
            </div>
            <div class="mtp-loss">
              <MathText text={`$\\mathcal{L}_{\\mathrm{MTP}}^{${d}}$`} mathClass="mtp-math" />
            </div>
          </section>
        ))}
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
