import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/param-decomposition.inline'
import style from '../styles/paramDecomposition.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; components?: number }

const COMP_COUNTS = [4, 6, 8] as const
const MATRIX = 3
const FLAT = MATRIX * MATRIX * 3
const SEED = [
  0.92, 0.18, 0.74, 0.41, 0.88, 0.12, 0.63, 0.29, 0.55, 0.81, 0.07, 0.46, 0.33, 0.97, 0.22, 0.68,
  0.14, 0.59, 0.85, 0.31, 0.76, 0.09, 0.52, 0.44, 0.71, 0.26, 0.93,
]

const hash = (i: number, c: number) => SEED[(i * 7 + c * 13) % SEED.length]

const sparseColumn = (c: number): number[] =>
  Array.from({ length: FLAT }, (_, i) => {
    const keep = hash(i, c) > 0.62
    return keep ? 0.25 + hash(i + 3, c) * 0.75 : 0
  })

const COLUMNS: Record<number, number[][]> = Object.fromEntries(
  COMP_COUNTS.map(n => [n, Array.from({ length: n }, (_, c) => sparseColumn(c))]),
)

const weights = Array.from({ length: 3 }, (_, m) =>
  Array.from({ length: MATRIX * MATRIX }, (_, i) => 0.35 + SEED[(m * 9 + i) % SEED.length] * 0.65),
)

let instance = 0

const ParamDecompositionImpl: QuartzMdxComponent<Props> = ({ caption, components = 6 }) => {
  const id = `apd-${++instance}`
  const n0 = (COMP_COUNTS as readonly number[]).includes(components) ? components : 6
  const state = Object.fromEntries(
    COMP_COUNTS.map(n => [
      n,
      {
        columns: COLUMNS[n],
        nonzero: COLUMNS[n].reduce(
          (acc, col) => acc + col.filter(v => v > 0).length / FLAT,
          0,
        ) / n,
      },
    ]),
  )

  return (
    <figure
      class="param-decomposition"
      data-param-decomposition
      data-apd-n={String(n0)}
      data-apd-state={JSON.stringify(state)}
    >
      <div class="apd-controls">
        <fieldset class="apd-segments">
          <legend class="apd-legend">
            <MathText text="components $C$" mathClass="apd-math" />
          </legend>
          {COMP_COUNTS.map(n => (
            <label class="apd-segment">
              <input
                type="radio"
                name={`${id}-n`}
                value={String(n)}
                checked={n === n0}
                data-apd-n-input
              />
              <span>{String(n)}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <div class="apd-flow">
        <div class="apd-stage">
          <div class="apd-mats">
            {weights.map((mat, mi) => (
              <div class="apd-mat" key={mi} aria-label={`weight matrix W${mi + 1}`}>
                {mat.map((v, i) => (
                  <span
                    class="apd-cell"
                    style={{ '--apd-a': String(v) } as Record<string, string>}
                    key={i}
                  />
                ))}
              </div>
            ))}
          </div>
          <p class="apd-label">
            <MathText text="weights $W_1,W_2,W_3$" mathClass="apd-math" />
          </p>
        </div>

        <div class="apd-arrow" aria-hidden="true">
          flatten →
        </div>

        <div class="apd-stage">
          <div class="apd-vec" aria-label="parameter vector">
            {Array.from({ length: FLAT }, (_, i) => (
              <span
                class="apd-cell apd-cell--tall"
                style={{ '--apd-a': String(0.3 + SEED[i % SEED.length] * 0.7) } as Record<
                  string,
                  string
                >}
                key={i}
              />
            ))}
          </div>
          <p class="apd-label">
            <MathText text="$\theta^\star \in \mathbb{R}^{N}$" mathClass="apd-math" />
          </p>
        </div>

        <div class="apd-arrow" aria-hidden="true">
          decompose →
        </div>

        <div class="apd-stage apd-stage--grow">
          <div class="apd-cols" data-apd-cols>
            {COMP_COUNTS.map(n => (
              <div
                class="apd-colset"
                data-apd-colset={String(n)}
                data-apd-off={n !== n0 ? 'true' : undefined}
                key={n}
              >
                {COLUMNS[n].map((col, ci) => (
                  <div class="apd-col" key={ci}>
                    {col.map((v, i) => (
                      <span
                        class={`apd-cell apd-cell--tall${v === 0 ? ' apd-cell--zero' : ''}`}
                        style={{ '--apd-a': String(v) } as Record<string, string>}
                        key={i}
                      />
                    ))}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p class="apd-label">
            <MathText text="$\theta^\star \approx \sum_c P_c$" mathClass="apd-math" />
          </p>
        </div>
      </div>

      <dl class="apd-readout">
        <div>
          <dt>N</dt>
          <dd>{FLAT}</dd>
        </div>
        <div>
          <dt>components</dt>
          <dd data-apd-out="n">{n0}</dd>
        </div>
        <div>
          <dt>mean fill</dt>
          <dd data-apd-out="fill">{(state[n0].nonzero * 100).toFixed(0)}%</dd>
        </div>
        <div>
          <dt>objective</dt>
          <dd>faithful, minimal, simple</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="apd-caption">
          <MathText text={caption} mathClass="apd-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ParamDecompositionComponent = ParamDecompositionImpl as QuartzMdxComponent<Props>
ParamDecompositionComponent.css = style
ParamDecompositionComponent.afterDOMLoaded = script

export const ParamDecomposition = registerMdxComponent(
  'ParamDecomposition',
  ParamDecompositionComponent,
)

export default (() => ParamDecomposition) satisfies (
  opts: undefined,
) => QuartzMdxComponent<Props>
