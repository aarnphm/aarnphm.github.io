import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/conv-locality.inline'
import style from '../styles/convLocality.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type View = 'pooling' | 'connectivity'
type Props = { caption?: string; view?: View }

// Pooling: the 4 x 4 example from the lecture's max-pooling figure.
const POOL_IN = [
  [1, 1, 2, 4],
  [5, 6, 7, 8],
  [3, 2, 1, 0],
  [1, 2, 3, 4],
]
const W1 = POOL_IN.length
const SIZES = [2, 3] as const
const STRIDES = [1, 2] as const

// Output width without padding. A window that would run past the edge is dropped (floor).
const outWidth = (f: number, s: number) => Math.floor((W1 - f) / s) + 1

type PoolState = { cells: string; max: string; rows: string; cols: string }
const POOL: Record<string, PoolState> = {}
const POOL_META: Record<string, { n: number; skipped: number; w2: string }> = {}
for (const f of SIZES) {
  for (const s of STRIDES) {
    const n = outWidth(f, s)
    const covered = new Set<number>()
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const idx: number[] = []
        for (let i = 0; i < f; i++)
          for (let j = 0; j < f; j++) idx.push((r * s + i) * W1 + c * s + j)
        idx.forEach(k => covered.add(k))
        const max = Math.max(...idx.map(k => POOL_IN[Math.floor(k / W1)][k % W1]))
        const cells = Array.from({ length: W1 * W1 }, (_, k) =>
          !idx.includes(k) ? '0' : POOL_IN[Math.floor(k / W1)][k % W1] === max ? 'm' : 'w',
        ).join('')
        POOL[`${f}:${s}:${r * n + c}`] = {
          cells,
          max: String(max),
          rows: `${r * s} to ${r * s + f - 1}`,
          cols: `${c * s} to ${c * s + f - 1}`,
        }
      }
    }
    POOL_META[`${f}:${s}`] = { n, skipped: W1 * W1 - covered.size, w2: `${n} × ${n}` }
  }
}
const poolOut = (f: number, s: number, i: number) => POOL[`${f}:${s}:${i}`].max

// Connectivity: a 6 x 6 input and 3 x 3 filters at stride 1 give a 4 x 4 layer of units.
const IN = 6
const F = 3
const HID = IN - F + 1
type Mode = 'dense' | 'local' | 'shared'
const MODES: readonly { id: Mode; label: string; rule: string }[] = [
  {
    id: 'dense',
    label: 'Dense',
    rule: 'Every unit reads all $36$ inputs through its own weights: $36 \\times 16$ weights.',
  },
  {
    id: 'local',
    label: 'Local',
    rule: 'Each unit reads a $3 \\times 3$ patch through its own weights: $9 \\times 16$ weights.',
  },
  {
    id: 'shared',
    label: 'Shared (convolution)',
    rule: 'Each unit reads a $3 \\times 3$ patch, and all units reuse one filter $w$: $9$ weights.',
  },
]

type ConnState = { ids: string[]; params: string; reads: string; range: string; share: string }
const CONN: Record<string, ConnState> = {}
for (const { id: m } of MODES) {
  for (let u = 0; u < HID * HID; u++) {
    const ur = Math.floor(u / HID)
    const uc = u % HID
    const ids = Array.from({ length: IN * IN }, (_, k) => {
      const r = Math.floor(k / IN)
      const c = k % IN
      const inField = r >= ur && r < ur + F && c >= uc && c < uc + F
      const offset = (r - ur) * F + (c - uc)
      // Ids index the layer's weight vector, so the largest id is the parameter count.
      if (m === 'dense') return String(u * IN * IN + k + 1)
      if (!inField) return ''
      return String(m === 'local' ? u * F * F + offset + 1 : offset + 1)
    })
    const lo = m === 'dense' ? u * IN * IN + 1 : m === 'local' ? u * F * F + 1 : 1
    const per = m === 'dense' ? IN * IN : F * F
    CONN[`${m}:${u}`] = {
      ids,
      params: String(
        m === 'dense' ? IN * IN * HID * HID : m === 'local' ? F * F * HID * HID : F * F,
      ),
      reads: String(per),
      range: `${lo} to ${lo + per - 1}`,
      share: m === 'shared' ? `all ${HID * HID}` : 'this unit only',
    }
  }
}

let instance = 0

const Segments = <T extends string | number>({
  name,
  legend,
  options,
  value,
  data,
}: {
  name: string
  legend: string
  options: readonly { id: T; label: string }[]
  value: T
  data: string
}) => (
  <fieldset class="cvl-segments">
    <legend class="cvl-legend">
      <MathText text={legend} mathClass="cvl-math" />
    </legend>
    {options.map(o => (
      <label class="cvl-segment">
        <input
          type="radio"
          name={name}
          value={String(o.id)}
          checked={o.id === value}
          {...{ [data]: true }}
        />
        <span>{o.label}</span>
      </label>
    ))}
  </fieldset>
)

const PoolingView = ({ id }: { id: string }) => {
  const f0 = 2
  const s0 = 2
  const st = POOL[`${f0}:${s0}:0`]
  return (
    <>
      <div class="cvl-controls">
        <Segments
          name={`${id}-size`}
          legend="Window $F$"
          options={SIZES.map(f => ({ id: f, label: `${f} × ${f}` }))}
          value={f0}
          data="data-cvl-size"
        />
        <Segments
          name={`${id}-stride`}
          legend="Stride $S$"
          options={STRIDES.map(s => ({ id: s, label: String(s) }))}
          value={s0}
          data="data-cvl-stride"
        />
      </div>
      <div class="cvl-stage">
        <div class="cvl-block">
          <p class="cvl-heading">Input, 4 × 4</p>
          <div class="cvl-grid cvl-grid--pool" style={`--cvl-n:${W1}`} aria-hidden="true">
            {POOL_IN.flat().map((v, k) => (
              <span class="cvl-cell" data-cvl-in={String(k)} data-cvl-state={st.cells[k]}>
                {v}
              </span>
            ))}
          </div>
        </div>
        <div class="cvl-block">
          <p class="cvl-heading">Max-pooled output</p>
          {SIZES.flatMap(f =>
            STRIDES.map(s => {
              const { n } = POOL_META[`${f}:${s}`]
              return (
                <fieldset
                  class="cvl-grid cvl-grid--pool cvl-outputs"
                  style={`--cvl-n:${n}`}
                  data-cvl-config={`${f}:${s}`}
                  hidden={f !== f0 || s !== s0}
                >
                  <legend class="cvl-sr">{`Output windows for F = ${f}, S = ${s}`}</legend>
                  {Array.from({ length: n * n }, (_, i) => (
                    <label class="cvl-cell cvl-pick">
                      <input
                        type="radio"
                        name={`${id}-win-${f}-${s}`}
                        value={String(i)}
                        checked={i === 0}
                        aria-label={`Window ${i + 1}, rows ${POOL[`${f}:${s}:${i}`].rows}, columns ${POOL[`${f}:${s}:${i}`].cols}, max ${poolOut(f, s, i)}`}
                        data-cvl-window
                      />
                      <span>{poolOut(f, s, i)}</span>
                    </label>
                  ))}
                </fieldset>
              )
            }),
          )}
        </div>
      </div>
      <dl class="cvl-readout" aria-live="polite">
        <div>
          <dt>
            <span>
              <MathText text="Output size $\lfloor (W_1-F)/S\rfloor+1$" mathClass="cvl-math" />
            </span>
          </dt>
          <dd data-cvl-out="w2">{POOL_META[`${f0}:${s0}`].w2}</dd>
        </div>
        <div>
          <dt>Window rows, columns</dt>
          <dd data-cvl-out="span">{`${st.rows}, ${st.cols}`}</dd>
        </div>
        <div>
          <dt>Window max</dt>
          <dd data-cvl-out="max">{st.max}</dd>
        </div>
        <div>
          <dt>Inputs no window reads</dt>
          <dd data-cvl-out="skipped">{String(POOL_META[`${f0}:${s0}`].skipped)}</dd>
        </div>
      </dl>
    </>
  )
}

const ConnectivityView = ({ id }: { id: string }) => {
  const m0: Mode = 'shared'
  const u0 = 5
  const st = CONN[`${m0}:${u0}`]
  return (
    <>
      <div class="cvl-controls">
        <Segments
          name={`${id}-mode`}
          legend="Connection"
          options={MODES}
          value={m0}
          data="data-cvl-mode"
        />
      </div>
      {MODES.map(m => (
        <p class="cvl-rule" data-cvl-rule={m.id}>
          <MathText text={m.rule} mathClass="cvl-math" />
        </p>
      ))}
      <div class="cvl-stage">
        <div class="cvl-block">
          <p class="cvl-heading">Input, 6 × 6: weight id per input</p>
          <div class="cvl-grid cvl-grid--conn" style={`--cvl-n:${IN}`} aria-hidden="true">
            {st.ids.map((v, k) => (
              <span class="cvl-cell" data-cvl-in={String(k)} data-cvl-state={v === '' ? '0' : 'w'}>
                {v}
              </span>
            ))}
          </div>
        </div>
        <div class="cvl-block">
          <p class="cvl-heading">Units, 4 × 4</p>
          <fieldset class="cvl-grid cvl-grid--conn cvl-outputs" style={`--cvl-n:${HID}`}>
            <legend class="cvl-sr">Selected unit</legend>
            {Array.from({ length: HID * HID }, (_, u) => (
              <label class="cvl-cell cvl-pick">
                <input
                  type="radio"
                  name={`${id}-unit`}
                  value={String(u)}
                  checked={u === u0}
                  aria-label={`Unit ${u + 1}, row ${Math.floor(u / HID)}, column ${u % HID}`}
                  data-cvl-unit
                />
                <span>{u + 1}</span>
              </label>
            ))}
          </fieldset>
        </div>
      </div>
      <dl class="cvl-readout" aria-live="polite">
        <div>
          <dt>Weights in the layer</dt>
          <dd data-cvl-out="params">{st.params}</dd>
        </div>
        <div>
          <dt>Inputs this unit reads</dt>
          <dd data-cvl-out="reads">{st.reads}</dd>
        </div>
        <div>
          <dt>Weight ids this unit uses</dt>
          <dd data-cvl-out="range">{st.range}</dd>
        </div>
        <div>
          <dt>Units that use these weights</dt>
          <dd data-cvl-out="share">{st.share}</dd>
        </div>
      </dl>
    </>
  )
}

const ConvLocalityImpl: QuartzMdxComponent<Props> = ({ caption, view = 'pooling' }) => {
  const id = `cvl-${++instance}`
  const v0: View = view === 'connectivity' ? 'connectivity' : 'pooling'
  return (
    <figure
      class="conv-locality"
      data-conv-locality
      data-cvl-view={v0}
      data-cvl-mode={v0 === 'connectivity' ? 'shared' : undefined}
      data-cvl-states={JSON.stringify(v0 === 'pooling' ? POOL : CONN)}
      data-cvl-meta={v0 === 'pooling' ? JSON.stringify(POOL_META) : undefined}
    >
      {v0 === 'pooling' ? <PoolingView id={id} /> : <ConnectivityView id={id} />}
      {caption ? (
        <figcaption class="cvl-caption">
          <MathText text={caption} mathClass="cvl-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ConvLocalityComponent = ConvLocalityImpl as QuartzMdxComponent<Props>
ConvLocalityComponent.css = style
ConvLocalityComponent.afterDOMLoaded = script

export const ConvLocality = registerMdxComponent('ConvLocality', ConvLocalityComponent)

export default (() => ConvLocality) satisfies (opts: undefined) => QuartzMdxComponent<Props>
