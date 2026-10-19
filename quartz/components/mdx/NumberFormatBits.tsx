import { renderInlineMath, MathText } from '../../util/math-text'
import {
  BF16,
  bitString,
  E2M1,
  E4M3,
  E5M2,
  FP16,
  FP32,
  isSubnormal,
  maxFinite,
  minNormal,
  minSubnormal,
  roundHalfEven,
  roundToFormat,
  type FloatFormat,
  type Overflow,
} from '../../util/minifloat'
//@ts-ignore
import script from '../scripts/number-format-bits.inline'
import style from '../styles/numberFormatBits.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; value?: string; format?: string }

type FloatRow = { id: string; label: string; fmt: FloatFormat; overflow: Overflow }
type Row = FloatRow | { id: 'int8'; label: string; fmt?: undefined }

const ROWS: readonly Row[] = [
  { id: 'fp32', label: 'FP32', fmt: FP32, overflow: 'ieee' },
  { id: 'bf16', label: 'BF16', fmt: BF16, overflow: 'ieee' },
  { id: 'fp16', label: 'FP16', fmt: FP16, overflow: 'ieee' },
  { id: 'e5m2', label: 'FP8 E5M2', fmt: E5M2, overflow: 'saturate' },
  { id: 'e4m3', label: 'FP8 E4M3', fmt: E4M3, overflow: 'saturate' },
  { id: 'e2m1', label: 'FP4 E2M1', fmt: E2M1, overflow: 'saturate' },
  { id: 'int8', label: 'INT8' },
]

// The note's symmetric int8 example: S = 1/127, Z = 0, levels [-127, 127].
const INT8_Q = 127
const INT8_S = 1 / INT8_Q

const VALUES: readonly { id: string; x: number; tex: string }[] = [
  { id: 'p03', x: 0.3, tex: '0.3' },
  { id: 'pi', x: Math.PI, tex: '\\pi' },
  { id: 'm300', x: -300, tex: '-300' },
  { id: 'k1', x: 1000, tex: '1000' },
  { id: 'k70', x: 70000, tex: '70\\,000' },
  { id: 'tiny', x: 1e-5, tex: '10^{-5}' },
]

type Field = 's' | 'e' | 'm'
const fieldsOf = (row: Row): Field[] =>
  row.fmt
    ? ['s', ...Array<Field>(row.fmt.ebits).fill('e'), ...Array<Field>(row.fmt.mbits).fill('m')]
    : ['s', ...Array<Field>(7).fill('m')]

const trimZeros = (s: string) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s)
const sci = (x: number, digits: number) => {
  const [m, e] = x.toExponential(digits - 1).split('e')
  return [trimZeros(m), Number(e)] as const
}
const plain = (x: number) => Math.abs(x) >= 1e-3 && Math.abs(x) < 1e6
const texNum = (x: number, digits = 8): string => {
  if (!Number.isFinite(x)) return x < 0 ? '-\\infty' : '\\infty'
  if (x === 0) return '0'
  if (plain(x)) return trimZeros(x.toPrecision(digits))
  const [m, e] = sci(x, Math.min(digits, 5))
  return `${m}\\times10^{${e}}`
}
const textNum = (x: number): string => {
  if (!Number.isFinite(x)) return x < 0 ? '−∞' : '∞'
  if (x === 0) return '0'
  const s = plain(x) ? trimZeros(x.toPrecision(8)) : sci(x, 5).join('e')
  return s.replace('-', '−')
}

type Encoding = {
  bits: string
  value: number
  eq: string
  outcome: string
  max: number
  minNormal: number
  minSub: number | null
}

const encodeFloat = (x: number, row: FloatRow): Encoding => {
  const f = row.fmt
  const r = roundToFormat(x, f, row.overflow)
  const B = f.bias
  let eq: string
  if (r.outcome === 'infinity')
    eq = `e=${r.exp}=2^{${f.ebits}}-1,\\ f=0 \\;\\Rightarrow\\; ${texNum(r.value)}`
  else if (r.value === 0) eq = `e=0,\\ f=0 \\;\\Rightarrow\\; ${r.sign ? '-' : '+'}0`
  else if (isSubnormal(r))
    eq = `(-1)^{${r.sign}}\\,\\frac{${r.frac}}{2^{${f.mbits}}}\\,2^{1-${B}} = ${texNum(r.value)}`
  else
    eq = `(-1)^{${r.sign}}\\left(1+\\frac{${r.frac}}{2^{${f.mbits}}}\\right)2^{${r.exp}-${B}} = ${texNum(r.value)}`
  const outcome = {
    exact: 'Exact',
    rounded: 'Rounded to nearest, ties to even',
    zero: 'Underflow, rounds to zero',
    infinity: 'Overflow, rounds to infinity',
    saturated: 'Overflow, saturates to the largest finite value',
  }[r.outcome]
  return {
    bits: bitString(r, f),
    value: r.value,
    eq,
    outcome: isSubnormal(r) && r.outcome === 'rounded' ? `${outcome}, subnormal` : outcome,
    max: maxFinite(f),
    minNormal: minNormal(f),
    minSub: minSubnormal(f),
  }
}

const encodeInt8 = (x: number): Encoding => {
  const raw = roundHalfEven(x / INT8_S)
  const q = Math.min(INT8_Q, Math.max(-INT8_Q, raw))
  const value = q * INT8_S
  return {
    bits: (q & 0xff).toString(2).padStart(8, '0'),
    value,
    eq: `S\\,x_q=\\frac{${q}}{${INT8_Q}} ${value === x ? '=' : '\\approx'} ${texNum(value, 6)}`,
    outcome:
      raw !== q
        ? `Clipped to x_q = ${q < 0 ? '−' : ''}${INT8_Q}`
        : q === 0 && x !== 0
          ? 'Underflow, rounds to zero'
          : 'Rounded to nearest, ties to even',
    max: INT8_Q * INT8_S,
    minNormal: INT8_S,
    minSub: null,
  }
}

const encode = (x: number, row: Row) => (row.fmt ? encodeFloat(x, row) : encodeInt8(x))

const math = (tex: string, display = false) => renderInlineMath(tex, display)
const errorTex = (x: number, v: number) => {
  const abs = Math.abs(v - x)
  return {
    abs: texNum(abs, 3),
    rel: Number.isFinite(abs) ? `${texNum((abs / Math.abs(x)) * 100, 3)}\\%` : '\\infty',
  }
}

type Detail = Record<
  'eq' | 'hex' | 'dec' | 'abs' | 'rel' | 'outcome' | 'max' | 'minN' | 'minS',
  string
>
type ValueState = {
  bits: Record<string, string>
  dec: Record<string, string>
  detail: Record<string, Detail>
}

const STATES: Record<string, ValueState> = {}
for (const v of VALUES) {
  const state: ValueState = { bits: {}, dec: {}, detail: {} }
  for (const row of ROWS) {
    const enc = encode(v.x, row)
    const err = errorTex(v.x, enc.value)
    const width = enc.bits.length / 4
    state.bits[row.id] = enc.bits
    state.dec[row.id] = textNum(enc.value)
    state.detail[row.id] = {
      eq: math(enc.eq, true),
      hex: `0x${parseInt(enc.bits, 2).toString(16).toUpperCase().padStart(width, '0')}`,
      dec: math(texNum(enc.value)),
      abs: math(err.abs),
      rel: math(err.rel),
      outcome: enc.outcome,
      max: math(texNum(enc.max, 5)),
      minN: math(texNum(enc.minNormal, 5)),
      minS: enc.minSub === null ? 'none' : math(texNum(enc.minSub, 5)),
    }
  }
  STATES[v.id] = state
}

const DETAIL_ROWS: readonly [keyof Detail, string][] = [
  ['hex', 'Stored bits'],
  ['dec', 'Decoded value'],
  ['abs', 'Absolute error'],
  ['rel', 'Relative error'],
  ['outcome', 'Conversion'],
  ['max', 'Largest finite'],
  ['minN', 'Smallest normal'],
  ['minS', 'Smallest subnormal'],
]

let instance = 0

const NumberFormatBitsImpl: QuartzMdxComponent<Props> = ({
  caption,
  value = 'pi',
  format = 'fp16',
}) => {
  const id = `nfb-${++instance}`
  const v0 = VALUES.some(v => v.id === value) ? value : 'pi'
  const f0 = ROWS.some(r => r.id === format) ? format : 'fp16'
  const s0 = STATES[v0]
  const d0 = s0.detail[f0]

  return (
    <figure
      class="number-format-bits"
      data-number-format-bits
      data-nfb-states={JSON.stringify(STATES)}
    >
      <fieldset class="nfb-segments">
        <legend class="nfb-legend">
          <MathText text="Input $x$" mathClass="nfb-math" />
        </legend>
        {VALUES.map(v => (
          <label class="nfb-segment">
            <input
              type="radio"
              name={`${id}-value`}
              value={v.id}
              checked={v.id === v0}
              data-nfb-value-input
            />
            <span dangerouslySetInnerHTML={{ __html: math(v.tex) }} />
          </label>
        ))}
      </fieldset>

      <fieldset class="nfb-rows">
        <legend class="nfb-rows-legend">Format: select a row to decode it</legend>
        {ROWS.map(row => (
          <label class="nfb-row" data-nfb-row={row.id}>
            <input
              type="radio"
              name={`${id}-format`}
              value={row.id}
              checked={row.id === f0}
              data-nfb-format-input
            />
            <span class="nfb-name">{row.label}</span>
            <span class="nfb-bits" aria-hidden="true">
              {fieldsOf(row).map((field, i) => (
                <span class="nfb-bit" data-nfb-field={field} data-nfb-on={s0.bits[row.id][i]} />
              ))}
            </span>
            <span class="nfb-dec" data-nfb-dec>
              {s0.dec[row.id]}
            </span>
          </label>
        ))}
      </fieldset>

      <ul class="nfb-key" aria-label="Bit fields">
        <li>
          <span class="nfb-swatch" data-nfb-field="s" />
          sign
        </li>
        <li>
          <span class="nfb-swatch" data-nfb-field="e" />
          exponent
        </li>
        <li>
          <span class="nfb-swatch" data-nfb-field="m" />
          fraction (integer for INT8)
        </li>
        <li>
          <span class="nfb-swatch" data-nfb-field="m" data-nfb-on="1" />
          bit set
        </li>
      </ul>

      <div class="nfb-detail">
        <div class="nfb-eq" data-nfb-out="eq" dangerouslySetInnerHTML={{ __html: d0.eq }} />
        <dl class="nfb-readout">
          {DETAIL_ROWS.map(([key, label]) => (
            <div>
              <dt>{label}</dt>
              <dd data-nfb-out={key} dangerouslySetInnerHTML={{ __html: d0[key] }} />
            </div>
          ))}
        </dl>
      </div>

      {caption ? (
        <figcaption class="nfb-caption">
          <MathText text={caption} mathClass="nfb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const NumberFormatBitsComponent = NumberFormatBitsImpl as QuartzMdxComponent<Props>
NumberFormatBitsComponent.css = style
NumberFormatBitsComponent.afterDOMLoaded = script

export const NumberFormatBits = registerMdxComponent('NumberFormatBits', NumberFormatBitsComponent)

export default (() => NumberFormatBits) satisfies (opts: undefined) => QuartzMdxComponent<Props>
