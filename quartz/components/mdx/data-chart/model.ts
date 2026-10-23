export type Value = number | null
export type SeriesSpec = {
  name: string
  // values align with `x`; points carry their own x; terms sample y = sum of c * x^p on [from, to].
  values?: Value[]
  points?: [number, number][]
  terms?: [number, number][]
  from?: number
  to?: number
}
export type MarkSpec = { x: number; xlabel?: string; ylabel?: string }
export type ProbeSpec = {
  x: number
  step?: number
  xname?: string
  yname?: string
  falling?: string
  flat?: string
  rising?: string
  state?: string
}
export type Scale = 'linear' | 'log'
export type XScale = 'linear' | 'log2' | 'log10'

// MDX hands JSON-valued props over as strings, so every structured prop also accepts a string.
export type Json<T> = T | string
export type Props = {
  caption?: string
  label?: string
  mode?: 'bar' | 'line'
  x?: Json<(string | number)[]>
  xscale?: XScale
  xlabel?: string
  xname?: string
  xmin?: number | string
  xmax?: number | string
  xticks?: Json<number[]>
  yscale?: Scale
  ylabel?: string
  ymin?: number | string
  ymax?: number | string
  yticks?: Json<number[]>
  unit?: string
  decimals?: number | string
  series: Json<SeriesSpec[]>
  labels?: Json<number[]> | 'all' | 'last' | 'none'
  ratio?: string
  active?: number | string
  marks?: Json<MarkSpec[]>
  probe?: Json<ProbeSpec>
}

export const VIEW_W = 440
export const X0 = 46
export const X1 = 428
export const Y_TOP = 30
export const Y0 = 228
export const VIEW_H = 270
export const BAR_MAX = 24
export const BAR_GAP = 2
export const SAMPLES = 120
export const MAX_SLOTS = 2

export const json = <T>(name: string, v: Json<T> | undefined, fallback: T): T => {
  if (v === undefined || v === null || v === '') return fallback
  if (typeof v !== 'string') return v
  try {
    return JSON.parse(v) as T
  } catch {
    throw new Error(`DataChart: prop "${name}" is not valid JSON`)
  }
}
export const num = (v: number | string | undefined): number | undefined =>
  v === undefined || v === '' ? undefined : Number(v)
export const r1 = (n: number) => Math.round(n * 10) / 10

export const fwd = (scale: XScale | Scale, v: number) =>
  scale === 'log2' ? Math.log2(v) : scale === 'log10' || scale === 'log' ? Math.log10(v) : v

export const SI: [number, string][] = [
  [1e9, 'G'],
  [1e6, 'M'],
  [1e3, 'k'],
]
export const trim = (v: number) => String(Number(v.toPrecision(6)))
export const siTick = (v: number) => {
  for (const [base, suffix] of SI) if (Math.abs(v) >= base) return `${trim(v / base)}${suffix}`
  return trim(v)
}

export const niceTicks = (lo: number, hi: number, target = 5): number[] => {
  const raw = (hi - lo) / target
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step)
    out.push(Number(v.toPrecision(12)))
  return out
}

export const expTex = (v: number) => {
  const e = Math.log2(v)
  const k = Math.round(e)
  return `2^{${Math.abs(e - k) < 1e-6 ? k : e.toFixed(2)}}`
}

export type Pt = { x: number; y: number }
export type Series = { name: string; slot: number; pts: Pt[]; analytic: boolean; spec: SeriesSpec }
export type Position = {
  x: number
  cx: number
  lo: number
  hi: number
  tex: string
  text: string
  values: Value[]
}

export const evalTerms = (terms: [number, number][], x: number) =>
  terms.reduce((a, [p, c]) => a + c * x ** p, 0)

export const formatRatio = (r: number) =>
  `${r >= 100 ? r.toFixed(0) : r >= 10 ? r.toFixed(1) : r.toFixed(2)}×`
