// Shared by the server render and the inline script, so the first paint and every update agree.

export type AccDType = 'bf16' | 'fp16' | 'fp32' | 'int8' | 'int4'
export type AccNumKey = 'dm' | 'nl' | 'nh' | 'sl' | 'bs'
export type AccState = Record<AccNumKey, number> & { dt: AccDType }

export const ACC_VALUES: Record<AccNumKey, number[]> = {
  dm: [128, 256, 512, 1024, 2048, 4096, 8192],
  nl: [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96],
  nh: [1, 2, 4, 8, 16, 32, 64],
  sl: [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
  bs: [1, 2, 4, 8, 16, 32, 64, 128, 256],
}

export const ACC_DTYPE_BYTES: Record<AccDType, number> = {
  bf16: 2,
  fp16: 2,
  fp32: 4,
  int8: 1,
  int4: 0.5,
}

export const ACC_INITIAL: AccState = { dm: 4096, nl: 32, nh: 32, sl: 4096, bs: 1, dt: 'bf16' }

// Weights stay BF16 while the KV dtype changes.
const WEIGHT_BYTES = 2

export type AccMetrics = {
  dh: number
  params: number
  flops: number
  kvtok: number
  kvtotal: number
  ratio: number
  paramBytes: number
  ffnBytes: number
}

export const accCompute = (s: AccState): AccMetrics => {
  const bytes = ACC_DTYPE_BYTES[s.dt]
  // Q, K, V and O projections: 4 d^2 weights per layer.
  const params = 4 * s.nl * s.dm * s.dm
  // One decoded token: 2 FLOPs per projection weight, plus QK^T and AV over L cached tokens
  // (2 L d each) in every layer.
  const flops = 2 * params + 4 * s.nl * s.sl * s.dm
  const kvtok = 2 * s.dm * bytes
  const kvtotal = s.bs * s.sl * s.nl * kvtok
  const paramBytes = params * WEIGHT_BYTES
  // A 4x MLP: two d x 4d matrices per layer.
  const ffnBytes = 8 * s.nl * s.dm * s.dm * WEIGHT_BYTES
  return {
    dh: s.dm / s.nh,
    params,
    flops,
    kvtok,
    kvtotal,
    ratio: kvtotal / paramBytes,
    paramBytes,
    ffnBytes,
  }
}

const sig3 = (v: number): string => v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)

/** A count as TeX: integers below 10^4, else three significant figures times a power of ten. */
export const accCountTex = (n: number): string => {
  if (n < 1e4) return Number.isInteger(n) ? String(n) : n.toFixed(2)
  const exp = Math.floor(Math.log10(n))
  const mantissa = n / 10 ** exp
  return `${mantissa.toFixed(2)}\\times 10^{${exp}}`
}

/** Bytes as TeX in binary units, which is what the 1024 steps measure. */
export const accBytesTex = (n: number): string => {
  if (n < 1024) return `${Number.isInteger(n) ? n : n.toFixed(1)}\\,\\text{B}`
  const units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${sig3(v)}\\,\\text{${units[i]}}`
}

export const accRatioTex = (r: number): string => `${r >= 10 ? r.toFixed(0) : r.toFixed(2)}\\times`

export const accMetricTex = (k: string, m: AccMetrics): string => {
  switch (k) {
    case 'dh':
      return accCountTex(m.dh)
    case 'params':
      return accCountTex(m.params)
    case 'flops':
      return accCountTex(m.flops)
    case 'kvtok':
      return accBytesTex(m.kvtok)
    case 'kvtotal':
      return accBytesTex(m.kvtotal)
    case 'ratio':
      return accRatioTex(m.ratio)
    default:
      return '\\text{–}'
  }
}

/** Signed change between two metric states, empty when nothing moved. */
export const accDeltaTex = (k: string, cur: AccMetrics, prev: AccMetrics): string => {
  const a = cur[k as keyof AccMetrics]
  const b = prev[k as keyof AccMetrics]
  if (a === b || !Number.isFinite(b)) return ''
  const sign = a > b ? '+' : '-'
  const diff = Math.abs(a - b)
  const body =
    k === 'kvtok' || k === 'kvtotal'
      ? accBytesTex(diff)
      : k === 'ratio'
        ? accRatioTex(diff)
        : accCountTex(diff)
  return `${sign}${body}`
}
