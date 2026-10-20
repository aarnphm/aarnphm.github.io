// Binary floating-point formats narrower than a double: IEEE binary32/binary16, bfloat16 and the
// OCP 8-, 6- and 4-bit element formats. Rounding is to nearest, ties to even. Every value here is
// computed in double precision, which holds each of these formats' values and quanta exactly.

/** `ieee`: all-ones exponent holds infinity and NaN. `nan-only`: only the all-ones exponent and
 * fraction is NaN (FP8 E4M3). `none`: every encoding is finite (FP6, FP4). */
export type FloatSpecials = 'ieee' | 'nan-only' | 'none'

export type FloatFormat = { ebits: number; mbits: number; bias: number; specials: FloatSpecials }

export const FP32: FloatFormat = { ebits: 8, mbits: 23, bias: 127, specials: 'ieee' }
export const BF16: FloatFormat = { ebits: 8, mbits: 7, bias: 127, specials: 'ieee' }
export const FP16: FloatFormat = { ebits: 5, mbits: 10, bias: 15, specials: 'ieee' }
export const E5M2: FloatFormat = { ebits: 5, mbits: 2, bias: 15, specials: 'ieee' }
export const E4M3: FloatFormat = { ebits: 4, mbits: 3, bias: 7, specials: 'nan-only' }
export const E3M2: FloatFormat = { ebits: 3, mbits: 2, bias: 3, specials: 'none' }
export const E2M3: FloatFormat = { ebits: 2, mbits: 3, bias: 1, specials: 'none' }
export const E2M1: FloatFormat = { ebits: 2, mbits: 1, bias: 1, specials: 'none' }

export const totalBits = (f: FloatFormat) => 1 + f.ebits + f.mbits

/** Unbiased exponent of the largest finite binade. */
export const emax = (f: FloatFormat) => 2 ** f.ebits - 1 - f.bias - (f.specials === 'ieee' ? 1 : 0)

export const maxFinite = (f: FloatFormat) => {
  // E4M3 gives up only the top fraction code of its top binade to NaN.
  const fracCodes = 2 ** f.mbits - (f.specials === 'nan-only' ? 2 : 1)
  return (1 + fracCodes / 2 ** f.mbits) * 2 ** emax(f)
}
export const minNormal = (f: FloatFormat) => 2 ** (1 - f.bias)
export const minSubnormal = (f: FloatFormat) => 2 ** (1 - f.bias - f.mbits)

export const floorLog2 = (a: number) => {
  let e = Math.floor(Math.log2(a))
  if (2 ** e > a) e -= 1
  else if (2 ** (e + 1) <= a) e += 1
  return e
}

export const roundHalfEven = (q: number) => {
  const f = Math.floor(q)
  const d = q - f
  if (d > 0.5) return f + 1
  if (d < 0.5) return f
  return f % 2 === 0 ? f : f + 1
}

/** What finite overflow does: IEEE conversion rounds to infinity, `saturate` clamps to the
 * largest finite value with the input's sign. Formats without infinity always saturate. */
export type Overflow = 'ieee' | 'saturate'

export type Rounded = {
  value: number
  sign: number
  exp: number
  frac: number
  /** `zero`: a nonzero input rounded to zero. `saturated`: clamped to the largest finite value. */
  outcome: 'exact' | 'rounded' | 'zero' | 'infinity' | 'saturated'
}

export const isSubnormal = (r: Rounded) => r.exp === 0 && r.value !== 0

export const roundToFormat = (x: number, f: FloatFormat, overflow: Overflow = 'ieee'): Rounded => {
  const sign = x < 0 || Object.is(x, -0) ? 1 : 0
  const a = Math.abs(x)
  const allOnes = 2 ** f.ebits - 1
  if (a === 0) return { value: sign ? -0 : 0, sign, exp: 0, frac: 0, outcome: 'exact' }
  const emin = 1 - f.bias
  const e = Math.max(floorLog2(a), emin)
  const quantum = 2 ** (e - f.mbits)
  let mag = roundHalfEven(a / quantum) * quantum
  const max = maxFinite(f)
  if (mag > max) {
    if (overflow === 'ieee' && f.specials === 'ieee')
      return {
        value: sign ? -Infinity : Infinity,
        sign,
        exp: allOnes,
        frac: 0,
        outcome: 'infinity',
      }
    mag = max
    return { ...encode(mag, sign, f), outcome: 'saturated' }
  }
  if (mag === 0) return { value: sign ? -0 : 0, sign, exp: 0, frac: 0, outcome: 'zero' }
  return { ...encode(mag, sign, f), outcome: mag === a ? 'exact' : 'rounded' }
}

const encode = (mag: number, sign: number, f: FloatFormat) => {
  const emin = 1 - f.bias
  const e = floorLog2(mag)
  const value = sign ? -mag : mag
  if (e < emin) return { value, sign, exp: 0, frac: mag / 2 ** (emin - f.mbits) }
  return { value, sign, exp: e + f.bias, frac: mag / 2 ** (e - f.mbits) - 2 ** f.mbits }
}

/** Sign, exponent and fraction fields as a bit string, most significant bit first. */
export const bitString = (r: Pick<Rounded, 'sign' | 'exp' | 'frac'>, f: FloatFormat) =>
  `${r.sign}${r.exp.toString(2).padStart(f.ebits, '0')}${r.frac.toString(2).padStart(f.mbits, '0')}`

/** Every nonnegative finite value of a format, ascending. */
export const levels = (f: FloatFormat): number[] => {
  const out: number[] = []
  const max = maxFinite(f)
  for (let exp = 0; exp < 2 ** f.ebits; exp++)
    for (let frac = 0; frac < 2 ** f.mbits; frac++) {
      const v =
        exp === 0
          ? (frac / 2 ** f.mbits) * 2 ** (1 - f.bias)
          : (1 + frac / 2 ** f.mbits) * 2 ** (exp - f.bias)
      if (v <= max) out.push(v)
    }
  return out
}

export type MXBlock = {
  /** Shared exponent: floor(log2 max|V|) - emax of the element format. */
  shared: number
  scale: number
  /** Element values P_i before scaling back. */
  elements: number[]
  dequantized: number[]
  clipped: number
}

/**
 * The example conversion in Rouhani et al. (2023), Algorithm 1: the shared E8M0 exponent maps the
 * block's largest magnitude into the element format's largest binade, then each element rounds
 * to nearest even and clamps finite overflow. An all-zero block takes scale 1.
 */
export const quantizeMXBlock = (block: number[], f: FloatFormat): MXBlock => {
  const amax = Math.max(...block.map(Math.abs))
  const shared = amax === 0 ? 0 : floorLog2(amax) - emax(f)
  const scale = 2 ** shared
  let clipped = 0
  const elements = block.map(v => {
    const r = roundToFormat(v / scale, f, 'saturate')
    if (r.outcome === 'saturated') clipped += 1
    return r.value
  })
  return { shared, scale, elements, dequantized: elements.map(p => p * scale), clipped }
}
