/**
 * Transcendental functions that give the same bits on every machine.
 *
 * `Math.sin` and friends are only approximations in the spec, and engines
 * disagree in the last bit: V8 on arm64 and on x64 already differ, and
 * Safari and Firefox are different again. The generator makes decisions on
 * these values (does this leaf fit, which size step is this), so one bit
 * can move a leaf, and the same seed would draw different pictures for
 * different readers.
 *
 * These are ports of fdlibm, the library V8 itself compiles from C. In
 * JavaScript they use only + - * / and comparisons, which IEEE 754 makes
 * exact everywhere, so the result is the same on every engine and CPU.
 * Accuracy is fdlibm's: under one ulp, except `pow` (a few ulp).
 */

const view = new DataView(new ArrayBuffer(8))

/** The high 32 bits of a double: sign, exponent, top of the mantissa. */
function high(x: number): number {
  view.setFloat64(0, x)
  return view.getInt32(0)
}

function low(x: number): number {
  view.setFloat64(0, x)
  return view.getUint32(4)
}

function withHigh(x: number, hi: number): number {
  view.setFloat64(0, x)
  view.setInt32(0, hi)
  return view.getFloat64(0)
}

// sin, cos ---------------------------------------------------------------

const S1 = -1.66666666666666324348e-1
const S2 = 8.33333333332248946124e-3
const S3 = -1.98412698298579493134e-4
const S4 = 2.75573137070700676789e-6
const S5 = -2.50507602534068634195e-8
const S6 = 1.58969099521155010221e-10

/** sin on [-π/4, π/4], with x + y the argument split in two for precision. */
function kernelSin(x: number, y: number, hasTail: boolean): number {
  if ((high(x) & 0x7fffffff) < 0x3e400000) return x
  const z = x * x
  const v = z * x
  const r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))
  return hasTail ? x - (z * (0.5 * y - v * r) - y - v * S1) : x + v * (S1 + z * r)
}

const C1 = 4.16666666666666019037e-2
const C2 = -1.38888888888741095749e-3
const C3 = 2.48015872894767294178e-5
const C4 = -2.75573143513906633035e-7
const C5 = 2.0875723212981748279e-9
const C6 = -1.13596475577881948265e-11

function kernelCos(x: number, y: number): number {
  const ix = high(x) & 0x7fffffff
  if (ix < 0x3e400000) return 1
  const z = x * x
  const r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))))
  if (ix < 0x3fd33333) return 1 - (0.5 * z - (z * r - x * y))
  const qx = ix > 0x3fe90000 ? 0.28125 : withHigh(0, ix - 0x00200000)
  const hz = 0.5 * z - qx
  return 1 - qx - (hz - (z * r - x * y))
}

const INV_PIO2 = 6.36619772367581382433e-1
const PIO2_1 = 1.57079632673412561417
const PIO2_1T = 6.07710050650619224932e-11
const PIO2_2 = 6.0771005063039659766e-11
const PIO2_2T = 2.02226624879595063154e-21
const PIO2_3 = 2.0222662487111664558e-21
const PIO2_3T = 8.47842766036889956997e-32

/**
 * x − n·π/2 as a head and tail, with π/2 carried to 118 bits so the
 * remainder stays exact. Returns n, or null past 2^19·π/2, where this
 * reduction runs out of bits. No ornament angle gets near that.
 */
function reducePio2(x: number, out: [number, number]): number | null {
  const ix = high(x) & 0x7fffffff
  if (ix > 0x413921fb) return null
  const t = Math.abs(x)
  const n = Math.floor(t * INV_PIO2 + 0.5)
  let r = t - n * PIO2_1
  let w = n * PIO2_1T
  const j = ix >> 20
  let y0 = r - w
  if (j - ((high(y0) >> 20) & 0x7ff) > 16) {
    // Near a multiple of π/2 the first subtraction cancelled; take more of π/2.
    let u = r
    w = n * PIO2_2
    r = u - w
    w = n * PIO2_2T - (u - r - w)
    y0 = r - w
    if (j - ((high(y0) >> 20) & 0x7ff) > 49) {
      u = r
      w = n * PIO2_3
      r = u - w
      w = n * PIO2_3T - (u - r - w)
      y0 = r - w
    }
  }
  const y1 = r - y0 - w
  if (x < 0) {
    out[0] = -y0
    out[1] = -y1
    return -n
  }
  out[0] = y0
  out[1] = y1
  return n
}

const parts: [number, number] = [0, 0]

export function sin(x: number): number {
  const ix = high(x) & 0x7fffffff
  if (ix <= 0x3fe921fb) return kernelSin(x, 0, false)
  if (ix >= 0x7ff00000) return NaN
  const n = reducePio2(x, parts)
  if (n === null) return Math.sin(x)
  const [a, b] = parts
  switch (n & 3) {
    case 0:
      return kernelSin(a, b, true)
    case 1:
      return kernelCos(a, b)
    case 2:
      return -kernelSin(a, b, true)
    default:
      return -kernelCos(a, b)
  }
}

export function cos(x: number): number {
  const ix = high(x) & 0x7fffffff
  if (ix <= 0x3fe921fb) return kernelCos(x, 0)
  if (ix >= 0x7ff00000) return NaN
  const n = reducePio2(x, parts)
  if (n === null) return Math.cos(x)
  const [a, b] = parts
  switch (n & 3) {
    case 0:
      return kernelCos(a, b)
    case 1:
      return -kernelSin(a, b, true)
    case 2:
      return -kernelCos(a, b)
    default:
      return kernelSin(a, b, true)
  }
}

// atan, atan2 ------------------------------------------------------------

const ATAN_HI = [
  4.63647609000806093515e-1, 7.85398163397448278999e-1, 9.82793723247329054082e-1,
  1.570796326794896558,
]
const ATAN_LO = [
  2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17,
  6.12323399573676603587e-17,
]
const AT = [
  3.33333333333329318027e-1, -1.99999999998764832476e-1, 1.42857142725034663711e-1,
  -1.1111110405462355788e-1, 9.09088713343650656196e-2, -7.69187620504482999495e-2,
  6.66107313738753120669e-2, -5.83357013379057348645e-2, 4.97687799461593236017e-2,
  -3.6531572744216915527e-2, 1.62858201153657823623e-2,
]

export function atan(x: number): number {
  const hx = high(x)
  const ix = hx & 0x7fffffff
  if (ix >= 0x44100000) {
    if (x !== x) return x
    return hx > 0 ? ATAN_HI[3] + ATAN_LO[3] : -ATAN_HI[3] - ATAN_LO[3]
  }
  let id: number
  if (ix < 0x3fdc0000) {
    if (ix < 0x3e200000) return x
    id = -1
  } else {
    x = Math.abs(x)
    if (ix < 0x3ff30000) {
      if (ix < 0x3fe60000) {
        id = 0
        x = (2 * x - 1) / (2 + x)
      } else {
        id = 1
        x = (x - 1) / (x + 1)
      }
    } else if (ix < 0x40038000) {
      id = 2
      x = (x - 1.5) / (1 + 1.5 * x)
    } else {
      id = 3
      x = -1 / x
    }
  }
  const z = x * x
  const w = z * z
  const s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))))
  const s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))))
  if (id < 0) return x - x * (s1 + s2)
  const r = ATAN_HI[id] - (x * (s1 + s2) - ATAN_LO[id] - x)
  return hx < 0 ? -r : r
}

const PI = 3.141592653589793116
const PI_LO = 1.2246467991473531772e-16
const PI_O_2 = 1.570796326794896558
const PI_O_4 = 7.85398163397448279e-1

export function atan2(y: number, x: number): number {
  if (x !== x || y !== y) return NaN
  if (x === 1) return atan(y)
  const hx = high(x)
  const hy = high(y)
  // Which quadrant, read from the sign bits so that −0 counts as negative.
  const m = (hy < 0 ? 1 : 0) | (hx < 0 ? 2 : 0)
  if (y === 0) return m === 2 ? PI : m === 3 ? -PI : y
  if (x === 0) return hy < 0 ? -PI_O_2 : PI_O_2
  if (x === Infinity || x === -Infinity) {
    if (y === Infinity || y === -Infinity) return [PI_O_4, -PI_O_4, 3 * PI_O_4, -3 * PI_O_4][m]
    return [0, -0, PI, -PI][m]
  }
  if (y === Infinity || y === -Infinity) return hy < 0 ? -PI_O_2 : PI_O_2
  const k = ((hy & 0x7fffffff) - (hx & 0x7fffffff)) >> 20
  let z: number
  if (k > 60) z = PI_O_2 + 0.5 * PI_LO
  else if (hx < 0 && k < -60) z = 0
  else z = atan(Math.abs(y / x))
  switch (m) {
    case 0:
      return z
    case 1:
      return -z
    case 2:
      return PI - (z - PI_LO)
    default:
      return z - PI_LO - PI
  }
}

// exp, log, pow ----------------------------------------------------------

const LN2_HI = 6.9314718036912381649e-1
const LN2_LO = 1.90821492927058770002e-10
const INV_LN2 = 1.442695040888963387
const P1 = 1.66666666666666019037e-1
const P2 = -2.77777777770155933842e-3
const P3 = 6.61375632143793436117e-5
const P4 = -1.6533902205465251539e-6
const P5 = 4.13813679705723846039e-8
const TWO_M1000 = 9.3326361850321887899e-302

export function exp(x: number): number {
  const hx0 = high(x)
  const negative = hx0 < 0
  const hx = hx0 & 0x7fffffff
  if (hx >= 0x40862e42) {
    if (x !== x) return x
    if (x > 7.09782712893383973096e2) return Infinity
    if (x < -7.4513321910194110842e2) return 0
  }
  let hi = 0
  let lo = 0
  let k = 0
  if (hx > 0x3fd62e42) {
    // x = k·ln2 + r with |r| ≤ ln2/2, ln2 split so k·ln2 is exact.
    if (hx < 0x3ff0a2b2) {
      hi = negative ? x + LN2_HI : x - LN2_HI
      lo = negative ? -LN2_LO : LN2_LO
      k = negative ? -1 : 1
    } else {
      k = Math.trunc(INV_LN2 * x + (negative ? -0.5 : 0.5))
      hi = x - k * LN2_HI
      lo = k * LN2_LO
    }
    x = hi - lo
  } else if (hx < 0x3e300000) {
    return 1 + x
  }
  const t = x * x
  const c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
  if (k === 0) return 1 - ((x * c) / (c - 2) - x)
  const y = 1 - (lo - (x * c) / (2 - c) - hi)
  // Scale by 2^k by adding to the exponent field directly.
  if (k >= -1021) return withHigh(y, high(y) + (k << 20))
  return withHigh(y, high(y) + ((k + 1000) << 20)) * TWO_M1000
}

const TWO_54 = 1.8014398509481984e16
const LG1 = 6.66666666666673513e-1
const LG2 = 3.999999999940941908e-1
const LG3 = 2.857142874366239149e-1
const LG4 = 2.222219843214978396e-1
const LG5 = 1.818357216161805012e-1
const LG6 = 1.531383769920937332e-1
const LG7 = 1.479819860511658591e-1

export function log(x: number): number {
  let hx = high(x)
  let k = 0
  if (hx < 0x00100000) {
    if (((hx & 0x7fffffff) | low(x)) === 0) return -Infinity
    if (hx < 0) return NaN
    // Subnormal: scale up so the exponent field means something.
    k -= 54
    x *= TWO_54
    hx = high(x)
  }
  if (hx >= 0x7ff00000) return x + x
  k += (hx >> 20) - 1023
  hx &= 0x000fffff
  // Normalize x into [√2/2, √2) so f = x − 1 is small.
  const i0 = (hx + 0x95f64) & 0x100000
  x = withHigh(x, hx | (i0 ^ 0x3ff00000))
  k += i0 >> 20
  const f = x - 1
  const dk = k
  if ((0x000fffff & (2 + hx)) < 3) {
    if (f === 0) return k === 0 ? 0 : dk * LN2_HI + dk * LN2_LO
    const R = f * f * (0.5 - 0.33333333333333333 * f)
    return k === 0 ? f - R : dk * LN2_HI - (R - dk * LN2_LO - f)
  }
  const s = f / (2 + f)
  const z = s * s
  const w = z * z
  const t1 = w * (LG2 + w * (LG4 + w * LG6))
  const t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
  const R = t2 + t1
  if (((hx - 0x6147a) | (0x6b851 - hx)) > 0) {
    const hfsq = 0.5 * f * f
    return k === 0
      ? f - (hfsq - s * (hfsq + R))
      : dk * LN2_HI - (hfsq - (s * (hfsq + R) + dk * LN2_LO) - f)
  }
  return k === 0 ? f - s * (f - R) : dk * LN2_HI - (s * (f - R) - dk * LN2_LO - f)
}

export function pow(x: number, y: number): number {
  if (y === 0) return 1
  if (x !== x || y !== y) return NaN
  // Zeros and infinities have exact answers in the spec; every engine agrees.
  if (x === 0 || !Number.isFinite(x) || !Number.isFinite(y)) return Math.pow(x, y)
  if (x < 0) {
    if (!Number.isInteger(y)) return NaN
    const r = powPositive(-x, y)
    return y % 2 === 0 ? r : -r
  }
  return powPositive(x, y)
}

function withLowZero(x: number): number {
  view.setFloat64(0, x)
  view.setUint32(4, 0)
  return view.getFloat64(0)
}

const BP = [1, 1.5]
const DP_H = [0, 5.84962487220764160156e-1]
const DP_L = [0, 1.35003920212974897128e-8]
const TWO_53 = 9007199254740992
const L1 = 5.99999999999994648725e-1
const L2 = 4.28571428578550184252e-1
const L3 = 3.33333329818377432918e-1
const L4 = 2.72728123808534006489e-1
const L5 = 2.30660745775561754067e-1
const L6 = 2.06975017800338417784e-1
const LN2_POW = 6.93147180559945286227e-1
const LN2_POW_H = 6.93147182464599609375e-1
const LN2_POW_L = -1.90465429995776804525e-9
const OVT = 8.0085662595372944372e-17
const CP = 9.61796693925975554329e-1
const CP_H = 9.61796700954437255859e-1
const CP_L = -7.02846165095275826516e-9
const IVLN2 = 1.442695040888963387
const IVLN2_H = 1.44269502162933349609
const IVLN2_L = 1.92596299112661746887e-8

/**
 * fdlibm's pow for finite x > 0 and finite y ≠ 0: log2(x) carried in two
 * halves to ~70 bits, times y, then 2^ of that. Plain e^(y·ln x) would
 * lose y·ln x's rounding error, tens of ulp for larger results.
 */
function powPositive(x: number, y: number): number {
  if (y === 1) return x
  if (y === -1) return 1 / x
  if (y === 2) return x * x
  if (y === 0.5) return Math.sqrt(x)
  let ix = high(x)
  const iy = high(y) & 0x7fffffff
  const yPositive = y > 0
  let t1: number
  let t2: number
  if (iy > 0x41e00000) {
    // |y| > 2^31: anything but x ≈ 1 over- or underflows.
    if (ix < 0x3fefffff) return yPositive ? 0 : Infinity
    if (ix > 0x3ff00000) return yPositive ? Infinity : 0
    const t = x - 1
    const w = t * t * (0.5 - t * (0.3333333333333333333333 - t * 0.25))
    const u = IVLN2_H * t
    const v = t * IVLN2_L - w * IVLN2
    t1 = withLowZero(u + v)
    t2 = v - (t1 - u)
  } else {
    let ax = x
    let n = 0
    if (ix < 0x00100000) {
      ax *= TWO_53
      n -= 53
      ix = high(ax)
    }
    n += (ix >> 20) - 0x3ff
    const j = ix & 0x000fffff
    ix = j | 0x3ff00000
    let k: number
    if (j <= 0x3988e) k = 0
    else if (j < 0xbb67a) k = 1
    else {
      k = 0
      n += 1
      ix -= 0x00100000
    }
    ax = withHigh(ax, ix)
    // log2(ax) via s = (ax − bp)/(ax + bp), each part split into head and tail.
    let u = ax - BP[k]
    let v = 1 / (ax + BP[k])
    const ss = u * v
    const sH = withLowZero(ss)
    let tH = withHigh(0, ((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18))
    let tL = ax - (tH - BP[k])
    const sL = v * (u - sH * tH - sH * tL)
    let s2 = ss * ss
    let r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))))
    r += sL * (sH + ss)
    s2 = sH * sH
    tH = withLowZero(3 + s2 + r)
    tL = r - (tH - 3 - s2)
    u = sH * tH
    v = sL * tH + tL * ss
    const pH = withLowZero(u + v)
    const pL = v - (pH - u)
    const zH = CP_H * pH
    const zL = CP_L * pH + pL * CP + DP_L[k]
    t1 = withLowZero(zH + zL + DP_H[k] + n)
    t2 = zL - (t1 - n - DP_H[k] - zH)
  }

  // y·log2(x) as pH + pL, then split off its integer part n.
  const y1 = withLowZero(y)
  const pL = (y - y1) * t1 + y * t2
  let pH = y1 * t1
  let z = pL + pH
  let j = high(z)
  const i = low(z)
  if (j >= 0x40900000) {
    if (((j - 0x40900000) | i) !== 0 || pL + OVT > z - pH) return Infinity
  } else if ((j & 0x7fffffff) >= 0x4090cc00) {
    if (((j - (0xc090cc00 | 0)) | i) !== 0 || pL <= z - pH) return 0
  }
  const ij = j & 0x7fffffff
  let k = (ij >> 20) - 0x3ff
  let n = 0
  if (ij > 0x3fe00000) {
    n = j + (0x00100000 >> (k + 1))
    k = ((n & 0x7fffffff) >> 20) - 0x3ff
    const t = withHigh(0, n & ~(0x000fffff >> k))
    n = ((n & 0x000fffff) | 0x00100000) >> (20 - k)
    if (j < 0) n = -n
    pH -= t
  }
  const t = withLowZero(pL + pH)
  const u = t * LN2_POW_H
  const v = (pL - (t - pH)) * LN2_POW + t * LN2_POW_L
  z = u + v
  const w = v - (z - u)
  const tt = z * z
  const c = z - tt * (P1 + tt * (P2 + tt * (P3 + tt * (P4 + tt * P5))))
  const r = (z * c) / (c - 2) - (w + z * w)
  z = 1 - (r - z)
  j = high(z) + (n << 20)
  if (j >> 20 <= 0) return withHigh(z, high(z) + ((n + 1000) << 20)) * TWO_M1000
  return withHigh(z, j)
}

/**
 * `Math.hypot` is approximate in the spec too, and engines scale it
 * differently. This is V8's method, scaled by the larger side (with two
 * terms its compensated sum never compensates), so it matches Chrome.
 */
export function hypot(x: number, y: number): number {
  x = Math.abs(x)
  y = Math.abs(y)
  if (x === Infinity || y === Infinity) return Infinity
  const max = Math.max(x, y)
  if (max === 0 || max !== max) return max
  const a = x / max
  const b = y / max
  return Math.sqrt(a * a + b * b) * max
}
