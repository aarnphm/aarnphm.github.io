// Okhsv after Björn Ottosson (https://bottosson.github.io/posts/colorpicker/, MIT): HSV laid out
// over Oklab, so equal steps on the plane look equal and the hue strip keeps its real lightness.

export type Rgb = [number, number, number]
/** h in degrees, s and v in 0..1. */
export type Hsv = { h: number; s: number; v: number }

type Lab = [number, number, number]
type Cusp = { L: number; C: number }

const toLinear = (x: number) => (x >= 0.04045 ? ((x + 0.055) / 1.055) ** 2.4 : x / 12.92)
const toGamma = (x: number) => (x >= 0.0031308 ? 1.055 * x ** (1 / 2.4) - 0.055 : 12.92 * x)

const linearToOklab = (r: number, g: number, b: number): Lab => {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

const oklabToLinear = (L: number, a: number, b: number): Rgb => {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ]
}

// Largest saturation C/L that stays in sRGB along hue (a, b): a polynomial guess and one Halley step.
const maxSaturation = (a: number, b: number) => {
  const [k0, k1, k2, k3, k4, wl, wm, ws] =
    -1.88170328 * a - 0.80936493 * b > 1
      ? [
          1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245, 4.0767416621, -3.3077115913,
          0.2309699292,
        ]
      : 1.81444104 * a - 1.19445276 * b > 1
        ? [
            0.73956515, -0.45954404, 0.08285427, 0.1254107, 0.14503204, -1.2684380046, 2.6097574011,
            -0.3413193965,
          ]
        : [
            1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167, -0.0041960863,
            -0.7034186147, 1.707614701,
          ]
  const S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b
  const kl = 0.3963377774 * a + 0.2158037573 * b
  const km = -0.1055613458 * a - 0.0638541728 * b
  const ks = -0.0894841775 * a - 1.291485548 * b
  const l_ = 1 + S * kl
  const m_ = 1 + S * km
  const s_ = 1 + S * ks
  const f = wl * l_ ** 3 + wm * m_ ** 3 + ws * s_ ** 3
  const f1 = wl * 3 * kl * l_ * l_ + wm * 3 * km * m_ * m_ + ws * 3 * ks * s_ * s_
  const f2 = wl * 6 * kl * kl * l_ + wm * 6 * km * km * m_ + ws * 6 * ks * ks * s_
  return S - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

const cuspOf = (a: number, b: number): Cusp => {
  const S = maxSaturation(a, b)
  const L = Math.cbrt(1 / Math.max(...oklabToLinear(1, S * a, S * b)))
  return { L, C: L * S }
}

const K1 = 0.206
const K2 = 0.03
const K3 = (1 + K1) / (1 + K2)
const toe = (x: number) => 0.5 * (K3 * x - K1 + Math.sqrt((K3 * x - K1) ** 2 + 4 * K2 * K3 * x))
const toeInv = (x: number) => (x * x + K1 * x) / (K3 * (x + K2))

const S0 = 0.5

/** A hue's cusp, shared by every pixel of the plane at that hue. */
export const hueBasis = (h: number) => {
  const a = Math.cos((h / 180) * Math.PI)
  const b = Math.sin((h / 180) * Math.PI)
  const cusp = cuspOf(a, b)
  return { a, b, Smax: cusp.C / cusp.L, Tmax: cusp.C / (1 - cusp.L) }
}

export const okhsvToRgb = (
  s: number,
  v: number,
  { a, b, Smax, Tmax }: ReturnType<typeof hueBasis>,
): Rgb => {
  if (v <= 0) return [0, 0, 0]
  const k = 1 - S0 / Smax
  const d = S0 + Tmax - Tmax * k * s
  const Lv = 1 - (s * S0) / d
  const Cv = (s * Tmax * S0) / d
  let L = v * Lv
  let C = v * Cv
  const Lvt = toeInv(Lv)
  const Cvt = (Cv * Lvt) / Lv
  const Lnew = toeInv(L)
  C = (C * Lnew) / L
  L = Lnew
  const scale = Math.cbrt(1 / Math.max(...oklabToLinear(Lvt, a * Cvt, b * Cvt), 0))
  L *= scale
  C *= scale
  return oklabToLinear(L, C * a, C * b).map(x => toGamma(Math.min(1, Math.max(0, x))) * 255) as Rgb
}

/** `previous` keeps hue for greys and saturation for black, which have neither. */
export const rgbToOkhsv = ([r, g, b]: Rgb, previous?: Hsv): Hsv => {
  const [L, A, B] = linearToOklab(toLinear(r / 255), toLinear(g / 255), toLinear(b / 255))
  const C = Math.hypot(A, B)
  if (L <= 1e-6) return { h: previous?.h ?? 0, s: previous?.s ?? 0, v: 0 }
  if (C < 1e-4) return { h: previous?.h ?? 0, s: 0, v: Math.min(1, toe(L)) }
  const h = ((Math.atan2(B, A) * 180) / Math.PI + 360) % 360
  const { a, b: bb, Smax, Tmax } = hueBasis(h)
  const k = 1 - S0 / Smax
  const t = Tmax / (C + L * Tmax)
  const Lv = t * L
  const Cv = t * C
  const Lvt = toeInv(Lv)
  const Cvt = (Cv * Lvt) / Lv
  const scale = Math.cbrt(1 / Math.max(...oklabToLinear(Lvt, a * Cvt, bb * Cvt), 0))
  const Ls = L / scale
  const Lt = toe(Ls)
  return {
    h,
    s: Math.min(1, Math.max(0, ((S0 + Tmax) * Cv) / (Tmax * S0 + Tmax * k * Cv))),
    v: Math.min(1, Math.max(0, Lt / Lv)),
  }
}

/** Oklab distance between two sRGB colours, for keeping sampled chips apart. */
export const oklabDistance = (x: Rgb, y: Rgb) => {
  const p = linearToOklab(...(x.map(c => toLinear(c / 255)) as Rgb))
  const q = linearToOklab(...(y.map(c => toLinear(c / 255)) as Rgb))
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])
}

export const oklabChroma = (x: Rgb) => {
  const [, a, b] = linearToOklab(...(x.map(c => toLinear(c / 255)) as Rgb))
  return Math.hypot(a, b)
}
