// Shared by the server render and the inline script, so the first paint and every update agree.
// Heavy ball on f(x) = (lambda/2) x^2. The rate depends on alpha only through a = alpha * lambda.

export const HB_ALPHA = { min: 0, max: 4, step: 0.05 } as const
export const HB_BETA = { min: 0, max: 0.95, step: 0.05 } as const
export const HB_STEPS = 24
const TOL = 1e-3
const MAX_ITERS = 2000

export const HB_W = 300
export const HB_H = 214
export const HB_X0 = 40
export const HB_X1 = 288
export const HB_Y0 = 186
export const HB_Y1 = 14
export const HB_RHO_MAX = 3
export const HB_X_ABS = 1.5

const r1 = (n: number) => Math.round(n * 10) / 10
export const hbRateX = (a: number) => r1(HB_X0 + (a / HB_ALPHA.max) * (HB_X1 - HB_X0))
export const hbRateY = (rho: number) => r1(HB_Y0 - (rho / HB_RHO_MAX) * (HB_Y0 - HB_Y1))
export const hbTrajX = (t: number) => r1(HB_X0 + (t / HB_STEPS) * (HB_X1 - HB_X0))
export const hbTrajY = (x: number) =>
  r1((HB_Y0 + HB_Y1) / 2 - (x / HB_X_ABS) * ((HB_Y0 - HB_Y1) / 2))

// Roots of r^2 - (1 + beta - a) r + beta = 0; the rate is the larger root magnitude.
export const hbRate = (a: number, beta: number): { rho: number; complex: boolean } => {
  const p = 1 + beta - a
  const disc = p * p - 4 * beta
  if (disc < 0) return { rho: Math.sqrt(beta), complex: true }
  const s = Math.sqrt(disc)
  return { rho: Math.max(Math.abs((p + s) / 2), Math.abs((p - s) / 2)), complex: false }
}

// Starts at rest: x_{-1} = x_0 = 1.
export const hbTrajectory = (a: number, beta: number, steps: number): number[] => {
  const xs = [1]
  let prev = 1
  let cur = 1
  for (let t = 0; t < steps; t++) {
    const next = (1 + beta - a) * cur - beta * prev
    prev = cur
    cur = next
    xs.push(cur)
  }
  return xs
}

// First t after which |x_t| stays below TOL, or null when it has not settled by MAX_ITERS.
const hbSettle = (a: number, beta: number): number | null => {
  const xs = hbTrajectory(a, beta, MAX_ITERS)
  let last = -1
  for (let t = 0; t < xs.length; t++) if (Math.abs(xs[t]) >= TOL) last = t
  return last === xs.length - 1 ? null : last + 1
}

const rateCurve = (beta: number): string => {
  const pts: number[] = []
  for (let i = 0; i <= 400; i++) pts.push((i / 400) * HB_ALPHA.max)
  // Corners of the curve sit where the roots turn complex; sample them exactly.
  const sb = Math.sqrt(beta)
  pts.push((1 - sb) ** 2, (1 + sb) ** 2)
  pts.sort((x, y) => x - y)
  return pts
    .filter(a => a <= HB_ALPHA.max)
    .map((a, i) => `${i ? 'L' : 'M'}${hbRateX(a)} ${hbRateY(hbRate(a, beta).rho)}`)
    .join('')
}

export const HB_REFERENCE = rateCurve(0)

export type HBView = {
  curve: string
  markX: number
  markY: number
  bandW: number
  traj: string
  points: { x: number; y: number }[]
  alpha: string
  beta: string
  rho: string
  roots: string
  settle: string
  valuetextAlpha: string
  valuetextBeta: string
}

export const hbView = (a: number, beta: number): HBView => {
  const { rho, complex } = hbRate(a, beta)
  const xs = hbTrajectory(a, beta, HB_STEPS)
  const points = xs.map((x, t) => ({
    x: hbTrajX(t),
    // Diverging iterates leave the panel; the clip hides them and the readout carries the value.
    y: Math.abs(x) > 1e6 ? hbTrajY(Math.sign(x) * 1e6) : hbTrajY(x),
  }))
  // At the band edges the larger root has magnitude 1; rounding must not call that convergent.
  const converges = a > 0 && rho < 1 - 1e-9
  const settle = converges ? hbSettle(a, beta) : null
  const convergentTo = Math.min(HB_ALPHA.max, 2 * (1 + beta))
  return {
    curve: rateCurve(beta),
    markX: hbRateX(a),
    markY: hbRateY(rho),
    bandW: r1(hbRateX(convergentTo) - HB_X0),
    traj: points.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(''),
    points,
    alpha: a.toFixed(2),
    beta: beta.toFixed(2),
    rho: rho.toFixed(3),
    roots: complex ? 'complex, |r| = √β' : 'real',
    settle: settle === null ? (converges ? `> ${MAX_ITERS}` : 'never') : String(settle),
    valuetextAlpha: `alpha lambda ${a.toFixed(2)}, rate ${rho.toFixed(3)}`,
    valuetextBeta: `beta ${beta.toFixed(2)}, convergent for alpha lambda below ${(2 * (1 + beta)).toFixed(2)}`,
  }
}
