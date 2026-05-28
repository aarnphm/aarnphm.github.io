export const mfaL = 8
export const mfaMaxM = 6
export const mfaMaxR = 4
export const mfaDefaultM = 2
export const mfaDefaultR = 2
export const mfaPalette = ['#fdb2a2', '#cdd597', '#4385be', '#8b7ec8', '#879a39', '#d14d41']

const mfaSigma = [1.0, 0.85, 0.72, 0.58, 0.42, 0.28, 0.16, 0.07]

function mfaSeededVectors(seed: number, count: number): number[][] {
  let state = seed >>> 0
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0xffffffff
  }
  const raw: number[][] = []
  for (let k = 0; k < count; k++) {
    const v: number[] = []
    for (let i = 0; i < mfaL; i++) v.push(next() * 2 - 1)
    raw.push(v)
  }
  const ortho: number[][] = []
  for (let k = 0; k < count; k++) {
    const v = raw[k].slice()
    for (let j = 0; j < k; j++) {
      const u = ortho[j]
      let dot = 0
      for (let i = 0; i < mfaL; i++) dot += v[i] * u[i]
      for (let i = 0; i < mfaL; i++) v[i] -= dot * u[i]
    }
    let norm = 0
    for (let i = 0; i < mfaL; i++) norm += v[i] * v[i]
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < mfaL; i++) v[i] /= norm
    ortho.push(v)
  }
  return ortho
}

const mfaU = mfaSeededVectors(1729, mfaL)
const mfaV = mfaSeededVectors(2718, mfaL)

function mfaBuildDense(): number[][] {
  const A: number[][] = Array.from({ length: mfaL }, () => Array(mfaL).fill(0))
  for (let k = 0; k < mfaL; k++) {
    const s = mfaSigma[k]
    for (let i = 0; i < mfaL; i++) {
      for (let j = 0; j < mfaL; j++) {
        A[i][j] += s * mfaU[k][i] * mfaV[k][j]
      }
    }
  }
  return A
}

export const mfaDense = mfaBuildDense()

export function mfaFactorMatrix(idx: number, r: number): number[][] {
  const M: number[][] = Array.from({ length: mfaL }, () => Array(mfaL).fill(0))
  const start = idx * r
  const end = Math.min(start + r, mfaL)
  for (let k = start; k < end; k++) {
    const s = mfaSigma[k]
    for (let i = 0; i < mfaL; i++) {
      for (let j = 0; j < mfaL; j++) {
        M[i][j] += s * mfaU[k][i] * mfaV[k][j]
      }
    }
  }
  return M
}

function mfaGateMask(m: number, gateOn: boolean): boolean[][] {
  const mask: boolean[][] = Array.from({ length: mfaL }, () => Array(m).fill(true))
  if (!gateOn) return mask
  for (let i = 0; i < mfaL; i++) {
    for (let k = 0; k < m; k++) {
      mask[i][k] = (i + k) % 2 === 0
    }
  }
  return mask
}

export function mfaApprox(m: number, r: number, gateOn: boolean): number[][] {
  const mask = mfaGateMask(m, gateOn)
  const factors: number[][][] = []
  for (let k = 0; k < m; k++) factors.push(mfaFactorMatrix(k, r))
  const A: number[][] = Array.from({ length: mfaL }, () => Array(mfaL).fill(0))
  for (let i = 0; i < mfaL; i++) {
    for (let j = 0; j < mfaL; j++) {
      for (let k = 0; k < m; k++) {
        if (mask[i][k]) A[i][j] += factors[k][i][j]
      }
    }
  }
  return A
}

export function mfaFrobError(A: number[][], B: number[][]): number {
  let s = 0
  for (let i = 0; i < mfaL; i++) {
    for (let j = 0; j < mfaL; j++) {
      const d = A[i][j] - B[i][j]
      s += d * d
    }
  }
  return Math.sqrt(s)
}

export function mfaStateError(m: number, r: number, gateOn: boolean): number {
  return mfaFrobError(mfaDense, mfaApprox(m, r, gateOn))
}
