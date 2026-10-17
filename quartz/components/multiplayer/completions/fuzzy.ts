import type { FuzzyMatch } from './types'

// Smith-Waterman / Gotoh O(n * m) affine-gap implementation with zero-allocation scratch buffers
const W = { MATCH: 16, GAP_START: -3, GAP_EXT: -1, BOUNDARY: 8, CAMEL: 7, CONSECUTIVE: 4, FIRST: 8 }

const SEPARATOR_LUT = new Uint8Array(128)
for (const ch of ['/', ':', '_', '-', ' ', '\t', '.', '[', ']']) {
  SEPARATOR_LUT[ch.charCodeAt(0)] = 1
}

function getBaseBonus(target: string, index: number): number {
  if (index === 0) return W.FIRST

  let bonus = 0
  const cur = target.charCodeAt(index)
  const prev = target.charCodeAt(index - 1)

  if (prev < 128 && SEPARATOR_LUT[prev] === 1) bonus += W.BOUNDARY

  const prevLower = prev >= 97 && prev <= 122
  const prevUpper = prev >= 65 && prev <= 90
  const prevDigit = prev >= 48 && prev <= 57
  const prevOther = !prevLower && !prevUpper && !prevDigit

  const curLower = cur >= 97 && cur <= 122
  const curUpper = cur >= 65 && cur <= 90

  if (prevLower && curUpper) bonus += W.CAMEL
  if (prevOther && (curLower || curUpper)) bonus += W.BOUNDARY

  return bonus
}

const NEG_INF = -1e9
let scratchCapacity = 4096
let matchBuf = new Int32Array(scratchCapacity)
let dpBuf = new Int32Array(scratchCapacity)
let traceBuf = new Int32Array(scratchCapacity)
let targetLowerBuf = new Uint16Array(512)
let queryLowerBuf = new Uint16Array(128)
let minJBuf = new Int32Array(128)

function ensureBuffers(n: number, m: number) {
  const size = (n + 1) * (m + 1)
  if (size > scratchCapacity) {
    scratchCapacity = Math.max(size, scratchCapacity * 2)
    matchBuf = new Int32Array(scratchCapacity)
    dpBuf = new Int32Array(scratchCapacity)
    traceBuf = new Int32Array(scratchCapacity)
  }
  if (m > targetLowerBuf.length) {
    targetLowerBuf = new Uint16Array(Math.max(m, targetLowerBuf.length * 2))
  }
  if (n + 1 > queryLowerBuf.length) {
    const cap = Math.max(n + 1, queryLowerBuf.length * 2)
    queryLowerBuf = new Uint16Array(cap)
    minJBuf = new Int32Array(cap)
  }
}

export function fuzzyMatch(query: string, target: string, minScore = -Infinity): FuzzyMatch | null {
  const n = query.length
  const m = target.length
  if (!n || n > m) return null

  ensureBuffers(n, m)

  for (let i = 0; i < n; i++) {
    const c = query.charCodeAt(i)
    queryLowerBuf[i] = c >= 65 && c <= 90 ? c + 32 : c
  }

  let qi = 0
  let qTarget = queryLowerBuf[0]!
  for (let ti = 0; ti < m; ti++) {
    const c = target.charCodeAt(ti)
    const lower = c >= 65 && c <= 90 ? c + 32 : c
    targetLowerBuf[ti] = lower
    if (qi < n && lower === qTarget) {
      minJBuf[qi + 1] = ti + 1
      qi++
      if (qi < n) qTarget = queryLowerBuf[qi]!
    }
  }
  if (qi < n) return null

  const stride = m + 1
  for (let j = 0; j <= m; j++) {
    dpBuf[j] = 0
    matchBuf[j] = NEG_INF
  }

  for (let i = 1; i <= n; i++) {
    const rowOff = i * stride
    const prevRowOff = (i - 1) * stride
    const qChar = queryLowerBuf[i - 1]!
    const startJ = minJBuf[i]!
    const prevStartJ = i === 1 ? 0 : minJBuf[i - 1]!

    dpBuf[rowOff + startJ - 1] = NEG_INF
    matchBuf[rowOff + startJ - 1] = NEG_INF
    if (startJ - 2 >= 0) {
      dpBuf[rowOff + startJ - 2] = NEG_INF
      matchBuf[rowOff + startJ - 2] = NEG_INF
    }

    let bestGap = NEG_INF
    let bestGapK = -1
    let affineGap = NEG_INF
    let affineGapK = -1

    for (let j = startJ; j <= m; j++) {
      const idx = rowOff + j
      if (j - 2 >= startJ) {
        const candStart = matchBuf[rowOff + (j - 2)]!
        const candStartPen = candStart > NEG_INF / 2 ? candStart + W.GAP_START : NEG_INF
        const candExt = affineGap > NEG_INF / 2 ? affineGap + W.GAP_EXT : NEG_INF
        if (candStartPen >= candExt) {
          affineGap = candStartPen
          affineGapK = j - 2
        } else {
          affineGap = candExt
        }
      }

      if (j - 1 >= startJ) {
        const zeroGapCand = matchBuf[rowOff + (j - 1)]!
        if (zeroGapCand >= affineGap) {
          bestGap = zeroGapCand
          bestGapK = j - 1
        } else {
          bestGap = affineGap
          bestGapK = affineGapK
        }
      } else {
        bestGap = NEG_INF
        bestGapK = -1
      }

      let curMatch = NEG_INF
      if (qChar === targetLowerBuf[j - 1] && j - 1 >= prevStartJ) {
        const prevMatch = matchBuf[prevRowOff + (j - 1)]!
        const prevDp = dpBuf[prevRowOff + (j - 1)]!
        const baseBonus = W.MATCH + getBaseBonus(target, j - 1)
        const scoreMatch = prevMatch > NEG_INF / 2 ? prevMatch + baseBonus + W.CONSECUTIVE : NEG_INF
        const scoreDp = prevDp > NEG_INF / 2 ? prevDp + baseBonus : NEG_INF
        curMatch = scoreMatch >= scoreDp ? scoreMatch : scoreDp
      }
      matchBuf[idx] = curMatch

      if (curMatch >= bestGap) {
        dpBuf[idx] = curMatch
        traceBuf[idx] = j
      } else {
        dpBuf[idx] = bestGap
        traceBuf[idx] = bestGapK
      }
    }
  }

  const lastRowOff = n * stride
  const lastStartJ = minJBuf[n]!
  let score = NEG_INF
  let end = -1
  for (let j = lastStartJ; j <= m; j++) {
    const v = dpBuf[lastRowOff + j]!
    if (v > score) {
      score = v
      end = j
    }
  }

  if (score <= NEG_INF / 2 || score <= minScore) return null

  let i = n
  let j = end
  while (i > 0) {
    const idx = i * stride + j
    const matchedAt = traceBuf[idx]!
    if (matchedAt === j && matchBuf[idx] === dpBuf[idx]) {
      minJBuf[i - 1] = j - 1
      i--
      j--
    } else {
      j = matchedAt
    }
  }

  const positions: number[] = []
  for (let k = 0; k < n; k++) {
    positions.push(minJBuf[k]!)
  }

  return { score, positions }
}

export function fuzzyMatchMultiple(
  query: string,
  targets: string[],
  minScore = -Infinity,
): { target: string; match: FuzzyMatch } | null {
  let best: { target: string; match: FuzzyMatch } | null = null
  let bestScore = minScore

  for (let i = 0; i < targets.length; i++) {
    const target = targets[i]!
    const match = fuzzyMatch(query, target, bestScore)
    if (match && match.score > bestScore) {
      bestScore = match.score
      best = { target, match }
    }
  }

  return best
}
