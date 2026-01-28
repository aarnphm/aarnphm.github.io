import type { FuzzyMatch } from "./types"

// Smith-Waterman implementation with position-aware bonuses for fuzzy match
const W = { MATCH: 16, GAP_START: -3, GAP_EXT: -1, BOUNDARY: 8, CAMEL: 7, CONSECUTIVE: 4, FIRST: 8 }

const SEPARATORS = new Set(["/", ":", "_", "-", " ", "\t", ".", "[", "]"])

function getBonus(t: string, i: number, continued: boolean) {
  let bonus = continued ? W.CONSECUTIVE : 0
  if (i === 0) return bonus + W.FIRST

  const cur = t[i]
  const prev = t[i - 1]

  if (SEPARATORS.has(prev)) bonus += W.BOUNDARY

  const prevLower = prev >= "a" && prev <= "z"
  const prevUpper = prev >= "A" && prev <= "Z"
  const prevDigit = prev >= "0" && prev <= "9"
  const prevOther = !prevLower && !prevUpper && !prevDigit

  const curLower = cur >= "a" && cur <= "z"
  const curUpper = cur >= "A" && cur <= "Z"

  if (prevLower && curUpper) bonus += W.CAMEL
  if (prevOther && (curLower || curUpper)) bonus += W.BOUNDARY

  return bonus
}

export function fuzzyMatch(query: string, target: string): FuzzyMatch | null {
  const n = query.length
  const m = target.length
  if (!n || n > m) return null

  const q = query.toLowerCase()
  const t = target.toLowerCase()

  // Quick check
  let qi = 0
  for (let ti = 0; ti < m && qi < n; ti++) {
    if (t[ti] === q[qi]) qi++
  }
  if (qi < n) return null

  const dp = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(-Infinity))
  const match = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(-Infinity))
  dp[0].fill(0)

  for (let i = 1; i <= n; i++) {
    for (let j = i; j <= m; j++) {
      if (q[i - 1] === t[j - 1]) {
        const prevMatch = match[i - 1][j - 1]
        const prevDp = dp[i - 1][j - 1]

        const scoreMatch = prevMatch + W.MATCH + getBonus(target, j - 1, prevMatch > -Infinity)
        const scoreDp = prevDp + W.MATCH + getBonus(target, j - 1, false)

        match[i][j] = Math.max(scoreMatch, scoreDp)
      }

      let bestGap = -Infinity
      for (let k = i; k < j; k++) {
        const gap = j - k - 1
        const pen = gap === 0 ? 0 : W.GAP_START + W.GAP_EXT * (gap - 1)
        const cand = match[i][k] + pen
        if (cand > bestGap) bestGap = cand
      }
      dp[i][j] = Math.max(match[i][j], bestGap)
    }
  }

  let score = -Infinity
  let end = -1
  for (let j = n; j <= m; j++) {
    if (dp[n][j] > score) {
      score = dp[n][j]
      end = j
    }
  }

  if (score === -Infinity) return null

  const positions: number[] = []
  let i = n
  let j = end

  while (i > 0) {
    if (match[i][j] === dp[i][j] && q[i - 1] === t[j - 1]) {
      positions.unshift(j - 1)
      i--
      j--
    } else {
      for (let k = j - 1; k >= i; k--) {
        const gap = j - k - 1
        const pen = gap === 0 ? 0 : W.GAP_START + W.GAP_EXT * (gap - 1)
        if (match[i][k] !== -Infinity && match[i][k] + pen === dp[i][j]) {
          j = k
          break
        }
      }
    }
  }

  return { score, positions }
}

export function fuzzyMatchMultiple(
  query: string,
  targets: string[],
): { target: string; match: FuzzyMatch } | null {
  let best: { target: string; match: FuzzyMatch } | null = null

  for (const target of targets) {
    const match = fuzzyMatch(query, target)
    if (match && (!best || match.score > best.match.score)) {
      best = { target, match }
    }
  }

  return best
}
