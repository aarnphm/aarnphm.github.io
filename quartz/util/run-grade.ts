// Minetti et al. (2002), doi:10.1152/japplphysiol.01177.2001: running cost in J/kg/m,
// normalised to level running (3.6 J/kg/m).
export function runGradeFactor(grade: number): number {
  const g = Math.min(0.3, Math.max(-0.3, grade))
  return (155.4 * g ** 5 - 30.4 * g ** 4 - 43.3 * g ** 3 + 46.3 * g ** 2 + 19.5 * g + 3.6) / 3.6
}

// Per-sample altitude deltas over ~3 m are mostly barometer noise; the cost curve is convex, so
// noise alone inflates grade-adjusted speed. Each sample takes the grade of its trailing 30 m.
const RUN_GRADE_SPAN_M = 30

/** Grade factor for each segment ending at sample i (index 0 is unused), or null without altitude. */
export function runGradeFactors(
  stream: { altitude?: readonly number[]; distance?: readonly number[] } | undefined,
): Float64Array | null {
  const alt = stream?.altitude
  const dist = stream?.distance
  if (!alt || !dist) return null
  const n = Math.min(alt.length, dist.length)
  if (n < 2) return null
  const factors = new Float64Array(n).fill(1)
  let start = 0
  for (let i = 1; i < n; i++) {
    while (start + 1 < i && dist[i] - dist[start + 1] >= RUN_GRADE_SPAN_M) start++
    const span = dist[i] - dist[start]
    const rise = alt[i] - alt[start]
    if (span > 0 && Number.isFinite(rise)) factors[i] = runGradeFactor(rise / span)
  }
  return factors
}
