export const alignedTrainingEffectMargins = (
  positions: readonly { activityTop: number; effectTop: number; marginTop: number }[],
): number[] => {
  const rows: { top: number; indices: number[] }[] = []
  for (const [index, position] of positions.entries()) {
    const row = rows.find(candidate => Math.abs(candidate.top - position.activityTop) < 1)
    if (row) row.indices.push(index)
    else rows.push({ top: position.activityTop, indices: [index] })
  }
  const margins = positions.map(() => 0)
  for (const row of rows) {
    if (row.indices.length < 2) continue
    const targetTop = Math.max(
      ...row.indices.map(index => positions[index].effectTop - positions[index].marginTop),
    )
    for (const index of row.indices)
      margins[index] = Math.max(
        0,
        targetTop - (positions[index].effectTop - positions[index].marginTop),
      )
  }
  return margins
}
