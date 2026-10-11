export function pixels(rows: readonly string[], tones: Record<string, string>, size = 3) {
  const paths = new Map<string, string[]>()
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const tone = tones[row[x]]
      if (!tone) continue
      const marks = paths.get(tone) ?? []
      marks.push(`M${x * size} ${y * size}h${size}v${size}h-${size}Z`)
      paths.set(tone, marks)
    }
  })
  return [...paths].map(([tone, marks]) => `<path class="${tone}" d="${marks.join('')}"/>`).join('')
}
