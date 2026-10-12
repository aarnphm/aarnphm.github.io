export function easeTrailDepth(progress: number) {
  const position = Math.max(0, Math.min(1, progress))
  return position * position * (3 - 2 * position)
}
