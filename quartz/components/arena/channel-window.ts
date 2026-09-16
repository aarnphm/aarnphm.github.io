export function arenaRowOffsets(
  count: number,
  columns: number,
  estimate: number,
  measured: ReadonlyMap<number, number>,
): number[] {
  const offsets = [0]
  for (let row = 0; row < Math.ceil(count / columns); row++) {
    offsets.push(offsets[row] + (measured.get(row) ?? estimate))
  }
  return offsets
}

export function arenaRowAt(offsets: readonly number[], position: number): number {
  let low = 0
  let high = Math.max(0, offsets.length - 2)
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (offsets[middle + 1] <= position) low = middle + 1
    else high = middle
  }
  return low
}

export function arenaVisibleRows(
  offsets: readonly number[],
  top: number,
  height: number,
  overscan: number,
): number[] {
  const total = offsets.at(-1) ?? 0
  if (total === 0 || top + height + overscan <= 0 || top - overscan >= total) return []
  const start = arenaRowAt(offsets, Math.max(0, top - overscan))
  const end = arenaRowAt(offsets, Math.min(total - 1, top + height + overscan))
  return Array.from({ length: end - start + 1 }, (_, index) => start + index)
}
