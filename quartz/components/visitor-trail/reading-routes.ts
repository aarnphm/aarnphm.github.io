import { trailRandom } from './random'

type ReadingLine = { top: number; bottom: number }
type Crossing = { y: number; room: number }

export function readingRoutes(
  width: number,
  bottom: number,
  left: number,
  right: number,
  lines: ReadingLine[],
  seed: number,
  surface?: ReadingLine,
) {
  const random = trailRandom(seed ^ 0x72c641a9)
  const ordered = [...lines].sort((a, b) => a.top - b.top)
  const gaps: Crossing[] = []
  let previous = ordered[0]?.bottom ?? 64
  for (const line of ordered.slice(1)) {
    const gap = line.top - previous
    if (gap >= 24) gaps.push({ y: (previous + line.top) / 2, room: (gap - 16) / 2 })
    previous = Math.max(previous, line.bottom)
  }
  const available = gaps.sort((a, b) => b.room - a.room)
  const firstLine = ordered[0]?.top ?? 64
  const crossings: Crossing[] = [
    { y: bottom + 76 + random() * 4, room: 3 },
    { y: Math.max(14, firstLine - 28), room: Math.min(5, firstLine / 4) },
    ...(surface
      ? [
          { y: bottom + 36 + random() * 4, room: 3 },
          { y: Math.max(14, firstLine - 48), room: Math.min(3, firstLine / 4) },
        ]
      : available.slice(0, width < 700 ? 1 : 2)),
  ]
  const marginLeft = Math.max(5, Math.min(width * 0.45, left - 12))
  const marginRight = Math.max(5, width - Math.max(width * 0.55, right + 12))
  return crossings.map(({ y, room }, index) => {
    const guardLeft = Math.max(5, marginLeft * (0.5 + random() * 0.4))
    const guardRight = width - Math.max(5, marginRight * (0.5 + random() * 0.4))
    const span = guardRight - guardLeft
    const crossing = y + (random() - 0.5) * room
    const bow = (random() - 0.5) * room * 1.5
    const entry = surface
      ? crossing + (random() - 0.5) * room
      : index === 0
        ? 36 + random() * firstLine
        : random() * bottom * 0.8 + 16
    const exit = surface
      ? crossing + (random() - 0.5) * room
      : Math.max(16, Math.min(bottom + 82, crossing + (random() - 0.5) * bottom * 0.35))
    const bendLeft = guardLeft * (0.25 + random() * 0.45)
    const bendRight = width - (width - guardRight) * (0.25 + random() * 0.45)
    const approach = entry + (crossing - entry) * (0.35 + random() * 0.25)
    const depart = crossing + (exit - crossing) * (0.4 + random() * 0.2)
    return (
      `M0 ${entry}` +
      `C${bendLeft * 0.6} ${entry - 18} ${bendLeft} ${approach - (crossing - entry) * 0.15} ${bendLeft} ${approach}` +
      `C${bendLeft} ${approach + (crossing - entry) * 0.15} ${bendLeft} ${crossing} ${guardLeft} ${crossing}` +
      `C${guardLeft + span * 0.32} ${crossing + bow} ${guardLeft + span * 0.68} ${crossing - bow} ${guardRight} ${crossing}` +
      `C${bendRight} ${crossing} ${bendRight} ${depart} ${bendRight} ${depart}` +
      `C${bendRight} ${depart} ${bendRight} ${exit + 18} ${width} ${exit}`
    )
  })
}
