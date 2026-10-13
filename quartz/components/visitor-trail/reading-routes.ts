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
  const marginLeft = Math.max(5, Math.min(width * 0.45, left - 12))
  const marginRight = Math.max(5, width - Math.max(width * 0.55, right + 12))
  if (surface) return boardRoutes(width, marginLeft, marginRight, surface, random)
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
    ...available.slice(0, width < 700 ? 1 : 2),
  ]
  return crossings.map(({ y, room }, index) => {
    const guardLeft = Math.max(5, marginLeft * (0.5 + random() * 0.4))
    const guardRight = width - Math.max(5, marginRight * (0.5 + random() * 0.4))
    const span = guardRight - guardLeft
    const crossing = y + (random() - 0.5) * room
    const bow = (random() - 0.5) * room * 1.5
    const entry = index === 0 ? 36 + random() * firstLine : random() * bottom * 0.8 + 16
    const exit = Math.max(16, Math.min(bottom + 82, crossing + (random() - 0.5) * bottom * 0.35))
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

type Point = { x: number; y: number }
type Band = { min: number; max: number }

const fixed = (value: number) => value.toFixed(1)

// Catmull–Rom through the waypoints as cubic Béziers, so a route bends smoothly at each one.
function smooth(points: Point[]) {
  const at = (index: number) => points[Math.max(0, Math.min(points.length - 1, index))]
  let d = `M${fixed(points[0].x)} ${fixed(points[0].y)}`
  for (let index = 0; index < points.length - 1; index++) {
    const [before, from, to, after] = [at(index - 1), at(index), at(index + 1), at(index + 2)]
    d +=
      `C${fixed(from.x + (to.x - before.x) / 6)} ${fixed(from.y + (to.y - before.y) / 6)} ` +
      `${fixed(to.x - (after.x - from.x) / 6)} ${fixed(to.y - (after.y - from.y) / 6)} ` +
      `${fixed(to.x)} ${fixed(to.y)}`
  }
  return d
}

// Routes around the hanging board meander in the bands above and below it. With wide margins they
// swing down or up beside the board, one may wrap from the top band around the right edge to
// leave under the board, and one lower route may roll a loop.
function boardRoutes(
  width: number,
  marginLeft: number,
  marginRight: number,
  board: ReadingLine,
  random: () => number,
) {
  const height = board.bottom - board.top
  const above: Band = { min: 8, max: Math.max(12, board.top - 10) }
  const below: Band = { min: board.bottom + 14, max: board.bottom + 110 }
  const wide = Math.min(marginLeft, marginRight) >= 48
  const loopRoute = random() < 0.5 ? 0 : 2
  const wrapRoute = wide && random() < 0.6 ? 3 : -1
  const between = (min: number, max: number) => min + random() * (max - min)
  // Upper routes hold their side swings to the top 60% of the board; the controls sit top right.
  const sideRange = (band: Band, right: boolean): Band =>
    band === above
      ? { min: right ? 72 : 16, max: board.top + height * 0.6 }
      : { min: board.top + height * 0.4, max: below.max }
  return [below, above, below, above].map((band, index) => {
    const guardLeft = Math.max(5, marginLeft * (0.5 + random() * 0.4))
    const guardRight = width - Math.max(5, marginRight * (0.5 + random() * 0.4))
    const span = guardRight - guardLeft
    const room = band.max - band.min
    const centre = band.min + room * (0.3 + random() * 0.4)
    const amplitude = room * (0.15 + random() * 0.3)
    const near = () => centre + (random() - 0.5) * Math.min(room, 8)
    const leftRange = sideRange(band, false)
    const rightRange = index === wrapRoute ? sideRange(below, true) : sideRange(band, true)
    const entry = wide ? between(leftRange.min, leftRange.max) : near()
    const exit = wide ? between(rightRange.min, rightRange.max) : near()
    const points: Point[] = [{ x: 0, y: entry }]
    if (wide)
      points.push({
        x: guardLeft * (0.3 + random() * 0.3),
        y: (entry + centre) / 2 + (random() - 0.5) * 24,
      })
    const lobes = (width < 700 ? 2 : 3) + Math.floor(random() * 3)
    const loopAt = index === loopRoute ? 1 + Math.floor(random() * (lobes - 1)) : -1
    const radius = Math.min(14, room / 5) * (0.7 + random() * 0.3)
    const sign = random() < 0.5 ? -1 : 1
    for (let lobe = 0; lobe <= lobes; lobe++) {
      const inner = lobe > 0 && lobe < lobes
      const x =
        guardLeft + (span * lobe) / lobes + (inner ? (random() - 0.5) * (span / lobes) * 0.3 : 0)
      const swing = (lobe % 2 ? 1 : -1) * sign * amplitude * (0.5 + random() * 0.5)
      const ceiling = lobe === loopAt ? band.max - radius * 2 : band.max
      const y = Math.max(band.min, Math.min(ceiling, centre + swing))
      points.push({ x, y })
      if (lobe === loopAt)
        points.push(
          { x: x + radius, y: y + radius },
          { x, y: y + radius * 2 },
          { x: x - radius, y: y + radius },
          { x: x + radius * 0.4, y: y - radius * 0.15 },
        )
    }
    if (wide)
      points.push({
        x: guardRight + (width - guardRight) * (0.4 + random() * 0.3),
        y: (centre + exit) / 2 + (random() - 0.5) * 24,
      })
    points.push({ x: width, y: exit })
    return smooth(points)
  })
}
