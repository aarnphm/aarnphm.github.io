import { trailRandom } from './random'

export type Point = { x: number; y: number }
export type Pond = Point & { rx: number; ry: number }
export type Grove = Point & { rx: number; ry: number }
export type Alpine = { left: number; top: number; width: number; height: number; base: number }
export type Landscape = {
  width: number
  height: number
  bottom: number
  alpine: Alpine
  treeline: number
  meadowTop: number
  camp: Point
  pond: Pond
  groves: Grove[]
  contours: Point[][]
  relief: { phase: number; amplitude: number }
}

export function groundOffset(x: number, landscape: Landscape) {
  const { width, relief } = landscape
  return (
    Math.sin((x / width) * Math.PI * 2 + relief.phase) * relief.amplitude +
    Math.sin((x / width) * Math.PI * 5 - relief.phase) * relief.amplitude * 0.3
  )
}

export function pondDistance(point: Point, pond: Pond) {
  return Math.hypot((point.x - pond.x) / pond.rx, (point.y - pond.y) / pond.ry)
}

export function trailLandscape(
  width: number,
  height: number,
  bottom: number,
  camp: Point,
  seed: number,
): Landscape {
  const random = trailRandom(seed ^ 0x51ed270b)
  const alpineWidth = Math.min(1180, width * (width < 700 ? 0.86 : 0.82))
  const alpineHeight = Math.min(width < 700 ? 112 : 206, alpineWidth * 0.27)
  const alpineTop = bottom + 44
  const alpine: Alpine = {
    left: Math.round((width - alpineWidth) / 2),
    top: alpineTop,
    width: alpineWidth,
    height: alpineHeight,
    base: alpineTop + alpineHeight,
  }
  const meadowTop = alpine.base + 24
  const clearing = camp.y - meadowTop
  const landscape: Landscape = {
    width,
    height,
    bottom,
    alpine,
    treeline: alpineTop + alpineHeight * 0.84,
    meadowTop,
    camp,
    relief: { phase: random() * Math.PI * 2, amplitude: Math.min(24, clearing * 0.06) },
    pond: {
      x: width * (0.25 + random() * 0.1),
      y: meadowTop + clearing * 0.64,
      rx: Math.min(108, width * 0.12),
      ry: Math.min(36, clearing * 0.085),
    },
    groves: [],
    contours: [],
  }
  landscape.pond.y += groundOffset(landscape.pond.x, landscape)
  const groves = [
    { x: 0.12, y: 0.36 },
    { x: 0.44, y: 0.31 },
    { x: 0.68, y: 0.57 },
    { x: 0.87, y: 0.78 },
  ]
  landscape.groves = groves.map(point => {
    const x = width * (point.x + (random() - 0.5) * 0.04)
    return {
      x,
      y: meadowTop + 64 + Math.max(1, clearing - 100) * point.y + groundOffset(x, landscape),
      rx: Math.min(112, width * 0.1),
      ry: Math.min(70, clearing * 0.13),
    }
  })
  landscape.contours = [0.43, 0.7, 0.96].map((level, index) => {
    const points: Point[] = []
    const step = width < 700 ? 8 : 16
    for (let x = 0; x < width + step; x += step) {
      const edge = Math.min(width, x)
      const y = meadowTop + clearing * level + groundOffset(edge + index * 36, landscape)
      points.push({ x: edge, y: Math.round(y / 4) * 4 })
    }
    return points
  })
  return landscape
}

export function onDryLand(point: Point, landscape: Landscape, margin = 1.5): Point {
  const { width, height, meadowTop, pond } = landscape
  let { x, y } = point
  const distance = pondDistance(point, pond)
  if (distance < margin) {
    const angle = Math.atan2((y - pond.y) / pond.ry, (x - pond.x) / pond.rx)
    x = pond.x + Math.cos(angle) * pond.rx * margin
    y = pond.y + Math.sin(angle) * pond.ry * margin
  }
  return {
    x: Math.max(26, Math.min(width - 30, x)),
    y: Math.max(meadowTop + 64, Math.min(height - 28, y)),
  }
}

export function followTerrain(points: Point[], landscape: Landscape) {
  const { pond, bottom, camp, alpine } = landscape
  const adjusted = points.map((point, index) => {
    if (point.y <= bottom || index === points.length - 1) return point
    const weight = Math.min(1, (point.y - bottom) / Math.max(1, camp.y - bottom))
    const raised = { x: point.x, y: point.y + groundOffset(point.x, landscape) * weight }
    if (pondDistance(raised, pond) >= 1.8) return raised
    const angle = Math.atan2((raised.y - pond.y) / pond.ry, (raised.x - pond.x) / pond.rx)
    return {
      x: pond.x + Math.cos(angle) * pond.rx * 1.8,
      y: pond.y + Math.sin(angle) * pond.ry * 1.8,
    }
  })
  const routed = [adjusted[0]]
  for (let index = 1; index < adjusted.length; index++) {
    const from = adjusted[index - 1]
    const to = adjusted[index]
    if (from.y < alpine.base && to.y >= alpine.base && from.y < to.y) {
      const atHeight = (y: number) =>
        from.x + (to.x - from.x) * Math.max(0, Math.min(1, (y - from.y) / (to.y - from.y)))
      const entry = atHeight(alpine.top - 16)
      const exit = atHeight(alpine.base + 16)
      if (
        Math.max(entry, exit) > alpine.left - 16 &&
        Math.min(entry, exit) < alpine.left + alpine.width + 16
      ) {
        const flank =
          from.x < landscape.width / 2
            ? Math.max(10, alpine.left - 26)
            : Math.min(landscape.width - 10, alpine.left + alpine.width + 26)
        routed.push(
          { x: flank, y: Math.min(alpine.top - 20, Math.max(from.y, bottom + 18)) },
          { x: flank, y: alpine.top + alpine.height * 0.5 },
          { x: flank, y: alpine.base + 36 },
        )
      }
    }
    const a = { x: (from.x - pond.x) / pond.rx, y: (from.y - pond.y) / pond.ry }
    const b = { x: (to.x - pond.x) / pond.rx, y: (to.y - pond.y) / pond.ry }
    const dx = b.x - a.x
    const dy = b.y - a.y
    const fraction = Math.max(0, Math.min(1, -(a.x * dx + a.y * dy) / (dx * dx + dy * dy || 1)))
    if (Math.hypot(a.x + fraction * dx, a.y + fraction * dy) < 1.7) {
      const start = Math.atan2(a.y, a.x)
      let sweep = Math.atan2(b.y, b.x) - start
      if (sweep > Math.PI) sweep -= Math.PI * 2
      if (sweep < -Math.PI) sweep += Math.PI * 2
      const steps = Math.max(2, Math.ceil(Math.abs(sweep) / (Math.PI / 6)))
      // An arc outside the bank leaves room for spline handles and walking marks.
      for (let step = 0; step <= steps; step++) {
        const angle = start + (sweep * step) / steps
        routed.push({
          x: pond.x + Math.cos(angle) * pond.rx * 1.8,
          y: pond.y + Math.sin(angle) * pond.ry * 1.8,
        })
      }
    }
    routed.push(to)
  }
  return routed
}
