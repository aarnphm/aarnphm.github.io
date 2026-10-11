import { followTerrain, trailLandscape, type Landscape } from './landscape'
import { trailRandom } from './random'
import { readingRoutes } from './reading-routes'

type Point = { x: number; y: number }
type Rect = { left: number; top: number; right: number; bottom: number }
export type TrailId = 'main' | 'pines' | 'grove' | 'ridge' | 'meadow' | 'switchback'
export const trailIds: TrailId[] = ['main', 'pines', 'grove', 'ridge', 'meadow', 'switchback']

function curvedPath(input: Point[], width: number, height: number, landscape?: Landscape) {
  const points = landscape ? followTerrain(input, landscape) : input
  const bounded = (point: Point) => ({
    x: Math.max(6, Math.min(width - 6, point.x)),
    y: Math.max(8, Math.min(height - 16, point.y)),
  })
  let route = `M${points[0].x} ${points[0].y}`
  for (let index = 0; index < points.length - 1; index++) {
    const before = points[Math.max(0, index - 1)]
    const start = points[index]
    const end = points[index + 1]
    const after = points[Math.min(points.length - 1, index + 2)]
    const first = bounded({
      x: start.x + (end.x - before.x) / 6,
      y: start.y + (end.y - before.y) / 6,
    })
    const second = bounded({
      x: end.x - (after.x - start.x) / 6,
      y: end.y - (after.y - start.y) / 6,
    })
    route += `C${first.x} ${first.y} ${second.x} ${second.y} ${end.x} ${end.y}`
  }
  return route
}

export function clearSpace(width: number, height: number, obstacles: Rect[]) {
  const clampY = (y: number) => Math.max(0, Math.min(height, y))
  const edges = [
    ...new Set([0, height, ...obstacles.flatMap(r => [clampY(r.top), clampY(r.bottom)])]),
  ].sort((a, b) => a - b)
  let path = ''
  // Disjoint open strips keep overlapping text bounds excluded from hit testing.
  for (let index = 1; index < edges.length; index++) {
    const top = edges[index - 1]
    const bottom = edges[index]
    const blocked = obstacles
      .filter(r => r.top < bottom && r.bottom > top)
      .sort((a, b) => a.left - b.left)
    let left = 0
    for (const rect of blocked) {
      const right = Math.max(0, Math.min(width, rect.left))
      if (right > left) path += `M${left} ${top}H${right}V${bottom}H${left}Z`
      left = Math.max(left, Math.min(width, rect.right))
    }
    if (left < width) path += `M${left} ${top}H${width}V${bottom}H${left}Z`
  }
  return path
}

export function trailLayout(scene: HTMLElement, seed: number) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const landing = document.querySelector<HTMLElement>('.landing')
  if (!view || !landing) return null
  const bounds = scene.getBoundingClientRect()
  const width = scene.clientWidth
  const height = scene.clientHeight
  const random = trailRandom(seed)
  const text: Rect[] = []
  const walker = document.createTreeWalker(landing, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!node.textContent?.trim() || !node.parentElement) continue
    const style = view.getComputedStyle(node.parentElement)
    if (style.visibility !== 'visible') continue
    const range = document.createRange()
    range.selectNodeContents(node)
    for (const rect of range.getClientRects()) {
      if (rect.width < 2 || rect.height < 2) continue
      text.push({
        left: rect.left - bounds.left,
        top: rect.top - bounds.top,
        right: rect.right - bounds.left,
        bottom: rect.bottom - bounds.top,
      })
    }
  }
  // Protect the entire reading line, including whitespace between links and words.
  const lines: Rect[] = []
  for (const rect of text.sort((a, b) => a.top - b.top || a.left - b.left)) {
    const line = lines.find(line => Math.abs(line.top - rect.top) < 4)
    if (line) {
      line.left = Math.min(line.left, rect.left)
      line.right = Math.max(line.right, rect.right)
      line.bottom = Math.max(line.bottom, rect.bottom)
    } else lines.push({ ...rect })
  }
  const obstacles = lines.map(line => ({
    left: Math.max(0, line.left - (width < 700 ? 4 : 10)),
    right: Math.min(width, line.right + (width < 700 ? 4 : 10)),
    top: Math.max(0, line.top - 6),
    bottom: line.bottom + 6,
  }))
  for (const control of document.querySelectorAll(
    '.visitor-weather-toggle, .homepage-trail-toggle',
  )) {
    const rect = control.getBoundingClientRect()
    obstacles.push({
      left: rect.left - bounds.left,
      right: rect.right - bounds.left,
      top: rect.top - bounds.top,
      bottom: rect.bottom - bounds.top,
    })
  }
  const bottom = Math.max(landing.getBoundingClientRect().bottom - bounds.top, 280)
  const camp = {
    x: Math.min(width - 72, width * (0.64 + random() * 0.12)),
    y: Math.min(height - 80, Math.max(height - 112, bottom + 225)),
  }
  const landscape = trailLandscape(width, height, bottom, camp, seed)
  const clearing = camp.y - landscape.meadowTop
  const wander = (x: number, y: number): Point => ({
    x: Math.max(10, Math.min(width - 10, width * (x + (random() - 0.5) * 0.12))),
    y: landscape.meadowTop + clearing * Math.max(0.05, y + (random() - 0.5) * 0.14),
  })
  const start = (fraction: number): Point => ({
    x: width * (fraction + (random() - 0.5) * 0.045),
    y: landscape.alpine.base - 12,
  })
  const end = { x: camp.x - 24, y: camp.y + 4 }
  const routes: Record<TrailId, string> = {
    main: curvedPath(
      [
        start(0.41),
        wander(0.4, 0.13),
        wander(0.27, 0.36),
        wander(0.44, 0.73),
        wander(0.61, 0.62),
        { x: camp.x - 42, y: camp.y - 24 },
        end,
      ],
      width,
      height,
      landscape,
    ),
    pines: curvedPath(
      [
        start(0.12),
        wander(0.07, 0.17),
        wander(0.15, 0.5),
        wander(0.28, 0.74),
        wander(0.53, 0.96),
        end,
      ],
      width,
      height,
      landscape,
    ),
    grove: curvedPath(
      [
        start(0.59),
        wander(0.64, 0.16),
        wander(0.8, 0.44),
        wander(0.93, 0.72),
        wander(0.81, 0.96),
        end,
      ],
      width,
      height,
      landscape,
    ),
    ridge: curvedPath(
      [
        start(0.9),
        wander(0.91, 0.1),
        wander(0.79, 0.35),
        wander(0.56, 0.23),
        wander(0.48, 0.52),
        wander(0.6, 0.8),
        end,
      ],
      width,
      height,
      landscape,
    ),
    meadow: curvedPath(
      [
        start(0.26),
        wander(0.15, 0.08),
        wander(0.23, 0.26),
        wander(0.12, 0.62),
        wander(0.34, 0.9),
        wander(0.56, 0.82),
        end,
      ],
      width,
      height,
      landscape,
    ),
    switchback: curvedPath(
      [
        start(0.75),
        wander(0.72, 0.13),
        wander(0.57, 0.32),
        wander(0.79, 0.47),
        wander(0.61, 0.64),
        wander(0.76, 0.8),
        end,
      ],
      width,
      height,
      landscape,
    ),
  }
  const clear = clearSpace(width, height, obstacles)
  const reading = landing.getBoundingClientRect()
  const upperRoutes = readingRoutes(
    width,
    bottom,
    reading.left - bounds.left,
    reading.right - bounds.left,
    lines,
    seed,
  )
  const backdropClear = clearSpace(width, height, [
    ...obstacles,
    {
      left: reading.left - bounds.left - 24,
      right: reading.right - bounds.left + 24,
      top: reading.top - bounds.top - 24,
      bottom: reading.bottom - bounds.top + 24,
    },
  ])
  return {
    width,
    height,
    bottom,
    camp,
    landscape,
    routes,
    readingRoutes: upperRoutes,
    clear,
    backdropClear,
    obstacles,
  }
}
