import {
  creekDistance,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { trailRandom } from './random'
import { contourFaces, faceAt, routePoints } from './ravine'

type Runs = Map<string, string[]>
type Stop = Point & { distance: number; dx: number; dy: number }

function run(runs: Runs, tone: string, x: number, y: number, width: number, height = 2) {
  if (width <= 0) return
  const list = runs.get(tone) ?? []
  list.push(`M${Math.round(x / 2) * 2} ${Math.round(y / 2) * 2}h${width}v${height}h${-width}Z`)
  runs.set(tone, list)
}

// Walks a route at an even pace, so features spaced by distance land evenly on the ground.
function walk(d: string, pace: number) {
  const points = routePoints(d)
  const stops: Stop[] = []
  let distance = 0
  let next = 0
  for (let index = 1; index < points.length; index++) {
    const a = points[index - 1]
    const b = points[index]
    const length = Math.hypot(b.x - a.x, b.y - a.y)
    while (length && next <= distance + length) {
      const t = (next - distance) / length
      stops.push({
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        distance: next,
        dx: (b.x - a.x) / length,
        dy: (b.y - a.y) / length,
      })
      next += pace
    }
    distance += length
  }
  return stops
}

const even = (value: number) => Math.round(value / 2) * 2
const leaves = ['leaf-red', 'leaf-orange', 'leaf-yellow', 'leaf-red', 'leaf-brown']

// Trailwork along each route: stone steps up the faces of the rises, log water bars across the
// shallower climbs, boardwalk where the trail runs by wet ground, posts with each route's blaze,
// cairns on the crests, a bench at the best rise, signposts at trailheads and junctions, puddles
// on the flats, and leaf litter that shows in autumn.
export function trailWayside(landscape: Landscape, routes: readonly string[], seed: number) {
  const runs: Runs = new Map()
  const random = trailRandom(seed ^ 0x5be0cd19)
  const faces = contourFaces(landscape)
  const { alpine, camp, pond, meadowTop } = landscape
  const walked: Stop[][] = []
  const cairns: Point[] = []
  routes.forEach((d, route) => {
    const stops = walk(d, 2)
    if (stops.length < 8) return
    const length = stops[stops.length - 1].distance
    let lastStep = -Infinity
    let lastBar = -Infinity
    let lastPuddle = 0
    let lastFace: unknown = null
    let bench: (Stop & { tall: number }) | null = null
    stops.forEach((stop, index) => {
      const { x, y, dx, dy, distance } = stop
      if (y < alpine.base + 4 || distance < 10 || length - distance < 40) return
      const depth = trailGroundScale(y, landscape)
      const face = faceAt(faces, stop)
      const creek = creekDistance(stop, landscape)
      const wet = (creek > 12 && creek < 28) || (pondDistance(stop, pond) < 1.9 && creek > 12)
      // Across the trail, to its left as walked.
      const nx = -dy
      const ny = dx
      if (face && face.below < face.tall) {
        if (face !== lastFace && face.below < 4) {
          lastFace = face
          if (!bench || face.tall > bench.tall) bench = { ...stop, tall: face.tall }
          if (cairns.every(other => Math.hypot(other.x - x, other.y - y) > 90)) {
            const side = random() < 0.5 ? -1 : 1
            const cx = x + nx * 12 * side
            const cy = y + ny * 6 * side
            cairns.push({ x: cx, y: cy })
            const size = depth > 0.6 ? 4 : 2
            ;[5, 4, 3, 2, 1].forEach((width, level) => {
              const w = width * size
              const top = cy - (level + 1) * size
              run(runs, 'stone', cx - w / 2, top, w, size)
              run(runs, 'stone-lit', cx - w / 2, top, Math.max(2, w / 2), Math.min(2, size))
            })
          }
        }
        const climb = Math.abs(dy) > Math.abs(dx) * 0.6
        const width = even(8 + depth * 8)
        // Steps stack up a steep face; treads lit, risers in shade, offset as the trail bends.
        if (climb && distance - lastStep >= 6) {
          lastStep = distance
          run(runs, 'tread', x - width / 2, y - 2, width)
          run(runs, 'riser', x - width / 2, y, width)
          run(runs, 'riser', x + width / 2, y - 2, 2, 4)
        }
        // A log set aslant across a shallow climb sheds water off the tread.
        if (!climb && distance - lastBar >= 34) {
          lastBar = distance
          const slant = dx > 0 ? 1 : -1
          for (let k = 0; k < 4; k++) {
            run(runs, 'log', x - width / 2 + k * (width / 4), y - 4 + k * 2 * slant, width / 4 + 2)
            run(runs, 'log-end', x - width / 2 + k * (width / 4), y - 2 + k * 2 * slant, 2)
          }
        }
      } else if (wet && index % 2 === 0) {
        // Boardwalk over the soft ground: boards laid across the trail, so they run up the screen
        // where the trail runs sideways and along it where the trail climbs.
        const width = even(10 + depth * 8)
        const across = Math.abs(dy)
        const board = even(4 + across * width)
        const tall = even(2 + (1 - across) * 4)
        run(runs, 'plank', x - board / 2, y - tall, board, tall)
        if (across > 0.6) run(runs, 'plank-gap', x - board / 2, y, board)
        if (index % 12 === 0) {
          run(runs, 'post', x + nx * width * 0.6, y - 4 + ny * 3, 2, 6)
          run(runs, 'post', x - nx * width * 0.6, y - 4 - ny * 3, 2, 6)
        }
      } else if (
        !face &&
        creek > 40 &&
        y > meadowTop &&
        distance - lastPuddle > 220 &&
        random() < 0.02
      ) {
        lastPuddle = distance
        const w = even(10 + depth * 10)
        run(runs, 'puddle', x - w / 2 + 2, y - 2, w - 4)
        run(runs, 'puddle', x - w / 2, y, w)
        run(runs, 'puddle', x - w / 2 + 4, y + 2, w - 8)
        run(runs, 'glint', x - w / 2 + 4, y, 4)
      }
      // A post with the route's blaze stands beside the trail every so often.
      if (Math.round(distance) % 360 === 180 && !wet) {
        const px = x + nx * 10
        const py = y + ny * 5
        const tall = even(8 + depth * 8)
        run(runs, 'post', px, py - tall, 2, tall)
        run(runs, `blaze-${route}`, px - 1, py - tall, 4, 4)
      }
      // Leaves drift onto the trail and pile in drifts at its edges.
      const pile = random() < 0.012 ? 5 : random() < 0.07 ? 1 : 0
      const side = random() < 0.5 ? -1 : 1
      const spread = 6 + random() * 6
      for (let leaf = 0; leaf < pile; leaf++)
        run(
          runs,
          leaves[Math.floor(random() * leaves.length)],
          x + nx * spread * side + (pile > 1 ? (random() - 0.5) * 8 : 0),
          y + ny * spread * 0.5 * side + (random() - 0.5) * (pile > 1 ? 4 : 6),
          2,
        )
    })
    if (bench) {
      // A bench on the highest rise, set back from the trail and facing down the meadow.
      const b: Stop & { tall: number } = bench
      const x = b.x - b.dy * 16
      const y = b.y + b.dx * 8
      run(runs, 'bench-back', x - 8, y - 8, 16)
      run(runs, 'bench', x - 10, y - 4, 20)
      run(runs, 'bench-leg', x - 8, y - 2, 2, 4)
      run(runs, 'bench-leg', x + 6, y - 2, 2, 4)
    }
    // The trailhead signpost points down the route, with its blaze on the arm.
    const head = stops.find(stop => stop.y > alpine.base + 8)
    if (head) signpost(runs, head, route)
    // A junction: where this route first meets one walked before it, away from camp.
    for (const stop of stops) {
      if (Math.hypot(stop.x - camp.x, stop.y - camp.y) < 140 || stop.y < alpine.base + 8) continue
      const meets = walked.some(other =>
        other.some(point => Math.hypot(point.x - stop.x, point.y - stop.y) < 4),
      )
      if (meets) {
        signpost(runs, stop, route)
        break
      }
    }
    walked.push(stops.filter((_, index) => index % 2 === 0))
  })
  return runs.size
    ? `<g class="trail-wayside" aria-hidden="true">${[...runs.entries()]
        .map(([tone, list]) => `<path class="trail-wayside-${tone}" d="${list.join('')}"/>`)
        .join('')}</g>`
    : ''
}

function signpost(runs: Runs, stop: Stop, route: number) {
  const x = stop.x - stop.dy * 10
  const y = stop.y + stop.dx * 5
  const toward = stop.dx >= 0 ? 1 : -1
  run(runs, 'post', x, y - 16, 2, 16)
  run(runs, 'sign', toward > 0 ? x : x - 10, y - 14, 12, 4)
  run(runs, 'sign', toward > 0 ? x + 12 : x - 12, y - 12, 2)
  run(runs, `blaze-${route}`, toward > 0 ? x + 2 : x - 8, y - 12, 4)
}
