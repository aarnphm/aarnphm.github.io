import { bayerThreshold } from '../../util/bayer'
import { pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

// Dappled sunlight over the groves. On the ground, under the trees: the canopy's shade, cast from
// the crowns along the sun's direction, with sun flecks punched through it where the leaves part.
// Over the crowns: small warm flecks on the sunward side. Everything is a dithered blob on the
// scene's pixel grid, so the shade is rebuilt from cells, never skewed, when the sun moves.
// The shade is two Bayer screens stacked, a wide soft one and a narrow deep one, so it steps
// through three tones toward its middle the way a crown does, and is never a solid fill.
type Blob = Point & { rx: number; ry: number; cover: number }
// Where the sun is, in the terms the foot shadows use: the cast direction on the ground (`dx`
// across, `dy` down the screen, toward the viewer) and how low the sun is (0 high, 1 on the
// horizon), which sets how far the crown's shade reaches from its foot.
export type SunCast = { dx: number; dy: number; low: number }
// A sun 25° left of the view, a third of the way down: the foot shadows' default when the page
// has no daylight clock.
export const defaultSun: SunCast = { dx: 0.42, dy: 0.32, low: 0.35 }
type Sweep = { from: number; to: number; dx: number; dy: number }

// Cells of a blob swept along a direction: the crown is a volume, and every level of it casts the
// same footprint a little further along the sun's line, so the shade is the footprint dragged
// from the crown's bottom to its top. `keep` drops cells that may not take shade (the pond).
function blobRuns(
  blob: Blob,
  pixel: number,
  phase: number,
  keep: (point: Point) => boolean = () => true,
  sweep: Sweep = { from: 0, to: 0, dx: 0, dy: 0 },
  // Covers in [0, 1] at the blob's centre and rim; `cover` above is the centre value.
  falloff = 0.8,
) {
  const marks: string[] = []
  const reachX = sweep.dx * sweep.to
  const reachY = sweep.dy * sweep.to
  const left = Math.floor((blob.x + Math.min(0, sweep.dx * sweep.from, reachX) - blob.rx) / pixel)
  const right = Math.ceil((blob.x + Math.max(0, sweep.dx * sweep.from, reachX) + blob.rx) / pixel)
  const top = Math.floor((blob.y + Math.min(0, sweep.dy * sweep.from, reachY) - blob.ry) / pixel)
  const bottom = Math.ceil((blob.y + Math.max(0, sweep.dy * sweep.from, reachY) + blob.ry) / pixel)
  // The sweep segment in the ellipse's own metric, so the nearest point on it is a projection.
  const vx = sweep.dx / blob.rx
  const vy = sweep.dy / blob.ry
  const vv = vx * vx + vy * vy
  for (let row = top; row <= bottom; row++) {
    const cy = row * pixel + pixel / 2
    let start = -1
    for (let column = left; column <= right + 1; column++) {
      const cx = column * pixel + pixel / 2
      const ux = (cx - blob.x) / blob.rx
      const uy = (cy - blob.y) / blob.ry
      const t = vv > 0 ? Math.max(sweep.from, Math.min(sweep.to, (ux * vx + uy * vy) / vv)) : 0
      const d = Math.hypot(ux - vx * t, uy - vy * t)
      const inside =
        column <= right &&
        d < 1 &&
        bayerThreshold(column + phase, row + phase * 3) < blob.cover * (1 - d * d) ** falloff &&
        keep({ x: cx, y: cy })
      if (inside && start < 0) start = column
      if (!inside && start >= 0) {
        const length = (column - start) * pixel
        marks.push(`M${start * pixel} ${row * pixel}h${length}v${pixel}h-${length}Z`)
        start = -1
      }
    }
  }
  return marks.join('')
}

export function trailSunlight(landscape: Landscape, seed: number, sun: SunCast = defaultSun) {
  const { width, groves } = landscape
  const pixel = width < 700 ? 2 : 4
  const random = trailRandom(seed ^ 0x5f3a9c1d)
  const ground: string[] = []
  const canopy: string[] = []
  // Shade and flecks fall on the land; the pond keeps its own surface.
  const land = (point: Point) => pondDistance(point, landscape.pond) > 1.08
  // How far the top of a crown throws its shade: short under a high sun, up to most of a grove
  // under a low one. The crown's bottom starts the sweep a third of the way along.
  const reach = 10 + sun.low * 62
  const sweep: Sweep = { from: reach * 0.3, to: reach, dx: sun.dx, dy: sun.dy }
  // A unit toward the sun along the ground, for the lit side of the crowns.
  const length = Math.hypot(sun.dx, sun.dy) || 1
  const toSunX = -sun.dx / length
  const toSunY = -sun.dy / length
  groves.forEach((grove, index) => {
    const soft: string[] = []
    const deep: string[] = []
    const flecks: string[] = []
    const lit: string[] = []
    const lobes = 2 + Math.floor(random() * 2)
    for (let i = 0; i < lobes; i++) {
      const lobe = {
        x: grove.x + (random() - 0.5) * grove.rx * 0.7,
        y: grove.y + (random() - 0.4) * grove.ry * 0.5,
        rx: grove.rx * (0.5 + random() * 0.3),
        ry: grove.ry * (0.4 + random() * 0.25),
      }
      soft.push(blobRuns({ ...lobe, cover: 0.62 }, pixel, i * 2, land, sweep, 0.6))
      deep.push(
        blobRuns(
          { ...lobe, rx: lobe.rx * 0.8, ry: lobe.ry * 0.8, cover: 0.4 },
          pixel,
          i * 2 + 1,
          land,
          sweep,
          1.2,
        ),
      )
    }
    // Flecks are gaps in the canopy, so they fall where the canopy's middle casts.
    const count = 6 + Math.floor(random() * 4)
    for (let i = 0; i < count; i++) {
      const angle = random() * Math.PI * 2
      const spread = Math.sqrt(random()) * 0.8
      const along = reach * (0.35 + random() * 0.55)
      const x = grove.x + Math.cos(angle) * grove.rx * spread + sun.dx * along
      const y = grove.y + Math.sin(angle) * grove.ry * spread + sun.dy * along
      const rx = 8 + random() * 14
      flecks.push(
        blobRuns({ x, y, rx, ry: rx * (0.4 + random() * 0.2), cover: 0.95 }, pixel, i + 1, land),
      )
    }
    // Light on the crowns: small flecks across the upper half of the grove, shifted to the side
    // the sun is on.
    const lamps = 6 + Math.floor(random() * 4)
    for (let i = 0; i < lamps; i++) {
      const x = grove.x + toSunX * grove.rx * 0.3 + (random() - 0.5) * grove.rx * 1.5
      const y = grove.y - grove.ry * (0.5 + random() * 1.1) + toSunY * grove.ry * 0.2
      const rx = 10 + random() * 12
      lit.push(blobRuns({ x, y, rx, ry: rx * (0.5 + random() * 0.3), cover: 0.85 }, pixel, i + 3))
    }
    const delay = (-random() * 11).toFixed(2)
    const style = `--trail-sunlight-delay:${delay}s;--trail-sunlight-phase:${(index % 2 ? -1 : 1).toFixed(0)}`
    ground.push(
      `<g class="trail-sunlight-grove" style="${style}"><path class="trail-leaf-shade" d="${soft.join('')}"/><path class="trail-leaf-shade trail-leaf-shade-deep" d="${deep.join('')}"/><path class="trail-sunfleck" d="${flecks.join('')}"/></g>`,
    )
    canopy.push(
      `<g class="trail-sunlight-grove" style="${style}"><path class="trail-crown-fleck" d="${lit.join('')}"/></g>`,
    )
  })
  return { ground: ground.join(''), canopy: canopy.join('') }
}
