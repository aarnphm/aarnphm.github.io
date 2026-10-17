import { bayerThreshold } from '../../util/bayer'
import { pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

// Dappled sunlight over the groves. On the ground, under the trees: the canopy's shade, with sun
// flecks punched through it where the leaves part. Over the crowns: small warm flecks where the
// sun catches the top of the canopy. Each is a dithered blob, dense at the centre and thinning
// to scattered cells at the rim, so the light reads as broken by leaves. The ground is pale, so
// the shade carries the effect and the flecks read as gaps in it, as they do under a real tree.
// A grove's light sways together with the wind, as the canopy above it would.
type Blob = Point & { rx: number; ry: number; cover: number }

function blobRuns(
  blob: Blob,
  pixel: number,
  phase: number,
  keep: (point: Point) => boolean = () => true,
) {
  const marks: string[] = []
  const top = Math.floor((blob.y - blob.ry) / pixel)
  const bottom = Math.ceil((blob.y + blob.ry) / pixel)
  const left = Math.floor((blob.x - blob.rx) / pixel)
  const right = Math.ceil((blob.x + blob.rx) / pixel)
  for (let row = top; row <= bottom; row++) {
    const cy = row * pixel + pixel / 2
    let start = -1
    for (let column = left; column <= right + 1; column++) {
      const cx = column * pixel + pixel / 2
      const d = Math.hypot((cx - blob.x) / blob.rx, (cy - blob.y) / blob.ry)
      const inside =
        column <= right &&
        d < 1 &&
        bayerThreshold(column + phase, row + phase * 3) < blob.cover * (1 - d * d) ** 0.8 &&
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

export function trailSunlight(landscape: Landscape, seed: number) {
  const { width, groves } = landscape
  const pixel = width < 700 ? 2 : 4
  const random = trailRandom(seed ^ 0x5f3a9c1d)
  const ground: string[] = []
  const canopy: string[] = []
  // Shade and flecks fall on the land; the pond keeps its own surface.
  const land = (point: Point) => pondDistance(point, landscape.pond) > 1.08
  groves.forEach((grove, index) => {
    const flecks: string[] = []
    const shade: string[] = []
    const lit: string[] = []
    // The sun comes from the upper left; the canopy's shade lies down and to the right of the
    // grove, in two or three overlapping lobes rather than one ellipse.
    const lobes = 2 + Math.floor(random() * 2)
    for (let i = 0; i < lobes; i++) {
      shade.push(
        blobRuns(
          {
            x: grove.x + 10 + (random() - 0.5) * grove.rx * 0.8,
            y: grove.y + 8 + (random() - 0.3) * grove.ry * 0.6,
            rx: grove.rx * (0.7 + random() * 0.4),
            ry: grove.ry * (0.55 + random() * 0.3),
            cover: 0.72,
          },
          pixel,
          i * 2,
          land,
        ),
      )
    }
    // Flecks fall inside the shade, where the leaves part.
    const count = 6 + Math.floor(random() * 4)
    for (let i = 0; i < count; i++) {
      const angle = random() * Math.PI * 2
      const reach = Math.sqrt(random()) * 0.9
      const x = grove.x + 10 + Math.cos(angle) * grove.rx * reach
      const y = grove.y + 8 + Math.sin(angle) * grove.ry * reach
      const rx = 8 + random() * 18
      flecks.push(
        blobRuns({ x, y, rx, ry: rx * (0.4 + random() * 0.2), cover: 0.9 }, pixel, i + 1, land),
      )
    }
    // Light on the crowns: small flecks across the upper half of the grove, on the sunward side.
    const lamps = 5 + Math.floor(random() * 4)
    for (let i = 0; i < lamps; i++) {
      const x = grove.x - grove.rx * 0.2 + (random() - 0.5) * grove.rx * 1.6
      const y = grove.y - grove.ry * (0.5 + random() * 1.1)
      const rx = 6 + random() * 12
      lit.push(blobRuns({ x, y, rx, ry: rx * (0.5 + random() * 0.3), cover: 0.7 }, pixel, i + 3))
    }
    const delay = (-random() * 11).toFixed(2)
    const style = `--trail-sunlight-delay:${delay}s;--trail-sunlight-phase:${(index % 2 ? -1 : 1).toFixed(0)}`
    ground.push(
      `<g class="trail-sunlight-grove" style="${style}"><path class="trail-leaf-shade" d="${shade.join('')}"/><path class="trail-sunfleck" d="${flecks.join('')}"/></g>`,
    )
    canopy.push(
      `<g class="trail-sunlight-grove" style="${style}"><path class="trail-crown-fleck" d="${lit.join('')}"/></g>`,
    )
  })
  return { ground: ground.join(''), canopy: canopy.join('') }
}
