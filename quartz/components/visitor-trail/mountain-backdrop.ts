import type { Alpine, Landscape, Point } from './landscape'
import { carvedTrails } from './carved-trail'
import {
  buildMountainRanges,
  buildMountainTracks,
  contains,
  heightAt,
  mountainProfileX,
  mountainRangesAt,
  mountainTerrain,
  type MountainRange,
} from './mountain-profile'
import { trailRandom } from './random'

type RockTone =
  | 'rock'
  | 'shade'
  | 'light'
  | 'snow'
  | 'fracture'
  | 'lichen'
  | 'basalt'
  | 'moss'
  | 'ochre'
  | 'rust'
  | 'ash'
type FoothillTone = 'far' | 'mid' | 'near'
// Snowy Alpine ranges sit behind the lower basalt gullies and sedimentary shelves.
type Geology = 'alpine' | 'fluted' | 'banded'
type MountainColumn = {
  range: MountainRange
  x: number
  crest: number
  floor: number
  far: number
  near: number
  edge: number
  verticalScale: number
  grain: number
  // How the crest runs over this column in screen space: negative rises to the right, so the face
  // below it turns toward the light from the upper left.
  slope: number
  geology: Geology
}

const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const alpineTones: readonly RockTone[] = ['rock', 'shade', 'light', 'snow', 'fracture', 'lichen']
const lowlandTones: readonly RockTone[] = [
  'rock',
  'shade',
  'light',
  'fracture',
  'basalt',
  'moss',
  'ochre',
  'rust',
  'ash',
]
const strata: readonly RockTone[] = [
  'ochre',
  'rust',
  'ash',
  'light',
  'rust',
  'ochre',
  'ash',
  'rust',
]

function geologyOf(range: MountainRange): Geology {
  if (range.tier !== 'foreground') return 'alpine'
  return range.form === 'limestone-wall' ? 'banded' : 'fluted'
}

// Rock below the crest, read as a slope face: t runs 0 at the crest to 1 at the range floor, and
// `crestY` is the crest in screen space. Light comes from the upper left.
function sedimentary(
  surface: MountainColumn,
  worldX: number,
  worldY: number,
  crestY: number,
  t: number,
  threshold: number,
): RockTone | undefined {
  const { geology, slope, range } = surface
  const depth = worldY - crestY
  const lit = slope < -0.12 ? 1 : slope > 0.12 ? -1 : 0
  if (geology === 'fluted') {
    // Gullies run straight downhill and widen into talus toward the foot.
    const lean = Math.max(-0.7, Math.min(0.7, -slope * 0.8))
    const u = worldX + depth * lean
    const phase = u / (14 + t * 14) + Math.sin(u / 53 + range.phase) * 0.8
    const rib = Math.sin(phase * Math.PI * 2)
    const facing = Math.cos(phase * Math.PI * 2)
    if (depth < 3) return 'light'
    if (rib < -0.8) return t < 0.4 ? 'light' : 'fracture'
    if (t > 0.62 && threshold < (t - 0.62) * 2.4) return 'moss'
    if (facing > 0.35 + lit * 0.2) return threshold < 0.5 ? 'rock' : 'shade'
    if (facing < -0.35 + lit * 0.2) return 'basalt'
    return lit > 0 ? 'rock' : 'shade'
  }
  if (geology === 'banded') {
    if (depth < 5) return threshold < 0.25 ? 'shade' : 'ash'
    const layer = worldY + Math.sin(worldX / 41 + range.phase) * 3 + Math.sin(worldX / 13) * 1.2
    const band = strata[((Math.floor(layer / 9) % strata.length) + strata.length) % strata.length]
    // Rills cut the soft layers into vertical flutes; a rill's shaded wall faces right.
    const rill = Math.sin(worldX / 4.5 + Math.sin(worldY / 17) * 2)
    if (rill > 0.82 - t * 0.2) return lit < 0 ? 'basalt' : 'shade'
    if (lit < 0 && threshold < 0.375) return 'shade'
    return band
  }
  return undefined
}

const foothillTones: readonly FoothillTone[] = ['far', 'mid', 'near']

function coordinate(value: number) {
  return Math.round(value * 100) / 100
}

// Horizontal runs keep the pixel detail in a fixed number of SVG paint layers.
function pixelLayers<Tone extends string>(
  alpine: Alpine,
  tones: readonly Tone[],
  classPrefix: string,
  sample: (x: number, y: number, column: number, row: number) => Tone | undefined,
) {
  // Paint beyond the clip edge to cover the perspective controller's lateral movement.
  const bleed = 8
  const paintWidth = alpine.width + bleed * 2
  const pixel = alpine.width < 700 ? 2 : 4
  const columns = Math.ceil(paintWidth / pixel)
  const rows = Math.ceil(alpine.height / pixel)
  const marks = new Map<Tone, string[]>()
  for (const tone of tones) marks.set(tone, [])

  for (let row = 0; row < rows; row++) {
    const top = row * pixel
    const height = Math.min(pixel, alpine.height - top)
    let runTone: Tone | undefined
    let start = -bleed
    for (let column = 0; column <= columns; column++) {
      const left = column * pixel - bleed
      const width = Math.min(pixel, alpine.width + bleed - left)
      const tone =
        column === columns
          ? undefined
          : sample(
              alpine.left + Math.max(0, Math.min(alpine.width, left + width / 2)),
              alpine.top + top,
              column,
              row,
            )
      if (tone === runTone) continue
      if (runTone) {
        const length = coordinate(Math.min(left, alpine.width + bleed) - start)
        marks.get(runTone)?.push(`M${start} ${top}h${length}v${coordinate(height)}h-${length}Z`)
      }
      runTone = tone
      start = left
    }
  }

  return tones
    .map(
      (tone, index) =>
        `<path class="trail-mountain-paint ${classPrefix}-${tone}" style="--trail-paint-step: ${index}" d="${marks.get(tone)?.join('') ?? ''}"/>`,
    )
    .join('')
}

export function trailMountainBackdrop(
  landscape: Landscape,
  seed: number,
  clipId: string,
  entrances?: readonly Point[],
): string {
  const { alpine } = landscape
  const ranges = buildMountainRanges(alpine, seed)
  const columns = new Map<number, MountainColumn[]>()
  const rangesInColumn = (x: number, column: number) => {
    const existing = columns.get(column)
    if (existing) return existing
    const visible = mountainRangesAt(x, ranges).map(range => {
      const profileX = mountainProfileX(x, range)
      const terrain = mountainTerrain(range)
      return {
        range,
        x: profileX,
        crest: heightAt(terrain.crest, profileX),
        floor: heightAt(terrain.floor, profileX),
        far: heightAt(terrain.far, profileX),
        near: heightAt(terrain.near, profileX),
        edge: 350,
        verticalScale: 350 / range.height,
        grain: Math.sin(profileX / 39 + range.phase),
        slope:
          ((heightAt(terrain.crest, profileX + 10) - heightAt(terrain.crest, profileX - 10)) / 20) *
          (range.mirrored ? -1 : 1),
        geology: geologyOf(range),
      }
    })
    columns.set(column, visible)
    return visible
  }
  const random = trailRandom(seed ^ 0x2b992ddf)
  const phaseX = Math.floor(random() * 4)
  const phaseY = Math.floor(random() * 4)
  const thresholdAt = (column: number, row: number, range: MountainRange) =>
    bayer[((row + phaseY + range.index) % 4) * 4 + ((column + phaseX + range.index * 3) % 4)] / 16

  const surfaceAt = (worldX: number, worldY: number, column: number, alpineFace: boolean) =>
    rangesInColumn(worldX, column).find(candidate => {
      const y = 350 + (worldY - candidate.range.base) * candidate.verticalScale
      return (
        (candidate.geology === 'alpine') === alpineFace &&
        y >= candidate.crest &&
        y <= candidate.floor
      )
    })
  const bleedDepth = Math.min(100, Math.max(48, (landscape.camp.y - alpine.base) * 0.24))
  const lowlandBounds = { ...alpine, height: alpine.height + bleedDepth }
  const scatterAt = (column: number, row: number, range: MountainRange) =>
    trailRandom(
      seed ^ Math.imul(column + 1, 374761393) ^ Math.imul(row + range.index * 31, 668265263),
    )()
  const apronAt = (worldX: number, worldY: number, column: number, row: number) =>
    rangesInColumn(worldX, column).find(surface => {
      if (surface.geology === 'alpine') return false
      const foot = surface.range.base + (surface.floor - 350) / surface.verticalScale
      const reach =
        bleedDepth *
        (0.6 + Math.sin(worldX / 73 + surface.range.phase) * 0.2 + Math.sin(worldX / 29) * 0.12)
      const distance = worldY - foot
      if (distance < 0 || distance >= reach) return false
      // Rock fans thin into scattered stones and moss instead of ending at a level paint edge.
      const cover = (1 - distance / reach) ** 1.5
      return scatterAt(column, row, surface.range) < cover
    })
  const ridge = pixelLayers(
    alpine,
    alpineTones,
    'trail-mountain',
    (worldX, worldY, column, row) => {
      const surface = surfaceAt(worldX, worldY, column, true)
      if (!surface) return undefined
      const { range, x } = surface
      const terrain = mountainTerrain(range)
      const y = 350 + (worldY - range.base) * surface.verticalScale
      const threshold = thresholdAt(column, row, range)
      const grain = surface.grain * Math.sin(y / 24 + x / 88)
      // Alpine faces catch a rim of light along the crest on the sunward side.
      if (y - surface.crest < 4 && surface.slope < -0.1) return 'light'
      if (terrain.snow.some(field => contains(x, y, field))) {
        if (y - surface.crest < 9) return 'light'
        return threshold < 0.0625 && grain > 0.2 ? 'light' : 'snow'
      }
      if (terrain.strata.some(shelf => contains(x, y, shelf))) {
        return threshold < 0.75 ? (range.depth < 0.88 ? 'shade' : 'fracture') : 'shade'
      }
      if (y > 243 && threshold < 0.125 && grain > 0.35) return 'lichen'
      const shaded = terrain.shade.some(face => contains(x, y, face))
      if (y > 328) return threshold < 0.125 ? 'rock' : 'light'
      if (threshold < 0.125 && grain > -0.2) return shaded ? 'rock' : 'light'
      if (threshold > 0.8125 && grain < -0.35) return shaded ? 'fracture' : 'shade'
      return shaded ? (range.depth < 0.88 ? 'rock' : 'shade') : 'rock'
    },
  )

  const lowlands = pixelLayers(
    lowlandBounds,
    lowlandTones,
    'trail-mountain',
    (worldX, worldY, column, row) => {
      const face = surfaceAt(worldX, worldY, column, false)
      const surface = face ?? apronAt(worldX, worldY, column, row)
      if (!surface) return undefined
      const y = 350 + (worldY - surface.range.base) * surface.verticalScale
      const crestY = surface.range.base + (surface.crest - 350) / surface.verticalScale
      const t = (y - surface.crest) / Math.max(1, surface.floor - surface.crest)
      return sedimentary(
        surface,
        worldX,
        worldY,
        crestY,
        face ? t : Math.min(0.9, t),
        face
          ? thresholdAt(column, row, surface.range)
          : scatterAt(column + 3, row + 1, surface.range),
      )
    },
  )

  const foothills = pixelLayers(
    lowlandBounds,
    foothillTones,
    'trail-foothill',
    (worldX, worldY, column, row) => {
      const surface = rangesInColumn(worldX, column).find(candidate => {
        const y = 350 + (worldY - candidate.range.base) * candidate.verticalScale
        return candidate.range.tier === 'foreground' && y >= candidate.far && y <= candidate.edge
      })
      if (!surface) {
        const apron = apronAt(worldX, worldY, column, row)
        if (!apron) return undefined
        const threshold = scatterAt(column + 2, row + 1, apron.range)
        return threshold < 0.1875 ? 'mid' : undefined
      }
      const { range, far, near, edge } = surface
      const y = 350 + (worldY - range.base) * surface.verticalScale
      const threshold = thresholdAt(column + 2, row + 1, range)
      // Moss hummocks, like an Icelandic lava field: each mound is shaded on its lower right.
      const mound =
        Math.sin(worldX / 9 + Math.sin(worldY / 7) * 1.6) * Math.sin(worldY / 4.5 + worldX / 31)
      if (y >= near) {
        if (y > edge - 8 && threshold < 0.25) return undefined
        return threshold < 0.125 || (mound > 0.55 && threshold < 0.6) ? 'mid' : 'near'
      }
      if (y > far + 6 && mound > 0.6 && threshold < 0.5) return 'mid'
      return threshold < 0.125 && y > far + 14 ? 'mid' : 'far'
    },
  )
  const placement = `translate(${coordinate(alpine.left)} ${coordinate(alpine.top)})`
  const atlas = ranges.map(({ index, form, tier, left, width, height, base, depth }) => ({
    index,
    form,
    tier,
    geology: geologyOf(ranges[index]),
    left,
    width,
    height,
    base,
    depth,
  }))
  const tracks = buildMountainTracks(alpine, seed, entrances)
  // The route paths carry the walking geometry; the carved bench underneath is what shows.
  const routes = tracks
    .map(track => {
      const [first, ...rest] = track.points
      const path = `M${coordinate(first.x)} ${coordinate(first.y)}${rest.map(point => `L${coordinate(point.x)} ${coordinate(point.y)}`).join('')}`
      return `<path class="trail-mountain-route" data-mountain-track="${track.id}" data-mountain-range="${track.rangeIndex}" data-mountain-kind="${track.kind}" data-mountain-depth="${coordinate(track.depth)}" d="${path}"/>`
    })
    .join('')

  return `<g class="trail-backdrop" data-trail-backdrop="" data-backdrop-placement="below-text" clip-path="url(#${clipId})">
      <g data-trail-depth="alpine">
        <g data-formation="bernese-alps" data-peaks="eiger monch jungfrau" data-mountain-ranges="${ranges.length}" data-mountain-atlas='${JSON.stringify(atlas)}' transform="${placement}">
          ${ridge}
        </g>
        <g class="trail-alpine-climbers" aria-hidden="true"></g>
      </g>
      <g data-trail-depth="foothill">
        <g data-formation="foothills" data-badlands-bleed="${coordinate(bleedDepth)}" transform="${placement}">
          ${lowlands}
          ${foothills}
        </g>
        <g class="trail-mountain-roads">${carvedTrails(alpine, ranges, tracks)}${routes}</g>
        <g class="trail-foothill-trees"></g>
        <g class="trail-alpine-tracks"></g>
        <g class="trail-alpine-travelers"></g>
        <g class="trail-foothill-objects"></g>
      </g>
    </g>`
}
