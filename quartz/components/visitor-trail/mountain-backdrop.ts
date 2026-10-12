import type { Alpine, Landscape, Point } from './landscape'
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

type RockTone = 'rock' | 'shade' | 'light' | 'snow' | 'fracture' | 'lichen'
type FoothillTone = 'far' | 'mid' | 'near'
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
}

const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const rockTones: readonly RockTone[] = ['rock', 'shade', 'light', 'snow', 'fracture', 'lichen']
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
  const pixel = Math.max(alpine.width < 700 ? 2 : 4, Math.ceil(paintWidth / 640))
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

  const ridge = pixelLayers(alpine, rockTones, 'trail-mountain', (worldX, worldY, column, row) => {
    const surface = rangesInColumn(worldX, column).find(candidate => {
      const y = 350 + (worldY - candidate.range.base) * candidate.verticalScale
      return y >= candidate.crest && y <= candidate.floor
    })
    if (!surface) return undefined
    const { range, x } = surface
    const terrain = mountainTerrain(range)
    const y = 350 + (worldY - range.base) * surface.verticalScale
    const threshold = thresholdAt(column, row, range)
    const grain = surface.grain * Math.sin(y / 24 + x / 88)
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
  })

  const foothills = pixelLayers(
    alpine,
    foothillTones,
    'trail-foothill',
    (worldX, worldY, column, row) => {
      const surface = rangesInColumn(worldX, column).find(candidate => {
        const y = 350 + (worldY - candidate.range.base) * candidate.verticalScale
        return y >= candidate.far && y <= candidate.edge
      })
      if (!surface) return undefined
      const { range, far, near, edge } = surface
      const y = 350 + (worldY - range.base) * surface.verticalScale
      const threshold = thresholdAt(column + 2, row + 1, range)
      if (y >= near) {
        if (y > edge - 8 && threshold < 0.25) return undefined
        return threshold < 0.125 ? 'mid' : 'near'
      }
      return threshold < 0.125 && y > far + 14 ? 'mid' : 'far'
    },
  )
  const placement = `translate(${coordinate(alpine.left)} ${coordinate(alpine.top)})`
  const atlas = ranges.map(({ index, form, tier, left, width, height, base, depth }) => ({
    index,
    form,
    tier,
    left,
    width,
    height,
    base,
    depth,
  }))
  const tracks = buildMountainTracks(alpine, seed, entrances)
    .map(track => {
      const [first, ...rest] = track.points
      const path = `M${coordinate(first.x)} ${coordinate(first.y)}${rest.map(point => `L${coordinate(point.x)} ${coordinate(point.y)}`).join('')}`
      return `<path class="trail-mountain-track" data-mountain-track="${track.id}" data-mountain-range="${track.rangeIndex}" data-mountain-kind="${track.kind}" data-mountain-depth="${coordinate(track.depth)}" d="${path}"/>`
    })
    .join('')

  return `<g class="trail-backdrop" data-trail-backdrop="" data-backdrop-placement="below-text" clip-path="url(#${clipId})">
      <g data-trail-depth="alpine">
        <g data-formation="bernese-alps" data-peaks="eiger monch jungfrau" data-mountain-ranges="${ranges.length}" data-mountain-atlas='${JSON.stringify(atlas)}' transform="${placement}">
          ${ridge}
        </g>
      </g>
      <g data-trail-depth="foothill">
        <g data-formation="foothills" transform="${placement}" style="--trail-mountain-delay: 900ms">
          ${foothills}
        </g>
        <g class="trail-mountain-roads">${tracks}</g>
        <g class="trail-foothill-trees"></g>
        <g class="trail-alpine-tracks"></g>
        <g class="trail-alpine-travelers"></g>
        <g class="trail-foothill-objects"></g>
      </g>
    </g>`
}
