import type { Alpine, Landscape } from './landscape'
import { trailRandom } from './random'

type Vertex = readonly [number, number]
type Profile = readonly Vertex[]
type Region = { profile: Profile; left: number; top: number; right: number; bottom: number }
type RockTone = 'rock' | 'shade' | 'light' | 'snow' | 'fracture' | 'lichen'
type FoothillTone = 'far' | 'mid' | 'near'

const ridgeProfile: Profile = [
  [0, 298],
  [35, 279],
  [63, 254],
  [78, 262],
  [104, 221],
  [128, 211],
  [146, 191],
  [162, 180],
  [174, 150],
  [187, 155],
  [199, 143],
  [213, 150],
  [232, 127],
  [245, 112],
  [250, 115],
  [267, 91],
  [280, 88],
  [294, 76],
  [304, 73],
  [310, 78],
  [321, 91],
  [345, 106],
  [350, 120],
  [369, 138],
  [377, 138],
  [399, 165],
  [425, 165],
  [448, 149],
  [465, 138],
  [480, 118],
  [491, 118],
  [510, 95],
  [517, 93],
  [535, 79],
  [541, 67],
  [550, 58],
  [566, 60],
  [582, 72],
  [594, 87],
  [605, 92],
  [614, 111],
  [627, 116],
  [642, 141],
  [657, 142],
  [673, 167],
  [691, 163],
  [709, 145],
  [718, 146],
  [736, 116],
  [749, 105],
  [760, 86],
  [777, 78],
  [794, 64],
  [805, 66],
  [818, 55],
  [833, 47],
  [842, 30],
  [854, 35],
  [863, 44],
  [877, 47],
  [886, 53],
  [900, 59],
  [912, 80],
  [925, 88],
  [935, 110],
  [949, 116],
  [957, 137],
  [973, 136],
  [990, 154],
  [1005, 166],
  [1018, 171],
  [1035, 199],
  [1053, 195],
  [1073, 227],
  [1090, 229],
  [1104, 248],
  [1124, 254],
  [1141, 279],
  [1165, 278],
  [1200, 309],
]

const valleyEdge: Profile = [
  [1150, 322],
  [1128, 312],
  [1080, 326],
  [1034, 317],
  [987, 329],
  [946, 323],
  [901, 337],
  [856, 332],
  [834, 323],
  [804, 334],
  [761, 324],
  [701, 340],
  [671, 331],
  [650, 333],
  [595, 324],
  [560, 338],
  [493, 329],
  [442, 334],
  [396, 323],
  [367, 332],
  [326, 325],
  [289, 336],
  [247, 323],
  [221, 332],
  [181, 318],
  [149, 324],
  [120, 316],
  [86, 327],
  [58, 316],
  [22, 325],
]

const farFoothill: Profile = [
  [0, 309],
  [42, 288],
  [75, 293],
  [103, 276],
  [145, 287],
  [179, 264],
  [201, 271],
  [231, 253],
  [259, 265],
  [291, 256],
  [320, 271],
  [352, 265],
  [389, 277],
  [417, 266],
  [449, 279],
  [482, 270],
  [510, 281],
  [544, 267],
  [574, 280],
  [605, 263],
  [630, 279],
  [663, 267],
  [691, 282],
  [724, 266],
  [758, 274],
  [787, 256],
  [811, 267],
  [844, 249],
  [873, 259],
  [904, 254],
  [928, 271],
  [960, 266],
  [991, 285],
  [1024, 273],
  [1060, 289],
  [1092, 281],
  [1126, 303],
  [1158, 294],
  [1200, 316],
]

const nearFoothill: Profile = [
  [0, 331],
  [42, 318],
  [81, 329],
  [111, 309],
  [143, 323],
  [183, 308],
  [211, 319],
  [247, 298],
  [271, 307],
  [302, 295],
  [335, 315],
  [363, 306],
  [401, 326],
  [433, 317],
  [471, 333],
  [504, 323],
  [540, 331],
  [574, 308],
  [608, 320],
  [640, 309],
  [677, 326],
  [716, 317],
  [751, 330],
  [786, 312],
  [820, 319],
  [853, 300],
  [884, 314],
  [921, 295],
  [953, 309],
  [991, 301],
  [1025, 319],
  [1058, 311],
  [1094, 330],
  [1134, 320],
  [1170, 336],
  [1200, 326],
]

const snowfields: readonly Profile[] = [
  [
    [267, 91],
    [280, 88],
    [294, 76],
    [304, 73],
    [310, 78],
    [321, 91],
    [345, 106],
    [350, 120],
    [369, 138],
    [356, 140],
    [345, 134],
    [330, 127],
    [321, 109],
    [313, 114],
    [307, 105],
    [301, 99],
    [296, 116],
    [288, 107],
    [281, 118],
    [273, 113],
    [266, 119],
    [253, 115],
  ],
  [
    [480, 118],
    [491, 118],
    [510, 95],
    [517, 93],
    [535, 79],
    [541, 67],
    [550, 58],
    [566, 60],
    [582, 72],
    [594, 87],
    [605, 92],
    [614, 111],
    [627, 116],
    [642, 141],
    [657, 142],
    [673, 167],
    [659, 178],
    [644, 167],
    [635, 174],
    [620, 155],
    [607, 159],
    [595, 145],
    [584, 148],
    [576, 127],
    [566, 119],
    [563, 100],
    [552, 86],
    [548, 104],
    [542, 111],
    [536, 136],
    [526, 141],
    [522, 158],
    [506, 157],
    [500, 143],
    [489, 145],
    [487, 129],
    [471, 135],
  ],
  [
    [718, 146],
    [736, 116],
    [749, 105],
    [760, 86],
    [777, 78],
    [794, 64],
    [805, 66],
    [818, 55],
    [833, 47],
    [842, 30],
    [854, 35],
    [863, 44],
    [877, 47],
    [886, 53],
    [900, 59],
    [912, 80],
    [925, 88],
    [935, 110],
    [949, 116],
    [957, 137],
    [973, 136],
    [990, 154],
    [1005, 166],
    [1018, 171],
    [1035, 199],
    [1023, 203],
    [1007, 190],
    [997, 191],
    [981, 168],
    [969, 172],
    [955, 157],
    [947, 163],
    [935, 149],
    [923, 148],
    [918, 130],
    [905, 122],
    [900, 112],
    [888, 108],
    [875, 91],
    [866, 99],
    [855, 78],
    [848, 85],
    [841, 69],
    [831, 84],
    [827, 97],
    [817, 103],
    [809, 122],
    [798, 131],
    [790, 148],
    [778, 151],
    [773, 173],
    [758, 181],
    [748, 169],
    [740, 170],
    [733, 156],
  ],
]

const rockFaces: readonly Profile[] = [
  [
    [199, 143],
    [213, 150],
    [232, 127],
    [245, 112],
    [250, 115],
    [267, 91],
    [281, 118],
    [288, 107],
    [296, 116],
    [301, 99],
    [307, 105],
    [302, 138],
    [310, 157],
    [302, 182],
    [309, 206],
    [293, 221],
    [284, 247],
    [261, 262],
    [239, 280],
    [211, 274],
    [217, 250],
    [200, 243],
    [214, 222],
    [207, 201],
    [214, 180],
    [198, 171],
  ],
  [
    [517, 93],
    [535, 79],
    [541, 67],
    [550, 58],
    [548, 104],
    [542, 111],
    [536, 136],
    [526, 141],
    [522, 158],
    [506, 157],
    [500, 143],
    [489, 145],
    [487, 129],
    [471, 135],
    [453, 170],
    [469, 181],
    [458, 207],
    [470, 221],
    [456, 247],
    [493, 263],
    [523, 253],
    [537, 228],
    [526, 206],
    [543, 187],
    [534, 168],
    [548, 147],
    [542, 132],
    [553, 108],
  ],
  [
    [794, 64],
    [805, 66],
    [818, 55],
    [833, 47],
    [842, 30],
    [841, 69],
    [831, 84],
    [827, 97],
    [817, 103],
    [809, 122],
    [798, 131],
    [790, 148],
    [778, 151],
    [773, 173],
    [758, 181],
    [767, 198],
    [751, 218],
    [760, 240],
    [790, 252],
    [827, 265],
    [852, 249],
    [842, 230],
    [854, 211],
    [833, 192],
    [839, 175],
    [823, 156],
    [830, 143],
    [814, 132],
    [824, 114],
    [819, 101],
    [836, 88],
    [833, 72],
  ],
]

const strata: readonly Profile[] = [
  [
    [223, 186],
    [243, 181],
    [254, 188],
    [282, 179],
    [285, 185],
    [257, 197],
    [244, 192],
    [221, 198],
  ],
  [
    [212, 222],
    [237, 215],
    [250, 226],
    [282, 220],
    [278, 228],
    [254, 236],
    [237, 225],
    [218, 232],
  ],
  [
    [468, 193],
    [487, 185],
    [500, 194],
    [516, 189],
    [523, 196],
    [500, 204],
    [487, 196],
    [472, 205],
  ],
  [
    [469, 231],
    [488, 225],
    [503, 233],
    [524, 225],
    [525, 233],
    [505, 245],
    [489, 236],
    [477, 243],
  ],
  [
    [774, 161],
    [793, 156],
    [807, 166],
    [831, 158],
    [828, 165],
    [807, 175],
    [794, 166],
    [775, 173],
  ],
  [
    [769, 204],
    [790, 194],
    [807, 206],
    [832, 197],
    [837, 204],
    [808, 216],
    [791, 205],
    [775, 218],
  ],
  [
    [785, 238],
    [810, 230],
    [826, 238],
    [840, 234],
    [837, 242],
    [825, 249],
    [809, 241],
    [793, 250],
  ],
  [
    [1071, 260],
    [1088, 253],
    [1099, 259],
    [1106, 275],
    [1098, 280],
    [1085, 265],
    [1073, 269],
  ],
]

const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const rockTones: readonly RockTone[] = ['rock', 'shade', 'light', 'snow', 'fracture', 'lichen']
const foothillTones: readonly FoothillTone[] = ['far', 'mid', 'near']
const ridgeBounds = region([...ridgeProfile, ...valleyEdge])
const snowRegions = snowfields.map(region)
const rockRegions = rockFaces.map(region)
const strataRegions = strata.map(region)

function region(profile: Profile): Region {
  const xs = profile.map(([x]) => x)
  const ys = profile.map(([, y]) => y)
  return {
    profile,
    left: Math.min(...xs),
    top: Math.min(...ys),
    right: Math.max(...xs),
    bottom: Math.max(...ys),
  }
}

function contains(x: number, y: number, bounds: Region) {
  if (x < bounds.left || x > bounds.right || y < bounds.top || y > bounds.bottom) return false
  const polygon = bounds.profile
  let inside = false
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [ax, ay] = polygon[index]
    const [bx, by] = polygon[previous]
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside
  }
  return inside
}

function heightAt(profile: Profile, x: number) {
  for (let index = 1; index < profile.length; index++) {
    const [ax, ay] = profile[index - 1]
    const [bx, by] = profile[index]
    if (x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
  }
  return profile[profile.length - 1][1]
}

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
  const pixel = alpine.width < 700 ? 2 : 4
  const columns = Math.ceil(alpine.width / pixel)
  const rows = Math.ceil(alpine.height / pixel)
  const marks = new Map<Tone, string[]>()
  for (const tone of tones) marks.set(tone, [])

  for (let row = 0; row < rows; row++) {
    const top = row * pixel
    const height = Math.min(pixel, alpine.height - top)
    let runTone: Tone | undefined
    let start = 0
    for (let column = 0; column <= columns; column++) {
      const left = column * pixel
      const width = Math.min(pixel, alpine.width - left)
      const tone =
        column === columns
          ? undefined
          : sample(
              ((left + width / 2) / alpine.width) * 1200,
              ((top + height / 2) / alpine.height) * 350,
              column,
              row,
            )
      if (tone === runTone) continue
      if (runTone) {
        const length = coordinate(Math.min(left, alpine.width) - start)
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

export function trailMountainBackdrop(landscape: Landscape, seed: number, clipId: string): string {
  const { alpine } = landscape
  const random = trailRandom(seed ^ 0x2b992ddf)
  const phaseX = Math.floor(random() * 4)
  const phaseY = Math.floor(random() * 4)
  const grainPhase = random() * Math.PI * 2
  const thresholdAt = (column: number, row: number) =>
    bayer[((row + phaseY) % 4) * 4 + ((column + phaseX) % 4)] / 16

  const ridge = pixelLayers(alpine, rockTones, 'trail-mountain', (x, y, column, row) => {
    if (!contains(x, y, ridgeBounds)) return undefined
    const threshold = thresholdAt(column, row)
    const grain = Math.sin(x / 39 + grainPhase) * Math.sin(y / 24 + x / 88)
    const onSnow = snowRegions.some(field => contains(x, y, field))
    if (onSnow) {
      if (y - heightAt(ridgeProfile, x) < 9) return 'light'
      return threshold < 0.0625 && grain > 0.2 ? 'light' : 'snow'
    }
    if (strataRegions.some(shelf => contains(x, y, shelf))) {
      return threshold < 0.75 ? 'fracture' : 'shade'
    }
    if (y > 243 && threshold < 0.125 && grain > 0.35) return 'lichen'
    const shaded = rockRegions.some(face => contains(x, y, face))
    if (threshold < 0.125 && grain > -0.2) return shaded ? 'rock' : 'light'
    if (threshold > 0.8125 && grain < -0.35) return shaded ? 'fracture' : 'shade'
    return shaded ? 'shade' : 'rock'
  })

  const foothills = pixelLayers(alpine, foothillTones, 'trail-foothill', (x, y, column, row) => {
    const far = heightAt(farFoothill, x)
    const near = heightAt(nearFoothill, x)
    if (y < far) return undefined
    const edge = Math.min(348, 340 + Math.sin(x / 67 + grainPhase) * 5 + Math.sin(x / 213) * 4)
    if (y > edge) return undefined
    const threshold = thresholdAt(column + 2, row + 1)
    if (y >= near) {
      if (y > edge - 8 && threshold < 0.25) return undefined
      return threshold < 0.125 ? 'mid' : 'near'
    }
    return threshold < 0.125 && y > far + 14 ? 'mid' : 'far'
  })
  const placement = `translate(${coordinate(alpine.left)} ${coordinate(alpine.top)})`

  return `<g class="trail-backdrop" data-trail-backdrop="" data-backdrop-placement="below-text" clip-path="url(#${clipId})">
      <g data-trail-depth="alpine">
        <g data-formation="bernese-alps" data-peaks="eiger monch jungfrau" transform="${placement}">
          ${ridge}
        </g>
      </g>
      <g data-trail-depth="foothill">
        <g data-formation="foothills" transform="${placement}" style="--trail-mountain-delay: 900ms">
          ${foothills}
        </g>
        <g class="trail-foothill-trees"></g>
        <g class="trail-alpine-tracks"></g>
        <g class="trail-alpine-travelers"></g>
      </g>
    </g>`
}
