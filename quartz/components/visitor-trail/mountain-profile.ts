import type { Alpine } from './landscape'
import { trailRandom } from './random'

export type Vertex = readonly [number, number]
export type Profile = readonly Vertex[]
type Region = { profile: Profile; left: number; top: number; right: number; bottom: number }

export const ridgeProfile: Profile = [
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

export const farFoothill: Profile = [
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

export const nearFoothill: Profile = [
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

const lowerEdge: Profile = valleyEdge.map(([x, y]): Vertex => [x, 341 + (y - 310) * 0.2])
export const mountainFloor: Profile = [[0, 350], ...[...lowerEdge].reverse(), [1200, 350]]
export const snowRegions = snowfields.map(region)
export const rockRegions = rockFaces.map(region)
export const strataRegions = strata.map(region)

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

export function contains(x: number, y: number, bounds: Region) {
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

export function heightAt(profile: Profile, x: number) {
  for (let index = 1; index < profile.length; index++) {
    const [ax, ay] = profile[index - 1]
    const [bx, by] = profile[index]
    if (x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
  }
  return profile[profile.length - 1][1]
}

type MountainForm = 'balanced' | 'high-saddle' | 'broken-ridge'
const forms: readonly MountainForm[] = ['balanced', 'high-saddle', 'broken-ridge']
const heightProfiles: Record<MountainForm, Profile> = {
  balanced: [
    [0, 1],
    [240, 0.96],
    [304, 1.02],
    [550, 0.97],
    [842, 1],
    [1200, 1],
  ],
  'high-saddle': [
    [0, 1],
    [160, 0.96],
    [304, 0.82],
    [440, 0.92],
    [550, 1.04],
    [730, 0.94],
    [842, 0.81],
    [1040, 0.95],
    [1200, 1],
  ],
  'broken-ridge': [
    [0, 1],
    [150, 0.93],
    [304, 0.98],
    [430, 0.92],
    [550, 0.78],
    [720, 0.93],
    [842, 1.04],
    [1060, 0.92],
    [1200, 1],
  ],
}

export type MountainRange = {
  left: number
  top: number
  width: number
  height: number
  base: number
  mirrored: boolean
  depth: number
  phase: number
  index: number
  form: MountainForm
}

export function buildMountainRanges(alpine: Alpine, seed: number): MountainRange[] {
  const random = trailRandom(seed ^ 0x13a5b79d)
  const firstForm = Math.floor(random() * forms.length)
  const overlap = 0.14
  const aspect = 1200 / 350
  const paintWidth = alpine.width + 16
  const count =
    alpine.width < 700
      ? 1
      : Math.max(1, Math.ceil(paintWidth / (alpine.height * aspect * (1 - overlap))))
  const parts = Array.from({ length: count }, (_, index) => ({
    weight: 0.86 + random() * 0.28,
    heightFactor: 0.88 + random() * 0.12,
    mirrored: random() > 0.5,
    phase: random() * Math.PI * 2,
    form: forms[(firstForm + index) % forms.length],
  }))
  const total =
    parts.reduce((sum, part) => sum + part.weight, 0) -
    parts
      .slice(1)
      .reduce((sum, part, index) => sum + overlap * Math.min(part.weight, parts[index].weight), 0)
  const unit = paintWidth / total
  const heights = parts.map(part =>
    Math.min(alpine.height, ((part.weight * unit) / aspect) * part.heightFactor),
  )
  const nearestHeight = Math.max(...heights)
  let left = alpine.left - 8

  return parts.map((part, index) => {
    const width = part.weight * unit
    const height = heights[index]
    const range: MountainRange = {
      left,
      top: alpine.base - height,
      width,
      height,
      base: alpine.base,
      mirrored: part.mirrored,
      depth: height / nearestHeight,
      phase: part.phase,
      index,
      form: part.form,
    }
    const next = parts[index + 1]
    left += width - (next ? overlap * Math.min(width, next.weight * unit) : 0)
    return range
  })
}

export function mountainRangesAt(
  worldX: number,
  ranges: readonly MountainRange[],
): MountainRange[] {
  return ranges
    .filter(range => worldX >= range.left && worldX <= range.left + range.width)
    .sort((left, right) => right.depth - left.depth || right.index - left.index)
}

export function mountainProfileX(worldX: number, range: MountainRange): number {
  const fraction = Math.max(0.000001, Math.min(0.999999, (worldX - range.left) / range.width))
  return (range.mirrored ? 1 - fraction : fraction) * 1200
}

export function mountainHeightScale(profileX: number, range: MountainRange): number {
  return heightAt(heightProfiles[range.form], profileX)
}

function rangeCrestAt(worldX: number, range: MountainRange): number {
  const x = mountainProfileX(worldX, range)
  return (
    range.base -
    ((350 - heightAt(ridgeProfile, x)) / 350) * range.height * mountainHeightScale(x, range)
  )
}

export function mountainCrestAt(
  worldX: number,
  ranges: readonly MountainRange[],
): number | undefined {
  const visible = mountainRangesAt(worldX, ranges)
  if (visible.length === 0) return undefined
  return Math.min(...visible.map(range => rangeCrestAt(worldX, range)))
}

// Depth follows the selected foreground slope, from its crest to the common valley base.
export function mountainGroundAt(
  worldX: number,
  depth: number,
  ranges: readonly MountainRange[],
): number | undefined {
  const foreground = mountainRangesAt(worldX, ranges)[0]
  if (!foreground) return undefined
  const crest = rangeCrestAt(worldX, foreground)
  return crest + (foreground.base - crest) * Math.max(0, Math.min(1, depth))
}
