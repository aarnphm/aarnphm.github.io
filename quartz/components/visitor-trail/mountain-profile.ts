import type { Alpine, Point } from './landscape'
import { trailRandom } from './random'

export type Vertex = readonly [number, number]
export type Profile = readonly Vertex[]
type Region = { profile: Profile; left: number; top: number; right: number; bottom: number }
type MountainForm =
  | 'bernese'
  | 'horn'
  | 'glacial-bowl'
  | 'serrated'
  | 'limestone-wall'
  | 'pasture-ridge'
type MountainTier = 'background' | 'middle' | 'foreground'
type TrackKind = 'switchback' | 'pass' | 'shelf'
type TrackKeys = { kind: TrackKind; points: Profile }
type Terrain = {
  crest: Profile
  far: Profile
  near: Profile
  floor: Profile
  snow: readonly Region[]
  shade: readonly Region[]
  strata: readonly Region[]
  tracks: readonly TrackKeys[]
  aspect: number
}
type TerrainDefinition = {
  crest: Profile
  foothill: Profile
  snow: readonly Profile[]
  shade: readonly Profile[]
  strata: readonly Profile[]
  tracks: readonly TrackKeys[]
  aspect: number
}

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

function terrain(definition: TerrainDefinition): Terrain {
  return {
    crest: definition.crest,
    far: definition.foothill,
    near: definition.foothill.map(([x, y]): Vertex => [x, 320 + (y - 260) * 0.28]),
    floor: definition.foothill.map(([x, y]): Vertex => [x, 348 + (y - 260) * 0.02]),
    snow: definition.snow.map(region),
    shade: definition.shade.map(region),
    strata: definition.strata.map(region),
    tracks: definition.tracks,
    aspect: definition.aspect,
  }
}

const terrains: Record<MountainForm, Terrain> = {
  bernese: terrain({
    aspect: 3.45,
    crest: [
      [0, 302],
      [56, 276],
      [98, 238],
      [141, 212],
      [166, 181],
      [183, 153],
      [209, 149],
      [235, 117],
      [267, 89],
      [292, 76],
      [306, 73],
      [339, 105],
      [367, 138],
      [397, 165],
      [427, 168],
      [464, 138],
      [490, 118],
      [519, 93],
      [540, 67],
      [551, 58],
      [568, 60],
      [596, 88],
      [621, 116],
      [646, 143],
      [674, 167],
      [694, 164],
      [723, 143],
      [749, 106],
      [777, 79],
      [806, 66],
      [835, 45],
      [846, 29],
      [863, 45],
      [900, 59],
      [927, 91],
      [951, 119],
      [972, 137],
      [1003, 166],
      [1033, 196],
      [1058, 199],
      [1091, 230],
      [1120, 256],
      [1160, 278],
      [1200, 309],
    ],
    foothill: [
      [0, 295],
      [99, 276],
      [179, 264],
      [258, 278],
      [334, 254],
      [418, 268],
      [502, 282],
      [585, 260],
      [658, 272],
      [739, 264],
      [810, 280],
      [904, 260],
      [987, 278],
      [1101, 288],
      [1200, 300],
    ],
    snow: [
      [
        [167, 185],
        [211, 150],
        [266, 89],
        [306, 73],
        [322, 92],
        [295, 102],
        [270, 153],
        [245, 144],
        [217, 192],
        [185, 219],
      ],
      [
        [467, 141],
        [519, 93],
        [551, 58],
        [582, 75],
        [613, 115],
        [582, 96],
        [548, 106],
        [532, 157],
        [496, 170],
      ],
      [
        [722, 146],
        [777, 79],
        [835, 45],
        [846, 29],
        [869, 46],
        [897, 59],
        [929, 94],
        [963, 142],
        [926, 120],
        [890, 94],
        [857, 97],
        [834, 139],
        [798, 133],
        [774, 181],
      ],
    ],
    shade: [
      [
        [305, 94],
        [340, 106],
        [395, 165],
        [455, 257],
        [403, 294],
        [353, 239],
        [324, 183],
      ],
      [
        [551, 101],
        [585, 115],
        [675, 169],
        [709, 258],
        [654, 293],
        [603, 223],
      ],
      [
        [848, 87],
        [905, 105],
        [971, 143],
        [1071, 253],
        [1082, 292],
        [1004, 280],
        [913, 202],
        [881, 154],
      ],
    ],
    strata: [
      [
        [363, 211],
        [395, 202],
        [435, 243],
        [425, 251],
        [392, 221],
        [368, 224],
      ],
      [
        [611, 210],
        [648, 223],
        [670, 241],
        [659, 250],
        [628, 231],
        [609, 224],
      ],
      [
        [894, 177],
        [927, 187],
        [965, 211],
        [960, 219],
        [919, 200],
        [891, 190],
      ],
    ],
    tracks: [
      {
        kind: 'switchback',
        points: [
          [0.18, 0.95],
          [0.23, 0.86],
          [0.43, 0.78],
          [0.28, 0.68],
          [0.45, 0.59],
          [0.34, 0.5],
          [0.47, 0.4],
          [0.43, 0.29],
        ],
      },
      {
        kind: 'pass',
        points: [
          [0.78, 0.95],
          [0.68, 0.86],
          [0.88, 0.75],
          [0.66, 0.64],
          [0.83, 0.53],
          [0.67, 0.42],
          [0.73, 0.31],
        ],
      },
    ],
  }),
  horn: terrain({
    aspect: 3.05,
    crest: [
      [0, 312],
      [56, 294],
      [94, 258],
      [133, 250],
      [167, 223],
      [205, 211],
      [247, 186],
      [282, 182],
      [312, 149],
      [339, 123],
      [354, 92],
      [370, 88],
      [387, 63],
      [406, 50],
      [421, 18],
      [431, 42],
      [450, 70],
      [463, 88],
      [486, 104],
      [503, 137],
      [524, 147],
      [542, 188],
      [565, 203],
      [584, 231],
      [623, 250],
      [670, 256],
      [719, 239],
      [754, 242],
      [790, 214],
      [827, 216],
      [857, 194],
      [891, 206],
      [930, 244],
      [981, 261],
      [1034, 283],
      [1082, 280],
      [1135, 304],
      [1200, 315],
    ],
    foothill: [
      [0, 302],
      [111, 282],
      [193, 276],
      [282, 262],
      [355, 273],
      [446, 250],
      [534, 271],
      [621, 293],
      [731, 278],
      [813, 281],
      [900, 269],
      [986, 290],
      [1087, 295],
      [1200, 308],
    ],
    snow: [
      [
        [334, 129],
        [370, 88],
        [405, 50],
        [421, 18],
        [433, 47],
        [456, 78],
        [476, 101],
        [449, 88],
        [429, 67],
        [417, 80],
        [396, 112],
        [379, 123],
        [357, 157],
      ],
      [
        [495, 137],
        [525, 149],
        [542, 188],
        [565, 204],
        [579, 224],
        [551, 218],
        [533, 184],
        [514, 176],
      ],
      [
        [788, 216],
        [857, 194],
        [891, 206],
        [910, 225],
        [878, 219],
        [854, 229],
        [827, 227],
      ],
    ],
    shade: [
      [
        [421, 73],
        [461, 102],
        [501, 146],
        [548, 205],
        [626, 274],
        [584, 309],
        [506, 271],
        [478, 210],
        [449, 163],
      ],
      [
        [290, 190],
        [332, 161],
        [365, 142],
        [383, 200],
        [358, 264],
        [301, 294],
        [268, 258],
      ],
      [
        [858, 219],
        [897, 228],
        [960, 266],
        [1014, 300],
        [944, 303],
        [891, 274],
      ],
    ],
    strata: [
      [
        [453, 176],
        [479, 187],
        [508, 217],
        [500, 224],
        [472, 203],
        [450, 194],
      ],
      [
        [298, 246],
        [334, 223],
        [347, 234],
        [321, 255],
        [300, 258],
      ],
    ],
    tracks: [
      {
        kind: 'switchback',
        points: [
          [0.72, 0.94],
          [0.75, 0.86],
          [0.54, 0.79],
          [0.68, 0.7],
          [0.46, 0.61],
          [0.57, 0.51],
          [0.37, 0.41],
          [0.4, 0.27],
        ],
      },
      {
        kind: 'shelf',
        points: [
          [0.2, 0.94],
          [0.3, 0.85],
          [0.24, 0.75],
          [0.43, 0.65],
          [0.31, 0.54],
          [0.45, 0.43],
          [0.39, 0.3],
        ],
      },
    ],
  }),
  'glacial-bowl': terrain({
    aspect: 4.1,
    crest: [
      [0, 303],
      [58, 286],
      [102, 249],
      [131, 210],
      [161, 169],
      [192, 147],
      [209, 119],
      [235, 100],
      [256, 64],
      [277, 73],
      [300, 55],
      [321, 72],
      [343, 88],
      [362, 124],
      [384, 149],
      [415, 190],
      [452, 209],
      [509, 224],
      [567, 231],
      [635, 232],
      [684, 220],
      [732, 202],
      [771, 171],
      [807, 126],
      [841, 89],
      [867, 103],
      [885, 118],
      [912, 134],
      [936, 170],
      [959, 176],
      [983, 218],
      [1015, 235],
      [1049, 270],
      [1087, 268],
      [1134, 294],
      [1200, 310],
    ],
    foothill: [
      [0, 296],
      [105, 278],
      [192, 265],
      [282, 260],
      [371, 280],
      [452, 291],
      [537, 302],
      [615, 307],
      [708, 293],
      [795, 276],
      [887, 259],
      [981, 282],
      [1092, 294],
      [1200, 306],
    ],
    snow: [
      [
        [188, 151],
        [235, 100],
        [256, 64],
        [277, 73],
        [300, 55],
        [321, 73],
        [357, 117],
        [346, 138],
        [318, 132],
        [294, 164],
        [277, 151],
        [248, 189],
        [219, 180],
      ],
      [
        [360, 145],
        [415, 190],
        [451, 209],
        [508, 224],
        [567, 231],
        [635, 232],
        [684, 220],
        [732, 202],
        [771, 171],
        [807, 126],
        [841, 89],
        [867, 103],
        [907, 133],
        [894, 160],
        [852, 144],
        [822, 194],
        [781, 210],
        [747, 252],
        [700, 264],
        [639, 273],
        [572, 265],
        [515, 257],
        [461, 241],
        [414, 224],
        [386, 193],
      ],
    ],
    shade: [
      [
        [296, 132],
        [337, 139],
        [390, 198],
        [448, 244],
        [405, 290],
        [352, 273],
        [318, 221],
      ],
      [
        [842, 145],
        [880, 170],
        [929, 198],
        [1009, 270],
        [997, 307],
        [933, 294],
        [878, 246],
      ],
    ],
    strata: [
      [
        [351, 231],
        [382, 244],
        [407, 271],
        [397, 280],
        [365, 255],
        [348, 245],
      ],
      [
        [877, 231],
        [916, 249],
        [940, 273],
        [930, 279],
        [899, 259],
        [876, 244],
      ],
    ],
    tracks: [
      {
        kind: 'pass',
        points: [
          [0.2, 0.94],
          [0.38, 0.83],
          [0.62, 0.75],
          [0.4, 0.65],
          [0.59, 0.56],
          [0.41, 0.45],
          [0.32, 0.32],
        ],
      },
      {
        kind: 'switchback',
        points: [
          [0.82, 0.94],
          [0.68, 0.83],
          [0.85, 0.73],
          [0.65, 0.63],
          [0.82, 0.5],
          [0.7, 0.38],
          [0.73, 0.28],
        ],
      },
    ],
  }),
  serrated: terrain({
    aspect: 3.8,
    crest: [
      [0, 309],
      [45, 276],
      [77, 254],
      [105, 216],
      [130, 222],
      [156, 181],
      [184, 197],
      [211, 151],
      [236, 169],
      [261, 132],
      [291, 162],
      [318, 114],
      [337, 135],
      [359, 86],
      [379, 109],
      [396, 73],
      [419, 101],
      [437, 91],
      [462, 127],
      [487, 108],
      [512, 155],
      [540, 132],
      [563, 168],
      [586, 156],
      [609, 199],
      [637, 178],
      [664, 212],
      [693, 167],
      [721, 185],
      [746, 141],
      [771, 158],
      [795, 119],
      [821, 139],
      [845, 99],
      [865, 113],
      [888, 88],
      [909, 111],
      [934, 98],
      [959, 144],
      [984, 157],
      [1004, 198],
      [1036, 205],
      [1066, 244],
      [1092, 239],
      [1132, 282],
      [1200, 316],
    ],
    foothill: [
      [0, 301],
      [88, 282],
      [159, 274],
      [235, 258],
      [304, 277],
      [381, 263],
      [465, 285],
      [549, 274],
      [631, 292],
      [713, 280],
      [793, 266],
      [877, 276],
      [964, 260],
      [1052, 282],
      [1125, 294],
      [1200, 307],
    ],
    snow: [
      [
        [305, 136],
        [358, 86],
        [379, 109],
        [396, 73],
        [419, 101],
        [437, 91],
        [462, 127],
        [483, 125],
        [458, 150],
        [421, 132],
        [396, 154],
        [374, 133],
        [344, 158],
      ],
      [
        [773, 159],
        [795, 119],
        [821, 139],
        [845, 99],
        [865, 113],
        [888, 88],
        [909, 111],
        [934, 98],
        [955, 139],
        [927, 130],
        [899, 143],
        [881, 130],
        [853, 154],
        [823, 154],
      ],
    ],
    shade: [
      [
        [177, 198],
        [225, 179],
        [272, 189],
        [321, 249],
        [293, 294],
        [223, 282],
      ],
      [
        [395, 155],
        [442, 146],
        [490, 173],
        [559, 251],
        [527, 296],
        [465, 278],
      ],
      [
        [888, 144],
        [933, 144],
        [987, 194],
        [1059, 264],
        [1038, 304],
        [961, 278],
      ],
    ],
    strata: [
      [
        [226, 245],
        [260, 236],
        [283, 257],
        [275, 266],
        [251, 249],
        [224, 255],
      ],
      [
        [468, 238],
        [492, 229],
        [527, 257],
        [520, 265],
        [488, 245],
        [466, 250],
      ],
      [
        [960, 245],
        [987, 253],
        [1011, 275],
        [1001, 282],
        [979, 267],
        [956, 257],
      ],
    ],
    tracks: [
      {
        kind: 'switchback',
        points: [
          [0.23, 0.94],
          [0.32, 0.84],
          [0.18, 0.74],
          [0.41, 0.65],
          [0.29, 0.54],
          [0.44, 0.42],
          [0.37, 0.29],
        ],
      },
      {
        kind: 'pass',
        points: [
          [0.73, 0.94],
          [0.63, 0.84],
          [0.84, 0.75],
          [0.67, 0.65],
          [0.86, 0.53],
          [0.76, 0.42],
          [0.78, 0.29],
        ],
      },
    ],
  }),
  'limestone-wall': terrain({
    aspect: 4.2,
    crest: [
      [0, 310],
      [65, 295],
      [112, 261],
      [156, 250],
      [194, 213],
      [221, 176],
      [243, 142],
      [266, 121],
      [300, 126],
      [328, 106],
      [369, 111],
      [402, 103],
      [438, 109],
      [470, 91],
      [499, 104],
      [537, 101],
      [563, 82],
      [591, 92],
      [620, 69],
      [644, 78],
      [666, 106],
      [688, 118],
      [704, 151],
      [720, 183],
      [731, 232],
      [751, 246],
      [780, 263],
      [815, 276],
      [845, 273],
      [873, 244],
      [903, 238],
      [931, 250],
      [959, 228],
      [993, 245],
      [1022, 264],
      [1051, 272],
      [1098, 282],
      [1147, 305],
      [1200, 315],
    ],
    foothill: [
      [0, 301],
      [112, 283],
      [206, 276],
      [291, 269],
      [389, 273],
      [489, 267],
      [579, 261],
      [673, 272],
      [765, 304],
      [849, 310],
      [939, 296],
      [1024, 281],
      [1111, 298],
      [1200, 308],
    ],
    snow: [
      [
        [242, 142],
        [266, 121],
        [300, 126],
        [328, 106],
        [369, 111],
        [401, 103],
        [437, 109],
        [470, 91],
        [499, 104],
        [537, 101],
        [563, 82],
        [591, 92],
        [620, 69],
        [644, 78],
        [666, 106],
        [649, 115],
        [620, 101],
        [590, 116],
        [551, 118],
        [521, 130],
        [483, 119],
        [448, 130],
        [407, 123],
        [368, 132],
        [327, 129],
        [295, 147],
      ],
    ],
    shade: [
      [
        [577, 134],
        [623, 121],
        [663, 144],
        [704, 178],
        [731, 232],
        [761, 266],
        [724, 306],
        [665, 296],
        [625, 246],
      ],
      [
        [294, 161],
        [358, 150],
        [387, 174],
        [365, 266],
        [322, 293],
        [278, 276],
      ],
    ],
    strata: [
      [
        [380, 168],
        [424, 173],
        [457, 164],
        [492, 174],
        [529, 166],
        [562, 177],
        [563, 185],
        [530, 176],
        [491, 183],
        [456, 174],
        [425, 182],
        [378, 177],
      ],
      [
        [390, 214],
        [426, 220],
        [466, 211],
        [503, 222],
        [543, 213],
        [581, 224],
        [583, 233],
        [543, 223],
        [505, 231],
        [465, 222],
        [426, 230],
        [390, 223],
      ],
      [
        [407, 260],
        [446, 253],
        [482, 267],
        [522, 258],
        [561, 270],
        [602, 262],
        [611, 272],
        [561, 279],
        [520, 268],
        [481, 276],
        [445, 263],
        [406, 270],
      ],
    ],
    tracks: [
      {
        kind: 'shelf',
        points: [
          [0.14, 0.94],
          [0.27, 0.86],
          [0.53, 0.79],
          [0.61, 0.67],
          [0.47, 0.56],
          [0.58, 0.43],
          [0.44, 0.32],
        ],
      },
      {
        kind: 'pass',
        points: [
          [0.83, 0.94],
          [0.73, 0.85],
          [0.84, 0.74],
          [0.69, 0.64],
          [0.73, 0.5],
          [0.59, 0.38],
          [0.54, 0.29],
        ],
      },
    ],
  }),
  'pasture-ridge': terrain({
    aspect: 4.9,
    crest: [
      [0, 307],
      [69, 285],
      [121, 279],
      [181, 249],
      [229, 242],
      [271, 216],
      [314, 210],
      [348, 188],
      [381, 184],
      [414, 167],
      [449, 175],
      [487, 166],
      [522, 173],
      [559, 194],
      [597, 207],
      [641, 223],
      [683, 241],
      [725, 247],
      [762, 236],
      [795, 220],
      [829, 208],
      [865, 189],
      [902, 198],
      [934, 217],
      [969, 224],
      [1012, 253],
      [1061, 267],
      [1107, 287],
      [1160, 294],
      [1200, 310],
    ],
    foothill: [
      [0, 300],
      [107, 284],
      [192, 269],
      [277, 250],
      [363, 264],
      [452, 248],
      [542, 269],
      [628, 287],
      [717, 293],
      [807, 270],
      [897, 258],
      [987, 281],
      [1101, 294],
      [1200, 306],
    ],
    snow: [],
    shade: [
      [
        [492, 183],
        [551, 203],
        [604, 239],
        [650, 278],
        [610, 300],
        [557, 268],
        [524, 225],
      ],
      [
        [866, 210],
        [914, 231],
        [968, 268],
        [1002, 290],
        [952, 299],
        [912, 274],
      ],
    ],
    strata: [
      [
        [325, 259],
        [359, 247],
        [387, 256],
        [406, 253],
        [419, 263],
        [386, 266],
        [359, 257],
        [324, 269],
      ],
    ],
    tracks: [
      {
        kind: 'shelf',
        points: [
          [0.16, 0.94],
          [0.3, 0.82],
          [0.53, 0.71],
          [0.67, 0.59],
          [0.49, 0.48],
          [0.42, 0.33],
        ],
      },
      {
        kind: 'pass',
        points: [
          [0.84, 0.94],
          [0.7, 0.81],
          [0.78, 0.67],
          [0.62, 0.54],
          [0.74, 0.4],
          [0.74, 0.29],
        ],
      },
    ],
  }),
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
  tier: MountainTier
}

export type MountainTrack = {
  id: number
  points: Point[]
  rangeIndex: number
  kind: TrackKind
  depth: number
}

const forms: readonly MountainForm[] = [
  'bernese',
  'horn',
  'glacial-bowl',
  'serrated',
  'limestone-wall',
  'pasture-ridge',
]
const tiers: readonly MountainTier[] = ['background', 'middle', 'foreground']

export function mountainTerrain(range: MountainRange): Terrain {
  return terrains[range.form]
}

export function buildMountainRanges(alpine: Alpine, seed: number): MountainRange[] {
  const sharedRandom = trailRandom(seed ^ 0x13a5b79d)
  const narrow = alpine.width < 700
  const paintWidth = alpine.width + 16
  const result: MountainRange[] = []

  for (const [tierIndex, tier] of tiers.entries()) {
    // Adding a background range must not reshape the existing tiers in front.
    const random = narrow
      ? sharedRandom
      : trailRandom(seed ^ 0x13a5b79d ^ Math.imul(tierIndex + 1, 0x9e3779b9))
    const span = tier === 'middle' ? 1.12 : tier === 'background' ? 0.97 : 0.83
    const shuffled = [...forms]
    for (let index = shuffled.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1))
      ;[shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]]
    }
    const bag = narrow
      ? shuffled.filter(
          form =>
            !result.some(range => range.form === form) &&
            (tier !== 'background' || (form !== 'pasture-ridge' && form !== 'limestone-wall')),
        )
      : shuffled
    const overlap = 0.21 + random() * 0.07
    let left = alpine.left - 8
    let index = 0

    while (index === 0 || (!narrow && left < alpine.left + alpine.width + 8)) {
      const weight = 0.78 + random() * 0.46
      const baseOffset =
        tier === 'foreground'
          ? 0
          : alpine.height * (tier === 'middle' ? 0.035 + random() * 0.045 : 0.1 + random() * 0.065)
      const elevation = tier === 'foreground' ? 0.82 + random() * 0.12 : 0.96 + random() * 0.1
      const mirrored = random() > 0.5
      const phase = random() * Math.PI * 2
      const form = bag[index % bag.length]
      const base = alpine.base - baseOffset
      const tierCap = narrow ? (tier === 'foreground' ? 0.56 : tier === 'middle' ? 0.73 : 1) : 1
      const availableHeight = (base - alpine.top) * tierCap
      const height = narrow
        ? Math.min(availableHeight, (paintWidth / terrains[form].aspect) * elevation)
        : availableHeight * Math.min(1, span * 0.75 * weight)
      const width = narrow ? paintWidth : (height * terrains[form].aspect) / elevation
      result.push({
        left,
        top: base - height,
        width,
        height,
        base,
        mirrored,
        depth:
          tier === 'foreground'
            ? 0.9 + elevation * 0.08
            : tier === 'middle'
              ? 0.6 + elevation * 0.05
              : 0.3 + elevation * 0.05,
        phase,
        index: result.length,
        form,
        tier,
      })
      left += width * (1 - overlap)
      index++
    }
  }
  return result
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

function rangeCrestAt(worldX: number, range: MountainRange): number {
  const x = mountainProfileX(worldX, range)
  return range.base - ((350 - heightAt(terrains[range.form].crest, x)) / 350) * range.height
}

export function mountainCrestAt(
  worldX: number,
  ranges: readonly MountainRange[],
): number | undefined {
  const visible = mountainRangesAt(worldX, ranges)
  if (visible.length === 0) return undefined
  return Math.min(...visible.map(range => rangeCrestAt(worldX, range)))
}

// Foreground ranges retain the common valley base for near-to-far ground queries.
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

export function buildMountainTracks(
  alpine: Alpine,
  seed: number,
  entrances?: readonly Point[],
): MountainTrack[] {
  const ranges = buildMountainRanges(alpine, seed)
  const onMountain = (x: number, depth: number): Point => {
    const ground = mountainGroundAt(x, depth, ranges) ?? alpine.base - 12
    return { x, y: Math.max(ground, (mountainCrestAt(x, ranges) ?? ground - 12) + 12) }
  }
  const supplied = entrances?.slice(0, 6)
  const starts = supplied?.length
    ? supplied
    : Array.from({ length: alpine.width < 700 ? 2 : 3 }, (_, index) => ({
        x: alpine.left + (alpine.width * (index + 1)) / (alpine.width < 700 ? 3 : 4),
        y: alpine.base - 12,
      }))

  return starts.map((entrance, id) => {
    const range = mountainRangesAt(entrance.x, ranges)[0]
    const terrain = terrains[range.form]
    const authored = terrain.tracks[id % terrain.tracks.length]
    const keys = authored.points.map(([fraction, depth]): Point => {
      const x = Math.max(
        alpine.left + 4,
        Math.min(
          alpine.left + alpine.width - 4,
          range.left + range.width * (range.mirrored ? 1 - fraction : fraction),
        ),
      )
      return onMountain(x, depth)
    })
    const lastPoint = keys.at(-1)
    const lastKey = authored.points.at(-1)
    const summitLimit = entrance.y - Math.max(4, alpine.height * 0.06)
    if (lastPoint && lastKey && lastPoint.y > summitLimit) {
      // An overlapping ridge can cover an authored summit; finish on visible uphill terrain.
      let closest = Infinity
      for (let step = 0; step <= 64; step++) {
        const x = Math.max(
          alpine.left + 4,
          Math.min(alpine.left + alpine.width - 4, range.left + (range.width * step) / 64),
        )
        if (mountainRangesAt(x, ranges)[0]?.index !== range.index) continue
        const candidate = onMountain(x, lastKey[1])
        const distance = Math.abs(candidate.x - lastPoint.x)
        if (candidate.y <= summitLimit && distance < closest) {
          closest = distance
          keys[keys.length - 1] = candidate
        }
      }
    }
    const points: Point[] = [{ ...entrance }]
    const anchors = [{ ...entrance }, ...keys]
    for (let segment = 1; segment < anchors.length; segment++) {
      const from = anchors[segment - 1]
      const to = anchors[segment]
      const steps = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / 3))
      for (let step = 1; step <= steps; step++) {
        const blend = step / steps
        const x = from.x + (to.x - from.x) * blend
        const linear = from.y + (to.y - from.y) * blend
        const crest = mountainCrestAt(x, ranges) ?? linear - 12
        points.push({ x, y: Math.min(alpine.base - 4, Math.max(linear, crest + 12)) })
      }
    }
    return { id, points, rangeIndex: range.index, kind: authored.kind, depth: range.depth }
  })
}
