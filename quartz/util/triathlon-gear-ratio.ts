export interface GearCassettePreset {
  id: string
  label: string
  speeds: number
  cogs: readonly number[]
  maximumChainrings: 1 | 2
}

export interface GearCassettePresetGroup {
  label: string
  presets: readonly GearCassettePreset[]
}

export interface GearChainringPreset {
  id: string
  label: string
  chainrings: readonly [number, number]
}

export interface GearRatioCell {
  cog: number
  ratio: number
  level: number
  drivetrainEfficiency: number
  drivetrainLossWatts: number
  crossChainLossWatts: number
}

export interface GearRatioRow {
  chainring: number
  cells: readonly GearRatioCell[]
}

export interface GearRatioMatrix {
  minimum: number
  maximum: number
  rows: readonly GearRatioRow[]
}

export const CERAMICSPEED_CROSS_CHAIN_RESEARCH = {
  sources: [
    {
      id: 1,
      title: 'CeramicSpeed Cross Chaining and Ring-Size Report summary',
      url: 'https://ceramicspeed.com/pages/cross-chaining-and-ring-size-report',
    },
    {
      id: 2,
      title:
        'Effects of Lateral Chain Misalignment (Cross-Chaining) on Drivetrain Efficiency & Effects of Chainring Size on Drivetrain Efficiency',
      publishedOn: '2015-04-17',
      url: 'https://cdn.shopify.com/s/files/1/0726/7542/6606/files/cross-chaining-and-ring-size-report.pdf?v=1687253624',
    },
  ],
  test: {
    sourceId: 2,
    protocolPage: 4,
    chainstayPage: 13,
    outputWatts: 250,
    cadenceRpm: 95,
    chainstayMm: 385,
    chainrings: [53, 39],
    cassetteCogs: [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 28],
    alignedCogNumbers: { largeChainring: 4, smallChainring: 6 },
  },
  alignedLoss: {
    sourceId: 2,
    page: 20,
    method:
      'ordinary least squares fit of average aligned loss watts against exact chainring/cog gear ratios',
    curves: [
      {
        chainring: 39,
        averageLossWatts: [8.24, 8.01, 7.58, 7.45, 7.27, 6.89, 6.71, 6.61, 6.49, 6.28, 6.14],
        fit: { slope: 0.9649512849225721, intercept: 4.777170268580869 },
      },
      {
        chainring: 53,
        averageLossWatts: [8.03, 7.54, 7.3, 6.97, 6.78, 6.49, 6.31, 6.08, 5.94, 5.79, 5.65],
        fit: { slope: 0.7821444353666108, intercept: 4.109867970360179 },
      },
    ],
  },
  crossChainLoss: {
    sourceId: 2,
    page: 15,
    wattsByCogOffset: [0, 0.19, 0.49, 0.79, 1.1, 1.4, 1.7, 2.01],
  },
  extrapolation: {
    cassetteAlignment: 'normalize the reported 4th and 6th cog positions across cassette length',
    oneByAlignment: 'cassette center',
    beyondReportedCogOffset: 'continue the final reported watt increment',
  },
} as const

export const CERAMICSPEED_TEST_OUTPUT_WATTS = CERAMICSPEED_CROSS_CHAIN_RESEARCH.test.outputWatts
export const CERAMICSPEED_TEST_CADENCE_RPM = CERAMICSPEED_CROSS_CHAIN_RESEARCH.test.cadenceRpm
export const CERAMICSPEED_TEST_CHAINSTAY_MM = CERAMICSPEED_CROSS_CHAIN_RESEARCH.test.chainstayMm

export const gearEfficiencyDeltaPercent = (efficiency: number): number => {
  const delta = Number((efficiency - 100).toFixed(6))
  return Object.is(delta, -0) ? 0 : delta
}

export const formatGearEfficiencyDeltaPercent = (
  efficiency: number,
  fractionDigits: number,
): string => {
  const delta = gearEfficiencyDeltaPercent(efficiency)
  const magnitude = Math.abs(delta).toFixed(fractionDigits)
  return delta > 0 ? `+${magnitude}` : delta < 0 ? `-${magnitude}` : magnitude
}

const ceramicSpeedAlignedLossWatts = (chainring: number, ratio: number): number => {
  const [small, large] = CERAMICSPEED_CROSS_CHAIN_RESEARCH.alignedLoss.curves
  const position = (chainring - small.chainring) / (large.chainring - small.chainring)
  const slope = small.fit.slope + (large.fit.slope - small.fit.slope) * position
  const intercept = small.fit.intercept + (large.fit.intercept - small.fit.intercept) * position
  return intercept + slope * ratio
}

const ceramicSpeedCrossChainLossWatts = (steps: number): number => {
  const losses = CERAMICSPEED_CROSS_CHAIN_RESEARCH.crossChainLoss.wattsByCogOffset
  const lastIndex = losses.length - 1
  if (steps >= lastIndex) {
    const last = losses[lastIndex]
    const previous = losses[lastIndex - 1]
    return last + (steps - lastIndex) * (last - previous)
  }
  const lowerIndex = Math.floor(steps)
  const lower = losses[lowerIndex]
  const upper = losses[lowerIndex + 1]
  return lower + (upper - lower) * (steps - lowerIndex)
}

const ceramicSpeedAlignedCogIndex = (
  cogCount: number,
  chainringIndex: number,
  largeChainringIndex: number,
  chainringCount: number,
): number => {
  if (chainringCount === 1) return (cogCount - 1) / 2
  const { alignedCogNumbers, cassetteCogs } = CERAMICSPEED_CROSS_CHAIN_RESEARCH.test
  const reportCogNumber =
    chainringIndex === largeChainringIndex
      ? alignedCogNumbers.largeChainring
      : alignedCogNumbers.smallChainring
  return ((reportCogNumber - 1) / (cassetteCogs.length - 1)) * (cogCount - 1)
}

const cassette = (
  id: string,
  label: string,
  speeds: number,
  cogs: readonly number[],
  maximumChainrings: 1 | 2 = 2,
): GearCassettePreset => ({ id, label, speeds, cogs, maximumChainrings })

export const DEFAULT_GEAR_CASSETTE: GearCassettePreset = cassette(
  'shimano-ultegra-r8100-11-34',
  'Ultegra R8100 · 11–34 · 12s',
  12,
  [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34],
)

const chainrings = (
  id: string,
  label: string,
  first: number,
  second: number,
): GearChainringPreset => ({ id, label, chainrings: [first, second] })

export const DEFAULT_GEAR_CHAINRING_PRESET: GearChainringPreset = chainrings(
  '54-40',
  '54–40',
  54,
  40,
)

export const GEAR_CHAINRING_PRESETS: readonly GearChainringPreset[] = [
  DEFAULT_GEAR_CHAINRING_PRESET,
  chainrings('53-39', '53–39', 53, 39),
  chainrings('52-36', '52–36', 52, 36),
]

export const GEAR_CASSETTE_PRESET_GROUPS: readonly GearCassettePresetGroup[] = [
  {
    label: 'Shimano 12-speed',
    presets: [
      DEFAULT_GEAR_CASSETTE,
      cassette(
        'shimano-ultegra-r8100-11-30',
        'Ultegra R8100 · 11–30 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'shimano-dura-ace-r9200-11-30',
        'Dura-Ace R9200 · 11–30 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'shimano-dura-ace-r9200-11-34',
        'Dura-Ace R9200 · 11–34 · 12s',
        12,
        [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34],
      ),
    ],
  },
  {
    label: 'Shimano 11-speed',
    presets: [
      cassette(
        'shimano-ultegra-r8000-11-25',
        'Ultegra R8000 · 11–25 · 11s',
        11,
        [11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25],
      ),
      cassette(
        'shimano-ultegra-r8000-11-28',
        'Ultegra R8000 · 11–28 · 11s',
        11,
        [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 28],
      ),
      cassette(
        'shimano-ultegra-r8000-11-30',
        'Ultegra R8000 · 11–30 · 11s',
        11,
        [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'shimano-ultegra-r8000-11-32',
        'Ultegra R8000 · 11–32 · 11s',
        11,
        [11, 12, 13, 14, 16, 18, 20, 22, 25, 28, 32],
      ),
      cassette(
        'shimano-ultegra-r8000-12-25',
        'Ultegra R8000 · 12–25 · 11s',
        11,
        [12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25],
      ),
      cassette(
        'shimano-ultegra-r8000-14-28',
        'Ultegra R8000 · 14–28 · 11s',
        11,
        [14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 28],
      ),
      cassette(
        'shimano-dura-ace-r9100-11-25',
        'Dura-Ace R9100 · 11–25 · 11s',
        11,
        [11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25],
      ),
      cassette(
        'shimano-dura-ace-r9100-11-28',
        'Dura-Ace R9100 · 11–28 · 11s',
        11,
        [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 28],
      ),
      cassette(
        'shimano-dura-ace-r9100-11-30',
        'Dura-Ace R9100 · 11–30 · 11s',
        11,
        [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'shimano-dura-ace-r9100-12-25',
        'Dura-Ace R9100 · 12–25 · 11s',
        11,
        [12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25],
      ),
      cassette(
        'shimano-dura-ace-r9100-12-28',
        'Dura-Ace R9100 · 12–28 · 11s',
        11,
        [12, 13, 14, 15, 16, 17, 19, 21, 23, 25, 28],
      ),
      cassette(
        'shimano-hg800-11-34',
        'Shimano HG800 · 11–34 · 11s',
        11,
        [11, 13, 15, 17, 19, 21, 23, 25, 27, 30, 34],
      ),
    ],
  },
  {
    label: 'SRAM AXS 12-speed',
    presets: [
      cassette(
        'sram-red-xg-1290-10-28',
        'RED XG-1290 · 10–28 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 28],
      ),
      cassette(
        'sram-red-xg-1290-10-30',
        'RED XG-1290 · 10–30 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'sram-red-xg-1290-10-33',
        'RED XG-1290 · 10–33 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 28, 33],
      ),
      cassette(
        'sram-red-xg-1290-10-36',
        'RED XG-1290 · 10–36 · 12s',
        12,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 36],
      ),
      cassette(
        'sram-force-xg-1270-10-28',
        'Force XG-1270 · 10–28 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 28],
      ),
      cassette(
        'sram-force-xg-1270-10-30',
        'Force XG-1270 · 10–30 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'sram-force-xg-1270-10-33',
        'Force XG-1270 · 10–33 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 28, 33],
      ),
      cassette(
        'sram-force-xg-1270-10-36',
        'Force XG-1270 · 10–36 · 12s',
        12,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 36],
      ),
      cassette(
        'sram-rival-xg-1250-10-30',
        'Rival XG-1250 · 10–30 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
      ),
      cassette(
        'sram-rival-xg-1250-10-36',
        'Rival XG-1250 · 10–36 · 12s',
        12,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 36],
      ),
    ],
  },
  {
    label: 'SRAM XPLR',
    presets: [
      cassette(
        'sram-red-xplr-xg-1391-10-46',
        'RED XPLR XG-1391 · 10–46 · 13s',
        13,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 38, 46],
        1,
      ),
      cassette(
        'sram-force-xplr-xg-1371-10-46',
        'Force XPLR XG-1371 · 10–46 · 13s',
        13,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 38, 46],
        1,
      ),
      cassette(
        'sram-rival-xplr-xg-1351-10-46',
        'Rival XPLR XG-1351 · 10–46 · 13s',
        13,
        [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 38, 46],
        1,
      ),
      cassette(
        'sram-xplr-xg-1271-10-44',
        'XPLR XG-1271 · 10–44 · 12s',
        12,
        [10, 11, 13, 15, 17, 19, 21, 24, 28, 32, 38, 44],
        1,
      ),
      cassette(
        'sram-xplr-xg-1251-10-44',
        'XPLR XG-1251 · 10–44 · 12s',
        12,
        [10, 11, 13, 15, 17, 19, 21, 24, 28, 32, 38, 44],
        1,
      ),
    ],
  },
  {
    label: 'Campagnolo 13-speed',
    presets: [
      cassette(
        'campagnolo-super-record-13-10-29',
        'Super Record 13 · 10–29 · 13s',
        13,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 26, 29],
      ),
      cassette(
        'campagnolo-super-record-13-10-33',
        'Super Record 13 · 10–33 · 13s',
        13,
        [10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 26, 29, 33],
      ),
      cassette(
        'campagnolo-super-record-13-11-32',
        'Super Record 13 · 11–32 · 13s',
        13,
        [11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 26, 29, 32],
      ),
      cassette(
        'campagnolo-super-record-13-11-36',
        'Super Record 13 · 11–36 · 13s',
        13,
        [11, 12, 13, 14, 15, 16, 18, 20, 23, 26, 29, 32, 36],
      ),
    ],
  },
  {
    label: 'Campagnolo 12-speed',
    presets: [
      cassette(
        'campagnolo-super-record-wireless-10-25',
        'Super Record Wireless · 10–25 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25],
      ),
      cassette(
        'campagnolo-super-record-wireless-10-27',
        'Super Record Wireless · 10–27 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27],
      ),
      cassette(
        'campagnolo-super-record-wireless-10-29',
        'Super Record Wireless · 10–29 · 12s',
        12,
        [10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 26, 29],
      ),
      cassette(
        'campagnolo-super-record-wireless-11-32',
        'Super Record Wireless · 11–32 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 22, 25, 28, 32],
      ),
      cassette(
        'campagnolo-chorus-12-11-29',
        'Chorus · 11–29 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 26, 29],
      ),
      cassette(
        'campagnolo-chorus-12-11-32',
        'Chorus · 11–32 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 22, 25, 28, 32],
      ),
      cassette(
        'campagnolo-chorus-12-11-34',
        'Chorus · 11–34 · 12s',
        12,
        [11, 12, 13, 14, 15, 16, 17, 19, 22, 25, 29, 34],
      ),
    ],
  },
  {
    label: 'Campagnolo 1×13',
    presets: [
      cassette(
        'campagnolo-super-record-x-9-42',
        'Super Record X · 9–42 · 13s',
        13,
        [9, 10, 11, 12, 13, 14, 16, 18, 21, 25, 30, 36, 42],
        1,
      ),
      cassette(
        'campagnolo-super-record-x-10-48',
        'Super Record X · 10–48 · 13s',
        13,
        [10, 11, 12, 13, 14, 16, 18, 21, 25, 30, 36, 42, 48],
        1,
      ),
      cassette(
        'campagnolo-record-x-10-48',
        'Record X · 10–48 · 13s',
        13,
        [10, 11, 12, 13, 14, 16, 18, 21, 25, 30, 36, 42, 48],
        1,
      ),
      cassette(
        'campagnolo-ekar-9-36',
        'Ekar · 9–36 · 13s',
        13,
        [9, 10, 11, 12, 13, 14, 16, 18, 20, 23, 27, 31, 36],
        1,
      ),
      cassette(
        'campagnolo-ekar-9-42',
        'Ekar · 9–42 · 13s',
        13,
        [9, 10, 11, 12, 13, 14, 16, 18, 21, 25, 30, 36, 42],
        1,
      ),
      cassette(
        'campagnolo-ekar-10-44',
        'Ekar · 10–44 · 13s',
        13,
        [10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 32, 38, 44],
        1,
      ),
    ],
  },
]

export const gearCassettePreset = (id: string): GearCassettePreset | null => {
  for (const group of GEAR_CASSETTE_PRESET_GROUPS)
    for (const preset of group.presets) if (preset.id === id) return preset
  return null
}

export const gearChainringPreset = (id: string): GearChainringPreset | null => {
  return GEAR_CHAINRING_PRESETS.find(preset => preset.id === id) ?? null
}

const validTeeth = (teeth: readonly number[]): boolean =>
  teeth.length > 0 && teeth.every(value => Number.isInteger(value) && value > 0)

export const gearRatioMatrix = (
  chainrings: readonly number[],
  cogs: readonly number[],
): GearRatioMatrix | null => {
  if (chainrings.length > 2 || !validTeeth(chainrings) || !validTeeth(cogs)) return null

  const ratios = chainrings.map(chainring => cogs.map(cog => chainring / cog))
  const values = ratios.flat()
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const span = maximum - minimum
  const largeChainringIndex = chainrings.reduce(
    (largest, chainring, index) => (chainring > chainrings[largest] ? index : largest),
    0,
  )

  return {
    minimum,
    maximum,
    rows: ratios.map((row, rowIndex) => ({
      chainring: chainrings[rowIndex],
      cells: row.map((ratio, cellIndex) => {
        const alignedCogIndex = ceramicSpeedAlignedCogIndex(
          cogs.length,
          rowIndex,
          largeChainringIndex,
          chainrings.length,
        )
        const crossChainLossWatts = ceramicSpeedCrossChainLossWatts(
          Math.abs(cellIndex - alignedCogIndex),
        )
        const drivetrainLossWatts =
          ceramicSpeedAlignedLossWatts(chainrings[rowIndex], ratio) + crossChainLossWatts
        return {
          cog: cogs[cellIndex],
          ratio,
          level: span === 0 ? 0.5 : (ratio - minimum) / span,
          drivetrainEfficiency:
            ((CERAMICSPEED_TEST_OUTPUT_WATTS - drivetrainLossWatts) /
              CERAMICSPEED_TEST_OUTPUT_WATTS) *
            100,
          drivetrainLossWatts,
          crossChainLossWatts,
        }
      }),
    })),
  }
}
