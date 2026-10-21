import type { GearCassettePreset } from './triathlon-gear-ratio/presets'
import { GEAR_CASSETTE_PRESET_GROUPS } from './triathlon-gear-ratio/cassette-presets'

export { GEAR_CASSETTE_PRESET_GROUPS } from './triathlon-gear-ratio/cassette-presets'
export {
  DEFAULT_GEAR_CASSETTE,
  DEFAULT_GEAR_CHAINRING_PRESET,
  GEAR_CHAINRING_PRESETS,
  gearChainringPreset,
  type GearCassettePreset,
} from './triathlon-gear-ratio/presets'

interface GearRatioCell {
  cog: number
  ratio: number
  level: number
  drivetrainEfficiency: number
  drivetrainLossWatts: number
  crossChainLossWatts: number
}

interface GearRatioRow {
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

const gearEfficiencyDeltaPercent = (efficiency: number): number => {
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

export const gearCassettePreset = (id: string): GearCassettePreset | null => {
  for (const group of GEAR_CASSETTE_PRESET_GROUPS)
    for (const preset of group.presets) if (preset.id === id) return preset
  return null
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
