import assert from 'node:assert/strict'
import test from 'node:test'
import {
  CERAMICSPEED_CROSS_CHAIN_RESEARCH,
  CERAMICSPEED_TEST_OUTPUT_WATTS,
  DEFAULT_GEAR_CASSETTE,
  GEAR_CASSETTE_PRESET_GROUPS,
  GEAR_CHAINRING_PRESETS,
  formatGearEfficiencyDeltaPercent,
  gearChainringPreset,
  gearEfficiencyDeltaPercent,
  gearCassettePreset,
  gearRatioMatrix,
} from './triathlon-gear-ratio'

test('the declared bike drivetrain is the default preset', () => {
  assert.equal(DEFAULT_GEAR_CASSETTE.id, 'shimano-ultegra-r8100-11-34')
  assert.deepEqual(DEFAULT_GEAR_CASSETTE.cogs, [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34])
})

test('chainring presets retain the three distinct tooth pairs', () => {
  assert.deepEqual(
    GEAR_CHAINRING_PRESETS.map(preset => [preset.id, preset.label, preset.chainrings]),
    [
      ['54-40', '54–40', [54, 40]],
      ['53-39', '53–39', [53, 39]],
      ['52-36', '52–36', [52, 36]],
    ],
  )
  assert.equal(gearChainringPreset('54-40'), GEAR_CHAINRING_PRESETS[0])
  assert.equal(gearChainringPreset('missing'), null)
})

test('cassette presets have unique ids and ordered sprocket counts', () => {
  const presets = GEAR_CASSETTE_PRESET_GROUPS.flatMap(group => group.presets)
  const ids = presets.map(preset => preset.id)

  assert.equal(new Set(ids).size, ids.length)
  for (const preset of presets) {
    assert.equal(preset.cogs.length, preset.speeds)
    assert.deepEqual(
      [...preset.cogs].sort((left, right) => left - right),
      preset.cogs,
    )
    assert.equal(gearCassettePreset(preset.id), preset)
    assert.equal(preset.maximumChainrings === 1 || preset.maximumChainrings === 2, true)
  }
  assert.equal(gearCassettePreset('missing'), null)
})

test('Campagnolo road and gravel presets use their declared sprocket progressions', () => {
  assert.deepEqual(
    gearCassettePreset('campagnolo-super-record-13-10-29')?.cogs,
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 26, 29],
  )
  assert.deepEqual(
    gearCassettePreset('campagnolo-super-record-wireless-10-29')?.cogs,
    [10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 26, 29],
  )
  assert.deepEqual(
    gearCassettePreset('campagnolo-chorus-12-11-34')?.cogs,
    [11, 12, 13, 14, 15, 16, 17, 19, 22, 25, 29, 34],
  )
  assert.deepEqual(
    gearCassettePreset('campagnolo-ekar-9-36')?.cogs,
    [9, 10, 11, 12, 13, 14, 16, 18, 20, 23, 27, 31, 36],
  )
})

test('wide-range one-by cassettes disable the two-chainring layout', () => {
  const oneByIds = GEAR_CASSETTE_PRESET_GROUPS.flatMap(group => group.presets)
    .filter(preset => preset.maximumChainrings === 1)
    .map(preset => preset.id)

  assert.deepEqual(oneByIds, [
    'sram-red-xplr-xg-1391-10-46',
    'sram-force-xplr-xg-1371-10-46',
    'sram-rival-xplr-xg-1351-10-46',
    'sram-xplr-xg-1271-10-44',
    'sram-xplr-xg-1251-10-44',
    'campagnolo-super-record-x-9-42',
    'campagnolo-super-record-x-10-48',
    'campagnolo-record-x-10-48',
    'campagnolo-ekar-9-36',
    'campagnolo-ekar-9-42',
    'campagnolo-ekar-10-44',
  ])
  assert.equal(gearCassettePreset('campagnolo-super-record-13-11-36')?.maximumChainrings, 2)
  assert.equal(gearCassettePreset('campagnolo-super-record-x-10-48')?.maximumChainrings, 1)
  assert.equal(gearCassettePreset('campagnolo-record-x-10-48')?.maximumChainrings, 1)
  assert.equal(gearCassettePreset('campagnolo-ekar-10-44')?.maximumChainrings, 1)
})

test('gear ratio matrices cover every chainring and cassette combination', () => {
  const matrix = gearRatioMatrix([52, 36], [11, 34])

  assert.ok(matrix)
  assert.equal(matrix.maximum, 52 / 11)
  assert.equal(matrix.minimum, 36 / 34)
  assert.deepEqual(
    matrix.rows.map(row => [row.chainring, row.cells.map(cell => [cell.cog, cell.ratio])]),
    [
      [
        52,
        [
          [11, 52 / 11],
          [34, 52 / 34],
        ],
      ],
      [
        36,
        [
          [11, 36 / 11],
          [34, 36 / 34],
        ],
      ],
    ],
  )
  assert.equal(matrix.rows[0].cells[0].level, 1)
  assert.equal(matrix.rows[1].cells[1].level, 0)
})

test('CeramicSpeed estimates reproduce the tested cross-chain loss curve', () => {
  const matrix = gearRatioMatrix([53, 39], [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 28])

  assert.ok(matrix)
  assert.equal(matrix.rows[0].cells[3].crossChainLossWatts, 0)
  assert.equal(matrix.rows[0].cells[10].crossChainLossWatts, 2.01)
  assert.equal(matrix.rows[1].cells[5].crossChainLossWatts, 0)
  assert.equal(matrix.rows[1].cells[0].crossChainLossWatts, 1.4)
  assert.ok(Math.abs(matrix.rows[0].cells[3].drivetrainLossWatts - 6.97) < 0.11)
  assert.ok(Math.abs(matrix.rows[1].cells[5].drivetrainLossWatts - 6.89) < 0.11)
  assert.equal(
    matrix.rows[0].cells[3].drivetrainEfficiency,
    ((CERAMICSPEED_TEST_OUTPUT_WATTS - matrix.rows[0].cells[3].drivetrainLossWatts) /
      CERAMICSPEED_TEST_OUTPUT_WATTS) *
      100,
  )
})

test('CeramicSpeed research numbers retain their original sources and reproduce the fitted curves', () => {
  const { sources, test: protocol, alignedLoss, crossChainLoss } = CERAMICSPEED_CROSS_CHAIN_RESEARCH

  assert.deepEqual(
    sources.map(source => source.id),
    [1, 2],
  )
  assert.equal(sources[0].url, 'https://ceramicspeed.com/pages/cross-chaining-and-ring-size-report')
  assert.equal(
    sources[1].url,
    'https://cdn.shopify.com/s/files/1/0726/7542/6606/files/cross-chaining-and-ring-size-report.pdf?v=1687253624',
  )
  assert.equal(sources[1].publishedOn, '2015-04-17')
  assert.equal(protocol.sourceId, 2)
  assert.equal(protocol.protocolPage, 4)
  assert.equal(protocol.chainstayPage, 13)
  assert.equal(alignedLoss.page, 20)
  assert.equal(crossChainLoss.page, 15)
  assert.deepEqual(crossChainLoss.wattsByCogOffset, [0, 0.19, 0.49, 0.79, 1.1, 1.4, 1.7, 2.01])

  for (const curve of alignedLoss.curves) {
    const ratios = protocol.cassetteCogs.map(cog => curve.chainring / cog)
    const ratioMean = ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length
    const lossMean =
      curve.averageLossWatts.reduce((sum, loss) => sum + loss, 0) / curve.averageLossWatts.length
    const slopeNumerator = ratios.reduce(
      (sum, ratio, index) => sum + (ratio - ratioMean) * (curve.averageLossWatts[index] - lossMean),
      0,
    )
    const slopeDenominator = ratios.reduce((sum, ratio) => sum + (ratio - ratioMean) ** 2, 0)
    const slope = slopeNumerator / slopeDenominator
    const intercept = lossMean - slope * ratioMean

    assert.ok(Math.abs(slope - curve.fit.slope) < 1e-12)
    assert.ok(Math.abs(intercept - curve.fit.intercept) < 1e-12)
  }
})

test('one-by drivetrains align around the center of the cassette', () => {
  const matrix = gearRatioMatrix([40], [10, 11, 12, 13, 14])

  assert.ok(matrix)
  assert.deepEqual(
    matrix.rows[0].cells.map(cell => cell.crossChainLossWatts),
    [0.49, 0.19, 0, 0.19, 0.49],
  )
})

test('drivetrain efficiency preserves and formats its signed difference from ideal', () => {
  assert.equal(gearEfficiencyDeltaPercent(96.5), -3.5)
  assert.equal(gearEfficiencyDeltaPercent(98.6), -1.4)
  assert.equal(gearEfficiencyDeltaPercent(100), 0)
  assert.equal(formatGearEfficiencyDeltaPercent(96.4786, 2), '-3.52')
  assert.equal(formatGearEfficiencyDeltaPercent(98.6044, 3), '-1.396')
  assert.equal(formatGearEfficiencyDeltaPercent(100.125, 2), '+0.13')
})

test('gear ratio matrices support one-by drivetrains and reject invalid teeth', () => {
  const matrix = gearRatioMatrix([40], [10, 20, 40])

  assert.ok(matrix)
  assert.deepEqual(
    matrix.rows[0].cells.map(cell => cell.ratio),
    [4, 2, 1],
  )
  assert.equal(gearRatioMatrix([], [11, 28]), null)
  assert.equal(gearRatioMatrix([52, 36, 24], [11, 28]), null)
  assert.equal(gearRatioMatrix([52.5], [11, 28]), null)
  assert.equal(gearRatioMatrix([52], [0, 28]), null)
})

test('a single gear receives a neutral chart intensity', () => {
  const matrix = gearRatioMatrix([40], [20])

  assert.ok(matrix)
  assert.equal(matrix.minimum, 2)
  assert.equal(matrix.maximum, 2)
  assert.equal(matrix.rows[0].cells[0].level, 0.5)
})
