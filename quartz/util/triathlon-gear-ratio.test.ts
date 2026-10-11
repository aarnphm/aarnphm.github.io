import assert from 'node:assert/strict'
import test from 'node:test'
import {
  initialGearRatioModel,
  updateGearRatio,
} from '../components/triathlon/tools/gear-ratio-model'
import {
  CERAMICSPEED_TEST_OUTPUT_WATTS,
  GEAR_CASSETTE_PRESET_GROUPS,
  formatGearEfficiencyDeltaPercent,
  gearEfficiencyDeltaPercent,
  gearCassettePreset,
  gearRatioMatrix,
} from './triathlon-gear-ratio'

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

test('cassette and chainring selections keep a compatible layout and invalidate edited presets', () => {
  const road = gearCassettePreset('shimano-ultegra-r8100-11-34')
  const cassette = gearCassettePreset('sram-xplr-xg-1271-10-44')
  assert.ok(road)
  assert.ok(cassette)
  const initial = initialGearRatioModel(road.id, '54-40', 54, 40)
  const compact = updateGearRatio(initial, {
    type: 'set-cassette',
    cassetteId: cassette.id,
    maximumChainrings: cassette.maximumChainrings,
  }).model
  const matrix = gearRatioMatrix(compact.chainrings.slice(0, compact.layout), cassette.cogs)
  assert.ok(matrix)
  assert.equal(matrix.rows.length, 1)

  const compatible = updateGearRatio(compact, {
    type: 'set-cassette',
    cassetteId: road.id,
    maximumChainrings: road.maximumChainrings,
  }).model
  const selected = updateGearRatio(compatible, {
    type: 'set-chainring-preset',
    presetId: '53-39',
    chainrings: [53, 39],
  }).model
  assert.equal(selected.layout, 2)
  const changed = updateGearRatio(selected, { type: 'set-chainring', index: 0, value: 54 }).model
  assert.equal(changed.chainringPresetId, null)
  assert.deepEqual(changed.chainrings, [54, 39])
})

test('a single gear receives a neutral chart intensity', () => {
  const matrix = gearRatioMatrix([40], [20])

  assert.ok(matrix)
  assert.equal(matrix.minimum, 2)
  assert.equal(matrix.maximum, 2)
  assert.equal(matrix.rows[0].cells[0].level, 0.5)
})
