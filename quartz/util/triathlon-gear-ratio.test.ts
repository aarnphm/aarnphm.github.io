import assert from 'node:assert/strict'
import test from 'node:test'
import {
  DEFAULT_GEAR_CASSETTE,
  GEAR_CASSETTE_PRESET_GROUPS,
  gearCassettePreset,
  gearRatioMatrix,
} from './triathlon-gear-ratio'

test('the declared bike drivetrain is the default preset', () => {
  assert.equal(DEFAULT_GEAR_CASSETTE.id, 'shimano-ultegra-r8100-11-34')
  assert.deepEqual(DEFAULT_GEAR_CASSETTE.cogs, [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34])
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
