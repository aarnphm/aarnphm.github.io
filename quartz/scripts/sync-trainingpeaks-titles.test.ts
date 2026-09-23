import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseTrainingPeaksTitleArgs,
  trainingPeaksSaunaTracking,
} from './sync-trainingpeaks-titles'

test('defaults to dry run and validates bounded title-sync arguments', () => {
  assert.equal(parseTrainingPeaksTitleArgs([]).write, false)
  assert.equal(parseTrainingPeaksTitleArgs(['--write', '--dry-run']).write, false)
  const args = parseTrainingPeaksTitleArgs([
    '--write',
    '--since',
    '2026-09-01',
    '--until',
    '2026-09-20',
    '--id',
    '20261500024',
    '--limit',
    '1',
  ])
  assert.equal(args.write, true)
  assert.deepEqual([...args.ids], [20261500024])
  assert.equal(args.limit, 1)
  for (const argv of [
    ['--sources', '--write'],
    ['--since', '2026-02-30'],
    ['--id', '0'],
    ['--limit', 'NaN'],
    ['--since'],
    ['--unknown'],
  ])
    assert.throws(() => parseTrainingPeaksTitleArgs(argv))
  assert.throws(() =>
    parseTrainingPeaksTitleArgs(['--since', '2026-09-20', '--until', '2026-09-01']),
  )
})

test('reads real Markdown tracking nodes and preserves explicit Strava identity', () => {
  const markdown =
    '```tracking\ntitle: Free Flow\ndate: 2026-09-20\ntime: 18:30\nduration: 75 mins\nactivity: sauna\ntemperature: 167F\nhumidity: 11%\ncooldown: cold plunge\nstrava: 20261500024\n```'
  const tracking = trainingPeaksSaunaTracking(markdown)
  assert.equal(tracking.length, 1)
  assert.equal(tracking[0].stravaActivityId, 20261500024)
  assert.equal(trainingPeaksSaunaTracking(markdown.replace('```tracking', '```text')).length, 0)
})
