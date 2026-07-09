import assert from 'node:assert/strict'
import test from 'node:test'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { BuiltinSummaryType } from './types'
import { computeColumnSummary } from './query'

const summarize = (
  values: unknown[],
  builtinType: BuiltinSummaryType,
): string | number | undefined => {
  const files: QuartzPluginData[] = values.map(value => ({ value }))
  return computeColumnSummary(
    'value',
    files,
    { type: 'builtin', builtinType },
    files,
    file => file.value,
    file => ({ file, allFiles: files }),
  )
}

test('string extrema use lexical ordering', () => {
  const values = ['gamma', 'alpha', 'beta']

  assert.equal(summarize(values, 'min'), 'alpha')
  assert.equal(summarize(values, 'max'), 'gamma')
})

test('numeric extrema remain numeric when strings are also present', () => {
  const values = ['alpha', 7, 2]

  assert.equal(summarize(values, 'min'), 2)
  assert.equal(summarize(values, 'max'), 7)
})

test('date summaries exclude invalid date values', () => {
  const invalidDate = new Date(Number.NaN)
  const validDate = new Date('2026-07-09T12:00:00Z')

  assert.equal(summarize([invalidDate], 'min'), undefined)
  assert.equal(summarize([invalidDate], 'max'), undefined)
  assert.equal(summarize([invalidDate], 'range'), undefined)
  assert.equal(summarize([invalidDate], 'unique'), 0)
  assert.equal(summarize([invalidDate, validDate], 'min'), '2026-07-09')
  assert.equal(summarize([invalidDate, validDate], 'max'), '2026-07-09')
})

test('date extrema ignore invalid strings and timestamps', () => {
  const values = ['2026-13-40', Number.POSITIVE_INFINITY, '2026-07-09']

  assert.equal(summarize(values, 'earliest'), '2026-07-09')
  assert.equal(summarize(values, 'latest'), '2026-07-09')
  assert.equal(summarize(['2026-13-40'], 'earliest'), undefined)
  assert.equal(summarize([Number.POSITIVE_INFINITY], 'latest'), undefined)
})

test('numeric summaries ignore non-finite values', () => {
  const values = [1, 3, Number.NaN, Number.POSITIVE_INFINITY]

  assert.equal(summarize(values, 'sum'), 4)
  assert.equal(summarize(values, 'average'), 2)
  assert.equal(summarize(values, 'median'), 2)
  assert.equal(summarize(values, 'stddev'), 1)
})
