import assert from 'node:assert/strict'
import test from 'node:test'
import { calculatorTabFromHash } from './calculator-tabs'

test('calculator tab hashes select only registered calculators', () => {
  assert.equal(calculatorTabFromHash(''), 'race')
  assert.equal(calculatorTabFromHash('#race'), 'race')
  assert.equal(calculatorTabFromHash('#gear-ratios'), 'gear-ratios')
  assert.equal(calculatorTabFromHash('#tire-pressure'), 'tire-pressure')
  assert.equal(calculatorTabFromHash('#calculator-arbitrary-share'), 'race')
})
