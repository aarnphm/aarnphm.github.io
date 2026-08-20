import assert from 'node:assert/strict'
import test from 'node:test'
import { calculatorTabFromHash, calculatorTabFromShortcut } from './calculator-tabs'

test('calculator tab hashes select only registered calculators', () => {
  assert.equal(calculatorTabFromHash(''), 'race')
  assert.equal(calculatorTabFromHash('#race'), 'race')
  assert.equal(calculatorTabFromHash('#gear-ratios'), 'gear-ratios')
  assert.equal(calculatorTabFromHash('#tire-pressure'), 'tire-pressure')
  assert.equal(calculatorTabFromHash('#calculator-arbitrary-share'), 'race')
})

test('calculator tab shortcuts use bare keys without consuming navigation suffixes', () => {
  assert.equal(calculatorTabFromShortcut('r'), 'race')
  assert.equal(calculatorTabFromShortcut('g'), 'gear-ratios')
  assert.equal(calculatorTabFromShortcut('t'), 'tire-pressure')
  assert.equal(calculatorTabFromShortcut('c'), null)
  assert.equal(calculatorTabFromShortcut('a'), null)
  assert.equal(calculatorTabFromShortcut('m'), null)
  assert.equal(calculatorTabFromShortcut('G'), null)
})
