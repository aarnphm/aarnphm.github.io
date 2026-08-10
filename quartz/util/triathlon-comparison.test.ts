import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activityComparisonEmbed,
  decodeActivityComparisonAnchor,
  encodeActivityComparisonAnchor,
} from './triathlon-comparison'

test('encodes comparison activity ids as a stable triathlon transclusion', () => {
  const activityIds = ['19471122670', '19476629599', '19477504076']

  assert.equal(
    encodeActivityComparisonAnchor(activityIds),
    'comparison-19471122670-19476629599-19477504076',
  )
  assert.equal(
    activityComparisonEmbed(activityIds),
    '![[triathlon#comparison-19471122670-19476629599-19477504076]]',
  )
  assert.deepEqual(
    decodeActivityComparisonAnchor('#comparison-19471122670-19476629599-19477504076'),
    activityIds,
  )
})

test('rejects malformed, duplicate, and undersized comparison anchors', () => {
  for (const value of [
    '',
    'calculator-1-a-i-120-120-2500-90-480',
    'comparison-19471122670',
    'comparison-19471122670-activity',
    'comparison-19471122670-19471122670',
    'comparison-0-19471122670',
  ]) {
    assert.equal(decodeActivityComparisonAnchor(value), null)
  }
  assert.equal(encodeActivityComparisonAnchor(['19471122670']), null)
  assert.equal(activityComparisonEmbed(['19471122670', '19471122670']), null)
})
