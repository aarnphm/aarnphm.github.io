import assert from 'node:assert/strict'
import test from 'node:test'
import { runAnalysisViewFromKey } from './run-analysis-tabs'

test('run analysis tabs wrap across arrows and respect boundary keys', () => {
  assert.equal(runAnalysisViewFromKey('workout', 'ArrowRight'), 'laps')
  assert.equal(runAnalysisViewFromKey('workout', 'ArrowLeft'), 'laps')
  assert.equal(runAnalysisViewFromKey('laps', 'ArrowRight'), 'workout')
  assert.equal(runAnalysisViewFromKey('laps', 'ArrowLeft'), 'workout')
  assert.equal(runAnalysisViewFromKey('laps', 'Home'), 'workout')
  assert.equal(runAnalysisViewFromKey('workout', 'End'), 'laps')
  assert.equal(runAnalysisViewFromKey('workout', 'Enter'), null)
})
