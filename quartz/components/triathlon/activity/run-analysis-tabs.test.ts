import assert from 'node:assert/strict'
import test from 'node:test'
import { runAnalysisViewFromKey } from './run-analysis-tabs'

test('run analysis tabs wrap across arrows and respect boundary keys', () => {
  assert.equal(runAnalysisViewFromKey('workout', 'ArrowRight'), 'laps')
  assert.equal(runAnalysisViewFromKey('workout', 'ArrowLeft'), 'pace')
  assert.equal(runAnalysisViewFromKey('laps', 'ArrowRight'), 'pace')
  assert.equal(runAnalysisViewFromKey('laps', 'ArrowLeft'), 'workout')
  assert.equal(runAnalysisViewFromKey('laps', 'Home'), 'workout')
  assert.equal(runAnalysisViewFromKey('workout', 'End'), 'pace')
  assert.equal(runAnalysisViewFromKey('workout', 'Enter'), null)
})

test('run analysis navigation follows the views available on one activity', () => {
  const views = ['workout', 'laps'] as const
  assert.equal(runAnalysisViewFromKey('workout', 'ArrowLeft', views), 'laps')
  assert.equal(runAnalysisViewFromKey('laps', 'ArrowRight', views), 'workout')
  assert.equal(runAnalysisViewFromKey('pace', 'ArrowRight', views), null)
})
