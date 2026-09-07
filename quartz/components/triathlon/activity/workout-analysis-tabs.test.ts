import assert from 'node:assert/strict'
import test from 'node:test'
import { workoutAnalysisViewFromKey } from './workout-analysis-tabs'

test('workout analysis tabs wrap across arrows and respect boundary keys', () => {
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowRight'), 'laps')
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowLeft'), 'pace')
  assert.equal(workoutAnalysisViewFromKey('laps', 'ArrowRight'), 'pace')
  assert.equal(workoutAnalysisViewFromKey('laps', 'ArrowLeft'), 'workout')
  assert.equal(workoutAnalysisViewFromKey('laps', 'Home'), 'workout')
  assert.equal(workoutAnalysisViewFromKey('workout', 'End'), 'pace')
  assert.equal(workoutAnalysisViewFromKey('workout', 'Enter'), null)
})

test('workout analysis navigation follows the views available on one activity', () => {
  const views = ['workout', 'laps'] as const
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowLeft', views), 'laps')
  assert.equal(workoutAnalysisViewFromKey('laps', 'ArrowRight', views), 'workout')
  assert.equal(workoutAnalysisViewFromKey('pace', 'ArrowRight', views), null)
})

test('single-view swim and cycling analysis tabs keep their active panel', () => {
  for (const key of ['ArrowLeft', 'ArrowRight', 'Home', 'End'])
    assert.equal(workoutAnalysisViewFromKey('workout', key, ['workout']), 'workout')
  assert.equal(workoutAnalysisViewFromKey('workout', 'Enter', ['workout']), null)
})
