import assert from 'node:assert/strict'
import test from 'node:test'
import { workoutAnalysisViewFromKey } from './workout-analysis-tabs'

test('workout analysis tabs wrap across arrows and respect boundary keys', () => {
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowRight'), 'laps')
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowLeft'), 'power')
  assert.equal(workoutAnalysisViewFromKey('laps', 'ArrowRight'), 'pace')
  assert.equal(workoutAnalysisViewFromKey('laps', 'ArrowLeft'), 'workout')
  assert.equal(workoutAnalysisViewFromKey('laps', 'Home'), 'workout')
  assert.equal(workoutAnalysisViewFromKey('workout', 'End'), 'power')
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

test('cycling power distribution supports arrows and boundary keys', () => {
  const views = ['workout', 'power'] as const
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowRight', views), 'power')
  assert.equal(workoutAnalysisViewFromKey('workout', 'ArrowLeft', views), 'power')
  assert.equal(workoutAnalysisViewFromKey('power', 'ArrowRight', views), 'workout')
  assert.equal(workoutAnalysisViewFromKey('power', 'Home', views), 'workout')
  assert.equal(workoutAnalysisViewFromKey('workout', 'End', views), 'power')
  assert.equal(workoutAnalysisViewFromKey('pace', 'ArrowRight', views), null)
})
