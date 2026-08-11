import assert from 'node:assert/strict'
import test from 'node:test'
import type { TrainingPlan } from '../../../plugins/stores/training'
import { filterTrainingPlans, initialTrainingModel, updateTraining } from './model'
import { deriveTrainingDocument } from './tree'

const plans: TrainingPlan[] = [
  {
    id: 'plan-0',
    meta: 'Toronto Olympic',
    distance: 'olympic',
    date: '2026-07-26',
    target: 'sub-3',
    author: 'Coach One, Coach Two',
    html: '<h2>Build</h2><h3>Week one</h3><h4>Bike</h4><h3>Week two</h3><section data-footnotes=""><h2>Notes</h2></section>',
  },
  {
    id: 'plan-1',
    meta: 'Montreal Sprint',
    distance: 'sprint',
    date: '2026-06-10',
    target: 'finish',
    author: '',
    html: '<h2>Base</h2>',
  },
]

test('training reducer preserves a valid selection across artifact loads', () => {
  const selected = updateTraining(initialTrainingModel(plans), {
    type: 'select-plan',
    id: 'plan-1',
  }).model
  const loading = updateTraining(selected, { type: 'load' })
  assert.equal(loading.model.status, 'loading')
  assert.deepEqual(loading.effects, [{ type: 'load-artifact' }])
  const loaded = updateTraining(loading.model, {
    type: 'loaded',
    plans: [...plans].reverse(),
  }).model
  assert.equal(loaded.status, 'ready')
  assert.equal(loaded.selectedPlanId, 'plan-1')
})

test('training reducer falls back to the first plan when a selection disappears', () => {
  const initial = { ...initialTrainingModel(plans), selectedPlanId: 'missing' }
  const loaded = updateTraining(initial, { type: 'loaded', plans }).model
  assert.equal(loaded.selectedPlanId, 'plan-0')
})

test('training filtering searches the stable plan metadata', () => {
  assert.deepEqual(
    filterTrainingPlans(plans, 'SPRINT finish').map(plan => plan.id),
    ['plan-1'],
  )
  assert.equal(filterTrainingPlans(plans, '   '), plans)
})

test('training document derivation assigns deterministic heading ids and excludes footnotes', () => {
  const document = deriveTrainingDocument(plans[0])
  assert.match(document.html, /id="tri-h-plan-0-0"/)
  assert.deepEqual(document.tree, [
    {
      id: 'tri-h-plan-0-0',
      label: 'Build',
      level: 2,
      children: [
        {
          id: 'tri-h-plan-0-1',
          label: 'Week one',
          level: 3,
          children: [{ id: 'tri-h-plan-0-2', label: 'Bike', level: 4, children: [] }],
        },
        { id: 'tri-h-plan-0-3', label: 'Week two', level: 3, children: [] },
      ],
    },
  ])
})
