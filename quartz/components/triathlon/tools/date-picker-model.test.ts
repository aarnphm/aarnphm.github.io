import assert from 'node:assert/strict'
import test from 'node:test'
import { initialDatePickerModel, updateDatePicker } from './date-picker-model'

test('date picker reducer owns open, month, selection, and clear transitions', () => {
  const opened = updateDatePicker(initialDatePickerModel(), {
    type: 'open',
    selected: '2026-08-10',
    viewMonth: '2026-08',
  })
  assert.equal(opened.model.open, true)
  assert.deepEqual(opened.effects, [{ type: 'render' }])

  const moved = updateDatePicker(opened.model, {
    type: 'focus-date',
    date: '2026-09-01',
    viewMonth: '2026-09',
  })
  assert.equal(moved.model.viewMonth, '2026-09')
  assert.deepEqual(moved.effects, [{ type: 'render', focusDate: '2026-09-01' }])

  const selected = updateDatePicker(moved.model, { type: 'select', date: '2026-09-01' })
  assert.equal(selected.model.selected, '2026-09-01')
  assert.equal(selected.model.open, false)

  const cleared = updateDatePicker(selected.model, { type: 'clear' })
  assert.equal(cleared.model.selected, undefined)
})
