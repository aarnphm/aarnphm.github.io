import assert from 'node:assert/strict'
import test from 'node:test'
import { visitorHoliday } from './visitor-holiday'

const local = (year: number, month: number, day: number, hour = 12) =>
  new Date(year, month - 1, day, hour)

test('Canadian Thanksgiving follows the second Monday and its local holiday weekend', () => {
  assert.equal(visitorHoliday(local(2026, 10, 8), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 10, 9, 0), 'CA'), 'thanksgiving')
  assert.equal(visitorHoliday(local(2026, 10, 12), 'CA'), 'thanksgiving')
  assert.equal(visitorHoliday(new Date(2026, 9, 12, 23, 59, 59, 999), 'CA'), 'thanksgiving')
  assert.equal(visitorHoliday(local(2026, 10, 13, 0), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 10, 14), 'CA'), null)
  assert.equal(visitorHoliday(local(2027, 10, 11), 'CA'), 'thanksgiving')
  assert.equal(visitorHoliday(local(2027, 10, 13), 'CA'), null)
})

test('US Thanksgiving follows the fourth Thursday without taking the Canadian date', () => {
  assert.equal(visitorHoliday(local(2026, 10, 12), 'US'), 'columbus-day')
  assert.equal(visitorHoliday(local(2026, 11, 22), 'US'), null)
  assert.equal(visitorHoliday(local(2026, 11, 26), 'US'), 'thanksgiving')
  assert.equal(visitorHoliday(new Date(2026, 10, 26, 23, 59, 59, 999), 'US'), 'thanksgiving')
  assert.equal(visitorHoliday(local(2026, 11, 27, 0), 'US'), null)
  assert.equal(visitorHoliday(local(2026, 11, 28), 'US'), null)
  assert.equal(visitorHoliday(local(2027, 11, 25), 'US'), 'thanksgiving')
  assert.equal(visitorHoliday(local(2026, 11, 26), 'CA'), null)
})

test('Columbus Day follows the second Monday in October for US visitors', () => {
  assert.equal(visitorHoliday(local(2026, 10, 8), 'US'), null)
  assert.equal(visitorHoliday(local(2026, 10, 9, 0), 'US'), 'columbus-day')
  assert.equal(visitorHoliday(local(2026, 10, 12), 'US'), 'columbus-day')
  assert.equal(visitorHoliday(new Date(2026, 9, 12, 23, 59, 59, 999), 'US'), 'columbus-day')
  assert.equal(visitorHoliday(local(2026, 10, 13, 0), 'US'), null)
  assert.equal(visitorHoliday(local(2026, 10, 14), 'US'), null)
  assert.equal(visitorHoliday(local(2027, 10, 11), 'US'), 'columbus-day')
  assert.equal(visitorHoliday(local(2027, 10, 13), 'US'), null)
  assert.equal(visitorHoliday(local(2026, 10, 12), ' us '), 'columbus-day')
  assert.equal(visitorHoliday(local(2026, 10, 12), 'ca'), 'thanksgiving')
})

test('unknown regions wait for country data instead of adopting Canadian Thanksgiving', () => {
  for (const country of [null, '', 'GB', 'FR', 'AU']) {
    assert.equal(visitorHoliday(local(2026, 10, 12), country), null)
    assert.equal(visitorHoliday(local(2026, 11, 26), country), null)
    assert.equal(visitorHoliday(local(2026, 10, 31), country), 'halloween')
    assert.equal(visitorHoliday(local(2026, 12, 25), country), 'christmas')
  }
})

test('Halloween crosses the month boundary without extending beyond its window', () => {
  assert.equal(visitorHoliday(local(2026, 10, 27), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 10, 28, 0), 'CA'), 'halloween')
  assert.equal(visitorHoliday(local(2026, 10, 31), 'US'), 'halloween')
  assert.equal(visitorHoliday(new Date(2026, 9, 31, 23, 59, 59, 999), 'CA'), 'halloween')
  assert.equal(visitorHoliday(local(2026, 11, 1, 0), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 11, 2), 'CA'), null)
})

test('Christmas is a local calendar window and ends at the next local midnight', () => {
  assert.equal(visitorHoliday(local(2026, 12, 19), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 12, 20, 0), 'CA'), 'christmas')
  assert.equal(visitorHoliday(local(2026, 12, 25), 'CA'), 'christmas')
  assert.equal(visitorHoliday(new Date(2026, 11, 25, 23, 59, 59, 999), 'CA'), 'christmas')
  assert.equal(visitorHoliday(local(2026, 12, 26, 0), 'CA'), null)
  assert.equal(visitorHoliday(local(2026, 12, 27), 'CA'), null)
  assert.equal(visitorHoliday(local(2027, 1, 1), 'CA'), null)
})

test('Gregorian Easter moves between March and April across years', () => {
  for (const [year, month, day] of [
    [2010, 4, 4],
    [2024, 3, 31],
    [2026, 4, 5],
    [2027, 3, 28],
  ]) {
    assert.equal(visitorHoliday(local(year, month, day - 3), 'CA'), null)
    assert.equal(visitorHoliday(local(year, month, day - 2, 0), 'CA'), 'easter')
    assert.equal(visitorHoliday(local(year, month, day), 'CA'), 'easter')
    assert.equal(visitorHoliday(new Date(year, month - 1, day, 23, 59, 59, 999), 'CA'), 'easter')
    assert.equal(visitorHoliday(local(year, month, day + 1, 0), 'CA'), null)
    assert.equal(visitorHoliday(local(year, month, day + 2), 'CA'), null)
  }
})

test('Easter retains the ecclesiastical dates documented by the Naval Observatory', () => {
  assert.equal(visitorHoliday(local(1954, 4, 18), 'CA'), 'easter')
  assert.equal(visitorHoliday(local(1954, 4, 20), 'CA'), null)
  assert.equal(visitorHoliday(local(1962, 4, 22), 'CA'), 'easter')
  assert.equal(visitorHoliday(local(1962, 4, 24), 'CA'), null)
})

test('invalid dates and dates outside the Gregorian service range are unavailable', () => {
  assert.equal(visitorHoliday(new Date(Number.NaN), 'CA'), null)
  assert.equal(visitorHoliday(local(1200, 4, 1), 'CA'), null)
})
