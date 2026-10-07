import type { Element, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarWorkout,
} from '../../../util/trainingpeaks-calendar'
import { parseTrainingPeaksCalendar } from '../../../util/trainingpeaks-calendar'
import { calendarToday } from './display'
import { trainingWeekStart } from './training-display'
import { TrainingCalendarView } from './TrainingCalendar'

const recordedRun: TrainingPeaksCalendarWorkout = {
  id: '101',
  source: 'trainingpeaks',
  date: '2020-01-06',
  title: 'Easy run',
  sport: 'run',
  workoutTypeId: 3,
  description: '',
  planned: { durationSeconds: 3600, distanceMeters: null, tss: null },
  actual: { durationSeconds: 3600, distanceMeters: null, tss: null },
  status: 'completed',
  startTime: '09:00:00',
  startTimePlanned: null,
  order: null,
}

function renderedWorkouts(workouts: TrainingPeaksCalendarWorkout[]): Element[] {
  const calendar: TrainingPeaksCalendar = {
    version: 1,
    source: 'trainingpeaks',
    fetchedAt: '2026-10-06T19:00:00.000Z',
    coverage: [{ since: '2020-01-01', until: '2099-12-31', fetchedAt: '2026-10-06T19:00:00.000Z' }],
    workouts,
  }
  const payload: unknown = JSON.parse(JSON.stringify(calendar))
  const parsed = parseTrainingPeaksCalendar(payload)
  assert.ok(parsed, 'the public payload must survive JSON and validation')
  const html = renderToString(
    <TrainingCalendarView
      calendar={parsed}
      week={trainingWeekStart(workouts[0].date)}
      id="test-training"
    />,
  )
  const found: Element[] = []
  const visit = (nodes: RootContent[]): void => {
    for (const node of nodes) {
      if (node.type !== 'element') continue
      if (node.properties.dataTrainingWorkout) found.push(node)
      visit(node.children)
    }
  }
  visit(fromHtml(html, { fragment: true }).children)
  return found
}

test('calendar renders execution boundaries from the serialized provider metrics', () => {
  const cases = [
    { seconds: 1799, status: 'far-off-target' },
    { seconds: 1800, status: 'off-target' },
    { seconds: 2879, status: 'off-target' },
    { seconds: 2880, status: 'on-target' },
    { seconds: 4320, status: 'on-target' },
    { seconds: 4321, status: 'off-target' },
    { seconds: 5400, status: 'off-target' },
    { seconds: 5401, status: 'far-off-target' },
  ]
  const cards = renderedWorkouts(
    cases.map((item, index) => ({
      ...recordedRun,
      id: String(index + 1),
      actual: { ...recordedRun.actual, durationSeconds: item.seconds },
    })),
  )
  assert.deepEqual(
    cards.map(card => card.properties.dataExecution),
    cases.map(item => item.status),
  )
})

test('calendar distinguishes missing metrics, unplanned work, and an explicitly recorded zero', () => {
  const cards = renderedWorkouts([
    {
      ...recordedRun,
      id: '1',
      planned: { durationSeconds: null, distanceMeters: null, tss: null },
    },
    { ...recordedRun, id: '2', actual: { durationSeconds: null, distanceMeters: 5000, tss: null } },
    { ...recordedRun, id: '3', actual: { durationSeconds: 0, distanceMeters: 5000, tss: null } },
    {
      ...recordedRun,
      id: '4',
      planned: { durationSeconds: 3600, distanceMeters: 5000, tss: null },
      actual: { durationSeconds: null, distanceMeters: 5000, tss: null },
    },
    {
      ...recordedRun,
      id: '5',
      planned: { durationSeconds: null, distanceMeters: null, tss: 100 },
      actual: { durationSeconds: null, distanceMeters: null, tss: 160 },
    },
  ])
  assert.deepEqual(
    cards.map(card => card.properties.dataExecution),
    ['unplanned', 'unavailable', 'far-off-target', 'on-target', 'far-off-target'],
  )
})

test('calendar marks only past incomplete sessions as missed', () => {
  for (const [date, expected] of [
    ['2020-01-06', 'missed'],
    [calendarToday(), 'planned'],
    ['2099-01-05', 'planned'],
  ]) {
    const [card] = renderedWorkouts([
      {
        ...recordedRun,
        date,
        status: 'planned',
        actual: { durationSeconds: null, distanceMeters: null, tss: null },
      },
    ])
    assert.equal(card.properties.dataExecution, expected, date)
  }
})
