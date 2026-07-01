import assert from 'node:assert/strict'
import test from 'node:test'
import {
  aggregateAppleRecords,
  aggregateSwimLaps,
  latestAppleDate,
  matchAppleRecord,
  matchStrokeStyle,
  matchSwimDistance,
  matchSwimStrokeOpen,
  mergeAppleDay,
  parseAppleJson,
  type AppleDaily,
  type SwimStroke,
} from './apple'

function day(date: string, values: Partial<AppleDaily> = {}): AppleDaily {
  return {
    date,
    burnKcal: null,
    activeKcal: null,
    intakeKcal: null,
    weightKg: null,
    vo2max: null,
    ...values,
  }
}

test('latestAppleDate returns the newest valid Apple day', () => {
  assert.equal(
    latestAppleDate({
      old: day('2026-06-07'),
      bad: day('today'),
      current: day('2026-06-09'),
      middle: day('2026-06-08'),
    }),
    '2026-06-09',
  )
})

test('mergeAppleDay keeps prior values when a fresh import omits a metric', () => {
  assert.deepEqual(
    mergeAppleDay(
      day('2026-06-08', {
        burnKcal: 2500,
        activeKcal: 600,
        intakeKcal: 2100,
        weightKg: 90.5,
        vo2max: 48.2,
      }),
      day('2026-06-08', { burnKcal: 2600, activeKcal: null, intakeKcal: null, weightKg: 90.1 }),
    ),
    day('2026-06-08', {
      burnKcal: 2600,
      activeKcal: 600,
      intakeKcal: 2100,
      weightKg: 90.1,
      vo2max: 48.2,
    }),
  )
})

test('Apple VO2 max survives XML aggregation and JSON import', () => {
  const record = matchAppleRecord(
    '<Record type="HKQuantityTypeIdentifierVO2Max" sourceName="Apple Watch" unit="mL/min/kg" startDate="2026-06-08 07:00:00 -0400" value="49.26"/>',
  )

  assert.deepEqual(record, {
    date: '2026-06-08',
    kind: 'vo2max',
    value: 49.26,
    unit: 'mL/min/kg',
    source: 'Apple Watch',
  })
  assert.deepEqual(aggregateAppleRecords(record ? [record] : []), [
    day('2026-06-08', { vo2max: 49.3 }),
  ])
  assert.deepEqual(parseAppleJson({ days: [{ date: '2026-06-08', vo2max: 49.26 }] }).days, [
    day('2026-06-08', { vo2max: 49.3 }),
  ])
})

test('JSON import preserves swim aggregates from HealthExporter', () => {
  assert.deepEqual(
    parseAppleJson({
      swims: [
        {
          date: '2026-06-19T07:30:00-04:00',
          totalM: 1500.2,
          laps: 60,
          strokes: { freestyle: 1450.4, breaststroke: 49.6, mixed: null },
        },
      ],
    }).swims,
    [
      {
        date: '2026-06-19',
        totalM: 1500,
        laps: 60,
        strokes: { freestyle: 1450, breaststroke: 50 },
      },
    ],
  )
})

test('JSON import preserves workout heart-rate streams from HealthExporter', () => {
  assert.deepEqual(
    parseAppleJson({
      workouts: [
        {
          id: '7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63',
          activity: 'cycling',
          start: '2026-07-01T01:11:00.000Z',
          end: '2026-07-01T02:07:45Z',
          durationS: 3405.4,
          heartRate: [
            { time: '2026-07-01T01:11:09Z', bpm: 122.2 },
            { time: '2026-07-01T01:11:04.000Z', bpm: 117.6 },
            { time: 'wat', bpm: 200 },
            { time: '2026-07-01T01:11:12Z', bpm: 0 },
          ],
        },
      ],
    }).workouts,
    [
      {
        id: '7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63',
        activity: 'cycling',
        start: '2026-07-01T01:11:00Z',
        end: '2026-07-01T02:07:45Z',
        durationS: 3405,
        heartRate: [
          { time: '2026-07-01T01:11:04Z', bpm: 118 },
          { time: '2026-07-01T01:11:09Z', bpm: 122 },
        ],
      },
    ],
  )
})

test('matchSwimDistance reads the lap start + meters, converting non-metric units', () => {
  assert.deepEqual(
    matchSwimDistance(
      '<Record type="HKQuantityTypeIdentifierDistanceSwimming" sourceName="appl-watch-ultra-3" unit="m" startDate="2026-05-17 13:19:25 -0400" endDate="2026-05-17 13:19:52 -0400" value="25"/>',
    ),
    { start: '2026-05-17 13:19:25 -0400', meters: 25 },
  )
  const yards = matchSwimDistance(
    '<Record type="HKQuantityTypeIdentifierDistanceSwimming" unit="yd" startDate="2026-05-17 13:19:25 -0400" value="25"/>',
  )
  assert.ok(yards && Math.abs(yards.meters - 22.86) < 1e-6)
})

test('matchSwimStrokeOpen captures the lap start only for an open stroke-count record', () => {
  assert.equal(
    matchSwimStrokeOpen(
      '<Record type="HKQuantityTypeIdentifierSwimmingStrokeCount" unit="count" startDate="2026-05-17 13:19:25 -0400" endDate="2026-05-17 13:19:52 -0400" value="20">',
    ),
    '2026-05-17 13:19:25 -0400',
  )
  assert.equal(
    matchSwimStrokeOpen(
      '<Record type="HKQuantityTypeIdentifierSwimmingStrokeCount" startDate="2026-05-17 13:19:25 -0400" value="20"/>',
    ),
    null,
  )
  assert.equal(
    matchSwimStrokeOpen('<Record type="HKQuantityTypeIdentifierStepCount" value="20">'),
    null,
  )
})

test('matchStrokeStyle maps the HealthKit enum to a stroke name', () => {
  assert.equal(
    matchStrokeStyle('   <MetadataEntry key="HKSwimmingStrokeStyle" value="2"/>'),
    'freestyle',
  )
  assert.equal(
    matchStrokeStyle('   <MetadataEntry key="HKSwimmingStrokeStyle" value="4"/>'),
    'breaststroke',
  )
  assert.equal(matchStrokeStyle('   <MetadataEntry key="HKWasUserEntered" value="0"/>'), null)
})

test('aggregateSwimLaps pairs laps to distances, falls back to the day pool length, groups by date', () => {
  const laps: { start: string; stroke: SwimStroke }[] = [
    { start: '2026-05-17 13:19:25 -0400', stroke: 'freestyle' },
    { start: '2026-05-17 13:20:00 -0400', stroke: 'breaststroke' },
    { start: '2026-05-17 13:21:00 -0400', stroke: 'freestyle' },
  ]
  const distByStart = new Map<string, number>([
    ['2026-05-17 13:19:25 -0400', 25],
    ['2026-05-17 13:20:00 -0400', 25],
  ])
  assert.deepEqual(aggregateSwimLaps(laps, distByStart), [
    { date: '2026-05-17', laps: 3, totalM: 75, strokes: { freestyle: 50, breaststroke: 25 } },
  ])
})
