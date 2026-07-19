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
  matchSwimWorkout,
  mergeAppleDay,
  parseAppleJson,
  type AppleDaily,
  type AppleSwimDistanceRecord,
  type AppleSwimLap,
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
        id: null,
        date: '2026-06-19',
        start: null,
        end: null,
        totalM: 1500,
        laps: 60,
        activeTimeS: null,
        strokeCount: null,
        strokeTimeS: null,
        strokes: { freestyle: 1450, breaststroke: 50 },
        intervals: [],
      },
    ],
  )
})

test('JSON import keeps separate swim sessions and their rate inputs', () => {
  assert.deepEqual(
    parseAppleJson({
      version: 3,
      swims: [
        {
          id: 'morning',
          date: '2026-06-19',
          start: '2026-06-19T11:00:00Z',
          end: '2026-06-19T12:00:00Z',
          totalM: 1500,
          laps: 60,
          activeTimeS: 1800.4,
          strokeCount: 960.2,
          strokeTimeS: 1700.4,
          strokes: { freestyle: 1500 },
          intervals: [
            {
              start: '2026-06-19T11:00:05.000Z',
              end: '2026-06-19T11:00:30Z',
              distanceM: 22.86,
              startElapsedS: 5.4,
              endElapsedS: 31.1,
              durationS: 25.7,
              strokeCount: 16.25,
              strokeTimeS: 20.25,
              stroke: 'freestyle',
            },
            { start: 'bad', end: '2026-06-19T11:01:00Z', distanceM: 25, stroke: 'backstroke' },
          ],
        },
        {
          id: 'evening',
          date: '2026-06-19',
          start: '2026-06-19T22:00:00Z',
          end: '2026-06-19T23:00:00Z',
          totalM: 1000,
          laps: 40,
          activeTimeS: 1400,
          strokeCount: 600,
          strokeTimeS: 1200,
          strokes: { breaststroke: 1000 },
        },
      ],
    }).swims,
    [
      {
        id: 'morning',
        date: '2026-06-19',
        start: '2026-06-19T11:00:00Z',
        end: '2026-06-19T12:00:00Z',
        totalM: 1500,
        laps: 60,
        activeTimeS: 1800,
        strokeCount: 960,
        strokeTimeS: 1700,
        strokes: { freestyle: 1500 },
        intervals: [
          {
            start: '2026-06-19T11:00:05Z',
            end: '2026-06-19T11:00:30Z',
            distanceM: 22.9,
            startElapsedS: 5.4,
            endElapsedS: 31.1,
            durationS: 25.7,
            strokeCount: 16.3,
            strokeTimeS: 20.3,
            stroke: 'freestyle',
          },
        ],
      },
      {
        id: 'evening',
        date: '2026-06-19',
        start: '2026-06-19T22:00:00Z',
        end: '2026-06-19T23:00:00Z',
        totalM: 1000,
        laps: 40,
        activeTimeS: 1400,
        strokeCount: 600,
        strokeTimeS: 1200,
        strokes: { breaststroke: 1000 },
        intervals: [],
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
          elapsedTimeS: 3769.4,
          distanceM: 9214.48,
          activeEnergyKcal: 741.04,
          averageHeartRateBpm: 155.6,
          averageRunningPowerW: 277.7,
          averageCadenceSpm: 159.2,
          lapCount: 6.2,
          source: ' Strava ',
          device: ' Apple Watch Ultra 3 49mm ',
          gpxFile: 'GPX/7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63.gpx',
          heartRate: [
            { time: '2026-07-01T01:11:09Z', bpm: 122.2 },
            { time: '2026-07-01T01:11:04.000Z', bpm: 117.6 },
            { time: 'wat', bpm: 200 },
            { time: '2026-07-01T01:11:12Z', bpm: 0 },
          ],
          strideLengthM: [
            { time: '2026-07-01T01:11:09Z', value: 1.21 },
            { time: '2026-07-01T01:11:04.000Z', value: 1.18 },
            { time: 'wat', value: 1.3 },
            { time: '2026-07-01T01:11:12Z', value: 4 },
          ],
          groundContactTimeMs: [{ time: '2026-07-01T01:11:04Z', value: 241 }],
          verticalOscillationCm: [{ time: '2026-07-01T01:11:04Z', value: 9.8 }],
        },
        {
          id: 'A45B1F35-9F51-4917-B656-C17BF2D07434',
          activity: 'swimming',
          start: '2026-07-02T11:00:00Z',
          end: '2026-07-02T12:00:00Z',
          durationS: 3600,
          heartRate: [],
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
        elapsedTimeS: 3769,
        distanceM: 9214.5,
        activeEnergyKcal: 741,
        averageHeartRateBpm: 156,
        averageRunningPowerW: 278,
        averageCadenceSpm: 159,
        lapCount: 6,
        source: 'Strava',
        device: 'Apple Watch Ultra 3 49mm',
        gpxFile: 'GPX/7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63.gpx',
        heartRate: [
          { time: '2026-07-01T01:11:04Z', bpm: 118 },
          { time: '2026-07-01T01:11:09Z', bpm: 122 },
        ],
        strideLengthM: [
          { time: '2026-07-01T01:11:04Z', value: 1.18 },
          { time: '2026-07-01T01:11:09Z', value: 1.21 },
        ],
        groundContactTimeMs: [{ time: '2026-07-01T01:11:04Z', value: 241 }],
        verticalOscillationCm: [{ time: '2026-07-01T01:11:04Z', value: 9.8 }],
      },
      {
        id: 'A45B1F35-9F51-4917-B656-C17BF2D07434',
        activity: 'swimming',
        start: '2026-07-02T11:00:00Z',
        end: '2026-07-02T12:00:00Z',
        durationS: 3600,
        heartRate: [],
        strideLengthM: [],
        groundContactTimeMs: [],
        verticalOscillationCm: [],
      },
    ],
  )
})

test('matchSwimDistance reads the lap start + meters, converting non-metric units', () => {
  assert.deepEqual(
    matchSwimDistance(
      '<Record type="HKQuantityTypeIdentifierDistanceSwimming" sourceName="appl-watch-ultra-3" unit="m" startDate="2026-05-17 13:19:25 -0400" endDate="2026-05-17 13:19:52 -0400" value="25"/>',
    ),
    { start: '2026-05-17 13:19:25 -0400', end: '2026-05-17 13:19:52 -0400', meters: 25 },
  )
  const yards = matchSwimDistance(
    '<Record type="HKQuantityTypeIdentifierDistanceSwimming" unit="yd" startDate="2026-05-17 13:19:25 -0400" value="25"/>',
  )
  assert.ok(yards && Math.abs(yards.meters - 22.86) < 1e-6)
  assert.equal(
    matchSwimDistance(
      '<Record type="HKQuantityTypeIdentifierDistanceSwimming" unit="poolLength" startDate="2026-05-17 13:19:25 -0400" value="25"/>',
    ),
    null,
  )
})

test('matchSwimStrokeOpen captures the stroke count interval', () => {
  assert.deepEqual(
    matchSwimStrokeOpen(
      '<Record type="HKQuantityTypeIdentifierSwimmingStrokeCount" unit="count" startDate="2026-05-17 13:19:25 -0400" endDate="2026-05-17 13:19:52 -0400" value="20">',
    ),
    { start: '2026-05-17 13:19:25 -0400', end: '2026-05-17 13:19:52 -0400', count: 20 },
  )
  assert.deepEqual(
    matchSwimStrokeOpen(
      '<Record type="HKQuantityTypeIdentifierSwimmingStrokeCount" startDate="2026-05-17 13:19:25 -0400" endDate="2026-05-17 13:19:52 -0400" value="20"/>',
    ),
    { start: '2026-05-17 13:19:25 -0400', end: '2026-05-17 13:19:52 -0400', count: 20 },
  )
  assert.equal(
    matchSwimStrokeOpen('<Record type="HKQuantityTypeIdentifierStepCount" value="20">'),
    null,
  )
})

test('matchSwimWorkout captures a stable XML session interval', () => {
  assert.deepEqual(
    matchSwimWorkout(
      '<Workout workoutActivityType="HKWorkoutActivityTypeSwimming" duration="30" durationUnit="min" totalDistance="1500" totalDistanceUnit="m" startDate="2026-05-17 13:00:00 -0400" endDate="2026-05-17 14:00:00 -0400">',
    ),
    {
      id: 'xml:2026-05-17 13:00:00 -0400|2026-05-17 14:00:00 -0400',
      start: '2026-05-17 13:00:00 -0400',
      end: '2026-05-17 14:00:00 -0400',
      totalM: 1500,
      activeTimeS: 1800,
    },
  )
  assert.equal(
    matchSwimWorkout(
      '<Workout workoutActivityType="HKWorkoutActivityTypeRunning" startDate="2026-05-17 13:00:00 -0400" endDate="2026-05-17 14:00:00 -0400">',
    ),
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

test('aggregateSwimLaps keeps sessions separate and unions overlapping rate intervals', () => {
  const laps: AppleSwimLap[] = [
    {
      start: '2026-05-17 13:19:25 -0400',
      end: '2026-05-17 13:19:52 -0400',
      count: 20,
      stroke: 'freestyle',
    },
    {
      start: '2026-05-17 13:19:25 -0400',
      end: '2026-05-17 13:19:52 -0400',
      count: 18,
      stroke: 'freestyle',
    },
    {
      start: '2026-05-17 13:19:40 -0400',
      end: '2026-05-17 13:20:00 -0400',
      count: 10,
      stroke: null,
    },
    {
      start: '2026-05-17 13:20:10 -0400',
      end: '2026-05-17 13:20:40 -0400',
      count: 0,
      stroke: 'kickboard',
    },
    {
      start: '2026-05-17 18:10:00 -0400',
      end: '2026-05-17 18:10:30 -0400',
      count: 16,
      stroke: 'breaststroke',
    },
  ]
  const distances: AppleSwimDistanceRecord[] = [
    { start: '2026-05-17 13:19:25 -0400', end: '2026-05-17 13:19:52 -0400', meters: 25 },
    { start: '2026-05-17 18:10:00 -0400', end: '2026-05-17 18:10:30 -0400', meters: 50 },
  ]
  const workouts = [
    {
      id: 'morning',
      start: '2026-05-17 13:00:00 -0400',
      end: '2026-05-17 14:00:00 -0400',
      totalM: 1500,
      activeTimeS: 1800,
    },
    {
      id: 'evening',
      start: '2026-05-17 18:00:00 -0400',
      end: '2026-05-17 19:00:00 -0400',
      totalM: null,
      activeTimeS: null,
    },
  ]
  assert.deepEqual(aggregateSwimLaps(laps, distances, workouts), [
    {
      id: 'morning',
      date: '2026-05-17',
      start: '2026-05-17T17:00:00Z',
      end: '2026-05-17T18:00:00Z',
      laps: 1,
      totalM: 1500,
      activeTimeS: 1800,
      strokeCount: 24,
      strokeTimeS: 35,
      strokes: { freestyle: 25 },
      intervals: [
        {
          start: '2026-05-17T17:19:25Z',
          end: '2026-05-17T17:19:52Z',
          distanceM: 25,
          startElapsedS: 1165,
          endElapsedS: 1192,
          durationS: 27,
          strokeCount: 20,
          strokeTimeS: 27,
          stroke: 'freestyle',
        },
      ],
    },
    {
      id: 'evening',
      date: '2026-05-17',
      start: '2026-05-17T22:00:00Z',
      end: '2026-05-17T23:00:00Z',
      laps: 1,
      totalM: 50,
      activeTimeS: 30,
      strokeCount: 16,
      strokeTimeS: 30,
      strokes: { breaststroke: 50 },
      intervals: [
        {
          start: '2026-05-17T22:10:00Z',
          end: '2026-05-17T22:10:30Z',
          distanceM: 50,
          startElapsedS: 600,
          endElapsedS: 630,
          durationS: 30,
          strokeCount: 16,
          strokeTimeS: 30,
          stroke: 'breaststroke',
        },
      ],
    },
  ])
})

test('aggregateSwimLaps prorates partial stroke coverage into interval count and time', () => {
  const swims = aggregateSwimLaps(
    [
      {
        start: '2026-05-17 13:00:00 -0400',
        end: '2026-05-17 13:00:20 -0400',
        count: 12,
        stroke: 'freestyle',
      },
      {
        start: '2026-05-17 13:00:30 -0400',
        end: '2026-05-17 13:00:50 -0400',
        count: 10,
        stroke: 'freestyle',
      },
    ],
    [{ start: '2026-05-17 13:00:10 -0400', end: '2026-05-17 13:00:40 -0400', meters: 25 }],
    [
      {
        id: 'partial',
        start: '2026-05-17 13:00:00 -0400',
        end: '2026-05-17 13:01:00 -0400',
        totalM: 25,
        activeTimeS: 30,
      },
    ],
  )

  assert.deepEqual(swims[0]?.intervals, [
    {
      start: '2026-05-17T17:00:10Z',
      end: '2026-05-17T17:00:40Z',
      distanceM: 25,
      startElapsedS: 10,
      endElapsedS: 40,
      durationS: 30,
      strokeCount: 11,
      strokeTimeS: 20,
      stroke: 'freestyle',
    },
  ])
})

test('aggregateSwimLaps preserves distance-only sessions without inventing distance from strokes', () => {
  const strokeOnly: AppleSwimLap = {
    start: '2026-05-17 22:10:00 -0400',
    end: '2026-05-17 22:10:30 -0400',
    count: 20,
    stroke: 'freestyle',
  }
  const distances: AppleSwimDistanceRecord[] = [
    { start: '2026-05-17 18:10:00 -0400', end: '2026-05-17 18:10:20 -0400', meters: 25 },
    { start: '2026-05-17 18:10:20 -0400', end: '2026-05-17 18:10:40 -0400', meters: 25 },
  ]
  const workouts = [
    {
      id: 'morning',
      start: '2026-05-17 13:00:00 -0400',
      end: '2026-05-17 14:00:00 -0400',
      totalM: 1000,
      activeTimeS: 1200,
    },
    {
      id: 'evening',
      start: '2026-05-17 18:00:00 -0400',
      end: '2026-05-17 19:00:00 -0400',
      totalM: null,
      activeTimeS: null,
    },
    {
      id: 'stroke-only',
      start: '2026-05-17 22:00:00 -0400',
      end: '2026-05-17 23:00:00 -0400',
      totalM: null,
      activeTimeS: null,
    },
  ]

  assert.deepEqual(aggregateSwimLaps([strokeOnly], distances, workouts), [
    {
      id: 'morning',
      date: '2026-05-17',
      start: '2026-05-17T17:00:00Z',
      end: '2026-05-17T18:00:00Z',
      laps: 0,
      totalM: 1000,
      activeTimeS: 1200,
      strokeCount: null,
      strokeTimeS: null,
      strokes: {},
      intervals: [],
    },
    {
      id: 'evening',
      date: '2026-05-17',
      start: '2026-05-17T22:00:00Z',
      end: '2026-05-17T23:00:00Z',
      laps: 2,
      totalM: 50,
      activeTimeS: 40,
      strokeCount: null,
      strokeTimeS: null,
      strokes: {},
      intervals: [
        {
          start: '2026-05-17T22:10:00Z',
          end: '2026-05-17T22:10:20Z',
          distanceM: 25,
          startElapsedS: 600,
          endElapsedS: 620,
          durationS: 20,
          strokeCount: null,
          strokeTimeS: null,
          stroke: null,
        },
        {
          start: '2026-05-17T22:10:20Z',
          end: '2026-05-17T22:10:40Z',
          distanceM: 25,
          startElapsedS: 620,
          endElapsedS: 640,
          durationS: 20,
          strokeCount: null,
          strokeTimeS: null,
          stroke: null,
        },
      ],
    },
  ])
  assert.deepEqual(aggregateSwimLaps([strokeOnly], [], []), [])
})
