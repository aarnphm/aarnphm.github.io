import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activityBridgeCreatedDestinations,
  activityBridgeReceiptKey,
  emptyActivityBridgeLedger,
  planActivityBridge,
  planTrainingPeaksBackfill,
  upsertActivityBridgeReceipt,
  type ActivityBridgeGarminActivity,
  type ActivityBridgeReceipt,
  type ActivityBridgeStravaActivity,
  type ActivityBridgeWahooActivity,
} from './activity-bridge'

const SHA = 'a'.repeat(64)

function strava(
  id: string,
  overrides: Partial<ActivityBridgeStravaActivity> = {},
): ActivityBridgeStravaActivity {
  return {
    id,
    name: `Ride ${id}`,
    sportType: 'Ride',
    startDate: '2026-08-27T12:00:00.000Z',
    startDateLocal: '2026-08-27T08:00:00',
    distanceM: 40_000,
    movingTimeS: 5_000,
    elapsedTimeS: 5_200,
    ...overrides,
  }
}

function garmin(
  id: string,
  overrides: Partial<ActivityBridgeGarminActivity> = {},
): ActivityBridgeGarminActivity {
  return {
    id,
    name: `Garmin ${id}`,
    sport: 'bike',
    startDate: '2026-08-27T12:01:00.000Z',
    startDateLocal: '2026-08-27T08:01:00',
    distanceM: 40_100,
    movingTimeS: 5_010,
    elapsedTimeS: 5_210,
    ...overrides,
  }
}

function wahoo(
  id: string,
  overrides: Partial<ActivityBridgeWahooActivity> = {},
): ActivityBridgeWahooActivity {
  return {
    id,
    name: `Wahoo ${id}`,
    workoutId: Number(id.replace(/\D/g, '')),
    sport: 'bike',
    startDate: '2026-08-27T12:01:00.000Z',
    startDateLocal: '2026-08-27T08:01:00',
    distanceM: 40_100,
    movingTimeS: 5_010,
    elapsedTimeS: 5_210,
    fitUrl: `https://cdn.wahooligan.com/${id}.fit`,
    fitSha256: SHA,
    ...overrides,
  }
}

function receipt(overrides: Partial<ActivityBridgeReceipt> = {}): ActivityBridgeReceipt {
  return {
    direction: 'wahoo-to-garmin',
    sourceProvider: 'wahoo',
    sourceActivityId: 'wahoo:1',
    sourceFitSha256: SHA,
    destinationProvider: 'garmin',
    destinationActivityId: 'connect:99',
    stravaActivityId: '1',
    uploadToken: null,
    uploadStatus: 'complete',
    createdAt: 100,
    updatedAt: 100,
    ...overrides,
  }
}

test('plans only the missing provider side from canonical Strava sport evidence', () => {
  assert.deepEqual(
    planActivityBridge(
      { strava: [strava('1')], garmin: [], wahoo: [wahoo('wahoo:1')] },
      emptyActivityBridgeLedger(),
    ),
    [
      {
        direction: 'wahoo-to-garmin',
        stravaActivityId: '1',
        title: 'Ride 1',
        source: wahoo('wahoo:1'),
      },
    ],
  )
  assert.deepEqual(
    planActivityBridge(
      { strava: [strava('1')], garmin: [garmin('connect:1')], wahoo: [] },
      emptyActivityBridgeLedger(),
    ).map(plan => plan.direction),
    ['garmin-to-wahoo'],
  )
  assert.deepEqual(
    planActivityBridge(
      { strava: [strava('1')], garmin: [garmin('connect:1')], wahoo: [wahoo('wahoo:1')] },
      emptyActivityBridgeLedger(),
    ),
    [],
  )
})

test('matches bike, run, and swim without crossing sports', () => {
  const activities = [
    strava('1'),
    strava('2', { sportType: 'Run', startDate: '2026-08-27T15:00:00.000Z' }),
    strava('3', { sportType: 'Swim', startDate: '2026-08-27T18:00:00.000Z' }),
    strava('4', { sportType: 'WeightTraining', startDate: '2026-08-27T21:00:00.000Z' }),
  ]
  const garminActivities = [
    garmin('connect:1'),
    garmin('connect:2', { sport: 'run', startDate: '2026-08-27T15:01:00.000Z' }),
    garmin('connect:3', { sport: 'swim', startDate: '2026-08-27T18:01:00.000Z' }),
    garmin('connect:4', { sport: 'bike', startDate: '2026-08-27T21:01:00.000Z' }),
  ]

  assert.deepEqual(
    planActivityBridge(
      { strava: activities, garmin: garminActivities, wahoo: [] },
      emptyActivityBridgeLedger(),
    ).map(plan => plan.source.id),
    ['connect:3', 'connect:2', 'connect:1'],
  )
})

test('prioritizes current Wahoo recordings before Garmin history', () => {
  const plans = planActivityBridge(
    {
      strava: [strava('1'), strava('2', { startDate: '2026-08-27T15:00:00.000Z' })],
      garmin: [garmin('connect:1')],
      wahoo: [wahoo('wahoo:2', { startDate: '2026-08-27T15:01:00.000Z' })],
    },
    emptyActivityBridgeLedger(),
  )
  assert.deepEqual(
    plans.map(plan => plan.direction),
    ['wahoo-to-garmin', 'garmin-to-wahoo'],
  )
})

test('rejects start-only candidates without distance and duration evidence', () => {
  const canonical = [strava('1')]
  assert.deepEqual(
    planActivityBridge(
      { strava: canonical, garmin: [], wahoo: [wahoo('wahoo:1', { distanceM: null })] },
      emptyActivityBridgeLedger(),
    ),
    [],
  )
  assert.deepEqual(
    planActivityBridge(
      {
        strava: canonical,
        garmin: [garmin('connect:1', { movingTimeS: null, elapsedTimeS: null })],
        wahoo: [],
      },
      emptyActivityBridgeLedger(),
    ),
    [],
  )
})

test('assigns each provider activity to at most one Strava activity', () => {
  const plans = planActivityBridge(
    {
      strava: [
        strava('1'),
        strava('2', { startDate: '2026-08-27T12:02:00.000Z', name: 'Second ride' }),
      ],
      garmin: [],
      wahoo: [wahoo('wahoo:1')],
    },
    emptyActivityBridgeLedger(),
  )
  assert.equal(plans.length, 1)
  assert.equal(plans[0].stravaActivityId, '1')
})

test('terminal receipts suppress retries and bridge-created destinations suppress reverse exports', () => {
  const ledger = upsertActivityBridgeReceipt(emptyActivityBridgeLedger(), receipt())
  assert.equal(
    activityBridgeReceiptKey('wahoo', 'wahoo:1', SHA, 'garmin'),
    `wahoo:wahoo%3A1:${SHA}:garmin`,
  )
  assert.deepEqual([...activityBridgeCreatedDestinations(ledger, 'garmin')], ['connect:99'])
  assert.deepEqual(
    planActivityBridge({ strava: [strava('1')], garmin: [], wahoo: [wahoo('wahoo:1')] }, ledger),
    [],
  )
  assert.deepEqual(
    planActivityBridge(
      { strava: [strava('1')], garmin: [garmin('connect:99')], wahoo: [] },
      ledger,
    ),
    [],
  )
})

test('nonterminal Wahoo upload receipts remain resumable', () => {
  const ledger = upsertActivityBridgeReceipt(
    emptyActivityBridgeLedger(),
    receipt({
      direction: 'garmin-to-wahoo',
      sourceProvider: 'garmin',
      sourceActivityId: 'connect:1',
      destinationProvider: 'wahoo',
      destinationActivityId: null,
      uploadToken: 'pending-token',
      uploadStatus: 'pending',
    }),
  )
  assert.deepEqual(
    planActivityBridge(
      {
        strava: [strava('1'), strava('2', { startDate: '2026-08-27T15:00:00.000Z' })],
        garmin: [
          garmin('connect:1'),
          garmin('connect:2', { startDate: '2026-08-27T15:01:00.000Z' }),
        ],
        wahoo: [],
      },
      ledger,
    ).map(plan => plan.source.id),
    ['connect:1', 'connect:2'],
  )
})

test('plans direct Garmin TrainingPeaks exports without requiring a Strava match', () => {
  const plans = planTrainingPeaksBackfill(
    {
      strava: [],
      garmin: [
        garmin('connect:1'),
        garmin('connect:2', {
          name: 'Evening run',
          sport: 'run',
          startDate: '2026-08-28T02:01:00.000Z',
          startDateLocal: '2026-08-27T22:01:00',
        }),
        garmin('connect:3', { sport: null }),
      ],
      wahoo: [],
    },
    emptyActivityBridgeLedger(),
    'garmin',
  )

  assert.deepEqual(
    plans.map(plan => ({
      sourceProvider: plan.sourceProvider,
      sourceActivityId: plan.source.id,
      localDate: plan.localDate,
      title: plan.title,
    })),
    [
      {
        sourceProvider: 'garmin',
        sourceActivityId: 'connect:2',
        localDate: '2026-08-27',
        title: 'Evening run',
      },
      {
        sourceProvider: 'garmin',
        sourceActivityId: 'connect:1',
        localDate: '2026-08-27',
        title: 'Garmin connect:1',
      },
    ],
  )
})

test('plans direct Strava TrainingPeaks exports for triathlon sports', () => {
  const plans = planTrainingPeaksBackfill(
    {
      strava: [
        strava('1'),
        strava('2', { sportType: 'TrailRun', startDateLocal: '2026-08-28T08:00:00' }),
        strava('3', { sportType: 'WeightTraining' }),
      ],
      garmin: [garmin('connect:1')],
      wahoo: [wahoo('wahoo:1')],
    },
    emptyActivityBridgeLedger(),
    'strava',
  )

  assert.deepEqual(
    plans.map(plan => ({ id: plan.source.id, sport: plan.sport, localDate: plan.localDate })),
    [
      { id: '2', sport: 'run', localDate: '2026-08-28' },
      { id: '1', sport: 'bike', localDate: '2026-08-27' },
    ],
  )
})

test('plans direct Wahoo TrainingPeaks exports from original FIT activities', () => {
  const plans = planTrainingPeaksBackfill(
    {
      strava: [],
      garmin: [],
      wahoo: [
        wahoo('wahoo:1'),
        wahoo('wahoo:2', {
          name: 'Evening run',
          sport: 'run',
          startDate: '2026-08-28T02:01:00.000Z',
          startDateLocal: '2026-08-27T22:01:00',
        }),
        wahoo('wahoo:3', { sport: null }),
      ],
    },
    emptyActivityBridgeLedger(),
    'wahoo',
  )

  assert.deepEqual(
    plans.map(plan => ({
      sourceProvider: plan.sourceProvider,
      sourceActivityId: plan.source.id,
      localDate: plan.localDate,
      title: plan.title,
    })),
    [
      {
        sourceProvider: 'wahoo',
        sourceActivityId: 'wahoo:2',
        localDate: '2026-08-27',
        title: 'Evening run',
      },
      {
        sourceProvider: 'wahoo',
        sourceActivityId: 'wahoo:1',
        localDate: '2026-08-27',
        title: 'Wahoo wahoo:1',
      },
    ],
  )
})

test('excludes provider activities created by the bridge from TrainingPeaks exports', () => {
  const withGarminMirror = upsertActivityBridgeReceipt(
    emptyActivityBridgeLedger(),
    receipt({ sourceActivityId: 'wahoo:1', destinationActivityId: 'connect:1' }),
  )
  const ledger = upsertActivityBridgeReceipt(
    withGarminMirror,
    receipt({
      direction: 'garmin-to-wahoo',
      sourceProvider: 'garmin',
      sourceActivityId: 'connect:2',
      destinationProvider: 'wahoo',
      destinationActivityId: 'wahoo:2',
      stravaActivityId: '2',
    }),
  )
  const inputs = {
    strava: [],
    garmin: [garmin('connect:1'), garmin('connect:3')],
    wahoo: [wahoo('wahoo:2'), wahoo('wahoo:4')],
  }

  assert.deepEqual(
    planTrainingPeaksBackfill(inputs, ledger, 'garmin').map(plan => plan.source.id),
    ['connect:3'],
  )
  assert.deepEqual(
    planTrainingPeaksBackfill(inputs, ledger, 'wahoo').map(plan => plan.source.id),
    ['wahoo:4'],
  )
})
