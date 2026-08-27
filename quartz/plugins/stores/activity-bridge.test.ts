import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activityBridgeCreatedDestinations,
  activityBridgeReceiptKey,
  emptyActivityBridgeLedger,
  planActivityBridge,
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
    sport: 'bike',
    startDate: '2026-08-27T12:01:00.000Z',
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
    workoutId: Number(id.replace(/\D/g, '')),
    sport: 'bike',
    startDate: '2026-08-27T12:01:00.000Z',
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

test('plans only the missing provider side from canonical Strava cycling evidence', () => {
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
      { strava: [strava('1')], garmin: [garmin('connect:1')], wahoo: [] },
      ledger,
    ).map(plan => plan.direction),
    ['garmin-to-wahoo'],
  )
})
