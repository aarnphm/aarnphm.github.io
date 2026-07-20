import assert from 'node:assert/strict'
import test from 'node:test'
import { fitImportActivityId, fitImportPollDelay, fitImportPollLocation } from './garmin-fit-upload'

test('accepts only one successful Garmin internal activity ID', () => {
  assert.equal(
    fitImportActivityId({
      detailedImportResult: { successes: [{ internalId: 123 }], failures: [] },
    }),
    '123',
  )
  assert.equal(
    fitImportActivityId({
      data: { detailedImportResult: { successes: [{ internalId: 456 }], failures: [] } },
    }),
    '456',
  )
  assert.throws(
    () =>
      fitImportActivityId({
        detailedImportResult: { successes: [{ externalId: 123 }], failures: [] },
      }),
    /internalId/,
  )
  assert.throws(
    () =>
      fitImportActivityId({
        detailedImportResult: { successes: [{ internalId: 123 }], failures: [{ internalId: 456 }] },
      }),
    /reported a failure/,
  )
  assert.throws(
    () =>
      fitImportActivityId({
        detailedImportResult: {
          successes: [{ internalId: 123 }, { internalId: 456 }],
          failures: [],
        },
      }),
    /exactly one success/,
  )
})

test('keeps FIT import polling on Garmin with bounded delays', () => {
  const request = 'https://connect.garmin.com/gc-api/upload-service/upload/.fit'
  assert.equal(
    fitImportPollLocation(request, '/gc-api/upload-service/status/123'),
    'https://connect.garmin.com/gc-api/upload-service/status/123',
  )
  assert.equal(
    fitImportPollLocation(
      request,
      'https://connectapi.garmin.com/activity-service/activity/status/123/opaque',
    ),
    'https://connect.garmin.com/gc-api/activity-service/activity/status/123/opaque',
  )
  assert.throws(
    () => fitImportPollLocation(request, 'https://example.com/gc-api/status/123'),
    /unsafe poll location/,
  )
  assert.throws(() => fitImportPollLocation(request, null), /omitted Location/)
  assert.equal(fitImportPollDelay(null), 1000)
  assert.equal(fitImportPollDelay('2500'), 2500)
  assert.equal(fitImportPollDelay('60000'), 10000)
  assert.equal(fitImportPollDelay('invalid'), 1000)
})
