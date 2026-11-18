import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { isActivityDetail } from '../components/triathlon/activity/data'
import { buildAnalytics, buildDataFeed } from '../plugins/stores/analytics'
import {
  buildPayload,
  applyManualFueling,
  applyManualMoves,
  applyManualSauna,
} from '../plugins/stores/strava'
import { parseTrackingBlock } from '../plugins/stores/tracking'
import { mapActivity } from '../scripts/sync-strava'
import { readActivityProvenance } from './activity-provenance'
import {
  myWindsockArchiveSchema,
  validateMyWindsockArchive,
  myWindsockCaptureSha256,
  type MyWindsockArchive,
} from './mywindsock-archive'
import { parseMyWindsockGraphs } from './mywindsock-graphs'
import { projectMyWindsockRoute } from './mywindsock-route'
import { encodePowerCurves, decodePowerCurves, powerCurvePointAt } from './power-curve'
import { readStravaCacheFile } from './strava-cache-file'
import {
  serializeStravaMapPayload,
  hydrateStravaMapPayload,
  serializeStravaDetails,
} from './strava-detail'
import { latestStravaActivities } from './strava-latest-activities'
import { linkTrainingPeaksCalendarActivities } from './trainingpeaks-activity-links'
import { parseTrainingPeaksCalendar, type TrainingPeaksCalendar } from './trainingpeaks-calendar'
import { buildTriathlonPreview, readTriathlonPreview } from './triathlon-preview'
import { routeWeatherFingerprint } from './weather-route-hours'

const ids = ['9007199254740992', '9007199254740993']
const date = '2026-11-16'

test('exact IDs survive cache ingestion, payload and map serialization, tracking, analytics and planned links', async () => {
  const root = await mkdtemp(join(tmpdir(), 'strava-pipeline-'))
  try {
    const cachePath = join(root, 'strava.json')
    const activities = ids.map((id, index) =>
      mapActivity({
        id: Number(id),
        id_str: id,
        name: `ride ${index}`,
        sport_type: 'Ride',
        distance: 5000,
        moving_time: 600,
        elapsed_time: 600,
        start_date: `${date}T12:00:00Z`,
        average_speed: 8.33,
        average_watts: 200,
        device_watts: true,
      }),
    )
    await writeFile(
      cachePath,
      JSON.stringify({
        version: 8,
        athleteId: ids[1],
        auth: { refreshToken: '', obtainedAt: 0 },
        lastSync: Date.parse(`${date}T23:00:00Z`),
        lastActivityStart: 0,
        activities: Object.fromEntries(activities.map(a => [a.id, a])),
        streams: Object.fromEntries(
          ids.map(id => [
            id,
            {
              time: [0, 1, 2],
              latlng: [
                [43, -79],
                [43.001, -79],
                [43.002, -79],
              ],
              distance: [0, 8, 16],
              altitude: [100, 100, 100],
              watts: [200, 250, 300],
              heartrate: [140, 140, 141],
            },
          ]),
        ),
      }),
    )
    const cache = await readStravaCacheFile(cachePath)
    assert.ok(cache)
    const payload = buildPayload(cache, null, null)
    assert.deepEqual(Object.keys(payload.details), ids)
    for (const id of ids) {
      assert.equal(payload.details[id].id, id)
      assert.equal(isActivityDetail(JSON.parse(JSON.stringify(payload.details[id]))), true)
    }
    const preview = buildTriathlonPreview({
      date,
      details: Object.values(payload.details),
      activities: new Map(),
      daily: null,
      event: null,
    })
    assert.deepEqual(
      readTriathlonPreview(JSON.parse(JSON.stringify(preview)), date)?.activities.map(a => a.id),
      ids,
    )
    const hashRoute = { timeS: [0], latlng: [[43, -79]] as [number, number][] }
    const oldHash = createHash('sha256')
      .update(JSON.stringify({ activityId: 42, start: date, end: date, route: [[0, 43, -79]] }))
      .digest('hex')
    assert.equal(routeWeatherFingerprint('42', date, date, hashRoute), oldHash)
    assert.notEqual(
      routeWeatherFingerprint(ids[0], date, date, hashRoute),
      routeWeatherFingerprint(ids[1], date, date, hashRoute),
    )
    const tracking = parseTrackingBlock(
      null,
      `date: ${date}\nactivity: ${ids[1]}\nfueling: 250\nmove: push-up | 10 reps\nskiptraining: true`,
    )
    assert.ok(tracking?.fueling)
    applyManualFueling(payload, [tracking.fueling])
    assert.ok(tracking.moves)
    applyManualMoves(payload, [tracking.moves])
    assert.equal(payload.details[ids[1]].moves?.entries[0].name, 'push-up')
    assert.equal(payload.details[ids[0]].moves, undefined)
    assert.equal(payload.details[ids[1]].fueling?.caloriesConsumed, 250)
    assert.notEqual(payload.details[ids[0]].fueling?.caloriesConsumed, 250)
    const legacyCache = { ...cache, activities: { '42': { ...activities[0], id: '42' } } }
    const legacyTracking = JSON.parse(
      '[{"activityId":42,"garminActivityId":null,"virtual":false,"computer":"fixture computer"}]',
    )
    const legacyPayload = buildPayload(
      legacyCache,
      null,
      null,
      undefined,
      null,
      null,
      undefined,
      undefined,
      null,
      undefined,
      undefined,
      undefined,
      legacyTracking,
    )
    assert.equal(legacyPayload.details['42'].computerOverride, 'fixture computer')
    const map = hydrateStravaMapPayload(JSON.parse(serializeStravaMapPayload(payload.details)))
    assert.deepEqual(Object.keys(map!.details), ids)
    const serialized = serializeStravaDetails({ details: payload.details, health: {} })
    assert.ok(JSON.stringify(serialized).includes(ids[1]))
    const curve = ids.map((activityId, i) => ({
      s: i + 1,
      w: 250 - i,
      activityId,
      activityDate: date,
    }))
    const decoded = decodePowerCurves(encodePowerCurves([curve]))[0]
    assert.ok(decoded)
    assert.deepEqual(
      ids.map((_, i) => powerCurvePointAt(decoded, i).activityId),
      ids,
    )
    const analytics = buildAnalytics(cache, { activityDetails: payload.details })
    assert.deepEqual(analytics.activities.map(a => a.id).sort(), ids)
    assert.ok(
      buildDataFeed(cache, analytics, { activityDetails: payload.details }).includes(ids[1]),
    )
    assert.deepEqual(
      latestStravaActivities(cachePath)
        ?.activities.map(a => a.id)
        .sort(),
      ids,
    )
    assert.ok(tracking.trainingExclusion)
    const feed = buildDataFeed(cache, analytics, {
      activityDetails: payload.details,
      trainingExclusions: [tracking.trainingExclusion],
    })
      .trim()
      .split('\n')
      .map(line => JSON.parse(line))
    assert.equal(feed.find(row => row.kind === 'activity' && row.id === ids[1]).skipTraining, true)
    assert.equal(feed.find(row => row.kind === 'activity' && row.id === ids[0]).skipTraining, false)
    const calendar: TrainingPeaksCalendar = {
      version: 1,
      source: 'trainingpeaks',
      fetchedAt: `${date}T23:00:00.000Z`,
      coverage: [{ since: date, until: date, fetchedAt: `${date}T23:00:00.000Z` }],
      workouts: [
        {
          id: '42',
          source: 'trainingpeaks',
          date,
          title: 'plan',
          sport: 'bike',
          workoutTypeId: 2,
          description: `https://www.strava.com/activities/${ids[1]}`,
          planned: { durationSeconds: 600, distanceMeters: 5000, tss: null },
          actual: { durationSeconds: 600, distanceMeters: 5000, tss: null },
          status: 'completed',
          startTime: '12:00:00',
          startTimePlanned: null,
          order: null,
        },
      ],
    }
    const linked = linkTrainingPeaksCalendarActivities(
      calendar,
      activities.map(a => ({ ...a, sport: 'bike' })),
    )
    assert.equal(linked.workouts[0].activity?.id, ids[1])
    assert.equal(
      parseTrainingPeaksCalendar(JSON.parse(JSON.stringify(linked)))?.workouts[0].activity?.id,
      ids[1],
    )
    const legacyCalendar = JSON.parse(JSON.stringify(linked))
    legacyCalendar.workouts[0].activity.id = 42
    assert.equal(parseTrainingPeaksCalendar(legacyCalendar)?.workouts[0].activity?.id, '42')
    legacyCalendar.workouts[0].activity.id = Number(ids[0])
    assert.equal(parseTrainingPeaksCalendar(legacyCalendar), null)
    const sauna = parseTrackingBlock(
      null,
      `date: ${date}\nactivity: sauna\ntime: 18:30\nduration: 30m\ntemperature: 80C\nhumidity: 10%\ncooldown: natural`,
    )?.sauna
    assert.ok(sauna)
    applyManualSauna(payload, [sauna], [])
    assert.equal(payload.details[sauna.id].id, sauna.id)
    assert.equal(payload.details[sauna.id].sport, 'sauna')
    await mkdir(join(root, 'triathlon'))
    await writeFile(
      join(root, 'triathlon/provenance.json'),
      JSON.stringify([
        {
          activityId: 42,
          source: 'athlete-estimate',
          note: 'legacy',
          sport: 'bike',
          startDate: `${date}T12:00:00Z`,
          trainingPeaksWorkoutId: 123,
        },
        {
          activityId: ids[1],
          source: 'athlete-estimate',
          note: 'exact',
          sport: 'bike',
          startDate: `${date}T12:00:00Z`,
        },
      ]),
    )
    assert.deepEqual(
      readActivityProvenance(root).map(a => a.activityId),
      ['42', ids[1]],
    )
    for (const id of ids) {
      const startedAt = `${date}T12:00:00.000Z`
      const capturedAt = `${date}T23:00:00.000Z`
      const capture: NonNullable<MyWindsockArchive['browserCapture']> = {
        adapterVersion: 'mywindsock-classic-browser-v1',
        transport: 'browser-runtime',
        valueKind: 'provider-analysis',
        conditionKind: 'historical',
        capturedAt,
        pageUrl: `https://mywindsock.com/activity/${id}/`,
        stravaId: id,
        viewOnStravaId: id,
        providerCourseId: '1',
        providerStravaId: id,
        rideTimestamp: Date.parse(startedAt) / 1000,
        pageType: 'activity',
        pageMode: 'analyst',
        runtime: {
          averages: {},
          series: {
            time: [0, 1 / 3600, 2 / 3600],
            point_time: [0, 1 / 3600, 1 / 3600],
            dist_acc: [0, 0.008, 0.016],
            groundspd: [30, 30, 30],
            adj_wind: [2, 2, 2],
            direction360: [0, 0, 0],
          },
          course: [],
          distancesM: [],
          elevationM: [],
        },
        weather: [],
        modelProfileCandidates: [],
        charts: { cda: [], gradient: [] },
        excludedFields: [],
        nonFiniteValues: [],
      }
      const fixture = {
        schemaVersion: 1,
        provider: 'mywindsock',
        activity: {
          stravaId: id,
          sport: 'bike',
          stravaSportType: 'Ride',
          startedAt,
          localDate: date,
          stravaBaseline: { distanceM: 16, movingTimeS: 2, elapsedTimeS: 2, elevationGainM: 0 },
        },
        entrypoints: {
          stravaUrl: `https://www.strava.com/activities/${id}`,
          myWindsockUrl: capture.pageUrl,
          providerCourseId: '1',
        },
        ingestion: {
          state: 'captured',
          pending: [],
          sections: {},
          attempts: [],
          events: [],
          captureSha256: myWindsockCaptureSha256(capture),
        },
        description: null,
        browserCapture: capture,
        mappings: Object.keys(capture.runtime.series).map(key => ({
          path: `/runtime/series/${key}`,
          meaning: 'fixture',
          unit: null,
          multiplier: null,
          status: 'verified',
          evidence: 'synthetic provider replay',
        })),
        normalized: null,
        context: { notes: [], equipment: null, position: null },
      }
      const validated = validateMyWindsockArchive(JSON.parse(JSON.stringify(fixture)), id)
      assert.equal(projectMyWindsockRoute(validated)?.activityId, id)
      assert.equal(
        myWindsockArchiveSchema.safeParse({
          ...fixture,
          browserCapture: { ...capture, providerStravaId: Number(id) },
        }).success,
        false,
      )
    }
    const archive = {
      provider: 'mywindsock',
      schemaVersion: 1,
      activity: { stravaId: ids[1], sport: 'bike' },
      graphCapture: {
        stravaId: ids[1],
        viewOnStravaId: ids[1],
        capturedAt: `${date}T23:00:00Z`,
        pageUrl: `https://mywindsock.com/activity/${ids[1]}/`,
        graphs: { power: { label: 'Power', state: 'empty', configuration: {} } },
      },
    }
    assert.equal(parseMyWindsockGraphs(archive, ids[1])?.activityId, ids[1])
    assert.equal(parseMyWindsockGraphs(archive, ids[0]), null)
    const artifact = process.env.STRAVA_ID_TEST_ARTIFACTS
    if (artifact) {
      await mkdir(artifact, { recursive: true })
      await writeFile(
        join(artifact, 'pipeline-replay.json'),
        JSON.stringify(
          {
            cache: JSON.parse(await readFile(cachePath, 'utf8')),
            map,
            activityIds: analytics.activities.map(a => a.id),
            planned: linked.workouts[0].activity,
            manualId: sauna.id,
          },
          null,
          2,
        ),
      )
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
