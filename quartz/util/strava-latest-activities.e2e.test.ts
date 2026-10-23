import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { promisify } from 'node:util'
import type { StravaRawCache } from '../plugins/stores/strava'

test('the homepage summary reads its default raw cache without initializing analytics', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'strava-latest-activities-'))
  const cacheDirectory = join(directory, 'quartz', '.quartz-cache')
  const syncedAt = Date.parse('2026-10-22T12:00:00Z')
  const cache: StravaRawCache = {
    athleteId: 7,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: syncedAt,
    lastActivityStart: 0,
    activities: {
      '1': {
        id: 1,
        name: 'Evening run',
        sportType: 'Run',
        distance: 5000,
        movingTime: 1800,
        elapsedTime: 1900,
        totalElevationGain: 20,
        startDate: '2026-10-22T01:00:00Z',
        startDateLocal: '2026-10-21T21:00:00',
        averageSpeed: 5000 / 1800,
      },
    },
  }

  try {
    await mkdir(cacheDirectory, { recursive: true })
    await writeFile(join(cacheDirectory, 'strava.json'), JSON.stringify(cache))
    const { stdout, stderr } = await promisify(execFile)(
      process.execPath,
      [
        '--import',
        import.meta.resolve('tsx'),
        '--input-type=module',
        '-e',
        `
          import { latestStravaActivities } from ${JSON.stringify(new URL('./strava-latest-activities.ts', import.meta.url).href)}
          console.log(JSON.stringify({
            summary: latestStravaActivities(),
            analyticsInitialized: Object.hasOwn(globalThis, '__quartzStravaPayloadMemo'),
          }))
        `,
      ],
      { cwd: directory, timeout: 30_000 },
    )
    assert.equal(stderr, '')
    const result: unknown = JSON.parse(stdout)
    assert.deepEqual(result, {
      summary: {
        athleteId: 7,
        syncedAt,
        activities: [
          {
            id: 1,
            kind: 'run',
            name: 'Evening run',
            date: '2026-10-21',
            distanceM: 5000,
            movingTimeS: 1800,
            elevationM: 20,
          },
        ],
        year: 2026,
        totals: { run: { count: 1, distanceM: 5000, movingTimeS: 1800 } },
      },
      analyticsInitialized: false,
    })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
