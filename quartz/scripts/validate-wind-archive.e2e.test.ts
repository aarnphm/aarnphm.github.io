import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import test from 'node:test'
import vm from 'node:vm'

const tsx = resolve('node_modules/tsx/dist/cli.mjs')
const validateScript = resolve('quartz/scripts/validate-wind-archive.ts')
const ingestScript = resolve('quartz/scripts/ingest-mywindsock.ts')
const captureScriptPath = resolve('quartz/scripts/mywindsock-browser-capture.js')
const realWindDir = resolve('content/triathlon/wind')

function runTsx(
  script: string,
  args: string[] = [],
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((done, reject) => {
    const child = spawn(process.execPath, [tsx, script, ...args], {
      cwd: resolve('.'),
      env: process.env,
    })
    let stdout = ''
    let stderr = ''
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', chunk => {
      stdout += chunk
    })
    child.stderr.on('data', chunk => {
      stderr += chunk
    })
    child.on('error', reject)
    child.on('close', code => done({ code, stdout, stderr }))
  })
}

function evaluateBrowserCapture(source: string, id: string, pageStravaId: unknown): string {
  const expression = source
    .replace('__MYWINDSOCK_CHARTS__', JSON.stringify({ cda: [], gradient: [] }))
    .replaceAll('__MYWINDSOCK_OUTPUT__', JSON.stringify('return'))
  return vm.runInNewContext(expression, {
    Date,
    Error,
    JSON,
    Number,
    Object,
    String,
    Array,
    location: { pathname: `/activity/${id}/` },
    document: {
      querySelectorAll: (selector: string) =>
        selector === 'a'
          ? [{ textContent: 'View on Strava', href: `https://www.strava.com/activities/${id}` }]
          : [],
      getElementById: () => ({ innerText: 'summary', textContent: 'nav' }),
    },
    type: 'activity',
    mode: 'analyst',
    courseID: 80642545,
    stravaID: pageStravaId,
    ride_time: Date.parse('2026-11-16T12:00:00Z') / 1000,
    data: {
      time: [0, 1 / 3600, 2 / 3600],
      point_time: [0, 1 / 3600, 1 / 3600],
      dist_acc: [0, 0.004, 0.008],
      groundspd: [14.4, 14.4, 14.4],
      adj_wind: [1.5, 1.5, 1.5],
      direction360: [90, 90, 90],
    },
    averages: { power: 0 },
    course: [
      [43.65, -79.46],
      [43.6501, -79.46],
      [43.6502, -79.46],
    ],
    distances: [0, 4, 8],
    elev: [100, 100, 100],
    forecast: [
      {
        latitude: 43.65,
        longitude: -79.46,
        timezone: 'America/Toronto',
        offset: -5,
        side_of_road: null,
        hourly: {},
        daily: {},
        currently: {},
      },
      {
        latitude: 43.6502,
        longitude: -79.46,
        timezone: 'America/Toronto',
        offset: -5,
        side_of_road: 'right',
        hourly: {},
        daily: {},
        currently: {},
      },
    ],
    profiles: [],
  }) as string
}

test('wind validation and ingestion preserve historical providerStravaId=0, string "0" captures, null side_of_road, and reject unsafe numeric stravaID', async () => {
  const root = await mkdtemp(join(tmpdir(), 'wind-archive-e2e-'))
  const windDir = join(root, 'wind')
  const cachePath = join(root, 'strava.json')
  const statePath = join(root, 'mywindsock-ingestion.json')
  const capturePath = join(root, 'capture.json')
  const exactId = '9007199254740993'
  const legacyId = '20441001789'
  const summaryId = '21028944948'
  const evidence: Record<string, unknown> = {}

  try {
    await mkdir(windDir, { recursive: true })
    await copyFile(join(realWindDir, `${legacyId}.json`), join(windDir, `${legacyId}.json`))
    await copyFile(join(realWindDir, `${summaryId}.json`), join(windDir, `${summaryId}.json`))
    const legacyBefore = await readFile(join(windDir, `${legacyId}.json`), 'utf8')

    // 1. Existing committed archive with checksum-covered numeric providerStravaId=0 passes deploy validator.
    const initialValidate = await runTsx(validateScript, [windDir])
    evidence.initialValidate = initialValidate
    assert.equal(initialValidate.code, 0, initialValidate.stderr || initialValidate.stdout)
    assert.match(initialValidate.stdout, /2\/2 records passed structural and capture checks/)

    // 2. Browser capture script preserves string "0" + null side_of_road and refuses unsafe numeric stravaID.
    const captureScript = await readFile(captureScriptPath, 'utf8')
    assert.throws(
      () => evaluateBrowserCapture(captureScript, exactId, Number(exactId)),
      /stravaID/i,
    )
    const capturedJson = evaluateBrowserCapture(captureScript, exactId, 0)
    await writeFile(capturePath, capturedJson)
    const parsedCapture = JSON.parse(capturedJson)
    assert.equal(parsedCapture.providerStravaId, '0')
    assert.equal(parsedCapture.weather[0].side_of_road, null)

    // 3. CLI import accepts string "0" + null side_of_road, then CLI queue and deploy validator succeed across all three records.
    await writeFile(
      cachePath,
      JSON.stringify({
        version: 8,
        athleteId: '9007199254740992',
        auth: { refreshToken: '', obtainedAt: 0 },
        lastSync: Date.parse('2026-11-18T12:00:00Z'),
        lastActivityStart: 0,
        activities: {
          [legacyId]: {
            id: legacyId,
            name: 'Tour de Simcoe',
            sportType: 'Ride',
            trainer: false,
            startDate: '2026-10-03T10:57:45Z',
            startDateLocal: '2026-10-03T06:57:45Z',
            distance: 335304,
            movingTime: 44588,
            elapsedTime: 53735,
            totalElevationGain: 1784,
          },
          [exactId]: {
            id: exactId,
            name: 'Future exact run',
            sportType: 'Run',
            trainer: false,
            startDate: '2026-11-16T12:00:00Z',
            startDateLocal: '2026-11-16T07:00:00Z',
            distance: 8,
            movingTime: 2,
            elapsedTime: 2,
            totalElevationGain: 0,
          },
          [summaryId]: {
            id: summaryId,
            name: 'Evening run',
            sportType: 'Run',
            trainer: false,
            startDate: '2026-11-18T00:28:41Z',
            startDateLocal: '2026-11-17T19:28:41Z',
            distance: 7108.9,
            movingTime: 2092,
            elapsedTime: 2100,
            totalElevationGain: 31,
          },
        },
      }),
    )

    const imported = await runTsx(ingestScript, [
      'import',
      exactId,
      capturePath,
      '--cache',
      cachePath,
      '--directory',
      windDir,
      '--state',
      statePath,
      '--until',
      '2026-11-17',
    ])
    evidence.imported = imported
    assert.equal(imported.code, 0, imported.stderr || imported.stdout)

    const savedExact = JSON.parse(await readFile(join(windDir, `${exactId}.json`), 'utf8'))
    assert.equal(savedExact.browserCapture.providerStravaId, '0')
    assert.equal(savedExact.browserCapture.weather[0].side_of_road, null)

    const queued = await runTsx(ingestScript, [
      'queue',
      '--cache',
      cachePath,
      '--directory',
      windDir,
      '--state',
      statePath,
      '--until',
      '2026-11-17',
    ])
    evidence.queued = {
      code: queued.code,
      stderr: queued.stderr,
      summary: queued.code === 0 ? JSON.parse(queued.stdout) : queued.stdout,
    }
    assert.equal(queued.code, 0, queued.stderr || queued.stdout)
    assert.equal(await readFile(join(windDir, `${legacyId}.json`), 'utf8'), legacyBefore)

    const finalValidate = await runTsx(validateScript, [windDir])
    evidence.finalValidate = finalValidate
    assert.equal(finalValidate.code, 0, finalValidate.stderr || finalValidate.stdout)
    assert.match(finalValidate.stdout, /3\/3 records passed structural and capture checks/)

    // 4. Full repository wind archive passes the deploy stage check.
    const repoValidate = await runTsx(validateScript, [realWindDir])
    evidence.repoValidate = {
      code: repoValidate.code,
      stderr: repoValidate.stderr,
      summaryLine: repoValidate.stdout.trim().split('\n').at(-1),
    }
    assert.equal(repoValidate.code, 0, repoValidate.stderr || repoValidate.stdout)
    assert.match(
      repoValidate.stdout,
      /^([1-9]\d*)\/\1 records passed structural and capture checks$/m,
    )
  } finally {
    const artifactDir = process.env.WIND_ARCHIVE_TEST_ARTIFACTS
    if (artifactDir) {
      await mkdir(artifactDir, { recursive: true })
      await writeFile(
        join(artifactDir, 'wind-archive-e2e.json'),
        JSON.stringify(
          {
            command:
              'WIND_ARCHIVE_TEST_ARTIFACTS=docs/verification/wind-archive-validation/artifacts ./node_modules/.bin/tsx --test quartz/scripts/validate-wind-archive.e2e.test.ts',
            inputs: { legacyId, summaryId, exactId },
            results: evidence,
          },
          null,
          2,
        ) + '\n',
      )
    }
    await rm(root, { recursive: true, force: true })
  }
})
