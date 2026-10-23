import { Decoder, Stream } from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { createServer } from 'node:http'
import { resolve } from 'node:path'
import test from 'node:test'
import type {
  ActivityBridgeGarminActivity,
  ActivityBridgeWahooActivity,
} from '../../plugins/stores/activity-bridge'
import {
  fetchGarminActivityFile,
  fetchGarminActivityFit,
  fetchWahooActivityFit,
} from '../activity-source-file'
import { WahooCloudClient } from '../wahoo-cloud'
import { fixtureArchive } from './fixture-archive'

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

const garminActivity: ActivityBridgeGarminActivity = {
  id: 'connect:123',
  name: 'Authored retrieval fixture',
  sport: 'run',
  startDate: '2026-07-19T12:00:00Z',
  startDateLocal: '2026-07-19T08:00:00',
  distanceM: 410,
  movingTimeS: 120,
  elapsedTimeS: 120,
}

interface Route {
  status: number
  bytes: Uint8Array
  contentType?: string
  location?: string
}

interface RequestReceipt {
  path: string
  status: number
  method: string | undefined
  authorization: string | undefined
  responseSha256: string
}

test('retrieves provider source files through real local HTTP', async t => {
  const fit = new Uint8Array(
    await fs.readFile(new URL('../fixtures/wahoo-garmin-stamina.fit', import.meta.url)),
  )
  const tcx = new Uint8Array(await fs.readFile(new URL('./running.tcx', import.meta.url)))
  assert.equal(fit.byteLength, 259)
  const stored = fixtureArchive([
    { name: 'earlier.tcx', bytes: tcx, method: 0 },
    { name: 'original.FIT', bytes: fit, method: 0 },
  ])
  const deflated = fixtureArchive([
    { name: 'earlier.tcx', bytes: tcx, method: 8 },
    { name: 'original.FIT', bytes: fit, method: 8 },
  ])
  const tcxArchive = fixtureArchive([{ name: 'original.TCX', bytes: tcx, method: 8 }])
  const truncated = Buffer.from(stored)
  const endOffset = truncated.length - 22
  const centralOffset = truncated.readUInt32LE(endOffset + 16)
  const secondEntry = centralOffset + 46 + Buffer.byteLength('earlier.tcx')
  truncated.writeUInt32LE(truncated.length - 10, secondEntry + 42)
  const missing = fixtureArchive([
    { name: 'readme.txt', bytes: Buffer.from('no activity'), method: 0 },
  ])
  const routes = new Map<string, Route>([
    ['/stored/download-service/files/activity/123', { status: 200, bytes: stored }],
    ['/deflated/download-service/files/activity/123', { status: 200, bytes: deflated }],
    ['/tcx/download-service/files/activity/123', { status: 200, bytes: tcxArchive }],
    [
      '/malformed/download-service/files/activity/123',
      { status: 200, bytes: Buffer.from('not a ZIP') },
    ],
    ['/truncated/download-service/files/activity/123', { status: 200, bytes: truncated }],
    ['/missing/download-service/files/activity/123', { status: 200, bytes: missing }],
    ['/wahoo.fit', { status: 200, bytes: fit }],
    ['/wahoo-redirect.fit', { status: 302, bytes: new Uint8Array(), location: '/wahoo.fit' }],
    ['/wahoo-forbidden.fit', { status: 403, bytes: Buffer.from('forbidden') }],
  ])
  for (const status of [401, 403])
    routes.set(`/auth-${status}/download-service/files/activity/123`, {
      status,
      bytes: Buffer.from('<html>sign in</html>'),
      contentType: 'text/html',
    })
  const requests: RequestReceipt[] = []
  const cases: { name: string; passed: boolean; error?: string }[] = []
  const artifacts = process.env.ACTIVITY_SOURCE_FILE_ARTIFACTS
  let decodedOutput: unknown = null
  let convertedBytes: Uint8Array | undefined
  const server = createServer((request, response) => {
    const path = request.url ?? '/'
    const route: Route = routes.get(path) ?? {
      status: 404,
      bytes: Buffer.from('unknown fixture route'),
    }
    requests.push({
      path,
      status: route.status,
      method: request.method,
      authorization: request.headers.authorization,
      responseSha256: sha256(route.bytes),
    })
    response.writeHead(route.status, {
      'Content-Type': route.contentType ?? 'application/octet-stream',
      ...(route.location ? { Location: route.location } : {}),
    })
    response.end(route.bytes)
  })
  await new Promise<void>((resolveListen, rejectListen) => {
    server.once('error', rejectListen)
    server.listen(0, '127.0.0.1', resolveListen)
  })
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const base = `http://127.0.0.1:${address.port}`
  const session = { headers: { authorization: 'Bearer local-fixture-only' } }
  const mappedRequests: { url: string; redirect: RequestRedirect | undefined }[] = []
  const request: typeof fetch = async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : input.toString())
    assert.equal(url.origin, 'https://activity-source-fixture.example.org')
    mappedRequests.push({ url: url.toString(), redirect: init?.redirect })
    return fetch(`${base}${url.pathname}${url.search}`, init)
  }
  const wahooClient = new WahooCloudClient(
    { clientId: 'fixture-client', clientSecret: 'fixture-secret', refreshToken: 'fixture-token' },
    { request },
  )
  const wahooActivity: ActivityBridgeWahooActivity = {
    ...garminActivity,
    id: 'wahoo:456',
    workoutId: 456,
    fitUrl: 'https://activity-source-fixture.example.org/wahoo.fit',
    fitSha256: sha256(fit).toUpperCase(),
  }
  async function check(name: string, run: () => Promise<void>): Promise<void> {
    await t.test(name, async () => {
      try {
        await run()
        cases.push({ name, passed: true })
      } catch (error) {
        cases.push({
          name,
          passed: false,
          error: error instanceof Error ? error.message : String(error),
        })
        throw error
      }
    })
  }
  const garminRetrievers: readonly [string, typeof fetchGarminActivityFile][] = [
    ['original', fetchGarminActivityFile],
    ['fit', fetchGarminActivityFit],
  ]
  const invalidArchives: readonly [string, RegExp][] = [
    ['malformed', /no ZIP end record/],
    ['truncated', /archive is truncated/],
    ['missing', /contains no FIT or TCX file/],
  ]
  try {
    for (const archive of ['stored', 'deflated']) {
      for (const [name, retrieve] of garminRetrievers) {
        await check(
          `Garmin ${archive} ${name} keeps FIT bytes and prefers FIT over earlier TCX`,
          async () => {
            const source = await retrieve(garminActivity, session, `${base}/${archive}`)
            assert.deepEqual(source.bytes, fit)
            assert.equal(source.sha256, sha256(fit))
            assert.equal(source.filename, 'garmin-123.fit')
            assert.equal(source.kind, 'fit')
            assert.equal(source.sourceKind, 'fit')
          },
        )
      }
    }
    await check('Garmin original TCX preserves XML, hash, and provenance', async () => {
      const source = await fetchGarminActivityFile(garminActivity, session, `${base}/tcx`)
      assert.deepEqual(source.bytes, tcx)
      assert.equal(new TextDecoder().decode(source.bytes).startsWith('<?xml'), true)
      assert.equal(source.sha256, sha256(tcx))
      assert.equal(source.filename, 'garmin-123.tcx')
      assert.equal(source.kind, 'tcx')
      assert.equal(source.sourceKind, 'tcx')
    })
    await check(
      'Garmin TCX conversion produces integral FIT with telemetry and two laps',
      async () => {
        const source = await fetchGarminActivityFit(garminActivity, session, `${base}/tcx`)
        convertedBytes = source.bytes
        assert.equal(source.sha256, sha256(source.bytes))
        assert.equal(source.filename, 'garmin-123.fit')
        assert.equal(source.kind, 'fit')
        assert.equal(source.sourceKind, 'tcx')
        assert.equal(Decoder.isFIT(Stream.fromByteArray(source.bytes)), true)
        assert.equal(new Decoder(Stream.fromByteArray(source.bytes)).checkIntegrity(), true)
        const decoded = new Decoder(Stream.fromByteArray(source.bytes)).read()
        assert.deepEqual(decoded.errors, [])
        decodedOutput = decoded.messages
        const records = decoded.messages.recordMesgs ?? []
        const laps = decoded.messages.lapMesgs ?? []
        const activitySession = decoded.messages.sessionMesgs?.[0]
        assert.equal(records.length, 4)
        assert.equal(records[0].heartRate, 120)
        assert.equal(records[0].positionLat, Math.round((43.6532 * 2 ** 31) / 180))
        assert.equal(records[0].cadence, 81)
        assert.equal(records[0].power, 240)
        assert.equal(records[0].temperature, 18)
        assert.equal(records[1].enhancedSpeed, 3.3)
        assert.equal(records[3].distance, 410)
        assert.equal(laps.length, 2)
        assert.deepEqual(
          laps.map(lap => lap.totalDistance),
          [200, 210],
        )
        assert.equal(activitySession?.sport, 'running')
        assert.equal(activitySession?.totalElapsedTime, 120)
        assert.equal(activitySession?.totalTimerTime, 120)
        assert.equal(activitySession?.totalDistance, 410)
        assert.equal(activitySession?.numLaps, 2)
        const firstTimestamp = records[0].timestamp
        const lastTimestamp = records[3].timestamp
        assert.ok(firstTimestamp instanceof Date)
        assert.ok(lastTimestamp instanceof Date)
        assert.equal(firstTimestamp.toISOString(), '2026-07-19T12:00:00.000Z')
        assert.equal(lastTimestamp.toISOString(), '2026-07-19T12:02:00.000Z')
      },
    )
    await check('Wahoo preserves raw FIT and accepts uppercase expected SHA', async () => {
      const source = await fetchWahooActivityFit(wahooActivity, wahooClient)
      assert.deepEqual(source.bytes, fit)
      assert.equal(source.sha256, sha256(fit))
      assert.equal(source.filename, 'wahoo-456.fit')
      assert.equal(source.kind, 'fit')
      assert.equal(source.sourceKind, 'fit')
    })
    await check('Wahoo follows redirects manually through the real client', async () => {
      const before = mappedRequests.length
      const source = await fetchWahooActivityFit(
        {
          ...wahooActivity,
          fitUrl: 'https://activity-source-fixture.example.org/wahoo-redirect.fit',
        },
        wahooClient,
      )
      assert.deepEqual(source.bytes, fit)
      assert.deepEqual(
        mappedRequests.slice(before).map(item => item.redirect),
        ['manual', 'manual'],
      )
    })
    await check('Wahoo rejects a checksum mismatch after HTTP download', async () => {
      const before = requests.length
      await assert.rejects(
        fetchWahooActivityFit({ ...wahooActivity, fitSha256: '0'.repeat(64) }, wahooClient),
        /Wahoo FIT SHA-256 changed for wahoo:456/,
      )
      assert.equal(requests.length, before + 1)
    })
    for (const status of [401, 403]) {
      await check(`Garmin ${status} preserves auth recovery before archive decoding`, async () => {
        await assert.rejects(
          fetchGarminActivityFit(garminActivity, session, `${base}/auth-${status}`),
          new RegExp(`Garmin bearer session rejected \\(${status}\\); run pnpm garmin:auth`),
        )
      })
    }
    await check('Wahoo 403 remains a provider download failure', async () => {
      await assert.rejects(
        fetchWahooActivityFit(
          {
            ...wahooActivity,
            fitUrl: 'https://activity-source-fixture.example.org/wahoo-forbidden.fit',
          },
          wahooClient,
        ),
        /Wahoo FIT download failed: 403/,
      )
    })
    await check('invalid Garmin IDs reject before HTTP', async () => {
      const before = requests.length
      for (const retrieve of [fetchGarminActivityFile, fetchGarminActivityFit]) {
        for (const id of ['', 'connect:', 'connect:abc', '123/../../456']) {
          await assert.rejects(
            retrieve({ ...garminActivity, id }, session, base),
            /Garmin activity id is invalid/,
          )
        }
      }
      assert.equal(requests.length, before)
    })
    for (const [route, expected] of invalidArchives) {
      await check(`Garmin rejects ${route} archive through retrieval`, async () => {
        await assert.rejects(
          fetchGarminActivityFile(garminActivity, session, `${base}/${route}`),
          expected,
        )
      })
    }
    await check('all Garmin HTTP downloads include the supplied bearer session', async () => {
      const garminRequests = requests.filter(item => item.path.includes('/download-service/'))
      assert.ok(garminRequests.length > 0)
      for (const entry of garminRequests) {
        assert.equal(entry.authorization, session.headers.authorization)
        assert.equal(entry.method, 'GET')
      }
    })
  } finally {
    await new Promise<void>((resolveClose, rejectClose) => {
      server.close(error => (error ? rejectClose(error) : resolveClose()))
      server.closeAllConnections()
    })
    if (artifacts) {
      const directory = resolve(artifacts)
      await fs.mkdir(directory, { recursive: true })
      const files = {
        'original.fit': fit,
        'original.tcx': tcx,
        'stored.zip': stored,
        'deflated.zip': deflated,
        'tcx.zip': tcxArchive,
        'truncated.zip': truncated,
        'missing.zip': missing,
        ...(convertedBytes ? { 'converted.fit': convertedBytes } : {}),
      }
      for (const [name, bytes] of Object.entries(files))
        await fs.writeFile(resolve(directory, name), bytes)
      await fs.writeFile(
        resolve(directory, 'receipt.json'),
        JSON.stringify(
          {
            command: `ACTIVITY_SOURCE_FILE_ARTIFACTS=${artifacts} pnpm test quartz/util/activity-source-file/retrieval.e2e.test.ts`,
            createdAt: new Date().toISOString(),
            transport:
              'loopback HTTP only, HTTPS fixture hostname mapped by real Wahoo client request option',
            cases,
            requests,
            mappedRequests,
            fixtures: Object.entries(files).map(([name, bytes]) => ({
              name,
              bytes: bytes.length,
              sha256: sha256(bytes),
            })),
            decodedConvertedFit: decodedOutput,
          },
          null,
          2,
        ),
      )
    }
  }
})
