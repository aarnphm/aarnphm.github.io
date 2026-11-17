import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import test from 'node:test'

const ids = ['9007199254740992', '9007199254740993']
const script = resolve('quartz/scripts/sync-strava.ts')
const tsx = resolve('node_modules/tsx/dist/cli.mjs')
const raw = (id: string, overrides: Record<string, unknown> = {}) => ({
  id: Number(id),
  id_str: id,
  name: `ride ${id}`,
  athlete: { id: Number(ids[1]), id_str: ids[1] },
  sport_type: 'Ride',
  start_date: '2026-11-16T12:00:00Z',
  ...overrides,
})

// Replay the real CLI with an HTTP provider fixture and a private cache; no live credentials.
// Invalid API identities and non-array responses must leave persisted bytes unchanged. Exact
// stream/detail paths, empty pages, legacy reads and overlapping refreshes are observable here.
test('Strava CLI migrates IDs without precision loss or destructive failed syncs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'strava-cli-ids-'))
  const cachePath = join(root, 'quartz/.quartz-cache/strava.json')
  const evidence: unknown[] = []
  let batch: unknown = ids.map(id => raw(id))
  let athleteResponse: unknown = { id: Number(ids[1]), id_str: ids[1], ftp: 250 }
  let detailIdOverride: unknown = undefined
  const requests: string[] = []
  const server = createServer((req, res) => {
    const url = new URL(req.url!, 'http://localhost')
    requests.push(url.pathname)
    res.setHeader('content-type', 'application/json')
    if (url.pathname === '/athlete/activities') res.end(JSON.stringify(batch))
    else if (url.pathname === '/athlete') res.end(JSON.stringify(athleteResponse))
    else if (url.pathname === '/athlete/zones') res.end('{}')
    else if (url.pathname.endsWith('/streams'))
      res.end(JSON.stringify({ time: { data: [0, 1] }, heartrate: { data: [140, 141] } }))
    else if (url.pathname.startsWith('/activities/')) {
      const id = url.pathname.split('/')[2]
      setTimeout(
        () =>
          res.end(
            JSON.stringify({
              ...raw(id, { athlete: athleteResponse }),
              ...(detailIdOverride === undefined ? {} : { id_str: detailIdOverride }),
              laps: [
                ...ids.map(id => ({
                  id: Number(id),
                  id_str: id,
                  elapsed_time: 1,
                  distance: 8,
                  start_index: 0,
                  end_index: 1,
                })),
                { id: Number(ids[0]), elapsed_time: 1, distance: 8 },
              ],
              description: id,
              calories: 100,
            }),
          ),
        20,
      )
    } else {
      res.statusCode = 404
      res.end('{}')
    }
  })
  await new Promise<void>(done => server.listen(0, '127.0.0.1', done))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const run = (args: string[] = []) =>
    new Promise<{ code: number | null; output: string }>((done, reject) => {
      const child = spawn(process.execPath, [tsx, script, ...args], {
        cwd: root,
        env: {
          PATH: process.env.PATH,
          HOME: process.env.HOME,
          STRAVA_ACCESS_TOKEN: 'test-only',
          STRAVA_API_BASE_URL: `http://127.0.0.1:${address.port}`,
          STRAVA_MAX_READ_REQUESTS: '90',
        },
      })
      let output = ''
      child.stdout.on('data', chunk => {
        output += chunk
      })
      child.stderr.on('data', chunk => {
        output += chunk
      })
      child.on('error', reject)
      child.on('exit', code => {
        evidence.push({ args, code, output })
        done({ code, output })
      })
    })
  try {
    await mkdir(join(root, 'content'), { recursive: true })
    await mkdir(join(root, 'quartz/.quartz-cache'), { recursive: true })
    await writeFile(join(root, 'content/triathlon.md'), '# Training\n')
    let result = await run()
    assert.equal(result.code, 0, result.output)
    const saved = JSON.parse(await readFile(cachePath, 'utf8'))
    assert.equal(saved.athleteId, ids[1])
    assert.deepEqual(Object.keys(saved.activities), ids)
    for (const id of ids) {
      assert.equal(saved.activities[id].id, id)
      assert.ok(requests.includes(`/activities/${id}/streams`))
      assert.ok(requests.includes(`/activities/${id}`))
      assert.equal(saved.activityDetails[id].description, id)
      assert.deepEqual(
        saved.activityDetails[id].laps.map((lap: { id: string }) => lap.id),
        [...ids, 'lap-3'],
      )
    }
    result = await run(['--gear-only'])
    assert.equal(result.code, 0, result.output)
    assert.equal(JSON.parse(await readFile(cachePath, 'utf8')).athleteId, ids[1])
    await writeFile(cachePath, JSON.stringify({ ...saved, activityDetails: {} }))
    const concurrent = await Promise.all(ids.map(id => run(['--activity-detail', id])))
    for (const result of concurrent) assert.equal(result.code, 0, result.output)
    const refreshed = JSON.parse(await readFile(cachePath, 'utf8'))
    for (const id of ids) assert.equal(refreshed.activityDetails[id].description, id)
    for (const invalid of [
      {},
      [raw(ids[0], { id_str: undefined })],
      [raw(ids[0], { id_str: '1e3' })],
      [raw(ids[0], { athlete: { id: Number(ids[1]) } })],
      [raw(ids[0], { athlete: {} })],
    ]) {
      batch = invalid
      const before = await readFile(cachePath, 'utf8')
      result = await run()
      assert.notEqual(result.code, 0, JSON.stringify(invalid))
      assert.equal(await readFile(cachePath, 'utf8'), before)
    }
    detailIdOverride = ids[1]
    const before = await readFile(cachePath, 'utf8')
    result = await run(['--activity-detail', ids[0]])
    assert.notEqual(result.code, 0)
    assert.equal(await readFile(cachePath, 'utf8'), before)
    detailIdOverride = undefined
    batch = []
    result = await run()
    assert.equal(result.code, 0, result.output)
    const empty = JSON.parse(await readFile(cachePath, 'utf8'))
    assert.equal(empty.athleteId, ids[1])
    assert.deepEqual(empty.activities, {})
    assert.deepEqual(empty.streams, {})
    assert.deepEqual(empty.activityDetails, {})
    await rm(cachePath)
    result = await run()
    assert.equal(result.code, 0, result.output)
    assert.equal(JSON.parse(await readFile(cachePath, 'utf8')).athleteId, ids[1])
    // Safe legacy fallback in both the authenticated athlete and nested MetaAthlete.
    await rm(cachePath)
    athleteResponse = { id: 7 }
    batch = [raw('42', { id: 42, id_str: undefined, athlete: { id: 7 } })]
    result = await run()
    assert.equal(result.code, 0, result.output)
    const legacy = JSON.parse(await readFile(cachePath, 'utf8'))
    assert.equal(legacy.athleteId, '7')
    assert.equal(legacy.activities['42'].id, '42')
  } finally {
    const artifact = process.env.STRAVA_ID_TEST_ARTIFACTS
    if (artifact) {
      await mkdir(artifact, { recursive: true })
      await writeFile(
        join(artifact, 'cli-replay.json'),
        JSON.stringify(
          {
            command: 'node_modules/.bin/tsx --test quartz/scripts/sync-strava-ids.e2e.test.ts',
            ids,
            requests,
            runs: evidence,
          },
          null,
          2,
        ),
      )
    }
    await new Promise<void>((done, reject) =>
      server.close(error => (error ? reject(error) : done())),
    )
    await rm(root, { recursive: true, force: true })
  }
})
