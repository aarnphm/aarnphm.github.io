import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { test } from 'node:test'
import { promisify } from 'node:util'
import { isRecord } from '../../quartz/util/type-guards'

const execute = promisify(execFile)

test(
  'remaining OCR time uses current-run work, measured mean and unique uncached pages',
  { timeout: 240_000 },
  async () => {
    const root = await mkdtemp(path.join(tmpdir(), 'embedding-timings-'))
    const python = path.resolve('.venv/bin/python')
    const snapshots: unknown[] = []
    await execute(python, ['harness/embeddings/timing-fixture.py', root, '0', String(process.pid)])
    const observer = spawn(python, [
      '-u',
      'harness/embeddings/snapshot.py',
      '--root',
      root,
      '--interval',
      '0.2',
    ])
    let error = ''
    observer.stderr.on('data', bytes => {
      error += bytes.toString()
    })
    let latest: Record<string, unknown> | undefined
    const lines = createInterface({ input: observer.stdout })
    lines.on('line', line => {
      const frame: unknown = JSON.parse(line)
      if (isRecord(frame) && frame.type === 'snapshot' && isRecord(frame.data)) latest = frame.data
      else error += line
    })
    const observe = async (pages: number, finish = false) => {
      const since = Date.now()
      await execute(python, [
        'harness/embeddings/timing-fixture.py',
        root,
        String(pages),
        String(process.pid),
        ...(finish ? ['--finish'] : []),
      ])
      const deadline = Date.now() + 180_000
      while (
        !(
          latest &&
          typeof latest.observedAt === 'string' &&
          Date.parse(latest.observedAt) >= since &&
          isRecord(latest.totals) &&
          latest.totals.ocrPages === (finish ? 11 : pages + 1)
        )
      ) {
        if (Date.now() >= deadline)
          throw new Error(`Missing timing observation: ${JSON.stringify(latest)}; ${error}`)
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      const data = latest
      assert.ok(isRecord(data.rate) && isRecord(data.totals))
      snapshots.push(data.rate)
      return { rate: data.rate, totals: data.totals }
    }
    try {
      const empty = await observe(0)
      assert.equal(
        empty.rate.samples,
        0,
        'migrated pages and earlier runs are not new timing samples',
      )
      assert.equal(empty.rate.remainingOcrSeconds, null)
      const collecting = await observe(2)
      assert.equal(collecting.rate.samples, 2)
      assert.equal(
        collecting.rate.remainingOcrSeconds,
        null,
        'a small sample does not produce an ETA',
      )
      const estimated = await observe(5)
      assert.equal(estimated.rate.samples, 5, 'repeated log records do not multiply timing samples')
      assert.equal(
        estimated.rate.secondsPerPage,
        30,
        'slow pages contribute to total expected work',
      )
      assert.equal(estimated.rate.pagesPerHour, 120)
      assert.equal(
        estimated.totals.remainingOcrPages,
        5,
        'coverage keeps both duplicate document references',
      )
      assert.equal(
        estimated.rate.remainingUniquePages,
        3,
        'inference only processes duplicate PDF bytes once',
      )
      assert.equal(estimated.rate.remainingOcrSeconds, 90)
      const advanced = await observe(6)
      assert.equal(advanced.rate.samples, 6)
      assert.equal(advanced.rate.remainingUniquePages, 2)
      assert.ok(typeof advanced.rate.remainingOcrSeconds === 'number')
      assert.ok(Math.abs(advanced.rate.remainingOcrSeconds - 200 / 3) < 1e-8)
      const complete = await observe(6, true)
      assert.equal(complete.rate.remainingUniquePages, 0)
      assert.equal(complete.rate.remainingOcrSeconds, 0)
      console.log(JSON.stringify({ snapshots }))
    } finally {
      observer.kill('SIGTERM')
      if (observer.exitCode === null) await once(observer, 'exit')
      lines.close()
      await rm(root, { recursive: true, force: true })
    }
  },
)
