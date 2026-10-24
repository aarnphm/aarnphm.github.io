import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { promisify } from 'node:util'
import { isRecord } from '../../quartz/util/type-guards'

const execute = promisify(execFile)

test(
  'observer tracks admitted, completed, queued and rejected physical pages',
  { timeout: 180_000 },
  async () => {
    const root = await mkdtemp(path.join(tmpdir(), 'embedding-stream-'))
    const python = path.resolve('.venv/bin/python')
    try {
      for (const failed of [false, true]) {
        await execute(python, [
          'harness/embeddings/streaming-fixture.py',
          root,
          String(process.pid),
          ...(failed ? ['--failed'] : []),
        ])
        const statusPath = path.join(
          root,
          'quartz/.quartz-cache/pdf-search/background-papers/status.json',
        )
        const heartbeat = setInterval(() => {
          const status: unknown = JSON.parse(readFileSync(statusPath, 'utf8'))
          assert.ok(isRecord(status))
          status.updatedAt = new Date().toISOString()
          writeFileSync(statusPath, JSON.stringify(status))
        }, 1000)
        const result = await execute(python, [
          'harness/embeddings/snapshot.py',
          '--root',
          root,
          '--once',
        ]).finally(() => clearInterval(heartbeat))
        const frame: unknown = JSON.parse(result.stdout)
        assert.ok(isRecord(frame) && isRecord(frame.data))
        const data = frame.data
        assert.ok(isRecord(data.runner) && isRecord(data.current))
        assert.equal(data.runner.batchCompleted, 3, 'duplicate completion records count once')
        assert.equal(data.runner.batchPlanned, 8, 'the final batch uses its actual planned size')
        assert.equal(data.runner.batchActive, failed ? 2 : 3)
        assert.equal(data.runner.batchQueued, 2)
        assert.deepEqual(
          data.current.pages,
          [2, 4],
          'later completed pages do not hide earlier active pages',
        )
        assert.ok(Array.isArray(data.runner.pages))
        assert.deepEqual(
          data.runner.pages.map(row => (isRecord(row) ? row.state : null)),
          [
            'active',
            'done',
            'active',
            'done',
            'queued',
            'queued',
            failed ? 'failed' : 'active',
            'done',
          ],
        )
        if (failed) {
          assert.ok(Array.isArray(data.issues))
          assert.ok(
            data.issues.some(
              message =>
                typeof message === 'string' &&
                message.includes('beta.pdf') &&
                message.includes('truncated region'),
            ),
          )
        }
        console.log(JSON.stringify({ failed, current: data.current, runner: data.runner }))
      }
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  },
)
