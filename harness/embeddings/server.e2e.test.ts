import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { test } from 'node:test'
import { promisify } from 'node:util'
import WebSocket from 'ws'
import { isRecord } from '../../quartz/util/type-guards'

const execute = promisify(execFile)

test(
  'the read-only harness pushes verified queue changes through HTTP and WebSocket',
  { timeout: 90_000 },
  async () => {
    const root = await mkdtemp(path.join(tmpdir(), 'embedding-harness-'))
    const fixture = path.resolve('harness/embeddings/fixture.py')
    const python = path.resolve('.venv/bin/python')
    await execute(python, [fixture, root, String(process.pid)])
    const server = spawn(
      process.execPath,
      [
        '--import',
        'tsx',
        'harness/embeddings/server.ts',
        '--root',
        root,
        '--port',
        '0',
        '--interval',
        '0.2',
      ],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    )
    let logs = ''
    server.stderr.on('data', chunk => {
      logs += chunk.toString()
    })
    let socket: WebSocket | undefined
    try {
      const address = await new Promise<string>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`Server did not start: ${logs}`)), 10_000)
        server.stdout.on('data', chunk => {
          const match = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/)
          if (match) {
            clearTimeout(timer)
            resolve(match[0])
          }
        })
        server.once('exit', code => {
          clearTimeout(timer)
          reject(new Error(`Server exited ${code}: ${logs}`))
        })
      })
      const frames: Record<string, unknown>[] = []
      socket = new WebSocket(address.replace('http:', 'ws:') + '/events')
      socket.on('message', bytes => {
        const frame: unknown = JSON.parse(bytes.toString())
        if (isRecord(frame) && frame.type === 'snapshot' && isRecord(frame.data))
          frames.push(frame.data)
        if (isRecord(frame) && frame.type === 'error') logs += JSON.stringify(frame)
      })
      const waitFor = async (predicate: (frame: Record<string, unknown>) => boolean) => {
        const deadline = Date.now() + 60_000
        while (Date.now() < deadline) {
          const found = frames.find(predicate)
          if (found) return found
          await new Promise(resolve => setTimeout(resolve, 50))
        }
        throw new Error(`Missing snapshot: ${JSON.stringify(frames.at(-1))}; ${logs}`)
      }
      const first = await waitFor(frame => isRecord(frame.totals) && frame.totals.ocrPages === 2)
      assert.ok(isRecord(first.totals))
      assert.equal(first.totals.totalPages, 5)
      assert.equal(first.totals.indexedPages, 1)
      assert.equal(first.totals.remainingOcrPages, 3)
      assert.equal(first.totals.documents, 2)
      assert.ok(isRecord(first.current))
      assert.equal(first.current.page, 3)
      assert.equal(first.current.slug, 'thoughts/papers/alpha.pdf')
      const response = await fetch(address)
      assert.equal(response.status, 200)
      assert.match(await response.text(), /embedding harness/)
      assert.equal((await fetch(address + '/api/status', { method: 'POST' })).status, 405)
      assert.equal((await fetch(address + '/.env')).status, 404)
      assert.equal((await fetch(address + '/snapshot.py')).status, 404)
      const denied = new WebSocket(address.replace('http:', 'ws:') + '/events', {
        origin: 'https://foreign.invalid',
      })
      denied.on('error', () => {})
      const denial = once(denied, 'unexpected-response')
      const [, rejection] = await denial
      assert.equal(rejection.statusCode, 403)
      denied.terminate()
      await execute(python, [fixture, root, String(process.pid), '--advance'])
      const next = await waitFor(frame => isRecord(frame.totals) && frame.totals.indexedPages === 3)
      assert.ok(isRecord(next.totals))
      assert.equal(next.totals.ocrPages, 3)
      assert.equal(next.totals.completeDocuments, 1)
      assert.equal(next.totals.remainingOcrPages, 2)
      assert.equal(next.totals.remainingIndexPages, 2)
      assert.ok(Array.isArray(next.indexes))
      assert.ok(
        next.indexes.some(
          index =>
            isRecord(index) && index.name === 'pdf' && index.rows === 3 && index.servedRows === 1,
        ),
      )
      const statusFile = path.join(
        root,
        'quartz/.quartz-cache/pdf-search/background-papers/status.json',
      )
      const status = JSON.parse(await readFile(statusFile, 'utf8'))
      await writeFile(
        statusFile,
        JSON.stringify({ ...status, phase: 'error', error: 'fixture extraction failure' }),
      )
      const failed = await waitFor(
        frame => isRecord(frame.runner) && frame.runner.phase === 'error',
      )
      assert.ok(isRecord(failed.runner))
      assert.equal(failed.runner.error, 'fixture extraction failure')
      console.log(
        JSON.stringify({
          initial: first.totals,
          updated: next.totals,
          frames: frames.length,
          writesRejected: true,
          foreignOriginRejected: true,
          errorPreserved: true,
        }),
      )
    } finally {
      socket?.terminate()
      server.kill('SIGTERM')
      if (server.exitCode === null) await once(server, 'exit')
      await rm(root, { recursive: true, force: true })
    }
  },
)
