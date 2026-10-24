import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { resolve } from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import { Worker as Thread } from 'node:worker_threads'
import { CodeSearchClient } from './code-search'

// Browser transport adapter backed by real worker_threads, with actual bundled provider code.
class ThreadWorker extends EventTarget {
  static started = 0
  static ended = 0
  private thread: Thread
  constructor(url: string) {
    super()
    ThreadWorker.started++
    this.thread = new Thread(new URL(url))
    this.thread.on('message', data => this.dispatchEvent(new MessageEvent('message', { data })))
    this.thread.on('error', error => {
      const event = new Event('error')
      Object.defineProperty(event, 'message', {
        value: error instanceof Error ? error.message : String(error),
      })
      this.dispatchEvent(event)
    })
    this.thread.on('exit', () => ThreadWorker.ended++)
  }
  postMessage(value: unknown): void {
    this.thread.postMessage(value)
  }
  terminate(): void {
    void this.thread.terminate()
  }
}

test('client drives real worker, cancels active queries, resumes and releases idle worker', async () => {
  const directory = resolve('docs/verification/code-search/artifacts/runtime')
  await mkdir(directory, { recursive: true })
  await build({
    entryPoints: ['quartz/workers/code-search.worker.ts'],
    outfile: `${directory}/worker.mjs`,
    bundle: true,
    format: 'esm',
    platform: 'browser',
    external: ['*/code-search-ast'],
  })
  await writeFile(
    `${directory}/bridge.mjs`,
    `import { parentPort } from 'node:worker_threads';\nglobalThis.self = globalThis; globalThis.postMessage = value => parentPort.postMessage(value); globalThis.addEventListener = (_, listener) => parentPort.on('message', data => listener({data}));\nawait import('./worker.mjs');\n`,
  )
  const text = 'foo(😀)\r\nfoo()'
  const hash = (value: string) => createHash('sha256').update(value).digest('hex')
  const id = hash(text)
  const shard = JSON.stringify({
    kind: 'quartz-code-sources-v1',
    generation: 'test',
    language: 'python',
    sources: [{ id, text }],
  })
  const requests: string[] = []
  const server = createServer((request, response) => {
    requests.push(request.url ?? '')
    response.end(
      request.url?.endsWith('manifest.json')
        ? JSON.stringify({
            kind: 'quartz-code-search-v1',
            version: 1,
            generation: 'test',
            units: [
              {
                id: 'unit',
                sourceId: id,
                ownerSlug: 'owner',
                title: 'Owner',
                path: 'owner.py',
                kind: 'attachment',
                language: 'python',
                originalLanguage: 'py',
                anchor: null,
                startLine: 1,
                endLine: 2,
              },
            ],
            shards: [
              {
                language: 'python',
                path: 'test/python.json',
                sha256: hash(shard),
                byteLength: Buffer.byteLength(shard),
                sourceIds: [id],
              },
            ],
            coverage: { owners: 1, units: 1, sources: 1 },
          })
        : shard,
    )
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const origin = `http://127.0.0.1:${address.port}`
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker')
  const locationDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'location')
  Object.defineProperty(globalThis, 'Worker', { value: ThreadWorker, configurable: true })
  Object.defineProperty(globalThis, 'location', { value: { href: origin }, configurable: true })
  const client = new CodeSearchClient({
    workerUrl: pathToFileURL(`${directory}/bridge.mjs`).href,
    idleMs: 30,
    timeoutMs: 30_000,
  })
  const query = { mode: 'literal', pattern: 'foo', caseSensitive: true } satisfies Parameters<
    CodeSearchClient['search']
  >[0]
  try {
    await assert.rejects(
      client.search(query, progress => {
        if (progress.phase === 'loading') client.cancel()
      }),
      { name: 'AbortError' },
    )
    const result = await client.search(query)
    assert.equal(result.totalMatches, 2)
    assert.equal(result.matches[0].ranges[1].start.offset, 9)
    assert.ok(requests.every(url => url.endsWith('.json')))
    const workersBeforeIdleCancel = ThreadWorker.started
    const requestsBeforeIdleCancel = requests.length
    client.cancelPending()
    assert.equal((await client.search(query)).totalMatches, 2)
    assert.equal(
      ThreadWorker.started,
      workersBeforeIdleCancel,
      'idle cancellation retains the worker',
    )
    assert.equal(
      requests.length,
      requestsBeforeIdleCancel,
      'idle cancellation retains manifest and source caches',
    )
    await assert.rejects(
      client.search(query, progress => {
        if (progress.phase === 'loading') client.cancelPending()
      }),
      { name: 'AbortError' },
    )
    assert.equal((await client.search(query)).totalMatches, 2)
    assert.equal(
      ThreadWorker.started,
      workersBeforeIdleCancel + 1,
      'active cancellation requires a fresh worker',
    )
    client.cancelPending()
    const releaseDeadline = performance.now() + 10_000
    while (ThreadWorker.started !== ThreadWorker.ended && performance.now() < releaseDeadline) {
      await new Promise(resolve => setTimeout(resolve, 20))
    }
    assert.equal(ThreadWorker.started, ThreadWorker.ended)
    assert.equal((await client.search(query)).matches.length, 1)
    client.dispose()
    await assert.rejects(client.search(query), /disposed/)
    await writeFile(
      `${directory}/receipt.json`,
      JSON.stringify(
        {
          command:
            'node node_modules/tsx/dist/cli.mjs --test quartz/components/scripts/code-search.e2e.test.ts',
          requests,
          startedWorkers: ThreadWorker.started,
          matches: result.totalMatches,
          checks: [
            'active cancellation',
            'idle cancelPending retains worker and caches',
            'active cancelPending rejects and restarts',
            'idle cancelPending preserves idle release timer',
            'new worker recovery',
            'idle release',
            'disposal',
            'literal JSON-only fetches',
          ],
        },
        null,
        2,
      ),
    )
  } finally {
    client.dispose()
    if (workerDescriptor) Object.defineProperty(globalThis, 'Worker', workerDescriptor)
    else Reflect.deleteProperty(globalThis, 'Worker')
    if (locationDescriptor) Object.defineProperty(globalThis, 'location', locationDescriptor)
    else Reflect.deleteProperty(globalThis, 'location')
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
