import type { AddressInfo } from 'node:net'
import { StringStream } from '@codemirror/language'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp } from 'node:fs/promises'
import { createServer, type RequestListener, type Server } from 'node:http'
import os from 'node:os'
import path from 'node:path'
import test, { describe } from 'node:test'
import type { CellId } from '../../util/notebook/types'
import type { RuntimeEvent } from '../notebook/kernel'
import { leanStreamParser } from '../../util/codemirror-lean'
import { handleLocalLeanVerifyNodeRequest, resolveLeanBinary } from '../../util/lean-local-verify'
import { createLeanKernel, getSharedLeanWebSocketSession, LeanWebSocketSession } from './kernel'

const cellId = 'cell-1' as CellId

async function collectEvents(iterable: AsyncIterable<RuntimeEvent>): Promise<RuntimeEvent[]> {
  const events: RuntimeEvent[] = []
  for await (const event of iterable) {
    events.push(event)
  }
  return events
}

async function withServer(
  handler: RequestListener,
  run: (url: string, server: Server) => Promise<void>,
): Promise<void> {
  const server = createServer(handler)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const { port } = server.address() as AddressInfo
  try {
    await run(`http://127.0.0.1:${port}/api/lean/verify`, server)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

describe('LeanKernel execution & InfoView rendering', () => {
  test('renders structured HTML InfoView alongside stdout stream and error diagnostics when goals or sorries exist', async () => {
    await withServer(
      (_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            messages: [
              {
                severity: 'information',
                pos: { line: 1, column: 0 },
                data: 'Nat.add_comm (n m : Nat) : n + m = m + n',
              },
              {
                severity: 'error',
                pos: { line: 3, column: 2 },
                data: 'unsolved goals\ncase succ\nn m : Nat\nh : n < m\n⊢ n + 0 < m',
              },
            ],
            sorries: [
              { pos: { line: 6, column: 4 }, goals: ['x : Nat\n⊢ x = x', 'y : Nat\n⊢ y + 0 = y'] },
            ],
          }),
        )
      },
      async endpoint => {
        const kernel = createLeanKernel({
          runtimeId: 'test-lean',
          sourcePath: 'notes/lean.md',
          indexUrl: endpoint,
        })
        const events = await collectEvents(kernel.execute(cellId, '#check Nat.add_comm'))

        const streamEvent = events.find(
          (
            e,
          ): e is Extract<RuntimeEvent, { type: 'output' }> & {
            output: { type: 'stream'; name: string; text: string }
          } => e.type === 'output' && 'type' in e.output && e.output.type === 'stream',
        )
        assert.ok(streamEvent)
        assert.equal(streamEvent.output.name, 'stdout')
        assert.match(streamEvent.output.text, /1:0: Nat\.add_comm/)

        const htmlEvent = events.find(
          (
            e,
          ): e is Extract<RuntimeEvent, { type: 'output' }> & {
            output: { type: 'html'; html: string }
          } => e.type === 'output' && 'type' in e.output && e.output.type === 'html',
        )
        assert.ok(htmlEvent)
        assert.match(htmlEvent.output.html, /class="lean-infoview"/)
        assert.match(htmlEvent.output.html, /class="lean-infoview-goal"/)
        assert.match(htmlEvent.output.html, /case succ/)
        assert.match(htmlEvent.output.html, /class="lean-infoview-hypothesis"/)
        assert.match(htmlEvent.output.html, /n &lt; m/)
        assert.match(htmlEvent.output.html, /class="lean-infoview-turnstile"[^>]*>⊢<\/span>/)
        assert.match(htmlEvent.output.html, /n \+ 0 &lt; m/)
        assert.match(htmlEvent.output.html, /x = x/)
        assert.match(htmlEvent.output.html, /y \+ 0 = y/)
        assert.match(htmlEvent.output.html, /Nat\.add_comm/)

        const errorEvent = events.find(
          (e): e is Extract<RuntimeEvent, { type: 'error' }> => e.type === 'error',
        )
        assert.ok(errorEvent)
        assert.equal(errorEvent.output.ename, 'LeanError')
        assert.ok(errorEvent.output.traceback.some(line => line.includes('3:2: unsolved goals')))
        assert.ok(
          errorEvent.output.traceback.some(
            line => line.includes('6:4: unsolved goal (sorry)') && line.includes('⊢ y + 0 = y'),
          ),
        )

        const doneEvent = events.at(-1)
        assert.deepEqual(doneEvent, { type: 'done', cellId, executionCount: null, failed: true })
      },
    )
  })

  test('falls back to Lean 4 playground handoff when verification endpoint returns 503', async () => {
    await withServer(
      (_req, res) => {
        res.writeHead(503, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'lean verification is not configured' }))
      },
      async endpoint => {
        const kernel = createLeanKernel({
          runtimeId: 'test-lean',
          sourcePath: 'notes/lean.md',
          indexUrl: endpoint,
        })
        const events = await collectEvents(kernel.execute(cellId, '#eval 1 + 1'))
        const htmlEvent = events.find(
          (
            e,
          ): e is Extract<RuntimeEvent, { type: 'output' }> & {
            output: { type: 'html'; html: string }
          } => e.type === 'output' && 'type' in e.output && e.output.type === 'html',
        )
        assert.ok(htmlEvent)
        assert.match(htmlEvent.output.html, /lean-handoff/)
        assert.match(htmlEvent.output.html, /https:\/\/live\.lean-lang\.org\//)
      },
    )
  })

  test('executes real local Lean 4 verification E2E via lean --json and caches on disk', async t => {
    const leanBin = await resolveLeanBinary()
    if (!leanBin) {
      t.skip('local lean binary not installed')
      return
    }
    const cacheDir = await mkdtemp(path.join(os.tmpdir(), 'quartz-lean-cache-'))
    await withServer(
      (req, res) => {
        void handleLocalLeanVerifyNodeRequest(req, res, { cacheDir })
      },
      async endpoint => {
        const source = [
          'import Mathlib',
          '#eval 21 + 21',
          'theorem set_diff_eq {α : Type*} (A B : Set α) : A \\ B = A ∩ Bᶜ := by',
          '  rw [Set.diff_eq]',
          'theorem cantor_diag {α : Type*} (f : α → Set α) : ¬ Function.Surjective f := by',
          '  intro hsurj',
          '  let D : Set α := {a | a ∉ f a}',
          '  obtain ⟨d, hd⟩ := hsurj D',
          '  have hiff : d ∈ f d ↔ d ∉ f d := by',
          '    conv => lhs; rw [hd]',
          '    rfl',
          '  exact (iff_not_self hiff)',
          '#check @cantor_diag',
        ].join('\n')

        const kernel = createLeanKernel({
          runtimeId: 'test-lean-e2e',
          sourcePath: 'notes/sets.md',
          indexUrl: endpoint,
        })
        const events = await collectEvents(kernel.execute(cellId, source))
        const streamEvent = events.find(
          (
            e,
          ): e is Extract<RuntimeEvent, { type: 'output' }> & {
            output: { type: 'stream'; name: string; text: string }
          } => e.type === 'output' && 'type' in e.output && e.output.type === 'stream',
        )
        assert.ok(streamEvent)
        assert.match(streamEvent.output.text, /import Mathlib/i)
        assert.match(streamEvent.output.text, /42/)
        assert.match(streamEvent.output.text, /cantor_diag/)

        const doneEvent = events.at(-1)
        assert.deepEqual(doneEvent, { type: 'done', cellId, executionCount: null, failed: false })

        // Verify disk cache hit on second HTTP request
        const expectedHash = createHash('sha256').update(`mathlib\0${source}`).digest('hex')
        const repeatRes = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: source, imports: 'mathlib' }),
        })
        assert.equal(repeatRes.status, 200)
        assert.equal(repeatRes.headers.get('x-lean-verify-cache'), 'HIT')
        assert.equal(repeatRes.headers.get('x-lean-verify-sha256'), expectedHash)
      },
    )
  })
})

describe('CodeMirror Lean 4 StreamParser', () => {
  test('highlights declarations, tactics, # commands, nested block comments, and unicode operators', () => {
    const state = leanStreamParser.startState!(2)
    const tokenizeLine = (line: string) => {
      const stream = new StringStream(line, 2, 2)
      const tokens: Array<{ text: string; token: string | null }> = []
      while (!stream.eol()) {
        stream.start = stream.pos
        const token = leanStreamParser.token(stream, state)
        tokens.push({ text: stream.current(), token })
      }
      return tokens
    }

    const line1 = tokenizeLine('#check ∀ n : Nat, n + 0 = n /- outer /- inner -/ -/')
    assert.equal(line1.find(t => t.text === '#check')?.token, 'keyword')
    assert.equal(line1.find(t => t.text === '∀')?.token, 'operator')
    assert.equal(line1.find(t => t.text === 'Nat')?.token, 'typeName')
    assert.equal(line1.at(-1)?.token, 'comment')
    assert.equal(state.commentDepth, 0)

    const line2 = tokenizeLine('opaque f : Nat → Nat := by congr; sorry')
    assert.equal(line2.find(t => t.text === 'opaque')?.token, 'keyword')
    assert.equal(line2.find(t => t.text === 'by')?.token, 'keyword')
    assert.equal(line2.find(t => t.text === 'congr')?.token, 'typeName')
    assert.equal(line2.find(t => t.text === 'sorry')?.token, 'invalid')
  })
})

describe('LeanWebSocketSession LSP complete & hover', () => {
  test('returns shared singleton session and returns immediately without blocking when cold', async () => {
    const s1 = getSharedLeanWebSocketSession()
    const s2 = getSharedLeanWebSocketSession()
    assert.equal(s1, s2)

    const session = new LeanWebSocketSession()
    let warmTriggered = false
    session.warm = async () => {
      warmTriggered = true
      return false
    }

    const items = await session.complete('#check Set.mem_', { line: 0, character: 15 }, 50)
    assert.deepEqual(items, [])
    assert.equal(warmTriggered, true)

    warmTriggered = false
    const hover = await session.hover('#check Set.mem_union', { line: 0, character: 12 }, 50)
    assert.equal(hover, undefined)
    assert.equal(warmTriggered, true)
  })
})
