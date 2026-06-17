import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'

type DevScriptResult = { code: number | null; stderr: string; stdout: string }
type DevScriptOptions = { env?: NodeJS.ProcessEnv }

function runDevScript(args: string[], options: DevScriptOptions = {}): Promise<DevScriptResult> {
  return new Promise((resolve, reject) => {
    const child = spawn('pnpm', ['exec', 'tsx', 'quartz/scripts/dev.ts', ...args], {
      cwd: process.cwd(),
      env: { ...process.env, ...options.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const stderr: Buffer[] = []
    const stdout: Buffer[] = []
    let settled = false

    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timeout)
      fn()
    }

    const timeout = setTimeout(() => {
      finish(() => {
        child.kill('SIGTERM')
        reject(new Error(`dev script timed out: ${Buffer.concat(stderr).toString('utf8')}`))
      })
    }, 3000)

    if (!child.stdout || !child.stderr) {
      finish(() => {
        child.kill('SIGTERM')
        reject(new Error('dev script did not expose stdio'))
      })
      return
    }

    child.stdout.on('data', (chunk: Buffer) => stdout.push(Buffer.from(chunk)))
    child.stderr.on('data', (chunk: Buffer) => stderr.push(Buffer.from(chunk)))
    child.once('error', error => {
      finish(() => reject(error))
    })
    child.once('close', code => {
      finish(() => {
        resolve({
          code,
          stderr: Buffer.concat(stderr).toString('utf8'),
          stdout: Buffer.concat(stdout).toString('utf8'),
        })
      })
    })
  })
}

test('dev script parses options after pnpm separator', async () => {
  const result = await runDevScript(['--', '--help'])

  assert.equal(result.code, 0)
  assert.match(result.stdout, /usage: pnpm swarm -- \[options\]/)
  assert.match(result.stdout, /--daemon/)
  assert.match(result.stdout, /--kill/)
  assert.doesNotMatch(`${result.stdout}\n${result.stderr}`, /launching pnpm dev/)
})

test('dev script kill is idempotent when no daemon pid exists', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-dev-cli-'))
  t.after(async () => {
    await rm(root, { recursive: true, force: true })
  })

  const result = await runDevScript(['--kill'], { env: { GIT_ROOT: root } })

  assert.equal(result.code, 0)
  assert.match(result.stdout, /\.dev\.pid/)
  assert.doesNotMatch(`${result.stdout}\n${result.stderr}`, /launching pnpm dev/)
})
