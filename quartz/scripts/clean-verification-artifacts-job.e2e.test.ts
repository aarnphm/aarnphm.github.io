import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { lstat, mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const repo = fileURLToPath(new URL('../..', import.meta.url))
const runner = path.join(repo, 'quartz/scripts/clean-verification-artifacts-job.sh')
const evidence = path.join(repo, 'docs/verification/artifact-cleanup-launchd/artifacts')

test('local cleanup job is silent on success and records real command failures', async t => {
  const fixture = await mkdtemp(path.join(tmpdir(), 'garden-cleanup-job-'))
  const logDirectory = path.join(fixture, 'logs with spaces')
  const configuredPnpm = process.env.npm_execpath
  assert.ok(configuredPnpm, 'run this E2E through pnpm test')
  const pnpm: string = configuredPnpm
  const commands: { args: string[]; status: number | null; stdout: string; stderr: string }[] = []
  const checks: string[] = []

  t.after(async () => {
    await mkdir(evidence, { recursive: true })
    await writeFile(
      path.join(evidence, 'results.json'),
      JSON.stringify(
        {
          command: 'pnpm test quartz/scripts/clean-verification-artifacts-job.e2e.test.ts',
          fixture,
          commands,
          checks,
          completed: checks.length === 4,
        },
        null,
        2,
      ) + '\n',
    )
    await rm(fixture, { recursive: true, force: true })
  })

  const artifacts = path.join(fixture, 'docs/verification/check/artifacts')
  await mkdir(artifacts, { recursive: true })
  const oldFile = path.join(artifacts, 'expired.json')
  const freshFile = path.join(artifacts, 'fresh.json')
  await writeFile(oldFile, 'expired evidence')
  await writeFile(freshFile, 'fresh evidence')
  const old = new Date(Date.now() - 73 * 60 * 60_000)
  await utimes(oldFile, old, old)

  async function run(directory: string, ...flags: string[]) {
    const args = [runner, directory, pnpm, logDirectory, ...flags]
    const result = await new Promise<{ status: number | null; stdout: string; stderr: string }>(
      (resolve, reject) => {
        const child = spawn('/bin/sh', args, {
          env: {
            ...process.env,
            PATH: `${path.dirname(process.execPath)}:/usr/bin:/bin:/usr/sbin:/sbin`,
            GARDEN_ARTIFACT_CLEANUP_NOTIFY: '0',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
          timeout: 10_000,
        })
        const stdout: Buffer[] = []
        const stderr: Buffer[] = []
        child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk))
        child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk))
        child.once('error', reject)
        child.once('close', status =>
          resolve({
            status,
            stdout: Buffer.concat(stdout).toString(),
            stderr: Buffer.concat(stderr).toString(),
          }),
        )
      },
    )
    commands.push({ args: ['/bin/sh', ...args], ...result })
    return result
  }

  const success = await run(repo, '--repo', fixture)
  assert.deepEqual(success, { status: 0, stdout: '', stderr: '' })
  await assert.rejects(lstat(oldFile), { code: 'ENOENT' })
  assert.equal(await readFile(freshFile, 'utf8'), 'fresh evidence')
  await assert.rejects(lstat(logDirectory), { code: 'ENOENT' })
  checks.push('success uses the real cleanup, emits nothing, and creates no log')

  const failure = await run(repo, '--invalid-cleanup-option')
  assert.equal(failure.status, 1)
  assert.equal(failure.stdout, '')
  const failureLog = path.join(logDirectory, 'failure.log')
  const details = await readFile(failureLog, 'utf8')
  assert.ok(details.includes('--invalid-cleanup-option'))
  assert.ok(details.includes('exit status: 1'))
  checks.push('real CLI failure preserves its exit status and diagnostic output')

  const missingRepo = await run(path.join(fixture, 'missing repository'))
  assert.ok(missingRepo.status !== 0)
  const latestFailure = await readFile(failureLog, 'utf8')
  assert.ok(latestFailure.includes('missing repository'))
  assert.ok(!latestFailure.includes('--invalid-cleanup-option'))
  checks.push('missing repository is reported and the failure log holds only the latest failure')

  const recovered = await run(repo, '--repo', fixture)
  assert.deepEqual(recovered, { status: 0, stdout: '', stderr: '' })
  assert.equal(await readFile(failureLog, 'utf8'), latestFailure)
  checks.push('a later success stays silent and preserves the last failure diagnostic')
})
