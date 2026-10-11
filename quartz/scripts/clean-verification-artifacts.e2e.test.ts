import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { lstat, mkdir, mkdtemp, readFile, rm, symlink, utimes, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execute = promisify(execFile)
const repo = fileURLToPath(new URL('../..', import.meta.url))
const script = path.join(repo, 'quartz/scripts/clean-verification-artifacts.ts')
const evidence = path.join(repo, 'docs/verification/artifact-retention/artifacts')

test('artifact cleanup CLI expires old evidence while preserving sources and fresh files', async t => {
  const fixture = await mkdtemp(path.join(tmpdir(), 'quartz-artifact-retention-'))
  const now = Date.now()
  const old = new Date(now - (72 * 60 + 1) * 60_000)
  const fresh = new Date(now - (72 * 60 - 1) * 60_000)
  const future = new Date(now + 60_000)
  const files = [
    { name: 'docs/verification/task/artifacts/expired.png', modified: old },
    { name: 'docs/verification/task/artifacts/mixed/expired.json', modified: old },
    { name: 'docs/verification/task/artifacts/empty-after/trace.json', modified: old },
    { name: 'docs/verification/task/artifacts/mixed/fresh.json', modified: fresh },
    { name: 'docs/verification/task/artifacts/future.json', modified: future },
    { name: 'docs/verification/task/replay.mjs', modified: old },
    { name: 'docs/verification/task/README.md', modified: old },
    { name: 'docs/verification/artifacts/report.json', modified: old },
    { name: 'quartz/.quartz-cache/design-system-v17/preview.html', modified: old },
    { name: 'quartz/.quartz-cache/strava.json', modified: old },
    { name: '.claude/design-system/ui.tsx', modified: old },
    { name: 'outside/expired.json', modified: old },
    { name: 'outside/task/artifacts/expired.json', modified: old },
    { name: 'outside/docs/verification/task/artifacts/expired.json', modified: old },
  ]
  const commands: { args: string[]; stdout: string; stderr: string }[] = []
  const checks: string[] = []

  t.after(async () => {
    await mkdir(evidence, { recursive: true })
    await writeFile(
      path.join(evidence, 'results.json'),
      JSON.stringify(
        {
          command: 'pnpm test quartz/scripts/clean-verification-artifacts.e2e.test.ts',
          fixture,
          files,
          commands,
          checks,
          completed: checks.length === 7,
        },
        null,
        2,
      ) + '\n',
    )
    await rm(fixture, { recursive: true, force: true })
  })

  for (const file of files) {
    const target = path.join(fixture, file.name)
    await mkdir(path.dirname(target), { recursive: true })
    await writeFile(target, file.name)
    await utimes(target, file.modified, file.modified)
  }

  const task = path.join(fixture, 'docs/verification/task')
  const artifacts = path.join(task, 'artifacts')
  await mkdir(path.join(artifacts, 'fresh-empty'))
  await utimes(path.join(artifacts, 'fresh-empty'), fresh, fresh)
  await mkdir(path.join(artifacts, 'old-empty'))
  await utimes(path.join(artifacts, 'old-empty'), old, old)
  await symlink(path.join(fixture, 'outside'), path.join(artifacts, 'linked-directory'))
  await symlink(path.join(fixture, 'outside/expired.json'), path.join(artifacts, 'linked-file'))
  await symlink(
    path.join(fixture, 'outside/task'),
    path.join(fixture, 'docs/verification/linked-task'),
  )
  await mkdir(path.join(fixture, 'docs/verification/linked-artifacts'))
  await symlink(
    path.join(fixture, 'outside'),
    path.join(fixture, 'docs/verification/linked-artifacts/artifacts'),
  )

  async function run(root: string, ...flags: string[]) {
    const args = ['--import', 'tsx', script, '--repo', root, ...flags]
    const result = await execute(process.execPath, args, { cwd: repo, timeout: 10_000 })
    const response: unknown = JSON.parse(result.stdout)
    assert.equal(typeof response, 'object')
    commands.push({ args: [process.execPath, ...args], ...result })
  }

  async function unchanged(name: string) {
    assert.equal(await readFile(path.join(fixture, name), 'utf8'), name)
  }

  await run(fixture, '--dry-run')
  for (const file of files) await unchanged(file.name)
  checks.push('dry-run preserves every file')

  await run(fixture)
  const expired = files.slice(0, 3)
  for (const file of expired) {
    await assert.rejects(lstat(path.join(fixture, file.name)), { code: 'ENOENT' })
  }
  checks.push('files older than 72 hours expire through the real CLI')
  for (const file of files.slice(3)) await unchanged(file.name)
  checks.push('fresh and future files, replay sources, caches, and Claude files survive')
  await assert.rejects(lstat(path.join(artifacts, 'empty-after')), { code: 'ENOENT' })
  await assert.rejects(lstat(path.join(artifacts, 'old-empty')), { code: 'ENOENT' })
  assert.ok((await lstat(path.join(artifacts, 'fresh-empty'))).isDirectory())
  assert.ok((await lstat(path.join(artifacts, 'mixed'))).isDirectory())
  assert.ok((await lstat(artifacts)).isDirectory())
  checks.push('expired empty directories are pruned; fresh, mixed, and artifact roots remain')
  for (const link of [
    path.join(artifacts, 'linked-directory'),
    path.join(artifacts, 'linked-file'),
    path.join(fixture, 'docs/verification/linked-task'),
    path.join(fixture, 'docs/verification/linked-artifacts/artifacts'),
  ])
    assert.ok((await lstat(link)).isSymbolicLink())
  checks.push('task, artifact, directory, and file symlinks are skipped')

  await run(fixture)
  for (const file of files.slice(3)) await unchanged(file.name)
  await run(path.join(fixture, 'missing-repo'))
  checks.push('repeat cleanup and missing verification roots succeed')

  const linkedRoot = path.join(fixture, 'linked-root')
  await mkdir(linkedRoot)
  await symlink(path.join(fixture, 'outside/docs'), path.join(linkedRoot, 'docs'))
  await run(linkedRoot)
  const linkedVerification = path.join(fixture, 'linked-verification')
  await mkdir(path.join(linkedVerification, 'docs'), { recursive: true })
  await symlink(
    path.join(fixture, 'outside/docs/verification'),
    path.join(linkedVerification, 'docs/verification'),
  )
  await run(linkedVerification)
  for (const file of files.slice(3)) await unchanged(file.name)
  checks.push('symlinked docs and verification ancestors are skipped')
})
