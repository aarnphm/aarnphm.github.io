import type { Stats } from 'node:fs'
import { lstat, readdir, rmdir, unlink } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

function hasCode(error: unknown, ...codes: string[]): boolean {
  return error instanceof Error && 'code' in error && codes.includes(String(error.code))
}

async function statIfPresent(target: string): Promise<Stats | undefined> {
  try {
    return await lstat(target)
  } catch (error) {
    if (hasCode(error, 'ENOENT')) return undefined
    throw error
  }
}

async function main() {
  const args = process.argv.slice(2)
  if (args[0] === '--') args.shift()
  const { values } = parseArgs({
    args,
    options: {
      repo: { type: 'string', default: fileURLToPath(new URL('../..', import.meta.url)) },
      'dry-run': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  })
  if (values.help) {
    console.log(
      'Usage: pnpm artifacts:clean [--dry-run] [--repo PATH]\n' +
        'Deletes regular files in docs/verification/<task>/artifacts older than 72 hours.\n' +
        'Uses file modification time, skips symlinks, and prunes empty nested directories.',
    )
    return
  }

  const repo = path.resolve(values.repo)
  const cutoff = Date.now() - 72 * 60 * 60_000
  const expiredFiles: string[] = []
  const removedDirectories: string[] = []
  const skipped: string[] = []
  const report = {
    repo,
    retentionHours: 72,
    cutoff: new Date(cutoff).toISOString(),
    dryRun: values['dry-run'],
    scannedFiles: 0,
    expiredFiles,
    expiredBytes: 0,
    removedDirectories,
    skipped,
  }

  async function directoryStat(target: string): Promise<Stats | undefined> {
    const stat = await statIfPresent(target)
    if (stat?.isSymbolicLink()) report.skipped.push(path.relative(repo, target))
    return stat?.isDirectory() ? stat : undefined
  }

  async function cleanDirectory(directory: string, prune: boolean) {
    const stat = await directoryStat(directory)
    if (!stat) return
    const expiredBefore = report.expiredFiles.length
    const removedBefore = report.removedDirectories.length
    const entries = await readdir(directory)
    for (const name of entries.sort()) {
      const target = path.join(directory, name)
      const stat = await statIfPresent(target)
      if (!stat) continue
      if (stat.isSymbolicLink()) {
        report.skipped.push(path.relative(repo, target))
      } else if (stat.isDirectory()) {
        await cleanDirectory(target, true)
      } else if (stat.isFile()) {
        report.scannedFiles++
        if (stat.mtimeMs >= cutoff) continue
        const current = await statIfPresent(target)
        // Preserve files refreshed or replaced by another verification run during this scan.
        if (
          !current?.isFile() ||
          current.dev !== stat.dev ||
          current.ino !== stat.ino ||
          current.size !== stat.size ||
          current.mtimeMs !== stat.mtimeMs ||
          current.ctimeMs !== stat.ctimeMs
        ) {
          report.skipped.push(path.relative(repo, target))
          continue
        }
        if (!report.dryRun) {
          try {
            await unlink(target)
          } catch (error) {
            if (hasCode(error, 'ENOENT')) continue
            throw error
          }
        }
        report.expiredFiles.push(path.relative(repo, target))
        report.expiredBytes += stat.size
      }
    }
    const expiredContents =
      report.expiredFiles.length > expiredBefore || report.removedDirectories.length > removedBefore
    if (prune && !report.dryRun && (stat.mtimeMs < cutoff || expiredContents)) {
      try {
        await rmdir(directory)
        report.removedDirectories.push(path.relative(repo, directory))
      } catch (error) {
        if (!hasCode(error, 'ENOENT', 'ENOTEMPTY', 'EEXIST')) throw error
      }
    }
  }

  const docs = path.join(repo, 'docs')
  const verification = path.join(docs, 'verification')
  if ((await directoryStat(docs)) && (await directoryStat(verification))) {
    for (const task of (await readdir(verification)).sort()) {
      const taskDirectory = path.join(verification, task)
      if (await directoryStat(taskDirectory)) {
        await cleanDirectory(path.join(taskDirectory, 'artifacts'), false)
      }
    }
  }
  console.log(JSON.stringify(report, null, 2))
}

await main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
