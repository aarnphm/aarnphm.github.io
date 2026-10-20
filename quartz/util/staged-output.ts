import { createHash } from 'node:crypto'
import { constants, createReadStream, rmSync, type BigIntStats } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { defaultIoConcurrency, mapConcurrent } from './async-pool'

type SeededFile = { source: BigIntStats; copy: BigIntStats }

declare global {
  var __quartzStageSeedsByRoot: Map<string, Map<string, SeededFile>> | undefined
  var __quartzActiveProcessStages: Set<string> | undefined
  var __quartzStageExitHookRegistered: boolean | undefined
}

const stageSeedsByRoot = (globalThis.__quartzStageSeedsByRoot ??= new Map<
  string,
  Map<string, SeededFile>
>())
const activeProcessStages = (globalThis.__quartzActiveProcessStages ??= new Set<string>())

if (!globalThis.__quartzStageExitHookRegistered) {
  globalThis.__quartzStageExitHookRegistered = true
  process.once('exit', () => {
    for (const stage of activeProcessStages) {
      try {
        rmSync(stage, { recursive: true, force: true })
      } catch {}
    }
  })
}

export async function allSettledOrThrow<T>(promises: readonly Promise<T>[]): Promise<T[]> {
  const results = await Promise.allSettled(promises)
  for (const result of results) {
    if (result.status === 'rejected') throw result.reason
  }
  return results.map(result => (result as PromiseFulfilledResult<T>).value)
}

async function entriesIn(
  root: string,
  preserved: ReadonlySet<string>,
): Promise<{ files: string[]; directories: string[] }> {
  const files: string[] = []
  const directories: string[] = []
  async function walk(dir: string): Promise<void> {
    const entries = await fs.readdir(path.join(root, dir), { withFileTypes: true }).catch(error => {
      if (error.code === 'ENOENT') return []
      throw error
    })
    directories.push(dir)
    const subdirs: string[] = []
    for (const entry of entries) {
      const relative = path.join(dir, entry.name)
      if (preserved.has(relative)) continue
      if (entry.isDirectory()) subdirs.push(relative)
      else if (entry.isFile()) files.push(relative)
      else throw new Error(`Unsupported output entry: ${relative}`)
    }
    if (subdirs.length > 0) await Promise.all(subdirs.map(walk))
  }
  await walk('')
  return { files, directories }
}

async function fileDigest(file: string): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  return hash.digest('hex')
}

async function stageEntries<T>(
  entries: readonly T[],
  action: (entry: T, index: number) => Promise<unknown>,
): Promise<void> {
  const failures: unknown[] = []
  await mapConcurrent(entries, defaultIoConcurrency, async (entry, index) => {
    try {
      await action(entry, index)
    } catch (error) {
      failures.push(error)
    }
  })
  // Cleanup must wait for every operation that can still write into the private stage.
  if (failures.length > 0) throw failures[0]
}

function unchangedFile(before: BigIntStats, after: BigIntStats): boolean {
  return (
    before.ino === after.ino &&
    before.size === after.size &&
    before.mtimeNs === after.mtimeNs &&
    before.ctimeNs === after.ctimeNs
  )
}

async function equalFiles(left: string, right: string, seed?: SeededFile): Promise<boolean> {
  const previous = await fs.stat(right, { bigint: true }).catch(error => {
    if (error.code === 'ENOENT') return undefined
    throw error
  })
  if (!previous) return false
  if (!previous.isFile()) throw new Error(`Output is not a file: ${right}`)
  const next = await fs.stat(left, { bigint: true })
  if (previous.size !== next.size) return false
  // Unmodified private copies still contain the source bytes. Include ctime and inode so writes
  // which restore mtime, and file replacements, fall through to the byte comparison.
  if (seed && unchangedFile(seed.source, previous) && unchangedFile(seed.copy, next)) return true
  if (next.size <= 1024n * 1024n) {
    const [a, b] = await Promise.all([fs.readFile(left), fs.readFile(right)])
    return a.equals(b)
  }
  const [a, b] = await Promise.all([fileDigest(left), fileDigest(right)])
  return a === b
}

function hasErrorCode(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code
}

function stageOwnerExited(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return false
  } catch (error) {
    return hasErrorCode(error, 'ESRCH')
  }
}

/** Private stages are disposable once their owner exits; published output supplies restart state. */
export async function pruneStaleOutputStages(output: string): Promise<string[]> {
  const publicRoot = path.resolve(output)
  const index = path.join(publicRoot, 'index.html')
  const publishedIndex = await fs.stat(index).catch(() => undefined)
  if (!publishedIndex?.isFile() || publishedIndex.size === 0) return []
  const readable = await fs.access(index, constants.R_OK).then(
    () => true,
    () => false,
  )
  if (!readable) return []
  const stagingRoot = path.join(path.dirname(publicRoot), '.quartz-cache')
  const cache = await fs.lstat(stagingRoot).catch((error: unknown) => {
    if (hasErrorCode(error, 'ENOENT')) return undefined
    throw error
  })
  if (!cache?.isDirectory()) return []
  const [cacheRoot, publishedRoot, publishedIndexPath] = await Promise.all([
    fs.realpath(stagingRoot),
    fs.realpath(publicRoot),
    fs.realpath(index),
  ])
  const entries = await fs.readdir(cacheRoot, { withFileTypes: true }).catch((error: unknown) => {
    if (hasErrorCode(error, 'ENOENT')) return []
    throw error
  })
  const prefix = `output-stage-${path.basename(publicRoot)}-`
  const removed: string[] = []
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith(prefix)) continue
    const owner = entry.name.slice(prefix.length)
    if (!/^[1-9]\d*$/.test(owner)) continue
    const pid = Number(owner)
    if (!Number.isSafeInteger(pid) || pid > 2147483647 || pid === process.pid) continue
    if (!stageOwnerExited(pid)) continue
    const stage = path.join(cacheRoot, entry.name)
    if (
      [publishedRoot, publishedIndexPath].some(
        published => published === stage || published.startsWith(`${stage}${path.sep}`),
      )
    )
      continue
    const current = await fs.lstat(stage).catch((error: unknown) => {
      if (hasErrorCode(error, 'ENOENT')) return undefined
      throw error
    })
    if (!current?.isDirectory() || !stageOwnerExited(pid)) continue
    await fs.rm(stage, { recursive: true, force: true })
    removed.push(stage)
  }
  return removed
}

/** Keep public readable while a full watch build works in a private copy. */
export async function withStagedOutput<T>(
  output: string,
  build: (stage: string) => Promise<T>,
  preserve: readonly string[] = [],
  onTiming?: (stage: string, elapsedMs: number) => void,
): Promise<T> {
  const started = performance.now()
  const publicRoot = path.resolve(output)
  // A stable path lets output-keyed runtime bundle caches survive successive rebuilds.
  const stagingRoot = path.join(path.dirname(publicRoot), '.quartz-cache')
  await fs.mkdir(stagingRoot, { recursive: true })
  const stage = path.join(stagingRoot, `output-stage-${path.basename(publicRoot)}-${process.pid}`)
  const preserved = new Set(preserve)
  await fs.mkdir(stage, { recursive: true })
  activeProcessStages.add(stage)
  let backup: string | undefined
  try {
    const existing = await fs.stat(publicRoot).catch(error => {
      if (error.code === 'ENOENT') return undefined
      throw error
    })
    const seeded = new Map<string, SeededFile>()
    if (existing) {
      const [{ files, directories }, stageInventory] = await Promise.all([
        entriesIn(publicRoot, preserved),
        entriesIn(stage, preserved),
      ])
      const previousSeeds = stageSeedsByRoot.get(stage)
      const [sources, stageStats] = await Promise.all([
        mapConcurrent(files, defaultIoConcurrency, file =>
          fs.stat(path.join(publicRoot, file), { bigint: true }),
        ),
        mapConcurrent(stageInventory.files, defaultIoConcurrency, file =>
          fs.stat(path.join(stage, file), { bigint: true }),
        ),
      ])
      const stageStatByFile = new Map(stageInventory.files.map((file, i) => [file, stageStats[i]]))
      const publicFileSet = new Set(files)
      const extraStageFiles = stageInventory.files.filter(file => !publicFileSet.has(file))
      if (extraStageFiles.length > 0) {
        await stageEntries(extraStageFiles, file => fs.rm(path.join(stage, file), { force: true }))
      }
      const stageDirSet = new Set(stageInventory.directories)
      const missingDirectories = directories.filter(dir => dir !== '' && !stageDirSet.has(dir))
      if (missingDirectories.length > 0) {
        await stageEntries(missingDirectories, directory =>
          fs.mkdir(path.join(stage, directory), { recursive: true }),
        )
      }
      await stageEntries(files, async (file, index) => {
        const source = sources[index]
        const existingCopy = stageStatByFile.get(file)
        const cachedSeed = previousSeeds?.get(file)
        if (
          cachedSeed &&
          existingCopy &&
          unchangedFile(cachedSeed.source, source) &&
          unchangedFile(cachedSeed.copy, existingCopy)
        ) {
          seeded.set(file, cachedSeed)
          return
        }
        const target = path.join(stage, file)
        await fs.mkdir(path.dirname(target), { recursive: true })
        await fs.copyFile(path.join(publicRoot, file), target, constants.COPYFILE_FICLONE)
        const copy = await fs.stat(target, { bigint: true })
        seeded.set(file, { source, copy })
      })
      stageSeedsByRoot.set(stage, new Map(seeded))
    } else {
      await fs.rm(stage, { recursive: true, force: true })
      await fs.mkdir(stage, { recursive: true })
      stageSeedsByRoot.delete(stage)
    }
    console.log(`Prepared staged output in ${Math.round(performance.now() - started)}ms`)
    onTiming?.('stage.prepare', performance.now() - started)
    const buildStarted = performance.now()
    const result = await build(stage)
    onTiming?.('stage.build', performance.now() - buildStarted)
    const publishStarted = performance.now()
    const [{ files: next }, { files: previous }] = await Promise.all([
      entriesIn(stage, preserved),
      entriesIn(publicRoot, preserved),
    ])
    const nextSet = new Set(next)
    const removed = previous.filter(file => !nextSet.has(file))
    const changed = (
      await mapConcurrent(next, defaultIoConcurrency, async file =>
        (await equalFiles(path.join(stage, file), path.join(publicRoot, file), seeded.get(file)))
          ? undefined
          : file,
      )
    ).filter(file => file !== undefined)
    console.log(
      `Compared ${next.length} output files in ${Math.round(performance.now() - publishStarted)}ms`,
    )
    onTiming?.('stage.compare', performance.now() - publishStarted)
    const backupStarted = performance.now()
    // Publish assets before the pages which reference them. Temporary files never enter public.
    changed.sort((a, b) => Number(a.endsWith('.html')) - Number(b.endsWith('.html')))
    backup = await fs.mkdtemp(path.join(stagingRoot, 'output-previous-'))
    const previousSet = new Set(previous)
    for (const file of [...changed, ...removed]) {
      if (!previousSet.has(file)) continue
      const saved = path.join(backup, file)
      await fs.mkdir(path.dirname(saved), { recursive: true })
      await fs.link(path.join(publicRoot, file), saved)
    }
    const published: string[] = []
    const deleted: string[] = []
    onTiming?.('stage.backup', performance.now() - backupStarted)
    const replaceStarted = performance.now()
    try {
      for (const file of changed) {
        const target = path.join(publicRoot, file)
        const stageFile = path.join(stage, file)
        const tempTarget = path.join(stagingRoot, `.publish-${process.pid}-${published.length}`)
        await fs.mkdir(path.dirname(target), { recursive: true })
        try {
          await fs.copyFile(stageFile, tempTarget, constants.COPYFILE_FICLONE)
          await fs.rename(tempTarget, target)
        } finally {
          await fs.rm(tempTarget, { force: true })
        }
        published.push(file)
      }
      for (const file of removed) {
        await fs.unlink(path.join(publicRoot, file))
        deleted.push(file)
      }
    } catch (error) {
      for (const file of [...published, ...deleted].reverse()) {
        const target = path.join(publicRoot, file)
        if (previousSet.has(file)) await fs.rename(path.join(backup, file), target)
        else await fs.rm(target, { force: true })
      }
      throw error
    }
    for (const file of removed) {
      seeded.delete(file)
    }
    if (changed.length > 0) {
      await mapConcurrent(changed, defaultIoConcurrency, async file => {
        const [source, copy] = await Promise.all([
          fs.stat(path.join(publicRoot, file), { bigint: true }),
          fs.stat(path.join(stage, file), { bigint: true }),
        ])
        seeded.set(file, { source, copy })
      })
    }
    stageSeedsByRoot.set(stage, seeded)
    console.log(
      `Published ${changed.length} changed files, removed ${removed.length}; kept ${next.length - changed.length} unchanged files in ${Math.round(performance.now() - publishStarted)}ms`,
    )
    onTiming?.('stage.publish', performance.now() - replaceStarted)
    return result
  } finally {
    const cleanupStarted = performance.now()
    if (backup) await fs.rm(backup, { recursive: true, force: true })
    onTiming?.('stage.cleanup', performance.now() - cleanupStarted)
    // Orphan removal must not hold up publication readiness or invalidate restart caches.
    void pruneStaleOutputStages(output).catch(error => {
      console.warn('Could not prune abandoned output stages:', error)
    })
  }
}
