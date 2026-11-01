import { EventEmitter } from 'node:events'
import { watch, type FSWatcher } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'

type EntryKind = { isFile(): boolean; isDirectory(): boolean }
type FileStats = { size: number; mtimeMs: number }

export type TreeWatcherOptions = {
  /** Emitted paths are relative to `cwd`; without it, relative roots emit relative paths. */
  cwd?: string
  /** Receives absolute paths, with the entry kind once it is known. */
  ignored?: (fp: string, stats?: EntryKind) => boolean
  /** Emit a file once its size and mtime hold for this long. */
  stabilityMs?: number
}

export type TreeWatcher = EventEmitter & { close(): Promise<void> }

const errorCode = (error: unknown) =>
  typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
const missing = (error: unknown) => errorCode(error) === 'ENOENT'
const unreadable = new Set(['ENOENT', 'ENOTDIR', 'EACCES', 'EPERM'])
// FSEvents (macOS) and ReadDirectoryChangesW (Windows) watch a tree natively. Elsewhere Node's
// recursive mode walks the tree and adds one watch per file, ignored subtrees included, so watch
// each kept directory instead: on Linux, one inotify watch per directory also reports its files.
const nativeRecursive = process.platform === 'darwin' || process.platform === 'win32'

/**
 * `add`/`change`/`unlink`/`ready` events (and `error` when startup fails) for files under `roots`.
 * Initial files emit nothing. chokidar keeps one descriptor per watched file; past macOS OPEN_MAX
 * (10240) descriptors, posix_spawn rejects the stdio pipes of every child process with EBADF.
 * Neither watch mode here holds a descriptor per file.
 */
export function watchTree(roots: readonly string[], options: TreeWatcherOptions = {}): TreeWatcher {
  const base = path.resolve(options.cwd ?? process.cwd())
  const stabilityMs = options.stabilityMs ?? 250
  const emitter = new EventEmitter() as TreeWatcher
  // Signature of each file as last seen, so FSEvents replays and duplicates of a write emit nothing.
  const known = new Map<string, string>()
  const lastSignature = new Map<string, string>()
  const timers = new Map<string, NodeJS.Timeout>()
  const handles: FSWatcher[] = []
  const directoryHandles = new Map<string, FSWatcher>()
  const absoluteOutput = new Set<string>()
  let closed = false

  const isIgnored = (abs: string, stats?: EntryKind) => options.ignored?.(abs, stats) ?? false
  const display = (abs: string, root: string) =>
    absoluteOutput.has(root) ? abs : path.relative(base, abs)
  const emitFile = (type: 'add' | 'change' | 'unlink', abs: string, root: string) => {
    if (!closed) emitter.emit(type, display(abs, root))
  }

  // A directory scan prunes ignored directories, so events below one are ignored too.
  const ignoredWithin = (abs: string, root: string) => {
    for (let dir = path.dirname(abs); dir.startsWith(root + path.sep); dir = path.dirname(dir)) {
      if (isIgnored(dir)) return true
    }
    return isIgnored(abs)
  }

  const signatureOf = (stats: FileStats) => `${stats.size}:${stats.mtimeMs}`

  async function scan(
    dir: string,
    root: string,
    found: (abs: string, signature: string) => void,
  ): Promise<void> {
    // Watch before reading, so a file written during the scan still reports.
    if (!nativeRecursive && !closed && !directoryHandles.has(dir))
      directoryHandles.set(dir, listen(dir, root, false))
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(error => {
      if (unreadable.has(errorCode(error) as string)) return []
      throw error
    })
    await Promise.all(
      entries.map(async entry => {
        const abs = path.join(dir, entry.name)
        if (isIgnored(abs, entry.isSymbolicLink() ? undefined : entry)) return
        if (entry.isDirectory()) return await scan(abs, root, found)
        const stats = await fs.stat(abs).catch(() => undefined)
        if (stats?.isFile() && !isIgnored(abs, stats)) found(abs, signatureOf(stats))
      }),
    )
  }

  async function settle(abs: string, root: string): Promise<void> {
    const stats = await fs.stat(abs).catch(error => {
      if (missing(error)) return undefined
      throw error
    })
    if (closed) return
    if (!stats) {
      lastSignature.delete(abs)
      const prefix = abs + path.sep
      for (const [dir, handle] of directoryHandles) {
        if (dir !== abs && !dir.startsWith(prefix)) continue
        handle.close()
        directoryHandles.delete(dir)
      }
      for (const file of known.keys()) {
        if (file !== abs && !file.startsWith(prefix)) continue
        known.delete(file)
        emitFile('unlink', file, root)
      }
      return
    }
    if (isIgnored(abs, stats)) return
    if (stats.isDirectory()) {
      // A directory moved in reports itself, not its files.
      await scan(abs, root, (file, signature) => {
        if (known.has(file)) return
        known.set(file, signature)
        emitFile('add', file, root)
      })
      return
    }
    if (!stats.isFile()) return
    const signature = signatureOf(stats)
    if (lastSignature.get(abs) !== signature) {
      lastSignature.set(abs, signature)
      schedule(abs, root)
      return
    }
    lastSignature.delete(abs)
    const previous = known.get(abs)
    if (previous === signature) return
    known.set(abs, signature)
    emitFile(previous === undefined ? 'add' : 'change', abs, root)
  }

  function schedule(abs: string, root: string): void {
    clearTimeout(timers.get(abs))
    timers.set(
      abs,
      setTimeout(() => {
        timers.delete(abs)
        settle(abs, root).catch(error => console.warn(`Could not inspect ${abs}:`, error))
      }, stabilityMs),
    )
  }

  function listen(dir: string, root: string, recursive: boolean, accept?: Set<string>): FSWatcher {
    const handle = watch(dir, { recursive, persistent: true }, (_event, filename) => {
      if (closed) return
      if (filename === null) {
        schedule(dir, root)
        return
      }
      const name = filename.toString()
      if (accept && !accept.has(name)) return
      const abs = path.join(dir, name)
      if (!ignoredWithin(abs, root)) schedule(abs, root)
    })
    handle.on('error', error => console.warn(`Stopped watching ${dir}:`, error))
    return handle
  }

  async function start(): Promise<void> {
    const directories: string[] = []
    const fileRoots = new Map<string, Set<string>>()
    for (const given of roots) {
      const root = path.resolve(base, given)
      const stats = await fs.stat(root).catch(() => undefined)
      const watched = stats?.isDirectory() ? root : path.dirname(root)
      if (path.isAbsolute(given) && !options.cwd) absoluteOutput.add(watched)
      if (stats?.isDirectory()) {
        directories.push(root)
        continue
      }
      if (!fileRoots.has(watched)) fileRoots.set(watched, new Set())
      fileRoots.get(watched)!.add(path.basename(root))
      if (stats?.isFile()) known.set(root, signatureOf(stats))
    }
    // Each new handle restarts libuv's shared FSEvents stream, which drops events in the gap, so
    // every root handle starts before the scan.
    if (nativeRecursive) for (const root of directories) handles.push(listen(root, root, true))
    for (const [parent, names] of fileRoots) handles.push(listen(parent, parent, false, names))
    await Promise.all(
      directories.map(root => scan(root, root, (abs, signature) => known.set(abs, signature))),
    )
  }

  emitter.close = async () => {
    closed = true
    for (const timer of timers.values()) clearTimeout(timer)
    timers.clear()
    for (const handle of [...handles, ...directoryHandles.values()]) handle.close()
  }
  start().then(
    () => {
      if (!closed) emitter.emit('ready')
    },
    error => emitter.emit('error', error),
  )
  return emitter
}
