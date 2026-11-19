import { Mutex } from 'async-mutex'
import { randomUUID } from 'crypto'
import esbuild from 'esbuild'
import { promises } from 'fs'
import { once } from 'node:events'
import { pathToFileURL } from 'node:url'
import { inspect, styleText } from 'node:util'
import path from 'path'
import { emitterSourceInputs, mergeEmitterSourceInputs } from '../../util/source-emission.ts'
import { watchTree } from '../../util/tree-watcher.ts'
import { cacheFile, fp, version } from '../constants.js'
import { createSourceRebuildQueue } from '../source-rebuild-queue.js'
import {
  courseSourceInputs,
  createBuildConfig,
  parseSourceInputs,
  printBundleInfo,
  stravaPayloadSourceInputs,
} from './bundle-info.js'
import { createPreviewServer, createReloadServer } from './preview-server.js'

const testSourcePattern =
  /(?:^|\/)(?:__(?:tests|specs)__|tests?|specs?)(?:\/|$)|(?:^|\/)(?:tests?|specs?|(?:test|spec)_[^/]+|[^/]+_(?:test|spec)|[^/]+\.(?:test|spec))\.[^/]+$/i
const sourceWatchWriteStabilityMs = 250
export const sourceWatchRoots = [
  'quartz.config.ts',
  'quartz.layout.ts',
  'quartz',
  'package.json',
  'pnpm-lock.yaml',
]
const normalizeWatchedPath = watchedFp => {
  const rawPath = watchedFp.toString()
  const relativePath = path.isAbsolute(rawPath) ? path.relative(process.cwd(), rawPath) : rawPath
  return relativePath.split(path.sep).join('/')
}
export const isTestSourcePath = watchedFp => testSourcePattern.test(normalizeWatchedPath(watchedFp))

export const isSourceWatchPath = watchedFp => {
  const normalized = normalizeWatchedPath(watchedFp)
  if (isTestSourcePath(normalized)) return false
  if (
    normalized === 'quartz.config.ts' ||
    normalized === 'quartz.layout.ts' ||
    normalized === 'package.json' ||
    normalized === 'pnpm-lock.yaml'
  ) {
    return true
  }
  if (!normalized.startsWith('quartz/')) return false
  if (normalized.startsWith('quartz/.quartz-cache/')) return false
  return true
}

const isIgnoredSourceWatchPath = (watchedFp, stats) => {
  const normalized = normalizeWatchedPath(watchedFp)
  if (
    normalized === 'node_modules' ||
    normalized.startsWith('node_modules/') ||
    normalized === 'public' ||
    normalized.startsWith('public/') ||
    normalized === '.quartz-cache' ||
    normalized.startsWith('.quartz-cache/') ||
    normalized === 'quartz/.quartz-cache' ||
    normalized.startsWith('quartz/.quartz-cache/') ||
    normalized.split('/').includes('__pycache__')
  ) {
    return true
  }
  return stats?.isFile() ? !isSourceWatchPath(normalized) : false
}

export function formatErrorReason(err) {
  if (typeof err === 'string') {
    return err
  }

  if (err instanceof Error) {
    return err.message.length > 0 ? err.message : String(err)
  }

  return inspect(err, { depth: 4, colors: false })
}

export function createSourceWatcher(onChange) {
  return watchTree(sourceWatchRoots, {
    stabilityMs: sourceWatchWriteStabilityMs,
    ignored: isIgnoredSourceWatchPath,
  })
    .on('add', watchedFp => onChange('add', watchedFp))
    .on('change', watchedFp => onChange('change', watchedFp))
    .on('unlink', watchedFp => onChange('delete', watchedFp))
}

/**
 * Handles `npx quartz build`
 * @param {import("../../util/ctx.ts").Argv} argv arguments for `build`
 */
export async function handleBuild(argv) {
  if (argv.serve) {
    argv.watch = true
  }

  console.log('\n' + styleText(['bgGreen', 'black'], `Quartz v${version}`) + '\n')
  const ctx = await esbuild.context({ ...createBuildConfig(), write: false })

  const buildMutex = new Mutex()
  let activeBuild = null
  // Bundle that ran activeBuild; a hard rebuild releases its module caches.
  let activeBundle = null
  // Modules the build bundle evaluates, from the latest and the previous metafile, so an edit that
  // drops a module from the closure still counts as a server change.
  let serverSourceInputs = new Set()
  let previousServerSourceInputs = new Set()
  let courseInputs = new Set()
  let emissionInputs = new Map()
  let previousEmissionInputs = new Map()

  const disposeActiveBuild = async () => {
    if (!activeBuild) return false
    const hadPending = Boolean(await activeBuild.dispose())
    activeBuild = null
    return hadPending
  }

  const rebuildQuartzBundle = async () => {
    try {
      const result = await ctx.rebuild()
      for (const output of result.outputFiles) {
        const unchanged = await promises.readFile(output.path).then(
          previous => previous.equals(output.contents),
          error => {
            if (error.code === 'ENOENT') return false
            throw error
          },
        )
        if (unchanged) continue
        await promises.mkdir(path.dirname(output.path), { recursive: true })
        await promises.writeFile(output.path, output.contents)
      }
      const parseInputs = parseSourceInputs(result.metafile)
      globalThis.__quartzParseSourceInputs = parseInputs
      const parseSignatures = await Promise.all(
        parseInputs.map(async inputFp => {
          try {
            const stat = await promises.stat(inputFp, { bigint: true })
            return `${inputFp}:${stat.mtimeNs}:${stat.ctimeNs}:${stat.size}:${stat.ino}`
          } catch {
            return `${inputFp}:missing`
          }
        }),
      )
      globalThis.__quartzParseSourceSalt = parseSignatures.join('|')
      globalThis.__quartzBuildSourceInputs = Object.keys(result.metafile.inputs)
      globalThis.__quartzStravaPayloadSourceInputs = stravaPayloadSourceInputs(result.metafile)
      previousServerSourceInputs = serverSourceInputs
      serverSourceInputs = new Set(Object.keys(result.metafile.inputs))
      courseInputs = courseSourceInputs(result.metafile)
      previousEmissionInputs = emissionInputs
      emissionInputs = emitterSourceInputs(result.metafile)
      if (argv.bundleInfo) {
        await printBundleInfo(result.metafile)
      }
      if (!argv.watch) {
        await ctx.dispose()
      }
      const moduleUrl = `${pathToFileURL(path.resolve(cacheFile)).href}?update=${randomUUID()}`
      return await import(moduleUrl)
    } catch (err) {
      console.error(`${styleText('red', "Couldn't parse Quartz configuration:")} ${fp}`)
      console.error(`Reason: ${styleText('gray', formatErrorReason(err))}`)
      if (!argv.watch) {
        process.exitCode = 1
        return null
      }
      console.log(styleText('yellow', 'Waiting for the next source change...'))
      return null
    }
  }

  const build = async (clientRefresh, takePaths = () => []) => {
    const release = await buildMutex.acquire()
    try {
      const buildReason = activeBuild ? 'source' : 'initial'
      const bundle = await rebuildQuartzBundle()
      // Take paths after the bundle rebuild so edits during that rebuild join this reset.
      if (!bundle) {
        takePaths()
        return false
      }
      const changedPaths = takePaths()

      const deletedSource = (
        await Promise.all(
          changedPaths.map(filename =>
            promises.access(filename).then(
              () => false,
              () => true,
            ),
          ),
        )
      ).some(Boolean)
      const scope =
        activeBuild?.ready && !deletedSource
          ? bundle.sourceRebuildScope(
              changedPaths,
              new Set([...previousServerSourceInputs, ...serverSourceInputs]),
              courseInputs,
              mergeEmitterSourceInputs(previousEmissionInputs, emissionInputs),
            )
          : { kind: 'full' }
      if (scope.kind === 'partial') {
        release()
        const ready = await bundle.buildSourceResources(argv, buildMutex, changedPaths, scope)
        bundle.releaseModuleCaches()
        if (ready) clientRefresh()
        return ready
      }

      const courses = scope.kind === 'courses'
      let hadPendingContent = false
      if (activeBuild) {
        hadPendingContent = await disposeActiveBuild()
        console.log(
          styleText(
            'yellow',
            courses && !hadPendingContent
              ? 'Detected a course source change, rebuilding the courses...'
              : scope.kind === 'emitters' && !hadPendingContent
                ? `Detected an emission source change, rebuilding ${scope.emitters.join(', ')}...`
                : 'Detected a source code change, doing a hard rebuild...',
          ),
        )
      }
      if (hadPendingContent) {
        globalThis.__quartzSourceEmissionSnapshot = undefined
      }
      activeBundle?.releaseModuleCaches()
      activeBundle = bundle
      release()

      activeBuild = await bundle.default(
        argv,
        buildMutex,
        clientRefresh,
        buildReason,
        hadPendingContent
          ? 'site'
          : courses
            ? 'courses'
            : scope.kind === 'emitters'
              ? { emitters: scope.emitters }
              : 'site',
      )
      if (activeBuild?.ready) clientRefresh()
      return argv.watch ? Boolean(activeBuild?.ready) : true
    } finally {
      release()
    }
  }

  const reload = argv.watch ? createReloadServer(argv.wsPort) : undefined
  const clientRefresh = () => reload?.refresh()
  let initialBuild = () => build(clientRefresh)
  if (argv.watch) {
    const sourceRebuildQueue = createSourceRebuildQueue({
      build: takePaths =>
        build(() => {
          if (sourceRebuildQueue.pending === 0) clientRefresh()
        }, takePaths),
      onError: error => console.error(`Source rebuild failed: ${formatErrorReason(error)}`),
      onReceipt: receipt => console.log(`__QUARTZ_SOURCE_QUEUE__${JSON.stringify(receipt)}`),
    })
    const sourceChanged = (type, watchedFp) => {
      if (isTestSourcePath(watchedFp)) return
      const normalized = normalizeWatchedPath(watchedFp)
      console.log(styleText('yellow', `Detected source ${type}: ${normalized}`))
      void sourceRebuildQueue.add(normalized)
    }
    const sourceWatcher = createSourceWatcher(sourceChanged)
    await once(sourceWatcher, 'ready')
    initialBuild = () => sourceRebuildQueue.start()
  }

  try {
    const ok = await initialBuild()
    if (!ok && !argv.watch) {
      process.exitCode = 1
      return
    }
  } finally {
    if (!argv.watch) {
      // Dispose the build context and stop esbuild's --ping service before Node tears down stdio
      // pipes on exit; abrupt stdin closure races esbuild's ping goroutine into a Go deadlock.
      await ctx.dispose().catch(() => {})
      await esbuild.stop()
    }
  }

  if (argv.serve) {
    if (argv.baseDir !== '' && !argv.baseDir.startsWith('/')) {
      argv.baseDir = '/' + argv.baseDir
    }

    const server = createPreviewServer(argv, buildMutex)

    server.listen(argv.port)
    console.log(
      styleText(
        'cyan',
        `[serve] Started a Quartz server listening at http://localhost:${argv.port}${argv.baseDir}`,
      ),
    )
  }

  if (argv.watch) {
    console.log(styleText('gray', 'hint: exit with ctrl+c'))
  }
}
