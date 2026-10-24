import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { randomUUID } from 'crypto'
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { constants, promises } from 'fs'
import { globby } from 'globby'
import http from 'http'
import { once } from 'node:events'
import { inspect, styleText } from 'node:util'
import path from 'path'
import prettyBytes from 'pretty-bytes'
import serveHandler from 'serve-handler'
import { WebSocket, WebSocketServer } from 'ws'
import { emitterSourceInputs, mergeEmitterSourceInputs } from '../util/source-emission.ts'
export { emitterSourceInputs } from '../util/source-emission.ts'
import { bundleInlineScript } from '../util/inline-script-bundler.ts'
import { handleLocalLeanVerifyNodeRequest } from '../util/lean-local-verify.ts'
import { version, fp, cacheFile } from './constants.js'
import { createSourceRebuildQueue } from './source-rebuild-queue.js'

const inlineScriptFilter = /\.inline\.(ts|js)$/
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
const normalizeWatchedPath = fp => {
  const rawPath = fp.toString()
  const relativePath = path.isAbsolute(rawPath) ? path.relative(process.cwd(), rawPath) : rawPath
  return relativePath.split(path.sep).join('/')
}
export const isTestSourcePath = fp => testSourcePattern.test(normalizeWatchedPath(fp))

export const isSourceWatchPath = fp => {
  const normalized = normalizeWatchedPath(fp)
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

const isIgnoredSourceWatchPath = (fp, stats) => {
  const normalized = normalizeWatchedPath(fp)
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

export const createBuildConfig = () => ({
  entryPoints: [fp],
  outfile: cacheFile,
  bundle: true,
  keepNames: true,
  minifyWhitespace: true,
  minifySyntax: true,
  platform: 'node',
  format: 'esm',
  jsx: 'automatic',
  jsxImportSource: 'preact',
  packages: 'external',
  loader: { '.py': 'text' },
  metafile: true,
  sourcemap: true,
  sourcesContent: false,
  plugins: [
    sassPlugin({ type: 'css-text', cssImports: true }),
    sassPlugin({ filter: /\.inline\.scss$/, type: 'css', cssImports: true }),
    {
      name: 'inline-script-loader',
      setup(build) {
        build.onLoad({ filter: inlineScriptFilter }, async args => {
          return { contents: await bundleInlineScript(args.path), loader: 'text' }
        })
      },
    },
  ],
})

const parseSourceSeedPrefixes = [
  'quartz/processors/parse.ts',
  'quartz/extensions/',
  'quartz/plugins/transformers/',
  'quartz/components/mdx/',
]

const sourceInputClosure = (metafile, isSeed, closure = new Set()) => {
  const inputs = metafile.inputs ?? {}
  const queue = Object.keys(inputs).filter(isSeed)
  while (queue.length > 0) {
    const fp = queue.pop()
    if (closure.has(fp)) continue
    closure.add(fp)
    for (const imported of inputs[fp]?.imports ?? []) {
      if (imported.external || !inputs[imported.path]) continue
      queue.push(imported.path)
    }
  }
  return [...closure].sort()
}

const nonParseAssetPattern = /\.(?:scss|css|inline\.[jt]s)$/

export const parseSourceInputs = metafile =>
  sourceInputClosure(
    metafile,
    fp => parseSourceSeedPrefixes.some(prefix => fp === prefix || fp.startsWith(prefix)),
    new Set(['quartz.config.ts']),
  ).filter(fp => !nonParseAssetPattern.test(fp))

export const stravaPayloadSourceInputs = metafile =>
  sourceInputClosure(metafile, fp => fp === 'quartz/util/strava-payload.ts')

// Course page components: they render only pages under `content/courses/`.
const courseSourceRoot = /^quartz\/components\/(?:Courses?[A-Z]\w*\.tsx$|course-[^/]+\/)/

/**
 * Server modules whose edits change only course pages: a course root, or a module that only course
 * modules import.
 */
export const courseSourceInputs = metafile => {
  const inputs = metafile.inputs ?? {}
  const importers = new Map()
  for (const [fp, input] of Object.entries(inputs)) {
    for (const imported of input.imports ?? []) {
      if (imported.external || !inputs[imported.path]) continue
      if (!importers.has(imported.path)) importers.set(imported.path, new Set())
      importers.get(imported.path).add(fp)
    }
  }
  const scoped = new Map()
  const isCourseInput = fp => {
    if (courseSourceRoot.test(fp)) return true
    // A cycle back to a module in progress proves nothing, so it counts against the scope.
    if (scoped.has(fp)) return scoped.get(fp) === true
    scoped.set(fp, 'pending')
    const from = importers.get(fp)
    const result = from !== undefined && [...from].every(isCourseInput)
    scoped.set(fp, result)
    return result
  }
  return new Set(Object.keys(inputs).filter(isCourseInput))
}

const bundleInfoOutputFile = () => cacheFile.replace(/^\.\//, '')

export const bundleInfoSummary = metafile => {
  const outputFileName = bundleInfoOutputFile()
  const meta = metafile.outputs[outputFileName]
  return {
    outputFile: outputFileName,
    inputCount: meta ? Object.keys(meta.inputs).length : 0,
    bytes: meta?.bytes ?? 0,
    bytesText: prettyBytes(meta?.bytes ?? 0),
  }
}

const useBundleInfoColor = () =>
  process.stdout.isTTY && process.env.TERM !== 'dumb' && process.env.NO_COLOR === undefined

export const resolveBundleInfoFormat = argv =>
  argv?.json || argv?.format === 'json' ? 'json' : 'table'

export const formatBundleInfoJson = metafile =>
  JSON.stringify({ version, summary: bundleInfoSummary(metafile), metafile }, null, 2)

export const formatBundleInfoTable = async (metafile, color = useBundleInfoColor()) => {
  const summary = bundleInfoSummary(metafile)
  const lines = []
  if (summary.inputCount > 0) {
    lines.push(
      `Successfully transpiled ${summary.inputCount} files (${prettyBytes(summary.bytes)})`,
    )
  }
  lines.push(await esbuild.analyzeMetafile(metafile, { color }))
  return lines.join('\n')
}

const printBundleInfo = async (metafile, format = 'table') => {
  if (format === 'json') {
    console.log(formatBundleInfoJson(metafile))
    return
  }

  console.log(await formatBundleInfoTable(metafile))
}

const printCurrentBundleInfo = async (format = 'table') => {
  const result = await esbuild.build({ ...createBuildConfig(), write: false })
  await printBundleInfo(result.metafile, format)
}

const printBundleInfoHeader = () => {
  console.log('\n' + styleText(['bgGreen', 'black'], `Quartz v${version}`) + '\n')
}

export function createSourceWatcher(onChange) {
  return chokidar
    .watch(sourceWatchRoots, {
      awaitWriteFinish: { stabilityThreshold: sourceWatchWriteStabilityMs },
      ignoreInitial: true,
      ignored: isIgnoredSourceWatchPath,
    })
    .on('add', fp => onChange('add', fp))
    .on('change', fp => onChange('change', fp))
    .on('unlink', fp => onChange('delete', fp))
}

export function createReloadServer(port) {
  const server = new WebSocketServer({ port })
  server.on('connection', socket => {
    socket.on('error', () => socket.terminate())
  })
  return {
    server,
    refresh() {
      for (const socket of server.clients) {
        if (socket.readyState === WebSocket.OPEN) socket.send('rebuild')
      }
    },
  }
}

export function createPreviewServer(argv, buildMutex) {
  return http.createServer(async (req, res) => {
    if (argv.baseDir && !req.url?.startsWith(argv.baseDir)) {
      console.log(
        styleText(
          'red',
          `[404] ${req.url} (warning: link outside of site, this is likely a Quartz bug)`,
        ),
      )
      res.writeHead(404)
      res.end()
      return
    }

    // Watch builds stage full output and replace individual files atomically. Requests can keep
    // reading the last successful output while the build mutex is held.
    if (!argv.watch) {
      const release = await buildMutex.acquire()
      if (res.destroyed) {
        release()
        return
      }
      res.once('finish', release)
      res.once('close', release)
    }

    // strip baseDir prefix
    req.url = req.url?.slice(argv.baseDir.length)

    const serve = async () => {
      const snapshots = new Map()
      const handles = new Map()
      let closed = false
      const closeReads = () => {
        if (closed) return
        closed = true
        void Promise.allSettled(
          [...snapshots.values()].map(pending =>
            pending.then(snapshot => snapshot.handle?.close()),
          ),
        )
      }
      res.once('finish', closeReads)
      res.once('close', closeReads)
      const snapshot = fp => {
        let pending = snapshots.get(fp)
        if (!pending) {
          pending = (async () => {
            const info = await promises.lstat(fp)
            if (!info.isFile()) return { info }
            if (closed) throw new Error('Preview response closed')
            const handle = await promises.open(fp, constants.O_RDONLY | constants.O_NOFOLLOW)
            try {
              const info = await handle.stat()
              handles.set(fp, handle)
              return { info, handle }
            } catch (error) {
              await handle.close()
              throw error
            }
          })()
          snapshots.set(fp, pending)
        }
        return pending
      }
      // Pin metadata and bytes to the same open inode while publication replaces its pathname.
      const reads = {
        async lstat(fp) {
          return (await snapshot(fp)).info
        },
        createReadStream(fp, options) {
          const handle = handles.get(fp)
          if (!handle) throw new Error(`Missing preview read snapshot: ${fp}`)
          const stream = handle.createReadStream({ start: 0, ...options })
          res.once('close', () => stream.destroy())
          return stream
        },
      }
      await serveHandler(
        req,
        res,
        {
          public: argv.output,
          directoryListing: false,
          headers: [
            { source: '**/*.*', headers: [{ key: 'Content-Disposition', value: 'inline' }] },
            { source: '**/*.webp', headers: [{ key: 'Content-Type', value: 'image/webp' }] },
            // fixes bug where avif images are displayed as text instead of images (future proof)
            { source: '**/*.avif', headers: [{ key: 'Content-Type', value: 'image/avif' }] },
          ],
        },
        reads,
      )
      const status = res.statusCode
      const statusString =
        status >= 200 && status < 300
          ? styleText('green', `[${status}]`)
          : styleText('red', `[${status}]`)
      console.log(statusString + styleText('gray', ` ${argv.baseDir}${req.url}`))
    }

    const redirect = newFp => {
      newFp = argv.baseDir + newFp
      res.writeHead(302, { Location: newFp })
      console.log(
        styleText('yellow', '[302]') + styleText('gray', ` ${argv.baseDir}${req.url} -> ${newFp}`),
      )
      res.end()
    }

    const outputRoot = path.resolve(argv.output)
    const normalizeRequestPath = value => {
      try {
        const parsed = new URL(value ?? '/', 'http://quartz.local')
        const decoded = decodeURIComponent(parsed.pathname)
        if (decoded.includes('\0')) return '/'
        return path.posix.normalize(`/${decoded}`)
      } catch {
        return '/'
      }
    }
    const resolveOutputPath = publicPath => {
      const resolved = path.resolve(outputRoot, `.${publicPath}`)
      if (resolved === outputRoot || resolved.startsWith(`${outputRoot}${path.sep}`)) {
        return resolved
      }
    }
    const existsInOutput = async publicPath => {
      const resolved = resolveOutputPath(publicPath)
      if (resolved === undefined) return false
      try {
        await promises.access(resolved)
        return true
      } catch {
        return false
      }
    }

    let fp = normalizeRequestPath(req.url)

    if (fp === '/api/lean/verify') {
      await handleLocalLeanVerifyNodeRequest(req, res)
      return
    }

    // handle redirects
    if (fp.endsWith('/')) {
      // /trailing/
      // does /trailing/index.html exist? if so, serve it
      const indexFp = path.posix.join(fp, 'index.html')
      if (await existsInOutput(indexFp)) {
        req.url = fp
        return serve()
      }

      // does /trailing.html exist? if so, redirect to /trailing
      let base = fp.slice(0, -1)
      if (path.extname(base) === '') {
        base += '.html'
      }
      if (await existsInOutput(base)) {
        return redirect(fp.slice(0, -1))
      }
    } else {
      // /regular
      // does /regular.html exist? if so, serve it
      let base = fp
      if (path.extname(base) === '') {
        base += '.html'
      }
      if (await existsInOutput(base)) {
        req.url = fp
        return serve()
      }

      // does /regular/index.html exist? if so, redirect to /regular/
      let indexFp = path.posix.join(fp, 'index.html')
      if (await existsInOutput(indexFp)) {
        return redirect(fp + '/')
      }
    }

    return serve()
  })
}

/**
 * Handles `npx quartz build`
 * @param {import("../util/ctx.ts").Argv} argv arguments for `build`
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
      return await import(`../../${cacheFile}?update=${randomUUID()}`)
    } catch (err) {
      console.log(`${styleText('red', "Couldn't parse Quartz configuration:")} ${fp}`)
      console.log(`Reason: ${styleText('gray', formatErrorReason(err))}`)
      if (!argv.watch) process.exit(1)
      console.log(styleText('yellow', 'Waiting for the next source change...'))
      return null
    }
  }

  const build = async (clientRefresh, takePaths = () => []) => {
    const release = await buildMutex.acquire()
    try {
      const changedPaths = takePaths()

      const buildReason = activeBuild ? 'source' : 'initial'
      const bundle = await rebuildQuartzBundle()
      if (!bundle) return false

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
      return Boolean(activeBuild?.ready)
    } finally {
      release()
    }
  }

  const reload = argv.watch ? createReloadServer(argv.wsPort) : undefined
  const clientRefresh = () => reload?.refresh()
  let initialBuild = () => build(clientRefresh)
  if (argv.watch) {
    const sourceRebuildQueue = createSourceRebuildQueue({
      build: takePaths => build(clientRefresh, takePaths),
      onError: error => console.error(`Source rebuild failed: ${formatErrorReason(error)}`),
      onReceipt: receipt => console.log(`__QUARTZ_SOURCE_QUEUE__${JSON.stringify(receipt)}`),
    })
    const sourceChanged = (type, fp) => {
      if (isTestSourcePath(fp)) return
      const normalized = normalizeWatchedPath(fp)
      console.log(styleText('yellow', `Detected source ${type}: ${normalized}`))
      void sourceRebuildQueue.add(normalized)
    }
    const sourceWatcher = createSourceWatcher(sourceChanged)
    await once(sourceWatcher, 'ready')
    initialBuild = () => sourceRebuildQueue.start()
  }
  await initialBuild()

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
  } else if (!argv.watch) {
    ctx.dispose()
  }

  if (argv.watch) {
    console.log(styleText('gray', 'hint: exit with ctrl+c'))
  }
}

export async function handleStats(argv) {
  console.log('\n' + styleText(['bgGreen', 'black'], `Quartz v${version}`) + '\n')
  const absDir = path.resolve(argv.directory)
  const files = await globby(['**/*'], { cwd: absDir, dot: true, onlyFiles: true })
  const fileStats = await Promise.all(
    files.map(async file => {
      const stat = await promises.stat(path.join(absDir, file))
      return { file, size: stat.size }
    }),
  )

  let totalBytes = 0
  let mdBytes = 0
  let mdFiles = 0
  let largestFile = null
  let largestBytes = 0

  for (const { file, size } of fileStats) {
    totalBytes += size
    if (path.extname(file).toLowerCase() === '.md') {
      mdFiles += 1
      mdBytes += size
    }
    if (size > largestBytes) {
      largestBytes = size
      largestFile = file
    }
  }

  console.log(styleText('cyan', 'Vault stats'))
  console.log(`Path: ${absDir}`)
  console.log(`Files: ${files.length} (${mdFiles} markdown, ${files.length - mdFiles} other)`)
  console.log(`Size: ${prettyBytes(totalBytes)} (${totalBytes} bytes)`)
  console.log(`Markdown: ${prettyBytes(mdBytes)} (${mdBytes} bytes)`)
  if (largestFile) {
    console.log(`Largest: ${largestFile} (${prettyBytes(largestBytes)})`)
  }
  console.log('')

  console.log(styleText('cyan', 'Bundle info'))
  await printCurrentBundleInfo()
}

export async function handleBundleInfo(argv) {
  const format = resolveBundleInfoFormat(argv)
  if (format === 'table') {
    printBundleInfoHeader()
  }
  await printCurrentBundleInfo(format)
}
