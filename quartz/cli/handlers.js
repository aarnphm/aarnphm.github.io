import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { randomUUID } from 'crypto'
import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { promises } from 'fs'
import { globby } from 'globby'
import http from 'http'
import { inspect, styleText } from 'node:util'
import path from 'path'
import prettyBytes from 'pretty-bytes'
import serveHandler from 'serve-handler'
import { WebSocket, WebSocketServer } from 'ws'
import { version, fp, cacheFile } from './constants.js'

const inlineScriptFilter = /\.inline\.(ts|js)$/
const sourceWatchWriteStabilityMs = 250
export const sourceWatchRoots = ['quartz.config.ts', 'quartz.layout.ts', 'quartz', 'package.json']
export const sourceWatchPatterns = [
  'quartz.config.ts',
  'quartz.layout.ts',
  'quartz/**/*.ts',
  'quartz/**/*.tsx',
  'quartz/**/*.scss',
  'quartz/**/*.py',
  'quartz/cli/*.js',
  'quartz/static/**/*',
  'quartz/extensions/**/*',
  'package.json',
]

const normalizeWatchedPath = fp => {
  const rawPath = fp.toString()
  const relativePath = path.isAbsolute(rawPath) ? path.relative(process.cwd(), rawPath) : rawPath
  return relativePath.split(path.sep).join('/')
}
export const isTestSourcePath = fp => {
  const normalized = normalizeWatchedPath(fp)
  return ['.test.ts', '.test.tsx', '.test.js', '.test.jsx'].some(suffix =>
    normalized.endsWith(suffix),
  )
}

export const isSourceWatchPath = fp => {
  const normalized = normalizeWatchedPath(fp)
  if (isTestSourcePath(normalized)) return false
  if (
    normalized === 'quartz.config.ts' ||
    normalized === 'quartz.layout.ts' ||
    normalized === 'package.json'
  ) {
    return true
  }
  if (!normalized.startsWith('quartz/')) return false
  if (normalized.startsWith('quartz/.quartz-cache/')) return false
  if (normalized.startsWith('quartz/static/') || normalized.startsWith('quartz/extensions/')) {
    return true
  }

  const ext = path.extname(normalized)
  if (ext === '.ts' || ext === '.tsx' || ext === '.scss' || ext === '.py') return true
  return normalized.startsWith('quartz/cli/') && ext === '.js'
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
    normalized.startsWith('quartz/.quartz-cache/')
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

async function bundleInlineScript(scriptPath) {
  let text = await promises.readFile(scriptPath, 'utf8')

  text = text.replace('export default', '')
  text = text.replace('export', '')

  const sourcefile = path.relative(path.resolve('.'), scriptPath)
  const resolveDir = path.dirname(sourcefile)
  const transpiled = await esbuild.build({
    stdin: {
      contents: text,
      loader: path.extname(scriptPath) === '.js' ? 'js' : 'ts',
      resolveDir,
      sourcefile,
    },
    write: false,
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    loader: { '.py': 'text' },
  })
  return transpiled.outputFiles[0].text
}

const createBuildConfig = () => ({
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

const printBundleInfo = async metafile => {
  const outputFileName = cacheFile.replace(/^\.\//, '')
  const meta = metafile.outputs[outputFileName]
  if (meta) {
    console.log(
      `Successfully transpiled ${Object.keys(meta.inputs).length} files (${prettyBytes(
        meta.bytes,
      )})`,
    )
  }
  console.log(await esbuild.analyzeMetafile(metafile, { color: true }))
}

const printCurrentBundleInfo = async () => {
  const result = await esbuild.build({ ...createBuildConfig(), write: false })
  await printBundleInfo(result.metafile)
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
  const ctx = await esbuild.context(createBuildConfig())

  const buildMutex = new Mutex()
  let lastBuildMs = 0
  let activeBuild = null

  const disposeActiveBuild = async () => {
    if (!activeBuild) return
    if (typeof activeBuild === 'function') {
      await activeBuild()
    } else {
      await activeBuild.dispose()
    }
    activeBuild = null
  }

  const rebuildQuartzBundle = async () => {
    const result = await ctx.rebuild().catch(err => {
      console.error(`${styleText('red', "Couldn't parse Quartz configuration:")} ${fp}`)
      console.error(`Reason: ${styleText('gray', formatErrorReason(err))}`)
      process.exit(1)
    })

    if (argv.bundleInfo) {
      await printBundleInfo(result.metafile)
    }

    return import(`../../${cacheFile}?update=${randomUUID()}`)
  }

  const build = async clientRefresh => {
    const buildStart = new Date().getTime()
    lastBuildMs = buildStart
    const release = await buildMutex.acquire()
    if (lastBuildMs > buildStart) {
      release()
      return
    }

    const buildReason = activeBuild ? 'source' : 'initial'
    if (activeBuild) {
      console.log(styleText('yellow', 'Detected a source code change, doing a hard rebuild...'))
      await disposeActiveBuild()
    }

    const { default: buildQuartz } = await rebuildQuartzBundle()
    release()

    activeBuild = await buildQuartz(argv, buildMutex, clientRefresh, buildReason)
    clientRefresh()
  }

  let clientRefresh = () => {}
  if (argv.serve) {
    const connections = new Set()
    clientRefresh = () => {
      for (const conn of connections) {
        if (conn.readyState === WebSocket.OPEN) {
          conn.send('rebuild')
        } else {
          connections.delete(conn)
        }
      }
    }

    if (argv.baseDir !== '' && !argv.baseDir.startsWith('/')) {
      argv.baseDir = '/' + argv.baseDir
    }

    await build(clientRefresh)
    const server = http.createServer(async (req, res) => {
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

      // strip baseDir prefix
      req.url = req.url?.slice(argv.baseDir.length)

      const serve = async () => {
        const release = await buildMutex.acquire()
        let released = false
        const releaseBuild = () => {
          if (released) return
          released = true
          release()
        }
        res.once('finish', releaseBuild)
        res.once('close', releaseBuild)
        try {
          await serveHandler(req, res, {
            public: argv.output,
            directoryListing: false,
            headers: [
              { source: '**/*.*', headers: [{ key: 'Content-Disposition', value: 'inline' }] },
              { source: '**/*.webp', headers: [{ key: 'Content-Type', value: 'image/webp' }] },
              // fixes bug where avif images are displayed as text instead of images (future proof)
              { source: '**/*.avif', headers: [{ key: 'Content-Type', value: 'image/avif' }] },
            ],
          })
          const status = res.statusCode
          const statusString =
            status >= 200 && status < 300
              ? styleText('green', `[${status}]`)
              : styleText('red', `[${status}]`)
          console.log(statusString + styleText('gray', ` ${argv.baseDir}${req.url}`))
        } finally {
          res.off('finish', releaseBuild)
          res.off('close', releaseBuild)
          releaseBuild()
        }
      }

      const redirect = newFp => {
        newFp = argv.baseDir + newFp
        res.writeHead(302, { Location: newFp })
        console.log(
          styleText('yellow', '[302]') +
            styleText('gray', ` ${argv.baseDir}${req.url} -> ${newFp}`),
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

    server.listen(argv.port)
    const wss = new WebSocketServer({ port: argv.wsPort })
    wss.on('connection', ws => {
      connections.add(ws)
      ws.on('close', () => connections.delete(ws))
      ws.on('error', () => connections.delete(ws))
    })
    console.log(
      styleText(
        'cyan',
        `[serve] Started a Quartz server listening at http://localhost:${argv.port}${argv.baseDir}`,
      ),
    )
  } else {
    await build(clientRefresh)
    if (!argv.watch) {
      ctx.dispose()
    }
  }

  if (argv.watch) {
    const sourceChanged = (type, fp) => {
      if (isTestSourcePath(fp)) return
      console.log(styleText('yellow', `Detected source ${type}: ${normalizeWatchedPath(fp)}`))
      return build(clientRefresh)
    }
    chokidar
      .watch(sourceWatchRoots, {
        awaitWriteFinish: { stabilityThreshold: sourceWatchWriteStabilityMs },
        ignoreInitial: true,
        ignored: isIgnoredSourceWatchPath,
      })
      .on('add', fp => sourceChanged('add', fp))
      .on('change', fp => sourceChanged('change', fp))
      .on('unlink', fp => sourceChanged('unlink', fp))

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

export async function handleBundleInfo() {
  console.log('\n' + styleText(['bgGreen', 'black'], `Quartz v${version}`) + '\n')
  await printCurrentBundleInfo()
}
