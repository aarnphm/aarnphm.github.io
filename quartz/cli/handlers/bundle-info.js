import esbuild from 'esbuild'
import { sassPlugin } from 'esbuild-sass-plugin'
import { promises } from 'fs'
import { globby } from 'globby'
import { styleText } from 'node:util'
import path from 'path'
import prettyBytes from 'pretty-bytes'
import { bundleInlineScript } from '../../util/inline-script-bundler.ts'
import { cacheFile, fp, version } from '../constants.js'

const inlineScriptFilter = /\.inline\.(ts|js)$/

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
    const inputFp = queue.pop()
    if (closure.has(inputFp)) continue
    closure.add(inputFp)
    for (const imported of inputs[inputFp]?.imports ?? []) {
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
    inputFp =>
      parseSourceSeedPrefixes.some(prefix => inputFp === prefix || inputFp.startsWith(prefix)),
    new Set(['quartz.config.ts']),
  ).filter(inputFp => !nonParseAssetPattern.test(inputFp))

export const stravaPayloadSourceInputs = metafile =>
  sourceInputClosure(metafile, inputFp => inputFp === 'quartz/util/strava-payload.ts')

// Course page components: they render only pages under `content/courses/`.
const courseSourceRoot = /^quartz\/components\/(?:Courses?[A-Z]\w*\.tsx$|course-[^/]+\/)/

/**
 * Server modules whose edits change only course pages: a course root, or a module that only course
 * modules import.
 */
export const courseSourceInputs = metafile => {
  const inputs = metafile.inputs ?? {}
  const importers = new Map()
  for (const [inputFp, input] of Object.entries(inputs)) {
    for (const imported of input.imports ?? []) {
      if (imported.external || !inputs[imported.path]) continue
      if (!importers.has(imported.path)) importers.set(imported.path, new Set())
      importers.get(imported.path).add(inputFp)
    }
  }
  const scoped = new Map()
  const isCourseInput = inputFp => {
    if (courseSourceRoot.test(inputFp)) return true
    // A cycle back to a module in progress proves nothing, so it counts against the scope.
    if (scoped.has(inputFp)) return scoped.get(inputFp) === true
    scoped.set(inputFp, 'pending')
    const from = importers.get(inputFp)
    const result = from !== undefined && [...from].every(isCourseInput)
    scoped.set(inputFp, result)
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

export const printBundleInfo = async (metafile, format = 'table') => {
  if (format === 'json') {
    console.log(formatBundleInfoJson(metafile))
    return
  }

  console.log(await formatBundleInfoTable(metafile))
}

const printCurrentBundleInfo = async (format = 'table') => {
  try {
    const result = await esbuild.build({ ...createBuildConfig(), write: false })
    await printBundleInfo(result.metafile, format)
  } finally {
    await esbuild.stop()
  }
}

const printBundleInfoHeader = () => {
  console.log('\n' + styleText(['bgGreen', 'black'], `Quartz v${version}`) + '\n')
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
