import { build as bundle, type Plugin } from 'esbuild'
import { globby } from 'globby'
import { existsSync as fileExistsSync } from 'node:fs'
import fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'path'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import type { ChunkedFileAssetDescriptor } from './chunked-file-assets'
import {
  chunkNotebookPyrightTypeshedFiles,
  notebookPyrightPackageStubFiles,
  notebookPyrightPyodidePackageImports,
  notebookPyrightTypeshedChunkBytes,
} from '../../../runtime/lsp/pyright-assets'
import {
  isNativeRuntimeLanguage,
  isNativeRuntimePackAvailableEntry,
  readNativeRuntimePackManifest,
  type NativeRuntimePackEntry,
  type NativeRuntimePackManifest,
} from '../../../runtime/native/runtime-pack'
import {
  notebookNativeRuntimeManifestAsset,
  notebookPyrightPackageStubsManifestAsset,
  notebookPyrightTypeshedManifestAsset,
  notebookPyrightWorkerManifestAsset,
  notebookRuntimeJavascriptWorkerAsset,
} from '../../../runtime/notebook/assets'
import { ensureAssetManifest, registerAsset, shouldHashAssets } from '../../../util/asset-manifest'
import { recordOutputAsset } from '../../../util/output-assets'
import { isJsonObject } from '../../../util/type-guards'
import {
  basedpyrightInternalSourcePackageName,
  basedpyrightSourcePackageName,
  notebookNativeRuntimeManifestSourcePath,
  notebookNativeRuntimeManifestPath,
  notebookNativeRuntimePacksDir,
  notebookPyrightPackageStubsBaseDir,
  notebookPyrightPackageStubsManifestPath,
  notebookPyrightTypeshedBaseDir,
  notebookPyrightTypeshedManifestPath,
  notebookPyrightWorkerManifestPath,
  notebookPyrightWorkerOutputName,
  notebookRuntimeClientEntry,
  notebookRuntimeJavascriptWorkerEntry,
  notebookRuntimeJavascriptWorkerPath,
  notebookRuntimePyrightGenericStubPath,
  notebookRuntimePyrightPackageStubsDir,
  notebookRuntimePyrightPyodideLockPath,
  notebookRuntimePyrightSitePackagesPath,
  notebookRuntimePyrightTypeshedDir,
  notebookRuntimePyrightWorkerEntry,
  notebookRuntimePyrightWorkerGlobalsEntry,
  notebookRuntimeWorkerEntry,
  notebookRuntimeWorkerPath,
  staticScriptsDir,
} from './asset-paths'
import {
  relativeAssetReference,
  relativeBundleAssetReference,
  resolveAssetPath,
  staticScriptAssetReference,
  writeAssetBundleOutput,
  writeRawAsset,
} from './asset-writer'
import {
  chunkedFileAssetChunkPath,
  chunkedFileAssetManifestPath,
  writeChunkedFileAsset,
} from './chunked-file-assets'

type NativeRuntimePackWrittenFile = {
  readonly filename: string
  readonly reference: string
  readonly files: readonly FilePath[]
}

export type NativeRuntimePackGitLfsPointer = { readonly oid: string; readonly size: number }

type CachedRuntimeAssetBundle = {
  readonly signature: string
  readonly files: readonly FilePath[]
  readonly assetEntries: readonly (readonly [string, string])[]
}

type NativeRuntimePackFileSpec = {
  readonly filename: string
  readonly mtimeMs: number
  readonly size: number
  readonly resolvedSize: number
}

declare global {
  var __quartzNativeRuntimePackAssetCache: Map<string, CachedRuntimeAssetBundle> | undefined
  var __quartzNotebookPyrightAssetCache: Map<string, CachedRuntimeAssetBundle> | undefined
}

const requireResolve = createRequire(import.meta.url).resolve
const notebookPyrightCommonJsEmptyModule = 'module.exports = {}'
export const nativeRuntimePackChunkBytes = 20 * 1024 * 1024
const nativeRuntimePackGitLfsPointerHeader = 'version https://git-lfs.github.com/spec/v1'
const notebookPyrightDisabledNodeModulePattern =
  /^(?:node:)?(?:child_process|crypto|fs|module|net|os|perf_hooks|stream|tls|v8|worker_threads)$/

const notebookPyrightTypeshedDescriptor: ChunkedFileAssetDescriptor = {
  baseDir: notebookPyrightTypeshedBaseDir,
  manifestName: 'manifest.json',
  chunkDir: 'chunks',
  maxBytes: notebookPyrightTypeshedChunkBytes,
}

const notebookPyrightPackageStubsDescriptor: ChunkedFileAssetDescriptor = {
  baseDir: notebookPyrightPackageStubsBaseDir,
  manifestName: 'manifest.json',
  chunkDir: 'chunks',
  maxBytes: notebookPyrightTypeshedChunkBytes,
}

const notebookPyrightLodashModule = `
function isObject(value) {
  return typeof value === "object" && value !== null
}

export function add(left, right) {
  return left + right
}

export function zip(...arrays) {
  const length = arrays.reduce((max, array) => Math.max(max, array?.length ?? 0), 0)
  return Array.from({ length }, (_, index) => arrays.map(array => array?.[index]))
}

export function isEqual(left, right) {
  if (Object.is(left, right)) return true
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
    return left.every((value, index) => isEqual(value, right[index]))
  }
  if (isObject(left) || isObject(right)) {
    if (!isObject(left) || !isObject(right)) return false
    const leftKeys = Object.keys(left)
    const rightKeys = Object.keys(right)
    if (leftKeys.length !== rightKeys.length) return false
    return leftKeys.every(key => Object.prototype.hasOwnProperty.call(right, key) && isEqual(left[key], right[key]))
  }
  return false
}
`

function packageDir(packageName: string): string {
  return path.dirname(requireResolve(`${packageName}/package.json`))
}

function packageNodeModules(packageName: string): string {
  return path.join(packageDir(packageName), 'node_modules')
}

function basedpyrightInternalSourceDir(): string {
  return path.join(packageDir(basedpyrightSourcePackageName), 'packages/pyright-internal/src')
}

function pyrightInternalSourcePath(importPath: string): string {
  const sourcePath = path.join(
    basedpyrightInternalSourceDir(),
    importPath.slice('pyright-internal/'.length),
  )
  if (fileExistsSync(sourcePath)) return sourcePath
  for (const ext of ['.ts', '.tsx', '.js', '.json']) {
    const candidate = `${sourcePath}${ext}`
    if (fileExistsSync(candidate)) return candidate
  }
  return sourcePath
}

function pyrightBrowserDependencyPaths(): string[] {
  return [packageNodeModules(basedpyrightInternalSourcePackageName), path.resolve('node_modules')]
}

function notebookPyrightWorkerPlugin(): Plugin {
  return {
    name: 'notebook-pyright-worker',
    setup(build) {
      build.onResolve({ filter: /^pyright-internal\// }, args => ({
        path: pyrightInternalSourcePath(args.path),
      }))
      build.onResolve({ filter: /^typeshed-json$/ }, () => ({
        namespace: 'notebook-pyright-empty',
        path: 'typeshed-json',
      }))
      build.onResolve({ filter: /^is-ci$/ }, () => ({
        namespace: 'notebook-pyright-is-ci',
        path: 'is-ci',
      }))
      build.onResolve({ filter: /^lodash$/ }, () => ({
        namespace: 'notebook-pyright-lodash',
        path: 'lodash',
      }))
      build.onResolve({ filter: /^(?:node:)?path$/ }, () => ({
        path: requireResolve('path-browserify'),
      }))
      build.onResolve({ filter: /^(?:node:)?buffer$/ }, () => ({ path: requireResolve('buffer/') }))
      build.onResolve({ filter: notebookPyrightDisabledNodeModulePattern }, args => ({
        namespace: 'notebook-pyright-empty',
        path: args.path,
      }))
      build.onLoad({ filter: /.*/, namespace: 'notebook-pyright-empty' }, () => ({
        contents: notebookPyrightCommonJsEmptyModule,
        loader: 'js',
      }))
      build.onLoad({ filter: /.*/, namespace: 'notebook-pyright-is-ci' }, () => ({
        contents: 'export default false',
        loader: 'js',
      }))
      build.onLoad({ filter: /.*/, namespace: 'notebook-pyright-lodash' }, () => ({
        contents: notebookPyrightLodashModule,
        loader: 'js',
      }))
    },
  }
}

function notebookPyrightWorkerEntryOutput(outputFiles: readonly { path: string; text: string }[]) {
  const entryName = `${notebookPyrightWorkerOutputName}.js`
  const entry = outputFiles.find(output => path.basename(output.path) === entryName)
  if (!entry) throw new Error('notebook pyright worker entry was not emitted')
  return entry
}

async function notebookPyrightTypeshedFiles(): Promise<Record<string, string>> {
  const entries = (
    await globby('**/*', { cwd: notebookRuntimePyrightTypeshedDir, onlyFiles: true, dot: true })
  ).sort()
  const files: Record<string, string> = {}
  for (const entry of entries) {
    const source = await fs.readFile(path.join(notebookRuntimePyrightTypeshedDir, entry), 'utf8')
    files[`/typeshed/stdlib/${entry.split(path.sep).join('/')}`] = source
  }
  return files
}

async function notebookPyrightLocalPackageStubFiles(): Promise<Record<string, string>> {
  const entries = (
    await globby('**/*.pyi', {
      cwd: notebookRuntimePyrightPackageStubsDir,
      onlyFiles: true,
      dot: true,
    })
  ).sort()
  const files: Record<string, string> = {}
  for (const entry of entries) {
    const source = await fs.readFile(
      path.join(notebookRuntimePyrightPackageStubsDir, entry),
      'utf8',
    )
    files[`${notebookRuntimePyrightSitePackagesPath}/${entry.split(path.sep).join('/')}`] = source
  }
  return files
}

async function notebookPyrightPyodidePackageStubFiles(): Promise<Record<string, string>> {
  const [lockSource, genericStub] = await Promise.all([
    fs.readFile(notebookRuntimePyrightPyodideLockPath, 'utf8'),
    fs.readFile(notebookRuntimePyrightGenericStubPath, 'utf8'),
  ])
  const value: unknown = JSON.parse(lockSource)
  if (!isJsonObject(value)) throw new Error('pyodide package lock is not a JSON object')
  return notebookPyrightPackageStubFiles(
    notebookRuntimePyrightSitePackagesPath,
    notebookPyrightPyodidePackageImports(value),
    genericStub,
  )
}

async function notebookPyrightPackageStubAssetFiles(): Promise<Record<string, string>> {
  const [pyodideFiles, localFiles] = await Promise.all([
    notebookPyrightPyodidePackageStubFiles(),
    notebookPyrightLocalPackageStubFiles(),
  ])
  return { ...pyodideFiles, ...localFiles }
}

async function writeNotebookPyrightTypeshedAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const chunks = chunkNotebookPyrightTypeshedFiles(
    await notebookPyrightTypeshedFiles(),
    notebookPyrightTypeshedDescriptor.maxBytes,
  )
  return writeChunkedFileAsset(
    ctx,
    notebookPyrightTypeshedDescriptor,
    chunks.map(chunk => JSON.stringify({ files: chunk.files })),
  )
}

async function writeNotebookPyrightPackageStubAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const chunks = chunkNotebookPyrightTypeshedFiles(
    await notebookPyrightPackageStubAssetFiles(),
    notebookPyrightPackageStubsDescriptor.maxBytes,
  )
  return writeChunkedFileAsset(
    ctx,
    notebookPyrightPackageStubsDescriptor,
    chunks.map(chunk => JSON.stringify({ files: chunk.files })),
  )
}

async function writeNotebookPyrightWorkerAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, staticScriptsDir)
  const worker = await bundle({
    entryPoints: { [notebookPyrightWorkerOutputName]: notebookRuntimePyrightWorkerEntry },
    bundle: true,
    minifyWhitespace: true,
    minifyIdentifiers: true,
    minifySyntax: false,
    target: 'es2022',
    supported: { 'template-literal': false },
    tsconfigRaw: { compilerOptions: { experimentalDecorators: true } },
    define: { __dirname: "'/'", __filename: "'/pyright-worker.js'" },
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    conditions: ['browser'],
    inject: [notebookRuntimePyrightWorkerGlobalsEntry],
    legalComments: 'eof',
    nodePaths: pyrightBrowserDependencyPaths(),
    plugins: [notebookPyrightWorkerPlugin()],
    write: false,
  })
  const entryOutput = notebookPyrightWorkerEntryOutput(worker.outputFiles)
  const workerFiles = await Promise.all(
    worker.outputFiles.map(output => writeAssetBundleOutput(ctx, output)),
  )
  const entryLogicalPath = path
    .relative(ctx.argv.output, entryOutput.path)
    .split(path.sep)
    .join('/')
  const manifest = {
    entry: relativeAssetReference(
      notebookPyrightWorkerManifestPath,
      resolveAssetPath(ctx, entryLogicalPath),
    ),
  }
  const manifestFile = await writeRawAsset(
    ctx,
    notebookPyrightWorkerManifestPath,
    JSON.stringify(manifest),
  )
  return [manifestFile, ...workerFiles]
}

const notebookPyrightWorkerLogicalPath = `${staticScriptsDir}/${notebookPyrightWorkerOutputName}.js`
const nativeRuntimePacksLogicalPrefix = `${path.posix.dirname(notebookNativeRuntimeManifestPath)}/`
const notebookPyrightLogicalPrefix = `${staticScriptsDir}/notebook-pyright/`

function isNotebookPyrightLogicalPath(logicalPath: string): boolean {
  return (
    logicalPath === notebookPyrightWorkerLogicalPath ||
    logicalPath.startsWith(notebookPyrightLogicalPrefix)
  )
}

function isNativeRuntimePackLogicalPath(logicalPath: string): boolean {
  return (
    logicalPath === notebookNativeRuntimeManifestPath ||
    logicalPath.startsWith(nativeRuntimePacksLogicalPrefix)
  )
}

function runtimeAssetCacheKey(ctx: BuildCtx): string {
  return `${path.resolve(ctx.argv.output)}\0${shouldHashAssets(ctx)}`
}

async function allFilesExistOnDisk(files: readonly FilePath[]): Promise<boolean> {
  if (files.length === 0) return false
  try {
    const stats = await Promise.all(files.map(file => fs.stat(file)))
    return stats.every(stat => stat.isFile() && stat.size > 0)
  } catch {
    return false
  }
}

function applyCachedRuntimeAssetBundle(
  ctx: BuildCtx,
  cached: CachedRuntimeAssetBundle,
): FilePath[] {
  for (const [logicalPath, emittedPath] of cached.assetEntries) {
    registerAsset(ctx, logicalPath, emittedPath)
  }
  recordOutputAsset('notebook-runtime', cached.files)
  return [...cached.files]
}

export function resetNotebookRuntimeAssetCache(): void {
  globalThis.__quartzNativeRuntimePackAssetCache?.clear()
  globalThis.__quartzNotebookPyrightAssetCache?.clear()
}

async function notebookPyrightSourceSignature(
  ctx: BuildCtx,
): Promise<{ signature: string; maxMtimeMs: number }> {
  const sourcePaths = [
    notebookRuntimePyrightWorkerEntry,
    notebookRuntimePyrightWorkerGlobalsEntry,
    notebookRuntimePyrightPyodideLockPath,
    notebookRuntimePyrightGenericStubPath,
  ]
  const stats = await Promise.all(sourcePaths.map(file => fs.stat(file)))
  let maxMtimeMs = 0
  const parts = [path.resolve(ctx.argv.output), String(shouldHashAssets(ctx))]
  for (let i = 0; i < sourcePaths.length; i += 1) {
    const stat = stats[i]
    if (stat.mtimeMs > maxMtimeMs) maxMtimeMs = stat.mtimeMs
    parts.push(`${sourcePaths[i]}:${stat.mtimeMs}:${stat.size}`)
  }
  return { signature: parts.join('\0'), maxMtimeMs }
}

async function tryRecoverUnhashedPyrightAssetBundle(
  ctx: BuildCtx,
  signature: string,
  maxMtimeMs: number,
): Promise<CachedRuntimeAssetBundle | undefined> {
  if (shouldHashAssets(ctx)) return undefined
  try {
    const outputRoot = path.resolve(ctx.argv.output)
    const workerManifestLogical = chunkedFileAssetManifestPath({
      baseDir: `${staticScriptsDir}/notebook-pyright/worker`,
      manifestName: 'manifest.json',
      chunkDir: 'chunks',
      maxBytes: 0,
    })
    const typeshedManifestLogical = chunkedFileAssetManifestPath(notebookPyrightTypeshedDescriptor)
    const stubsManifestLogical = chunkedFileAssetManifestPath(notebookPyrightPackageStubsDescriptor)
    const workerManifestFile = path.join(outputRoot, workerManifestLogical) as FilePath
    const typeshedManifestFile = path.join(outputRoot, typeshedManifestLogical) as FilePath
    const stubsManifestFile = path.join(outputRoot, stubsManifestLogical) as FilePath
    const workerEntryFile = path.join(outputRoot, notebookPyrightWorkerLogicalPath) as FilePath

    const [workerManifestStat, typeshedStat, stubsStat, workerEntryStat] = await Promise.all([
      fs.stat(workerManifestFile),
      fs.stat(typeshedManifestFile),
      fs.stat(stubsManifestFile),
      fs.stat(workerEntryFile),
    ])
    for (const stat of [workerManifestStat, typeshedStat, stubsStat, workerEntryStat]) {
      if (!stat.isFile() || stat.size === 0 || stat.mtimeMs < maxMtimeMs) return undefined
    }

    const [typeshedRaw, stubsRaw] = await Promise.all([
      fs.readFile(typeshedManifestFile, 'utf8'),
      fs.readFile(stubsManifestFile, 'utf8'),
    ])
    const typeshedParsed: unknown = JSON.parse(typeshedRaw)
    const stubsParsed: unknown = JSON.parse(stubsRaw)
    if (
      !isJsonObject(typeshedParsed) ||
      !Array.isArray(typeshedParsed.chunks) ||
      !isJsonObject(stubsParsed) ||
      !Array.isArray(stubsParsed.chunks)
    ) {
      return undefined
    }

    const assetEntries: Array<readonly [string, string]> = [
      [notebookPyrightWorkerLogicalPath, notebookPyrightWorkerLogicalPath],
      [notebookPyrightWorkerManifestPath, notebookPyrightWorkerManifestPath],
    ]
    const files: FilePath[] = [workerManifestFile, workerEntryFile]

    for (let index = 0; index < typeshedParsed.chunks.length; index += 1) {
      const logical = chunkedFileAssetChunkPath(notebookPyrightTypeshedDescriptor, index)
      assetEntries.push([logical, logical])
      files.push(path.join(outputRoot, logical) as FilePath)
    }
    assetEntries.push([typeshedManifestLogical, typeshedManifestLogical])
    files.push(typeshedManifestFile)

    for (let index = 0; index < stubsParsed.chunks.length; index += 1) {
      const logical = chunkedFileAssetChunkPath(notebookPyrightPackageStubsDescriptor, index)
      assetEntries.push([logical, logical])
      files.push(path.join(outputRoot, logical) as FilePath)
    }
    assetEntries.push([stubsManifestLogical, stubsManifestLogical])
    files.push(stubsManifestFile)

    if (!(await allFilesExistOnDisk(files))) return undefined
    return { signature, files, assetEntries }
  } catch {
    return undefined
  }
}

export async function writeNotebookPyrightAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const cache = (globalThis.__quartzNotebookPyrightAssetCache ??= new Map())
  const key = runtimeAssetCacheKey(ctx)
  const { signature, maxMtimeMs } = await notebookPyrightSourceSignature(ctx)
  const cached = cache.get(key)
  if (cached && cached.signature === signature && (await allFilesExistOnDisk(cached.files))) {
    return applyCachedRuntimeAssetBundle(ctx, cached)
  }

  const recovered = await tryRecoverUnhashedPyrightAssetBundle(ctx, signature, maxMtimeMs)
  if (recovered) {
    cache.set(key, recovered)
    return applyCachedRuntimeAssetBundle(ctx, recovered)
  }

  const [workerFiles, typeshedFiles, packageStubFiles] = await Promise.all([
    writeNotebookPyrightWorkerAssets(ctx),
    writeNotebookPyrightTypeshedAssets(ctx),
    writeNotebookPyrightPackageStubAssets(ctx),
  ])
  const files = [...workerFiles, ...typeshedFiles, ...packageStubFiles]
  const assetEntries = [...ensureAssetManifest(ctx).entries()].filter(([logicalPath]) =>
    isNotebookPyrightLogicalPath(logicalPath),
  )
  cache.set(key, { signature, files, assetEntries })
  recordOutputAsset('notebook-runtime', files)
  return files
}

function normalizeNativeRuntimePackPath(value: string): string {
  const normalized = value.split(path.sep).join('/')
  if (normalized.length === 0 || path.posix.isAbsolute(normalized)) {
    throw new Error(`native runtime pack asset path is invalid: ${value}`)
  }
  if (normalized.split('/').includes('..')) {
    throw new Error(`native runtime pack asset path escapes its pack: ${value}`)
  }
  return normalized
}

async function readNativeRuntimePackSourceManifest(): Promise<{
  raw: string
  manifest: NativeRuntimePackManifest
}> {
  const raw = await fs.readFile(notebookNativeRuntimeManifestSourcePath, 'utf8')
  const value: unknown = JSON.parse(raw)
  const manifest = readNativeRuntimePackManifest(value)
  if (!manifest) throw new Error('native runtime pack manifest is invalid')
  return { raw, manifest }
}

function nativeRuntimePackFiles(manifest: NativeRuntimePackManifest): readonly string[] {
  const files = new Set<string>()
  for (const entry of Object.values(manifest.runtimes)) {
    if (!entry || !isNativeRuntimePackAvailableEntry(entry)) continue
    files.add(normalizeNativeRuntimePackPath(entry.worker))
    for (const asset of entry.assets) files.add(normalizeNativeRuntimePackPath(asset))
  }
  return Array.from(files).sort()
}

async function inspectNativeRuntimePackFileSpec(
  filename: string,
): Promise<NativeRuntimePackFileSpec> {
  const sourcePath = path.join(notebookNativeRuntimePacksDir, filename)
  const stat = await fs.stat(sourcePath)
  if (stat.size <= 1024) {
    const content = await fs.readFile(sourcePath)
    const pointer = readNativeRuntimePackGitLfsPointer(content, filename)
    if (pointer) {
      return { filename, mtimeMs: stat.mtimeMs, size: stat.size, resolvedSize: pointer.size }
    }
  }
  return { filename, mtimeMs: stat.mtimeMs, size: stat.size, resolvedSize: stat.size }
}

async function nativeRuntimePackSignatureAndSpecs(
  ctx: BuildCtx,
): Promise<{
  sourceManifest: NativeRuntimePackManifest
  specs: readonly NativeRuntimePackFileSpec[]
  signature: string
}> {
  const { raw, manifest: sourceManifest } = await readNativeRuntimePackSourceManifest()
  const filenames = nativeRuntimePackFiles(sourceManifest)
  const specs = await Promise.all(filenames.map(inspectNativeRuntimePackFileSpec))
  const signature = [
    path.resolve(ctx.argv.output),
    String(shouldHashAssets(ctx)),
    raw,
    ...specs.map(spec => `${spec.filename}:${spec.mtimeMs}:${spec.size}:${spec.resolvedSize}`),
  ].join('\0')
  return { sourceManifest, specs, signature }
}

async function tryRecoverUnhashedNativeRuntimePackBundle(
  ctx: BuildCtx,
  specs: readonly NativeRuntimePackFileSpec[],
  signature: string,
): Promise<CachedRuntimeAssetBundle | undefined> {
  if (shouldHashAssets(ctx)) return undefined
  try {
    const outputRoot = path.resolve(ctx.argv.output)
    const manifestFile = path.join(outputRoot, notebookNativeRuntimeManifestPath) as FilePath
    const manifestStat = await fs.stat(manifestFile)
    if (!manifestStat.isFile() || manifestStat.size === 0) return undefined

    const files: FilePath[] = [manifestFile]
    const assetEntries: Array<readonly [string, string]> = []
    const baseDir = path.posix.dirname(notebookNativeRuntimeManifestPath)

    for (const spec of specs) {
      const logicalPath = `${baseDir}/${spec.filename}`
      if (spec.resolvedSize > nativeRuntimePackChunkBytes) {
        const chunkCount = Math.ceil(spec.resolvedSize / nativeRuntimePackChunkBytes)
        const manifestLogicalPath = `${logicalPath}.chunks.json`
        const chunkManifestFile = path.join(outputRoot, manifestLogicalPath) as FilePath
        const chunkManifestStat = await fs.stat(chunkManifestFile)
        if (!chunkManifestStat.isFile() || chunkManifestStat.size === 0) return undefined
        files.push(chunkManifestFile)
        for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
          const index = chunkIndex.toString().padStart(4, '0')
          const chunkLogicalPath = `${logicalPath}.chunk-${index}.bin`
          const expectedSize =
            chunkIndex === chunkCount - 1
              ? spec.resolvedSize - chunkIndex * nativeRuntimePackChunkBytes
              : nativeRuntimePackChunkBytes
          const chunkFile = path.join(outputRoot, chunkLogicalPath) as FilePath
          const chunkStat = await fs.stat(chunkFile)
          if (!chunkStat.isFile() || chunkStat.size !== expectedSize) return undefined
          assetEntries.push([chunkLogicalPath, chunkLogicalPath])
          files.push(chunkFile)
        }
        assetEntries.push([manifestLogicalPath, manifestLogicalPath])
      } else {
        const packFile = path.join(outputRoot, logicalPath) as FilePath
        const packStat = await fs.stat(packFile)
        if (!packStat.isFile() || packStat.size !== spec.resolvedSize) return undefined
        assetEntries.push([logicalPath, logicalPath])
        files.push(packFile)
      }
    }

    assetEntries.push([notebookNativeRuntimeManifestPath, notebookNativeRuntimeManifestPath])
    return { signature, files, assetEntries }
  } catch {
    return undefined
  }
}

async function writeNativeRuntimePackFile(
  ctx: BuildCtx,
  filename: string,
): Promise<NativeRuntimePackWrittenFile> {
  const logicalPath = `${path.posix.dirname(notebookNativeRuntimeManifestPath)}/${filename}`
  const content = await readNativeRuntimePackFile(filename)
  if (content.byteLength > nativeRuntimePackChunkBytes) {
    return writeChunkedNativeRuntimePackFile(ctx, filename, logicalPath, content)
  }
  const file = await writeRawAsset(ctx, logicalPath, content)
  return {
    filename,
    reference: relativeAssetReference(
      notebookNativeRuntimeManifestPath,
      resolveAssetPath(ctx, logicalPath),
    ),
    files: [file],
  }
}

export function readNativeRuntimePackGitLfsPointer(
  content: Buffer,
  filename: string,
): NativeRuntimePackGitLfsPointer | undefined {
  if (content.byteLength > 1024) return undefined
  const source = content.toString('utf8').trimEnd()
  if (!source.startsWith(nativeRuntimePackGitLfsPointerHeader)) return undefined
  const oid = /^oid sha256:([0-9a-f]{64})$/m.exec(source)?.[1]
  const rawSize = /^size ([0-9]+)$/m.exec(source)?.[1]
  const size = rawSize === undefined ? Number.NaN : Number(rawSize)
  if (!oid || !Number.isSafeInteger(size)) {
    throw new Error(`native runtime pack asset ${filename} has an invalid Git LFS pointer`)
  }
  return { oid, size }
}

async function readNativeRuntimeGitDir(): Promise<string> {
  const gitPath = path.resolve('.git')
  const stat = await fs.stat(gitPath)
  if (stat.isDirectory()) return gitPath
  const source = await fs.readFile(gitPath, 'utf8')
  const gitDir = /^gitdir: (.+)$/m.exec(source)?.[1]
  if (!gitDir) throw new Error('native runtime pack Git directory is invalid')
  return path.resolve(path.dirname(gitPath), gitDir)
}

async function readNativeRuntimeGitCommonDir(): Promise<string> {
  const gitDir = await readNativeRuntimeGitDir()
  const commonDirPath = path.join(gitDir, 'commondir')
  let commonDir: string
  try {
    commonDir = (await fs.readFile(commonDirPath, 'utf8')).trim()
  } catch {
    return gitDir
  }
  return path.resolve(gitDir, commonDir)
}

async function readNativeRuntimePackGitLfsObject(
  filename: string,
  pointer: NativeRuntimePackGitLfsPointer,
): Promise<Buffer> {
  const commonDir = await readNativeRuntimeGitCommonDir()
  const objectPath = path.join(
    commonDir,
    'lfs/objects',
    pointer.oid.slice(0, 2),
    pointer.oid.slice(2, 4),
    pointer.oid,
  )
  let content: Buffer
  try {
    content = await fs.readFile(objectPath)
  } catch {
    throw new Error(
      `native runtime pack asset ${filename} is a Git LFS pointer; run git lfs pull --include=${path.join(
        notebookNativeRuntimePacksDir,
        filename,
      )}`,
    )
  }
  if (content.byteLength !== pointer.size) {
    throw new Error(`native runtime pack asset ${filename} Git LFS object size is invalid`)
  }
  return content
}

async function readNativeRuntimePackFile(filename: string): Promise<Buffer> {
  const sourcePath = path.join(notebookNativeRuntimePacksDir, filename)
  const content = await fs.readFile(sourcePath)
  const pointer = readNativeRuntimePackGitLfsPointer(content, filename)
  return pointer ? await readNativeRuntimePackGitLfsObject(filename, pointer) : content
}

export function nativeRuntimePackFileChunks(
  content: Buffer,
  maxBytes = nativeRuntimePackChunkBytes,
): readonly Buffer[] {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) {
    throw new Error('native runtime pack chunk size is invalid')
  }
  const chunks: Buffer[] = []
  for (let offset = 0; offset < content.byteLength; offset += maxBytes) {
    chunks.push(content.subarray(offset, offset + maxBytes))
  }
  return chunks
}

async function writeChunkedNativeRuntimePackFile(
  ctx: BuildCtx,
  filename: string,
  logicalPath: string,
  content: Buffer,
): Promise<NativeRuntimePackWrittenFile> {
  const chunks = nativeRuntimePackFileChunks(content)
  const chunkLogicalPaths = chunks.map((_chunk, chunkIndex) => {
    const index = chunkIndex.toString().padStart(4, '0')
    return `${logicalPath}.chunk-${index}.bin`
  })
  const chunkFiles = await Promise.all(
    chunks.map((chunk, chunkIndex) => writeRawAsset(ctx, chunkLogicalPaths[chunkIndex], chunk)),
  )
  const manifestLogicalPath = `${logicalPath}.chunks.json`
  const manifest = {
    version: 1,
    size: content.byteLength,
    chunks: chunkLogicalPaths.map(chunkPath =>
      relativeAssetReference(manifestLogicalPath, resolveAssetPath(ctx, chunkPath)),
    ),
  }
  const manifestFile = await writeRawAsset(ctx, manifestLogicalPath, JSON.stringify(manifest))
  return {
    filename,
    reference: relativeAssetReference(
      notebookNativeRuntimeManifestPath,
      resolveAssetPath(ctx, manifestLogicalPath),
    ),
    files: [manifestFile, ...chunkFiles],
  }
}

function runtimePackAssetReference(references: Map<string, string>, filename: string): string {
  const reference = references.get(normalizeNativeRuntimePackPath(filename))
  if (!reference) throw new Error(`native runtime pack asset is missing: ${filename}`)
  return reference
}

function rewriteNativeRuntimePackEntry(
  references: Map<string, string>,
  entry: NativeRuntimePackEntry,
): NativeRuntimePackEntry {
  if (!isNativeRuntimePackAvailableEntry(entry)) return entry
  return {
    worker: runtimePackAssetReference(references, entry.worker),
    assets: entry.assets.map(asset => runtimePackAssetReference(references, asset)),
  }
}

function rewriteNativeRuntimePackManifest(
  manifest: NativeRuntimePackManifest,
  references: Map<string, string>,
): NativeRuntimePackManifest {
  const runtimes: NativeRuntimePackManifest['runtimes'] = {}
  for (const [language, entry] of Object.entries(manifest.runtimes)) {
    if (!isNativeRuntimeLanguage(language) || !entry) continue
    runtimes[language] = rewriteNativeRuntimePackEntry(references, entry)
  }
  return { version: 1, runtimes }
}

export async function writeNativeRuntimePackAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const cache = (globalThis.__quartzNativeRuntimePackAssetCache ??= new Map())
  const key = runtimeAssetCacheKey(ctx)
  const { sourceManifest, specs, signature } = await nativeRuntimePackSignatureAndSpecs(ctx)
  const cached = cache.get(key)
  if (cached && cached.signature === signature && (await allFilesExistOnDisk(cached.files))) {
    return applyCachedRuntimeAssetBundle(ctx, cached)
  }

  const recovered = await tryRecoverUnhashedNativeRuntimePackBundle(ctx, specs, signature)
  if (recovered) {
    cache.set(key, recovered)
    return applyCachedRuntimeAssetBundle(ctx, recovered)
  }

  const files = await Promise.all(specs.map(spec => writeNativeRuntimePackFile(ctx, spec.filename)))
  const references = new Map(files.map(file => [file.filename, file.reference]))
  const manifest = rewriteNativeRuntimePackManifest(sourceManifest, references)
  const manifestFile = await writeRawAsset(
    ctx,
    notebookNativeRuntimeManifestPath,
    JSON.stringify(manifest),
  )
  const emittedFiles = [manifestFile, ...files.flatMap(file => file.files)]
  const assetEntries = [...ensureAssetManifest(ctx).entries()].filter(([logicalPath]) =>
    isNativeRuntimePackLogicalPath(logicalPath),
  )
  cache.set(key, { signature, files: emittedFiles, assetEntries })
  recordOutputAsset('notebook-runtime', emittedFiles)
  return emittedFiles
}

function replaceNotebookRuntimeWorkerReference(text: string, fromFile: string, workerPath: string) {
  const placeholder = '\0quartz-notebook-runtime-worker\0'
  return text
    .replaceAll('../notebook-runtime.worker.js', placeholder)
    .replaceAll('notebook-runtime.worker.js', placeholder)
    .replaceAll(placeholder, relativeBundleAssetReference(fromFile, workerPath))
}

export async function writeNotebookRuntimeAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, staticScriptsDir)
  const [worker, javascriptWorker] = await Promise.all([
    bundle({
      entryPoints: [notebookRuntimeWorkerEntry],
      bundle: true,
      minify: true,
      platform: 'browser',
      format: 'esm',
      outfile: path.join(outdir, 'notebook-runtime.worker.js'),
      define: { 'globalThis.process': 'undefined' },
      external: ['fs'],
      loader: { '.py': 'text' },
      write: false,
    }),
    bundle({
      entryPoints: [notebookRuntimeJavascriptWorkerEntry],
      bundle: true,
      minify: true,
      platform: 'browser',
      format: 'esm',
      outfile: path.join(outdir, 'notebook-runtime.javascript.worker.js'),
      write: false,
    }),
  ])
  const workerFiles = await Promise.all(
    [...worker.outputFiles, ...javascriptWorker.outputFiles].map(output =>
      writeAssetBundleOutput(ctx, output),
    ),
  )
  const [nativeRuntimeFiles, pyrightFiles] = await Promise.all([
    writeNativeRuntimePackAssets(ctx),
    writeNotebookPyrightAssets(ctx),
  ])
  const client = await bundle({
    entryPoints: { 'notebook-runtime.client': notebookRuntimeClientEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    loader: { '.html': 'text' },
    write: false,
  })
  const workerPath = path.join(outdir, staticScriptAssetReference(ctx, notebookRuntimeWorkerPath))
  const clientOutputs = client.outputFiles.map(output => ({
    ...output,
    text: replaceNotebookRuntimeWorkerReference(output.text, output.path, workerPath)
      .replaceAll(
        notebookRuntimeJavascriptWorkerAsset,
        staticScriptAssetReference(ctx, notebookRuntimeJavascriptWorkerPath),
      )
      .replaceAll(
        notebookNativeRuntimeManifestAsset,
        staticScriptAssetReference(ctx, notebookNativeRuntimeManifestPath),
      )
      .replaceAll(
        notebookPyrightWorkerManifestAsset,
        staticScriptAssetReference(ctx, notebookPyrightWorkerManifestPath),
      )
      .replaceAll(
        notebookPyrightTypeshedManifestAsset,
        staticScriptAssetReference(ctx, notebookPyrightTypeshedManifestPath),
      )
      .replaceAll(
        notebookPyrightPackageStubsManifestAsset,
        staticScriptAssetReference(ctx, notebookPyrightPackageStubsManifestPath),
      ),
  }))
  const clientFiles = await Promise.all(
    clientOutputs.map(output => writeAssetBundleOutput(ctx, output)),
  )
  const allFiles = [...workerFiles, ...nativeRuntimeFiles, ...pyrightFiles, ...clientFiles]
  recordOutputAsset('notebook-runtime', allFiles)
  return allFiles
}
