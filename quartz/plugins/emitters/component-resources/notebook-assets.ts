import { build as bundle, type Plugin } from 'esbuild'
import { globby } from 'globby'
import { existsSync } from 'node:fs'
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
  notebookRuntimePyrightPyodideLockUrl,
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
import { writeChunkedFileAsset } from './chunked-file-assets'

type NativeRuntimePackWrittenFile = {
  readonly filename: string
  readonly reference: string
  readonly files: readonly FilePath[]
}

export type NativeRuntimePackGitLfsPointer = { readonly oid: string; readonly size: number }

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
  if (existsSync(sourcePath)) return sourcePath
  for (const ext of ['.ts', '.tsx', '.js', '.json']) {
    const candidate = `${sourcePath}${ext}`
    if (existsSync(candidate)) return candidate
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
  const [response, genericStub] = await Promise.all([
    fetch(notebookRuntimePyrightPyodideLockUrl),
    fs.readFile(notebookRuntimePyrightGenericStubPath, 'utf8'),
  ])
  if (!response.ok) throw new Error(`pyodide package lock request failed with ${response.status}`)
  const value: unknown = await response.json()
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

async function writeNotebookPyrightAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const [workerFiles, typeshedFiles, packageStubFiles] = await Promise.all([
    writeNotebookPyrightWorkerAssets(ctx),
    writeNotebookPyrightTypeshedAssets(ctx),
    writeNotebookPyrightPackageStubAssets(ctx),
  ])
  return [...workerFiles, ...typeshedFiles, ...packageStubFiles]
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

async function readNativeRuntimePackSourceManifest(): Promise<NativeRuntimePackManifest> {
  const source = await fs.readFile(notebookNativeRuntimeManifestSourcePath, 'utf8')
  const value: unknown = JSON.parse(source)
  const manifest = readNativeRuntimePackManifest(value)
  if (!manifest) throw new Error('native runtime pack manifest is invalid')
  return manifest
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
  if (!existsSync(commonDirPath)) return gitDir
  const commonDir = (await fs.readFile(commonDirPath, 'utf8')).trim()
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
  if (!existsSync(objectPath)) {
    throw new Error(
      `native runtime pack asset ${filename} is a Git LFS pointer; run git lfs pull --include=${path.join(
        notebookNativeRuntimePacksDir,
        filename,
      )}`,
    )
  }
  const content = await fs.readFile(objectPath)
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

async function writeNativeRuntimePackAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const sourceManifest = await readNativeRuntimePackSourceManifest()
  const files = await Promise.all(
    nativeRuntimePackFiles(sourceManifest).map(filename =>
      writeNativeRuntimePackFile(ctx, filename),
    ),
  )
  const references = new Map(files.map(file => [file.filename, file.reference]))
  const manifest = rewriteNativeRuntimePackManifest(sourceManifest, references)
  const manifestFile = await writeRawAsset(
    ctx,
    notebookNativeRuntimeManifestPath,
    JSON.stringify(manifest),
  )
  return [manifestFile, ...files.flatMap(file => file.files)]
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
  return [...workerFiles, ...nativeRuntimeFiles, ...pyrightFiles, ...clientFiles]
}
