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
  notebookPyrightPackageStubsManifestAsset,
  notebookPyrightTypeshedManifestAsset,
  notebookPyrightWorkerManifestAsset,
  notebookRuntimeJavascriptWorkerAsset,
} from '../../../runtime/notebook/assets'
import { isJsonObject } from '../../../util/type-guards'
import {
  basedpyrightInternalSourcePackageName,
  basedpyrightSourcePackageName,
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

const requireResolve = createRequire(import.meta.url).resolve
const notebookPyrightCommonJsEmptyModule = 'module.exports = {}'
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
  const pyrightFiles = await writeNotebookPyrightAssets(ctx)
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
  return [...workerFiles, ...pyrightFiles, ...clientFiles]
}
