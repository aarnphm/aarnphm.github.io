import { transform as transpile, build as bundle, type Plugin } from 'esbuild'
import { globby } from 'globby'
import { Features, transform } from 'lightningcss'
import { existsSync } from 'node:fs'
import fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'path'
import type { QuartzMdxComponent } from '../../components/mdx/registry'
import type { StaticResources } from '../../util/resources'
import { getMdxComponents } from '../../components/mdx/registry'
// @ts-ignore
import notFoundScript from '../../components/scripts/404.inline'
//@ts-ignore
import audioScript from '../../components/scripts/audio.inline'
// @ts-ignore
import baseMapScript from '../../components/scripts/base-map.inline'
// @ts-ignore
import pseudoScript from '../../components/scripts/clipboard-pseudo.inline'
// @ts-ignore
import clipboardScript from '../../components/scripts/clipboard.inline'
// @ts-ignore
import collaborativeCommentsScript from '../../components/scripts/collaborative-comments.inline'
// @ts-ignore
import collapseHeaderScript from '../../components/scripts/collapse-header.inline'
// @ts-ignore
import markerScript from '../../components/scripts/marker.inline'
// @ts-ignore
import petScript from '../../components/scripts/pet.inline'
// @ts-ignore
import popoverScript from '../../components/scripts/popover.inline'
//@ts-ignore
import protectedScript from '../../components/scripts/protected.inline'
// @ts-ignore
import spaRouterScript from '../../components/scripts/spa.inline'
// @ts-ignore
import transcludeScript from '../../components/scripts/transclude.inline.ts'
import audioStyle from '../../components/styles/audio.scss'
import clipboardStyle from '../../components/styles/clipboard.scss'
import collapseHeaderStyle from '../../components/styles/collapseHeader.inline.scss'
import popoverStyle from '../../components/styles/popover.scss'
import pseudoStyle from '../../components/styles/pseudocode.scss'
import '../../components/mdx'
import { chunkNotebookPyrightTypeshedFiles } from '../../runtime/lsp/pyright-assets'
import baseStyles from '../../styles/base.scss'
import customStyles from '../../styles/custom.scss'
import { QuartzComponent } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import {
  assetManifestRecord,
  assetPath,
  assetSlugForContent,
  contentHashSlug,
  registerExtractedStaticResource,
  resolveAsset,
} from '../../util/asset-manifest'
import { BuildCtx } from '../../util/ctx'
import { FilePath, FullSlug, isFullSlug, joinSegments } from '../../util/path'
import {
  splitCssBundles,
  splitJsBundles,
  componentCssResourceKey,
  staticCssBundleKey,
  staticCssBundleSlug,
  staticJsBundleKey,
  staticJsBundleSlug,
} from '../../util/resource-bundles'
import { googleFontHref, joinStyles, processGoogleFonts } from '../../util/theme'
import { isWorkerEntryPath, workerEntryPattern } from '../../util/workers'
import { write } from './helpers'

const name = 'ComponentResources'
const collaborativeCommentsClientEntry =
  'quartz/components/scripts/collaborative-comments.client.ts'
const notebookRuntimeInlineEntry = 'quartz/components/scripts/notebook-runtime.inline.ts'
const notebookRuntimeClientEntry = 'quartz/runtime/notebook/client.ts'
const notebookRuntimeLspEntry = 'quartz/runtime/lsp/pyright.ts'
const notebookRuntimeWorkerEntry = 'quartz/runtime/python/pyodide-worker.js'
const notebookRuntimeBootstrapEntry = 'quartz/runtime/python/pyodide-bridge.py'
const notebookRuntimeMlBridgeEntry = 'quartz/runtime/python/ml-bridge.js'
const notebookRuntimePyrightWorkerEntry = 'quartz/util/pyright-browser-worker.js'
const notebookRuntimePyrightWorkerGlobalsEntry = 'quartz/util/pyright-worker-globals.js'
const notebookRuntimePyrightTypeshedDir = 'node_modules/basedpyright/dist/typeshed-fallback/stdlib'
const notebookRuntimePyrightWorkerManifestPath = 'static/scripts/notebook-pyright-worker.json'
const notebookRuntimePyrightTypeshedPath = 'static/scripts/notebook-pyright-typeshed.json'
const notebookRuntimePyrightTypeshedChunkPrefix = 'static/scripts/notebook-pyright-typeshed'
const basedpyrightSourcePackageName = 'basedpyright-source'
const basedpyrightInternalSourcePackageName = 'basedpyright-internal-source'
const notebookPyrightWorkerOutputName = 'notebook-pyright-worker'
const notebookPyrightCommonJsEmptyModule = 'module.exports = {}'
const notebookPyrightDisabledNodeModulePattern =
  /^(?:node:)?(?:child_process|crypto|fs|module|net|os|perf_hooks|stream|tls|v8|worker_threads)$/
const semanticWorkerEntry = 'quartz/workers/semantic.worker.ts'
const semanticWorkerPath = 'static/scripts/semantic.worker.js'
const emojiAssetSourceDir = 'quartz/util/emojimap'
const requireResolve = createRequire(import.meta.url).resolve
const notebookRuntimeAssetEntries = new Set([
  'quartz/plugins/emitters/componentResources.tsx',
  notebookRuntimeClientEntry,
  notebookRuntimeLspEntry,
  notebookRuntimeWorkerEntry,
  notebookRuntimeBootstrapEntry,
  notebookRuntimeMlBridgeEntry,
  notebookRuntimePyrightWorkerEntry,
  notebookRuntimePyrightWorkerGlobalsEntry,
  'quartz/runtime/lsp/pyright-assets.ts',
  'quartz/util/type-guards.ts',
  'quartz/runtime/notebook/assets.ts',
  'quartz/runtime/editor/code-editor.ts',
])

const notebookRuntimeAssetPrefixes = ['quartz/util/notebook/', 'quartz/runtime/']
const semanticWorkerAssetEntries = new Set([semanticWorkerEntry, 'package.json'])

type ComponentResources = {
  css: string[]
  componentCss: string[]
  beforeDOMLoaded: string[]
  afterDOMLoaded: string[]
}

export function normalizeResource(resource: string | string[] | undefined): string[] {
  if (!resource) return []
  if (Array.isArray(resource)) return resource
  return [resource]
}

function getComponentResources(ctx: BuildCtx): ComponentResources {
  const allComponents: Set<QuartzComponent | QuartzMdxComponent> = new Set()
  for (const emitter of ctx.cfg.plugins.emitters) {
    const components = emitter.getQuartzComponents?.(ctx) ?? []
    for (const component of components) {
      allComponents.add(component)
    }
  }
  for (const component of getMdxComponents()) {
    allComponents.add(component)
  }

  const componentResources = {
    css: new Set<string>(),
    beforeDOMLoaded: new Set<string>(),
    afterDOMLoaded: new Set<string>(),
  }

  for (const component of allComponents) {
    const { css, beforeDOMLoaded, afterDOMLoaded } = component
    const normalizedCss = normalizeResource(css)
    const normalizedBeforeDOMLoaded = normalizeResource(beforeDOMLoaded)
    const normalizedAfterDOMLoaded = normalizeResource(afterDOMLoaded)

    normalizedCss.forEach(c => componentResources.css.add(c))
    normalizedBeforeDOMLoaded.forEach(b => componentResources.beforeDOMLoaded.add(b))
    normalizedAfterDOMLoaded.forEach(a => componentResources.afterDOMLoaded.add(a))
  }

  return {
    css: [...componentResources.css],
    componentCss: [...componentResources.css],
    beforeDOMLoaded: [...componentResources.beforeDOMLoaded],
    afterDOMLoaded: [...componentResources.afterDOMLoaded],
  }
}

async function joinScripts(scripts: string[]): Promise<string> {
  // wrap with iife to prevent scope collision
  const script = scripts.map(script => `(function () {${script}})();`).join('\n')

  // minify with esbuild
  const res = await transpile(script, { minify: true })

  return res.code
}

function notebookRuntimeInlineResourceIndex(componentResources: ComponentResources): number {
  return componentResources.afterDOMLoaded.findIndex(
    script =>
      script.includes('notebookRuntimeScriptUrl') && script.includes('data-notebook-runtime-data'),
  )
}

async function refreshNotebookRuntimeInlineResource(componentResources: ComponentResources) {
  const index = notebookRuntimeInlineResourceIndex(componentResources)
  if (index < 0) return
  componentResources.afterDOMLoaded[index] = await fs.readFile(notebookRuntimeInlineEntry, 'utf8')
}

async function currentComponentResources(ctx: BuildCtx): Promise<ComponentResources> {
  const componentResources = getComponentResources(ctx)
  await refreshNotebookRuntimeInlineResource(componentResources)
  addGlobalPageResources(ctx, componentResources)
  return componentResources
}

async function writeAssetBundleOutput(ctx: BuildCtx, outputFile: { path: string; text: string }) {
  const rel = path.relative(ctx.argv.output, outputFile.path).split(path.sep).join('/')
  const ext = path.extname(rel) as `.${string}`
  const logicalSlug = rel.slice(0, -ext.length)
  const slug = rel.includes('/chunks/')
    ? (logicalSlug as FullSlug)
    : assetSlugForContent(ctx, logicalSlug, ext, outputFile.text)
  return write({ ctx, slug, ext, content: outputFile.text })
}

async function writeRawAsset(ctx: BuildCtx, logicalPath: string, content: string | Buffer) {
  const ext = path.extname(logicalPath) as `.${string}`
  const logicalSlug = logicalPath.slice(0, -ext.length)
  const slug = assetSlugForContent(ctx, logicalSlug, ext, content)
  return write({ ctx, slug, ext, content })
}

async function writeAssetManifest(ctx: BuildCtx): Promise<FilePath> {
  return write({
    ctx,
    slug: 'static/scripts/asset-manifest' as FullSlug,
    ext: '.json',
    content: JSON.stringify(assetManifestRecord(ctx)),
  })
}

function minifyStylesheet(filename: string, stylesheet: string) {
  return transform({
    filename,
    code: Buffer.from(stylesheet),
    minify: true,
    targets: {
      safari: (15 << 16) | (6 << 8),
      ios_saf: (15 << 16) | (6 << 8),
      edge: 115 << 16,
      firefox: 102 << 16,
      chrome: 109 << 16,
    },
    include: Features.MediaQueries,
  }).code.toString()
}

async function* writeStaticResourceBundles(ctx: BuildCtx, resources: StaticResources) {
  for (const part of splitCssBundles(resources.css, [collapseHeaderStyle])) {
    if (part.type !== 'bundle') continue
    const content = minifyStylesheet('resource-style.css', part.content)
    const slug = contentHashSlug(staticCssBundleSlug, content)
    registerExtractedStaticResource(ctx, staticCssBundleKey(part.content), assetPath(slug, '.css'))
    yield write({ ctx, slug, ext: '.css', content })
  }

  for (const loadTime of ['beforeDOMReady', 'afterDOMReady'] as const) {
    const leadingInline =
      loadTime === 'afterDOMReady' ? [transcludeScript, collapseHeaderScript] : []
    for (const part of splitJsBundles(resources.js, loadTime, leadingInline)) {
      if (part.type !== 'bundle') continue
      const content = await joinScripts(part.scripts)
      const slug = contentHashSlug(staticJsBundleSlug(part.loadTime), content)
      registerExtractedStaticResource(
        ctx,
        staticJsBundleKey(part.loadTime, part.scripts),
        assetPath(slug, '.js'),
      )
      yield write({ ctx, slug, ext: '.js', content })
    }
  }
}

async function* writeComponentStyles(ctx: BuildCtx, resources: ComponentResources) {
  for (const stylesheet of resources.componentCss) {
    const content = minifyStylesheet('component.css', `@layer quartz-base {\n${stylesheet}\n}`)
    const slug = contentHashSlug('component', content)
    registerExtractedStaticResource(
      ctx,
      componentCssResourceKey(stylesheet),
      assetPath(slug, '.css'),
    )
    yield write({ ctx, slug, ext: '.css', content })
  }
}

async function writeAfterDomLoadedScripts(ctx: BuildCtx, scripts: string[]) {
  const entries = await Promise.all(
    scripts.map(async script => {
      const content = await joinScripts([script])
      const slug = contentHashSlug('static/scripts/script', content)
      return {
        filename: assetPath(slug, '.js'),
        file: await write({ ctx, slug, ext: '.js', content }),
      }
    }),
  )

  const postscript = entries.map(({ filename }) => `await import("./${filename}");`).join('\n')

  return { postscript, files: entries.map(({ file }) => file) }
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

function packageDir(packageName: string) {
  return path.dirname(requireResolve(`${packageName}/package.json`))
}

function packageNodeModules(packageName: string) {
  return path.join(packageDir(packageName), 'node_modules')
}

function basedpyrightInternalSourceDir() {
  return path.join(packageDir(basedpyrightSourcePackageName), 'packages/pyright-internal/src')
}

function pyrightInternalSourcePath(importPath: string) {
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

function pyrightBrowserDependencyPaths() {
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

function notebookRuntimePyrightChunkPath(prefix: string, index: number, ext: string) {
  return `${prefix}-${index}${ext}`
}

async function writeNotebookPyrightChunkedAsset(
  ctx: BuildCtx,
  manifestPath: string,
  chunkPrefix: string,
  ext: string,
  chunks: string[],
): Promise<FilePath[]> {
  const chunkPaths = chunks.map((_chunk, index) =>
    notebookRuntimePyrightChunkPath(chunkPrefix, index, ext),
  )
  const chunkFiles = await Promise.all(
    chunks.map((chunk, index) => writeRawAsset(ctx, chunkPaths[index], chunk)),
  )
  const manifest = {
    chunks: chunkPaths.map(chunkPath => path.basename(resolveAsset(ctx, chunkPath))),
  }
  const manifestFile = await writeRawAsset(ctx, manifestPath, JSON.stringify(manifest))
  return [manifestFile, ...chunkFiles]
}

async function notebookPyrightTypeshedFiles() {
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

async function writeNotebookPyrightTypeshedAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const chunks = chunkNotebookPyrightTypeshedFiles(await notebookPyrightTypeshedFiles())
  return writeNotebookPyrightChunkedAsset(
    ctx,
    notebookRuntimePyrightTypeshedPath,
    notebookRuntimePyrightTypeshedChunkPrefix,
    '.json',
    chunks.map(chunk => JSON.stringify({ files: chunk.files })),
  )
}

async function writeNotebookPyrightWorkerAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
  const worker = await bundle({
    entryPoints: { [notebookPyrightWorkerOutputName]: notebookRuntimePyrightWorkerEntry },
    bundle: true,
    minify: true,
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
  const manifest = { entry: path.basename(resolveAsset(ctx, entryLogicalPath)) }
  const manifestFile = await writeRawAsset(
    ctx,
    notebookRuntimePyrightWorkerManifestPath,
    JSON.stringify(manifest),
  )
  return [manifestFile, ...workerFiles]
}

async function writeNotebookPyrightAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const [workerFiles, typeshedFiles] = await Promise.all([
    writeNotebookPyrightWorkerAssets(ctx),
    writeNotebookPyrightTypeshedAssets(ctx),
  ])
  return [...workerFiles, ...typeshedFiles]
}

async function writeNotebookRuntimeAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
  const worker = await bundle({
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
  })
  const workerFiles = await Promise.all(
    worker.outputFiles.map(output => writeAssetBundleOutput(ctx, output)),
  )
  const pyrightFiles = await writeNotebookPyrightAssets(ctx)
  const workerName = path.basename(resolveAsset(ctx, 'static/scripts/notebook-runtime.worker.js'))
  const pyrightWorkerName = path.basename(
    resolveAsset(ctx, notebookRuntimePyrightWorkerManifestPath),
  )
  const pyrightTypeshedName = path.basename(resolveAsset(ctx, notebookRuntimePyrightTypeshedPath))
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
  const workerPath = path.join(outdir, workerName)
  const clientOutputs = client.outputFiles.map(output => ({
    ...output,
    text: replaceNotebookRuntimeWorkerReference(output.text, output.path, workerPath)
      .replaceAll('notebook-pyright-worker.json', pyrightWorkerName)
      .replaceAll('notebook-pyright-typeshed.json', pyrightTypeshedName),
  }))
  const clientFiles = await Promise.all(
    clientOutputs.map(output => writeAssetBundleOutput(ctx, output)),
  )
  return [...workerFiles, ...pyrightFiles, ...clientFiles]
}

async function writeCollaborativeCommentsAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
  const client = await bundle({
    entryPoints: { 'collaborative-comments.client': collaborativeCommentsClientEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    write: false,
  })
  return await Promise.all(client.outputFiles.map(output => writeAssetBundleOutput(ctx, output)))
}

async function writeSemanticWorkerAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const outdir = path.join(ctx.argv.output, 'static/scripts')
  const worker = await bundle({
    entryPoints: { 'semantic.worker': semanticWorkerEntry },
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    splitting: true,
    outdir,
    entryNames: '[name]',
    chunkNames: 'chunks/[name]-[hash]',
    write: false,
  })
  return await Promise.all(worker.outputFiles.map(output => writeAssetBundleOutput(ctx, output)))
}

async function removeSemanticWorkerAsset(ctx: BuildCtx) {
  await fs.rm(path.join(ctx.argv.output, semanticWorkerPath), { force: true })
}

async function writeEmojiAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const files = await globby([`${emojiAssetSourceDir}/**/*.json`])
  return await Promise.all(
    files.map(async file => {
      const rel = path.relative(emojiAssetSourceDir, file).split(path.sep).join('/')
      const slug = joinSegments('static', 'scripts', 'emoji', rel.slice(0, -'.json'.length))
      if (!isFullSlug(slug)) throw new Error(`invalid emoji asset slug ${slug}`)
      const content = await fs.readFile(file)
      return await write({
        ctx,
        slug: assetSlugForContent(ctx, slug, '.json', content),
        ext: '.json',
        content,
      })
    }),
  )
}

function assetBasename(ctx: BuildCtx, logicalPath: string): string {
  return path.basename(resolveAsset(ctx, logicalPath))
}

function relativeBundleAssetReference(fromFile: string, toFile: string): string {
  return path.relative(path.dirname(fromFile), toFile).split(path.sep).join('/')
}

function replaceNotebookRuntimeWorkerReference(text: string, fromFile: string, workerPath: string) {
  const placeholder = '\0quartz-notebook-runtime-worker\0'
  return text
    .replaceAll('../notebook-runtime.worker.js', placeholder)
    .replaceAll('notebook-runtime.worker.js', placeholder)
    .replaceAll(placeholder, relativeBundleAssetReference(fromFile, workerPath))
}

function resolveComponentResourceAssets(ctx: BuildCtx, componentResources: ComponentResources) {
  componentResources.afterDOMLoaded = componentResources.afterDOMLoaded.map(script =>
    script
      .replaceAll(
        'notebook-runtime.client.js',
        assetBasename(ctx, 'static/scripts/notebook-runtime.client.js'),
      )
      .replaceAll(
        'notebook-runtime.worker.js',
        assetBasename(ctx, 'static/scripts/notebook-runtime.worker.js'),
      )
      .replaceAll(
        'notebook-pyright-worker.json',
        assetBasename(ctx, notebookRuntimePyrightWorkerManifestPath),
      )
      .replaceAll(
        'notebook-pyright-typeshed.json',
        assetBasename(ctx, notebookRuntimePyrightTypeshedPath),
      )
      .replaceAll(
        'collaborative-comments.client.js',
        assetBasename(ctx, 'static/scripts/collaborative-comments.client.js'),
      )
      .replaceAll('semantic.worker.js', assetBasename(ctx, semanticWorkerPath)),
  )
}

function isNotebookRuntimeAssetChange(changePath: string): boolean {
  if (notebookRuntimeAssetEntries.has(changePath)) return true
  return notebookRuntimeAssetPrefixes.some(prefix => changePath.startsWith(prefix))
}

function isNotebookRuntimePageScriptChange(changePath: string): boolean {
  return changePath === notebookRuntimeInlineEntry
}

function isCollaborativeCommentsAssetChange(changePath: string): boolean {
  return (
    changePath === collaborativeCommentsClientEntry ||
    changePath === 'quartz/util/type-guards.ts' ||
    changePath === 'quartz/components/scripts/markdown-editor.ts' ||
    changePath === 'quartz/components/scripts/search-index.ts' ||
    changePath.startsWith('quartz/components/multiplayer/') ||
    changePath.startsWith('quartz/functional/')
  )
}

function isSemanticWorkerAssetChange(changePath: string): boolean {
  return semanticWorkerAssetEntries.has(changePath)
}

function isEmojiAssetChange(changePath: string): boolean {
  return changePath.startsWith(`${emojiAssetSourceDir}/`) && changePath.endsWith('.json')
}

function addGlobalPageResources(ctx: BuildCtx, componentResources: ComponentResources) {
  const cfg = ctx.cfg.configuration
  const resolvedPetScript = petScript.replace(
    '__QUARTZ_PETS_DEFAULT_ENABLED__',
    ctx.argv.watch || ctx.argv.serve ? '0' : '1',
  )

  // popovers
  if (cfg.enablePopovers) {
    componentResources.afterDOMLoaded.push(popoverScript)
    componentResources.css.push(popoverStyle)
  }

  componentResources.beforeDOMLoaded.push(markerScript)

  componentResources.css.push(clipboardStyle, pseudoStyle, audioStyle)
  componentResources.afterDOMLoaded.push(
    clipboardScript,
    pseudoScript,
    protectedScript,
    audioScript,
    baseMapScript,
    collaborativeCommentsScript,
    resolvedPetScript,
  )

  if (cfg.analytics?.provider === 'plausible') {
    const plausibleHost = cfg.analytics.host ?? 'https://plausible.io'
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.outbound-links.manual.js"
      plausibleScript.setAttribute("data-domain", [location.hostname, "stream.aarnphm.xyz"].join(','))
      plausibleScript.setAttribute("data-api", "/_plausible/event")
      plausibleScript.defer = true
      plausibleScript.onload = () => {
        window.plausible = window.plausible || function () { (window.plausible.q = window.plausible.q || []).push(arguments); };
        plausible('pageview')
        document.addEventListener('nav', () => {
          plausible('pageview')
        })
      }

      document.head.appendChild(plausibleScript)
    `)
  }

  componentResources.afterDOMLoaded.push(notFoundScript, spaRouterScript)
}

export const ComponentResources: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, _content, resources) {
      const cfg = ctx.cfg.configuration
      // component specific scripts and styles
      const componentResources = await currentComponentResources(ctx)
      const notebookRuntimeFiles = await writeNotebookRuntimeAssets(ctx)
      const collaborativeCommentsFiles = await writeCollaborativeCommentsAssets(ctx)
      const semanticWorkerFiles = await writeSemanticWorkerAssets(ctx)
      const emojiFiles = await writeEmojiAssets(ctx)
      resolveComponentResourceAssets(ctx, componentResources)
      let googleFontsStyleSheet = ''
      if (cfg.theme.fontOrigin === 'local') {
        // let the user do it themselves in css
      } else if (cfg.theme.fontOrigin === 'googleFonts' && !cfg.theme.cdnCaching) {
        const response = await fetch(googleFontHref(ctx.cfg.configuration.theme))
        googleFontsStyleSheet = await response.text()

        if (!cfg.baseUrl) {
          throw new Error(
            'baseUrl must be defined when using Google Fonts without cfg.theme.cdnCaching',
          )
        }

        const { processedStylesheet, fontFiles } = await processGoogleFonts(
          googleFontsStyleSheet,
          cfg.baseUrl,
        )
        googleFontsStyleSheet = processedStylesheet

        // Download and save font files
        for (const fontFile of fontFiles) {
          const res = await fetch(fontFile.url)
          if (!res.ok) {
            throw new Error(`failed to fetch font ${fontFile.filename}`)
          }

          const buf = await res.arrayBuffer()
          yield write({
            ctx,
            slug: joinSegments('static', 'fonts', fontFile.filename) as FullSlug,
            ext: `.${fontFile.extension}`,
            content: Buffer.from(buf),
          })
        }
      }

      yield* writeComponentStyles(ctx, componentResources)

      const componentCss = new Set(componentResources.componentCss)
      const quartzBase = joinStyles(
        ctx.cfg.configuration.theme,
        googleFontsStyleSheet,
        ...componentResources.css.filter(css => !componentCss.has(css)),
        baseStyles,
      )
      const stylesheet = `@layer quartz-base {\n${quartzBase}\n}\n${customStyles}`
      const [prescript, postscriptResult] = await Promise.all([
        joinScripts(componentResources.beforeDOMLoaded),
        writeAfterDomLoadedScripts(ctx, componentResources.afterDOMLoaded),
      ])
      const { postscript, files: postscriptFiles } = postscriptResult

      const manifest = {
        name: cfg.pageTitle,
        short_name: cfg.baseUrl,
        icons: [
          { src: '/android-chrome-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: '/android-chrome-512x512.png', sizes: '512x512', type: 'image/png' },
        ],
        theme_color: cfg.theme.colors['lightMode'].light,
        background_color: cfg.theme.colors['lightMode'].light,
        display: 'standalone',
        lang: cfg.locale,
        dir: 'auto',
      }

      const stylesheetContent = minifyStylesheet('index.css', stylesheet)
      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'index', '.css', stylesheetContent),
        ext: '.css',
        content: stylesheetContent,
      })

      yield* writeStaticResourceBundles(ctx, resources)

      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'prescript', '.js', prescript),
        ext: '.js',
        content: prescript,
      })

      yield write({
        ctx,
        slug: assetSlugForContent(ctx, 'postscript', '.js', postscript),
        ext: '.js',
        content: postscript,
      })

      for (const file of postscriptFiles) {
        yield file
      }

      yield write({
        ctx,
        slug: 'site' as FullSlug,
        ext: '.webmanifest',
        content: JSON.stringify(manifest),
      })

      const workerFiles = (await globby([workerEntryPattern])).filter(
        src => src !== semanticWorkerEntry,
      )
      for (const src of workerFiles) {
        const result = await bundle({
          entryPoints: [src],
          bundle: true,
          minify: true,
          platform: 'browser',
          format: 'esm',
          write: false,
        })
        const code = result.outputFiles[0].text
        const name = path.basename(src).replace(/\.ts$/, '')
        yield write({ ctx, slug: name as FullSlug, ext: '.js', content: code })
      }

      for (const file of notebookRuntimeFiles) {
        yield file
      }

      for (const file of collaborativeCommentsFiles) {
        yield file
      }

      for (const file of semanticWorkerFiles) {
        yield file
      }

      for (const file of emojiFiles) {
        yield file
      }

      yield writeAssetManifest(ctx)
    },
    async *partialEmit(ctx, _content, resources, changeEvents) {
      const componentResources = await currentComponentResources(ctx)
      yield* writeComponentStyles(ctx, componentResources)
      yield* writeStaticResourceBundles(ctx, resources)

      if (changeEvents.some(changeEvent => isNotebookRuntimeAssetChange(changeEvent.path))) {
        for (const file of await writeNotebookRuntimeAssets(ctx)) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isNotebookRuntimePageScriptChange(changeEvent.path))) {
        resolveComponentResourceAssets(ctx, componentResources)
        const [prescript, postscriptResult] = await Promise.all([
          joinScripts(componentResources.beforeDOMLoaded),
          writeAfterDomLoadedScripts(ctx, componentResources.afterDOMLoaded),
        ])
        const { postscript, files: postscriptFiles } = postscriptResult

        yield write({
          ctx,
          slug: assetSlugForContent(ctx, 'prescript', '.js', prescript),
          ext: '.js',
          content: prescript,
        })
        yield write({
          ctx,
          slug: assetSlugForContent(ctx, 'postscript', '.js', postscript),
          ext: '.js',
          content: postscript,
        })

        for (const file of postscriptFiles) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isCollaborativeCommentsAssetChange(changeEvent.path))) {
        for (const file of await writeCollaborativeCommentsAssets(ctx)) {
          yield file
        }
      }

      const semanticWorkerDeleted = changeEvents.some(
        changeEvent => changeEvent.path === semanticWorkerEntry && changeEvent.type === 'delete',
      )
      if (semanticWorkerDeleted) {
        await removeSemanticWorkerAsset(ctx)
      } else if (changeEvents.some(changeEvent => isSemanticWorkerAssetChange(changeEvent.path))) {
        for (const file of await writeSemanticWorkerAssets(ctx)) {
          yield file
        }
      }

      if (changeEvents.some(changeEvent => isEmojiAssetChange(changeEvent.path))) {
        for (const file of await writeEmojiAssets(ctx)) {
          yield file
        }
      }

      for (const changeEvent of changeEvents) {
        if (!isWorkerEntryPath(changeEvent.path)) continue
        if (changeEvent.path === semanticWorkerEntry) continue
        if (changeEvent.type === 'delete') {
          const name = path.basename(changeEvent.path).replace(/\.ts$/, '')
          const dest = joinSegments(ctx.argv.output, `${name}.js`)
          await fs.unlink(dest)
          continue
        }
        const result = await bundle({
          entryPoints: [changeEvent.path],
          bundle: true,
          minify: true,
          platform: 'browser',
          format: 'esm',
          write: false,
        })
        const code = result.outputFiles[0].text
        const name = path.basename(changeEvent.path).replace(/\.ts$/, '')
        yield write({ ctx, slug: name as FullSlug, ext: '.js', content: code })
      }

      yield writeAssetManifest(ctx)
    },
    externalResources: ({ cfg }) => ({
      additionalHead: [
        <link rel="manifest" href={`https://${cfg.configuration.baseUrl}/site.webmanifest`} />,
        <link rel="shortcut icon" href={`https://${cfg.configuration.baseUrl}/favicon.ico`} />,
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href={`https://${cfg.configuration.baseUrl}/favicon-32x32.png`}
        />,
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href={`https://${cfg.configuration.baseUrl}/favicon-16x16.png`}
        />,
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href={`https://${cfg.configuration.baseUrl}/apple-touch-icon.png`}
        />,
        <link
          rel="android-chrome"
          sizes="192x192"
          href={`https://${cfg.configuration.baseUrl}/android-chrome-192x192.png`}
        />,
        <link
          rel="android-chrome"
          sizes="512x512"
          href={`https://${cfg.configuration.baseUrl}/android-chrome-512x512.png`}
        />,
      ],
    }),
  }
}
