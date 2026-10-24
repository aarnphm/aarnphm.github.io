import { build, type Plugin } from 'esbuild'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import test from 'node:test'
import { pathToFileURL } from 'node:url'
import ts from 'typescript'
import type { BuildCtx } from '../../../util/ctx'
import { assetManifestRecord, hashAssetContent } from '../../../util/asset-manifest'
import { bundleInlineScript } from '../../../util/inline-script-bundler'
import { resetNotebookRuntimeAssetCache } from './notebook-assets'
import { writePageScripts } from './script-assets'

function testCtx(output: string): BuildCtx {
  return {
    buildId: 'test-page-script-publication',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'page script publication fixture',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        baseUrl: 'example.com',
        locale: 'en-US',
        theme: {
          fontOrigin: 'local',
          cdnCaching: true,
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          colors: {
            lightMode: {
              light: '#fff',
              lightgray: '#eee',
              gray: '#888',
              darkgray: '#444',
              dark: '#000',
              secondary: '#456',
              tertiary: '#789',
              highlight: '#abc',
              textHighlight: '#def',
            },
            darkMode: {
              light: '#fff',
              lightgray: '#eee',
              gray: '#888',
              darkgray: '#444',
              dark: '#000',
              secondary: '#456',
              tertiary: '#789',
              highlight: '#abc',
              textHighlight: '#def',
            },
          },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

type Edge = { target: string; dynamic: boolean }
type ModuleInfo = { imports: Edge[]; pierrePayloads: number; bytes: number }

function inspectModule(file: string, code: string, wasmHash: string): ModuleInfo {
  const tree = ts.createSourceFile(file, code, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
  const imports: Edge[] = []
  let pierrePayloads = 0
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      if (node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier))
        imports.push({ target: node.moduleSpecifier.text, dynamic: false })
    }
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      const target = node.arguments[0]
      if (target && ts.isStringLiteral(target)) imports.push({ target: target.text, dynamic: true })
    }
    if (ts.isStringLiteral(node) && node.text.startsWith('AGFzbQ')) {
      const bytes = Buffer.from(node.text, 'base64')
      if (createHash('sha256').update(bytes).digest('hex') === wasmHash) pierrePayloads++
    }
    ts.forEachChild(node, visit)
  }
  visit(tree)
  return { imports, pierrePayloads, bytes: Buffer.byteLength(code) }
}

function isPlugin(value: unknown): value is Plugin {
  return (
    typeof value === 'object' &&
    value !== null &&
    'name' in value &&
    typeof value.name === 'string' &&
    'setup' in value &&
    typeof value.setup === 'function'
  )
}

async function bootstrapInlineScript(run: string): Promise<string> {
  const handlers: unknown = await import(pathToFileURL(path.resolve('quartz/cli/handlers.js')).href)
  assert(
    handlers &&
      typeof handlers === 'object' &&
      'createBuildConfig' in handlers &&
      typeof handlers.createBuildConfig === 'function',
  )
  const config: unknown = handlers.createBuildConfig()
  assert(
    config && typeof config === 'object' && 'plugins' in config && Array.isArray(config.plugins),
  )
  const loader = config.plugins.find(
    (value: unknown) => isPlugin(value) && value.name === 'inline-script-loader',
  )
  assert(isPlugin(loader))
  const output = path.join(run, 'bootstrap-stage-one.mjs')
  await build({
    stdin: {
      contents:
        'import script from "./quartz/components/scripts/search.inline.ts"; export default script',
      resolveDir: process.cwd(),
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    plugins: [loader],
    outfile: output,
  })
  const imported: unknown = await import(pathToFileURL(output).href)
  assert(
    imported &&
      typeof imported === 'object' &&
      'default' in imported &&
      typeof imported.default === 'string',
  )
  return imported.default
}

// Real emitted graphs catch duplicate WASM bytes, stage-one import hoisting, unresolved
// shared URLs, and production hash drift without replacing the highlighter or its callers.
test('all client builds share one lazy Pierre payload through both inline bundling stages', async () => {
  const artifacts = path.resolve('docs/verification/highlights/artifacts')
  await mkdir(artifacts, { recursive: true })
  const run = await mkdtemp(path.join(artifacts, 'shared-bundle-'))
  const output = path.join(run, 'output')
  const ctx = testCtx(output)
  const scripts = await Promise.all([
    bundleInlineScript('quartz/components/scripts/palette.inline.ts'),
    bootstrapInlineScript(run),
  ])
  await writeFile(path.join(run, 'stage-one-palette.js'), scripts[0])
  await writeFile(path.join(run, 'stage-one-search.js'), scripts[1])
  const files = await writePageScripts(ctx, {
    css: [],
    componentCss: [],
    beforeDOMLoaded: [],
    afterDOMLoaded: scripts,
  })
  const manifest = assetManifestRecord(ctx)
  const resolve = createRequire(import.meta.url).resolve
  const packageRoot = path.dirname(resolve('@pierre/highlights/package.json'))
  const wasm = await readFile(path.join(packageRoot, 'dist/highlights.wasm'))
  const wasmHash = createHash('sha256').update(wasm).digest('hex')
  const page = files.find(file => file.includes('/static/scripts/script-'))
  assert(page)
  const roots = [
    page,
    ...['collaborative-comments.client', 'notebook-runtime.client'].map(name => {
      const relative = manifest[`static/scripts/${name}.js`]
      assert(relative, `missing ${name}`)
      return path.join(output, relative)
    }),
  ]
  const graph = new Map<string, ModuleInfo>()
  const visit = async (file: string): Promise<void> => {
    if (graph.has(file)) return
    const info = inspectModule(file, await readFile(file, 'utf8'), wasmHash)
    graph.set(file, info)
    for (const edge of info.imports) {
      if (!edge.target.startsWith('.') && !edge.target.startsWith('/')) continue
      const resolved = edge.target.startsWith('/')
        ? path.join(output, edge.target)
        : path.resolve(path.dirname(file), edge.target)
      await visit(resolved)
    }
  }
  for (const root of roots) await visit(root)
  const payloads = [...graph.entries()].filter(([, info]) => info.pierrePayloads > 0)
  const report = {
    roots,
    wasmHash,
    payloads: payloads.map(([file, info]) => ({ file, count: info.pierrePayloads })),
    modules: Object.fromEntries(graph),
  }
  await writeFile(path.join(run, 'graph.json'), JSON.stringify(report, null, 2))
  await writeFile(path.join(run, 'asset-manifest.json'), JSON.stringify(manifest, null, 2))
  assert.equal(
    payloads.reduce((count, [, info]) => count + info.pierrePayloads, 0),
    1,
    `duplicate Pierre payloads; ${run}`,
  )
  const logical = 'static/scripts/highlight-source.js'
  const emitted = manifest[logical]
  assert(emitted, 'shared highlighter must register its URL')
  const shared = path.join(output, emitted)
  const bytes = await readFile(shared)
  assert.equal(emitted, `static/scripts/highlight-source-${hashAssetContent(bytes)}.js`)
  assert.equal(payloads[0][0], shared)
  assert(files.some(file => file === shared))
  const sharedUrl = '/' + emitted
  for (const root of roots) {
    const seen = new Set<string>()
    const reachable = (file: string, staticOnly: boolean): string[] => {
      if (seen.has(file)) return []
      seen.add(file)
      const info = graph.get(file)
      assert(info)
      const children = info.imports.filter(
        edge =>
          (!staticOnly || !edge.dynamic) &&
          (edge.target.startsWith('.') || edge.target.startsWith('/')),
      )
      return [
        file,
        ...children.flatMap(edge =>
          reachable(
            edge.target.startsWith('/')
              ? path.join(output, edge.target)
              : path.resolve(path.dirname(file), edge.target),
            staticOnly,
          ),
        ),
      ]
    }
    assert(!reachable(root, true).includes(shared), `${root} eagerly loads Pierre`)
    seen.clear()
    const all = reachable(root, false)
    assert(all.includes(shared), `${root} does not reach shared highlighter`)
    assert(
      all.some(file =>
        graph.get(file)?.imports.some(edge => edge.dynamic && edge.target === sharedUrl),
      ),
      `${root} lacks shared lazy URL`,
    )
  }
  const highlighter: unknown = await import(pathToFileURL(shared).href)
  assert(
    highlighter &&
      typeof highlighter === 'object' &&
      'highlightCode' in highlighter &&
      typeof highlighter.highlightCode === 'function',
  )
  const highlighted: unknown = highlighter.highlightCode('const answer = 42', 'javascript')
  assert(
    highlighted &&
      typeof highlighted === 'object' &&
      'lines' in highlighted &&
      highlighted.lines === 1 &&
      'html' in highlighted &&
      typeof highlighted.html === 'string',
  )
  resetNotebookRuntimeAssetCache()
  console.log(`shared highlighter evidence: ${run}`)
})
