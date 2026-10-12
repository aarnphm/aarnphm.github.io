import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import { notebookPyrightTypeshedFiles } from '../../../runtime/lsp/pyright-assets'
import { isJsonObject } from '../../../util/type-guards'

function testCtx(output: string): BuildCtx {
  const colors = {
    light: '#fff',
    lightgray: '#eee',
    gray: '#888',
    darkgray: '#444',
    dark: '#000',
    secondary: '#456',
    tertiary: '#789',
    highlight: '#abc',
    textHighlight: '#def',
  }
  return {
    buildId: 'notebook-offline',
    argv: {
      directory: 'content',
      output,
      verbose: false,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'offline notebook fixture',
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
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

async function readObject(file: string) {
  const value: unknown = JSON.parse(await readFile(file, 'utf8'))
  assert(isJsonObject(value), `expected an object in ${file}`)
  return value
}

test(
  'emits production notebook assets with Node network access denied',
  { skip: !process.allowedNodeEnvironmentFlags.has('--allow-net') },
  async () => {
    const artifacts = path.resolve('docs/verification/component-resources-offline/artifacts')
    await mkdir(artifacts, { recursive: true })
    const run = await mkdtemp(path.join(artifacts, 'run-'))
    const output = path.join(run, 'output')
    const ctx = testCtx(output)
    const manifestPath = path.join(run, 'asset-manifest.json')
    const runner = path.join(run, 'emit.mjs')
    const emitterPath = path.resolve(
      'quartz/plugins/emitters/component-resources/notebook-assets.ts',
    )
    const manifestSourcePath = path.resolve('quartz/util/asset-manifest.ts')
    const contents = `
      import assert from 'node:assert/strict'
      import { writeFile } from 'node:fs/promises'
      import { writeNotebookRuntimeAssets } from ${JSON.stringify(emitterPath)}
      import { assetManifestRecord } from ${JSON.stringify(manifestSourcePath)}
      assert.equal(process.permission.has('net'), false)
      const ctx = ${JSON.stringify(ctx)}
      const files = await writeNotebookRuntimeAssets(ctx)
      await writeFile(${JSON.stringify(manifestPath)}, JSON.stringify(assetManifestRecord(ctx), null, 2))
      console.log(JSON.stringify({ networkAllowed: false, files: files.length }))
    `
    await build({
      stdin: { contents, resolveDir: process.cwd(), loader: 'js' },
      outfile: runner,
      bundle: true,
      packages: 'external',
      platform: 'node',
      format: 'esm',
    })
    const args = [
      '--permission',
      '--allow-fs-read=*',
      `--allow-fs-write=${run}`,
      '--allow-child-process',
      runner,
    ]
    await writeFile(path.join(run, 'fixture.json'), JSON.stringify(ctx, null, 2))
    await writeFile(
      path.join(run, 'command.json'),
      JSON.stringify({ cwd: process.cwd(), executable: process.execPath, args }, null, 2),
    )
    const result = spawnSync(process.execPath, args, {
      cwd: process.cwd(),
      env: { ...process.env, NODE_ENV: 'production' },
      encoding: 'utf8',
      timeout: 120_000,
      maxBuffer: 8 * 1024 * 1024,
    })
    await writeFile(path.join(run, 'process.log'), result.stdout + result.stderr)
    await writeFile(
      path.join(run, 'result.json'),
      JSON.stringify(
        { status: result.status, signal: result.signal, error: result.error?.message },
        null,
        2,
      ),
    )
    assert.equal(result.status, 0, `emitter failed; evidence: ${run}\n${result.stderr}`)
    const manifest = await readObject(manifestPath)
    for (const emittedPath of Object.values(manifest)) {
      assert.equal(typeof emittedPath, 'string')
      if (typeof emittedPath !== 'string') continue
      assert((await stat(path.join(output, emittedPath))).size > 0)
    }
    const stubLogicalPath = 'static/scripts/notebook-pyright/site-packages/manifest.json'
    const stubManifestPath = manifest[stubLogicalPath]
    assert.equal(typeof stubManifestPath, 'string')
    if (typeof stubManifestPath !== 'string') return
    const stubManifest = await readObject(path.join(output, stubManifestPath))
    assert(Array.isArray(stubManifest.chunks))
    const stubFiles: Record<string, string> = {}
    for (const chunk of stubManifest.chunks) {
      assert.equal(typeof chunk, 'string')
      if (typeof chunk !== 'string') continue
      const chunkData = await readObject(path.join(output, path.dirname(stubManifestPath), chunk))
      Object.assign(stubFiles, notebookPyrightTypeshedFiles(chunkData))
    }
    for (const moduleName of ['numpy', 'scipy', 'pandas', 'micropip', 'matplotlib', 'sympy']) {
      assert(stubFiles[`/site-packages/${moduleName}/__init__.pyi`], `missing ${moduleName} stub`)
    }
    const genericStub = await readFile('quartz/util/pyright-stubs/generic-module.pyi', 'utf8')
    assert.equal(stubFiles['/site-packages/sympy/__init__.pyi'], genericStub)
    const ipythonStub = await readFile(
      'quartz/util/pyright-stubs/site-packages/IPython/__init__.pyi',
      'utf8',
    )
    assert.equal(stubFiles['/site-packages/IPython/__init__.pyi'], ipythonStub)
    for (const logicalPath of [
      'static/scripts/notebook-runtime.worker.js',
      'static/scripts/notebook-runtime.javascript.worker.js',
      'static/scripts/notebook-runtime.client.js',
      'static/scripts/notebook-pyright/worker/manifest.json',
      'static/scripts/notebook-pyright/typeshed/manifest.json',
      'static/scripts/notebook-runtimes/manifest.json',
    ]) {
      assert.equal(typeof manifest[logicalPath], 'string', `missing ${logicalPath}`)
    }
    console.log(`offline notebook asset evidence: ${run}`)
  },
)
