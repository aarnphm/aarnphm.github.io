import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { WebSocket } from 'ws'
import {
  createReloadServer,
  formatBundleInfoJson,
  formatBundleInfoTable,
  isSourceWatchPath,
  isTestSourcePath,
  parseSourceInputs,
  resolveBundleInfoFormat,
} from './handlers.js'

test('watch reload server broadcasts successive builds without an HTTP server', async t => {
  const reload = createReloadServer(0)
  t.after(() => new Promise(resolve => reload.server.close(resolve)))
  await once(reload.server, 'listening')
  const address = reload.server.address()
  assert.ok(address && typeof address !== 'string')
  const socket = new WebSocket(`ws://localhost:${address.port}`)
  t.after(() => socket.terminate())
  await once(socket, 'open')

  for (let index = 0; index < 2; index++) {
    const message = once(socket, 'message')
    reload.refresh()
    assert.equal((await message)[0].toString(), 'rebuild')
  }

  const closed = once(socket, 'close')
  socket.close()
  await closed
  reload.refresh()
})

const bundleMetafile = {
  inputs: {
    'quartz/build.ts': { bytes: 32, imports: [] },
    'quartz/components/PageTitle.tsx': { bytes: 10, imports: [] },
  },
  outputs: {
    'quartz/.quartz-cache/transpiled-build.mjs': {
      bytes: 42,
      entryPoint: 'quartz/build.ts',
      exports: [],
      imports: [],
      inputs: {
        'quartz/build.ts': { bytesInOutput: 32 },
        'quartz/components/PageTitle.tsx': { bytesInOutput: 10 },
      },
    },
  },
}

test('source watcher ignores test files across repository conventions', () => {
  for (const fp of [
    'quartz/cli/handlers.test.js',
    'quartz/components/PageTitle.spec.tsx',
    'quartz/runtime/native/worker.test.mjs',
    'quartz/scripts/test_sync.py',
    'quartz/scripts/sync_test.py',
    'quartz/components/__tests__/PageTitle.tsx',
    'quartz/components/tests/fixtures.ts',
    'quartz/components/spec/render.ts',
    'quartz/components/page_spec.rb',
    'quartz/components/test.ts',
  ]) {
    assert.equal(isTestSourcePath(fp), true, fp)
    assert.equal(isSourceWatchPath(fp), false, fp)
  }

  assert.equal(isTestSourcePath('quartz/components/Latest.tsx'), false)
})

test('source watcher accepts newly added Quartz source files', () => {
  assert.equal(isSourceWatchPath('quartz/util/transclude-props.ts'), true)
  assert.equal(isSourceWatchPath('quartz/components/renderPage.tsx'), true)
  assert.equal(isSourceWatchPath('.claude/skills/core/SKILL.md'), false)
  assert.equal(isSourceWatchPath('quartz/.quartz-cache/transpiled-build.mjs'), false)
  assert.equal(isSourceWatchPath('quartz/util/transclude-props.test.ts'), false)
})

test('bundle info table can render without ANSI escapes for pipes', async () => {
  const output = await formatBundleInfoTable(bundleMetafile, false)
  assert.match(output, /Successfully transpiled 2 files/)
  assert.match(output, /quartz\/\.quartz-cache\/transpiled-build\.mjs/)
  assert.equal(output.includes(String.fromCharCode(27)), false)
})

test('bundle info JSON is machine readable', () => {
  const payload = JSON.parse(formatBundleInfoJson(bundleMetafile))
  assert.equal(payload.summary.outputFile, 'quartz/.quartz-cache/transpiled-build.mjs')
  assert.equal(payload.summary.inputCount, 2)
  assert.equal(payload.summary.bytes, 42)
  assert.equal(payload.metafile.outputs['quartz/.quartz-cache/transpiled-build.mjs'].bytes, 42)
})

test('bundle info format resolves json flag first', () => {
  assert.equal(resolveBundleInfoFormat({ format: 'table', json: false }), 'table')
  assert.equal(resolveBundleInfoFormat({ format: 'json', json: false }), 'json')
  assert.equal(resolveBundleInfoFormat({ format: 'table', json: true }), 'json')
})

const parseGraphMetafile = {
  inputs: {
    'quartz.config.ts': { bytes: 1, imports: [{ path: 'quartz/plugins/emitters/assets.ts' }] },
    'quartz/processors/parse.ts': { bytes: 1, imports: [{ path: 'quartz/util/path.ts' }] },
    'quartz/plugins/transformers/ofm.ts': {
      bytes: 1,
      imports: [
        { path: 'quartz/util/references/literal.ts' },
        { path: 'remark-parse', external: true },
        { path: 'quartz/util/absent.ts' },
      ],
    },
    'quartz/util/references/literal.ts': { bytes: 1, imports: [{ path: 'quartz/util/path.ts' }] },
    'quartz/util/path.ts': { bytes: 1, imports: [] },
    'quartz/plugins/emitters/assets.ts': { bytes: 1, imports: [] },
    'quartz/components/Head.tsx': { bytes: 1, imports: [] },
  },
  outputs: {},
}

test('parse source closure follows transformer imports transitively', () => {
  const closure = parseSourceInputs(parseGraphMetafile)
  assert.equal(closure.includes('quartz/util/references/literal.ts'), true)
  assert.equal(closure.includes('quartz/util/path.ts'), true)
})

test('parse source closure excludes emit-only and unresolved imports', () => {
  const closure = parseSourceInputs(parseGraphMetafile)
  assert.equal(closure.includes('quartz/components/Head.tsx'), false)
  assert.equal(closure.includes('quartz/plugins/emitters/assets.ts'), false)
  assert.equal(closure.includes('remark-parse'), false)
  assert.equal(closure.includes('quartz/util/absent.ts'), false)
})

test('parse source closure treats the config as a leaf', () => {
  const closure = parseSourceInputs(parseGraphMetafile)
  assert.equal(closure.includes('quartz.config.ts'), true)
  assert.equal(closure.includes('quartz/plugins/emitters/assets.ts'), false)
})

test('source watcher includes browser JavaScript and runtime data dependencies', () => {
  for (const filename of [
    'quartz/runtime/javascript/browser-worker.js',
    'quartz/runtime/python/ml-bridge.js',
    'quartz/runtime/lsp/pyodide-lock-v0.29.4.json',
    'quartz/util/pyright-browser-worker.mjs',
    'quartz/components/styles/custom.css',
    'pnpm-lock.yaml',
  ])
    assert.equal(isSourceWatchPath(filename), true, filename)
})

test('non-watch build flushes configuration import failure over pipes without esbuild deadlock', async t => {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'quartz-handlers-pipe-'))
  t.after(() => fs.rm(tmpDir, { recursive: true, force: true }))

  await fs.writeFile(
    path.join(tmpDir, 'package.json'),
    JSON.stringify({ name: 'quartz-pipe-fixture', version: '4.6.0', type: 'module' }),
  )
  await fs.symlink(path.join(repoRoot, 'node_modules'), path.join(tmpDir, 'node_modules'))
  await fs.mkdir(path.join(tmpDir, 'quartz/chunks'), { recursive: true })

  // Generate enough >4 KiB module inputs so --bundleInfo exceeds the 64 KiB OS pipe buffer.
  const imports = []
  for (let i = 0; i < 900; i++) {
    const name = `very-long-component-module-name-${String(i).padStart(4, '0')}.ts`
    const pad = 'x'.repeat(5000 + i)
    await fs.writeFile(path.join(tmpDir, 'quartz/chunks', name), `export const v${i} = '${pad}'\n`)
    imports.push(`import { v${i} } from './chunks/${name}'`)
  }
  await fs.writeFile(
    path.join(tmpDir, 'quartz/build.ts'),
    `${imports.join('\n')}\nimport 'missing-external-dependency-sentinel'\nexport default async () => ({ ready: true })\nexport const sum = [${Array.from({ length: 900 }, (_, i) => `v${i}`).join(',')}].length\n`,
  )

  const handlersEntry = path.join(repoRoot, 'quartz/cli/handlers.js')
  const runner = path.join(tmpDir, 'run.mjs')
  await fs.writeFile(
    runner,
    `import { handleBuild } from ${JSON.stringify(handlersEntry)}\nawait handleBuild({ directory: 'content', output: 'public', concurrency: 1, bundleInfo: true, watch: false, serve: false, verbose: false })\n`,
  )

  const child = spawn(process.execPath, [runner], {
    cwd: tmpDir,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  child.stdout.setEncoding('utf8')
  child.stderr.setEncoding('utf8')
  let stdout = ''
  let stderr = ''
  const drainStdout = (async () => {
    for await (const chunk of child.stdout) {
      stdout += chunk
      await new Promise(resolve => setTimeout(resolve, 15))
    }
  })()
  const drainStderr = (async () => {
    for await (const chunk of child.stderr) {
      stderr += chunk
    }
  })()
  const [[code]] = await Promise.all([once(child, 'close'), drainStdout, drainStderr])

  assert.equal(code, 1)
  const combined = `${stdout}\n${stderr}`
  assert.match(combined, /Couldn't parse Quartz configuration:/)
  assert.match(combined, /missing-external-dependency-sentinel/)
  assert.doesNotMatch(stderr, /all goroutines are asleep - deadlock!/)
})
