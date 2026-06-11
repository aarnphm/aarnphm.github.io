import assert from 'node:assert/strict'
import test from 'node:test'
import {
  bundleInfoSummary,
  formatBundleInfoJson,
  formatBundleInfoTable,
  isSourceWatchPath,
  isTestSourcePath,
  resolveBundleInfoFormat,
  sourceWatchPatterns,
  sourceWatchRoots,
} from './handlers.js'

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

test('source watcher includes top-level Quartz source inputs', () => {
  assert.equal(sourceWatchPatterns.includes('quartz.config.ts'), true)
  assert.equal(sourceWatchPatterns.includes('quartz.layout.ts'), true)
  assert.equal(sourceWatchRoots.includes('quartz'), true)
})

test('source watcher ignores cli test files', () => {
  assert.equal(isTestSourcePath('quartz/cli/handlers.test.js'), true)
})

test('source watcher accepts newly added Quartz source files', () => {
  assert.equal(isSourceWatchPath('quartz/util/transclude-props.ts'), true)
  assert.equal(isSourceWatchPath('quartz/components/renderPage.tsx'), true)
  assert.equal(isSourceWatchPath('quartz/.quartz-cache/transpiled-build.mjs'), false)
  assert.equal(isSourceWatchPath('quartz/util/transclude-props.test.ts'), false)
})

test('bundle info summarizes the configured Quartz output', () => {
  assert.deepEqual(bundleInfoSummary(bundleMetafile), {
    outputFile: 'quartz/.quartz-cache/transpiled-build.mjs',
    inputCount: 2,
    bytes: 42,
    bytesText: '42 B',
  })
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
