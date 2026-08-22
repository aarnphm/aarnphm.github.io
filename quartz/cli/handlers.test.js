import assert from 'node:assert/strict'
import test from 'node:test'
import {
  bundleInfoSummary,
  formatBundleInfoJson,
  formatBundleInfoTable,
  isSourceWatchPath,
  isTestSourcePath,
  parseSourceInputs,
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
  assert.equal(sourceWatchPatterns.includes('.claude/skills/**/*'), false)
  assert.equal(sourceWatchRoots.includes('.claude/skills'), false)
})

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

const parseGraphMetafile = {
  inputs: {
    'quartz.config.ts': { bytes: 1, imports: [{ path: 'quartz/plugins/emitters/assets.ts' }] },
    'quartz/processors/parse.ts': { bytes: 1, imports: [{ path: 'quartz/util/path.ts' }] },
    'quartz/plugins/transformers/ofm.ts': {
      bytes: 1,
      imports: [
        { path: 'quartz/util/wikilinks.ts' },
        { path: 'remark-parse', external: true },
        { path: 'quartz/util/absent.ts' },
      ],
    },
    'quartz/util/wikilinks.ts': { bytes: 1, imports: [{ path: 'quartz/util/path.ts' }] },
    'quartz/util/path.ts': { bytes: 1, imports: [] },
    'quartz/plugins/emitters/assets.ts': { bytes: 1, imports: [] },
    'quartz/components/Head.tsx': { bytes: 1, imports: [] },
  },
  outputs: {},
}

test('parse source closure follows transformer imports transitively', () => {
  const closure = parseSourceInputs(parseGraphMetafile)
  assert.equal(closure.includes('quartz/util/wikilinks.ts'), true)
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
