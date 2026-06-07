import assert from 'node:assert/strict'
import test from 'node:test'
import {
  isSourceWatchPath,
  isTestSourcePath,
  sourceWatchPatterns,
  sourceWatchRoots,
} from './handlers.js'

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
