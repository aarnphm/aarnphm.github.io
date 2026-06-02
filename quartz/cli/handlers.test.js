import assert from 'node:assert/strict'
import test from 'node:test'
import { isTestSourcePath, sourceWatchPatterns } from './handlers.js'

test('source watcher includes top-level Quartz source inputs', () => {
  assert.equal(sourceWatchPatterns.includes('quartz.config.ts'), true)
  assert.equal(sourceWatchPatterns.includes('quartz.layout.ts'), true)
})

test('source watcher ignores cli test files', () => {
  assert.equal(isTestSourcePath('quartz/cli/handlers.test.js'), true)
})
