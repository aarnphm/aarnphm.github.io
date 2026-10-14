import assert from 'node:assert/strict'
import test from 'node:test'
import { sourceRebuildScope } from './change-classifier'

test('scopes source rebuilds by the server bundle closure', () => {
  const server = new Set([
    'quartz/util/triathlon-card.ts',
    'quartz/components/scripts/spa.inline.ts',
  ])
  const partial = (overrides: object) => ({
    kind: 'partial',
    staticFiles: false,
    styles: false,
    scripts: false,
    ...overrides,
  })

  assert.deepEqual(sourceRebuildScope([], server), { kind: 'full' })
  assert.deepEqual(sourceRebuildScope(['quartz/util/triathlon-card.ts'], server), { kind: 'full' })
  assert.deepEqual(sourceRebuildScope(['quartz.config.ts'], server), { kind: 'full' })
  assert.deepEqual(sourceRebuildScope(['quartz/worker.ts'], server), { kind: 'full' })
  assert.deepEqual(
    sourceRebuildScope(['quartz/components/styles/triathlon.scss', 'quartz/build.ts'], server),
    { kind: 'full' },
  )
  assert.deepEqual(
    sourceRebuildScope(
      [
        'quartz/components/scripts/spa.inline.ts',
        'quartz/components/triathlon/activity/workspace.ts',
      ],
      server,
    ),
    partial({ scripts: true }),
  )
  assert.deepEqual(
    sourceRebuildScope(['quartz/styles/pages/triathlon.scss'], server),
    partial({ styles: true }),
  )
  assert.deepEqual(
    sourceRebuildScope(['quartz/static/triathlon/mywindsock.svg'], server),
    partial({ staticFiles: true }),
  )
  assert.deepEqual(
    sourceRebuildScope(
      [
        'quartz/scripts/verify-semantic-index.ts',
        'quartz/embed_build.py',
        'quartz/util/garmin-connect.ts',
        'quartz/util/wahoo-cloud.ts',
        'quartz/util/lesswrong-preview.ts',
      ],
      server,
    ),
    { kind: 'full' },
  )
})

// A helper can gain a server caller after being introduced as a standalone utility.
test('server dependency closure takes precedence over historical exclusions', () => {
  for (const filename of [
    'quartz/util/lesswrong-preview.ts',
    'quartz/scripts/render.ts',
    'quartz/util/garmin-connect.ts',
  ]) {
    assert.deepEqual(sourceRebuildScope([filename], new Set([filename])), { kind: 'full' })
  }
  assert.deepEqual(sourceRebuildScope(['quartz/util/new-browser-helper.ts'], new Set()), {
    kind: 'partial',
    staticFiles: false,
    styles: false,
    scripts: true,
  })
  assert.deepEqual(sourceRebuildScope(['quartz/unknown/data.json'], new Set()), { kind: 'full' })
})

test('source resource scope includes shared styles, JavaScript helpers, and runtime data', () => {
  for (const filename of [
    'quartz/styles/variables.scss',
    'quartz/components/styles/collapseHeader.inline.scss',
    'quartz/components/styles/pet.scss',
  ]) {
    assert.deepEqual(sourceRebuildScope([filename], new Set()), {
      kind: 'partial',
      staticFiles: false,
      styles: true,
      scripts: false,
    })
  }
  for (const filename of [
    'quartz/components/scripts/root-lifecycle.ts',
    'quartz/runtime/python/ml-bridge.js',
  ]) {
    assert.deepEqual(sourceRebuildScope([filename], new Set()), {
      kind: 'partial',
      staticFiles: false,
      styles: false,
      scripts: true,
    })
  }
  assert.deepEqual(
    sourceRebuildScope(['quartz/runtime/lsp/pyodide-lock-v0.29.4.json'], new Set()),
    { kind: 'full' },
  )
})
