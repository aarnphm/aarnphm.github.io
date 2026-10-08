import assert from 'node:assert/strict'
import test from 'node:test'
import type { ChangeEvent } from '../../../types/plugin'
import {
  classifyResourceChanges,
  hasComponentResourceChanges,
  sourceRebuildScope,
} from './change-classifier'

function change(path: string, type: ChangeEvent['type'] = 'change'): ChangeEvent {
  return { path, type } as ChangeEvent
}

test('classifies component resource partial emit changes by asset family', () => {
  const changes = classifyResourceChanges([
    change('quartz/styles/custom.scss'),
    change('quartz/runtime/notebook/client.ts'),
    change('quartz/components/scripts/notebook-runtime.inline.ts'),
    change('quartz/components/scripts/popover.inline.ts'),
    change('quartz/components/scripts/nav-lifecycle.ts'),
    change('quartz/components/ArticleTitle.tsx'),
    change('quartz/components/multiplayer/ws.ts'),
    change('quartz/workers/semantic.worker.ts'),
    change('quartz/util/emojimap/codepoint-to-name.json'),
    change('quartz/plugins/emitters/component-resources/xslt-polyfill-assets.ts'),
    change('quartz/workers/example.worker.ts', 'add'),
    change('quartz/workers/stale.worker.ts', 'delete'),
  ])

  assert.equal(changes.indexStylesheet, true)
  assert.equal(changes.componentStyles, false)
  assert.equal(changes.staticStyles, false)
  assert.equal(changes.staticScripts, false)
  assert.equal(changes.notebookRuntime, true)
  assert.equal(changes.notebookRuntimePageScript, true)
  assert.equal(changes.pageScripts, true)
  assert.equal(changes.collaborativeComments, true)
  assert.equal(changes.semanticWorker, true)
  assert.equal(changes.semanticWorkerDeleted, false)
  assert.equal(changes.emoji, true)
  assert.equal(changes.xsltPolyfill, true)
  assert.deepEqual(
    changes.genericWorkerChanges.map(changeEvent => [changeEvent.type, changeEvent.path]),
    [
      ['add', 'quartz/workers/example.worker.ts'],
      ['delete', 'quartz/workers/stale.worker.ts'],
    ],
  )
})

test('classifies shared browser utilities as page script changes', () => {
  const changes = classifyResourceChanges([
    change('quartz/util/stacked-notes.ts'),
    change('quartz/util/fetch-canonical.ts'),
    change('quartz/util/search-text.ts'),
  ])

  assert.equal(changes.notebookRuntimePageScript, false)
  assert.equal(changes.pageScripts, true)
  assert.equal(
    classifyResourceChanges([change('quartz/util/fetch-canonical.test.ts')]).pageScripts,
    false,
  )
})

test('classifies global component styles as index stylesheet changes', () => {
  for (const stylesheet of ['popover', 'twitter']) {
    const changes = classifyResourceChanges([change(`quartz/components/styles/${stylesheet}.scss`)])
    assert.equal(changes.indexStylesheet, true)
    assert.equal(changes.componentStyles, false)
    assert.equal(changes.staticStyles, false)
  }
})

test('classifies mdx component styles as component stylesheet changes', () => {
  const changes = classifyResourceChanges([
    change('quartz/components/styles/attentionCircuits.scss'),
  ])

  assert.equal(changes.indexStylesheet, false)
  assert.equal(changes.componentStyles, true)
  assert.equal(changes.staticStyles, false)
  assert.equal(hasComponentResourceChanges(changes), true)
})

test('classifies extracted static resources separately from page scripts and component styles', () => {
  const changes = classifyResourceChanges([
    change('quartz/components/styles/collapseHeader.inline.scss'),
    change('quartz/components/scripts/transclude.inline.ts'),
  ])

  assert.equal(changes.componentStyles, false)
  assert.equal(changes.staticStyles, true)
  assert.equal(changes.staticScripts, true)
  assert.equal(changes.pageScripts, false)
  assert.equal(hasComponentResourceChanges(changes), true)
})

test('classifies semantic worker deletion separately from generic workers', () => {
  const changes = classifyResourceChanges([change('quartz/workers/semantic.worker.ts', 'delete')])

  assert.equal(changes.semanticWorker, true)
  assert.equal(changes.semanticWorkerDeleted, true)
  assert.deepEqual(changes.genericWorkerChanges, [])
})

test('detects when component resources can skip content-only partial emits', () => {
  const contentChanges = classifyResourceChanges([change('content/index.md')])
  const scriptChanges = classifyResourceChanges([
    change('quartz/components/scripts/popover.inline.ts'),
  ])

  assert.equal(hasComponentResourceChanges(contentChanges), false)
  assert.equal(hasComponentResourceChanges(scriptChanges), true)
})

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
      ['quartz/scripts/verify-semantic-index.ts', 'quartz/embed_build.py'],
      server,
    ),
    partial({}),
  )
})

test('recompiles component styles when a shared partial changes', () => {
  assert.equal(
    classifyResourceChanges([change('quartz/styles/variables.scss')]).componentStyles,
    true,
  )
  assert.equal(
    classifyResourceChanges([change('quartz/styles/pages/triathlon.scss')]).componentStyles,
    false,
  )
})
