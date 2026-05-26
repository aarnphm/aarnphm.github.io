import assert from 'node:assert/strict'
import test from 'node:test'
import type { ChangeEvent } from '../../../types/plugin'
import { classifyResourceChanges } from './change-classifier'

function change(path: string, type: ChangeEvent['type'] = 'change'): ChangeEvent {
  return { path, type } as ChangeEvent
}

test('classifies component resource partial emit changes by asset family', () => {
  const changes = classifyResourceChanges([
    change('quartz/styles/custom.scss'),
    change('quartz/runtime/notebook/client.ts'),
    change('quartz/components/scripts/notebook-runtime.inline.ts'),
    change('quartz/components/scripts/popover.inline.ts'),
    change('quartz/components/scripts/util.ts'),
    change('quartz/components/ArticleTitle.tsx'),
    change('quartz/components/multiplayer/ws.ts'),
    change('quartz/workers/semantic.worker.ts'),
    change('quartz/util/emojimap/codepoint-to-name.json'),
    change('quartz/workers/example.worker.ts', 'add'),
    change('quartz/workers/stale.worker.ts', 'delete'),
  ])

  assert.equal(changes.indexStylesheet, true)
  assert.equal(changes.notebookRuntime, true)
  assert.equal(changes.notebookRuntimePageScript, true)
  assert.equal(changes.pageScripts, true)
  assert.equal(changes.collaborativeComments, true)
  assert.equal(changes.semanticWorker, true)
  assert.equal(changes.semanticWorkerDeleted, false)
  assert.equal(changes.emoji, true)
  assert.deepEqual(
    changes.genericWorkerChanges.map(changeEvent => [changeEvent.type, changeEvent.path]),
    [
      ['add', 'quartz/workers/example.worker.ts'],
      ['delete', 'quartz/workers/stale.worker.ts'],
    ],
  )
})

test('classifies shared browser utilities as page script changes', () => {
  const changes = classifyResourceChanges([change('quartz/util/stacked-notes.ts')])

  assert.equal(changes.notebookRuntimePageScript, false)
  assert.equal(changes.pageScripts, true)
})

test('classifies semantic worker deletion separately from generic workers', () => {
  const changes = classifyResourceChanges([change('quartz/workers/semantic.worker.ts', 'delete')])

  assert.equal(changes.semanticWorker, true)
  assert.equal(changes.semanticWorkerDeleted, true)
  assert.deepEqual(changes.genericWorkerChanges, [])
})
