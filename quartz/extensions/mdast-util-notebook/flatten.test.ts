import assert from 'node:assert'
import test, { describe } from 'node:test'
import type { NotebookDoc } from '../../util/notebook/types'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { notebookDocToFlatMdast } from './flatten'

function fixtureDoc(): NotebookDoc {
  return parseNotebookDoc(
    JSON.stringify({
      metadata: { language_info: { name: 'python' } },
      cells: [
        { cell_type: 'markdown', source: '# Heading\n\nbody text' },
        {
          cell_type: 'code',
          source: 'print("hi")',
          execution_count: 1,
          outputs: [{ output_type: 'stream', name: 'stdout', text: ['hi\n'] }],
        },
      ],
    }),
    'fixture.ipynb',
  ) as NotebookDoc
}

describe('notebookDocToFlatMdast', () => {
  test('produces only standard mdast node types at the root', () => {
    const tree = notebookDocToFlatMdast(fixtureDoc())
    for (const child of tree.children) {
      assert.notStrictEqual(child.type, 'notebookCell')
      assert.notStrictEqual(child.type, 'notebookOutput')
    }
  })

  test('emits html wrappers carrying cell frame metadata', () => {
    const tree = notebookDocToFlatMdast(fixtureDoc())
    const opens = tree.children
      .filter((child): child is { type: 'html'; value: string } => child.type === 'html')
      .map(child => child.value)
    assert.ok(opens.some(value => /data-notebook-cell-frame="cell-1"/.test(value)))
    assert.ok(opens.some(value => /notebook-flat-cell-code/.test(value)))
    assert.ok(opens.some(value => /notebook-flat-cell-markdown/.test(value)))
  })

  test('serializes stream outputs as html nodes', () => {
    const tree = notebookDocToFlatMdast(fixtureDoc())
    const htmlValues = tree.children
      .filter((child): child is { type: 'html'; value: string } => child.type === 'html')
      .map(child => child.value)
    assert.ok(htmlValues.some(value => /notebook-output-stream-stdout/.test(value)))
  })
})
