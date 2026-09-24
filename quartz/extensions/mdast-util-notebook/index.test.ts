import assert from 'node:assert'
import test, { describe } from 'node:test'
import type { NotebookDoc } from '../../util/notebook/types'
import type { NotebookCell, NotebookOutput } from './types'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { notebookToMdast } from './index'

function parsedDoc(): NotebookDoc {
  const doc = parseNotebookDoc(
    JSON.stringify({
      metadata: { language_info: { name: 'python' } },
      cells: [
        { cell_type: 'markdown', source: '# Heading\n\nbody' },
        {
          cell_type: 'code',
          source: 'print("hi")',
          execution_count: 1,
          outputs: [{ output_type: 'stream', name: 'stdout', text: ['hi\n'] }],
        },
        {
          cell_type: 'code',
          source: 'x + 1',
          outputs: [
            { output_type: 'execute_result', data: { 'text/plain': '2' }, execution_count: 2 },
          ],
        },
      ],
    }),
    'fixture.ipynb',
  ) as NotebookDoc
  return doc
}

describe('mdast-util-notebook', () => {
  test('markdown cells get mdast children, code cells get code + output children', () => {
    const tree = notebookToMdast(parsedDoc())
    const mdCell = tree.children[0] as unknown as NotebookCell
    assert.strictEqual(mdCell.data.notebookCell.cellType, 'markdown')
    assert.ok(mdCell.children.some(child => child.type === 'heading'))
    assert.ok(mdCell.children.some(child => child.type === 'paragraph'))

    const codeCell = tree.children[1] as unknown as NotebookCell
    assert.ok(codeCell.children.some(child => child.type === 'code'))
    assert.ok(codeCell.children.some(child => child.type === 'notebookOutput'))
  })

  test('execute_result outputs carry executionCount', () => {
    const tree = notebookToMdast(parsedDoc())
    const codeCell = tree.children[2] as unknown as NotebookCell
    const output = codeCell.children.find(
      (child): child is NotebookOutput => child.type === 'notebookOutput',
    )
    assert.ok(output)
    assert.strictEqual(output.data.notebookOutput.kind, 'execute_result')
    assert.strictEqual(output.data.notebookOutput.executionCount, 2)
  })
})
