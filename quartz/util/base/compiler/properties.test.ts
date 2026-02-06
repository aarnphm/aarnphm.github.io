import assert from 'node:assert'
import test from 'node:test'
import { parseExpressionSource } from './parser'
import { buildPropertyExpressionSource } from './properties'

test('builds property expression sources', () => {
  const cases: Array<{ input: string; expected: string }> = [
    { input: 'status', expected: 'note.status' },
    { input: 'note.status', expected: 'note.status' },
    { input: 'file.name', expected: 'file.name' },
    { input: 'file.my-field', expected: 'file["my-field"]' },
    { input: 'my-field', expected: 'note["my-field"]' },
    { input: 'note["my field"]', expected: 'note["my field"]' },
    { input: 'formula.total', expected: 'formula.total' },
    { input: 'this.file.name', expected: 'this.file.name' },
    { input: 'a.b-c.d', expected: 'note.a["b-c"].d' },
    { input: 'date(file.ctime)', expected: 'date(file.ctime)' },
  ]

  for (const entry of cases) {
    const result = buildPropertyExpressionSource(entry.input)
    assert.strictEqual(result, entry.expected)
    const parsed = parseExpressionSource(entry.expected)
    assert.strictEqual(parsed.diagnostics.length, 0)
    assert.ok(parsed.program.body)
  }
})
