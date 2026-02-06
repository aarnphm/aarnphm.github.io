import assert from 'node:assert'
import test from 'node:test'
import { parseExpressionSource } from './parser'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null

const strip = (node: unknown): unknown => {
  if (!isRecord(node)) return node
  const type = node.type
  if (type === 'Identifier') {
    return { type, name: node.name }
  }
  if (type === 'Literal') {
    const kind = node.kind
    const value = node.value
    const flags = node.flags
    return flags !== undefined ? { type, kind, value, flags } : { type, kind, value }
  }
  if (type === 'UnaryExpr') {
    return { type, operator: node.operator, argument: strip(node.argument) }
  }
  if (type === 'BinaryExpr' || type === 'LogicalExpr') {
    return { type, operator: node.operator, left: strip(node.left), right: strip(node.right) }
  }
  if (type === 'CallExpr') {
    const args = Array.isArray(node.args) ? node.args.map(strip) : []
    return { type, callee: strip(node.callee), args }
  }
  if (type === 'MemberExpr') {
    return { type, object: strip(node.object), property: node.property }
  }
  if (type === 'IndexExpr') {
    return { type, object: strip(node.object), index: strip(node.index) }
  }
  if (type === 'ListExpr') {
    const elements = Array.isArray(node.elements) ? node.elements.map(strip) : []
    return { type, elements }
  }
  if (type === 'ErrorExpr') {
    return { type, message: node.message }
  }
  return node
}

test('ebnf to ast mapping snapshots', () => {
  const cases: Array<{ source: string; expected: unknown }> = [
    {
      source: 'status == "done"',
      expected: {
        type: 'BinaryExpr',
        operator: '==',
        left: { type: 'Identifier', name: 'status' },
        right: { type: 'Literal', kind: 'string', value: 'done' },
      },
    },
    {
      source: '!done',
      expected: {
        type: 'UnaryExpr',
        operator: '!',
        argument: { type: 'Identifier', name: 'done' },
      },
    },
    {
      source: 'file.ctime',
      expected: {
        type: 'MemberExpr',
        object: { type: 'Identifier', name: 'file' },
        property: 'ctime',
      },
    },
    {
      source: 'note["my-field"]',
      expected: {
        type: 'IndexExpr',
        object: { type: 'Identifier', name: 'note' },
        index: { type: 'Literal', kind: 'string', value: 'my-field' },
      },
    },
    {
      source: 'date(due) < today()',
      expected: {
        type: 'BinaryExpr',
        operator: '<',
        left: {
          type: 'CallExpr',
          callee: { type: 'Identifier', name: 'date' },
          args: [{ type: 'Identifier', name: 'due' }],
        },
        right: { type: 'CallExpr', callee: { type: 'Identifier', name: 'today' }, args: [] },
      },
    },
    {
      source: 'now() - file.ctime',
      expected: {
        type: 'BinaryExpr',
        operator: '-',
        left: { type: 'CallExpr', callee: { type: 'Identifier', name: 'now' }, args: [] },
        right: {
          type: 'MemberExpr',
          object: { type: 'Identifier', name: 'file' },
          property: 'ctime',
        },
      },
    },
    {
      source: '(pages * 2).round(0)',
      expected: {
        type: 'CallExpr',
        callee: {
          type: 'MemberExpr',
          object: {
            type: 'BinaryExpr',
            operator: '*',
            left: { type: 'Identifier', name: 'pages' },
            right: { type: 'Literal', kind: 'number', value: 2 },
          },
          property: 'round',
        },
        args: [{ type: 'Literal', kind: 'number', value: 0 }],
      },
    },
    {
      source: 'tags.containsAny("a","b")',
      expected: {
        type: 'CallExpr',
        callee: {
          type: 'MemberExpr',
          object: { type: 'Identifier', name: 'tags' },
          property: 'containsAny',
        },
        args: [
          { type: 'Literal', kind: 'string', value: 'a' },
          { type: 'Literal', kind: 'string', value: 'b' },
        ],
      },
    },
    {
      source: 'list(links).filter(value.isTruthy())',
      expected: {
        type: 'CallExpr',
        callee: {
          type: 'MemberExpr',
          object: {
            type: 'CallExpr',
            callee: { type: 'Identifier', name: 'list' },
            args: [{ type: 'Identifier', name: 'links' }],
          },
          property: 'filter',
        },
        args: [
          {
            type: 'CallExpr',
            callee: {
              type: 'MemberExpr',
              object: { type: 'Identifier', name: 'value' },
              property: 'isTruthy',
            },
            args: [],
          },
        ],
      },
    },
    {
      source: '["a", "b", "c"].length',
      expected: {
        type: 'MemberExpr',
        object: {
          type: 'ListExpr',
          elements: [
            { type: 'Literal', kind: 'string', value: 'a' },
            { type: 'Literal', kind: 'string', value: 'b' },
            { type: 'Literal', kind: 'string', value: 'c' },
          ],
        },
        property: 'length',
      },
    },
    {
      source: 'this.file.name',
      expected: {
        type: 'MemberExpr',
        object: {
          type: 'MemberExpr',
          object: { type: 'Identifier', name: 'this' },
          property: 'file',
        },
        property: 'name',
      },
    },
    {
      source: 'a || b && c',
      expected: {
        type: 'LogicalExpr',
        operator: '||',
        left: { type: 'Identifier', name: 'a' },
        right: {
          type: 'LogicalExpr',
          operator: '&&',
          left: { type: 'Identifier', name: 'b' },
          right: { type: 'Identifier', name: 'c' },
        },
      },
    },
    {
      source: 'values[0]',
      expected: {
        type: 'IndexExpr',
        object: { type: 'Identifier', name: 'values' },
        index: { type: 'Literal', kind: 'number', value: 0 },
      },
    },
  ]

  for (const entry of cases) {
    const result = parseExpressionSource(entry.source)
    assert.strictEqual(result.diagnostics.length, 0)
    assert.deepStrictEqual(strip(result.program.body), entry.expected)
  }
})

test('syntax doc samples parse', () => {
  const samples = [
    'note["price"]',
    'file.size > 10',
    'file.hasLink(this.file)',
    'date("2024-12-01") + "1M" + "4h" + "3m"',
    'now() - file.ctime',
    'property[0]',
    'link("filename", icon("plus"))',
    'file.mtime > now() - "1 week"',
    '/abc/.matches("abcde")',
    'name.replace(/:/g, "-")',
    'values.filter(value.isType("number")).reduce(if(acc == null || value > acc, value, acc), null)',
  ]

  for (const source of samples) {
    const result = parseExpressionSource(source)
    assert.strictEqual(result.diagnostics.length, 0)
    assert.ok(result.program.body)
  }
})

test('string escapes are decoded', () => {
  const result = parseExpressionSource('"a\\n\\"b"')
  assert.strictEqual(result.diagnostics.length, 0)
  const literal = strip(result.program.body)
  if (!isRecord(literal)) {
    throw new Error('expected literal record')
  }
  assert.strictEqual(literal.type, 'Literal')
  assert.strictEqual(literal.kind, 'string')
  assert.strictEqual(literal.value, 'a\n"b')
})

test('parser reports errors and recovers', () => {
  const result = parseExpressionSource('status ==')
  assert.ok(result.diagnostics.length > 0)
  assert.ok(result.program.body)
})
