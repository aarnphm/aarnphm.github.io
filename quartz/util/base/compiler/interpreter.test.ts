import assert from 'node:assert'
import test from 'node:test'
import { ContentLayout } from '../../../plugins/emitters/contentIndex'
import { FilePath, FullSlug, SimpleSlug } from '../../path'
import { evaluateExpression, valueToUnknown, EvalContext } from './interpreter'
import { compileExpression } from './ir'
import { parseExpressionSource } from './parser'

const parseExpr = (source: string) => {
  const result = parseExpressionSource(source, 'test')
  if (!result.program.body) {
    throw new Error(`expected expression for ${source}`)
  }
  return compileExpression(result.program.body)
}

const makeCtx = (): EvalContext => {
  const fileA = {
    slug: 'a' as FullSlug,
    filePath: 'a.md' as FilePath,
    frontmatter: { title: 'A', pageLayout: 'default' as ContentLayout },
    links: [] as SimpleSlug[],
  }
  const fileB = {
    slug: 'b' as FullSlug,
    filePath: 'b.md' as FilePath,
    frontmatter: { title: 'B', pageLayout: 'default' as ContentLayout },
    links: ['a'] as SimpleSlug[],
  }
  return { file: fileA, allFiles: [fileA, fileB] }
}

test('link equality resolves to file targets', () => {
  const expr = parseExpr('link("a") == file("a")')
  const value = valueToUnknown(evaluateExpression(expr, makeCtx()))
  assert.strictEqual(value, true)
})

test('link equality matches raw string targets', () => {
  const expr = parseExpr('link("a") == "a"')
  const value = valueToUnknown(evaluateExpression(expr, makeCtx()))
  assert.strictEqual(value, true)
})

test('date arithmetic handles month additions', () => {
  const expr = parseExpr('date("2025-01-01") + "1M"')
  const value = valueToUnknown(evaluateExpression(expr, makeCtx()))
  assert.ok(value instanceof Date)
  assert.strictEqual(value.toISOString().split('T')[0], '2025-02-01')
})

test('date subtraction returns duration in ms', () => {
  const expr = parseExpr('date("2025-01-02") - date("2025-01-01")')
  const value = valueToUnknown(evaluateExpression(expr, makeCtx()))
  assert.strictEqual(value, 86400000)
})

test('list summary helpers compute statistics', () => {
  const meanExpr = parseExpr('([1, 2, 3]).mean()')
  const medianExpr = parseExpr('([1, 2, 3]).median()')
  const stddevExpr = parseExpr('([1, 2, 3]).stddev()')
  const sumExpr = parseExpr('([1, 2, 3]).sum()')
  const ctx = makeCtx()
  assert.strictEqual(valueToUnknown(evaluateExpression(meanExpr, ctx)), 2)
  assert.strictEqual(valueToUnknown(evaluateExpression(medianExpr, ctx)), 2)
  assert.strictEqual(valueToUnknown(evaluateExpression(sumExpr, ctx)), 6)
  const stddev = valueToUnknown(evaluateExpression(stddevExpr, ctx))
  assert.strictEqual(typeof stddev, 'number')
  if (typeof stddev === 'number') {
    assert.ok(Math.abs(stddev - Math.sqrt(2 / 3)) < 1e-6)
  }
})
