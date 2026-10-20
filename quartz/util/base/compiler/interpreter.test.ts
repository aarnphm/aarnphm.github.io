import assert from 'node:assert'
import test from 'node:test'
import { ContentLayout } from '../../../plugins/emitters/contentIndex'
import { FilePath, FullSlug, SimpleSlug } from '../../path'
import { evaluateExpression, valueToUnknown, EvalContext } from './interpreter'
import { compileExpression, JumpInstruction } from './ir'
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

test('Minato-Morreale ISOP recurrence eliminates absorption, consensus, contradiction, and tautology', async () => {
  const { optimizeFilterExprWithIsop } = await import('./isop')
  const parseRaw = (source: string) => {
    const result = parseExpressionSource(source, 'test')
    if (!result.program.body) throw new Error(`expected expression for ${source}`)
    return result.program.body
  }

  // 1. Absorption: A || (A && B) === A
  const absorbed = compileExpression(
    optimizeFilterExprWithIsop(
      parseRaw('file.inFolder("library") || (file.inFolder("library") && file.hasTag("book"))'),
    ),
  )
  assert.strictEqual(
    absorbed.instructions.filter(ins => ins.op === 'call_method' && ins.name === 'hasTag').length,
    0,
  )

  // 2. Consensus: (A && B) || (!A && C) || (B && C) === (A && B) || (!A && C)
  const consensusAst = optimizeFilterExprWithIsop(
    parseRaw(
      '(file.inFolder("library") && note.finished == true) || (!file.inFolder("library") && file.hasTag("book")) || (note.finished == true && file.hasTag("book"))',
    ),
  )
  const consensusIr = compileExpression(consensusAst)
  // In the reduced 2-cube DNF, hasTag("book") is evaluated only once (not twice)
  assert.strictEqual(
    consensusIr.instructions.filter(ins => ins.op === 'call_method' && ins.name === 'hasTag')
      .length,
    1,
  )

  // 3. Contradiction & Tautology
  const contradictionIr = compileExpression(
    optimizeFilterExprWithIsop(parseRaw('file.inFolder("library") && !file.inFolder("library")')),
  )
  assert.strictEqual(contradictionIr.instructions.length, 1)
  assert.strictEqual(contradictionIr.instructions[0]?.op, 'const')
  assert.strictEqual(
    (contradictionIr.instructions[0] as { op: 'const'; literal: { value: unknown } }).literal.value,
    false,
  )

  const tautologyIr = compileExpression(
    optimizeFilterExprWithIsop(parseRaw('file.inFolder("library") || !file.inFolder("library")')),
  )
  assert.strictEqual(tautologyIr.instructions.length, 1)
  assert.strictEqual(tautologyIr.instructions[0]?.op, 'const')
  assert.strictEqual(
    (tautologyIr.instructions[0] as { op: 'const'; literal: { value: unknown } }).literal.value,
    true,
  )
})

test('Minato-Morreale ISOP orders cheaper predicates first and compiles flat N-ary short-circuit jumps', async () => {
  const { optimizeFilterExprWithIsop } = await import('./isop')
  const raw = parseExpressionSource(
    'formula.heavy && note.category.containsAny("books") && file.inFolder("library") && file.ext == "md"',
    'test',
  ).program.body!
  const optimized = optimizeFilterExprWithIsop(raw)
  const ir = compileExpression(optimized)

  // All (k - 1) jump_if_false instructions in a k-term conjunction jump directly to the same final false target
  const falseJumps = ir.instructions.filter(
    (ins): ins is JumpInstruction => ins.op === 'jump_if_false',
  )
  assert.strictEqual(falseJumps.length, 3)
  const targets = new Set(falseJumps.map(j => j.target))
  assert.strictEqual(targets.size, 1)

  // Cheap scalar check (file.ext == "md") is evaluated before formula.heavy so non-md files never evaluate formula.heavy
  let formulaEvaluated = false
  const nonMdCtx: EvalContext = {
    file: {
      slug: 'library/poster' as FullSlug,
      filePath: 'library/poster.png' as FilePath,
      frontmatter: { title: 'Poster', pageLayout: 'default' as ContentLayout },
      links: [] as SimpleSlug[],
    },
    allFiles: [],
    formulas: {
      get heavy() {
        formulaEvaluated = true
        return parseExpr('true')
      },
    } as unknown as EvalContext['formulas'],
  }
  const result = valueToUnknown(evaluateExpression(ir, nonMdCtx))
  assert.strictEqual(result, false)
  assert.strictEqual(formulaEvaluated, false)
})
