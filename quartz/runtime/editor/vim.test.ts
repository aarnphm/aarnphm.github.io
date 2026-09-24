import { Text } from '@codemirror/state'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  notebookCursorCoordinateInText,
  notebookLeapMotionForKey,
  notebookLeapTargets,
  notebookLeapTargetsInText,
  notebookSurroundingPairRangeInText,
  notebookSurroundKeyPlan,
  notebookWhitespaceGlyphs,
  notebookWordRangeAt,
} from './code-editor'

describe('notebook whitespace glyphs', () => {
  test('mirrors local vim listchars for leading spaces, tabs, nbsp, and trails', () => {
    assert.deepStrictEqual(notebookWhitespaceGlyphs('    value  ', 10), [
      { from: 10, to: 11, text: '»', kind: 'lead' },
      { from: 11, to: 12, text: '·', kind: 'lead' },
      { from: 12, to: 13, text: '·', kind: 'lead' },
      { from: 13, to: 14, text: '·', kind: 'lead' },
      { from: 19, to: 20, text: '·', kind: 'trail' },
      { from: 20, to: 21, text: '·', kind: 'trail' },
    ])

    assert.deepStrictEqual(notebookWhitespaceGlyphs('\t\u00a0x\t ', 2), [
      { from: 2, to: 3, text: '»·', kind: 'tab' },
      { from: 3, to: 4, text: '+', kind: 'nbsp' },
      { from: 5, to: 6, text: '»·', kind: 'tab' },
      { from: 6, to: 7, text: '·', kind: 'trail' },
    ])
  })

  test('keeps interior ordinary spaces invisible', () => {
    assert.deepStrictEqual(notebookWhitespaceGlyphs('value = item'), [])
  })
})

describe('notebook surround ranges', () => {
  test('finds the vim word under the cursor', () => {
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 2), { from: 0, to: 5 })
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 5), { from: 0, to: 5 })
    assert.deepStrictEqual(notebookWordRangeAt('alpha beta', 7), { from: 6, to: 10 })
    assert.strictEqual(notebookWordRangeAt('  ', 1), undefined)
  })

  test('finds bracket and quote surrounds from CodeMirror Text', () => {
    assert.deepStrictEqual(notebookSurroundingPairRangeInText(Text.of(['call(foo)']), 6, ')'), {
      openFrom: 4,
      openTo: 5,
      closeFrom: 8,
      closeTo: 9,
    })
    assert.deepStrictEqual(notebookSurroundingPairRangeInText(Text.of(['call(foo)']), 8, '('), {
      openFrom: 4,
      openTo: 5,
      closeFrom: 8,
      closeTo: 9,
    })
    assert.deepStrictEqual(notebookSurroundingPairRangeInText(Text.of(['"foo"']), 2, '"'), {
      openFrom: 0,
      openTo: 1,
      closeFrom: 4,
      closeTo: 5,
    })
    assert.deepStrictEqual(notebookSurroundingPairRangeInText(Text.of(['"foo"']), 4, '"'), {
      openFrom: 0,
      openTo: 1,
      closeFrom: 4,
      closeTo: 5,
    })
  })

  test('finds surrounds across CodeMirror Text line break chunks', () => {
    assert.deepStrictEqual(
      notebookSurroundingPairRangeInText(Text.of(['call(', 'foo', ')']), 7, ')'),
      { openFrom: 4, openTo: 5, closeFrom: 10, closeTo: 11 },
    )
  })

  test('ignores unmatched closes before the selected CodeMirror Text surround', () => {
    assert.deepStrictEqual(notebookSurroundingPairRangeInText(Text.of([') call(foo)']), 8, ')'), {
      openFrom: 6,
      openTo: 7,
      closeFrom: 10,
      closeTo: 11,
    })
  })
})

describe('notebook surround key planning', () => {
  test('captures operator-prefixed surround sequences before vim consumes the operator', () => {
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'y', false), {
      kind: 'pending',
      buffer: 'y',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ysi', 'w', false), {
      kind: 'pending',
      buffer: 'ysiw',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ysiw', ')', false), {
      kind: 'surroundWord',
      token: ')',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('yss', 'B', false), {
      kind: 'surroundLine',
      token: 'B',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ds', '"', false), {
      kind: 'deleteSurround',
      token: '"',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('cs)', ']', false), {
      kind: 'changeSurround',
      oldToken: ')',
      replacementToken: ']',
    })
  })

  test('captures visual surround and flushes non-surround continuations', () => {
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'S', true), { kind: 'pending', buffer: 'S' })
    assert.deepStrictEqual(notebookSurroundKeyPlan('S', '}', true), {
      kind: 'surroundSelection',
      token: '}',
    })
    assert.deepStrictEqual(notebookSurroundKeyPlan('ys', 'x', false), { kind: 'flush' })
    assert.deepStrictEqual(notebookSurroundKeyPlan('', 'w', false), { kind: 'pass' })
  })
})

describe('notebook leap char motions', () => {
  test('applies backward and till offsets', () => {
    const backward = notebookLeapMotionForKey('F')
    const forwardTill = notebookLeapMotionForKey('t')
    const backwardTill = notebookLeapMotionForKey('T')
    assert(backward)
    assert(forwardTill)
    assert(backwardTill)

    assert.deepStrictEqual(notebookLeapTargets('abacad', 5, 'a', backward), [
      { matchFrom: 4, matchTo: 5, target: 4 },
      { matchFrom: 2, matchTo: 3, target: 2 },
      { matchFrom: 0, matchTo: 1, target: 0 },
    ])
    assert.deepStrictEqual(notebookLeapTargets('abacad', 0, 'a', forwardTill), [
      { matchFrom: 2, matchTo: 3, target: 1 },
      { matchFrom: 4, matchTo: 5, target: 3 },
    ])
    assert.deepStrictEqual(notebookLeapTargets('abacad', 5, 'a', backwardTill), [
      { matchFrom: 4, matchTo: 5, target: 5 },
      { matchFrom: 2, matchTo: 3, target: 3 },
      { matchFrom: 0, matchTo: 1, target: 1 },
    ])
  })

  test('finds targets from CodeMirror Text without flattening the document', () => {
    const motion = notebookLeapMotionForKey('f')
    assert(motion)
    const doc = Text.of(['alpha beta', 'banana'])
    assert.deepStrictEqual(notebookLeapTargetsInText(doc, 0, 'a', motion), [
      { matchFrom: 4, matchTo: 5, target: 4 },
      { matchFrom: 9, matchTo: 10, target: 9 },
      { matchFrom: 12, matchTo: 13, target: 12 },
      { matchFrom: 14, matchTo: 15, target: 14 },
      { matchFrom: 16, matchTo: 17, target: 16 },
    ])
  })

  test('finds targets across CodeMirror Text line break chunks', () => {
    const motion = notebookLeapMotionForKey('f')
    assert(motion)
    const doc = Text.of(['a', 'a'])
    assert.deepStrictEqual(notebookLeapTargetsInText(doc, 0, 'a', motion), [
      { matchFrom: 2, matchTo: 3, target: 2 },
    ])
  })
})

describe('notebook editor status text', () => {
  test('reports one-based cursor coordinates from CodeMirror Text', () => {
    const doc = Text.of(['one', 'two'])
    assert.deepStrictEqual(notebookCursorCoordinateInText(doc, 0), { line: 1, column: 1 })
    assert.deepStrictEqual(notebookCursorCoordinateInText(doc, 3), { line: 1, column: 4 })
    assert.deepStrictEqual(notebookCursorCoordinateInText(doc, 4), { line: 2, column: 1 })
    assert.deepStrictEqual(notebookCursorCoordinateInText(doc, 99), { line: 2, column: 4 })
  })
})
