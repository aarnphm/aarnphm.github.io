import type { Paragraph } from 'mdast'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  canonicalizeCallout,
  formatMathCalloutTitle,
  isMathCallout,
  isStandaloneProofLine,
  italicizeProofLine,
  mathCalloutTitlePrefix,
} from './callouts'

describe('callout helpers', () => {
  test('canonicalizes math-specific callout aliases', () => {
    const cases = [
      ['math', 'math'],
      ['proof', 'proof'],
      ['lemma', 'lemma'],
      ['theory', 'theory'],
      ['propos', 'proposition'],
      ['thm', 'theorem'],
      ['def', 'definition'],
    ]

    for (const [inputType, expectedType] of cases) {
      const calloutType = canonicalizeCallout(inputType)

      assert.strictEqual(calloutType, expectedType)
      assert.strictEqual(isMathCallout(calloutType), true)
    }
  })

  test('keeps ordinary callouts outside the math family', () => {
    const calloutType = canonicalizeCallout('info')

    assert.strictEqual(calloutType, 'info')
    assert.strictEqual(isMathCallout(calloutType), false)
  })

  test('recognizes math callouts before alias canonicalization', () => {
    assert.strictEqual(isMathCallout('thm'), true)
    assert.strictEqual(isMathCallout('def'), true)
  })

  test('prefixes formal math callout titles when the type is absent', () => {
    assert.strictEqual(mathCalloutTitlePrefix('lemma', '3.3 nullspace penalty'), 'Lemma')
    assert.strictEqual(mathCalloutTitlePrefix('thm', 'Rank-Nullity Theorem'), undefined)
    assert.strictEqual(mathCalloutTitlePrefix('math', 'grouped cache reuse'), undefined)
  })

  test('formats numbered math callout titles', () => {
    assert.strictEqual(
      formatMathCalloutTitle('lemma', 'nullspace penalty', 'nullspace penalty', 2),
      'Lemma 2. nullspace penalty',
    )
    assert.strictEqual(
      formatMathCalloutTitle('lemma', '3.3, nullspace penalty', '3.3, nullspace penalty', 2),
      'Lemma 3.3. nullspace penalty',
    )
    assert.strictEqual(
      formatMathCalloutTitle(
        'proposition',
        '3.1 (sample-complexity gap)',
        '3.1 (sample-complexity gap)',
        2,
      ),
      'Proposition 3.1. sample-complexity gap',
    )
    assert.strictEqual(
      formatMathCalloutTitle('theorem', 'Rank-Nullity Theorem', 'Rank-Nullity Theorem', 1),
      'Theorem 1. Rank-Nullity Theorem',
    )
    assert.strictEqual(
      formatMathCalloutTitle('math', 'grouped cache reuse', 'grouped cache reuse', 1),
      '1. grouped cache reuse',
    )
  })

  test('italicizes leading proof labels in paragraphs', () => {
    const paragraph: Paragraph = {
      type: 'paragraph',
      children: [{ type: 'text', value: 'Proof: for any proposed function' }],
    }

    assert.strictEqual(italicizeProofLine(paragraph), true)
    assert.deepStrictEqual(paragraph.children, [
      { type: 'emphasis', children: [{ type: 'text', value: 'Proof' }] },
      { type: 'text', value: ': for any proposed function' },
    ])
  })

  test('detects standalone proof markers', () => {
    const standalone: Paragraph = {
      type: 'paragraph',
      children: [{ type: 'text', value: 'Proof:' }],
    }
    const inline: Paragraph = {
      type: 'paragraph',
      children: [{ type: 'text', value: 'Proof: because the diagonal flips membership' }],
    }

    assert.strictEqual(isStandaloneProofLine(standalone), true)
    assert.strictEqual(isStandaloneProofLine(inline), false)
  })
})
