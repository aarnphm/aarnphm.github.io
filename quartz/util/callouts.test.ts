import type { Paragraph } from 'mdast'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  canonicalizeCallout,
  isMathCallout,
  isStandaloneProofLine,
  italicizeProofLine,
} from './callouts'

describe('callout helpers', () => {
  test('canonicalizes math-specific callout aliases', () => {
    const cases = [
      ['math', 'math'],
      ['proof', 'proof'],
      ['lemma', 'lemma'],
      ['theory', 'theory'],
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
