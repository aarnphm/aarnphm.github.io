import { ChangeSet, Text } from '@codemirror/state'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import { codemirrorChangedTextIsBlank, codemirrorTextIsBlank } from './codemirror-text'

describe('CodeMirror Text helpers', () => {
  test('detects blank documents from rope chunks', () => {
    assert.strictEqual(codemirrorTextIsBlank(Text.of(['', '   ', '\t'])), true)
    assert.strictEqual(codemirrorTextIsBlank(Text.of(['', '  ok'])), false)
  })

  test('updates blank state from CodeMirror change sets', () => {
    const blank = Text.of([''])
    const insertWhitespace = ChangeSet.of({ from: 0, insert: '  ' }, blank.length)
    assert.strictEqual(
      codemirrorChangedTextIsBlank(true, insertWhitespace.apply(blank), insertWhitespace),
      true,
    )

    const insertText = ChangeSet.of({ from: 0, insert: 'x' }, blank.length)
    assert.strictEqual(
      codemirrorChangedTextIsBlank(true, insertText.apply(blank), insertText),
      false,
    )

    const text = Text.of(['ok'])
    const appendWhitespace = ChangeSet.of({ from: text.length, insert: ' ' }, text.length)
    assert.strictEqual(
      codemirrorChangedTextIsBlank(false, appendWhitespace.apply(text), appendWhitespace),
      false,
    )

    const deleteText = ChangeSet.of({ from: 0, to: text.length }, text.length)
    assert.strictEqual(
      codemirrorChangedTextIsBlank(false, deleteText.apply(text), deleteText),
      true,
    )
  })
})
