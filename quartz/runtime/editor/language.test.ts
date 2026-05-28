import type { Extension } from '@codemirror/state'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import { codemirrorCodeLanguage } from '../../util/codemirror-language'
import { notebookCodeEditorLanguageExtension } from './code-editor'

function isEmptyExtension(extension: Extension): boolean {
  return Array.isArray(extension) && extension.length === 0
}

describe('CodeMirror language extensions', () => {
  test('notebook editor loads every executable runtime language', async () => {
    const languages = [
      'python',
      'javascript',
      'c',
      'cpp',
      'go',
      'rust',
      'haskell',
      'ocaml',
      'mojo',
      'wasm',
      'bash',
    ]

    for (const language of languages) {
      const extension = await notebookCodeEditorLanguageExtension(language)
      assert.strictEqual(isEmptyExtension(extension), false, language)
    }
  })

  test('markdown code editor loads runtime and shell languages', () => {
    const languages = ['c', 'cpp', 'wasm', 'rust', 'haskell', 'ocaml', 'mojo', 'bash']

    for (const language of languages) {
      const extension = codemirrorCodeLanguage(language)
      assert.strictEqual(isEmptyExtension(extension), false, language)
    }
  })
})
