import assert from 'node:assert/strict'
import test, { describe } from 'node:test'
import { haskellNotebookSource } from './worker.js'

describe('haskell worker source preparation', () => {
  test('adds a no-op main for definition-only cells', () => {
    assert.strictEqual(
      haskellNotebookSource('data Exp = Id Char deriving (Eq, Show)\nexpression s = s'),
      'data Exp = Id Char deriving (Eq, Show)\nexpression s = s\n\nmain :: IO ()\nmain = pure ()\n',
    )
  })

  test('keeps executable cells unchanged when they define main', () => {
    const source = 'main :: IO ()\nmain = putStrLn "hi"\n'
    assert.strictEqual(haskellNotebookSource(source), source)
  })

  test('does not confuse nested bindings for the program entrypoint', () => {
    assert.strictEqual(
      haskellNotebookSource('outer = let main = 1 in main\n'),
      'outer = let main = 1 in main\n\nmain :: IO ()\nmain = pure ()\n',
    )
  })
})
