import assert from 'node:assert'
import test, { describe } from 'node:test'
import { extractArxivId } from './citations'

describe('arxiv citation urls', () => {
  test('extracts arxiv ids from the arxiv host only', () => {
    assert.strictEqual(extractArxivId('https://arxiv.org/abs/2605.12290'), '2605.12290')
    assert.strictEqual(extractArxivId('https://arxiv.org/pdf/2605.12290v1.pdf'), '2605.12290')
    assert.strictEqual(extractArxivId('https://arxiv.org.evil.test/abs/2605.12290'), null)
    assert.strictEqual(extractArxivId('https://evil.test/https://arxiv.org/abs/2605.12290'), null)
  })
})
