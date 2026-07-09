import assert from 'node:assert/strict'
import test from 'node:test'
import { encode, highlight, tokenizeTerm } from './search-text'

test('tokenizeTerm emits the longest phrase first', () => {
  assert.deepEqual(tokenizeTerm('alpha beta gamma'), [
    'alpha beta gamma',
    'alpha beta',
    'alpha',
    'gamma',
    'beta',
  ])
})

test('highlight treats regular expression characters as text', () => {
  assert.equal(highlight('a+b', 'a+b aab'), '<span class="highlight">a+b</span> aab')
})

test('highlight trims around the strongest matching window', () => {
  const words = Array.from({ length: 100 }, (_, index) =>
    index >= 70 && index <= 72 ? `target${index}` : `word${index}`,
  )
  const result = highlight('target', words.join(' '), true)
  assert.equal(result.startsWith('...'), true)
  assert.equal(result.endsWith('...'), false)
  assert.equal(result.includes('<span class="highlight">target</span>70'), true)
})

test('encode separates CJK characters from word tokens', () => {
  assert.deepEqual(encode('Alpha 日本 beta'), ['alpha', '日', '本', 'beta'])
})
