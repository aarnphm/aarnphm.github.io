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

test('highlight does not match later query terms inside generated markup', () => {
  assert.equal(
    highlight('estimating a robot position', 'estimating a robot position'),
    '<span class="highlight">estimating a robot position</span>',
  )
  assert.equal(
    highlight('robot a', 'robot range'),
    '<span class="highlight">robot</span> r<span class="highlight">a</span>nge',
  )
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

test('highlight keeps the last equally strong excerpt window', () => {
  const words = Array.from({ length: 100 }, (_, index) =>
    index === 30 || index === 61 ? `target${index}` : `word${index}`,
  )
  const expected = words
    .slice(31, 92)
    .map(word => (word === 'target61' ? '<span class="highlight">target</span>61' : word))
    .join(' ')
  assert.equal(highlight('target', words.join(' '), true), `...${expected}...`)
})

test('highlight preserves unmatched excerpt selection and short documents', () => {
  const words = Array.from({ length: 100 }, (_, index) => `word${index}`)
  assert.equal(highlight('absent', words.join(' '), true), `...${words.slice(40).join(' ')}`)
  assert.equal(
    highlight('target', 'before TARGET after', true),
    'before <span class="highlight">TARGET</span> after',
  )
  assert.equal(highlight('target', '', true), '')
})

test('encode separates CJK characters from word tokens', () => {
  assert.deepEqual(encode('Alpha 日本 beta'), ['alpha', '日', '本', 'beta'])
})
