import assert from 'node:assert/strict'
import test from 'node:test'
import { compactQuoteText, findQuoteMatch } from './quote-highlights'

test('a saved quote finds its article passage across whitespace differences', () => {
  const text = compactQuoteText('Searching for their life’s work was a multi-turn endeavor.')
  const match = findQuoteMatch(text, {
    exact: 'their life’s work was a multi-turn endeavor',
    prefix: 'Searching for ',
    suffix: '.',
  })
  assert.deepEqual(match, {
    start: compactQuoteText('Searching for ').length,
    end: text.length - 1,
  })
})

test('surrounding text locates the intended copy of a repeated quote', () => {
  const text = compactQuoteText('alpha target beta gamma target delta')
  const quote = { exact: 'target', prefix: 'gamma ', suffix: ' delta' }
  assert.deepEqual(findQuoteMatch(text, quote), {
    start: compactQuoteText('alpha target beta gamma ').length,
    end: compactQuoteText('alpha target beta gamma target').length,
  })
  assert.equal(findQuoteMatch(text, { exact: 'target', prefix: '', suffix: '' }), null)
})
