import assert from 'node:assert'
import test from 'node:test'
import { lex } from './lexer'

test('lexes regex literals with flags', () => {
  const result = lex('name.replace(/:/g, "-")')
  const regexToken = result.tokens.find(token => token.type === 'regex')
  if (!regexToken || regexToken.type !== 'regex') {
    throw new Error('expected regex token')
  }
  assert.strictEqual(regexToken.pattern, ':')
  assert.strictEqual(regexToken.flags, 'g')
})

test('lexes regex literals with escaped slashes', () => {
  const result = lex('path.matches(/\\//)')
  const regexToken = result.tokens.find(token => token.type === 'regex')
  if (!regexToken || regexToken.type !== 'regex') {
    throw new Error('expected regex token')
  }
  assert.strictEqual(regexToken.pattern, '\\/')
  assert.strictEqual(regexToken.flags, '')
})

test('lexes division as operator, not regex', () => {
  const result = lex('a / b')
  const operatorToken = result.tokens.find(
    token => token.type === 'operator' && token.value === '/',
  )
  assert.ok(operatorToken)
  const regexToken = result.tokens.find(token => token.type === 'regex')
  assert.strictEqual(regexToken, undefined)
})
