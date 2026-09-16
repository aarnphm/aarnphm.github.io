import assert from 'node:assert/strict'
import test from 'node:test'
import { mathVariantText } from './math'

test('MathJax vectors, calligraphic functions, and real numbers retain their notation', () => {
  assert.equal(mathVariantText('XkIvA', 'bold'), '𝐗𝐤𝐈𝐯𝐀')
  assert.equal(mathVariantText('F(X)', 'script'), 'ℱ(𝒳)')
  assert.equal(mathVariantText('RCHNPQZ', 'double-struck'), 'ℝℂℍℕℙℚℤ')
})

test('Greek symbols, digits, and legacy letterlike code points map without changing operators', () => {
  assert.equal(mathVariantText('αβΓ∇∂ϵϑϰϕϱϖ + 123', 'bold'), '𝛂𝛃𝚪𝛁𝛛𝛜𝛝𝛞𝛟𝛠𝛡 + 𝟏𝟐𝟑')
  assert.equal(mathVariantText('h', 'italic'), 'ℎ')
  assert.equal(mathVariantText('BEFHILMRego', 'script'), 'ℬℰℱℋℐℒℳℛℯℊℴ')
  assert.equal(mathVariantText('CHIRZ', 'fraktur'), 'ℭℌℑℜℨ')
  assert.equal(mathVariantText('012', 'double-struck'), '𝟘𝟙𝟚')
  assert.equal(mathVariantText('x2', 'monospace'), '𝚡𝟸')
})

test('already styled, normal, and unsupported characters remain unchanged', () => {
  assert.equal(mathVariantText('𝐗 + ℝ ∈ ∞', 'bold'), '𝐗 + ℝ ∈ ∞')
  assert.equal(mathVariantText('x + β', 'normal'), 'x + β')
  assert.equal(mathVariantText('x', 'unknown'), 'x')
  assert.equal(mathVariantText('x', 'constructor'), 'x')
})
