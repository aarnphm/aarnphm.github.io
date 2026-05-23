import assert from 'node:assert'
import test, { describe } from 'node:test'
import { hostnameMatches, parseExternalUrl } from './url'

describe('url helpers', () => {
  test('matches exact hosts and subdomains only', () => {
    assert.strictEqual(hostnameMatches(new URL('https://github.com/a'), 'github.com'), true)
    assert.strictEqual(hostnameMatches(new URL('https://docs.github.com/a'), 'github.com'), true)
    assert.strictEqual(hostnameMatches(new URL('https://evilgithub.com/a'), 'github.com'), false)
    assert.strictEqual(
      hostnameMatches(new URL('https://github.com.evil.test/a'), 'github.com'),
      false,
    )
  })

  test('parses absolute and protocol-relative external URLs', () => {
    assert.strictEqual(parseExternalUrl('/internal/path'), undefined)
    assert.strictEqual(parseExternalUrl('github.com/a'), undefined)
    assert.strictEqual(parseExternalUrl('https://github.com/a')?.hostname, 'github.com')
    assert.strictEqual(parseExternalUrl('//github.com/a')?.hostname, 'github.com')
  })
})
