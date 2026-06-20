import assert from 'node:assert/strict'
import test from 'node:test'
import { triathlonShortcutRedirectUrl } from './triathlon-shortcut'

test('redirects triathlon shortcut documents to canonical triathlon paths', () => {
  const cases: [string, string][] = [
    ['https://t.aarnphm.xyz', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/analytics', 'https://aarnphm.xyz/triathlon/analytics'],
    ['https://t.aarnphm.xyz/tools', 'https://aarnphm.xyz/triathlon/tools'],
    ['https://t.aarnphm.xyz/maps', 'https://aarnphm.xyz/triathlon/maps'],
    ['https://t.aarnphm.xyz/training', 'https://aarnphm.xyz/triathlon/training'],
    ['https://t.aarnphm.xyz/triathlon', 'https://aarnphm.xyz/triathlon'],
    ['https://t.aarnphm.xyz/triathlon/tools', 'https://aarnphm.xyz/triathlon/tools'],
  ]

  for (const [source, expected] of cases) {
    assert.equal(triathlonShortcutRedirectUrl('https://t.aarnphm.xyz', source, true), expected)
  }
})

test('preserves triathlon shortcut search and hash state', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/analytics?window=42d#fitness',
      true,
    ),
    'https://aarnphm.xyz/triathlon/analytics?window=42d#fitness',
  )
})

test('redirects triathlon shortcut assets without inventing nested triathlon paths', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://t.aarnphm.xyz',
      'https://t.aarnphm.xyz/static/analytics.json',
      false,
    ),
    'https://aarnphm.xyz/static/analytics.json',
  )
})

test('honors configured canonical base URLs', () => {
  assert.equal(
    triathlonShortcutRedirectUrl(
      'https://preview.aarnphm.xyz',
      'https://t.aarnphm.xyz/tools',
      true,
    ),
    'https://preview.aarnphm.xyz/triathlon/tools',
  )
})
