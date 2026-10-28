import assert from 'node:assert/strict'
import test from 'node:test'
import { homepageShortcutKeys } from './homepage-shortcuts'

// Allocation can fail on punctuation, accented initials, repeated abbreviations, or a
// multi-word abbreviation already claimed by a repeated single-word initial.
test('allocates in document order, using combinations only after a repeated initial', () => {
  assert.deepEqual(
    homepageShortcutKeys([
      'focus',
      'friends',
      'fast-twitching muscles',
      'fern',
      'making internet zines',
      'more internet zines',
    ]),
    ['F', 'FF', 'FM', 'FFFF', 'M', 'MIZ'],
  )
})

test('keeps repeated abbreviations and repeated-letter combinations unique', () => {
  assert.deepEqual(
    homepageShortcutKeys(['focus', 'friends', 'free food', 'fast muscles', 'fine music']),
    ['F', 'FF', 'FFF', 'FM', 'FFFFF'],
  )
  const keys = homepageShortcutKeys(Array.from({ length: 30 }, () => 'same same'))
  assert.equal(new Set(keys).size, keys.length)
  assert.equal(keys.at(-1), 'S'.repeat(30))
})

test('normalizes labels without treating punctuation or hyphens as word breaks', () => {
  assert.deepEqual(
    homepageShortcutKeys([
      '',
      '  ',
      '🎞 cinema',
      '“colophon”',
      'écriture',
      'email',
      'are.na',
      'aesthetic',
    ]),
    ['', '', 'C', 'CC', 'E', 'EE', 'A', 'AA'],
  )
})
