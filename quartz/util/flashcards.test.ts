import assert from 'node:assert'
import { describe, test } from 'node:test'
import { parseFlashcards } from './flashcards'
import {
  deckPathsForSource,
  flashcardsSlug,
  isFlashcardPath,
  sourceSlugForDeck,
} from './flashcards-path'

describe('path + slug helpers', () => {
  test('isFlashcardPath matches both suffixes', () => {
    assert.equal(isFlashcardPath('thoughts/Sets.flashcards.md'), true)
    assert.equal(isFlashcardPath('thoughts/Sets.fc.md'), true)
    assert.equal(isFlashcardPath('thoughts/Sets.fc'), true)
    assert.equal(isFlashcardPath('thoughts/Sets.flashcards'), true)
    assert.equal(isFlashcardPath('thoughts/Sets.md'), false)
  })

  test('sourceSlugForDeck strips the deck suffix', () => {
    assert.equal(sourceSlugForDeck('thoughts/Sets.flashcards'), 'thoughts/Sets')
    assert.equal(sourceSlugForDeck('thoughts/Sets.fc'), 'thoughts/Sets')
    assert.equal(sourceSlugForDeck('/thoughts/Sets.fc.md'), '/thoughts/Sets')
    assert.equal(sourceSlugForDeck('/thoughts/Sets.flashcards.md'), '/thoughts/Sets')
  })

  test('deckPathsForSource and flashcardsSlug', () => {
    assert.deepEqual(deckPathsForSource('thoughts/Sets.md'), [
      'thoughts/Sets.fc',
      'thoughts/Sets.flashcards',
      'thoughts/Sets.fc.md',
      'thoughts/Sets.flashcards.md',
    ])
    assert.equal(flashcardsSlug('thoughts/Sets'), 'thoughts/Sets/flashcards')
  })
})

describe('parseFlashcards: qa', () => {
  test('parses a basic Q/A card', () => {
    const { cards, errors } = parseFlashcards('Q: how many neurons?\nA: ~80 billion.')
    assert.equal(errors.length, 0)
    assert.equal(cards.length, 1)
    assert.equal(cards[0].kind, 'qa')
    assert.equal(cards[0].front, 'how many neurons?')
    assert.equal(cards[0].back, '~80 billion.')
    assert.match(cards[0].id, /^[0-9a-f]{8}$/)
  })

  test('strips leading frontmatter before parsing', () => {
    const src = '---\ntitle: Sets.flashcards\ntags: [seed]\n---\nQ: a\nA: b'
    const { cards } = parseFlashcards(src)
    assert.equal(cards.length, 1)
    assert.equal(cards[0].front, 'a')
  })

  test('multi-line faces and --- separators', () => {
    const src = 'Q: list the platinum group\nA:\n\n- ruthenium\n- rhodium\n---\nQ: x\nA: y'
    const { cards } = parseFlashcards(src)
    assert.equal(cards.length, 2)
    assert.equal(cards[0].back, '- ruthenium\n- rhodium')
    assert.equal(cards[1].front, 'x')
  })

  test('consecutive Q: cards split without a separator', () => {
    const { cards } = parseFlashcards('Q: a\nA: b\nQ: c\nA: d')
    assert.equal(cards.length, 2)
  })

  test('Q: without A: is an error, not a card', () => {
    const { cards, errors } = parseFlashcards('Q: dangling')
    assert.equal(cards.length, 0)
    assert.equal(errors.length, 1)
  })
})

describe('parseFlashcards: cloze', () => {
  test('single deletion yields one card with a blank front', () => {
    const { cards } = parseFlashcards('C: an [agonist] binds a receptor.')
    assert.equal(cards.length, 1)
    assert.equal(cards[0].kind, 'cloze')
    assert.match(cards[0].front, /cloze-blank/)
    assert.match(cards[0].back, /cloze-answer">agonist</)
  })

  test('multiple deletions become siblings sharing a groupId', () => {
    const { cards } = parseFlashcards('C: an [agonist] binds and [activates it].')
    assert.equal(cards.length, 2)
    assert.equal(cards[0].groupId, cards[1].groupId)
    assert.notEqual(cards[0].id, cards[1].id)
    assert.match(cards[0].front, /cloze-blank">\[…\]/)
    assert.match(cards[0].front, /activates it/)
  })

  test('hint after a pipe is shown on the front', () => {
    const { cards } = parseFlashcards('C: the capital is [Ottawa|city].')
    assert.match(cards[0].front, /cloze-blank">city</)
    assert.match(cards[0].back, /cloze-answer">Ottawa</)
  })

  test('C: without deletions is an error', () => {
    const { cards, errors } = parseFlashcards('C: no brackets here.')
    assert.equal(cards.length, 0)
    assert.equal(errors.length, 1)
  })

  test('wikilink brackets are not mistaken for deletions', () => {
    const { cards } = parseFlashcards(
      'C: ZFC blocks [[thoughts/x|Russell]] via the [axiom of separation] over an [existing set].',
    )
    assert.equal(cards.length, 2)
    assert.ok(cards.every(card => card.front.includes('[[thoughts/x|Russell]]')))
    assert.ok(cards.every(card => card.back.includes('[[thoughts/x|Russell]]')))
  })

  test('latex deletion inside inline math keeps markdown math parseable', () => {
    const { cards, errors } = parseFlashcards(
      String.raw`C: De Morgan's law: $A \setminus (B \cup C) = (A \setminus B) [\cap] (A \setminus C)$.`,
    )

    assert.equal(errors.length, 0)
    assert.equal(cards.length, 1)
    assert.equal(
      cards[0].front,
      String.raw`De Morgan's law: $A \setminus (B \cup C) = (A \setminus B) $<span class="cloze-blank">[…]</span>$ (A \setminus C)$.`,
    )
    assert.equal(
      cards[0].back,
      String.raw`De Morgan's law: $A \setminus (B \cup C) = (A \setminus B) $<span class="cloze-answer">$\cap$</span>$ (A \setminus C)$.`,
    )
  })

  test('deletion spanning a whole math region absorbs the delimiters', () => {
    const { cards, errors } = parseFlashcards(
      String.raw`C: The naive set-builder form $[\{x \mid P(x)\}|unsafe form]$ is dangerous.`,
    )

    assert.equal(errors.length, 0)
    assert.equal(cards.length, 1)
    assert.equal(
      cards[0].front,
      String.raw`The naive set-builder form <span class="cloze-blank">unsafe form</span> is dangerous.`,
    )
    assert.equal(
      cards[0].back,
      String.raw`The naive set-builder form <span class="cloze-answer">$\{x \mid P(x)\}$</span> is dangerous.`,
    )
  })
})

describe('content-addressed identity', () => {
  test('editing a face changes the id (reset-on-edit)', () => {
    const a = parseFlashcards('Q: a\nA: b').cards[0].id
    const b = parseFlashcards('Q: a\nA: c').cards[0].id
    assert.notEqual(a, b)
  })

  test('whitespace-only reflow keeps the id stable', () => {
    const a = parseFlashcards('Q: hello world\nA: yes').cards[0].id
    const b = parseFlashcards('Q:   hello   world\nA:  yes  ').cards[0].id
    assert.equal(a, b)
  })
})
