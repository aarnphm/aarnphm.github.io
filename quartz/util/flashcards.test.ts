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

  test('deletion inside a code span splits the span around the blank', () => {
    const { cards } = parseFlashcards('C: Dates read `le [premier] mars` but `le [deux] mars`.')
    assert.equal(cards.length, 2)
    assert.equal(
      cards[0].front,
      'Dates read `le` <span class="cloze-blank">[…]</span> `mars` but `le deux mars`.',
    )
    assert.equal(
      cards[0].back,
      'Dates read `le` <span class="cloze-answer">`premier`</span> `mars` but `le deux mars`.',
    )
    assert.equal(
      cards[1].front,
      'Dates read `le premier mars` but `le` <span class="cloze-blank">[…]</span> `mars`.',
    )
  })

  test('deletion at a code span edge absorbs that delimiter', () => {
    const whole = parseFlashcards('C: Plural of `le`: `[les]`.').cards[0]
    assert.equal(whole.front, 'Plural of `le`: <span class="cloze-blank">[…]</span>.')
    assert.equal(whole.back, 'Plural of `le`: <span class="cloze-answer">`les`</span>.')

    const edges = parseFlashcards("C: `[D'où] viens-tu [?]`").cards
    assert.equal(edges[0].front, '<span class="cloze-blank">[…]</span> `viens-tu ?`')
    assert.equal(edges[1].front, '`D\'où viens-tu` <span class="cloze-blank">[…]</span>')

    const glued = parseFlashcards("C: `Je viens [d']Italie.`").cards[0]
    assert.equal(glued.front, '`Je viens` <span class="cloze-blank">[…]</span>`Italie.`')
  })

  test('code spans hide dollar signs and honour longer fences', () => {
    const dollars = parseFlashcards('C: `15,50 $` uses a [comma] and `$` follows.').cards
    assert.equal(dollars.length, 1)
    assert.equal(
      dollars[0].back,
      '`15,50 $` uses a <span class="cloze-answer">comma</span> and `$` follows.',
    )

    const fence = parseFlashcards('C: ``a`b [c]`` stays').cards[0]
    assert.equal(fence.front, '``a`b`` <span class="cloze-blank">[…]</span> stays')

    const padded = parseFlashcards('C: ``a ` [b]`` stays').cards[0]
    assert.equal(padded.front, '``a ` `` <span class="cloze-blank">[…]</span> stays')
  })

  test('a deletion wrapping a whole code span keeps the code inside the blank', () => {
    const { cards } = parseFlashcards('C: Only [`-et-un`] takes `et`.')
    assert.equal(cards.length, 1)
    assert.equal(cards[0].back, 'Only <span class="cloze-answer">`-et-un`</span> takes `et`.')
  })
})

describe('parseFlashcards: notes', () => {
  test('N: shows on the back only and leaves the id unchanged', () => {
    const plain = parseFlashcards("C: J'ai [25] ans.\n---\nQ: 20?\nA: `vingt`").cards
    const noted = parseFlashcards(
      "C: J'ai [25] ans.\n\nN: age uses `avoir`.\n---\nQ: 20?\nA: `vingt`\nN: silent `t`\nsecond line",
    ).cards
    assert.deepEqual(
      noted.map(card => card.id),
      plain.map(card => card.id),
    )
    assert.equal(noted[0].front, 'J\'ai <span class="cloze-blank">[…]</span> ans.')
    assert.equal(noted[0].note, 'age uses `avoir`.')
    assert.equal(noted[1].back, '`vingt`')
    assert.equal(noted[1].note, 'silent `t`\nsecond line')
    assert.equal(plain[0].note, undefined)
  })

  test('N: before A: is an error', () => {
    const { cards, errors } = parseFlashcards('Q: a\nN: b\nA: c')
    assert.equal(cards.length, 0)
    assert.equal(errors.length, 1)
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

describe('ID: pins', () => {
  test('a pinned Q/A keeps the old id after rewording', () => {
    const before = parseFlashcards('Q: What is ∂∂?\nA: zero')
    const old = before.cards[0].id
    const deck = parseFlashcards(
      `Q: Why is ∂∂ = 0?\nA: faces cancel in pairs\nN: see notes\nID: ${old}`,
    )
    assert.deepEqual(deck.errors, [])
    assert.equal(deck.cards[0].id, old)
    assert.equal(deck.cards[0].note, 'see notes')
    assert.equal(deck.cards[0].raw.includes('ID:'), false)
  })

  test('a cloze pins one id per deletion, in order', () => {
    const deck = parseFlashcards('C: [a] then [b].\nID: 0123abcd 89efcdef')
    assert.deepEqual(deck.errors, [])
    assert.deepEqual(
      deck.cards.map(c => c.id),
      ['0123abcd', '89efcdef'],
    )
  })

  test('a pin count that does not match the deletions is an error', () => {
    const deck = parseFlashcards('C: [a] then [b].\nID: 0123abcd')
    assert.equal(deck.cards.length, 0)
    assert.match(deck.errors[0].message, /1 ids for 2 deletions/)
  })

  test('non-hex ID: text stays part of the face', () => {
    const deck = parseFlashcards('Q: name it\nA: the map\nID: identity morphism')
    assert.deepEqual(deck.errors, [])
    assert.equal(deck.cards[0].back, 'the map\nID: identity morphism')
  })

  test('ID: before A: and a second ID: are errors', () => {
    assert.match(parseFlashcards('Q: q\nID: 0123abcd\nA: a').errors[0].message, /before A:/)
    assert.match(
      parseFlashcards('Q: q\nA: a\nID: 0123abcd\nID: 0123abce').errors[0].message,
      /twice/,
    )
  })

  test('duplicate ids in one deck are an error', () => {
    const deck = parseFlashcards('Q: a\nA: b\n---\nQ: a\nA: b')
    assert.equal(deck.cards.length, 2)
    assert.match(deck.errors[0].message, /duplicate card id/)
  })
})
