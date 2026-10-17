import type { EditorView } from '@codemirror/view'
import { CompletionContext } from '@codemirror/autocomplete'
import { EditorState } from '@codemirror/state'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  LeanAbbreviationIndex,
  acceptLeanAbbreviation,
  computeLeanAbbreviationEdit,
  getActiveLeanAbbreviation,
  getLeanAbbreviationIndex,
  leanUnicodeCompletionSource,
  leanUnicodeInputExtension,
} from './unicode-input'

function typeChars(initialState: EditorState, text: string): EditorState {
  let state = initialState
  for (const ch of text) {
    const pos = state.selection.main.head
    state = state.update({
      changes: { from: pos, to: pos, insert: ch },
      selection: { anchor: pos + ch.length },
    }).state
  }
  return state
}

function createFakeView(initialState: EditorState): {
  view: EditorView
  getState: () => EditorState
} {
  let currentState = initialState
  const fakeView = {
    get state() {
      return currentState
    },
    dispatch(...specs: Parameters<EditorState['update']>) {
      currentState = currentState.update(...specs).state
    },
  } as unknown as EditorView
  return { view: fakeView, getState: () => currentState }
}

describe('LeanAbbreviationIndex', () => {
  const index = getLeanAbbreviationIndex()

  test('resolves exact and prefix symbol lookups', () => {
    const expectedExact: ReadonlyArray<readonly [string, string]> = [
      ['alpha', 'α'],
      ['to', '→'],
      ['r', '→'],
      ['iff', '↔'],
      ['cap', '∩'],
      ['cup', '∪'],
      ['empty', '∅'],
      ['emptyset', '∅'],
      ['compl', 'ᶜ'],
      ['bigcup', '⋃'],
      ['bigcap', '⋂'],
      ['inv', '⁻¹'],
      ['land', '∧'],
      ['lor', '∨'],
      ['<', '⟨'],
      ['>', '⟩'],
    ]

    for (const [abbrev, symbol] of expectedExact) {
      assert.strictEqual(index.getSymbol(abbrev), symbol, `Expected \\${abbrev} -> ${symbol}`)
      assert.deepStrictEqual(index.resolveReplacement(abbrev), { text: symbol })
    }

    assert.strictEqual(index.getSymbol('<>'), '⟨⟩')
    assert.deepStrictEqual(index.resolveReplacement('<>'), { text: '⟨⟩', cursorOffset: 1 })

    // Prefix resolution picks the shortest matching abbreviation
    assert.deepStrictEqual(index.resolveReplacement('alp'), { text: 'α' })
    assert.deepStrictEqual(index.resolveReplacement(' bigcu'.trim()), { text: '⋃' })
  })

  test('detects unique and complete abbreviations vs extendable prefixes', () => {
    assert.strictEqual(index.isUniqueAndComplete('alpha'), true)
    assert.strictEqual(index.isUniqueAndComplete('emptyset'), true)
    assert.strictEqual(index.isUniqueAndComplete('forall'), true)
    assert.strictEqual(index.isUniqueAndComplete('exists'), true)
    assert.strictEqual(index.isUniqueAndComplete('notin'), true)
    assert.strictEqual(index.isUniqueAndComplete('symmDiff'), true)
    assert.strictEqual(index.isUniqueAndComplete('<>'), true)

    // Has longer extensions (\top, \inter, \emptyset, \rho, \subset, \<>), so not unique-complete
    assert.strictEqual(index.isUniqueAndComplete('to'), false)
    assert.strictEqual(index.hasPrefix('to'), true)

    assert.strictEqual(index.isUniqueAndComplete('in'), false)
    assert.strictEqual(index.hasPrefix('in'), true)

    assert.strictEqual(index.isUniqueAndComplete('empty'), false)
    assert.strictEqual(index.hasPrefix('empty'), true)

    assert.strictEqual(index.isUniqueAndComplete('r'), false)
    assert.strictEqual(index.hasPrefix('r'), true)

    assert.strictEqual(index.isUniqueAndComplete('sub'), false)
    assert.strictEqual(index.hasPrefix('sub'), true)

    assert.strictEqual(index.isUniqueAndComplete('<'), false)
    assert.strictEqual(index.hasPrefix('<'), true)
  })

  test('supports recursive longest-prefix fallback replacement', () => {
    assert.deepStrictEqual(index.resolveReplacement('alpha1'), { text: 'α1' })
    assert.deepStrictEqual(index.resolveReplacement('to)'), { text: '→)' })
    assert.strictEqual(index.resolveReplacement(''), undefined)
    assert.strictEqual(index.resolveReplacement('@'), undefined)
  })

  test('ranks matchPrefix by exact match first, then shortest length, then lexicographically', () => {
    const toMatches = index.matchPrefix('to', 10)
    assert.ok(toMatches.length > 1)
    assert.strictEqual(toMatches[0].abbrev, 'to')
    assert.strictEqual(toMatches[0].symbol, '→')

    for (let i = 1; i < toMatches.length; i += 1) {
      const prev = toMatches[i - 1]
      const curr = toMatches[i]
      assert.ok(curr.abbrev.startsWith('to'))
      assert.ok(
        prev.abbrev.length < curr.abbrev.length ||
          (prev.abbrev.length === curr.abbrev.length && prev.abbrev <= curr.abbrev),
      )
    }

    const angleMatches = index.matchPrefix('<>', 5)
    assert.deepStrictEqual(angleMatches[0], { abbrev: '<>', symbol: '⟨⟩', cursorOffset: 1 })
  })

  test('provides reverse symbol hover info for single-char, multi-char, and auto-closing symbols', () => {
    const interHover = index.hoverInfoForTextAt('A ∩ Bᶜ', 2)
    assert.ok(interHover)
    assert.strictEqual(interHover.symbol, '∩')
    assert.strictEqual(interHover.fromCol, 2)
    assert.strictEqual(interHover.toCol, 3)
    assert.ok(interHover.abbreviations.includes('cap'))
    assert.ok(interHover.abbreviations.includes('inter'))
    assert.ok(interHover.markdown.includes('`\\cap`'))
    assert.ok(interHover.markdown.includes('`\\inter`'))

    const complHover = index.hoverInfoForTextAt('A ∩ Bᶜ', 5)
    assert.ok(complHover)
    assert.strictEqual(complHover.symbol, 'ᶜ')
    assert.ok(complHover.abbreviations.includes('compl'))

    const invHoverStart = index.hoverInfoForTextAt('f⁻¹ x', 1)
    const invHoverInner = index.hoverInfoForTextAt('f⁻¹ x', 2)
    assert.ok(invHoverStart)
    assert.strictEqual(invHoverStart.symbol, '⁻¹')
    assert.strictEqual(invHoverStart.fromCol, 1)
    assert.strictEqual(invHoverStart.toCol, 3)
    assert.ok(invHoverStart.abbreviations.includes('inv'))
    assert.deepStrictEqual(invHoverInner, invHoverStart)

    const angleHover = index.hoverInfoForTextAt('⟨x, y⟩', 0)
    assert.ok(angleHover)
    assert.strictEqual(angleHover.symbol, '⟨')
    assert.ok(angleHover.abbreviations.includes('<'))
    assert.ok(angleHover.abbreviations.includes('langle'))
    assert.ok(angleHover.autoClosing.some(([abbrev, close]) => abbrev === '<>' && close === '⟩'))
    assert.ok(angleHover.markdown.includes('`\\<>`'))

    // Plain ASCII characters should not return reverse symbol hover
    assert.strictEqual(index.hoverInfoForTextAt('A ∩ Bᶜ', 0), undefined)
  })
})

describe('leanUnicodeInputExtension and computeLeanAbbreviationEdit', () => {
  test('eagerly replaces unique complete abbreviations on final character', () => {
    const initial = EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] })

    const afterAlpha = typeChars(initial, '\\alpha')
    assert.strictEqual(afterAlpha.doc.toString(), 'α')
    assert.strictEqual(afterAlpha.selection.main.head, 1)
    assert.strictEqual(getActiveLeanAbbreviation(afterAlpha), null)

    const afterForall = typeChars(initial, 'theorem t : \\forall x, x = x')
    assert.strictEqual(afterForall.doc.toString(), 'theorem t : ∀ x, x = x')
  })

  test('tracks pending prefix abbreviations and triggers on space or delimiter', () => {
    let state = EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] })

    state = typeChars(state, '\\to')
    assert.strictEqual(state.doc.toString(), '\\to')
    assert.deepStrictEqual(getActiveLeanAbbreviation(state), { from: 0, to: 3, abbrev: 'to' })

    // Typing space replaces \to with → and keeps the trailing space
    state = typeChars(state, ' ')
    assert.strictEqual(state.doc.toString(), '→ ')
    assert.strictEqual(state.selection.main.head, 2)
    assert.strictEqual(getActiveLeanAbbreviation(state), null)

    // Delimiter characters like '(' or ',' or '1' replace the abbreviation before the delimiter
    const withParen = typeChars(
      EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] }),
      '\\in(A)',
    )
    assert.strictEqual(withParen.doc.toString(), '∈(A)')

    const withDigit = typeChars(
      EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] }),
      '\\alpha1',
    )
    assert.strictEqual(withDigit.doc.toString(), 'α1')

    // Consecutive abbreviations separated by backslash (\to\alpha -> →α)
    const chained = typeChars(
      EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] }),
      '\\to\\alpha',
    )
    assert.strictEqual(chained.doc.toString(), '→α')
  })

  test('positions cursor at $CURSOR offset for paired abbreviations like \\<>', () => {
    let state = EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] })

    state = typeChars(state, '\\<>')
    assert.strictEqual(state.doc.toString(), '⟨⟩')
    assert.strictEqual(state.selection.main.head, 1)

    // Typing further inserts inside ⟨...⟩
    state = typeChars(state, 'a, b')
    assert.strictEqual(state.doc.toString(), '⟨a, b⟩')
    assert.strictEqual(state.selection.main.head, 5)
  })

  test('confirms pending abbreviation with Tab without inserting whitespace', () => {
    const initial = EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] })
    const pendingTo = typeChars(initial, '\\to')
    const { view, getState } = createFakeView(pendingTo)

    const handled = acceptLeanAbbreviation(view)
    assert.strictEqual(handled, true)
    assert.strictEqual(getState().doc.toString(), '→')
    assert.strictEqual(getState().selection.main.head, 1)
    assert.strictEqual(getActiveLeanAbbreviation(getState()), null)
  })

  test('escapes double backslash \\\\ to literal backslash without triggering subsequent abbreviation', () => {
    const state = typeChars(
      EditorState.create({ doc: '', extensions: [leanUnicodeInputExtension()] }),
      '\\\\to ',
    )
    assert.strictEqual(state.doc.toString(), '\\to ')

    const untrackedEdit = computeLeanAbbreviationEdit('\\\\alpha', 7)
    assert.strictEqual(untrackedEdit.kind, 'none')
  })
})

describe('leanUnicodeCompletionSource', () => {
  test('returns ranked symbol completions for backslash prefixes and applies cursor offset', () => {
    const state = EditorState.create({ doc: 'have h : A \\ca', selection: { anchor: 14 } })
    const context = new CompletionContext(state, 14, false)
    const result = leanUnicodeCompletionSource(context)
    assert.ok(result && !(result instanceof Promise))
    assert.strictEqual(result.from, 11)
    assert.strictEqual(result.to, 14)
    assert.strictEqual(result.filter, false)
    assert.ok(result.options.some(opt => opt.label === '\\cap' && opt.detail === '∩'))

    // Applying a completion with $CURSOR (e.g. \<> -> ⟨|⟩) positions cursor between brackets
    const angleState = EditorState.create({
      doc: 'exact \\<',
      selection: { anchor: 8 },
      extensions: [leanUnicodeInputExtension()],
    })
    const angleCtx = new CompletionContext(angleState, 8, false)
    const angleResult = leanUnicodeCompletionSource(angleCtx)
    assert.ok(angleResult && !(angleResult instanceof Promise))
    const pairOption = angleResult.options.find(opt => opt.label === '\\<>')
    assert.ok(pairOption && typeof pairOption.apply === 'function')

    const { view, getState } = createFakeView(angleState)
    pairOption.apply(view, pairOption, angleResult.from, angleResult.to ?? 8)
    assert.strictEqual(getState().doc.toString(), 'exact ⟨⟩')
    assert.strictEqual(getState().selection.main.head, 7)

    // Escaped backslash \\ca should not trigger completion
    const escapedState = EditorState.create({ doc: '\\\\ca', selection: { anchor: 4 } })
    const escapedResult = leanUnicodeCompletionSource(new CompletionContext(escapedState, 4, false))
    assert.strictEqual(escapedResult, null)

    // Custom small dictionary test
    const customIndex = new LeanAbbreviationIndex({ alpha: 'α', '()': '($CURSOR)' })
    assert.strictEqual(customIndex.getSymbol('alpha'), 'α')
  })
})
