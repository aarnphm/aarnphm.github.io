import {
  type Completion,
  type CompletionContext,
  type CompletionResult,
  type CompletionSource,
} from '@codemirror/autocomplete'
import {
  Annotation,
  EditorState,
  Prec,
  RangeSetBuilder,
  StateEffect,
  StateField,
  type Extension,
  type Transaction,
} from '@codemirror/state'
import { Decoration, EditorView, keymap, type Command } from '@codemirror/view'
import abbreviationsData from './abbreviations.json'

const CURSOR_PLACEHOLDER = '$CURSOR'
const COMPLETION_PREFIX_REGEX = /\\[A-Za-z0-9_^<>|~*+=!?:.,/\-[\]{}()'"`#&]*$/

export type LeanAbbreviationReplacement = { text: string; cursorOffset?: number }

export type LeanAbbreviationMatch = { abbrev: string; symbol: string; cursorOffset?: number }

export type LeanSymbolHoverInfo = {
  symbol: string
  fromCol: number
  toCol: number
  abbreviations: readonly string[]
  autoClosing: ReadonlyArray<readonly [string, string]>
  markdown: string
}

export type LeanAbbreviationState = { from: number; to: number; abbrev: string }

export type LeanAbbreviationTrigger = 'input' | 'tab'

export type LeanAbbreviationEditOptions = {
  trigger?: LeanAbbreviationTrigger
  trackedFrom?: number | null
  allowUntracked?: boolean
  index?: LeanAbbreviationIndex
}

export type LeanAbbreviationEditResult =
  | {
      kind: 'eager' | 'delimiter' | 'tab'
      from: number
      to: number
      abbrev: string
      insert: string
      selectionAnchor: number
      cursorOffset?: number
      active: LeanAbbreviationState | null
    }
  | { kind: 'pending'; from: number; to: number; abbrev: string; active: LeanAbbreviationState }
  | { kind: 'none'; active: null }

function parseSymbolReplacement(rawSymbol: string): LeanAbbreviationReplacement {
  const cursorIndex = rawSymbol.indexOf(CURSOR_PLACEHOLDER)
  if (cursorIndex === -1) {
    return { text: rawSymbol }
  }
  return {
    text:
      rawSymbol.slice(0, cursorIndex) + rawSymbol.slice(cursorIndex + CURSOR_PLACEHOLDER.length),
    cursorOffset: cursorIndex,
  }
}

function compareAbbreviations(a: string, b: string): number {
  if (a.length !== b.length) return a.length - b.length
  return a < b ? -1 : a > b ? 1 : 0
}

function hasNonAsciiCodeUnit(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    if (text.charCodeAt(i) > 0x7f) return true
  }
  return false
}

function formatAbbreviationListMarkdown(
  symbol: string,
  abbreviations: readonly string[],
  autoClosing: ReadonlyArray<readonly [string, string]>,
): string {
  const formatted = abbreviations.map(abbrev => `\`\\${abbrev}\``)
  let mainText: string
  if (formatted.length === 0) {
    mainText = `Symbol \`${symbol}\``
  } else if (formatted.length === 1) {
    mainText = `Type \`${symbol}\` using ${formatted[0]}`
  } else if (formatted.length === 2) {
    mainText = `Type \`${symbol}\` using ${formatted[0]} or ${formatted[1]}`
  } else {
    mainText = `Type \`${symbol}\` using ${formatted.slice(0, -1).join(', ')}, or ${formatted[formatted.length - 1]}`
  }

  if (autoClosing.length === 0) {
    return mainText
  }

  const autoClosingNotes = autoClosing.map(
    ([abbrev, closing]) => `\`${symbol}…${closing}\` with \`\\${abbrev}\``,
  )
  return `${mainText} (auto-close ${autoClosingNotes.join(', ')})`
}

export class LeanAbbreviationIndex {
  readonly symbolsByAbbreviation: Readonly<Record<string, string>>
  readonly strictPrefixCount: Map<string, number>
  readonly validPrefixSet: Set<string>
  readonly reverseSymbols: Map<string, readonly string[]>
  readonly autoClosingByOpening: Map<string, ReadonlyArray<readonly [string, string]>>

  private readonly sortedAbbreviations: readonly string[]
  private readonly abbreviationsByPrefix: Map<string, readonly string[]>
  private readonly maxReverseSymbolLength: number

  constructor(rawAbbreviations: Readonly<Record<string, string>> = abbreviationsData) {
    this.symbolsByAbbreviation = rawAbbreviations
    this.strictPrefixCount = new Map<string, number>()
    this.validPrefixSet = new Set<string>()

    const keys = Object.keys(rawAbbreviations).sort(compareAbbreviations)
    this.sortedAbbreviations = keys

    const prefixBuckets = new Map<string, string[]>()
    const reverseBuckets = new Map<string, string[]>()
    const autoClosingBuckets = new Map<string, Array<readonly [string, string]>>()
    let maxSymbolLen = 0

    for (const abbrev of keys) {
      for (let i = 1; i <= abbrev.length; i += 1) {
        const prefix = abbrev.slice(0, i)
        this.validPrefixSet.add(prefix)
        if (i < abbrev.length) {
          this.strictPrefixCount.set(prefix, (this.strictPrefixCount.get(prefix) ?? 0) + 1)
        }
        let bucket = prefixBuckets.get(prefix)
        if (!bucket) {
          bucket = []
          prefixBuckets.set(prefix, bucket)
        }
        bucket.push(abbrev)
      }

      const rawSymbol = rawAbbreviations[abbrev]
      if (!rawSymbol) continue

      const cursorIdx = rawSymbol.indexOf(CURSOR_PLACEHOLDER)
      if (cursorIdx !== -1) {
        const opening = rawSymbol.slice(0, cursorIdx)
        const closing = rawSymbol.slice(cursorIdx + CURSOR_PLACEHOLDER.length)
        if (opening.length > 0) {
          let pairs = autoClosingBuckets.get(opening)
          if (!pairs) {
            pairs = []
            autoClosingBuckets.set(opening, pairs)
          }
          pairs.push([abbrev, closing] as const)
          if (opening.length > maxSymbolLen) {
            maxSymbolLen = opening.length
          }
        }
        continue
      }

      // Skip pure ASCII control/escape sequences (such as "\\" or "\n") in reverse symbol hover
      if (!hasNonAsciiCodeUnit(rawSymbol)) continue

      let revList = reverseBuckets.get(rawSymbol)
      if (!revList) {
        revList = []
        reverseBuckets.set(rawSymbol, revList)
      }
      revList.push(abbrev)
      if (rawSymbol.length > maxSymbolLen) {
        maxSymbolLen = rawSymbol.length
      }
    }

    this.abbreviationsByPrefix = prefixBuckets
    this.reverseSymbols = reverseBuckets
    this.autoClosingByOpening = autoClosingBuckets
    this.maxReverseSymbolLength = maxSymbolLen
  }

  private normalizeAbbrevKey(abbrev: string): string {
    if (
      abbrev.length > 1 &&
      abbrev.startsWith('\\') &&
      !Object.hasOwn(this.symbolsByAbbreviation, abbrev)
    ) {
      return abbrev.slice(1)
    }
    return abbrev
  }

  getSymbol(abbrev: string): string | undefined {
    const key = this.normalizeAbbrevKey(abbrev)
    if (!Object.hasOwn(this.symbolsByAbbreviation, key)) return undefined
    return parseSymbolReplacement(this.symbolsByAbbreviation[key]).text
  }

  hasPrefix(prefix: string): boolean {
    const key = this.normalizeAbbrevKey(prefix)
    return this.validPrefixSet.has(key)
  }

  hasAbbreviationWithPrefix(prefix: string): boolean {
    return this.hasPrefix(prefix)
  }

  isUniqueAndComplete(abbrev: string): boolean {
    const key = this.normalizeAbbrevKey(abbrev)
    return (
      Object.hasOwn(this.symbolsByAbbreviation, key) &&
      Boolean(this.symbolsByAbbreviation[key]) &&
      (this.strictPrefixCount.get(key) ?? 0) === 0
    )
  }

  isUniqueCompleteAbbreviation(abbrev: string): boolean {
    return this.isUniqueAndComplete(abbrev)
  }

  private findRawReplacementText(abbrev: string): string | undefined {
    if (abbrev.length === 0) return undefined
    if (Object.hasOwn(this.symbolsByAbbreviation, abbrev)) {
      return this.symbolsByAbbreviation[abbrev]
    }
    const matching = this.abbreviationsByPrefix.get(abbrev)
    if (matching && matching.length > 0) {
      return this.symbolsByAbbreviation[matching[0]]
    }
    const prefixReplacement = this.findRawReplacementText(abbrev.slice(0, -1))
    if (prefixReplacement === undefined) return undefined
    return prefixReplacement + abbrev.slice(-1)
  }

  resolveReplacement(abbrev: string): LeanAbbreviationReplacement | undefined {
    const key = this.normalizeAbbrevKey(abbrev)
    if (key.length === 0) return undefined
    const raw = this.findRawReplacementText(key)
    if (raw === undefined) return undefined
    return parseSymbolReplacement(raw)
  }

  matchPrefix(prefix: string, limit = 60): ReadonlyArray<LeanAbbreviationMatch> {
    if (limit <= 0) return []
    const normalized =
      prefix.length > 1 && prefix.startsWith('\\') && !this.validPrefixSet.has(prefix)
        ? prefix.slice(1)
        : prefix
    const sourceKeys =
      normalized.length === 0
        ? this.sortedAbbreviations
        : (this.abbreviationsByPrefix.get(normalized) ?? [])
    const count = Math.min(sourceKeys.length, limit)
    const results: LeanAbbreviationMatch[] = []
    for (let i = 0; i < count; i += 1) {
      const abbrev = sourceKeys[i]
      const parsed = parseSymbolReplacement(this.symbolsByAbbreviation[abbrev])
      if (parsed.cursorOffset !== undefined) {
        results.push({ abbrev, symbol: parsed.text, cursorOffset: parsed.cursorOffset })
      } else {
        results.push({ abbrev, symbol: parsed.text })
      }
    }
    return results
  }

  hoverInfoForTextAt(lineText: string, column: number): LeanSymbolHoverInfo | undefined {
    if (column < 0 || column >= lineText.length || this.maxReverseSymbolLength === 0) {
      return undefined
    }

    const minStart = Math.max(0, column - this.maxReverseSymbolLength + 1)
    let bestSymbol = ''
    let bestFrom = -1
    let bestTo = -1

    for (let start = column; start >= minStart; start -= 1) {
      const maxLen = Math.min(this.maxReverseSymbolLength, lineText.length - start)
      for (let len = maxLen; len >= 1; len -= 1) {
        const end = start + len
        if (end <= column) break
        const candidate = lineText.slice(start, end)
        if (this.reverseSymbols.has(candidate) || this.autoClosingByOpening.has(candidate)) {
          if (
            candidate.length > bestSymbol.length ||
            (candidate.length === bestSymbol.length && start === column && bestFrom !== column)
          ) {
            bestSymbol = candidate
            bestFrom = start
            bestTo = end
          }
          break
        }
      }
    }

    if (!bestSymbol || bestFrom === -1) return undefined

    const abbreviations = this.reverseSymbols.get(bestSymbol) ?? []
    const autoClosing = this.autoClosingByOpening.get(bestSymbol) ?? []
    if (abbreviations.length === 0 && autoClosing.length === 0) return undefined

    return {
      symbol: bestSymbol,
      fromCol: bestFrom,
      toCol: bestTo,
      abbreviations,
      autoClosing,
      markdown: formatAbbreviationListMarkdown(bestSymbol, abbreviations, autoClosing),
    }
  }
}

let singletonIndex: LeanAbbreviationIndex | null = null

export function getLeanAbbreviationIndex(): LeanAbbreviationIndex {
  if (!singletonIndex) {
    singletonIndex = new LeanAbbreviationIndex()
  }
  return singletonIndex
}

function findUntrackedSlashOffset(docText: string, cursorPos: number): number | null {
  let lineStart = cursorPos
  while (lineStart > 0 && docText[lineStart - 1] !== '\n') {
    lineStart -= 1
  }

  for (let i = cursorPos - 1; i >= lineStart; i -= 1) {
    if (docText[i] === '\\') {
      if (i > lineStart && docText[i - 1] === '\\') {
        // Allow `\\` at the cursor to resolve the literal backslash abbreviation `"\\"`
        if (i === cursorPos - 1 && (i - 1 === lineStart || docText[i - 2] !== '\\')) {
          return i - 1
        }
        return null
      }
      return i
    }
  }
  return null
}

export function computeLeanAbbreviationEdit(
  docText: string,
  cursorPos = docText.length,
  options: LeanAbbreviationEditOptions = {},
): LeanAbbreviationEditResult {
  const index = options.index ?? getLeanAbbreviationIndex()
  const trigger = options.trigger ?? 'input'
  const clampedCursor = Math.max(0, Math.min(cursorPos, docText.length))

  let slashPos: number | null = null
  if (
    options.trackedFrom !== undefined &&
    options.trackedFrom !== null &&
    options.trackedFrom >= 0 &&
    options.trackedFrom < clampedCursor &&
    docText[options.trackedFrom] === '\\'
  ) {
    slashPos = options.trackedFrom
  } else if (options.allowUntracked !== false) {
    slashPos = findUntrackedSlashOffset(docText, clampedCursor)
  }

  if (slashPos === null) {
    return { kind: 'none', active: null }
  }

  const rawAfterSlash = docText.slice(slashPos + 1, clampedCursor)

  if (trigger === 'tab') {
    if (rawAfterSlash.length === 0) {
      return { kind: 'none', active: null }
    }
    const resolved = index.resolveReplacement(rawAfterSlash)
    if (!resolved) {
      return { kind: 'none', active: null }
    }
    const cursorOffset = resolved.cursorOffset ?? resolved.text.length
    return {
      kind: 'tab',
      from: slashPos,
      to: clampedCursor,
      abbrev: rawAfterSlash,
      insert: resolved.text,
      selectionAnchor: slashPos + cursorOffset,
      cursorOffset: resolved.cursorOffset,
      active: null,
    }
  }

  if (rawAfterSlash.length === 0) {
    const active: LeanAbbreviationState = { from: slashPos, to: clampedCursor, abbrev: '' }
    return { kind: 'pending', from: slashPos, to: clampedCursor, abbrev: '', active }
  }

  if (index.isUniqueAndComplete(rawAfterSlash)) {
    const resolved = index.resolveReplacement(rawAfterSlash)
    if (resolved) {
      const cursorOffset = resolved.cursorOffset ?? resolved.text.length
      return {
        kind: 'eager',
        from: slashPos,
        to: clampedCursor,
        abbrev: rawAfterSlash,
        insert: resolved.text,
        selectionAnchor: slashPos + cursorOffset,
        cursorOffset: resolved.cursorOffset,
        active: null,
      }
    }
  }

  if (index.hasPrefix(rawAfterSlash)) {
    const active: LeanAbbreviationState = {
      from: slashPos,
      to: clampedCursor,
      abbrev: rawAfterSlash,
    }
    return { kind: 'pending', from: slashPos, to: clampedCursor, abbrev: rawAfterSlash, active }
  }

  // Non-prefix continuation (Case C): find longest valid abbreviation prefix before trailing delimiter/chars
  let validPrefixLen = 0
  for (let len = rawAfterSlash.length - 1; len >= 1; len -= 1) {
    const candidate = rawAfterSlash.slice(0, len)
    if (index.hasPrefix(candidate)) {
      validPrefixLen = len
      break
    }
  }

  if (validPrefixLen === 0) {
    return { kind: 'none', active: null }
  }

  const abbrev = rawAfterSlash.slice(0, validPrefixLen)
  const trailing = rawAfterSlash.slice(validPrefixLen)
  const resolved = index.resolveReplacement(abbrev)
  if (!resolved) {
    return { kind: 'none', active: null }
  }

  const abbrevEnd = slashPos + 1 + validPrefixLen
  const selectionAnchor =
    resolved.cursorOffset !== undefined
      ? slashPos + resolved.cursorOffset + trailing.length
      : slashPos + resolved.text.length + trailing.length
  const nextActive: LeanAbbreviationState | null =
    trailing === '\\'
      ? {
          from: slashPos + resolved.text.length,
          to: slashPos + resolved.text.length + 1,
          abbrev: '',
        }
      : null

  return {
    kind: 'delimiter',
    from: slashPos,
    to: abbrevEnd,
    abbrev,
    insert: resolved.text,
    selectionAnchor,
    cursorOffset: resolved.cursorOffset,
    active: nextActive,
  }
}

const leanAbbreviationRewriteAnnotation = Annotation.define<boolean>()
const setLeanAbbreviationEffect = StateEffect.define<LeanAbbreviationState | null>()
const pendingAbbreviationMark = Decoration.mark({ class: 'cm-leanAbbreviation' })

function extractSingleChangeEnd(tr: Transaction): {
  fromA: number
  toA: number
  fromB: number
  toB: number
  insertedText: string
  changeCount: number
} {
  let fromA = 0
  let toA = 0
  let fromB = 0
  let toB = 0
  let insertedText = ''
  let changeCount = 0
  tr.changes.iterChanges((fA, tA, fB, tB, inserted) => {
    changeCount += 1
    fromA = fA
    toA = tA
    fromB = fB
    toB = tB
    insertedText = inserted.toString()
  })
  return { fromA, toA, fromB, toB, insertedText, changeCount }
}

function initialAbbreviationState(state: EditorState): LeanAbbreviationState | null {
  const index = getLeanAbbreviationIndex()
  const head = state.selection.main.head
  if (!state.selection.main.empty) return null
  const line = state.doc.lineAt(head)
  const linePrefix = line.text.slice(0, head - line.from)
  const slashInLine = findUntrackedSlashOffset(linePrefix, linePrefix.length)
  if (slashInLine === null) return null
  const abbrev = linePrefix.slice(slashInLine + 1)
  if (abbrev.length === 0 || (index.hasPrefix(abbrev) && !index.isUniqueAndComplete(abbrev))) {
    return { from: line.from + slashInLine, to: head, abbrev }
  }
  return null
}

const leanAbbreviationField = StateField.define<LeanAbbreviationState | null>({
  create(state) {
    return initialAbbreviationState(state)
  },
  update(prev, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setLeanAbbreviationEffect)) {
        return effect.value
      }
    }
    if (tr.annotation(leanAbbreviationRewriteAnnotation)) {
      return null
    }
    if (!tr.docChanged) {
      if (tr.selection) {
        const main = tr.newSelection.main
        if (!main.empty || !prev || main.head !== prev.to) {
          return null
        }
      }
      return prev
    }
    return prev
  },
  provide: field =>
    EditorView.decorations.from(field, active => {
      if (!active || active.to <= active.from) return Decoration.none
      const builder = new RangeSetBuilder<Decoration>()
      builder.add(active.from, active.to, pendingAbbreviationMark)
      return builder.finish()
    }),
})

export function getActiveLeanAbbreviation(state: EditorState): LeanAbbreviationState | null {
  return state.field(leanAbbreviationField, false) ?? null
}

export const acceptLeanAbbreviation: Command = (view: EditorView): boolean => {
  const index = getLeanAbbreviationIndex()
  const state = view.state
  const active = getActiveLeanAbbreviation(state)

  if (active && active.abbrev.length > 0) {
    const resolved = index.resolveReplacement(active.abbrev)
    if (resolved) {
      const cursorOffset = resolved.cursorOffset ?? resolved.text.length
      view.dispatch({
        changes: { from: active.from, to: active.to, insert: resolved.text },
        selection: { anchor: active.from + cursorOffset },
        effects: setLeanAbbreviationEffect.of(null),
        annotations: leanAbbreviationRewriteAnnotation.of(true),
      })
      return true
    }
  }

  if (!state.selection.main.empty) return false
  const head = state.selection.main.head
  const line = state.doc.lineAt(head)
  const edit = computeLeanAbbreviationEdit(line.text, head - line.from, { trigger: 'tab', index })
  if (edit.kind === 'tab') {
    const from = line.from + edit.from
    const to = line.from + edit.to
    const selectionAnchor = line.from + edit.selectionAnchor
    view.dispatch({
      changes: { from, to, insert: edit.insert },
      selection: { anchor: selectionAnchor },
      effects: setLeanAbbreviationEffect.of(null),
      annotations: leanAbbreviationRewriteAnnotation.of(true),
    })
    return true
  }

  return false
}

const leanAbbreviationTransactionFilter = EditorState.transactionFilter.of(tr => {
  if (tr.annotation(leanAbbreviationRewriteAnnotation)) {
    return tr
  }
  if (!tr.docChanged) {
    return tr
  }

  const index = getLeanAbbreviationIndex()
  const prevActive = tr.startState.field(leanAbbreviationField, false) ?? null
  const { fromA, toA, fromB, toB, insertedText, changeCount } = extractSingleChangeEnd(tr)

  if (changeCount !== 1) {
    return [tr, { effects: setLeanAbbreviationEffect.of(null), sequential: true }]
  }

  // Determine the effective cursor position after the edit
  const effectiveHead = tr.selection ? tr.newSelection.main.head : toB

  let trackedFrom: number | null = null
  let allowUntracked = false

  if (prevActive !== null) {
    if (
      fromA >= prevActive.from + 1 &&
      toA <= prevActive.to &&
      effectiveHead >= prevActive.from + 1
    ) {
      trackedFrom = prevActive.from
    } else if (fromA <= prevActive.from && toA >= prevActive.to && insertedText.includes('\\')) {
      allowUntracked = true
    } else {
      return [tr, { effects: setLeanAbbreviationEffect.of(null), sequential: true }]
    }
  } else if (insertedText.includes('\\')) {
    allowUntracked = true
  } else {
    return tr
  }

  const windowStart =
    trackedFrom !== null
      ? Math.max(0, trackedFrom - 1)
      : tr.newDoc.lineAt(Math.min(fromB, effectiveHead)).from
  const sliceText = tr.newDoc.sliceString(windowStart, effectiveHead)
  const relativeTrackedFrom = trackedFrom !== null ? trackedFrom - windowStart : null

  const edit = computeLeanAbbreviationEdit(sliceText, sliceText.length, {
    trigger: 'input',
    trackedFrom: relativeTrackedFrom,
    allowUntracked,
    index,
  })

  if (edit.kind === 'eager' || edit.kind === 'delimiter') {
    const absFrom = windowStart + edit.from
    const absTo = windowStart + edit.to
    const absSelection = windowStart + edit.selectionAnchor
    const nextActive =
      edit.active !== null
        ? {
            from: windowStart + edit.active.from,
            to: windowStart + edit.active.to,
            abbrev: edit.active.abbrev,
          }
        : null
    return [
      tr,
      {
        changes: { from: absFrom, to: absTo, insert: edit.insert },
        selection: { anchor: absSelection },
        effects: setLeanAbbreviationEffect.of(nextActive),
        annotations: leanAbbreviationRewriteAnnotation.of(true),
        sequential: true,
      },
    ]
  }

  if (edit.kind === 'pending') {
    const nextActive: LeanAbbreviationState = {
      from: windowStart + edit.active.from,
      to: windowStart + edit.active.to,
      abbrev: edit.active.abbrev,
    }
    return [tr, { effects: setLeanAbbreviationEffect.of(nextActive), sequential: true }]
  }

  if (prevActive !== null) {
    return [tr, { effects: setLeanAbbreviationEffect.of(null), sequential: true }]
  }

  return tr
})

const leanAbbreviationTheme = EditorView.baseTheme({
  '.cm-leanAbbreviation': {
    textDecorationLine: 'underline',
    textDecorationStyle: 'dotted',
    textUnderlineOffset: '2px',
  },
})

export function leanUnicodeInputExtension(): Extension {
  return [
    leanAbbreviationField,
    leanAbbreviationTransactionFilter,
    Prec.highest(keymap.of([{ key: 'Tab', run: acceptLeanAbbreviation }])),
    leanAbbreviationTheme,
  ]
}

export const leanUnicodeCompletionSource: CompletionSource = (
  context: CompletionContext,
): CompletionResult | null => {
  const line = context.state.doc.lineAt(context.pos)
  const textBefore = line.text.slice(0, context.pos - line.from)
  const match = COMPLETION_PREFIX_REGEX.exec(textBefore)
  if (!match) return null

  const matchIndex = match.index
  if (matchIndex > 0 && textBefore[matchIndex - 1] === '\\') {
    return null
  }

  const from = line.from + matchIndex
  const prefixWithoutSlash = match[0].slice(1)
  const index = getLeanAbbreviationIndex()
  const matches = index.matchPrefix(prefixWithoutSlash, 48)
  if (matches.length === 0) return null

  const options: Completion[] = matches.map(item => ({
    label: `\\${item.abbrev}`,
    detail: item.symbol,
    type: 'constant',
    apply(view: EditorView, _completion: Completion, applyFrom: number, applyTo: number) {
      const cursorOffset = item.cursorOffset ?? item.symbol.length
      view.dispatch({
        changes: { from: applyFrom, to: applyTo, insert: item.symbol },
        selection: { anchor: applyFrom + cursorOffset },
        effects: setLeanAbbreviationEffect.of(null),
        annotations: leanAbbreviationRewriteAnnotation.of(true),
      })
    },
  }))

  return { from, to: context.pos, filter: false, options }
}
