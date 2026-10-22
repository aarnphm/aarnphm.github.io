import type { Text } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView } from '@codemirror/view'

export type NotebookTextRange = { from: number; to: number }
export type NotebookSurroundingPairRange = {
  openFrom: number
  openTo: number
  closeFrom: number
  closeTo: number
}

export type NotebookSurroundKeyPlan =
  | { kind: 'pass' }
  | { kind: 'flush' }
  | { kind: 'pending'; buffer: string }
  | { kind: 'surroundSelection'; token: string }
  | { kind: 'surroundWord'; token: string }
  | { kind: 'surroundLine'; token: string }
  | { kind: 'deleteSurround'; token: string }
  | { kind: 'changeSurround'; oldToken: string; replacementToken: string }

const notebookSurroundPairs = new Map<string, readonly [string, string]>([
  ['(', ['(', ')']],
  [')', ['(', ')']],
  ['b', ['(', ')']],
  ['[', ['[', ']']],
  [']', ['[', ']']],
  ['{', ['{', '}']],
  ['}', ['{', '}']],
  ['B', ['{', '}']],
  ['<', ['<', '>']],
  ['>', ['<', '>']],
  ["'", ["'", "'"]],
  ['"', ['"', '"']],
  ['`', ['`', '`']],
  ['*', ['*', '*']],
  ['_', ['_', '_']],
  ['~', ['~', '~']],
  ['$', ['$', '$']],
])
const notebookSurroundPrefixes = new Set(['ys', 'ysi', 'ysa', 'cs', 'ds'])

export function notebookSurroundPair(token: string | undefined) {
  if (token === undefined) return undefined
  return notebookSurroundPairs.get(token)
}

export function notebookSurroundKeyPlan(
  buffer: string,
  key: string,
  visualMode: boolean,
  pendingOperator?: string,
): NotebookSurroundKeyPlan {
  if (visualMode) {
    if (buffer === '' && key === 'S') return { kind: 'pending', buffer: 'S' }
    if (buffer === 'S') {
      return notebookSurroundPair(key)
        ? { kind: 'surroundSelection', token: key }
        : { kind: 'flush' }
    }
    return { kind: 'pass' }
  }

  if (buffer === '') {
    if (key === 's') {
      if (pendingOperator === 'yank') return { kind: 'pending', buffer: 'ys' }
      if (pendingOperator === 'delete') return { kind: 'pending', buffer: 'ds' }
      if (pendingOperator === 'change') return { kind: 'pending', buffer: 'cs' }
    }
    return { kind: 'pass' }
  }

  if (buffer === 'ysiw' || buffer === 'ysw' || buffer === 'ysaw') {
    return notebookSurroundPair(key) ? { kind: 'surroundWord', token: key } : { kind: 'flush' }
  }
  if (buffer === 'yss') {
    return notebookSurroundPair(key) ? { kind: 'surroundLine', token: key } : { kind: 'flush' }
  }
  if (buffer === 'ds') {
    return notebookSurroundPair(key) ? { kind: 'deleteSurround', token: key } : { kind: 'flush' }
  }
  if (buffer === 'cs') {
    return notebookSurroundPair(key) ? { kind: 'pending', buffer: `cs${key}` } : { kind: 'flush' }
  }
  if (buffer.startsWith('cs') && buffer.length === 3) {
    return notebookSurroundPair(key)
      ? { kind: 'changeSurround', oldToken: buffer[2] ?? '', replacementToken: key }
      : { kind: 'flush' }
  }

  const nextBuffer = buffer + key
  if (
    notebookSurroundPrefixes.has(nextBuffer) ||
    nextBuffer === 'ysiw' ||
    nextBuffer === 'ysaw' ||
    nextBuffer === 'ysw' ||
    nextBuffer === 'yss' ||
    nextBuffer === 'ds' ||
    nextBuffer === 'cs'
  ) {
    return { kind: 'pending', buffer: nextBuffer }
  }
  return { kind: 'flush' }
}

function isNotebookWordChar(char: string | undefined) {
  return char !== undefined && /^[A-Za-z0-9_]$/.test(char)
}

export function notebookWordRangeAt(source: string, cursor: number): NotebookTextRange | undefined {
  if (source.length === 0) return undefined
  let index = Math.min(Math.max(0, cursor), source.length)
  if (index === source.length) index -= 1
  if (!isNotebookWordChar(source[index])) {
    if (index > 0 && isNotebookWordChar(source[index - 1])) index -= 1
    else return undefined
  }
  let from = index
  while (from > 0 && isNotebookWordChar(source[from - 1])) from -= 1
  let to = index + 1
  while (to < source.length && isNotebookWordChar(source[to])) to += 1
  return { from, to }
}

export function currentWordRange(view: CodeMirrorEditorView) {
  const head = view.state.selection.main.head
  const line = view.state.doc.lineAt(Math.min(head, view.state.doc.length))
  const range = notebookWordRangeAt(line.text, head - line.from)
  if (!range) return undefined
  return { from: line.from + range.from, to: line.from + range.to }
}

export function currentLineRange(view: CodeMirrorEditorView) {
  const line = view.state.doc.lineAt(view.state.selection.main.head)
  return { from: line.from, to: line.to }
}

export function currentSelectionRange(view: CodeMirrorEditorView) {
  const selection = view.state.selection.main
  const from = Math.min(selection.anchor, selection.head)
  const to = Math.max(selection.anchor, selection.head)
  return from === to ? undefined : { from, to }
}

function previousSurroundOpen(
  source: string,
  cursor: number,
  open: string,
  close: string,
): number | undefined {
  let depth = 0
  for (let index = Math.min(cursor, source.length - 1); index >= 0; index -= 1) {
    const char = source[index]
    if (char === close) {
      depth += 1
    } else if (char === open) {
      if (depth === 0) return index
      depth -= 1
    }
  }
}

function nextSurroundClose(
  source: string,
  cursor: number,
  open: string,
  close: string,
): number | undefined {
  let depth = 0
  for (let index = Math.max(0, cursor); index < source.length; index += 1) {
    const char = source[index]
    if (char === open) {
      depth += 1
    } else if (char === close) {
      if (depth === 0) return index
      depth -= 1
    }
  }
}

export function notebookSurroundingPairRange(
  source: string,
  cursor: number,
  token: string | undefined,
): NotebookSurroundingPairRange | undefined {
  const pair = notebookSurroundPair(token)
  if (!pair) return undefined
  const [open, close] = pair
  if (open === close) {
    const quoteCursor = Math.min(Math.max(0, cursor), source.length - 1)
    const nextQuote = source.indexOf(close, quoteCursor + close.length)
    const openFrom =
      source[quoteCursor] === open && nextQuote < 0
        ? source.lastIndexOf(open, quoteCursor - open.length)
        : source.lastIndexOf(open, quoteCursor)
    if (openFrom < 0) return undefined
    const closeFrom = source.indexOf(close, Math.max(openFrom + open.length, cursor))
    if (closeFrom < 0 || closeFrom === openFrom) return undefined
    return {
      openFrom,
      openTo: openFrom + open.length,
      closeFrom,
      closeTo: closeFrom + close.length,
    }
  }
  const searchCursor = Math.min(Math.max(0, cursor), source.length - 1)
  const openSearchCursor =
    source[searchCursor] === close ? searchCursor - close.length : searchCursor
  const openFrom = previousSurroundOpen(source, openSearchCursor, open, close)
  if (openFrom === undefined) return undefined
  const closeFrom = nextSurroundClose(source, Math.max(cursor, openFrom + open.length), open, close)
  if (closeFrom === undefined) return undefined
  return { openFrom, openTo: openFrom + open.length, closeFrom, closeTo: closeFrom + close.length }
}

function textIndexOf(
  doc: Text,
  character: string,
  from: number,
  to = doc.length,
): number | undefined {
  const start = Math.min(Math.max(0, from), doc.length)
  const end = Math.min(Math.max(start, to), doc.length)
  const iterator = doc.iterRange(start, end)
  let offset = start
  while (!iterator.next().done) {
    const index = iterator.value.indexOf(character)
    if (index >= 0) return offset + index
    offset += iterator.value.length
  }
}

function textLastIndexOf(doc: Text, character: string, toInclusive: number): number | undefined {
  const end = Math.min(Math.max(0, toInclusive + 1), doc.length)
  const iterator = doc.iterRange(end, 0)
  let offset = end
  while (!iterator.next().done) {
    offset -= iterator.value.length
    const index = iterator.value.lastIndexOf(character)
    if (index >= 0) return offset + index
  }
}

function previousSurroundOpenInText(
  doc: Text,
  cursor: number,
  open: string,
  close: string,
): number | undefined {
  const end = Math.min(Math.max(0, cursor + 1), doc.length)
  const iterator = doc.iterRange(end, 0)
  let offset = end
  let depth = 0
  while (!iterator.next().done) {
    const text = iterator.value
    offset -= text.length
    for (let index = text.length - 1; index >= 0; index -= 1) {
      const character = text[index]
      if (character === close) {
        depth += 1
      } else if (character === open) {
        if (depth === 0) return offset + index
        depth -= 1
      }
    }
  }
}

function nextSurroundCloseInText(
  doc: Text,
  cursor: number,
  open: string,
  close: string,
): number | undefined {
  const start = Math.min(Math.max(0, cursor), doc.length)
  const iterator = doc.iterRange(start, doc.length)
  let offset = start
  let depth = 0
  while (!iterator.next().done) {
    const text = iterator.value
    for (let index = 0; index < text.length; index += 1) {
      const character = text[index]
      if (character === open) {
        depth += 1
      } else if (character === close) {
        if (depth === 0) return offset + index
        depth -= 1
      }
    }
    offset += text.length
  }
}

export function notebookSurroundingPairRangeInText(
  doc: Text,
  cursor: number,
  token: string | undefined,
): NotebookSurroundingPairRange | undefined {
  const pair = notebookSurroundPair(token)
  if (!pair || doc.length === 0) return undefined
  const [open, close] = pair
  if (open === close) {
    const quoteCursor = Math.min(Math.max(0, cursor), doc.length - 1)
    const nextQuote = textIndexOf(doc, close, quoteCursor + close.length)
    const atQuote = doc.sliceString(quoteCursor, quoteCursor + open.length) === open
    const openFrom =
      atQuote && nextQuote === undefined
        ? textLastIndexOf(doc, open, quoteCursor - open.length)
        : textLastIndexOf(doc, open, quoteCursor)
    if (openFrom === undefined) return undefined
    const closeFrom = textIndexOf(doc, close, Math.max(openFrom + open.length, cursor))
    if (closeFrom === undefined || closeFrom === openFrom) return undefined
    return {
      openFrom,
      openTo: openFrom + open.length,
      closeFrom,
      closeTo: closeFrom + close.length,
    }
  }
  const searchCursor = Math.min(Math.max(0, cursor), doc.length - 1)
  const openSearchCursor =
    doc.sliceString(searchCursor, searchCursor + close.length) === close
      ? searchCursor - close.length
      : searchCursor
  const openFrom = previousSurroundOpenInText(doc, openSearchCursor, open, close)
  if (openFrom === undefined) return undefined
  const closeFrom = nextSurroundCloseInText(
    doc,
    Math.max(cursor, openFrom + open.length),
    open,
    close,
  )
  if (closeFrom === undefined) return undefined
  return { openFrom, openTo: openFrom + open.length, closeFrom, closeTo: closeFrom + close.length }
}

export function surroundRange(
  view: CodeMirrorEditorView,
  range: NotebookTextRange | undefined,
  token: string | undefined,
) {
  const pair = notebookSurroundPair(token)
  if (!pair || !range) return
  const [open, close] = pair
  view.dispatch({
    changes: [
      { from: range.from, insert: open },
      { from: range.to, insert: close },
    ],
    selection: { anchor: range.from + open.length, head: range.to + open.length },
    scrollIntoView: true,
  })
}

export function deleteSurroundRange(view: CodeMirrorEditorView, token: string | undefined) {
  const head = view.state.selection.main.head
  const range = notebookSurroundingPairRangeInText(view.state.doc, head, token)
  if (!range) return
  const pair = notebookSurroundPair(token)
  if (!pair) return
  const [open] = pair
  view.dispatch({
    changes: [
      { from: range.openFrom, to: range.openTo },
      { from: range.closeFrom, to: range.closeTo },
    ],
    selection: {
      anchor: Math.min(Math.max(range.openFrom, head - open.length), range.closeFrom - open.length),
    },
    scrollIntoView: true,
  })
}

export function changeSurroundRange(
  view: CodeMirrorEditorView,
  oldToken: string | undefined,
  replacementToken: string | undefined,
) {
  const range = notebookSurroundingPairRangeInText(
    view.state.doc,
    view.state.selection.main.head,
    oldToken,
  )
  const replacement = notebookSurroundPair(replacementToken)
  if (!range || !replacement) return
  const [open, close] = replacement
  view.dispatch({
    changes: [
      { from: range.openFrom, to: range.openTo, insert: open },
      { from: range.closeFrom, to: range.closeTo, insert: close },
    ],
    scrollIntoView: true,
  })
}
