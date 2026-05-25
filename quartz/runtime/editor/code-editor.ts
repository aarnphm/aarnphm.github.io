import type { EditorState as CodeMirrorEditorState, Extension, Text } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView, KeyBinding } from '@codemirror/view'
import type { CodeMirror } from '@replit/codemirror-vim'
import type { LanguageFn } from 'highlight.js'
import { ensureSyntaxTree, forceParsing, syntaxTreeAvailable } from '@codemirror/language'
import type { NotebookLspConfig } from '../lsp/pyright'
import { codemirrorCodeLanguage } from '../../util/codemirror-language'
import { backendFor } from '../notebook/registry'

export type NotebookCodeEditor = {
  getValue(): string
  setValue(content: string): void
  highlightedLines(): HTMLElement[]
  setVimMode(enabled: boolean): Promise<void>
  focus(): void
  destroy(): void
}

type NotebookCodeEditorConfig = {
  parent: HTMLElement
  initialContent?: string
  language?: string
  vimMode?: boolean
  onEdited?: () => void
  onChange?: (content: string) => void
  onSubmit?: () => void
  onSubmitAndAdvance?: () => void
  onSave?: () => void
  onCancel?: () => void
  lsp?: NotebookLspConfig
}

type NotebookCodeEditorWarmupConfig = {
  languages?: readonly string[]
  lsp?: boolean
  vimMode?: boolean
}

type VimApi = typeof import('@replit/codemirror-vim')
type NotebookCodeMirror = NonNullable<ReturnType<VimApi['getCM']>>
type NotebookVimModule = Pick<VimApi, 'Vim'>
export type NotebookVimBindingsApi = Pick<
  NotebookVimModule['Vim'],
  'defineAction' | 'mapCommand' | '_mapCommand'
>
export type NotebookTextRange = { from: number; to: number }
export type NotebookSurroundingPairRange = {
  openFrom: number
  openTo: number
  closeFrom: number
  closeTo: number
}
type NotebookSurroundActionArgs = {
  token?: string
  selectedCharacter?: string
  oldToken?: string
  replacementToken?: string
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
export type NotebookLeapMotion = {
  key: string
  backward: boolean
  offset: -1 | 0 | 1
  forwardKey: string
  backwardKey: string
}
export type NotebookLeapTarget = { matchFrom: number; matchTo: number; target: number }
export type NotebookLeapSearchRange = { from: number; to: number }
export type NotebookCursorCoordinate = { line: number; column: number }
export type NotebookLeapStatus = {
  key: string
  character?: string
  activeIndex?: number
  targetCount?: number
}
type NotebookLeapSession = {
  motion: NotebookLeapMotion
  character: string
  targets: readonly NotebookLeapTarget[]
  activeIndex: number
  visualMode: boolean
  anchor: number
}

let notebookVimBindingsConfigured = false
let notebookCodeEditorCoreWarmup: Promise<void> | undefined
let notebookCodeEditorLspWarmup: Promise<void> | undefined
let notebookCodeEditorVimWarmup: Promise<void> | undefined
const notebookLanguageWarmups = new Map<string, Promise<Extension>>()
const notebookRunKeys = ['Ctrl-Enter', 'Shift-Enter', 'Alt-Enter'] as const
const notebookHighlightJsLanguages = new Set<string>()
const notebookVimReservedControlKeys = new Set([
  'Ctrl-b',
  'Ctrl-d',
  'Ctrl-e',
  'Ctrl-f',
  'Ctrl-u',
  'Ctrl-y',
])
const notebookHighlightParseTimeoutMs = 1500
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
])
const notebookSurroundPrefixes = new Set(['y', 'ys', 'ysi', 'ysa', 'c', 'd'])
const notebookLeapPendingDelayMs = 900
const notebookLeapTraversalDelayMs = 2800
const notebookLeapSliceSearchMaxLength = 16384
export const notebookVimNoremaps = [
  ['insert', 'jj', '<Esc>'],
  ['insert', 'jk', '<Esc>'],
  ['insert', '<A-BS>', '<C-w>'],
  ['insert', '<M-BS>', '<C-w>'],
  ['normal', 'Y', 'y$'],
  ['normal', 'D', 'd$'],
  ['normal', '\\', ':noh<CR>'],
  ['visual', '<', '<gv'],
  ['visual', '>', '>gv'],
] as const

function languageWarmupKey(language: string | undefined): string {
  return (language ?? '').trim().toLowerCase() || 'python'
}

export async function notebookCodeEditorLanguageExtension(
  language: string | undefined,
): Promise<Extension> {
  const name = (language ?? '').trim().toLowerCase()
  const backendExtension = await backendFor(name)?.editor?.languageExtension?.()
  if (backendExtension) return backendExtension
  return codemirrorCodeLanguage(language)
}

function warmLanguageExtension(language: string | undefined): Promise<Extension> {
  const key = languageWarmupKey(language)
  let warmup = notebookLanguageWarmups.get(key)
  if (!warmup) {
    warmup = notebookCodeEditorLanguageExtension(language)
    notebookLanguageWarmups.set(key, warmup)
  }
  return warmup
}

async function notebookHighlightJsLanguage(
  language: string | undefined,
): Promise<{ name: string; define: LanguageFn } | undefined> {
  const name = languageWarmupKey(language)
  if (name === 'python' || name === 'py' || name === 'ipython') {
    return { name: 'python', define: (await import('highlight.js/lib/languages/python')).default }
  }
  if (name === 'javascript' || name === 'js' || name === 'jsx') {
    return {
      name: 'javascript',
      define: (await import('highlight.js/lib/languages/javascript')).default,
    }
  }
  if (name === 'typescript' || name === 'ts' || name === 'tsx') {
    return {
      name: 'typescript',
      define: (await import('highlight.js/lib/languages/typescript')).default,
    }
  }
  if (name === 'go' || name === 'golang') {
    return { name: 'go', define: (await import('highlight.js/lib/languages/go')).default }
  }
  if (name === 'rust' || name === 'rs') {
    return { name: 'rust', define: (await import('highlight.js/lib/languages/rust')).default }
  }
  if (name === 'haskell' || name === 'hs') {
    return { name: 'haskell', define: (await import('highlight.js/lib/languages/haskell')).default }
  }
  if (name === 'ocaml' || name === 'ml') {
    return { name: 'ocaml', define: (await import('highlight.js/lib/languages/ocaml')).default }
  }
  if (name === 'bash' || name === 'sh' || name === 'shell' || name === 'zsh') {
    return { name: 'bash', define: (await import('highlight.js/lib/languages/bash')).default }
  }
}

async function highlightNotebookSource(source: string, language: string | undefined) {
  const languageDefinition = await notebookHighlightJsLanguage(language)
  if (!languageDefinition) return undefined
  const hljs = (await import('highlight.js/lib/core')).default
  if (!notebookHighlightJsLanguages.has(languageDefinition.name)) {
    hljs.registerLanguage(languageDefinition.name, languageDefinition.define)
    notebookHighlightJsLanguages.add(languageDefinition.name)
  }
  return hljs.highlight(source, { language: languageDefinition.name, ignoreIllegals: true }).value
}

function createHighlightedLine(): HTMLElement {
  const line = document.createElement('span')
  line.dataset.line = ''
  return line
}

function appendWrappedText(line: HTMLElement, wrappers: readonly HTMLElement[], text: string) {
  if (text.length === 0) return
  let node: Node = document.createTextNode(text)
  for (let index = wrappers.length - 1; index >= 0; index -= 1) {
    const wrapper = wrappers[index]?.cloneNode(false)
    if (!(wrapper instanceof HTMLElement)) continue
    wrapper.append(node)
    node = wrapper
  }
  line.append(node)
}

function highlightedHtmlLineSpans(html: string): HTMLElement[] {
  const template = document.createElement('template')
  template.innerHTML = html
  const lines = [createHighlightedLine()]
  const wrappers: HTMLElement[] = []
  const currentLine = () => lines[lines.length - 1] ?? lines[0]
  const startLine = () => {
    lines.push(createHighlightedLine())
  }
  const visit = (node: ChildNode) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const parts = (node.textContent ?? '').split('\n')
      parts.forEach((part, index) => {
        appendWrappedText(currentLine(), wrappers, part)
        if (index < parts.length - 1) startLine()
      })
      return
    }
    if (node instanceof HTMLElement) {
      wrappers.push(node)
      for (const child of Array.from(node.childNodes)) visit(child)
      wrappers.pop()
      return
    }
    for (const child of Array.from(node.childNodes)) visit(child)
  }
  for (const child of Array.from(template.content.childNodes)) visit(child)
  return lines
}

function notebookKeyBindingUsesVimControl(binding: KeyBinding): boolean {
  return [binding.key, binding.mac, binding.win, binding.linux].some(
    key => key !== undefined && notebookVimReservedControlKeys.has(key),
  )
}

async function lspExtensions(config: NotebookLspConfig | undefined): Promise<readonly Extension[]> {
  if (!config?.enabled) return []
  const bridgeFactory = backendFor(config.language)?.editor?.lspBridge
  if (!bridgeFactory) return []
  try {
    const bridge = await bridgeFactory()
    return await bridge.extensions(config)
  } catch (error) {
    console.warn('notebook lsp extension failed to load', error)
    return []
  }
}

function moveSelectedLines(cm: NotebookCodeMirror, direction: -1 | 1) {
  const view = cm.cm6
  const doc = view.state.doc
  const range = view.state.selection.main
  const from = Math.min(range.anchor, range.head)
  const to = Math.max(range.anchor, range.head)
  const startLineNumber = doc.lineAt(from).number
  const endLineNumber = doc.lineAt(to > from ? to - 1 : to).number
  if (direction < 0 && startLineNumber === 1) return
  if (direction > 0 && endLineNumber === doc.lines) return

  const startLine = doc.line(startLineNumber)
  const endLine = doc.line(endLineNumber)
  const selectedText = doc.sliceString(startLine.from, endLine.to)

  if (direction < 0) {
    const previousLine = doc.line(startLineNumber - 1)
    const previousText = doc.sliceString(previousLine.from, previousLine.to)
    view.dispatch({
      changes: {
        from: previousLine.from,
        to: endLine.to,
        insert: `${selectedText}\n${previousText}`,
      },
      selection: { anchor: previousLine.from, head: previousLine.from + selectedText.length },
      scrollIntoView: true,
    })
    return
  }

  const nextLine = doc.line(endLineNumber + 1)
  const nextText = doc.sliceString(nextLine.from, nextLine.to)
  const nextSelectionFrom = startLine.from + nextText.length + 1
  view.dispatch({
    changes: { from: startLine.from, to: nextLine.to, insert: `${nextText}\n${selectedText}` },
    selection: { anchor: nextSelectionFrom, head: nextSelectionFrom + selectedText.length },
    scrollIntoView: true,
  })
}

export function notebookSurroundPair(token: string | undefined) {
  if (token === undefined) return undefined
  return notebookSurroundPairs.get(token)
}

function notebookSurroundToken(args: NotebookSurroundActionArgs) {
  return args.selectedCharacter ?? args.token
}

export function notebookSurroundKeyPlan(
  buffer: string,
  key: string,
  visualMode: boolean,
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
    return key === 'y' || key === 'd' || key === 'c'
      ? { kind: 'pending', buffer: key }
      : { kind: 'pass' }
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

export function notebookLeapMotionForKey(key: string): NotebookLeapMotion | undefined {
  if (key === 'f') return { key, backward: false, offset: 0, forwardKey: 'f', backwardKey: 'F' }
  if (key === 'F') return { key, backward: true, offset: 0, forwardKey: 'f', backwardKey: 'F' }
  if (key === 't') return { key, backward: false, offset: -1, forwardKey: 't', backwardKey: 'T' }
  if (key === 'T') return { key, backward: true, offset: 1, forwardKey: 't', backwardKey: 'T' }
}

function notebookLeapTargetPosition(matchFrom: number, sourceLength: number, offset: -1 | 0 | 1) {
  return Math.min(Math.max(0, matchFrom + offset), sourceLength)
}

export function notebookLeapTargets(
  source: string,
  cursor: number,
  character: string,
  motion: NotebookLeapMotion,
  ranges: readonly NotebookLeapSearchRange[] = [{ from: 0, to: source.length }],
): readonly NotebookLeapTarget[] {
  if (character.length !== 1) return []
  const matches: NotebookLeapTarget[] = []
  for (const range of ranges) {
    const from = Math.min(Math.max(0, range.from), source.length)
    const to = Math.min(Math.max(from, range.to), source.length)
    let index = source.indexOf(character, from)
    while (index >= 0 && index < to) {
      if ((!motion.backward && index > cursor) || (motion.backward && index < cursor)) {
        matches.push({
          matchFrom: index,
          matchTo: index + character.length,
          target: notebookLeapTargetPosition(index, source.length, motion.offset),
        })
      }
      index = source.indexOf(character, index + character.length)
    }
  }
  return matches.sort((left, right) =>
    motion.backward ? right.matchFrom - left.matchFrom : left.matchFrom - right.matchFrom,
  )
}

export function notebookLeapTargetsInText(
  doc: Text,
  cursor: number,
  character: string,
  motion: NotebookLeapMotion,
  ranges: readonly NotebookLeapSearchRange[] = [{ from: 0, to: doc.length }],
): readonly NotebookLeapTarget[] {
  if (character.length !== 1) return []
  const matches: NotebookLeapTarget[] = []
  for (const range of ranges) {
    const from = Math.min(Math.max(0, range.from), doc.length)
    const to = Math.min(Math.max(from, range.to), doc.length)
    if (to - from <= notebookLeapSliceSearchMaxLength) {
      const text = doc.sliceString(from, to)
      let relativeIndex = text.indexOf(character)
      while (relativeIndex >= 0) {
        const index = from + relativeIndex
        if ((!motion.backward && index > cursor) || (motion.backward && index < cursor)) {
          matches.push({
            matchFrom: index,
            matchTo: index + character.length,
            target: notebookLeapTargetPosition(index, doc.length, motion.offset),
          })
        }
        relativeIndex = text.indexOf(character, relativeIndex + character.length)
      }
    } else {
      const iterator = doc.iterRange(from, to)
      let offset = from
      while (!iterator.next().done) {
        const text = iterator.value
        let relativeIndex = text.indexOf(character)
        while (relativeIndex >= 0) {
          const index = offset + relativeIndex
          if ((!motion.backward && index > cursor) || (motion.backward && index < cursor)) {
            matches.push({
              matchFrom: index,
              matchTo: index + character.length,
              target: notebookLeapTargetPosition(index, doc.length, motion.offset),
            })
          }
          relativeIndex = text.indexOf(character, relativeIndex + character.length)
        }
        offset += text.length
      }
    }
  }
  return matches.sort((left, right) =>
    motion.backward ? right.matchFrom - left.matchFrom : left.matchFrom - right.matchFrom,
  )
}

export function notebookCursorCoordinate(source: string, cursor: number): NotebookCursorCoordinate {
  const clampedCursor = Math.min(Math.max(0, cursor), source.length)
  let line = 1
  let lineStart = 0
  for (let index = 0; index < clampedCursor; index++) {
    if (source.charCodeAt(index) === 10) {
      line++
      lineStart = index + 1
    }
  }
  return { line, column: clampedCursor - lineStart + 1 }
}

export function notebookCursorCoordinateInText(
  doc: Text,
  cursor: number,
): NotebookCursorCoordinate {
  const clampedCursor = Math.min(Math.max(0, cursor), doc.length)
  const line = doc.lineAt(clampedCursor)
  return { line: line.number, column: clampedCursor - line.from + 1 }
}

function notebookEditorStatusTextForCoordinate(
  coordinate: NotebookCursorCoordinate,
  leap?: NotebookLeapStatus,
) {
  const position = `${coordinate.line}:${coordinate.column}`
  if (!leap) return position
  if (leap.character === undefined) return `${leap.key}_ | ${position}`
  const targetPosition =
    leap.activeIndex !== undefined && leap.targetCount !== undefined && leap.targetCount > 0
      ? ` ${leap.activeIndex + 1}/${leap.targetCount}`
      : ''
  return `${leap.key}${leap.character}${targetPosition} | ${position}`
}

export function notebookEditorStatusText(
  source: string,
  cursor: number,
  leap?: NotebookLeapStatus,
): string {
  return notebookEditorStatusTextForCoordinate(notebookCursorCoordinate(source, cursor), leap)
}

export function notebookEditorStatusTextInText(
  doc: Text,
  cursor: number,
  leap?: NotebookLeapStatus,
): string {
  return notebookEditorStatusTextForCoordinate(notebookCursorCoordinateInText(doc, cursor), leap)
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

function currentWordRange(view: CodeMirrorEditorView) {
  const head = view.state.selection.main.head
  const line = view.state.doc.lineAt(Math.min(head, view.state.doc.length))
  const range = notebookWordRangeAt(line.text, head - line.from)
  if (!range) return undefined
  return { from: line.from + range.from, to: line.from + range.to }
}

function currentLineRange(view: CodeMirrorEditorView) {
  const line = view.state.doc.lineAt(view.state.selection.main.head)
  return { from: line.from, to: line.to }
}

function currentSelectionRange(view: CodeMirrorEditorView) {
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

function surroundRange(
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

function deleteSurroundRange(view: CodeMirrorEditorView, token: string | undefined) {
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

function changeSurroundRange(
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

export function registerNotebookSurroundBindings(Vim: NotebookVimBindingsApi) {
  Vim.defineAction('notebookSurroundSelection', (cm, actionArgs) =>
    surroundRange(cm.cm6, currentSelectionRange(cm.cm6), notebookSurroundToken(actionArgs)),
  )
  Vim.defineAction('notebookSurroundWord', (cm, actionArgs) =>
    surroundRange(cm.cm6, currentWordRange(cm.cm6), notebookSurroundToken(actionArgs)),
  )
  Vim.defineAction('notebookSurroundLine', (cm, actionArgs) =>
    surroundRange(cm.cm6, currentLineRange(cm.cm6), notebookSurroundToken(actionArgs)),
  )
  Vim.defineAction('notebookDeleteSurround', (cm, actionArgs) =>
    deleteSurroundRange(cm.cm6, notebookSurroundToken(actionArgs)),
  )
  Vim.defineAction('notebookChangeSurround', (cm, actionArgs: NotebookSurroundActionArgs) =>
    changeSurroundRange(cm.cm6, actionArgs.oldToken, actionArgs.replacementToken),
  )
  Vim._mapCommand({
    keys: 'S<character>',
    type: 'action',
    action: 'notebookSurroundSelection',
    actionArgs: {},
    context: 'visual',
    isEdit: true,
  })
  Vim._mapCommand({
    keys: 'ysiw<character>',
    type: 'action',
    action: 'notebookSurroundWord',
    actionArgs: {},
    context: 'normal',
    isEdit: true,
  })
  Vim._mapCommand({
    keys: 'ysw<character>',
    type: 'action',
    action: 'notebookSurroundWord',
    actionArgs: {},
    context: 'normal',
    isEdit: true,
  })
  Vim._mapCommand({
    keys: 'yss<character>',
    type: 'action',
    action: 'notebookSurroundLine',
    actionArgs: {},
    context: 'normal',
    isEdit: true,
  })
  Vim._mapCommand({
    keys: 'ds<character>',
    type: 'action',
    action: 'notebookDeleteSurround',
    actionArgs: {},
    context: 'normal',
    isEdit: true,
  })
  for (const oldToken of notebookSurroundPairs.keys()) {
    for (const replacementToken of notebookSurroundPairs.keys()) {
      Vim.mapCommand(
        `cs${oldToken}${replacementToken}`,
        'action',
        'notebookChangeSurround',
        { oldToken, replacementToken },
        { context: 'normal', isEdit: true },
      )
    }
  }
}

function configureNotebookVimBindings(vimApi: NotebookVimModule) {
  if (notebookVimBindingsConfigured) return
  notebookVimBindingsConfigured = true
  const { Vim } = vimApi
  for (const [context, lhs, rhs] of notebookVimNoremaps) {
    Vim.noremap(lhs, rhs, context)
  }
  Vim.defineAction('notebookNoop', () => {})
  Vim.defineAction('notebookMoveSelectedLinesDown', cm => moveSelectedLines(cm, 1))
  Vim.defineAction('notebookMoveSelectedLinesUp', cm => moveSelectedLines(cm, -1))
  Vim.mapCommand('<Space>', 'action', 'notebookNoop', {}, { context: 'normal' })
  Vim.mapCommand('<Space>', 'action', 'notebookNoop', {}, { context: 'visual' })
  Vim.mapCommand(
    'J',
    'action',
    'notebookMoveSelectedLinesDown',
    {},
    { context: 'visual', isEdit: true },
  )
  Vim.mapCommand(
    'K',
    'action',
    'notebookMoveSelectedLinesUp',
    {},
    { context: 'visual', isEdit: true },
  )
  registerNotebookSurroundBindings(Vim)
}

export async function warmNotebookCodeEditorAssets(
  config: NotebookCodeEditorWarmupConfig = {},
): Promise<void> {
  notebookCodeEditorCoreWarmup ??= Promise.all([
    import('@codemirror/commands'),
    import('@codemirror/autocomplete'),
    import('@codemirror/language'),
    import('@codemirror/state'),
    import('@codemirror/view'),
  ]).then(() => {})

  const languages = config.languages && config.languages.length > 0 ? config.languages : ['python']
  const warmups: Promise<unknown>[] = [
    notebookCodeEditorCoreWarmup,
    ...languages.map(language => warmLanguageExtension(language)),
  ]

  if (config.lsp) {
    notebookCodeEditorLspWarmup ??= Promise.all(
      languages.map(async language => {
        const bridge = await backendFor(language)?.editor?.lspBridge?.()
        if (bridge) await import('../lsp/pyright').then(module => module.warmNotebookLspAssets())
      }),
    ).then(() => {})
    warmups.push(notebookCodeEditorLspWarmup)
  }

  if (config.vimMode) {
    notebookCodeEditorVimWarmup ??= import('@replit/codemirror-vim').then(vimApi => {
      configureNotebookVimBindings(vimApi)
    })
    warmups.push(notebookCodeEditorVimWarmup)
  }

  await Promise.all(warmups)
}

function inlineHighlightStyle(source: HTMLElement, target: HTMLElement) {
  const style = getComputedStyle(source)
  if (style.color) {
    target.style.color = style.color
    target.style.setProperty('--shiki-light', style.color)
    target.style.setProperty('--shiki-dark', style.color)
  }
  if (style.fontStyle && style.fontStyle !== 'normal') target.style.fontStyle = style.fontStyle
  if (style.fontWeight && style.fontWeight !== '400') target.style.fontWeight = style.fontWeight
  if (style.textDecorationLine && style.textDecorationLine !== 'none') {
    target.style.textDecorationLine = style.textDecorationLine
  }
  const sourceChildren = Array.from(source.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement,
  )
  const targetChildren = Array.from(target.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement,
  )
  sourceChildren.forEach((child, index) => {
    const targetChild = targetChildren[index]
    if (targetChild) inlineHighlightStyle(child, targetChild)
  })
}

function cloneHighlightedNode(node: ChildNode): Node {
  const cloned = node.cloneNode(true)
  if (node instanceof HTMLElement && cloned instanceof HTMLElement) {
    inlineHighlightStyle(node, cloned)
  }
  return cloned
}

function highlightedLineSpans(view: CodeMirrorEditorView): HTMLElement[] {
  const upto = view.state.doc.length
  if (!syntaxTreeAvailable(view.state, upto)) {
    ensureSyntaxTree(view.state, upto, notebookHighlightParseTimeoutMs)
    forceParsing(view, upto, notebookHighlightParseTimeoutMs)
  }
  return Array.from(view.dom.querySelectorAll<HTMLElement>('.cm-content > .cm-line')).map(line => {
    const span = document.createElement('span')
    span.dataset.line = ''
    for (const child of Array.from(line.childNodes)) {
      span.append(cloneHighlightedNode(child))
    }
    return span
  })
}

export async function renderNotebookHighlightedLines(
  source: string,
  languageName: string | undefined,
): Promise<HTMLElement[]> {
  const highlightedSource = await highlightNotebookSource(source, languageName)
  if (highlightedSource !== undefined) return highlightedHtmlLineSpans(highlightedSource)

  const [{ syntaxHighlighting, defaultHighlightStyle }, { EditorState }, { EditorView }, language] =
    await Promise.all([
      import('@codemirror/language'),
      import('@codemirror/state'),
      import('@codemirror/view'),
      notebookCodeEditorLanguageExtension(languageName),
    ])
  const lineCount = source.split(/\r?\n/).length
  const host = document.createElement('div')
  host.style.position = 'fixed'
  host.style.left = '-10000px'
  host.style.top = '0'
  host.style.width = '1000px'
  host.style.height = `${Math.max(1, lineCount) * 24}px`
  host.style.visibility = 'hidden'
  host.style.pointerEvents = 'none'
  document.body.append(host)
  const state = EditorState.create({
    doc: source,
    extensions: [
      language,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      EditorView.theme({
        '&': { fontFamily: 'monospace', fontSize: '16px', lineHeight: '24px' },
        '.cm-scroller': { overflow: 'visible' },
      }),
    ],
  })
  const view = new EditorView({ state, parent: host })
  highlightedLineSpans(view)
  await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
  const lines = highlightedLineSpans(view)
  view.destroy()
  host.remove()
  return lines
}

export async function createNotebookCodeEditor(
  config: NotebookCodeEditorConfig,
): Promise<NotebookCodeEditor> {
  const [
    { defaultKeymap, historyKeymap, history, indentWithTab },
    {
      acceptCompletion,
      clearSnippet,
      closeCompletion,
      completionStatus,
      moveCompletionSelection,
      nextSnippetField,
      prevSnippetField,
      snippetKeymap,
      startCompletion,
    },
    { syntaxHighlighting, defaultHighlightStyle },
    { Compartment, EditorState, Prec, StateEffect, StateField },
    { Decoration, EditorView, keymap, lineNumbers },
    language,
    lsp,
  ] = await Promise.all([
    import('@codemirror/commands'),
    import('@codemirror/autocomplete'),
    import('@codemirror/language'),
    import('@codemirror/state'),
    import('@codemirror/view'),
    notebookCodeEditorLanguageExtension(config.language),
    lspExtensions(config.lsp),
  ])

  const initialVimMode = config.vimMode === true
  let vimModeEnabled = initialVimMode
  const editorKeymap = (): Extension =>
    keymap.of([
      indentWithTab,
      ...(vimModeEnabled
        ? defaultKeymap.filter(binding => !notebookKeyBindingUsesVimControl(binding))
        : defaultKeymap),
      ...historyKeymap,
    ])
  const submit = () => {
    config.onSubmit?.()
    return true
  }
  const submitAndAdvance = () => {
    if (config.onSubmitAndAdvance) {
      config.onSubmitAndAdvance()
    } else {
      config.onSubmit?.()
    }
    return true
  }
  const notebookKeymap = (): Extension =>
    Prec.highest(
      keymap.of([
        { key: 'Mod-Enter', run: submitAndAdvance },
        ...notebookRunKeys.map(key => ({ key, run: submit })),
        {
          key: 'Mod-s',
          run: () => {
            config.onSave?.()
            return true
          },
        },
        {
          key: 'Ctrl-s',
          run: () => {
            config.onSave?.()
            return true
          },
        },
        {
          key: 'Escape',
          run: () => {
            if (vimModeEnabled) return false
            config.onCancel?.()
            return true
          },
        },
      ]),
    )

  let vimApi: Pick<VimApi, 'getCM' | 'vim' | 'Vim'> | undefined
  const loadVimApi = async (): Promise<Pick<VimApi, 'getCM' | 'vim' | 'Vim'>> => {
    vimApi ??= await import('@replit/codemirror-vim')
    configureNotebookVimBindings(vimApi)
    return vimApi
  }

  const vimExtension = async (enabled: boolean): Promise<Extension> => {
    if (!enabled) return []
    const [{ vim }, { drawSelection }] = await Promise.all([
      loadVimApi(),
      import('@codemirror/view'),
    ])
    return [vim(), drawSelection()]
  }

  const vimCompartment = new Compartment()
  const keymapCompartment = new Compartment()
  const editorKeymapCompartment = new Compartment()
  const vimAcceptsText = (view: CodeMirrorEditorView) =>
    !vimModeEnabled || vimApi?.getCM(view)?.state.vim?.insertMode === true
  const runTextCommand = (
    view: CodeMirrorEditorView,
    command: (target: CodeMirrorEditorView) => boolean,
  ) => (vimAcceptsText(view) ? command(view) : false)
  const moveActiveCompletion = (
    view: CodeMirrorEditorView,
    command: (target: CodeMirrorEditorView) => boolean,
  ) => (vimAcceptsText(view) && completionStatus(view.state) === 'active' ? command(view) : false)
  const closeEditorCompletion = (view: CodeMirrorEditorView) => {
    const closed = closeCompletion(view)
    return vimModeEnabled ? false : closed
  }
  const clearEditorSnippet = (view: CodeMirrorEditorView) => {
    const cleared = clearSnippet(view)
    return vimModeEnabled ? false : cleared
  }
  const moveCompletionDown = moveCompletionSelection(true)
  const moveCompletionUp = moveCompletionSelection(false)
  const moveCompletionPageDown = moveCompletionSelection(true, 'page')
  const moveCompletionPageUp = moveCompletionSelection(false, 'page')
  const completionKeymap = Prec.highest(
    keymap.of([
      { key: 'Tab', run: view => runTextCommand(view, acceptCompletion) },
      { key: 'Ctrl-Space', run: view => runTextCommand(view, startCompletion) },
      { mac: 'Alt-`', run: view => runTextCommand(view, startCompletion) },
      { mac: 'Alt-i', run: view => runTextCommand(view, startCompletion) },
      { key: 'Escape', run: closeEditorCompletion },
      { key: 'ArrowDown', run: view => runTextCommand(view, moveCompletionDown) },
      { key: 'ArrowUp', run: view => runTextCommand(view, moveCompletionUp) },
      { key: 'Ctrl-n', run: view => moveActiveCompletion(view, moveCompletionDown) },
      { key: 'Ctrl-p', run: view => moveActiveCompletion(view, moveCompletionUp) },
      { key: 'PageDown', run: view => runTextCommand(view, moveCompletionPageDown) },
      { key: 'PageUp', run: view => runTextCommand(view, moveCompletionPageUp) },
      { key: 'Enter', run: view => runTextCommand(view, acceptCompletion) },
    ]),
  )
  const editorSnippetKeymap = snippetKeymap.of([
    {
      key: 'Tab',
      run: view => runTextCommand(view, nextSnippetField),
      shift: view => runTextCommand(view, prevSnippetField),
    },
    { key: 'Escape', run: clearEditorSnippet },
  ])
  const leapHighlightEffect = StateEffect.define<
    { targets: readonly NotebookLeapTarget[]; activeIndex: number } | undefined
  >()
  const leapDecorationSet = (targets: readonly NotebookLeapTarget[], activeIndex: number) =>
    Decoration.set(
      targets.map((target, index) =>
        Decoration.mark({
          class: index === activeIndex ? 'cm-notebookLeapMatch' : 'cm-notebookLeapBackdrop',
        }).range(target.matchFrom, target.matchTo),
      ),
      true,
    )
  const leapHighlightField = StateField.define({
    create: () => Decoration.none,
    update(highlights, transaction) {
      let next = transaction.docChanged ? highlights.map(transaction.changes) : highlights
      for (const effect of transaction.effects) {
        if (effect.is(leapHighlightEffect)) {
          next = effect.value
            ? leapDecorationSet(effect.value.targets, effect.value.activeIndex)
            : Decoration.none
        }
      }
      return next
    },
    provide: field => EditorView.decorations.from(field),
  })
  let leapPendingMotion: NotebookLeapMotion | undefined
  let leapSession: NotebookLeapSession | undefined
  let leapClearTimer: number | undefined
  const statusElement = document.createElement('div')
  statusElement.className = 'cm-notebookStatus'
  const currentLeapStatus = (): NotebookLeapStatus | undefined =>
    leapSession
      ? {
          key: leapSession.motion.key,
          character: leapSession.character,
          activeIndex: leapSession.activeIndex,
          targetCount: leapSession.targets.length,
        }
      : leapPendingMotion
        ? { key: leapPendingMotion.key }
        : undefined
  const updateStatusElement = (state: CodeMirrorEditorState) => {
    const leap = currentLeapStatus()
    statusElement.textContent = notebookEditorStatusTextInText(
      state.doc,
      state.selection.main.head,
      leap,
    )
    statusElement.dataset.active = leap ? 'true' : 'false'
  }
  const clearLeapTimer = () => {
    if (leapClearTimer !== undefined) {
      window.clearTimeout(leapClearTimer)
      leapClearTimer = undefined
    }
  }
  const clearLeap = () => {
    leapPendingMotion = undefined
    leapSession = undefined
    clearLeapTimer()
    view.dispatch({ effects: leapHighlightEffect.of(undefined) })
  }
  const scheduleLeapClear = (delayMs: number) => {
    clearLeapTimer()
    leapClearTimer = window.setTimeout(() => clearLeap(), delayMs)
  }
  const applyLeapTarget = (
    session: NotebookLeapSession,
    activeIndex: number,
    targetView: CodeMirrorEditorView,
  ) => {
    const target = session.targets[activeIndex]
    if (!target) return
    leapSession = { ...session, activeIndex }
    targetView.dispatch({
      selection: session.visualMode
        ? { anchor: session.anchor, head: target.target }
        : { anchor: target.target },
      effects: leapHighlightEffect.of({ targets: session.targets, activeIndex }),
      scrollIntoView: true,
    })
    scheduleLeapClear(notebookLeapTraversalDelayMs)
  }
  const visibleLeapRanges = (
    targetView: CodeMirrorEditorView,
  ): readonly NotebookLeapSearchRange[] =>
    targetView.visibleRanges.map(range => ({ from: range.from, to: range.to }))
  const eventCharacterKey = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return undefined
    return event.key.length === 1 ? event.key : undefined
  }
  const leapKeyHandler = Prec.highest(
    EditorView.domEventHandlers({
      keydown(event, target) {
        if (!vimModeEnabled) {
          clearLeap()
          return false
        }
        const cm = vimApi?.getCM(target)
        const vimState = cm?.state.vim
        if (!cm || !vimApi || !vimState || vimState.insertMode) {
          clearLeap()
          return false
        }
        const key = eventCharacterKey(event)
        if (!key) {
          clearLeap()
          return false
        }

        if (leapSession) {
          if (key === leapSession.motion.forwardKey || key === leapSession.motion.backwardKey) {
            event.preventDefault()
            event.stopPropagation()
            const direction = key === leapSession.motion.forwardKey ? 1 : -1
            const nextIndex =
              (leapSession.activeIndex + direction + leapSession.targets.length) %
              leapSession.targets.length
            applyLeapTarget(leapSession, nextIndex, target)
            return true
          }
          clearLeap()
          return false
        }

        if (leapPendingMotion) {
          event.preventDefault()
          event.stopPropagation()
          const motion = leapPendingMotion
          leapPendingMotion = undefined
          const selection = target.state.selection.main
          const cursor = selection.head
          const targets = notebookLeapTargetsInText(
            target.state.doc,
            cursor,
            key,
            motion,
            visibleLeapRanges(target),
          )
          if (targets.length === 0) {
            clearLeap()
            return true
          }
          applyLeapTarget(
            {
              motion,
              character: key,
              targets,
              activeIndex: 0,
              visualMode: vimState.visualMode === true,
              anchor: selection.anchor,
            },
            0,
            target,
          )
          return true
        }

        const motion = notebookLeapMotionForKey(key)
        if (!motion || vimState.inputState.operator) return false
        event.preventDefault()
        event.stopPropagation()
        clearLeapTimer()
        leapPendingMotion = motion
        leapSession = undefined
        updateStatusElement(target.state)
        scheduleLeapClear(notebookLeapPendingDelayMs)
        return true
      },
    }),
  )
  let surroundPendingBuffer = ''
  let surroundPendingFlushTimer: number | undefined
  const clearSurroundPending = () => {
    surroundPendingBuffer = ''
    if (surroundPendingFlushTimer !== undefined) {
      window.clearTimeout(surroundPendingFlushTimer)
      surroundPendingFlushTimer = undefined
    }
  }
  const replaySurroundPending = (target: CodeMirrorEditorView) => {
    if (surroundPendingBuffer.length === 0) return
    const pending = Array.from(surroundPendingBuffer)
    clearSurroundPending()
    const cm = vimApi?.getCM(target)
    if (!cm || !vimApi) return
    for (const key of pending) {
      vimApi.Vim.handleKey(cm, key, 'mapping')
    }
  }
  const scheduleSurroundPendingFlush = (target: CodeMirrorEditorView) => {
    if (surroundPendingFlushTimer !== undefined) window.clearTimeout(surroundPendingFlushTimer)
    surroundPendingFlushTimer = window.setTimeout(() => replaySurroundPending(target), 900)
  }
  const eventSurroundKey = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return undefined
    return event.key.length === 1 ? event.key : undefined
  }
  const surroundKeyHandler = Prec.highest(
    EditorView.domEventHandlers({
      keydown(event, target) {
        if (!vimModeEnabled) {
          clearSurroundPending()
          return false
        }
        const cm = vimApi?.getCM(target)
        const vimState = cm?.state.vim
        if (!cm || !vimApi || !vimState || vimState.insertMode) {
          clearSurroundPending()
          return false
        }
        const key = eventSurroundKey(event)
        if (!key) {
          replaySurroundPending(target)
          return false
        }
        const plan = notebookSurroundKeyPlan(
          surroundPendingBuffer,
          key,
          vimState.visualMode === true,
        )
        if (plan.kind === 'pass') return false
        if (plan.kind === 'flush') {
          replaySurroundPending(target)
          return false
        }

        event.preventDefault()
        event.stopPropagation()
        if (plan.kind === 'pending') {
          surroundPendingBuffer = plan.buffer
          scheduleSurroundPendingFlush(target)
          return true
        }

        clearSurroundPending()
        if (plan.kind === 'surroundSelection') {
          surroundRange(target, currentSelectionRange(target), plan.token)
          vimApi.Vim.handleKey(cm, '<Esc>', 'mapping')
        } else if (plan.kind === 'surroundWord') {
          surroundRange(target, currentWordRange(target), plan.token)
        } else if (plan.kind === 'surroundLine') {
          surroundRange(target, currentLineRange(target), plan.token)
        } else if (plan.kind === 'deleteSurround') {
          deleteSurroundRange(target, plan.token)
        } else {
          changeSurroundRange(target, plan.oldToken, plan.replacementToken)
        }
        return true
      },
    }),
  )
  const initialVimExtension = await vimExtension(initialVimMode)

  let suppressChangeNotification = false
  const updateListener = EditorView.updateListener.of(update => {
    const userDocChanged = update.docChanged && !suppressChangeNotification
    if (update.docChanged) {
      if (userDocChanged) config.onEdited?.()
      if (config.onChange && userDocChanged) config.onChange(update.state.doc.toString())
    }
    const leapChanged = update.transactions.some(transaction =>
      transaction.effects.some(effect => effect.is(leapHighlightEffect)),
    )
    if (update.docChanged || update.selectionSet || update.focusChanged || leapChanged) {
      updateStatusElement(update.state)
    }
  })

  const transparentTheme = EditorView.theme({
    '&': {
      backgroundColor: 'transparent !important',
      border: 'none !important',
      outline: 'none !important',
      fontSize: 'inherit',
      fontFamily: 'inherit !important',
      lineHeight: 'inherit',
      color: 'inherit',
      height: 'auto',
      padding: '0',
      position: 'relative',
    },
    '&.cm-focused': {
      outline: 'none !important',
      border: 'none !important',
      boxShadow: 'none !important',
    },
    '.cm-content': {
      padding: '0 0 1.15rem 0 !important',
      minHeight: '20px',
      caretColor: 'inherit',
    },
    '.cm-gutter': { minHeight: '20px' },
    '.cm-scroller': { overflow: 'visible', fontFamily: 'inherit !important', fontSize: 'inherit' },
    '.cm-line': { padding: '0' },
    '.cm-cursor': { borderLeftColor: 'inherit' },
    '.cm-notebookLeapBackdrop': {
      backgroundColor: 'transparent',
      color: 'var(--code-comment, var(--gray)) !important',
      fontStyle: 'italic',
      transitionProperty: 'color, background-color',
      transitionDuration: '120ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookLeapMatch': {
      borderRadius: '2px',
      backgroundColor: 'color-mix(in srgb, var(--notebook-active-green) 22%, transparent)',
      color: 'var(--dark) !important',
      fontWeight: '800',
      textDecorationLine: 'none',
      transitionProperty: 'color, background-color',
      transitionDuration: '80ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookStatus': {
      position: 'absolute',
      right: '4px',
      bottom: '1px',
      zIndex: '4',
      maxWidth: 'min(60%, 12rem)',
      overflow: 'hidden',
      padding: '1px 5px',
      borderRadius: '4px',
      backgroundColor: 'color-mix(in srgb, var(--light) 82%, transparent)',
      color: 'color-mix(in srgb, var(--gray) 78%, var(--dark) 22%)',
      fontSize: '0.72em',
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: '0',
      lineHeight: '1.25',
      opacity: '0.72',
      pointerEvents: 'none',
      textAlign: 'right',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      transitionProperty: 'color, background-color, opacity',
      transitionDuration: '120ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookStatus[data-active="true"]': {
      backgroundColor: 'color-mix(in srgb, var(--notebook-active-green) 18%, var(--light))',
      color: 'var(--notebook-active-green)',
      opacity: '1',
    },
  })

  const state = EditorState.create({
    doc: config.initialContent ?? '',
    extensions: [
      lineNumbers(),
      language,
      lsp,
      history(),
      leapHighlightField,
      vimCompartment.of(initialVimExtension),
      leapKeyHandler,
      surroundKeyHandler,
      completionKeymap,
      editorSnippetKeymap,
      keymapCompartment.of(notebookKeymap()),
      editorKeymapCompartment.of(editorKeymap()),
      updateListener,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      transparentTheme,
    ],
  })

  const view = new EditorView({ state, parent: config.parent })
  view.dom.append(statusElement)
  updateStatusElement(view.state)
  const bindVimSave = async () => {
    if (!config.onSave) return
    const { getCM } = await loadVimApi()
    const cm: CodeMirror | null = getCM(view)
    if (cm) cm.save = config.onSave
  }
  if (initialVimMode) await bindVimSave()

  return {
    getValue: () => view.state.doc.toString(),
    setValue(content: string) {
      if (view.state.doc.length === content.length && view.state.doc.sliceString(0) === content)
        return
      suppressChangeNotification = true
      try {
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: content } })
      } finally {
        suppressChangeNotification = false
      }
    },
    highlightedLines: () => highlightedLineSpans(view),
    async setVimMode(enabled: boolean) {
      clearSurroundPending()
      clearLeap()
      vimModeEnabled = enabled
      view.dispatch({
        effects: [
          vimCompartment.reconfigure(await vimExtension(enabled)),
          keymapCompartment.reconfigure(notebookKeymap()),
          editorKeymapCompartment.reconfigure(editorKeymap()),
        ],
      })
      if (enabled) await bindVimSave()
    },
    focus: () => view.focus(),
    destroy() {
      clearSurroundPending()
      clearLeapTimer()
      view.destroy()
    },
  }
}
