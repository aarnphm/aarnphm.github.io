import type { Text } from '@codemirror/state'
import type { KeyBinding } from '@codemirror/view'
import {
  changeSurroundRange,
  currentLineRange,
  currentSelectionRange,
  currentWordRange,
  deleteSurroundRange,
  surroundRange,
} from './surround'

export type VimApi = typeof import('@replit/codemirror-vim')
type NotebookCodeMirror = NonNullable<ReturnType<VimApi['getCM']>>
type NotebookVimModule = Pick<VimApi, 'Vim'>
export type NotebookVimBindingsApi = Pick<
  NotebookVimModule['Vim'],
  'defineAction' | 'mapCommand' | '_mapCommand'
>

type NotebookSurroundActionArgs = {
  token?: string
  selectedCharacter?: string
  oldToken?: string
  replacementToken?: string
}

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
export type NotebookLeapSession = {
  motion: NotebookLeapMotion
  character: string
  targets: readonly NotebookLeapTarget[]
  activeIndex: number
  visualMode: boolean
  anchor: number
}

let notebookVimBindingsConfigured = false

export const notebookRunKeys = ['Ctrl-Enter', 'Shift-Enter', 'Alt-Enter'] as const

const notebookVimReservedControlKeys = new Set([
  'Ctrl-b',
  'Ctrl-d',
  'Ctrl-e',
  'Ctrl-f',
  'Ctrl-u',
  'Ctrl-y',
])

export const notebookLeapPendingDelayMs = 900
export const notebookLeapTraversalDelayMs = 2800
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

export function notebookKeyBindingUsesVimControl(binding: KeyBinding): boolean {
  return [binding.key, binding.mac, binding.win, binding.linux].some(
    key => key !== undefined && notebookVimReservedControlKeys.has(key),
  )
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

function notebookSurroundToken(args: NotebookSurroundActionArgs) {
  return args.selectedCharacter ?? args.token
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
}

export function configureNotebookVimBindings(vimApi: NotebookVimModule) {
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
