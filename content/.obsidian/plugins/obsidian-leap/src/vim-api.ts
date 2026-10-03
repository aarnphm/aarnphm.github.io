import { EditorView } from '@codemirror/view'

export interface VimPosition {
  line: number
  ch: number
}

export interface VimMotionArgs {
  forward?: boolean
  inclusive?: boolean
  linewise?: boolean
  repeat?: number
  selectedCharacter?: string
  gardenLeapAncestor?: number
}

export interface VimInputState {
  operator?: string | null
  keyBuffer: string[]
}

export interface VimState {
  insertMode: boolean
  visualMode: boolean
  visualLine: boolean
  visualBlock: boolean
  sel: { anchor?: VimPosition; head?: VimPosition }
  inputState: unknown
  lastEditInputState?: unknown
  lastEditActionCommand?: unknown
}

export interface VimInsertModeChanges {
  changes: unknown[]
  expectCursorActivityForChange: boolean
  visualBlock?: number
  [key: string]: unknown
}

export interface VimMacroState {
  lastInsertModeChanges: VimInsertModeChanges
}

export interface VimRepeatRecord {
  input: unknown
  action: unknown
  macro?: VimMacroState
  buffer?: VimInsertModeChanges
  insertion?: VimInsertModeChanges
}

export interface VimEditor {
  cm6: unknown
  state: { vim?: unknown }
  indexFromPos(position: VimPosition): number
  posFromIndex(index: number): VimPosition
  signal(type: string, event: { mode: string; subMode?: string }): void
}

export type VimMotion = (
  editor: unknown,
  head: unknown,
  args: unknown,
  state: unknown,
  input: unknown,
) => VimPosition | [VimPosition, VimPosition] | null

export interface VimMapping {
  keys: string
  type: 'motion'
  motion: string
  motionArgs: VimMotionArgs
  context: string
}

export interface VimApi {
  defineMotion(name: string, motion: VimMotion): void
  _mapCommand(mapping: VimMapping): void
  unmap(keys: string, context: string): unknown
  handleKey(editor: VimEditor, keys: string, origin?: string): unknown
  getVimGlobalState_(): unknown
}

function isObject(value: unknown): value is object {
  return (typeof value === 'object' && value !== null) || typeof value === 'function'
}

export function isPosition(value: unknown): value is VimPosition {
  return (
    isObject(value) &&
    'line' in value &&
    typeof value.line === 'number' &&
    'ch' in value &&
    typeof value.ch === 'number'
  )
}

export function isMotionArgs(value: unknown): value is VimMotionArgs {
  return (
    isObject(value) &&
    (!('selectedCharacter' in value) || typeof value.selectedCharacter === 'string') &&
    (!('repeat' in value) || typeof value.repeat === 'number')
  )
}

export function isInputState(value: unknown): value is VimInputState {
  return (
    isObject(value) &&
    'keyBuffer' in value &&
    Array.isArray(value.keyBuffer) &&
    value.keyBuffer.every(key => typeof key === 'string') &&
    (!('operator' in value) || value.operator === null || typeof value.operator === 'string')
  )
}

export function isVimState(value: unknown): value is VimState {
  return (
    isObject(value) &&
    'insertMode' in value &&
    typeof value.insertMode === 'boolean' &&
    'visualMode' in value &&
    typeof value.visualMode === 'boolean' &&
    'visualLine' in value &&
    typeof value.visualLine === 'boolean' &&
    'visualBlock' in value &&
    typeof value.visualBlock === 'boolean' &&
    'sel' in value &&
    isObject(value.sel) &&
    'inputState' in value
  )
}

export function isVimEditor(value: unknown): value is VimEditor {
  return (
    isObject(value) &&
    'cm6' in value &&
    'state' in value &&
    isObject(value.state) &&
    'indexFromPos' in value &&
    typeof value.indexFromPos === 'function' &&
    'posFromIndex' in value &&
    typeof value.posFromIndex === 'function' &&
    'signal' in value &&
    typeof value.signal === 'function'
  )
}

export function nativeEditor(view: EditorView): VimEditor | undefined {
  if (!('cm' in view) || !isVimEditor(view.cm)) return
  const nativeView: unknown = view.cm.cm6
  return nativeView instanceof EditorView && nativeView === view ? view.cm : undefined
}

function isVimApi(api: unknown): api is VimApi {
  return (
    isObject(api) &&
    'defineMotion' in api &&
    typeof api.defineMotion === 'function' &&
    '_mapCommand' in api &&
    typeof api._mapCommand === 'function' &&
    'unmap' in api &&
    typeof api.unmap === 'function' &&
    'handleKey' in api &&
    typeof api.handleKey === 'function' &&
    'getVimGlobalState_' in api &&
    typeof api.getVimGlobalState_ === 'function'
  )
}

export function nativeVim(window: Window | null): VimApi | undefined {
  if (!window || !('CodeMirrorAdapter' in window)) return
  const adapter: unknown = window.CodeMirrorAdapter
  if (!isObject(adapter) || !('Vim' in adapter)) return
  return isVimApi(adapter.Vim) ? adapter.Vim : undefined
}

function isInsertModeChanges(value: unknown): value is VimInsertModeChanges {
  return (
    isObject(value) &&
    'changes' in value &&
    Array.isArray(value.changes) &&
    'expectCursorActivityForChange' in value &&
    typeof value.expectCursorActivityForChange === 'boolean' &&
    (!('visualBlock' in value) ||
      value.visualBlock === undefined ||
      typeof value.visualBlock === 'number')
  )
}

function isMacroState(value: unknown): value is VimMacroState {
  return (
    isObject(value) &&
    'lastInsertModeChanges' in value &&
    isInsertModeChanges(value.lastInsertModeChanges)
  )
}

export function captureRepeat(state: VimState, api?: VimApi): VimRepeatRecord {
  const record: VimRepeatRecord = {
    input: state.lastEditInputState,
    action: state.lastEditActionCommand,
  }
  const global: unknown = api?.getVimGlobalState_()
  if (!isObject(global) || !('macroModeState' in global) || !isMacroState(global.macroModeState))
    return record
  const macro = global.macroModeState
  const buffer = macro.lastInsertModeChanges
  record.macro = macro
  record.buffer = buffer
  record.insertion = { ...buffer, changes: buffer.changes.slice() }
  return record
}

export function restoreRepeat(state: VimState, record: VimRepeatRecord): void {
  state.lastEditInputState = record.input
  state.lastEditActionCommand = record.action
  const { macro, buffer, insertion } = record
  if (!macro || !buffer || !insertion || macro.lastInsertModeChanges !== buffer) return
  for (const key of Object.keys(buffer)) {
    if (!(key in insertion)) delete buffer[key]
  }
  Object.assign(buffer, insertion, { changes: insertion.changes.slice() })
}
