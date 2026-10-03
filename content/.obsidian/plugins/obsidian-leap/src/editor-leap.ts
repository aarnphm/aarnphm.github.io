import { EditorSelection, type Extension } from '@codemirror/state'
import { EditorView, ViewPlugin, type ViewUpdate } from '@codemirror/view'
import { Component, editorInfoField, type App } from 'obsidian'
import type { CharacterMotion, LeapOptions, LeapTarget, SyntaxTarget } from './types'
import { characterLabelAlphabet, characterTargets } from './character-targets'
import { LeapOverlay, type OverlayTarget } from './overlay'
import { syntaxTargets } from './syntax-targets'
import {
  captureRepeat,
  isInputState,
  isMotionArgs,
  isPosition,
  isVimEditor,
  isVimState,
  nativeEditor,
  nativeVim,
  restoreRepeat,
  type VimApi,
  type VimEditor,
  type VimInputState,
  type VimMapping,
  type VimMotion,
  type VimMotionArgs,
  type VimPosition,
  type VimRepeatRecord,
  type VimState,
} from './vim-api'

interface QuerySession {
  kind: 'query'
  view: EditorView
  motion: CharacterMotion
}

interface CharacterSession {
  kind: 'character'
  view: EditorView
  motion: CharacterMotion
  targets: LeapTarget[]
  active: number
  native?: VimEditor
  awaitingNative: boolean
}

interface SyntaxSession {
  kind: 'syntax'
  view: EditorView
  linewise: boolean
  targets: SyntaxTarget[]
  active: number
  native?: VimEditor
  pending?: VimInputState
  previousEdit?: VimRepeatRecord
  restoreEdit?: boolean
  awaitingNative: boolean
}

type Session = QuerySession | CharacterSession | SyntaxSession

const motions: CharacterMotion[] = ['f', 'F', 't', 'T']
const traversalMotion = 'gardenLeapTraverse'
const traversalKeys = '<GardenLeapTarget>'
const syntaxMotion = 'gardenLeapSyntax'
const syntaxLabels = Array.from('asfnut/SFNLHMUGTZ?')
const linewiseSyntaxLabels = Array.from('sfnut/SFNLHMUGTZ?')
const operatorSyntaxLabels = Array.from('sfnjklhodweimbuyvrgtaqpcxz/SFNJKLHODWEIMBUYVRGTAQPCXZ?')
let disposalCounter = 0

function positionBefore(view: EditorView, position: number): number {
  const tail = Array.from(view.state.doc.sliceString(Math.max(0, position - 2), position)).at(-1)
  return position - (tail?.length ?? 0)
}

function clampIndex(index: number, length: number): number {
  return Math.max(0, Math.min(index, length - 1))
}

export class EditorLeap extends Component {
  readonly extension: Extension
  private views = new Set<EditorView>()
  private session?: Session
  private overlay: LeapOverlay
  private api?: VimApi
  private mappings: VimMapping[] = []
  private mutating = 0
  private traversalPosition?: VimPosition
  private syntaxChoice?: { target: SyntaxTarget; ancestor: number }
  private previous?: { character: string; motion: CharacterMotion }
  private lastEdits = new WeakMap<VimEditor, VimRepeatRecord>()

  constructor(
    private app: App,
    private options: () => LeapOptions,
  ) {
    super()
    this.overlay = new LeapOverlay(app.workspace.containerEl.ownerDocument)
    this.extension = [
      ViewPlugin.define(view => {
        this.views.add(view)
        this.rememberEdit(view)
        const onScroll = () => this.clear(view)
        const onBlur = () => this.clear(view)
        const onKeydown = (event: KeyboardEvent) => {
          if (this.keydown(event, view)) {
            event.preventDefault()
            event.stopImmediatePropagation()
          }
        }
        view.scrollDOM.addEventListener('scroll', onScroll)
        // Obsidian installs its highest-priority Vim handler before vault
        // extensions. Capture receives owned session keys before that handler.
        view.contentDOM.addEventListener('keydown', onKeydown, true)
        view.contentDOM.addEventListener('blur', onBlur, true)
        return {
          update: (update: ViewUpdate) => this.update(update),
          destroy: () => {
            this.clear(view)
            this.views.delete(view)
            view.scrollDOM.removeEventListener('scroll', onScroll)
            view.contentDOM.removeEventListener('keydown', onKeydown, true)
            view.contentDOM.removeEventListener('blur', onBlur, true)
          },
        }
      }),
    ]
  }

  onload(): void {
    this.refreshBindings()
    this.registerEvent(this.app.workspace.on('active-leaf-change', () => this.clear()))
    this.registerEvent(this.app.workspace.on('layout-change', () => this.clear()))
    const document = this.app.workspace.containerEl.ownerDocument
    this.registerDomEvent(document, 'pointerdown', () => this.clear(), true)
    if (document.defaultView)
      this.registerDomEvent(document.defaultView, 'resize', () => this.clear())
  }

  onunload(): void {
    this.clear()
    this.removeBindings()
    this.views.clear()
  }

  refreshBindings(): void {
    this.clear()
    this.removeBindings()
    if (!this.options().vimMotions) return
    this.api = nativeVim(this.app.workspace.containerEl.ownerDocument.defaultView)
    if (!this.api) return
    for (const motion of motions) {
      const name = `gardenLeapCharacter${motion}`
      this.api.defineMotion(
        name,
        this.guardMotion((view, editor, head, args, state, input) =>
          this.characterMotion(view, editor, head, args, state, input, motion),
        ),
      )
      this.map(`${motion}<character>`, name, {
        forward: motion === 'f' || motion === 't',
        inclusive: motion === 'f' || motion === 't',
      })
    }
    this.api.defineMotion(
      syntaxMotion,
      this.guardMotion((view, editor, head, args, state, input) =>
        this.syntaxMotion(view, editor, head, args, state, input),
      ),
    )
    this.map('ga', syntaxMotion, { inclusive: false, linewise: false })
    this.map('gA', syntaxMotion, { inclusive: false, linewise: true })
    this.api.defineMotion(traversalMotion, () => this.traversalPosition ?? null)
    this.map(traversalKeys, traversalMotion, { inclusive: true })
  }

  startCharacter(motion: CharacterMotion): boolean {
    const view = this.activeView()
    if (!view) return false
    this.clear()
    view.focus()
    this.session = { kind: 'query', view, motion }
    return true
  }

  startSyntax(linewise: boolean): boolean {
    const view = this.activeView()
    if (!view) return false
    this.clear()
    const targets = syntaxTargets(view, view.state.selection.main.head, linewise)
    if (!targets.length) return false
    view.focus()
    const editor = nativeEditor(view)
    if (this.api && editor && isVimState(editor.state.vim) && !editor.state.vim.insertMode) {
      this.api.handleKey(editor, linewise ? 'gA' : 'ga', 'garden-leap')
      return true
    }
    this.session = { kind: 'syntax', view, linewise, targets, active: 0, awaitingNative: false }
    this.selectSyntax(this.session)
    if (targets.length === 1) this.clear()
    return true
  }

  private activeView(): EditorView | undefined {
    const active = this.app.workspace.activeEditor?.editor
    for (const view of this.views) {
      if (active && view.state.field(editorInfoField, false)?.editor === active && view.inView)
        return view
    }
    return Array.from(this.views).find(view => view.hasFocus && view.inView)
  }

  private map(keys: string, motion: string, motionArgs: VimMotionArgs): void {
    if (!this.api) return
    for (const context of ['normal', 'visual', 'operatorPending']) {
      const mapping: VimMapping = { keys, type: 'motion', motion, motionArgs, context }
      this.mappings.push(mapping)
      this.api._mapCommand(mapping)
    }
  }

  private removeBindings(): void {
    if (!this.api) return
    const context = `garden-leap-disposed-${++disposalCounter}`
    for (const mapping of this.mappings) {
      // The retained object identifies our exact entry even if another plugin
      // subsequently installed a newer mapping for the same keys.
      mapping.context = context
      this.api.unmap(mapping.keys, context)
    }
    for (const name of [
      syntaxMotion,
      traversalMotion,
      ...motions.map(motion => `gardenLeapCharacter${motion}`),
    ]) {
      this.api.defineMotion(name, () => null)
    }
    this.mappings = []
    this.api = undefined
  }

  private guardMotion(
    motion: (
      view: EditorView,
      editor: VimEditor,
      head: VimPosition,
      args: VimMotionArgs,
      state: VimState,
      input: VimInputState,
    ) => ReturnType<VimMotion>,
  ): VimMotion {
    return (editor, head, args, state, input) => {
      if (
        !isVimEditor(editor) ||
        !isPosition(head) ||
        !isMotionArgs(args) ||
        !isVimState(state) ||
        !isInputState(input)
      )
        return null
      const view: unknown = editor.cm6
      if (!(view instanceof EditorView) || !this.views.has(view)) return null
      return motion(view, editor, head, args, state, input)
    }
  }

  private characterMotion(
    view: EditorView,
    editor: VimEditor,
    head: VimPosition,
    args: VimMotionArgs,
    _state: VimState,
    input: VimInputState,
    motion: CharacterMotion,
  ): VimPosition | null {
    this.clear()
    let character = args.selectedCharacter
    if (character === '\n' && this.previous) character = this.previous.character
    if (!character || Array.from(character).length !== 1) return null
    this.previous = { character, motion }
    const targets = characterTargets(
      view,
      editor.indexFromPos(head),
      character,
      motion === 'F' || motion === 'T',
      motion === 't' || motion === 'T',
    )
    const index = Math.max(1, args.repeat ?? 1) - 1
    const target = targets[index]
    if (!target) return null
    if (!input.operator && index === 0 && targets.length > 1) {
      this.session = {
        kind: 'character',
        view,
        motion,
        targets,
        active: 0,
        native: editor,
        awaitingNative: true,
      }
      this.afterNative(this.session)
    }
    return editor.posFromIndex(target.cursor)
  }

  private syntaxMotion(
    view: EditorView,
    editor: VimEditor,
    head: VimPosition,
    args: VimMotionArgs,
    state: VimState,
    input: VimInputState,
  ): [VimPosition, VimPosition] | null {
    const linewise = args.linewise === true
    const chosen = this.syntaxChoice
    this.syntaxChoice = undefined
    this.clear()
    const targets = syntaxTargets(view, editor.indexFromPos(head), linewise)
    const index = Math.max(1, args.gardenLeapAncestor ?? 1) - 1
    const target = chosen?.target ?? targets[index]
    if (!target) return null
    if (input.operator && !chosen && args.gardenLeapAncestor === undefined) {
      this.session = {
        kind: 'syntax',
        view,
        linewise,
        targets,
        active: 0,
        native: editor,
        pending: input,
        previousEdit: this.lastEdits.get(editor),
        restoreEdit: this.lastEdits.has(editor),
        awaitingNative: false,
      }
      this.show(this.session)
      return null
    }
    if (chosen) args.gardenLeapAncestor = chosen.ancestor
    if (!input.operator) {
      // Native evalInput captured this existing sel object before calling us.
      // Switching its flags preserves that reference and native Visual updates.
      state.visualMode = true
      state.visualLine = linewise
      state.visualBlock = false
      editor.signal('vim-mode-change', { mode: 'visual', subMode: linewise ? 'linewise' : '' })
      if (index === 0 && !chosen && targets.length > 1) {
        this.session = {
          kind: 'syntax',
          view,
          linewise,
          targets,
          active: 0,
          native: editor,
          awaitingNative: true,
        }
        this.afterNative(this.session)
      }
    }
    const end = input.operator && !linewise ? target.to : positionBefore(view, target.to)
    if (!input.operator && !linewise) this.normalizeSyntaxSelection(view, editor, target)
    return [editor.posFromIndex(target.from), editor.posFromIndex(end)]
  }

  private normalizeSyntaxSelection(
    view: EditorView,
    editor: VimEditor,
    target: SyntaxTarget,
  ): void {
    const doc = view.state.doc
    const finalCharacter = doc.sliceString(Math.max(target.from, target.to - 2), target.to)
    const codePoint = finalCharacter.codePointAt(0)
    if (codePoint === undefined || codePoint <= 0xffff) return
    queueMicrotask(() => {
      const state = editor.state.vim
      const selection = view.state.selection.main
      if (
        !this.views.has(view) ||
        view.state.doc !== doc ||
        !isVimState(state) ||
        !state.visualMode ||
        state.visualLine ||
        selection.from !== target.from ||
        selection.to !== target.to - 1
      )
        return
      // The native Visual engine adds one UTF-16 unit to its inclusive head.
      // Complete an astral final character and retain the native Visual range.
      this.mutating++
      try {
        view.dispatch({ selection: EditorSelection.range(target.from, target.to) })
        state.sel.anchor = editor.posFromIndex(target.from)
        state.sel.head = editor.posFromIndex(target.to - 1)
      } finally {
        this.mutating--
      }
    })
  }

  private afterNative(session: CharacterSession | SyntaxSession): void {
    queueMicrotask(() => {
      if (this.session !== session) return
      session.awaitingNative = false
      this.show(session)
    })
  }

  private update(update: ViewUpdate): void {
    const session = this.session
    if (!(session?.kind === 'syntax' && session.pending)) this.rememberEdit(update.view)
    if (
      !session ||
      session.view !== update.view ||
      this.mutating ||
      (session.kind !== 'query' && session.awaitingNative)
    )
      return
    if (
      update.docChanged ||
      update.selectionSet ||
      update.viewportChanged ||
      update.geometryChanged
    )
      this.clear(update.view)
  }

  private clear(view?: EditorView): void {
    if (view && this.session?.view !== view) return
    const session = this.session
    if (
      session?.kind === 'syntax' &&
      session.pending &&
      session.restoreEdit &&
      session.previousEdit
    ) {
      const state = session.native?.state.vim
      if (isVimState(state) && state.lastEditInputState === session.pending) {
        restoreRepeat(state, session.previousEdit)
      }
    }
    this.session = undefined
    this.overlay.clear()
  }

  private keydown(event: KeyboardEvent, view: EditorView): boolean {
    if (event.key === 'Shift') return false
    const session = this.session
    if (!(session?.kind === 'syntax' && session.pending)) this.rememberEdit(view)
    if (!session || session.view !== view) return false
    if (event.isComposing) {
      this.clear()
      return false
    }
    if (event.key === 'Escape') {
      this.clear()
      return true
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      this.clear()
      return false
    }
    if (session.kind === 'query') {
      if (event.key === 'Enter' && this.previous) {
        this.queryCharacter(session, this.previous.character)
        return true
      }
      if (Array.from(event.key).length === 1) {
        this.queryCharacter(session, event.key)
        return true
      }
      this.clear()
      return false
    }
    const forward = session.kind === 'character' ? session.motion.toLowerCase() : 'a'
    const backward = forward.toUpperCase()
    if (session.kind === 'syntax' && session.pending) {
      if (event.key === 'Enter' || (!session.linewise && event.key === 'a')) {
        this.commitSyntax(session, 0)
        return true
      }
      const index = this.labels(session).findIndex(label => label === event.key)
      if (index >= 0) {
        this.commitSyntax(session, index)
        return true
      }
      this.clear()
      return false
    }
    if (
      event.key === 'Enter' ||
      (session.kind === 'character' && event.key === ' ') ||
      (event.key === forward && !(session.kind === 'syntax' && session.linewise))
    ) {
      this.traverse(session, 1)
      return true
    }
    if (
      event.key === 'Backspace' ||
      (event.key === backward && !(session.kind === 'syntax' && session.linewise))
    ) {
      this.traverse(session, -1)
      return true
    }
    const labels = this.labels(session)
    const index = labels.findIndex(label => label === event.key)
    if (index >= 0) {
      session.active = index
      this.move(session)
      this.clear()
      return true
    }
    this.clear()
    return false
  }

  private queryCharacter(session: QuerySession, character: string): void {
    const { view, motion } = session
    const editor = nativeEditor(view)
    if (
      this.api &&
      editor &&
      isVimState(editor.state.vim) &&
      !editor.state.vim.insertMode &&
      character.length === 1
    ) {
      this.clear()
      this.api.handleKey(
        editor,
        `${motion}${character === '\n' ? '<CR>' : character}`,
        'garden-leap',
      )
      return
    }
    const targets = characterTargets(
      view,
      view.state.selection.main.head,
      character,
      motion === 'F' || motion === 'T',
      motion === 't' || motion === 'T',
    )
    this.previous = { character, motion }
    this.clear()
    if (!targets.length) return
    const next: CharacterSession = {
      kind: 'character',
      view,
      motion,
      targets,
      active: 0,
      awaitingNative: false,
    }
    this.session = next
    this.move(next)
    if (targets.length === 1) this.clear()
  }

  private traverse(session: CharacterSession | SyntaxSession, direction: number): void {
    session.active =
      session.active === 0 && direction < 0
        ? session.targets.length - 1
        : clampIndex(session.active + direction, session.targets.length)
    this.move(session)
  }

  private move(session: CharacterSession | SyntaxSession): void {
    if (session.kind === 'syntax') {
      this.selectSyntax(session)
      return
    }
    const target = session.targets[session.active]
    if (!target) return
    this.mutating++
    try {
      if (session.native && this.api) {
        this.traversalPosition = session.native.posFromIndex(target.cursor)
        this.api.handleKey(session.native, traversalKeys, 'garden-leap')
        this.traversalPosition = undefined
      } else {
        session.view.dispatch({
          selection: EditorSelection.cursor(target.cursor),
          scrollIntoView: true,
        })
      }
    } finally {
      this.mutating--
    }
    this.show(session)
  }

  private selectSyntax(session: SyntaxSession): void {
    const target = session.targets[session.active]
    if (!target) return
    if (session.pending) {
      this.show(session)
      return
    }
    this.mutating++
    try {
      if (session.native && this.api) {
        this.syntaxChoice = { target, ancestor: session.active + 1 }
        const current = this.session
        this.api.handleKey(session.native, session.linewise ? 'gA' : 'ga', 'garden-leap')
        this.session = current
      } else {
        session.view.dispatch({
          selection: EditorSelection.range(target.from, target.to),
          scrollIntoView: true,
        })
      }
    } finally {
      this.mutating--
    }
    this.show(session)
  }

  private commitSyntax(session: SyntaxSession, index: number): void {
    const target = session.targets[index]
    const editor = session.native
    const state = editor?.state.vim
    if (!target || !editor || !this.api || !isVimState(state) || !session.pending) return
    this.mutating++
    try {
      session.restoreEdit = false
      session.pending.keyBuffer = []
      state.inputState = session.pending
      this.syntaxChoice = { target, ancestor: index + 1 }
      this.api.handleKey(editor, session.linewise ? 'gA' : 'ga', 'garden-leap')
    } finally {
      this.mutating--
      this.clear()
    }
  }

  private rememberEdit(view: EditorView): void {
    const editor = nativeEditor(view)
    const state = editor?.state.vim
    if (editor && isVimState(state)) this.lastEdits.set(editor, captureRepeat(state, this.api))
  }

  private labels(
    session: CharacterSession | SyntaxSession,
    visible = false,
  ): Array<string | undefined> {
    if (visible && !this.options().showLabels) return []
    const alphabet =
      session.kind === 'character'
        ? characterLabelAlphabet(session.motion)
        : session.pending
          ? operatorSyntaxLabels
          : session.linewise
            ? linewiseSyntaxLabels
            : syntaxLabels
    const pending = session.kind === 'syntax' && session.pending
    return session.targets.map((_target, targetIndex) => {
      if (pending) return alphabet[targetIndex]
      if (targetIndex === 0 || (visible && targetIndex <= session.active)) return undefined
      return alphabet[targetIndex - 1]
    })
  }

  private show(session: CharacterSession | SyntaxSession): void {
    const labels = this.labels(session, true)
    const markers: OverlayTarget[] = []
    for (let index = 0; index < session.targets.length; index++) {
      const target = session.targets[index]
      const rect =
        session.kind === 'character'
          ? (session.view.coordsForChar(target.from) ?? session.view.coordsAtPos(target.from))
          : session.view.coordsAtPos(target.from)
      if (!rect) continue
      const clip = session.view.scrollDOM.getBoundingClientRect()
      if (rect.bottom <= clip.top || rect.top >= clip.bottom) continue
      markers.push({
        rect: new DOMRect(
          rect.left,
          rect.top,
          Math.max(rect.right - rect.left, 2),
          rect.bottom - rect.top,
        ),
        label: labels[index],
        active: index === session.active,
      })
    }
    this.overlay.show(markers)
  }
}
