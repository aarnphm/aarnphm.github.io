import type { EditorState as CodeMirrorEditorState, Extension } from '@codemirror/state'
import type { EditorView as CodeMirrorEditorView } from '@codemirror/view'
import type { CodeMirror } from '@replit/codemirror-vim'
import { completionTheme } from '../../../util/codemirror-completion-theme'
import { lspExtensions, notebookCodeEditorLanguageExtension } from './highlighting'
import {
  changeSurroundRange,
  currentLineRange,
  currentSelectionRange,
  currentWordRange,
  deleteSurroundRange,
  notebookSurroundKeyPlan,
  surroundRange,
} from './surround'
import {
  createLeapHighlight,
  createTransparentTheme,
  createWhitespaceExtension,
  type NotebookCodeEditor,
  type NotebookCodeEditorConfig,
} from './view-extensions'
import {
  configureNotebookVimBindings,
  notebookEditorStatusTextInText,
  notebookKeyBindingUsesVimControl,
  notebookLeapMotionForKey,
  notebookLeapPendingDelayMs,
  notebookLeapTargetsInText,
  notebookLeapTraversalDelayMs,
  notebookRunKeys,
  type NotebookLeapMotion,
  type NotebookLeapSearchRange,
  type NotebookLeapSession,
  type NotebookLeapStatus,
  type VimApi,
} from './vim-navigation'

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
    { Compartment, EditorState, Prec, RangeSetBuilder, StateEffect, StateField },
    { Decoration, EditorView, ViewPlugin, WidgetType, keymap, lineNumbers },
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
  const { leapHighlightEffect, leapHighlightField } = createLeapHighlight({
    StateEffect,
    Decoration,
    StateField,
    EditorView,
  })
  let leapPendingMotion: NotebookLeapMotion | undefined
  let leapSession: NotebookLeapSession | undefined
  let leapClearTimer: number | undefined
  const statusElement = document.createElement('div')
  statusElement.className = 'cm-notebookStatus'
  const statusTextNode = document.createTextNode('')
  statusElement.append(statusTextNode)
  let fatCursorStabilized = false
  const stabilizeVimFatCursorText = (root: HTMLElement) => {
    if (fatCursorStabilized) return
    const cursor = root.querySelector<HTMLElement>('.cm-fat-cursor')
    if (!cursor) return
    fatCursorStabilized = true
    let textNode = cursor.firstChild
    if (!(textNode instanceof Text)) {
      textNode = document.createTextNode(cursor.textContent || ' ')
      cursor.replaceChildren(textNode)
    }
    const node = textNode
    Object.defineProperty(cursor, 'textContent', {
      configurable: true,
      get() {
        return node.nodeValue ?? ''
      },
      set(value: string | null) {
        const next = value ?? ''
        if (node.nodeValue !== next) node.nodeValue = next
      },
    })
  }
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
  let lastStatusText = ''
  let lastStatusActive = ''
  const updateStatusElement = (state: CodeMirrorEditorState) => {
    if (vimModeEnabled && !fatCursorStabilized && statusElement.parentElement) {
      stabilizeVimFatCursorText(statusElement.parentElement)
    }
    const leap = currentLeapStatus()
    const nextText = notebookEditorStatusTextInText(state.doc, state.selection.main.head, leap)
    if (nextText !== lastStatusText) {
      lastStatusText = nextText
      statusTextNode.nodeValue = nextText
    }
    const nextActive = leap ? 'true' : 'false'
    if (nextActive !== lastStatusActive) {
      lastStatusActive = nextActive
      statusElement.dataset.active = nextActive
    }
  }
  const clearLeapTimer = () => {
    if (leapClearTimer !== undefined) {
      window.clearTimeout(leapClearTimer)
      leapClearTimer = undefined
    }
  }
  const clearLeap = () => {
    const hadPending = leapPendingMotion !== undefined
    const hadSession = leapSession !== undefined
    leapPendingMotion = undefined
    leapSession = undefined
    clearLeapTimer()
    if (hadSession) {
      view.dispatch({ effects: leapHighlightEffect.of(undefined) })
    } else if (hadPending) {
      updateStatusElement(view.state)
    }
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
  const scheduleSurroundPendingFlush = () => {
    if (surroundPendingFlushTimer !== undefined) window.clearTimeout(surroundPendingFlushTimer)
    surroundPendingFlushTimer = window.setTimeout(() => clearSurroundPending(), 900)
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
          if (surroundPendingBuffer.length > 0) {
            clearSurroundPending()
            if (event.key === 'Escape') {
              event.preventDefault()
              event.stopPropagation()
              return true
            }
          }
          return false
        }
        const wasEmptyBuffer = surroundPendingBuffer === ''
        const pendingOperator =
          wasEmptyBuffer && !vimState.visualMode
            ? (vimState.inputState?.operator ?? undefined)
            : undefined
        const plan = notebookSurroundKeyPlan(
          surroundPendingBuffer,
          key,
          vimState.visualMode === true,
          pendingOperator,
        )
        if (plan.kind === 'pass') return false
        if (plan.kind === 'flush') {
          clearSurroundPending()
          event.preventDefault()
          event.stopPropagation()
          return true
        }

        event.preventDefault()
        event.stopPropagation()
        if (wasEmptyBuffer && pendingOperator) {
          vimApi.Vim.handleKey(cm, '<Esc>', 'mapping')
        }
        if (plan.kind === 'pending') {
          surroundPendingBuffer = plan.buffer
          scheduleSurroundPendingFlush()
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
  let lastSyncedContent: string | undefined = config.initialContent ?? ''
  const updateListener = EditorView.updateListener.of(update => {
    const userDocChanged = update.docChanged && !suppressChangeNotification
    if (update.docChanged) {
      lastSyncedContent = undefined
      if (userDocChanged) config.onEdited?.()
      if (config.onChange && userDocChanged) {
        const nextDoc = update.state.doc.toString()
        lastSyncedContent = nextDoc
        config.onChange(nextDoc)
      }
    }
    const leapChanged = update.transactions.some(transaction =>
      transaction.effects.some(effect => effect.is(leapHighlightEffect)),
    )
    if (update.docChanged || update.selectionSet || update.focusChanged || leapChanged) {
      updateStatusElement(update.state)
    }
  })
  const { whitespaceExtension } = createWhitespaceExtension({
    WidgetType,
    Decoration,
    RangeSetBuilder,
    ViewPlugin,
  })
  const { transparentTheme } = createTransparentTheme({ EditorView })

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
      whitespaceExtension,
      transparentTheme,
      completionTheme,
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
    getValue: () => {
      if (lastSyncedContent === undefined) {
        lastSyncedContent = view.state.doc.toString()
      }
      return lastSyncedContent
    },
    setValue(content: string) {
      if (lastSyncedContent === content) return
      if (view.state.doc.length === content.length && view.state.doc.sliceString(0) === content) {
        lastSyncedContent = content
        return
      }
      suppressChangeNotification = true
      try {
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: content } })
        lastSyncedContent = content
      } finally {
        suppressChangeNotification = false
      }
    },
    async setVimMode(enabled: boolean) {
      clearSurroundPending()
      clearLeap()
      vimModeEnabled = enabled
      fatCursorStabilized = false
      view.dispatch({
        effects: [
          vimCompartment.reconfigure(await vimExtension(enabled)),
          keymapCompartment.reconfigure(notebookKeymap()),
          editorKeymapCompartment.reconfigure(editorKeymap()),
        ],
      })
      if (enabled) {
        stabilizeVimFatCursorText(view.dom)
        await bindVimSave()
      }
    },
    focus: () => view.focus(),
    destroy() {
      clearSurroundPending()
      clearLeapTimer()
      view.destroy()
    },
  }
}
