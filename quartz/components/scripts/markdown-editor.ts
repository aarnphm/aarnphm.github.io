import { autocompletion, completionStatus, moveCompletionSelection } from '@codemirror/autocomplete'
import {
  defaultKeymap,
  historyKeymap,
  history,
  indentLess,
  indentMore,
  indentWithTab,
} from '@codemirror/commands'
import { markdown } from '@codemirror/lang-markdown'
import { syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language'
import { EditorState, Prec, type Extension } from '@codemirror/state'
import { EditorView, keymap, lineNumbers, placeholder } from '@codemirror/view'
import TurndownService from 'turndown'
import { completionTheme } from '../../util/codemirror-completion-theme'
import { codemirrorCodeLanguage } from '../../util/codemirror-language'
import { codemirrorChangedTextIsBlank, codemirrorTextIsBlank } from '../../util/codemirror-text'
import { completionSources } from '../multiplayer/completions'
import { togglePreview, cleanupPreview, onEditorUpdate } from '../multiplayer/completions/preview'

const turndown = new TurndownService({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  bulletListMarker: '-',
})

const MARKDOWN_WRAP_PAIRS: Readonly<Record<string, readonly [string, string]>> = {
  '*': ['*', '*'],
  _: ['_', '_'],
  '`': ['`', '`'],
  $: ['$', '$'],
  '~': ['~', '~'],
  '(': ['(', ')'],
  '[': ['[', ']'],
}

const LIST_LINE_RE = /^\s*(?:[-*+]|\d+[.)])\s/

function toggleMarkdownInlineMarker(view: EditorView, marker: string): boolean {
  const { from, to } = view.state.selection.main
  const len = marker.length
  const doc = view.state.doc
  if (
    from >= len &&
    to + len <= doc.length &&
    doc.sliceString(from - len, from) === marker &&
    doc.sliceString(to, to + len) === marker
  ) {
    view.dispatch({
      changes: [
        { from: from - len, to: from },
        { from: to, to: to + len },
      ],
      selection: { anchor: from - len, head: to - len },
    })
    return true
  }
  const selected = doc.sliceString(from, to)
  if (selected.length >= len * 2 && selected.startsWith(marker) && selected.endsWith(marker)) {
    const inner = selected.slice(len, selected.length - len)
    view.dispatch({
      changes: { from, to, insert: inner },
      selection: { anchor: from, head: from + inner.length },
    })
    return true
  }
  view.dispatch({
    changes: { from, to, insert: `${marker}${selected}${marker}` },
    selection: { anchor: from + len, head: to + len },
  })
  return true
}

const markdownSelectionWrapHandler = EditorView.inputHandler.of((view, from, to, text) => {
  if (from === to) return false
  const pair = MARKDOWN_WRAP_PAIRS[text]
  if (!pair) return false
  const [open, close] = pair
  view.dispatch({
    changes: [
      { from, insert: open },
      { from: to, insert: close },
    ],
    selection: { anchor: from + open.length, head: to + open.length },
  })
  return true
})

const markdownErgonomicsKeymap = keymap.of([
  {
    key: 'Tab',
    run: view => {
      const line = view.state.doc.lineAt(view.state.selection.main.from)
      if (!LIST_LINE_RE.test(line.text)) return false
      return indentMore(view)
    },
    shift: view => {
      const line = view.state.doc.lineAt(view.state.selection.main.from)
      if (!LIST_LINE_RE.test(line.text)) return false
      return indentLess(view)
    },
  },
  { key: 'Mod-b', run: view => toggleMarkdownInlineMarker(view, '**') },
  { key: 'Mod-i', run: view => toggleMarkdownInlineMarker(view, '*') },
  { key: 'Mod-e', run: view => toggleMarkdownInlineMarker(view, '`') },
  { key: 'Mod-Shift-m', run: view => toggleMarkdownInlineMarker(view, '$') },
])

const sharedTransparentTheme = EditorView.theme({
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
  },
  '&.cm-focused': {
    outline: 'none !important',
    border: 'none !important',
    boxShadow: 'none !important',
  },
  '.cm-content': { padding: '0 !important', minHeight: '20px', caretColor: 'inherit' },
  '.cm-gutter': { minHeight: '20px' },
  '.cm-scroller': { overflow: 'visible', fontFamily: 'inherit !important', fontSize: 'inherit' },
  '.cm-line': { padding: '0' },
  '.cm-cursor': { borderLeftColor: 'inherit' },
  '.cm-tooltip-autocomplete': {
    backgroundColor: 'var(--light)',
    border: '1px solid var(--lightgray)',
    borderRadius: '12px',
    boxShadow: '0 12px 32px rgba(0, 0, 0, 0.12)',
    padding: '8px 0',
    fontFamily: 'var(--bodyFont)',
    fontSize: '13px',
    width: '240px',
  },
  '.cm-completionLabel': {
    fontWeight: '600',
    color: 'var(--dark)',
    overflow: 'hidden !important',
    whiteSpace: 'nowrap !important',
    textOverflow: 'ellipsis !important',
  },
  '.cm-completionDetail': {
    fontSize: '12px',
    color: 'var(--gray)',
    fontStyle: 'normal',
    overflow: 'hidden !important',
    whiteSpace: 'nowrap !important',
    textOverflow: 'ellipsis !important',
  },
  '.cm-completionIcon': { display: 'none' },
  "li[role='option']": {
    cursor: 'pointer',
    borderRadius: '6px',
    margin: '1px 6px',
    justifyContent: 'center',
    alignItems: 'center',
    display: 'grid',
    gridTemplateColumns: '1fr 160px',
  },
  "li[role='option'][aria-selected]": { background: 'var(--foam) !important' },
})

export interface MarkdownEditorConfig {
  parent: HTMLElement
  initialContent?: string
  onEdited?: (state: { empty: boolean }) => void
  onChange?: (content: string) => void
  onSubmit?: () => void
  onCancel?: () => void
  mode?: 'markdown' | 'code'
  language?: string
  lineWrapping?: boolean
  ariaLabel?: string
  placeholder?: string
  maxLength?: number
  changeDebounceMs?: number
}

export class MarkdownEditor {
  private view: EditorView
  private empty: boolean
  private lastSyncedContent: string | undefined
  private suppressChangeNotification = false
  private changeTimer: number | undefined
  private pendingChangeNotification = false
  private readonly onChangeCallback?: (content: string) => void
  private readonly onPageHide: () => void

  constructor(config: MarkdownEditorConfig) {
    const mode = config.mode ?? 'markdown'
    this.lastSyncedContent = config.initialContent || ''
    this.onChangeCallback = config.onChange
    const debounceMs = config.changeDebounceMs ?? 0
    this.onPageHide = () => this.flush()
    window.addEventListener('pagehide', this.onPageHide)

    const customKeymap = Prec.highest(
      keymap.of([
        {
          key: 'Mod-Enter',
          run: () => {
            this.flush()
            config.onSubmit?.()
            return true
          },
        },
        {
          key: 'Escape',
          run: () => {
            this.flush()
            config.onCancel?.()
            return true
          },
        },
        {
          key: 'Ctrl-n',
          run: view => {
            if (completionStatus(view.state) === 'active') {
              moveCompletionSelection(true)(view)
              return true
            }
            return false
          },
        },
        {
          key: 'Ctrl-p',
          run: view => {
            if (completionStatus(view.state) === 'active') {
              moveCompletionSelection(false)(view)
              return true
            }
            return false
          },
        },
        {
          key: 'Ctrl-d',
          run: view => {
            if (completionStatus(view.state) === 'active') {
              return togglePreview(view)
            }
            return false
          },
        },
      ]),
    )

    const updateListener = EditorView.updateListener.of(update => {
      if (update.docChanged) {
        this.lastSyncedContent = undefined
        this.empty = codemirrorChangedTextIsBlank(this.empty, update.state.doc, update.changes)
        if (!this.suppressChangeNotification) {
          config.onEdited?.({ empty: this.empty })
          if (config.onChange) {
            if (debounceMs > 0) {
              this.pendingChangeNotification = true
              if (this.changeTimer !== undefined) window.clearTimeout(this.changeTimer)
              this.changeTimer = window.setTimeout(() => this.flush(), debounceMs)
            } else {
              this.pendingChangeNotification = false
              const next = update.state.doc.toString()
              this.lastSyncedContent = next
              config.onChange(next)
            }
          }
        }
      }
      if (update.focusChanged && !update.view.hasFocus) {
        this.flush()
      }
      if (mode === 'markdown') onEditorUpdate(update)
    })

    const extensions: Extension[] = [
      history(),
      customKeymap,
      ...(mode === 'markdown' ? [markdownErgonomicsKeymap, markdownSelectionWrapHandler] : []),
      keymap.of(
        mode === 'code'
          ? [indentWithTab, ...defaultKeymap, ...historyKeymap]
          : [...defaultKeymap, ...historyKeymap],
      ),
      updateListener,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      sharedTransparentTheme,
    ]
    if (config.ariaLabel) {
      extensions.push(EditorView.contentAttributes.of({ 'aria-label': config.ariaLabel }))
    }
    if (config.placeholder) extensions.push(placeholder(config.placeholder))
    if (config.maxLength !== undefined) {
      const limit = config.maxLength
      extensions.push(
        EditorState.changeFilter.of(transaction => transaction.newDoc.length <= limit),
      )
    }

    if (config.lineWrapping ?? mode === 'markdown') {
      extensions.push(EditorView.lineWrapping)
    }

    if (mode === 'markdown') {
      extensions.unshift(markdown())
      extensions.push(
        autocompletion({ override: completionSources, closeOnBlur: false, activateOnTyping: true }),
        completionTheme,
      )
      extensions.push(
        EditorView.domEventHandlers({
          paste(event, view) {
            const html = event.clipboardData?.getData('text/html')
            if (!html) return false

            if (
              !html.includes('<p>') &&
              !html.includes('<div>') &&
              !html.includes('<h') &&
              !html.includes('<li>')
            ) {
              return false
            }

            event.preventDefault()
            const md = turndown.turndown(html)

            view.dispatch({
              changes: {
                from: view.state.selection.main.from,
                to: view.state.selection.main.to,
                insert: md,
              },
            })
            return true
          },
        }),
      )
    } else {
      extensions.unshift(lineNumbers(), codemirrorCodeLanguage(config.language))
    }

    const state = EditorState.create({ doc: config.initialContent || '', extensions })
    this.empty = codemirrorTextIsBlank(state.doc)

    this.view = new EditorView({ state, parent: config.parent })
  }

  flush(): void {
    if (this.changeTimer !== undefined) {
      window.clearTimeout(this.changeTimer)
      this.changeTimer = undefined
    }
    if (!this.pendingChangeNotification || !this.onChangeCallback) return
    this.pendingChangeNotification = false
    const next = this.getValue()
    this.onChangeCallback(next)
  }

  getValue(): string {
    if (this.lastSyncedContent === undefined) {
      this.lastSyncedContent = this.view.state.doc.toString()
    }
    return this.lastSyncedContent
  }

  setValue(content: string): void {
    if (this.lastSyncedContent === content) return
    if (
      this.view.state.doc.length === content.length &&
      this.view.state.doc.sliceString(0) === content
    ) {
      this.lastSyncedContent = content
      return
    }
    if (this.changeTimer !== undefined) {
      window.clearTimeout(this.changeTimer)
      this.changeTimer = undefined
    }
    this.pendingChangeNotification = false
    this.suppressChangeNotification = true
    try {
      this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: content } })
      this.lastSyncedContent = content
    } finally {
      this.suppressChangeNotification = false
    }
  }

  focus(): void {
    this.view.focus()
  }

  destroy(): void {
    this.flush()
    window.removeEventListener('pagehide', this.onPageHide)
    cleanupPreview()
    this.view.destroy()
  }
}
