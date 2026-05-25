import { autocompletion, completionStatus, moveCompletionSelection } from '@codemirror/autocomplete'
import { defaultKeymap, historyKeymap, history, indentWithTab } from '@codemirror/commands'
import { markdown } from '@codemirror/lang-markdown'
import { syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language'
import { EditorState, Prec, type Extension } from '@codemirror/state'
import { EditorView, keymap, lineNumbers } from '@codemirror/view'
import TurndownService from 'turndown'
import { codemirrorCodeLanguage } from '../../util/codemirror-language'
import { codemirrorChangedTextIsBlank, codemirrorTextIsBlank } from '../../util/codemirror-text'
import { completionSources } from '../multiplayer/completions'
import { togglePreview, cleanupPreview, onEditorUpdate } from '../multiplayer/completions/preview'

const turndown = new TurndownService({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  bulletListMarker: '-',
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
}

export class MarkdownEditor {
  private view: EditorView
  private empty: boolean

  constructor(config: MarkdownEditorConfig) {
    const mode = config.mode ?? 'markdown'
    const customKeymap = Prec.highest(
      keymap.of([
        {
          key: 'Mod-Enter',
          run: () => {
            config.onSubmit?.()
            return true
          },
        },
        {
          key: 'Escape',
          run: () => {
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
        this.empty = codemirrorChangedTextIsBlank(this.empty, update.state.doc, update.changes)
        config.onEdited?.({ empty: this.empty })
        if (config.onChange) config.onChange(update.state.doc.toString())
      }
      if (mode === 'markdown') onEditorUpdate(update)
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
      },
      '&.cm-focused': {
        outline: 'none !important',
        border: 'none !important',
        boxShadow: 'none !important',
      },
      '.cm-content': { padding: '0 !important', minHeight: '20px', caretColor: 'inherit' },
      '.cm-gutter': { minHeight: '20px' },
      '.cm-scroller': {
        overflow: 'visible',
        fontFamily: 'inherit !important',
        fontSize: 'inherit',
      },
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

    const extensions: Extension[] = [
      history(),
      customKeymap,
      keymap.of(
        mode === 'code'
          ? [indentWithTab, ...defaultKeymap, ...historyKeymap]
          : [...defaultKeymap, ...historyKeymap],
      ),
      updateListener,
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      EditorState.tabSize.of(2),
      transparentTheme,
    ]

    if (config.lineWrapping ?? mode === 'markdown') {
      extensions.push(EditorView.lineWrapping)
    }

    if (mode === 'markdown') {
      extensions.unshift(markdown())
      extensions.push(
        autocompletion({ override: completionSources, closeOnBlur: false, activateOnTyping: true }),
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

  getValue(): string {
    return this.view.state.doc.toString()
  }

  setValue(content: string): void {
    if (
      this.view.state.doc.length === content.length &&
      this.view.state.doc.sliceString(0) === content
    )
      return
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: content } })
  }

  focus(): void {
    this.view.focus()
  }

  destroy(): void {
    cleanupPreview()
    this.view.destroy()
  }
}
