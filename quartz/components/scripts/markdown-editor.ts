import { EditorView, keymap } from "@codemirror/view"
import { EditorState } from "@codemirror/state"
import { markdown } from "@codemirror/lang-markdown"
import { defaultKeymap, historyKeymap, history } from "@codemirror/commands"
import { syntaxHighlighting, defaultHighlightStyle } from "@codemirror/language"

export interface MarkdownEditorConfig {
  parent: HTMLElement
  initialContent?: string
  onChange?: (content: string) => void
  onSubmit?: () => void
  onCancel?: () => void
}

export class MarkdownEditor {
  private view: EditorView

  constructor(config: MarkdownEditorConfig) {
    const customKeymap = keymap.of([
      {
        key: "Mod-Enter",
        run: () => {
          config.onSubmit?.()
          return true
        },
      },
      {
        key: "Escape",
        run: () => {
          config.onCancel?.()
          return true
        },
      },
    ])

    const updateListener = EditorView.updateListener.of((update) => {
      if (update.docChanged && config.onChange) {
        config.onChange(update.state.doc.toString())
      }
    })

    const transparentTheme = EditorView.theme({
      "&": {
        backgroundColor: "transparent !important",
        border: "none !important",
        outline: "none !important",
        fontSize: "inherit",
        fontFamily: "inherit !important",
        lineHeight: "inherit",
        color: "inherit",
        height: "100%",
        padding: "0",
      },
      "&.cm-focused": {
        outline: "none !important",
        border: "none !important",
        boxShadow: "none !important",
      },
      ".cm-content": {
        padding: "0 !important",
        minHeight: "inherit",
        caretColor: "inherit",
      },
      ".cm-scroller": {
        overflow: "inherit",
        fontFamily: "inherit !important",
        fontSize: "inherit",
      },
      ".cm-line": {
        padding: "0",
      },
      ".cm-cursor": {
        borderLeftColor: "inherit",
      },
    })

    const extensions = [
      markdown(),
      history(),
      customKeymap,
      keymap.of([...defaultKeymap, ...historyKeymap]),
      updateListener,
      syntaxHighlighting(defaultHighlightStyle),
      EditorView.lineWrapping,
      transparentTheme,
    ]

    const state = EditorState.create({
      doc: config.initialContent || "",
      extensions,
    })

    this.view = new EditorView({
      state,
      parent: config.parent,
    })
  }

  getValue(): string {
    return this.view.state.doc.toString()
  }

  setValue(content: string): void {
    this.view.dispatch({
      changes: {
        from: 0,
        to: this.view.state.doc.length,
        insert: content,
      },
    })
  }

  focus(): void {
    this.view.focus()
  }

  destroy(): void {
    this.view.destroy()
  }
}
