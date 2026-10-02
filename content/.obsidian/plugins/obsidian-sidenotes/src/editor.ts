import { StateField, type EditorState } from '@codemirror/state'
import { Decoration, EditorView, WidgetType, type DecorationSet } from '@codemirror/view'
import {
  editorInfoField,
  editorLivePreviewField,
  type MarkdownRenderChild,
  type Plugin,
} from 'obsidian'
import type { SidenoteMatch } from './types'
import { findSidenotes } from './parser'
import { createSidenote } from './renderer'

class SidenoteWidget extends WidgetType {
  private child?: MarkdownRenderChild

  constructor(
    private match: SidenoteMatch,
    private number: number,
    private plugin: Plugin,
    private sourcePath: string,
    private sourceMode: boolean,
  ) {
    super()
  }

  eq(other: SidenoteWidget): boolean {
    return (
      this.match.data.raw === other.match.data.raw &&
      this.number === other.number &&
      this.sourcePath === other.sourcePath &&
      this.sourceMode === other.sourceMode
    )
  }

  toDOM(view: EditorView): HTMLElement {
    const rendered = createSidenote(
      this.match.data,
      this.number,
      this.plugin,
      this.sourcePath,
      () => view.requestMeasure(),
    )
    this.child = rendered.child
    if (this.sourceMode) {
      rendered.element.classList.add('sidenote-source')
      rendered.element.hidden = true
    }
    this.plugin.addChild(rendered.child)
    void rendered.ready.catch(error => console.error('Sidenote rendering failed', error))
    return rendered.element
  }

  ignoreEvent(): boolean {
    return true
  }

  destroy(): void {
    if (this.child) this.plugin.removeChild(this.child)
    this.child = undefined
  }
}

function decorate(state: EditorState, matches: SidenoteMatch[], plugin: Plugin): DecorationSet {
  const livePreview = state.field(editorLivePreviewField, false) === true
  const sourcePath = state.field(editorInfoField, false)?.file?.path ?? ''
  const ranges = matches.flatMap((match, index) => {
    const editing = state.selection.ranges.some(
      range => range.from <= match.to && range.to >= match.from,
    )
    if (livePreview && editing) return []
    const widget = new SidenoteWidget(match, index + 1, plugin, sourcePath, !livePreview)
    if (!livePreview) return [Decoration.widget({ widget, side: -1 }).range(match.from)]
    return [Decoration.replace({ widget }).range(match.from, match.to)]
  })
  return Decoration.set(ranges, true)
}

export function sidenoteEditorExtension(
  plugin: Plugin,
): StateField<{ matches: SidenoteMatch[]; decorations: DecorationSet }> {
  return StateField.define({
    create(state) {
      const matches = findSidenotes(state.doc.toString())
      return { matches, decorations: decorate(state, matches, plugin) }
    },
    update(value, transaction) {
      const matches = transaction.docChanged
        ? findSidenotes(transaction.state.doc.toString())
        : value.matches
      return { matches, decorations: decorate(transaction.state, matches, plugin) }
    },
    provide: field => EditorView.decorations.from(field, value => value.decorations),
  })
}
