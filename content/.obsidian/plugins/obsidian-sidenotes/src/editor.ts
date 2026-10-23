import { StateField, type EditorState } from '@codemirror/state'
import { Decoration, EditorView, WidgetType, type DecorationSet } from '@codemirror/view'
import {
  editorInfoField,
  editorLivePreviewField,
  type MarkdownRenderChild,
  type Plugin,
} from 'obsidian'
import type { SidenoteMatch, SidenoteDefinition } from './types'
import { parseSidenotes } from './parser'
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
      this.match.data.content === other.match.data.content &&
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

function decorate(
  state: EditorState,
  matches: SidenoteMatch[],
  definitions: SidenoteDefinition[],
  plugin: Plugin,
): DecorationSet {
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
  if (livePreview) {
    for (const definition of definitions) {
      const editing = state.selection.ranges.some(
        range => range.from <= definition.to && range.to >= definition.from,
      )
      if (!editing) ranges.push(Decoration.replace({}).range(definition.from, definition.to))
    }
  }
  return Decoration.set(ranges, true)
}

export function sidenoteEditorExtension(
  plugin: Plugin,
): StateField<{
  matches: SidenoteMatch[]
  definitions: SidenoteDefinition[]
  decorations: DecorationSet
}> {
  return StateField.define({
    create(state) {
      const parsed = parseSidenotes(state.doc.toString())
      return { ...parsed, decorations: decorate(state, parsed.matches, parsed.definitions, plugin) }
    },
    update(value, transaction) {
      const parsed = transaction.docChanged
        ? parseSidenotes(transaction.state.doc.toString())
        : value
      return {
        ...parsed,
        decorations: decorate(transaction.state, parsed.matches, parsed.definitions, plugin),
      }
    },
    provide: field => EditorView.decorations.from(field, value => value.decorations),
  })
}
