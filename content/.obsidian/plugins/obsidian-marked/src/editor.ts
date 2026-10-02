import { StateField, type EditorState } from '@codemirror/state'
import { Decoration, EditorView, WidgetType, type DecorationSet } from '@codemirror/view'
import { editorInfoField, editorLivePreviewField } from 'obsidian'
import { MarkerRenderChild, type MarkerAnnotations } from './annotations'
import { findMarkers, type MarkerMatch } from './parser'
import { renderMarkerContent } from './renderer'

class MarkerWidget extends WidgetType {
  private child?: MarkerRenderChild

  constructor(
    private match: MarkerMatch,
    private sourcePath: string,
    private manager: MarkerAnnotations,
  ) {
    super()
  }

  eq(other: MarkerWidget): boolean {
    return this.match.raw === other.match.raw && this.sourcePath === other.sourcePath
  }

  toDOM(view: EditorView): HTMLElement {
    const element = document.createElement('span')
    const child = new MarkerRenderChild(element, this.match.intensity, this.manager)
    this.child = child
    this.manager.plugin.addChild(child)
    void renderMarkerContent(child, this.match.content, this.sourcePath, this.manager)
      .then(() => view.requestMeasure())
      .catch(error => console.error('Marker rendering failed', error))
    return element
  }

  ignoreEvent(): boolean {
    return false
  }

  destroy(): void {
    if (this.child) this.manager.plugin.removeChild(this.child)
    this.child = undefined
  }
}

function decorate(
  state: EditorState,
  matches: MarkerMatch[],
  manager: MarkerAnnotations,
): DecorationSet {
  if (state.field(editorLivePreviewField, false) !== true) return Decoration.none
  const sourcePath = state.field(editorInfoField, false)?.file?.path ?? ''
  return Decoration.set(
    matches.flatMap(match => {
      if (state.selection.ranges.some(range => range.from <= match.to && range.to >= match.from))
        return []
      return [
        Decoration.replace({ widget: new MarkerWidget(match, sourcePath, manager) }).range(
          match.from,
          match.to,
        ),
      ]
    }),
    true,
  )
}

export function markerEditorExtension(
  manager: MarkerAnnotations,
): StateField<{ matches: MarkerMatch[]; decorations: DecorationSet }> {
  return StateField.define({
    create(state) {
      const matches = findMarkers(state.doc.toString())
      return { matches, decorations: decorate(state, matches, manager) }
    },
    update(value, transaction) {
      const matches = transaction.docChanged
        ? findMarkers(transaction.state.doc.toString())
        : value.matches
      return { matches, decorations: decorate(transaction.state, matches, manager) }
    },
    provide: field => EditorView.decorations.from(field, value => value.decorations),
  })
}
