import type {
  Decoration as CodeMirrorDecoration,
  EditorView as CodeMirrorEditorView,
  ViewUpdate as CodeMirrorViewUpdate,
  WidgetType as CodeMirrorWidgetType,
} from '@codemirror/view'
import type { NotebookLspConfig } from '../../lsp/pyright'
import type { NotebookLeapTarget } from './vim-navigation'

export type NotebookCodeEditor = {
  getValue(): string
  setValue(content: string): void
  setVimMode(enabled: boolean): Promise<void>
  focus(): void
  destroy(): void
}

export type NotebookCodeEditorConfig = {
  parent: HTMLElement
  initialContent?: string
  language?: string
  vimMode?: boolean
  onEdited?: () => void
  onChange?: (content: string) => void
  onSubmit?: () => void
  onSubmitAndAdvance?: () => void
  onSave?: () => void
  onCancel?: () => void
  lsp?: NotebookLspConfig
}

export type NotebookWhitespaceKind = 'lead' | 'trail' | 'tab' | 'nbsp'
export type NotebookWhitespaceGlyph = {
  from: number
  to: number
  text: string
  kind: NotebookWhitespaceKind
}

const notebookWhitespaceLeadMultispace = ['»', '·', '·', '·'] as const
const notebookWhitespaceTab = '»·'
const notebookWhitespaceNbsp = '+'
const notebookWhitespaceTrail = '·'

export function notebookWhitespaceGlyphs(
  line: string,
  lineFrom = 0,
): readonly NotebookWhitespaceGlyph[] {
  const glyphs: NotebookWhitespaceGlyph[] = []
  const contentStart = line.search(/[^ \t\u00a0]/)
  const leadingTo = contentStart < 0 ? line.length : contentStart
  const trailingMatch = /[ \t\u00a0]+$/.exec(line)
  const trailingFrom = trailingMatch && leadingTo < line.length ? trailingMatch.index : line.length
  let leadingSpaceIndex = 0

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index]
    const from = lineFrom + index
    if (character === '\t') {
      glyphs.push({ from, to: from + 1, text: notebookWhitespaceTab, kind: 'tab' })
    } else if (character === '\u00a0') {
      glyphs.push({ from, to: from + 1, text: notebookWhitespaceNbsp, kind: 'nbsp' })
    } else if (character === ' ' && index < leadingTo) {
      const text =
        notebookWhitespaceLeadMultispace[
          leadingSpaceIndex % notebookWhitespaceLeadMultispace.length
        ]
      glyphs.push({ from, to: from + 1, text, kind: 'lead' })
      leadingSpaceIndex += 1
    } else if (character === ' ' && index >= trailingFrom) {
      glyphs.push({ from, to: from + 1, text: notebookWhitespaceTrail, kind: 'trail' })
    }
  }

  return glyphs
}

export function createWhitespaceExtension({
  WidgetType,
  Decoration,
  RangeSetBuilder,
  ViewPlugin,
}: {
  WidgetType: (typeof import('@codemirror/view'))['WidgetType']
  Decoration: (typeof import('@codemirror/view'))['Decoration']
  RangeSetBuilder: (typeof import('@codemirror/state'))['RangeSetBuilder']
  ViewPlugin: (typeof import('@codemirror/view'))['ViewPlugin']
}) {
  class NotebookWhitespaceWidget extends WidgetType {
    constructor(
      readonly text: string,
      readonly kind: NotebookWhitespaceKind,
    ) {
      super()
    }

    eq(other: CodeMirrorWidgetType): boolean {
      return (
        this === other ||
        (other instanceof NotebookWhitespaceWidget &&
          other.text === this.text &&
          other.kind === this.kind)
      )
    }

    toDOM() {
      const element = document.createElement('span')
      element.className = `cm-notebookWhitespace cm-notebookWhitespace-${this.kind}`
      element.textContent = this.text
      element.setAttribute('aria-hidden', 'true')
      return element
    }

    ignoreEvent() {
      return true
    }
  }

  const whitespaceDecorationsByKey = {
    'lead:»': Decoration.replace({
      widget: new NotebookWhitespaceWidget('»', 'lead'),
      inclusive: false,
    }),
    'lead:·': Decoration.replace({
      widget: new NotebookWhitespaceWidget('·', 'lead'),
      inclusive: false,
    }),
    'trail:·': Decoration.replace({
      widget: new NotebookWhitespaceWidget('·', 'trail'),
      inclusive: false,
    }),
    'tab:»·': Decoration.replace({
      widget: new NotebookWhitespaceWidget('»·', 'tab'),
      inclusive: false,
    }),
    'nbsp:+': Decoration.replace({
      widget: new NotebookWhitespaceWidget('+', 'nbsp'),
      inclusive: false,
    }),
  } as const

  const decorationForWhitespaceGlyph = (glyph: NotebookWhitespaceGlyph): CodeMirrorDecoration => {
    const key = `${glyph.kind}:${glyph.text}` as keyof typeof whitespaceDecorationsByKey
    return (
      whitespaceDecorationsByKey[key] ??
      Decoration.replace({
        widget: new NotebookWhitespaceWidget(glyph.text, glyph.kind),
        inclusive: false,
      })
    )
  }

  const notebookWhitespaceDecorations = (targetView: CodeMirrorEditorView) => {
    const builder = new RangeSetBuilder<CodeMirrorDecoration>()
    let lastLineNumber = 0
    for (const range of targetView.visibleRanges) {
      const startLine = targetView.state.doc.lineAt(range.from)
      const endLine = targetView.state.doc.lineAt(range.to)
      const firstLineNumber = Math.max(startLine.number, lastLineNumber + 1)
      for (let lineNumber = firstLineNumber; lineNumber <= endLine.number; lineNumber += 1) {
        const line = targetView.state.doc.line(lineNumber)
        for (const glyph of notebookWhitespaceGlyphs(line.text, line.from)) {
          builder.add(glyph.from, glyph.to, decorationForWhitespaceGlyph(glyph))
        }
      }
      lastLineNumber = Math.max(lastLineNumber, endLine.number)
    }
    return builder.finish()
  }

  class NotebookWhitespaceView {
    decorations: ReturnType<typeof notebookWhitespaceDecorations>

    constructor(view: CodeMirrorEditorView) {
      this.decorations = notebookWhitespaceDecorations(view)
    }

    update(update: CodeMirrorViewUpdate) {
      if (update.viewportChanged) {
        this.decorations = notebookWhitespaceDecorations(update.view)
        return
      }
      if (!update.docChanged) return

      let mapped = this.decorations.map(update.changes)
      const doc = update.state.doc
      const lineSpans: Array<{ from: number; to: number; startLine: number; endLine: number }> = []
      update.changes.iterChangedRanges((_fromA, _toA, fromB, toB) => {
        const firstLine = doc.lineAt(fromB)
        const lastLine = doc.lineAt(toB)
        const prev = lineSpans[lineSpans.length - 1]
        if (prev && firstLine.number <= prev.endLine + 1) {
          prev.to = Math.max(prev.to, lastLine.to)
          prev.endLine = Math.max(prev.endLine, lastLine.number)
        } else {
          lineSpans.push({
            from: firstLine.from,
            to: lastLine.to,
            startLine: firstLine.number,
            endLine: lastLine.number,
          })
        }
      })

      for (const span of lineSpans) {
        const add = []
        for (let lineNum = span.startLine; lineNum <= span.endLine; lineNum += 1) {
          const line = doc.line(lineNum)
          for (const glyph of notebookWhitespaceGlyphs(line.text, line.from)) {
            add.push(decorationForWhitespaceGlyph(glyph).range(glyph.from, glyph.to))
          }
        }
        mapped = mapped.update({
          filterFrom: span.from,
          filterTo: span.to,
          filter: () => false,
          add,
        })
      }
      this.decorations = mapped
    }
  }

  const whitespaceExtension = ViewPlugin.fromClass(NotebookWhitespaceView, {
    decorations: value => value.decorations,
  })
  return { whitespaceExtension }
}

export function createTransparentTheme({
  EditorView,
}: {
  EditorView: (typeof import('@codemirror/view'))['EditorView']
}) {
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
      position: 'relative',
    },
    '&.cm-focused': {
      outline: 'none !important',
      border: 'none !important',
      boxShadow: 'none !important',
    },
    '.cm-content': {
      padding: '0 0 1.15rem 0 !important',
      minHeight: '20px',
      caretColor: 'inherit',
    },
    '.cm-gutter': { minHeight: '20px' },
    '.cm-scroller': { overflow: 'visible', fontFamily: 'inherit !important', fontSize: 'inherit' },
    '.cm-line': { padding: '0' },
    '.cm-cursor': { borderLeftColor: 'inherit' },
    '.cm-notebookWhitespace': {
      color: 'color-mix(in srgb, var(--gray) 64%, transparent)',
      fontStyle: 'normal',
      fontWeight: '400',
      opacity: '0.72',
      textDecorationLine: 'none',
    },
    '.cm-notebookWhitespace-trail': {
      color: 'color-mix(in srgb, var(--gray) 74%, var(--notebook-active-green) 26%)',
    },
    '.cm-notebookLeapBackdrop': {
      backgroundColor: 'transparent',
      color: 'var(--code-comment, var(--gray)) !important',
      fontStyle: 'italic',
      transitionProperty: 'color, background-color',
      transitionDuration: '120ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookLeapMatch': {
      borderRadius: '2px',
      backgroundColor: 'color-mix(in srgb, var(--notebook-active-green) 22%, transparent)',
      color: 'var(--dark) !important',
      fontWeight: '800',
      textDecorationLine: 'none',
      transitionProperty: 'color, background-color',
      transitionDuration: '80ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookStatus': {
      position: 'absolute',
      right: '4px',
      bottom: '1px',
      zIndex: '4',
      maxWidth: 'min(60%, 12rem)',
      overflow: 'hidden',
      padding: '1px 5px',
      borderRadius: '4px',
      backgroundColor: 'color-mix(in srgb, var(--light) 82%, transparent)',
      color: 'color-mix(in srgb, var(--gray) 78%, var(--dark) 22%)',
      fontSize: '0.72em',
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: '0',
      lineHeight: '1.25',
      opacity: '0.72',
      pointerEvents: 'none',
      textAlign: 'right',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      transitionProperty: 'color, background-color, opacity',
      transitionDuration: '120ms',
      transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
    },
    '.cm-notebookStatus[data-active="true"]': {
      backgroundColor: 'color-mix(in srgb, var(--notebook-active-green) 18%, var(--light))',
      color: 'var(--notebook-active-green)',
      opacity: '1',
    },
  })
  return { transparentTheme }
}

export function createLeapHighlight({
  StateEffect,
  Decoration,
  StateField,
  EditorView,
}: {
  StateEffect: (typeof import('@codemirror/state'))['StateEffect']
  Decoration: (typeof import('@codemirror/view'))['Decoration']
  StateField: (typeof import('@codemirror/state'))['StateField']
  EditorView: (typeof import('@codemirror/view'))['EditorView']
}) {
  const leapHighlightEffect = StateEffect.define<
    { targets: readonly NotebookLeapTarget[]; activeIndex: number } | undefined
  >()

  const leapDecorationSet = (targets: readonly NotebookLeapTarget[], activeIndex: number) =>
    Decoration.set(
      targets.map((target, index) =>
        Decoration.mark({
          class: index === activeIndex ? 'cm-notebookLeapMatch' : 'cm-notebookLeapBackdrop',
        }).range(target.matchFrom, target.matchTo),
      ),
      true,
    )

  const leapHighlightField = StateField.define({
    create: () => Decoration.none,
    update(highlights, transaction) {
      let next = transaction.docChanged ? highlights.map(transaction.changes) : highlights
      for (const effect of transaction.effects) {
        if (effect.is(leapHighlightEffect)) {
          next = effect.value
            ? leapDecorationSet(effect.value.targets, effect.value.activeIndex)
            : Decoration.none
        }
      }
      return next
    },
    provide: field => EditorView.decorations.from(field),
  })
  return { leapHighlightEffect, leapHighlightField }
}
